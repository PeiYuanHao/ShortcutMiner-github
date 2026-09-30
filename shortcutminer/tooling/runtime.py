"""Per-invocation budget, persistence, and event tracing for tools."""

from dataclasses import dataclass, replace
from threading import Lock
import json
from pathlib import Path
from typing import Any

from pydantic import Field, TypeAdapter

from shortcutminer.protocol.executive import TaskId
from shortcutminer.protocol.identify import (
    CandidateId,
    SafeRelativePath,
    StrictProtocolModel,
)
from shortcutminer.experiment.check_resolution import (
    QUANTITATIVE_EXPERIMENT_TYPES,
    resolve_quantitative_checks,
)
from shortcutminer.storage.events import append_event
from shortcutminer.storage.tool_artifacts import (
    save_tool_result,
    tool_result_reference,
)
from shortcutminer.tooling.base import ToolExecutionContext
from shortcutminer.tooling.errors import ToolBudgetExceeded, ToolRuntimeError
from shortcutminer.tooling.protocol import AgentName, ToolCall, ToolResult, ToolSpec
from shortcutminer.tooling.registry import ToolRegistry


class ToolBudget(StrictProtocolModel):
    max_tool_calls: int = Field(default=8, ge=1, le=100)
    timeout_seconds: float = Field(default=30.0, gt=0, le=3600)
    max_selected_samples: int = Field(default=100, ge=1, le=100_000)


class RunToolBudgetState:
    """Atomic run-wide tool-call counter shared by task runtime forks."""

    def __init__(self, max_tool_calls: int, *, used: int = 0) -> None:
        self.max_tool_calls = max_tool_calls
        if used < 0 or used > max_tool_calls:
            raise ValueError("invalid restored run tool budget")
        self.used = used
        self._lock = Lock()

    def reserve(self) -> None:
        with self._lock:
            if self.used >= self.max_tool_calls:
                raise ToolBudgetExceeded("Run tool-call budget is exhausted")
            self.used += 1


@dataclass(frozen=True)
class TaskExecutionScope:
    """Authoritative correlation and explicit inputs for one AnalysisTask."""

    candidate_id: str
    task_id: str
    explicit_artifacts: frozenset[str]


class ToolRuntime:
    """Execute one Agent's permitted tools without storing runtime objects in State."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        run_id: str,
        agent_name: AgentName,
        dataset_root: str | Path | None = None,
        runs_root: str | Path | None = None,
        workspace_root: str | Path | None = None,
        approved_artifacts: set[str] | frozenset[str] = frozenset(),
        approved_samples: set[str] | frozenset[str] = frozenset(),
        runtime_constraints: set[str] | frozenset[str] = frozenset(),
        model_capabilities: set[str] | frozenset[str] = frozenset(),
        allowed_experiment_types: set[str] | frozenset[str] = frozenset(),
        budget: ToolBudget | None = None,
        task_scope: TaskExecutionScope | None = None,
        restore_persisted_task_results: bool = False,
        run_budget_state: RunToolBudgetState | None = None,
    ) -> None:
        self.registry = registry
        self.budget = budget or ToolBudget()
        self.context = ToolExecutionContext(
            run_id=run_id,
            agent_name=agent_name,
            dataset_root=(Path(dataset_root) if dataset_root is not None else None),
            runs_root=(Path(runs_root) if runs_root is not None else None),
            workspace_root=(
                Path(workspace_root) if workspace_root is not None else None
            ),
            approved_artifacts=frozenset(approved_artifacts),
            approved_samples=frozenset(approved_samples),
            runtime_constraints=frozenset(runtime_constraints),
            model_capabilities=frozenset(model_capabilities),
            allowed_experiment_types=frozenset(allowed_experiment_types),
            max_selected_samples=self.budget.max_selected_samples,
        )
        self.history: list[ToolResult] = []
        self._call_ids: set[str] = set()
        self.task_scope = task_scope
        self.restore_persisted_task_results = restore_persisted_task_results
        self.run_budget_state = run_budget_state or RunToolBudgetState(self.budget.max_tool_calls)

    @property
    def agent_name(self) -> AgentName:
        return self.context.agent_name

    @property
    def run_id(self) -> str:
        return self.context.run_id

    def tool_specs(self) -> list[ToolSpec]:
        return self.registry.specs_for(self.agent_name)

    @property
    def explicit_task_artifacts(self) -> frozenset[str]:
        if self.task_scope is None:
            return frozenset(self.context.approved_artifacts)
        return self.task_scope.explicit_artifacts

    def fork_for_task(
        self,
        *,
        candidate_id: str,
        task_id: str,
        runtime_constraints: list[str],
        requested_checks: list[str],
        explicitly_authorized_artifacts: list[str] | set[str] | frozenset[str],
    ) -> "ToolRuntime":
        """Create an isolated mutable runtime while sharing static infrastructure."""

        if self.agent_name not in {"quantitative", "qualitative"}:
            raise ToolRuntimeError("Only task Agents may be task-scoped")
        validated_candidate = TypeAdapter(CandidateId).validate_python(candidate_id)
        validated_task = TypeAdapter(TaskId).validate_python(task_id)
        task_candidate = validated_task.removeprefix("TASK-").rsplit("-", 1)[0]
        if task_candidate != validated_candidate:
            raise ToolRuntimeError("Task scope candidate/task correlation is invalid")
        explicit = frozenset(
            TypeAdapter(SafeRelativePath).validate_python(reference)
            for reference in explicitly_authorized_artifacts
        )
        child = ToolRuntime(
            self.registry,
            run_id=self.run_id,
            agent_name=self.agent_name,
            dataset_root=self.context.dataset_root,
            runs_root=self.context.runs_root,
            workspace_root=self.context.workspace_root,
            approved_artifacts=explicit,
            approved_samples=frozenset(),
            runtime_constraints=runtime_constraints,
            model_capabilities=self.context.model_capabilities,
            allowed_experiment_types=(
                resolve_quantitative_checks(requested_checks)
                if self.agent_name == "quantitative" else frozenset()
            ),
            budget=self.budget,
            task_scope=TaskExecutionScope(
                candidate_id=validated_candidate,
                task_id=validated_task,
                explicit_artifacts=explicit,
            ),
            restore_persisted_task_results=self.restore_persisted_task_results,
            run_budget_state=self.run_budget_state,
        )
        if child.restore_persisted_task_results:
            child.restore_task_history()
        return child

    def assert_task_scope(self, *, candidate_id: str, task_id: str) -> None:
        """Reject ambient or mismatched mutable state at a provider boundary."""

        if self.task_scope is None:
            return
        if (
            self.task_scope.candidate_id != candidate_id
            or self.task_scope.task_id != task_id
        ):
            raise ToolRuntimeError("Tool Runtime is bound to another AnalysisTask")
        if any(
            result.run_id != self.run_id
            or result.agent_name != self.agent_name
            or result.candidate_id != candidate_id
            or result.task_id != task_id
            for result in self.history
        ):
            raise ToolRuntimeError("Tool Runtime contains cross-task history")

    def restore_task_history(self) -> None:
        """Restore only persisted ToolResults correlated to this exact task."""

        if self.task_scope is None:
            raise ToolRuntimeError("Persisted history restoration requires task scope")
        if self.context.runs_root is None:
            return
        tool_root = (
            self.context.runs_root / self.run_id / "tools" / self.agent_name
        )
        if not tool_root.is_dir():
            return
        restored: list[ToolResult] = []
        for path in sorted(tool_root.glob("*.json")):
            try:
                result = ToolResult.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ToolRuntimeError(
                    "Persisted ToolResult cannot be restored safely"
                ) from exc
            if (
                result.run_id == self.run_id
                and result.agent_name == self.agent_name
                and result.candidate_id == self.task_scope.candidate_id
                and result.task_id == self.task_scope.task_id
            ):
                restored.append(result)
        self.history = restored
        self._call_ids = {result.tool_call_id for result in restored}
        generated = {
            artifact
            for result in restored
            if result.status == "success"
            for artifact in result.artifacts
        }
        self.context = replace(
            self.context,
            approved_artifacts=frozenset(
                {*self.task_scope.explicit_artifacts, *generated}
            ),
        )

    def approve_samples(self, references: list[str]) -> None:
        """Add control-layer-selected dataset references to the runtime boundary."""

        validated = {
            TypeAdapter(SafeRelativePath).validate_python(reference)
            for reference in references
        }
        combined = {*self.context.approved_samples, *validated}
        if len(combined) > self.budget.max_selected_samples:
            raise ToolBudgetExceeded(
                "Approved sample set exceeds the runtime sample budget"
            )
        self.context = replace(
            self.context,
            approved_samples=frozenset(combined),
        )

    def approve_artifacts(self, references: list[str]) -> None:
        """Add control-layer-generated run artifacts to the readable allowlist."""

        validated = {
            TypeAdapter(SafeRelativePath).validate_python(reference)
            for reference in references
        }
        self.context = replace(
            self.context,
            approved_artifacts=frozenset(
                {*self.context.approved_artifacts, *validated}
            ),
        )

    def bind_analysis_task(
        self,
        *,
        runtime_constraints: list[str],
        requested_checks: list[str],
    ) -> None:
        """Bind Executive policy to the mutable tool boundary before Agent use."""

        resolved_types = resolve_quantitative_checks(requested_checks)
        self.context = replace(
            self.context,
            runtime_constraints=frozenset(runtime_constraints),
            allowed_experiment_types=frozenset(resolved_types),
        )

    def can_attempt_run_experiment(
        self,
        *,
        requested_checks: list[str],
        required_model_capabilities: list[str],
        requires_sample_access: bool,
        requires_external_services: bool,
        runtime_constraints: list[str],
    ) -> bool:
        """Report deterministic runtime feasibility without executing a tool."""

        if self.agent_name != "quantitative":
            return False
        if not any(spec.name == "run_experiment" for spec in self.tool_specs()):
            return False
        resolved_types = resolve_quantitative_checks(requested_checks)
        if not resolved_types or not set(resolved_types).issubset(
            QUANTITATIVE_EXPERIMENT_TYPES
        ):
            return False
        if not set(required_model_capabilities).issubset(
            self.context.model_capabilities
        ):
            return False
        if requires_sample_access and self.context.dataset_root is None:
            return False
        if (
            "no_retraining" in runtime_constraints
            and "task_model_training" in resolved_types
        ):
            return False
        if requires_external_services and "no_external_services" in runtime_constraints:
            return False
        return True

    def execute(self, call: ToolCall | dict[str, Any]) -> ToolResult:
        parsed = call if isinstance(call, ToolCall) else ToolCall.model_validate(call)
        if self.task_scope is not None:
            self.assert_task_scope(
                candidate_id=parsed.candidate_id or "",
                task_id=parsed.task_id or "",
            )
        if parsed.tool_call_id in self._call_ids:
            raise ToolBudgetExceeded(
                f"Duplicate tool_call_id: {parsed.tool_call_id}"
            )
        if len(self.history) >= self.budget.max_tool_calls:
            raise ToolBudgetExceeded("Agent tool-call budget is exhausted")
        self.run_budget_state.reserve()
        self._call_ids.add(parsed.tool_call_id)
        schemas = tuple(spec.model_dump(mode="json") for spec in self.tool_specs())
        context = replace(self.context, tool_schemas=schemas)

        self._event(
            "TOOL_STARTED",
            parsed,
            {
                "tool_call_id": parsed.tool_call_id,
                "tool": parsed.tool_name,
                "candidate_id": parsed.candidate_id,
                "task_id": parsed.task_id,
                "arguments": parsed.arguments,
            },
        )
        try:
            result = self.registry.execute(
                parsed,
                context,
                timeout_seconds=self.budget.timeout_seconds,
            )
        except ToolRuntimeError as exc:
            self._event(
                "TOOL_REJECTED",
                parsed,
                {
                    "tool_call_id": parsed.tool_call_id,
                    "tool": parsed.tool_name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise

        if self.context.runs_root is not None:
            reference = tool_result_reference(result)
            result = ToolResult.model_validate(
                {
                    **result.model_dump(mode="json"),
                    "artifacts": [*result.artifacts, reference],
                }
            )
            save_tool_result(self.context.runs_root, result)
        if result.status == "success" and result.artifacts:
            self.context = replace(
                self.context,
                approved_artifacts=frozenset(
                    {*self.context.approved_artifacts, *result.artifacts}
                ),
            )
        self.history.append(result)
        self._event(
            "TOOL_COMPLETED" if result.status == "success" else "TOOL_FAILED",
            parsed,
            {
                "tool_call_id": result.tool_call_id,
                "tool": result.tool_name,
                "status": result.status,
                "duration_ms": result.duration_ms,
                "artifacts": result.artifacts,
                **(
                    {"error": result.error.model_dump(mode="json")}
                    if result.error is not None
                    else {}
                ),
            },
        )
        return result

    def _event(
        self,
        name: str,
        call: ToolCall,
        details: dict[str, Any],
    ) -> None:
        if self.context.runs_root is None:
            return
        append_event(
            self.context.runs_root,
            call.run_id,
            name,
            stage=f"tool:{call.agent_name}",
            details=details,
        )
