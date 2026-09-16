"""Per-invocation budget, persistence, and event tracing for tools."""

from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import Field, TypeAdapter

from shortcutminer.protocol.identify import SafeRelativePath, StrictProtocolModel
from shortcutminer.storage.events import append_event
from shortcutminer.storage.tool_artifacts import (
    save_tool_result,
    tool_result_reference,
)
from shortcutminer.tooling.base import ToolExecutionContext
from shortcutminer.tooling.errors import ToolBudgetExceeded, ToolRuntimeError
from shortcutminer.tooling.protocol import AgentName, ToolCall, ToolResult, ToolSpec
from shortcutminer.tooling.registry import ToolRegistry


QUANTITATIVE_EXPERIMENT_TYPES = frozenset({
    "statistical_check",
    "model_evaluation",
    "intervention",
    "ablation",
    "subgroup_evaluation",
    "metric_computation",
    "task_model_training",
    "task_model_evaluation",
    "ood_evaluation",
    "counterfactual_intervention",
})


class ToolBudget(StrictProtocolModel):
    max_tool_calls: int = Field(default=8, ge=1, le=100)
    timeout_seconds: float = Field(default=30.0, gt=0, le=3600)
    max_selected_samples: int = Field(default=100, ge=1, le=100_000)


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

    @property
    def agent_name(self) -> AgentName:
        return self.context.agent_name

    @property
    def run_id(self) -> str:
        return self.context.run_id

    def tool_specs(self) -> list[ToolSpec]:
        return self.registry.specs_for(self.agent_name)

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

        self.context = replace(
            self.context,
            runtime_constraints=frozenset(runtime_constraints),
            allowed_experiment_types=frozenset(
                check
                for check in requested_checks
                if check in QUANTITATIVE_EXPERIMENT_TYPES
            ),
        )

    def can_attempt_run_experiment(
        self,
        *,
        requested_checks: list[str],
        required_model_capabilities: list[str],
        requires_sample_access: bool,
    ) -> bool:
        """Report deterministic runtime feasibility without executing a tool."""

        if self.agent_name != "quantitative":
            return False
        if not any(spec.name == "run_experiment" for spec in self.tool_specs()):
            return False
        if not requested_checks or not set(requested_checks).issubset(
            QUANTITATIVE_EXPERIMENT_TYPES
        ):
            return False
        if not set(required_model_capabilities).issubset(
            self.context.model_capabilities
        ):
            return False
        if requires_sample_access and self.context.dataset_root is None:
            return False
        return True

    def execute(self, call: ToolCall | dict[str, Any]) -> ToolResult:
        parsed = call if isinstance(call, ToolCall) else ToolCall.model_validate(call)
        if parsed.tool_call_id in self._call_ids:
            raise ToolBudgetExceeded(
                f"Duplicate tool_call_id: {parsed.tool_call_id}"
            )
        if len(self.history) >= self.budget.max_tool_calls:
            raise ToolBudgetExceeded("Agent tool-call budget is exhausted")
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
