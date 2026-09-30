"""Permission-aware registry and typed execution boundary."""

import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from pydantic import ValidationError

from shortcutminer.tooling.base import Tool, ToolExecutionContext
from shortcutminer.tooling.errors import (
    ToolInputValidationError,
    ToolNotFound,
    ToolPermissionError,
)
from shortcutminer.tooling.protocol import (
    AgentName,
    ToolCall,
    ToolErrorInfo,
    ToolResult,
    ToolSpec,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _publish_experiment_artifacts(
    staged_root: Path, authoritative_root: Path, references: list[str]
) -> None:
    """Publish only completed experiment outputs, with no path escape."""
    checked: list[tuple[Path, Path]] = []
    for reference in references:
        source = (staged_root / reference).resolve(strict=True)
        destination = (authoritative_root / reference).resolve(strict=False)
        if not source.is_file() or not source.is_relative_to(staged_root.resolve()):
            raise ValueError("Experiment artifact path escapes its staging directory")
        if not destination.is_relative_to(authoritative_root.resolve()):
            raise ValueError("Experiment artifact path escapes the run directory")
        checked.append((source, destination))
    for source, destination in checked:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".publish-", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


class ToolRegistry:
    """Register named tools, enforce Agent permissions, and execute safely."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._permissions: dict[str, frozenset[AgentName]] = {}

    def register(self, tool: Tool, *, agents: set[AgentName]) -> None:
        spec = tool.spec()
        if spec.name in self._tools:
            raise ValueError(f"Tool is already registered: {spec.name}")
        if not agents:
            raise ValueError(f"Tool must grant at least one Agent: {spec.name}")
        self._tools[spec.name] = tool
        self._permissions[spec.name] = frozenset(agents)

    def resolve(self, tool_name: str) -> Tool:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise ToolNotFound(f"Unknown tool: {tool_name}") from exc

    def is_allowed(self, agent_name: AgentName, tool_name: str) -> bool:
        self.resolve(tool_name)
        return agent_name in self._permissions[tool_name]

    def specs_for(self, agent_name: AgentName) -> list[ToolSpec]:
        return [
            self._tools[name].spec()
            for name in sorted(self._tools)
            if agent_name in self._permissions[name]
        ]

    def permission_matrix(self) -> dict[str, dict[AgentName, bool]]:
        agents: tuple[AgentName, ...] = (
            "identify",
            "executive",
            "quantitative",
            "qualitative",
            "summarize",
        )
        return {
            name: {agent: agent in self._permissions[name] for agent in agents}
            for name in sorted(self._tools)
        }

    def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        *,
        timeout_seconds: float,
    ) -> ToolResult:
        tool = self.resolve(call.tool_name)
        if call.agent_name not in self._permissions[call.tool_name]:
            raise ToolPermissionError(
                f"Agent {call.agent_name!r} cannot call {call.tool_name!r}"
            )
        if call.agent_name != context.agent_name or call.run_id != context.run_id:
            raise ToolPermissionError(
                "ToolCall identity does not match its execution context"
            )
        try:
            arguments = tool.input_model.model_validate(call.arguments)
        except ValidationError as exc:
            raise ToolInputValidationError(
                f"Invalid input for tool {call.tool_name}: {exc}"
            ) from exc
        for field_name in ("candidate_id", "task_id"):
            argument_value = getattr(arguments, field_name, None)
            call_value = getattr(call, field_name)
            if argument_value is not None and call_value != argument_value:
                raise ToolInputValidationError(
                    f"ToolCall {field_name} does not match its arguments"
                )

        started_at = _utc_now()
        started = monotonic()
        stage_directory: Path | None = None
        execution_context = context
        executor: ThreadPoolExecutor | None = None
        future = None
        cleanup_deferred = False
        try:
            if call.tool_name == "run_experiment" and context.runs_root is not None:
                authoritative_root = (context.runs_root / call.run_id).resolve(strict=True)
                stage_directory = Path(tempfile.mkdtemp(prefix="shortcutminer-experiment-"))
                staged_root = stage_directory / call.run_id
                staged_root.mkdir()
                if call.task_id is not None:
                    previous_task_artifacts = authoritative_root / "experiments" / call.task_id
                    resolved_task_artifacts = previous_task_artifacts.resolve(strict=False)
                    if (
                        resolved_task_artifacts.is_relative_to(authoritative_root)
                        and previous_task_artifacts.is_dir()
                        and not previous_task_artifacts.is_symlink()
                    ):
                        if any(path.is_symlink() for path in previous_task_artifacts.rglob("*")):
                            raise ValueError("Experiment task artifact tree contains a symlink")
                        shutil.copytree(
                            previous_task_artifacts,
                            staged_root / "experiments" / call.task_id,
                        )
                for reference in context.approved_artifacts:
                    source = (authoritative_root / reference).resolve(strict=False)
                    if not source.is_relative_to(authoritative_root) or not source.is_file():
                        continue
                    destination = staged_root / reference
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                execution_context = replace(context, runs_root=stage_directory)
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(tool.execute, arguments, execution_context)
            raw_output = future.result(timeout=timeout_seconds)
            output = tool.output_model.model_validate(raw_output)
            serialized = output.model_dump(mode="json")
            json.dumps(serialized, allow_nan=False)
            if stage_directory is not None:
                _publish_experiment_artifacts(
                    stage_directory / call.run_id,
                    authoritative_root,
                    list(getattr(output, "artifact_references", [])),
                )
            status = "success"
            error = None
            artifacts = list(getattr(output, "artifact_references", []))
        except FutureTimeoutError:
            if future is not None:
                future.cancel()
            if stage_directory is not None:
                future.add_done_callback(
                    lambda _future, path=stage_directory: shutil.rmtree(path, ignore_errors=True)
                )
                cleanup_deferred = True
            serialized = None
            status = "failed"
            error = ToolErrorInfo(
                error_type="ToolTimeoutError",
                message=(
                    f"Tool {call.tool_name} exceeded {timeout_seconds:g} seconds"
                ),
            )
            artifacts = []
        except Exception as exc:
            serialized = None
            status = "failed"
            error = ToolErrorInfo(
                error_type="ToolExecutionError",
                message=str(exc).strip() or type(exc).__name__,
            )
            artifacts = []
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            if stage_directory is not None and not cleanup_deferred:
                shutil.rmtree(stage_directory, ignore_errors=True)

        return ToolResult(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            agent_name=call.agent_name,
            run_id=call.run_id,
            candidate_id=call.candidate_id,
            task_id=call.task_id,
            plan_step_id=call.plan_step_id,
            planning_check=call.planning_check,
            experiment_type=call.experiment_type,
            status=status,
            output=serialized,
            error=error,
            artifacts=artifacts,
            metadata={
                "timeout_seconds": timeout_seconds,
                **(
                    {"failure_category": "backend_execution_failure"}
                    if status == "failed" else {}
                ),
            },
            started_at=started_at,
            completed_at=_utc_now(),
            duration_ms=(monotonic() - started) * 1000,
        )
