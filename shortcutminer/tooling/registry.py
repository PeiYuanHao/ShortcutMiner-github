"""Permission-aware registry and typed execution boundary."""

import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
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
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(tool.execute, arguments, context)
        try:
            raw_output = future.result(timeout=timeout_seconds)
            output = tool.output_model.model_validate(raw_output)
            serialized = output.model_dump(mode="json")
            json.dumps(serialized, allow_nan=False)
            status = "success"
            error = None
        except FutureTimeoutError:
            future.cancel()
            serialized = None
            status = "failed"
            error = ToolErrorInfo(
                error_type="ToolTimeoutError",
                message=(
                    f"Tool {call.tool_name} exceeded {timeout_seconds:g} seconds"
                ),
            )
        except Exception as exc:
            serialized = None
            status = "failed"
            error = ToolErrorInfo(
                error_type="ToolExecutionError",
                message=str(exc).strip() or type(exc).__name__,
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return ToolResult(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            agent_name=call.agent_name,
            run_id=call.run_id,
            candidate_id=call.candidate_id,
            task_id=call.task_id,
            status=status,
            output=serialized,
            error=error,
            artifacts=[],
            metadata={"timeout_seconds": timeout_seconds},
            started_at=started_at,
            completed_at=_utc_now(),
            duration_ms=(monotonic() - started) * 1000,
        )
