"""Optional provider-neutral tool loop used by Agent executors."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from shortcutminer.agent_client import AgentClient, invoke_agent_client
from shortcutminer.protocol.executive import RoutingPlan
from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.protocol.qualitative import QualitativeResult
from shortcutminer.protocol.quantitative import QuantitativeResult
from shortcutminer.protocol.summarize import SummaryReport
from shortcutminer.tooling.errors import ToolBudgetExceeded
from shortcutminer.tooling.protocol import ToolCall, ToolResult
from shortcutminer.tooling.runtime import ToolRuntime


@dataclass(frozen=True)
class AgentToolInvocation:
    output: object
    tool_results: tuple[ToolResult, ...]


_FINAL_SCHEMAS = {
    "identify": CandidateReport,
    "executive": RoutingPlan,
    "quantitative": QuantitativeResult,
    "qualitative": QualitativeResult,
    "summarize": SummaryReport,
}


def _validate_provider_output(agent_name: str, output: dict[str, Any]) -> None:
    if "tool_calls" in output:
        if set(output) != {"tool_calls"} or not isinstance(output["tool_calls"], list):
            raise ValueError("Tool-call response must contain only a tool_calls list")
        if not output["tool_calls"]:
            raise ValueError("tool_calls must not be empty")
        for raw_call in output["tool_calls"]:
            ToolCall.model_validate(raw_call)
        return
    schema = _FINAL_SCHEMAS.get(agent_name)
    if schema is None:
        raise ValueError(f"No final response schema is registered for {agent_name}")
    schema.model_validate(output)


def _invoke(
    client: AgentClient,
    agent_name: str,
    payload: dict[str, Any],
    *,
    system_prompt: str | None,
) -> object:
    validated = getattr(client, "invoke_agent_validated", None)
    if callable(validated):
        return validated(
            agent_name,
            payload,
            system_prompt=system_prompt,
            validator=lambda output: _validate_provider_output(agent_name, output),
        )
    return invoke_agent_client(
        client,
        agent_name,
        payload,
        system_prompt=system_prompt,
    )


def invoke_agent_with_tools(
    client: AgentClient,
    agent_name: str,
    payload: dict[str, Any],
    *,
    system_prompt: str | None,
    tool_runtime: ToolRuntime | None,
) -> AgentToolInvocation:
    """Invoke once normally, or service bounded ToolCall batches until final output."""
    if tool_runtime is None:
        return AgentToolInvocation(
            output=_invoke(
                client, agent_name, payload, system_prompt=system_prompt
            ),
            tool_results=(),
        )
    if tool_runtime.agent_name != agent_name:
        raise ValueError("Tool Runtime is bound to a different Agent")

    base_payload = deepcopy(payload)
    iterations = 0
    while True:
        iterations += 1
        if iterations > tool_runtime.budget.max_tool_calls + 1:
            raise ToolBudgetExceeded("Agent tool loop exceeded its iteration budget")
        invocation_payload = {
            **deepcopy(base_payload),
            "tool_runtime": {
                "tools": [
                    spec.model_dump(mode="json") for spec in tool_runtime.tool_specs()
                ],
                "results": [
                    result.model_dump(mode="json")
                    for result in tool_runtime.history
                ],
                "remaining_calls": (
                    tool_runtime.budget.max_tool_calls - len(tool_runtime.history)
                ),
            },
        }
        response = _invoke(
            client,
            agent_name,
            invocation_payload,
            system_prompt=system_prompt,
        )
        if not isinstance(response, dict) or "tool_calls" not in response:
            return AgentToolInvocation(
                output=response,
                tool_results=tuple(tool_runtime.history),
            )
        if set(response) != {"tool_calls"} or not isinstance(
            response["tool_calls"], list
        ):
            raise ValueError("Tool-call response must contain only a tool_calls list")
        if not response["tool_calls"]:
            raise ValueError("tool_calls must not be empty")
        for raw_call in response["tool_calls"]:
            call = ToolCall.model_validate(raw_call)
            tool_runtime.execute(call)
