"""Optional provider-neutral tool loop used by Agent executors."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from shortcutminer.agent_client import AgentClient, invoke_agent_client
from shortcutminer.protocol.executive import RoutingPlan
from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.protocol.qualitative import QualitativeAgentDraft
from shortcutminer.protocol.quantitative import (
    QuantitativeAgentDraft,
    QuantitativeExecutionRequirement,
    QuantitativeExecutionResources,
)
from shortcutminer.protocol.summarize import SummaryReport
from shortcutminer.tooling.errors import ToolBudgetExceeded, ToolNotFound
from shortcutminer.tooling.errors import ToolInputValidationError, ToolPermissionError
from shortcutminer.tooling.protocol import ToolCall, ToolCallDraft, ToolResult
from shortcutminer.tooling.runtime import ToolRuntime
from shortcutminer.experiment.execution_contract import (
    AuthoritativeResourceMissing,
    CanonicalExperimentInputInvalid,
    ToolCallDraftInvalid,
    ToolPermissionFailure,
    classify_experiment_availability,
    finalize_run_experiment_draft,
    provider_run_experiment_schema,
)


@dataclass(frozen=True)
class AgentToolInvocation:
    output: object
    tool_results: tuple[ToolResult, ...]
    execution_obligation_retry_count: int = 0
    provider_tool_draft_count: int = 0
    canonical_tool_call_count: int = 0


class RequiredExperimentNotAttempted(RuntimeError):
    """A mandatory quantitative task finalized before its experiment attempt."""

    category = "required_experiment_not_attempted"

    def __init__(self, *, retry_count: int) -> None:
        super().__init__(self.category)
        self.retry_count = retry_count


_FINAL_SCHEMAS = {
    "identify": CandidateReport,
    "executive": RoutingPlan,
    "quantitative": QuantitativeAgentDraft,
    "qualitative": QualitativeAgentDraft,
    "summarize": SummaryReport,
}


def _validate_provider_output(agent_name: str, output: dict[str, Any]) -> None:
    if "tool_calls" in output:
        if set(output) != {"tool_calls"} or not isinstance(output["tool_calls"], list):
            raise ValueError("Tool-call response must contain only a tool_calls list")
        if not output["tool_calls"]:
            raise ValueError("tool_calls must not be empty")
        for raw_call in output["tool_calls"]:
            ToolCallDraft.model_validate(raw_call)
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


def _authoritative_correlation(
    base_payload: dict[str, Any],
    tool_runtime: ToolRuntime,
) -> tuple[str | None, str | None]:
    request = base_payload.get("request")
    if not isinstance(request, dict):
        request = {}
    request_run_id = request.get("run_id")
    if request_run_id is not None and request_run_id != tool_runtime.run_id:
        raise ValueError("Authoritative request run_id does not match Tool Runtime")
    candidate = request.get("candidate")
    task = request.get("task")
    candidate_id = (
        candidate.get("candidate_id") if isinstance(candidate, dict) else None
    )
    task_id = task.get("task_id") if isinstance(task, dict) else None
    return candidate_id, task_id


def finalize_tool_call_draft(
    draft: ToolCallDraft | dict[str, Any],
    *,
    agent_name: str,
    base_payload: dict[str, Any],
    tool_runtime: ToolRuntime,
    iteration: int,
    call_index: int,
) -> ToolCall:
    """Inject authoritative identity into one strict provider tool-call draft."""

    try:
        parsed = (
            draft if isinstance(draft, ToolCallDraft)
            else ToolCallDraft.model_validate(draft)
        )
    except Exception as exc:
        raise ToolCallDraftInvalid("provider tool-call draft is invalid") from exc
    if tool_runtime.agent_name != agent_name:
        raise ValueError("Tool Runtime is bound to a different Agent")
    available_names = {spec.name for spec in tool_runtime.tool_specs()}
    if parsed.name not in available_names:
        raise ToolNotFound(
            f"ToolCallDraft names unavailable tool: {parsed.name}"
        )
    candidate_id, task_id = _authoritative_correlation(base_payload, tool_runtime)
    arguments = parsed.arguments
    if parsed.name == "run_experiment":
        if candidate_id is None or task_id is None:
            raise AuthoritativeResourceMissing(
                "run_experiment requires authoritative candidate/task correlation"
            )
        try:
            resources = QuantitativeExecutionResources.model_validate(
                base_payload.get("execution_resources")
            )
        except Exception as exc:
            raise AuthoritativeResourceMissing(
                "quantitative execution resources are missing or invalid"
            ) from exc
        requirement = QuantitativeExecutionRequirement.model_validate(
            base_payload.get("execution_requirement")
        )
        dataset_root = tool_runtime.context.dataset_root
        if dataset_root is None:
            raise AuthoritativeResourceMissing("Tool Runtime has no dataset root")
        runs_root = tool_runtime.context.runs_root
        run_root = (
            runs_root / tool_runtime.run_id if runs_root is not None else None
        )
        canonical = finalize_run_experiment_draft(
            arguments,
            resources=resources,
            requirement=requirement,
            run_id=tool_runtime.run_id,
            candidate_id=candidate_id,
            task_id=task_id,
            dataset_root=dataset_root,
            run_root=run_root,
            approved_artifacts=set(tool_runtime.context.approved_artifacts),
            history=tool_runtime.history,
        )
        arguments = canonical.model_dump(mode="json")
    arguments_hash = hashlib.sha256(
        json.dumps(
            arguments,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    history_hash = hashlib.sha256(
        json.dumps(
            [item.tool_call_id for item in tool_runtime.history],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    identity = {
        "run_id": tool_runtime.run_id,
        "agent_name": agent_name,
        "candidate_id": candidate_id,
        "task_id": task_id,
        "tool_name": parsed.name,
        "iteration": iteration,
        "call_index": call_index,
        "arguments_hash": arguments_hash,
        "history_hash": history_hash,
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()[:32]
    return ToolCall.model_validate({
        "protocol_version": "1.0",
        "tool_call_id": f"TC-{digest}",
        "tool_name": parsed.name,
        "agent_name": agent_name,
        "run_id": tool_runtime.run_id,
        "candidate_id": candidate_id,
        "task_id": task_id,
        "arguments": arguments,
    })


def invoke_agent_with_tools(
    client: AgentClient,
    agent_name: str,
    payload: dict[str, Any],
    *,
    system_prompt: str | None,
    tool_runtime: ToolRuntime | None,
) -> AgentToolInvocation:
    """Invoke once normally, or service bounded ToolCall batches until final output."""
    requirement = None
    if agent_name == "quantitative" and "execution_requirement" in payload:
        requirement = QuantitativeExecutionRequirement.model_validate(
            payload["execution_requirement"]
        )
    obligation_required = bool(
        requirement is not None and requirement.must_attempt_run_experiment
    )
    if tool_runtime is None:
        if obligation_required:
            raise RequiredExperimentNotAttempted(retry_count=0)
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
    execution_obligation_retry_count = 0
    provider_tool_draft_count = 0
    canonical_tool_call_count = 0

    def required_attempt_exists() -> bool:
        if not obligation_required:
            return True
        request = base_payload.get("request", {})
        task = request.get("task", {})
        candidate = request.get("candidate", {})
        return any(
            result.tool_name == "run_experiment"
            and result.agent_name == "quantitative"
            and result.run_id == request.get("run_id")
            and result.task_id == task.get("task_id")
            and result.candidate_id == candidate.get("candidate_id")
            for result in tool_runtime.history
        )

    def correct_or_fail() -> None:
        nonlocal execution_obligation_retry_count
        if execution_obligation_retry_count >= 1:
            raise RequiredExperimentNotAttempted(
                retry_count=execution_obligation_retry_count
            )
        execution_obligation_retry_count += 1

    while True:
        iterations += 1
        if iterations > tool_runtime.budget.max_tool_calls + 2:
            raise ToolBudgetExceeded("Agent tool loop exceeded its iteration budget")
        current_requirement = requirement
        if requirement is not None:
            available, pending, unsupported = classify_experiment_availability(
                requirement.resolved_experiment_types,
                history=tool_runtime.history,
                approved_artifacts=set(tool_runtime.context.approved_artifacts),
            )
            current_requirement = requirement.model_copy(update={
                "available_now_experiment_types": available,
                "pending_dependency_experiment_types": pending,
                "unsupported_experiment_types": unsupported,
            })
        tool_specs = []
        for spec in tool_runtime.tool_specs():
            value = spec.model_dump(mode="json")
            if agent_name == "quantitative" and spec.name == "run_experiment":
                value["input_schema"] = provider_run_experiment_schema()
            tool_specs.append(value)
        invocation_payload = {
            **deepcopy(base_payload),
            **(
                {
                    "execution_requirement": current_requirement.model_dump(
                        mode="json"
                    )
                }
                if current_requirement is not None else {}
            ),
            "tool_runtime": {
                "tools": tool_specs,
                "results": [
                    result.model_dump(mode="json")
                    for result in tool_runtime.history
                ],
                "remaining_calls": (
                    tool_runtime.budget.max_tool_calls - len(tool_runtime.history)
                ),
            },
        }
        if execution_obligation_retry_count:
            invocation_payload["execution_obligation_correction"] = {
                "category": "required_experiment_not_attempted",
                "instruction": (
                    "The assigned quantitative task has already passed deterministic "
                    "feasibility checks and an authorized run_experiment tool is "
                    "available. You must attempt the required measurement before "
                    "returning a final QuantitativeAgentDraft. Return a tool_calls "
                    "object invoking run_experiment using the supplied tool schema. "
                    "Do not return blocked merely because no measurement has been "
                    "run yet."
                ),
            }
        response = _invoke(
            client,
            agent_name,
            invocation_payload,
            system_prompt=system_prompt,
        )
        if not isinstance(response, dict) or "tool_calls" not in response:
            if not required_attempt_exists():
                correct_or_fail()
                continue
            return AgentToolInvocation(
                output=response,
                tool_results=tuple(tool_runtime.history),
                execution_obligation_retry_count=execution_obligation_retry_count,
                provider_tool_draft_count=provider_tool_draft_count,
                canonical_tool_call_count=canonical_tool_call_count,
            )
        if set(response) != {"tool_calls"} or not isinstance(
            response["tool_calls"], list
        ):
            raise ValueError("Tool-call response must contain only a tool_calls list")
        if not response["tool_calls"]:
            raise ValueError("tool_calls must not be empty")
        try:
            drafts = [
                ToolCallDraft.model_validate(item) for item in response["tool_calls"]
            ]
        except Exception as exc:
            raise ToolCallDraftInvalid("provider tool-call draft is invalid") from exc
        provider_tool_draft_count += len(drafts)
        calls = [
            finalize_tool_call_draft(
                draft,
                agent_name=agent_name,
                base_payload=base_payload,
                tool_runtime=tool_runtime,
                iteration=iterations,
                call_index=call_index,
            )
            for call_index, draft in enumerate(drafts)
        ]
        canonical_tool_call_count += len(calls)
        if not required_attempt_exists():
            if any(
                call.tool_name != "run_experiment"
                for call in calls
            ):
                correct_or_fail()
                continue
        for call in calls:
            try:
                tool_runtime.execute(call)
            except ToolPermissionError as exc:
                raise ToolPermissionFailure(
                    "canonical tool call was rejected by runtime permissions"
                ) from exc
            except ToolInputValidationError as exc:
                raise CanonicalExperimentInputInvalid(
                    "canonical tool input was rejected by Tool Runtime"
                ) from exc
