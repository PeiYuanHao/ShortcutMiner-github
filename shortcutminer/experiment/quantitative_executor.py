"""Validated execution boundary for the Quantitative Agent."""

from pathlib import Path

from pydantic import ValidationError

from shortcutminer.agent_client import AgentClient
from shortcutminer.experiment.agent_workspace import load_agent_workspace_prompt
from shortcutminer.protocol.quantitative import (
    MetricValue,
    QuantitativeAgentDraft,
    QuantitativeRequest,
    QuantitativeResult,
    QuantitativeResultPayload,
)
from shortcutminer.storage.events import append_event
from shortcutminer.storage.stage_artifacts import save_stage_json
from shortcutminer.tooling.agent_loop import invoke_agent_with_tools
from shortcutminer.tooling.builtins import ExperimentToolOutput
from shortcutminer.tooling.protocol import ToolResult
from shortcutminer.tooling.runtime import ToolRuntime


class QuantitativeExecutionError(RuntimeError):
    """Raised when Quantitative cannot produce a valid result."""


class QuantitativeDraftInvariantError(QuantitativeExecutionError):
    """Raised when a schema-valid draft cites unusable quantitative evidence."""


def build_quantitative_agent_payload(request: QuantitativeRequest) -> dict:
    return {"request": request.model_dump(mode="json")}


def validate_quantitative_request(request: QuantitativeRequest) -> None:
    if request.in_reply_to != request.task.message_id:
        raise QuantitativeExecutionError(
            "QuantitativeRequest in_reply_to does not match AnalysisTask."
        )
    if request.task.target != "quantitative":
        raise QuantitativeExecutionError(
            "QuantitativeRequest contains a non-quantitative task."
        )
    if request.task.run_id != request.run_id:
        raise QuantitativeExecutionError(
            "AnalysisTask run_id does not match QuantitativeRequest."
        )
    if request.task.candidate_id != request.candidate.candidate_id:
        raise QuantitativeExecutionError(
            "AnalysisTask candidate_id does not match Candidate."
        )
    if request.task.hypothesis != request.candidate.hypothesis:
        raise QuantitativeExecutionError(
            "AnalysisTask changed the Candidate hypothesis."
        )


def validate_quantitative_result(
    result: QuantitativeResult,
    request: QuantitativeRequest,
    *,
    tool_results: tuple[ToolResult, ...] = (),
) -> None:
    validate_quantitative_request(request)
    payload = result.payload
    if result.protocol_version != request.protocol_version:
        raise QuantitativeExecutionError("Quantitative protocol_version changed.")
    if result.run_id != request.run_id:
        raise QuantitativeExecutionError("Quantitative result run_id changed.")
    if result.in_reply_to != request.message_id:
        raise QuantitativeExecutionError(
            "QuantitativeResult in_reply_to does not match request."
        )
    if result.message_id in {request.message_id, request.task.message_id}:
        raise QuantitativeExecutionError(
            "QuantitativeResult message_id must identify a new message."
        )
    if payload.task_id != request.task.task_id:
        raise QuantitativeExecutionError("Quantitative result changed task_id.")
    if payload.candidate_id != request.candidate.candidate_id:
        raise QuantitativeExecutionError("Quantitative result changed candidate_id.")
    if payload.hypothesis != request.candidate.hypothesis:
        raise QuantitativeExecutionError("Quantitative result changed hypothesis.")
    allowed_artifacts = set(request.task.inputs.artifact_references)
    for tool_result in tool_results:
        if tool_result.status == "success":
            allowed_artifacts.update(tool_result.artifacts)
    if not set(payload.artifact_references).issubset(allowed_artifacts):
        raise QuantitativeExecutionError(
            "Quantitative result contains an unauthorized artifact reference."
        )
    missing = [
        limitation
        for limitation in request.limitations
        if limitation not in payload.limitations
    ]
    if missing:
        raise QuantitativeExecutionError(
            "Quantitative result omitted upstream limitations: "
            + "; ".join(missing)
        )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _selected_evidence(
    draft: QuantitativeAgentDraft,
    request: QuantitativeRequest,
    tool_results: tuple[ToolResult, ...],
) -> list[ToolResult]:
    by_id = {result.tool_call_id: result for result in tool_results}
    selected: list[ToolResult] = []
    for tool_call_id in draft.evidence_tool_call_ids:
        result = by_id.get(tool_call_id)
        if result is None:
            raise QuantitativeDraftInvariantError(
                f"unknown evidence_tool_call_id: {tool_call_id}"
            )
        if result.agent_name != "quantitative":
            raise QuantitativeDraftInvariantError(
                f"evidence ToolResult {tool_call_id} belongs to another agent"
            )
        if result.run_id != request.run_id:
            raise QuantitativeDraftInvariantError(
                f"evidence ToolResult {tool_call_id} belongs to another run"
            )
        if result.candidate_id != request.candidate.candidate_id:
            raise QuantitativeDraftInvariantError(
                f"evidence ToolResult {tool_call_id} belongs to another candidate"
            )
        if result.task_id != request.task.task_id:
            raise QuantitativeDraftInvariantError(
                f"evidence ToolResult {tool_call_id} belongs to another task"
            )
        if result.status != "success":
            raise QuantitativeDraftInvariantError(
                f"evidence ToolResult {tool_call_id} did not succeed"
            )
        if result.tool_name != "run_experiment":
            raise QuantitativeDraftInvariantError(
                f"evidence ToolResult {tool_call_id} is not a run_experiment result"
            )
        try:
            ExperimentToolOutput.model_validate(result.output)
        except ValidationError as exc:
            raise QuantitativeDraftInvariantError(
                f"evidence ToolResult {tool_call_id} has an invalid typed output"
            ) from exc
        selected.append(result)
    if draft.status == "completed" and not selected:
        raise QuantitativeDraftInvariantError(
            "completed quantitative draft requires selected successful evidence"
        )
    return selected


def extract_quantitative_metrics(
    selected_results: list[ToolResult],
) -> dict[str, MetricValue]:
    """Extract typed scalar metrics without interpreting their scientific meaning."""

    outputs = [
        (result.tool_call_id, ExperimentToolOutput.model_validate(result.output))
        for result in selected_results
    ]
    counts: dict[str, int] = {}
    for _, output in outputs:
        for name in output.metrics:
            counts[name] = counts.get(name, 0) + 1
    metrics: dict[str, MetricValue] = {}
    for tool_call_id, output in outputs:
        for name, value in output.metrics.items():
            key = f"{tool_call_id}.{name}" if counts[name] > 1 else name
            metrics[key] = value
    return metrics


def finalize_quantitative_result(
    draft: QuantitativeAgentDraft,
    request: QuantitativeRequest,
    *,
    tool_results: tuple[ToolResult, ...] = (),
) -> QuantitativeResult:
    """Construct the canonical result from provider reasoning and authoritative tools."""

    selected = _selected_evidence(draft, request, tool_results)
    execution_limitations: list[str] = []
    for result in selected:
        output = ExperimentToolOutput.model_validate(result.output)
        raw_limitations = output.metadata.get("limitations")
        if isinstance(raw_limitations, list) and all(
            isinstance(item, str) and item.strip() for item in raw_limitations
        ):
            execution_limitations.extend(raw_limitations)
    metrics = (
        extract_quantitative_metrics(selected)
        if draft.status == "completed"
        else {}
    )
    artifacts = _unique(
        [
            *request.task.inputs.artifact_references,
            *(artifact for result in selected for artifact in result.artifacts),
        ]
    )
    limitations = _unique(
        [
            *request.limitations,
            *draft.additional_limitations,
            *execution_limitations,
        ]
    )
    try:
        result = QuantitativeResult(
            protocol_version=request.protocol_version,
            run_id=request.run_id,
            message_id=(
                f"{request.run_id}:quantitative:{request.task.task_id}:result"
            ),
            source="quantitative",
            target="summarize",
            message_type="quantitative_analysis_result",
            in_reply_to=request.message_id,
            payload=QuantitativeResultPayload(
                task_id=request.task.task_id,
                candidate_id=request.candidate.candidate_id,
                hypothesis=request.candidate.hypothesis,
                status=draft.status,
                analysis_type=draft.analysis_type,
                metrics=metrics,
                evidence_level=draft.evidence_level,
                evidence_direction=draft.evidence_direction,
                evidence_summary=draft.evidence_summary,
                limitations=limitations,
                artifact_references=artifacts,
                recommended_followup=draft.recommended_followup,
            ),
        )
    except ValidationError as exc:
        raise QuantitativeDraftInvariantError(
            "Quantitative draft cannot be finalized as a canonical result."
        ) from exc
    validate_quantitative_result(result, request, tool_results=tool_results)
    return result


def execute_quantitative(
    request: QuantitativeRequest,
    client: AgentClient,
    runs_root: str | Path | None = None,
    workspace_dir: str | Path | None = None,
    tool_runtime: ToolRuntime | None = None,
) -> QuantitativeResult:
    """Invoke, validate, and optionally persist one quantitative task."""
    try:
        if runs_root is not None:
            append_event(
                runs_root,
                request.run_id,
                "QUANTITATIVE_STARTED",
                stage="quantitative",
                details={"task_id": request.task.task_id},
            )
        payload = build_quantitative_agent_payload(request)
        prompt = load_agent_workspace_prompt("quantitative", workspace_dir)
        if tool_runtime is not None:
            tool_runtime.bind_analysis_task(
                runtime_constraints=request.task.inputs.runtime_constraints,
                requested_checks=request.task.requested_checks,
            )
        if runs_root is not None:
            save_stage_json(
                runs_root,
                run_id=request.run_id,
                stage="quantitative",
                scope=request.task.task_id,
                filename="input.json",
                payload=payload,
            )
        try:
            invocation = invoke_agent_with_tools(
                client,
                "quantitative",
                payload,
                system_prompt=prompt,
                tool_runtime=tool_runtime,
            )
            raw_output = invocation.output
        except Exception as exc:
            raise QuantitativeExecutionError(
                f"Quantitative agent invocation failed: {exc}"
            ) from exc
        if not isinstance(raw_output, dict):
            raise QuantitativeExecutionError(
                "Quantitative agent returned a non-object response."
            )
        try:
            draft = QuantitativeAgentDraft.model_validate(raw_output)
        except ValidationError as exc:
            raise QuantitativeExecutionError(
                "Quantitative agent returned an invalid QuantitativeAgentDraft."
            ) from exc
        if runs_root is not None:
            save_stage_json(
                runs_root,
                run_id=request.run_id,
                stage="quantitative",
                scope=request.task.task_id,
                filename="draft.json",
                payload=draft,
            )
        result = finalize_quantitative_result(
            draft,
            request,
            tool_results=invocation.tool_results,
        )
        if runs_root is not None:
            save_stage_json(
                runs_root,
                run_id=result.run_id,
                stage="quantitative",
                scope=request.task.task_id,
                filename="result.json",
                payload=result,
            )
            append_event(
                runs_root,
                result.run_id,
                "QUANTITATIVE_COMPLETED",
                stage="quantitative",
                details={
                    "task_id": request.task.task_id,
                    "status": result.payload.status,
                    "evidence_level": result.payload.evidence_level,
                },
            )
        return result
    except Exception as exc:
        if runs_root is not None:
            try:
                append_event(
                    runs_root,
                    request.run_id,
                    "AGENT_FAILED",
                    stage="quantitative",
                    details={
                        "task_id": request.task.task_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            except Exception as event_exc:
                exc.add_note(f"Could not persist failure event: {event_exc}")
        if isinstance(exc, QuantitativeExecutionError):
            raise
        raise QuantitativeExecutionError(
            f"Quantitative execution failed: {exc}"
        ) from exc
