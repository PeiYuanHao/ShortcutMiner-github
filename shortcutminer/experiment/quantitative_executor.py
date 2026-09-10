"""Validated execution boundary for the Quantitative Agent."""

from pathlib import Path

from pydantic import ValidationError

from shortcutminer.agent_client import AgentClient
from shortcutminer.experiment.agent_workspace import load_agent_workspace_prompt
from shortcutminer.protocol.quantitative import (
    QuantitativeRequest,
    QuantitativeResult,
)
from shortcutminer.storage.events import append_event
from shortcutminer.storage.stage_artifacts import save_stage_json
from shortcutminer.tooling.agent_loop import invoke_agent_with_tools
from shortcutminer.tooling.protocol import ToolResult
from shortcutminer.tooling.runtime import ToolRuntime


class QuantitativeExecutionError(RuntimeError):
    """Raised when Quantitative cannot produce a valid result."""


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
        if runs_root is not None:
            save_stage_json(
                runs_root,
                run_id=request.run_id,
                stage="quantitative",
                scope=request.task.task_id,
                filename="raw_output.json",
                payload=raw_output,
            )
        try:
            result = QuantitativeResult.model_validate(raw_output)
        except ValidationError as exc:
            raise QuantitativeExecutionError(
                "Quantitative agent returned an invalid QuantitativeResult."
            ) from exc
        validate_quantitative_result(
            result,
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
