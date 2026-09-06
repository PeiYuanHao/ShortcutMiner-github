"""Validated execution boundary for the Summarize Agent."""

from pathlib import Path

from pydantic import ValidationError

from shortcutminer.agent_client import AgentClient
from shortcutminer.experiment.agent_workspace import load_agent_workspace_prompt
from shortcutminer.protocol.summarize import SummarizeRequest, SummaryReport
from shortcutminer.storage.events import append_event
from shortcutminer.storage.stage_artifacts import save_stage_json
from shortcutminer.tooling.agent_loop import invoke_agent_with_tools
from shortcutminer.tooling.runtime import ToolRuntime


class SummarizeExecutionError(RuntimeError):
    """Raised when Summarize cannot produce a valid final report."""


def build_summarize_agent_payload(request: SummarizeRequest) -> dict:
    return {"request": request.model_dump(mode="json")}


def validate_summarize_request(request: SummarizeRequest) -> None:
    if request.in_reply_to != request.executive_result.message_id:
        raise SummarizeExecutionError(
            "SummarizeRequest in_reply_to does not match RoutingPlan."
        )
    inputs = (
        request.identify_result,
        request.executive_result,
        request.quantitative_result,
        request.qualitative_result,
    )
    if any(item.run_id != request.run_id for item in inputs):
        raise SummarizeExecutionError("Summarize inputs do not share one run_id.")
    quantitative_tasks = [
        task.task_id
        for task in request.executive_result.tasks
        if task.target == "quantitative"
    ]
    qualitative_tasks = [
        task.task_id
        for task in request.executive_result.tasks
        if task.target == "qualitative"
    ]
    if [
        result.payload.task_id
        for result in request.quantitative_result.results
    ] != quantitative_tasks:
        raise SummarizeExecutionError(
            "Quantitative branch results do not match planned tasks."
        )
    qualitative_result_tasks = [
        result.payload.task_id
        for result in request.qualitative_result.results
    ]
    expected_qualitative_tasks = [
        task.task_id
        for task in request.executive_result.tasks
        if task.target == "qualitative"
    ]
    if qualitative_result_tasks != expected_qualitative_tasks:
        raise SummarizeExecutionError(
            "Qualitative branch results do not match planned tasks."
        )
    for task, result in zip(
        (
            task
            for task in request.executive_result.tasks
            if task.target == "quantitative"
        ),
        request.quantitative_result.results,
        strict=True,
    ):
        expected = f"{request.run_id}:quantitative:{task.task_id}:request"
        if result.in_reply_to != expected:
            raise SummarizeExecutionError(
                "Quantitative branch result has invalid request correlation."
            )
    for task, result in zip(
        (
            task
            for task in request.executive_result.tasks
            if task.target == "qualitative"
        ),
        request.qualitative_result.results,
        strict=True,
    ):
        expected = f"{request.run_id}:qualitative:{task.task_id}:request"
        if result.in_reply_to != expected:
            raise SummarizeExecutionError(
                "Qualitative branch result has invalid request correlation."
            )


def validate_summary_report(
    report: SummaryReport,
    request: SummarizeRequest,
) -> None:
    validate_summarize_request(request)
    if report.protocol_version != request.protocol_version:
        raise SummarizeExecutionError("Summary protocol_version changed.")
    if report.run_id != request.run_id:
        raise SummarizeExecutionError("Summary run_id changed.")
    if report.in_reply_to != request.message_id:
        raise SummarizeExecutionError(
            "SummaryReport in_reply_to does not match request."
        )
    if report.message_id in {
        request.message_id,
        request.executive_result.message_id,
        request.identify_result.message_id,
    }:
        raise SummarizeExecutionError(
            "SummaryReport message_id must identify a new message."
        )
    candidate_ids = [
        candidate.candidate_id for candidate in request.identify_result.candidates
    ]
    finding_ids = [finding.candidate_id for finding in report.payload.findings]
    if finding_ids != candidate_ids:
        raise SummarizeExecutionError(
            "SummaryReport must contain one ordered finding per candidate."
        )
    quantitative_ids = {
        result.payload.candidate_id: result.message_id
        for result in request.quantitative_result.results
    }
    qualitative_ids = {
        result.payload.candidate_id: result.message_id
        for result in request.qualitative_result.results
    }
    for finding in report.payload.findings:
        expected_quantitative = quantitative_ids.get(finding.candidate_id)
        if set(finding.quantitative_result_ids) != (
            {expected_quantitative} if expected_quantitative else set()
        ):
            raise SummarizeExecutionError(
                "Summary finding does not reference its quantitative result."
            )
        expected_qualitative = qualitative_ids.get(finding.candidate_id)
        if set(finding.qualitative_result_ids) != (
            {expected_qualitative} if expected_qualitative else set()
        ):
            raise SummarizeExecutionError(
                "Summary finding does not reference its qualitative result."
            )

    allowed_artifacts = set(request.identify_result.artifact_references)
    allowed_artifacts.update(request.executive_result.artifact_references)
    for task in request.executive_result.tasks:
        allowed_artifacts.update(task.inputs.artifact_references)
    for result in request.quantitative_result.results:
        allowed_artifacts.update(result.payload.artifact_references)
    for result in request.qualitative_result.results:
        allowed_artifacts.update(result.payload.artifact_references)
        for failure in result.payload.failures:
            allowed_artifacts.update(failure.artifact_references)
    unknown_artifacts = set(report.payload.artifact_references) - allowed_artifacts
    if unknown_artifacts:
        raise SummarizeExecutionError(
            "SummaryReport references artifacts absent from validated upstream "
            "inputs: " + ", ".join(sorted(unknown_artifacts))
        )


def execute_summarize(
    request: SummarizeRequest,
    client: AgentClient,
    runs_root: str | Path | None = None,
    workspace_dir: str | Path | None = None,
    tool_runtime: ToolRuntime | None = None,
) -> SummaryReport:
    """Invoke, validate, and optionally persist the final report."""
    try:
        validate_summarize_request(request)
        if runs_root is not None:
            append_event(
                runs_root,
                request.run_id,
                "SUMMARIZE_STARTED",
                stage="summarize",
            )
        payload = build_summarize_agent_payload(request)
        prompt = load_agent_workspace_prompt("summarize", workspace_dir)
        if runs_root is not None:
            save_stage_json(
                runs_root,
                run_id=request.run_id,
                stage="summarize",
                filename="input.json",
                payload=payload,
            )
        try:
            invocation = invoke_agent_with_tools(
                client,
                "summarize",
                payload,
                system_prompt=prompt,
                tool_runtime=tool_runtime,
            )
            raw_output = invocation.output
        except Exception as exc:
            raise SummarizeExecutionError(
                f"Summarize agent invocation failed: {exc}"
            ) from exc
        if not isinstance(raw_output, dict):
            raise SummarizeExecutionError(
                "Summarize agent returned a non-object response."
            )
        if runs_root is not None:
            save_stage_json(
                runs_root,
                run_id=request.run_id,
                stage="summarize",
                filename="raw_output.json",
                payload=raw_output,
            )
        try:
            report = SummaryReport.model_validate(raw_output)
        except ValidationError as exc:
            raise SummarizeExecutionError(
                "Summarize agent returned an invalid SummaryReport."
            ) from exc
        validate_summary_report(report, request)
        if runs_root is not None:
            save_stage_json(
                runs_root,
                run_id=report.run_id,
                stage="summarize",
                filename="result.json",
                payload=report,
            )
            append_event(
                runs_root,
                report.run_id,
                "SUMMARIZE_COMPLETED",
                stage="summarize",
                details={"finding_count": len(report.payload.findings)},
            )
        return report
    except Exception as exc:
        if runs_root is not None:
            try:
                append_event(
                    runs_root,
                    request.run_id,
                    "AGENT_FAILED",
                    stage="summarize",
                    details={
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            except Exception as event_exc:
                exc.add_note(f"Could not persist failure event: {event_exc}")
        if isinstance(exc, SummarizeExecutionError):
            raise
        raise SummarizeExecutionError(
            f"Summarize execution failed: {exc}"
        ) from exc
