"""Validated execution boundary for the Summarize Agent."""

from pathlib import Path

from pydantic import ValidationError

from shortcutminer.agent_client import AgentClient
from shortcutminer.experiment.agent_workspace import load_agent_workspace_prompt
from shortcutminer.protocol.summarize import (
    SummarizeAgentDraft, SummarizeRequest, SummaryFinding, SummaryReport,
    SummaryReportPayload,
)
from shortcutminer.storage.events import append_event
from shortcutminer.storage.stage_artifacts import save_stage_json
from shortcutminer.tooling.agent_loop import ProviderContractValidationError, invoke_agent_with_tools
from shortcutminer.tooling.runtime import ToolRuntime


class SummarizeExecutionError(RuntimeError):
    """Raised when Summarize cannot produce a valid final report."""


def allowed_assessments(consistency: str) -> set[str]:
    return {
        "no_usable_evidence": {"insufficient_evidence"},
        "inconclusive": {"inconclusive", "insufficient_evidence"},
        "mixed": {"partially_supported", "inconclusive"},
        "consistent_support": {"supported", "partially_supported", "inconclusive"},
        "consistent_contradiction": {"contradicted", "inconclusive"},
    }[consistency]


def validate_summarize_draft(draft: SummarizeAgentDraft, request: SummarizeRequest) -> None:
    expected = [item.candidate_id for item in request.identify_result.candidates]
    actual = [item.candidate_ref for item in draft.findings]
    if actual != expected:
        raise ProviderContractValidationError("candidate_coverage_mismatch", "findings")
    entries = {item.candidate_id: item for item in request.evidence_ledger.candidates}
    for index, finding in enumerate(draft.findings):
        if finding.assessment not in allowed_assessments(entries[finding.candidate_ref].profile.consistency):
            raise ProviderContractValidationError("assessment_overclaim", f"findings.{index}.assessment")


def finalize_summarize_draft(draft: SummarizeAgentDraft, request: SummarizeRequest) -> SummaryReport:
    validate_summarize_draft(draft, request)
    quant = {item.payload.candidate_id: item.message_id for item in request.quantitative_result.results}
    qual = {item.payload.candidate_id: item.message_id for item in request.qualitative_result.results}
    entries = {item.candidate_id: item for item in request.evidence_ledger.candidates}
    artifacts = list(dict.fromkeys([
        *request.identify_result.artifact_references,
        *request.executive_result.artifact_references,
        *(ref for item in request.quantitative_result.results for ref in item.payload.artifact_references),
        *(ref for item in request.qualitative_result.results for ref in item.payload.artifact_references),
        request.evidence_ledger.artifact_reference,
    ]))
    report = SummaryReport(
        protocol_version=request.protocol_version,
        run_id=request.run_id,
        message_id=f"{request.run_id}:summarize:report",
        source="summarize", target="orchestrator", message_type="summary_report",
        in_reply_to=request.message_id,
        payload=SummaryReportPayload(
            overview=draft.overview,
            findings=[SummaryFinding(
                candidate_id=item.candidate_ref,
                assessment=item.assessment,
                summary=item.summary,
                quantitative_result_ids=([quant[item.candidate_ref]] if item.candidate_ref in quant else []),
                qualitative_result_ids=([qual[item.candidate_ref]] if item.candidate_ref in qual else []),
                evidence_record_ids=entries[item.candidate_ref].record_ids,
                limitations=list(dict.fromkeys([*entries[item.candidate_ref].profile.limitations, *item.additional_limitations])),
                recommendations=item.recommendations,
            ) for item in draft.findings],
            limitations=list(dict.fromkeys([*request.evidence_ledger.limitations, *draft.additional_limitations])),
            artifact_references=artifacts,
            evidence_ledger_reference=request.evidence_ledger.artifact_reference,
        ),
    )
    validate_summary_report(report, request)
    return report


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
    if request.evidence_ledger is not None:
        if request.evidence_ledger.run_id != request.run_id:
            raise SummarizeExecutionError("EvidenceLedger run_id does not match request.")
        ledger_ids = [entry.candidate_id for entry in request.evidence_ledger.candidates]
        candidate_ids = [candidate.candidate_id for candidate in request.identify_result.candidates]
        if ledger_ids != candidate_ids:
            raise SummarizeExecutionError(
                "EvidenceLedger must contain every candidate in report order."
            )
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

    if request.evidence_ledger is not None:
        entries = {entry.candidate_id: entry for entry in request.evidence_ledger.candidates}
        for finding in report.payload.findings:
            entry = entries[finding.candidate_id]
            profile = entry.profile
            has_support = bool(profile.supporting_record_ids or profile.mixed_record_ids)
            has_contradiction = bool(profile.contradicting_record_ids or profile.mixed_record_ids)
            if finding.assessment in {"supported", "partially_supported"} and not has_support:
                raise SummarizeExecutionError(
                    f"Summary overclaim: {finding.candidate_id} has no supporting evidence."
                )
            if finding.assessment == "contradicted" and not has_contradiction:
                raise SummarizeExecutionError(
                    f"Summary overclaim: {finding.candidate_id} has no contradictory evidence."
                )
            if profile.consistency == "no_usable_evidence" and finding.assessment != "insufficient_evidence":
                raise SummarizeExecutionError(
                    f"Summary overclaim: {finding.candidate_id} has no usable evidence."
                )
            if profile.consistency == "inconclusive" and finding.assessment not in {
                "inconclusive", "insufficient_evidence"
            }:
                raise SummarizeExecutionError(
                    f"Summary overclaim: {finding.candidate_id} has only inconclusive evidence."
                )
            if profile.consistency == "mixed" and finding.assessment not in {
                "partially_supported", "inconclusive"
            }:
                raise SummarizeExecutionError(
                    f"Summary overclaim: {finding.candidate_id} must preserve mixed evidence."
                )
            unknown_records = set(finding.evidence_record_ids) - set(entry.record_ids)
            if unknown_records:
                raise SummarizeExecutionError(
                    f"Summary finding references unknown EvidenceRecord IDs for {finding.candidate_id}."
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
    if request.evidence_ledger is not None:
        allowed_artifacts.add(request.evidence_ledger.artifact_reference)
    unknown_artifacts = set(report.payload.artifact_references) - allowed_artifacts
    if unknown_artifacts:
        raise SummarizeExecutionError(
            "SummaryReport references artifacts absent from validated upstream "
            "inputs: " + ", ".join(sorted(unknown_artifacts))
        )
    if report.payload.evidence_ledger_reference is not None:
        if request.evidence_ledger is None or report.payload.evidence_ledger_reference != request.evidence_ledger.artifact_reference:
            raise SummarizeExecutionError(
                "SummaryReport evidence ledger reference does not match its request."
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
                provider_contract_validator=lambda output: finalize_summarize_draft(
                    SummarizeAgentDraft.model_validate(output), request
                ),
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
        try:
            draft = SummarizeAgentDraft.model_validate(raw_output)
            report = finalize_summarize_draft(draft, request)
        except ValidationError as exc:
            raise SummarizeExecutionError(
                "Summarize agent returned an invalid SummarizeAgentDraft."
            ) from exc
        if runs_root is not None:
            save_stage_json(
                runs_root, run_id=request.run_id, stage="summarize",
                filename="draft.json", payload=draft,
            )
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
