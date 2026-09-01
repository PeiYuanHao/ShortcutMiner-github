from pathlib import Path

from pydantic import ValidationError

from shortcutminer.agent_client import AgentClient
from shortcutminer.experiment.identify_context import (
    IdentifyDiscoveryContext,
    build_identify_context,
)
from shortcutminer.protocol.identify import (
    CandidateReport,
    IdentifyRequest,
)
from shortcutminer.storage.events import append_event
from shortcutminer.storage.identify_artifacts import (
    save_identify_input,
    save_identify_output,
    save_identify_raw_output,
)
class IdentifyExecutionError(RuntimeError):
    """Raised when Identify execution cannot produce a valid report."""


def build_identify_agent_payload(
    request: IdentifyRequest,
    context: IdentifyDiscoveryContext,
) -> dict:
    """
    Build the structured payload sent to the Identify Agent.
    """

    return {
        "request": request.model_dump(mode="json"),
        "discovery_context": context.model_dump(mode="json"),
    }


def validate_candidate_report(
    report: CandidateReport,
    request: IdentifyRequest,
    context: IdentifyDiscoveryContext,
) -> None:
    """
    Validate run-level invariants that are not enforced
    directly by the CandidateReport schema.
    """

    if context.run_id != request.run_id:
        raise IdentifyExecutionError(
            "IdentifyDiscoveryContext run_id does not match "
            "IdentifyRequest."
        )

    if context.protocol_version != request.protocol_version:
        raise IdentifyExecutionError(
            "IdentifyDiscoveryContext protocol_version does not match "
            "IdentifyRequest."
        )

    if context.dataset_name != request.dataset.name:
        raise IdentifyExecutionError(
            "IdentifyDiscoveryContext dataset_name does not match "
            "IdentifyRequest."
        )

    if context.task_type != request.task.type:
        raise IdentifyExecutionError(
            "IdentifyDiscoveryContext task_type does not match "
            "IdentifyRequest."
        )

    if context.label_column != request.task.label_column:
        raise IdentifyExecutionError(
            "IdentifyDiscoveryContext label_column does not match "
            "IdentifyRequest."
        )

    if context.metadata_rows_selected != len(context.selected_samples):
        raise IdentifyExecutionError(
            "IdentifyDiscoveryContext metadata_rows_selected does not "
            "match selected_samples."
        )

    if context.metadata_rows_selected > request.budget.max_samples:
        raise IdentifyExecutionError(
            "IdentifyDiscoveryContext exceeds max_samples metadata-row "
            "selection budget."
        )

    if context.metadata_rows_selected > request.budget.max_metadata_rows:
        raise IdentifyExecutionError(
            "IdentifyDiscoveryContext exceeds max_metadata_rows "
            "metadata-row selection budget."
        )

    entries_scanned = context.dataset_inspection.get("entries_scanned")
    if (
        not isinstance(entries_scanned, int)
        or isinstance(entries_scanned, bool)
        or entries_scanned < 0
        or entries_scanned > request.budget.max_dataset_entries
    ):
        raise IdentifyExecutionError(
            "IdentifyDiscoveryContext has an invalid or over-budget "
            "entries_scanned value."
        )

    if context.metadata_profile is not None:
        rows_profiled = context.metadata_profile.get("rows_profiled")
        if (
            not isinstance(rows_profiled, int)
            or isinstance(rows_profiled, bool)
            or rows_profiled < 0
            or rows_profiled > request.budget.max_metadata_rows
        ):
            raise IdentifyExecutionError(
                "IdentifyDiscoveryContext has an invalid or over-budget "
                "rows_profiled value."
            )

    if not context.metadata_available and (
        context.metadata_profile is not None or context.selected_samples
    ):
        raise IdentifyExecutionError(
            "IdentifyDiscoveryContext exposes metadata evidence while "
            "metadata_available is false."
        )

    if report.protocol_version != request.protocol_version:
        raise IdentifyExecutionError(
            "CandidateReport protocol_version does not match "
            "IdentifyRequest."
        )

    if report.run_id != request.run_id:
        raise IdentifyExecutionError(
            "CandidateReport run_id does not match IdentifyRequest."
        )

    if report.in_reply_to != request.message_id:
        raise IdentifyExecutionError(
            "CandidateReport in_reply_to does not match "
            "IdentifyRequest message_id."
        )

    if report.message_id == request.message_id:
        raise IdentifyExecutionError(
            "CandidateReport message_id must identify a new message."
        )

    if report.artifact_references:
        raise IdentifyExecutionError(
            "CandidateReport contains artifact references that were not "
            "supplied authoritatively in the current invocation payload."
        )

    if report.dataset_summary.dataset != request.dataset.name:
        raise IdentifyExecutionError(
            "CandidateReport dataset does not match IdentifyRequest."
        )

    if (
        report.dataset_summary.samples_inspected
        > request.budget.max_samples
    ):
        raise IdentifyExecutionError(
            "CandidateReport exceeds max_samples budget."
        )

    if (
        report.dataset_summary.samples_inspected
        != context.samples_inspected
    ):
        raise IdentifyExecutionError(
            "CandidateReport samples_inspected does not match the "
            "deterministic discovery context."
        )

    if (
        report.dataset_summary.metadata_used
        and not context.metadata_available
    ):
        raise IdentifyExecutionError(
            "CandidateReport claims metadata use when metadata was "
            "not available."
        )

    if len(report.candidates) > request.budget.max_candidates:
        raise IdentifyExecutionError(
            "CandidateReport exceeds max_candidates budget."
        )

    candidate_ids = [
        candidate.candidate_id
        for candidate in report.candidates
    ]

    if len(candidate_ids) != len(set(candidate_ids)):
        raise IdentifyExecutionError(
            "CandidateReport contains duplicate candidate IDs."
        )

    expected_candidate_ids = [
        f"C{index:03d}"
        for index in range(1, len(candidate_ids) + 1)
    ]

    if candidate_ids != expected_candidate_ids:
        raise IdentifyExecutionError(
            "CandidateReport candidate IDs must be sequential from C001."
        )

    for candidate in report.candidates:
        if len(candidate.evidence_sources) != len(
            set(candidate.evidence_sources)
        ):
            raise IdentifyExecutionError(
                "CandidateReport contains duplicate evidence sources."
            )

        evidence_sources = set(candidate.evidence_sources)

        if (
            "metadata" in evidence_sources
            and not report.dataset_summary.metadata_used
        ):
            raise IdentifyExecutionError(
                "CandidateReport uses metadata evidence while "
                "metadata_used is false."
            )

        if (
            "metadata" in evidence_sources
            and not context.metadata_available
        ):
            raise IdentifyExecutionError(
                "CandidateReport uses unavailable metadata evidence."
            )

        if (
            "sample_inspection" in evidence_sources
            and context.samples_inspected == 0
        ):
            raise IdentifyExecutionError(
                "CandidateReport claims sample inspection evidence, but "
                "no sample content was inspected."
            )

        if "image_statistics" in evidence_sources:
            raise IdentifyExecutionError(
                "CandidateReport claims image statistics that are not "
                "present in the current discovery context."
            )

    missing_limitations = [
        limitation
        for limitation in context.limitations
        if limitation not in report.limitations
    ]

    if missing_limitations:
        raise IdentifyExecutionError(
            "CandidateReport omitted deterministic context limitations: "
            + "; ".join(missing_limitations)
        )


def execute_identify(
    request: IdentifyRequest,
    seed: int,
    client: AgentClient,
    runs_root: str | Path | None = None,
) -> CandidateReport:
    """
    Execute one Identify stage.

    When run storage is enabled, the input, output, and
    stage-level execution events are persisted.
    """

    try:
        if runs_root is not None:
            append_event(
                runs_root=runs_root,
                run_id=request.run_id,
                event="IDENTIFY_STARTED",
                stage="identify",
            )

        context = build_identify_context(
            request=request,
            seed=seed,
        )

        payload = build_identify_agent_payload(
            request=request,
            context=context,
        )

        if runs_root is not None:
            save_identify_input(
                runs_root=runs_root,
                run_id=request.run_id,
                payload=payload,
            )

        try:
            raw_output = client.invoke_agent(
                agent_name="identify",
                payload=payload,
            )

        except Exception as exc:
            raise IdentifyExecutionError(
                f"Identify agent invocation failed: {exc}"
            ) from exc

        if not isinstance(raw_output, dict):
            raise IdentifyExecutionError(
                "Identify agent returned a non-object response."
            )

        if runs_root is not None:
            save_identify_raw_output(
                runs_root=runs_root,
                run_id=request.run_id,
                raw_output=raw_output,
            )

        try:
            report = CandidateReport.model_validate(
                raw_output
            )

        except ValidationError as exc:
            raise IdentifyExecutionError(
                "Identify agent returned an invalid "
                "CandidateReport."
            ) from exc

        validate_candidate_report(
            report=report,
            request=request,
            context=context,
        )

        if runs_root is not None:
            save_identify_output(
                runs_root=runs_root,
                report=report,
            )

            append_event(
                runs_root=runs_root,
                run_id=request.run_id,
                event="IDENTIFY_COMPLETED",
                stage="identify",
                details={
                    "candidate_count": len(
                        report.candidates
                    ),
                    "samples_inspected": (
                        report.dataset_summary.samples_inspected
                    ),
                },
            )

        return report

    except Exception as exc:
        if runs_root is not None:
            try:
                append_event(
                    runs_root=runs_root,
                    run_id=request.run_id,
                    event="AGENT_FAILED",
                    stage="identify",
                    details={
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )

            except Exception:
                # Never hide the original Identify failure
                # because failure logging also failed.
                pass

        if isinstance(exc, IdentifyExecutionError):
            raise

        raise IdentifyExecutionError(
            f"Identify execution failed: {exc}"
        ) from exc
