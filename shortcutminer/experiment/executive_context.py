from collections.abc import Iterable

from shortcutminer.experiment.identify_context import IdentifyDiscoveryContext
from shortcutminer.protocol.executive import (
    ExecutiveAuditOptions,
    ExecutiveCapabilities,
    ExecutiveRequest,
    ExecutiveRoutingContext,
)
from shortcutminer.protocol.identify import CandidateReport, IdentifyRequest


class ExecutiveContextError(RuntimeError):
    """Raised when authoritative Identify state cannot form Executive input."""


def _unique_strings(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _available_metadata_fields(
    context: IdentifyDiscoveryContext,
) -> set[str]:
    fields: set[str] = set()

    if context.metadata_profile is not None:
        columns = context.metadata_profile.get("columns")
        if isinstance(columns, list):
            for column in columns:
                if isinstance(column, dict) and isinstance(
                    column.get("name"), str
                ):
                    fields.add(column["name"])

    for selection in context.selected_samples:
        if not isinstance(selection, dict):
            continue
        record = selection.get("record")
        if isinstance(record, dict):
            fields.update(str(name) for name in record)

    return fields


def build_executive_request(
    *,
    identify_request: IdentifyRequest,
    identify_context: IdentifyDiscoveryContext,
    candidate_report: CandidateReport,
    message_id: str,
    audit_options: ExecutiveAuditOptions,
    capabilities: ExecutiveCapabilities,
    artifact_references: list[str] | None = None,
    limitations: list[str] | None = None,
) -> ExecutiveRequest:
    """Build bounded, traceable Executive input from one Identify run."""

    if identify_context.protocol_version != identify_request.protocol_version:
        raise ExecutiveContextError(
            "Identify context protocol_version does not match request."
        )
    if candidate_report.protocol_version != identify_request.protocol_version:
        raise ExecutiveContextError(
            "CandidateReport protocol_version does not match IdentifyRequest."
        )
    if identify_context.run_id != identify_request.run_id:
        raise ExecutiveContextError("Identify context run_id does not match request.")
    if candidate_report.run_id != identify_request.run_id:
        raise ExecutiveContextError("CandidateReport run_id does not match request.")
    if candidate_report.in_reply_to != identify_request.message_id:
        raise ExecutiveContextError(
            "CandidateReport in_reply_to does not match IdentifyRequest."
        )
    if identify_context.dataset_name != identify_request.dataset.name:
        raise ExecutiveContextError(
            "Identify context dataset does not match IdentifyRequest."
        )
    if candidate_report.dataset_summary.dataset != identify_request.dataset.name:
        raise ExecutiveContextError(
            "CandidateReport dataset does not match IdentifyRequest."
        )
    if identify_context.task_type != identify_request.task.type:
        raise ExecutiveContextError(
            "Identify context task_type does not match IdentifyRequest."
        )
    if identify_context.label_column != identify_request.task.label_column:
        raise ExecutiveContextError(
            "Identify context label_column does not match IdentifyRequest."
        )
    if (
        candidate_report.dataset_summary.samples_inspected
        != identify_context.samples_inspected
    ):
        raise ExecutiveContextError(
            "CandidateReport samples_inspected does not match Identify context."
        )
    if (
        candidate_report.dataset_summary.metadata_used
        and not identify_context.metadata_available
    ):
        raise ExecutiveContextError(
            "CandidateReport claims metadata use when Identify metadata was "
            "unavailable."
        )

    missing_identify_limitations = [
        limitation
        for limitation in identify_context.limitations
        if limitation not in candidate_report.limitations
    ]
    if missing_identify_limitations:
        raise ExecutiveContextError(
            "CandidateReport omitted Identify context limitations: "
            + "; ".join(missing_identify_limitations)
        )

    profiled_fields = _available_metadata_fields(identify_context)
    claimed_fields = set(capabilities.available_metadata_fields)
    if claimed_fields and not identify_context.metadata_available:
        raise ExecutiveContextError(
            "Executive capabilities claim metadata fields when metadata was "
            "not available to Identify."
        )
    if not claimed_fields.issubset(profiled_fields):
        raise ExecutiveContextError(
            "Executive capabilities include metadata fields absent from the "
            "bounded Identify metadata profile."
        )

    authoritative_artifacts = _unique_strings(
        [
            *candidate_report.artifact_references,
            *(artifact_references or []),
        ]
    )
    combined_limitations = _unique_strings(
        [
            *identify_context.limitations,
            *candidate_report.limitations,
            *(limitations or []),
        ]
    )

    routing_context = ExecutiveRoutingContext(
        protocol_version=identify_request.protocol_version,
        run_id=identify_request.run_id,
        dataset=identify_request.dataset,
        task=identify_request.task,
        audit_options=audit_options,
        capabilities=capabilities,
        artifact_references=authoritative_artifacts,
        limitations=combined_limitations,
    )

    return ExecutiveRequest(
        protocol_version=identify_request.protocol_version,
        run_id=identify_request.run_id,
        message_id=message_id,
        source="orchestrator",
        target="executive",
        message_type="executive_request",
        in_reply_to=candidate_report.message_id,
        candidate_report=candidate_report,
        routing_context=routing_context,
    )
