"""Strict request and final report contracts for the Summarize Agent."""

from typing import Literal

from pydantic import Field, field_validator

from shortcutminer.protocol.batches import (
    QualitativeResultBatch,
    QuantitativeResultBatch,
)
from shortcutminer.protocol.executive import RoutingPlan
from shortcutminer.protocol.evidence import EvidenceLedger
from shortcutminer.protocol.identify import (
    CandidateId,
    CandidateReport,
    NonEmptyText,
    ProtocolVersion,
    SafeRelativePath,
    SafeRunId,
    StrictProtocolModel,
)


SummaryAssessment = Literal[
    "supported",
    "partially_supported",
    "inconclusive",
    "contradicted",
    "insufficient_evidence",
]


def _reject_duplicates(values: list[str], field_name: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


class SummarizeRequest(StrictProtocolModel):
    protocol_version: ProtocolVersion
    run_id: SafeRunId
    message_id: NonEmptyText
    source: Literal["orchestrator"]
    target: Literal["summarize"]
    message_type: Literal["summarize_request"]
    in_reply_to: NonEmptyText
    identify_result: CandidateReport
    executive_result: RoutingPlan
    quantitative_result: QuantitativeResultBatch
    qualitative_result: QualitativeResultBatch
    evidence_ledger: EvidenceLedger | None = None


class SummaryFinding(StrictProtocolModel):
    candidate_id: CandidateId
    assessment: SummaryAssessment
    summary: NonEmptyText
    quantitative_result_ids: list[NonEmptyText]
    qualitative_result_ids: list[NonEmptyText]
    evidence_record_ids: list[NonEmptyText] = Field(default_factory=list)
    limitations: list[NonEmptyText]
    recommendations: list[NonEmptyText]

    @field_validator(
        "quantitative_result_ids",
        "qualitative_result_ids",
        "evidence_record_ids",
        "limitations",
        "recommendations",
    )
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)


class SummaryReportPayload(StrictProtocolModel):
    overview: NonEmptyText
    findings: list[SummaryFinding]
    limitations: list[NonEmptyText]
    artifact_references: list[SafeRelativePath]
    evidence_ledger_reference: SafeRelativePath | None = None

    @field_validator("limitations", "artifact_references")
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)


class SummaryReport(StrictProtocolModel):
    protocol_version: ProtocolVersion
    run_id: SafeRunId
    message_id: NonEmptyText
    source: Literal["summarize"]
    target: Literal["orchestrator"]
    message_type: Literal["summary_report"]
    in_reply_to: NonEmptyText
    payload: SummaryReportPayload
