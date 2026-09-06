"""Strict messages for one Quantitative Agent task and result."""

from typing import Literal

from pydantic import Field, FiniteFloat, field_validator, model_validator

from shortcutminer.protocol.executive import AnalysisTask
from shortcutminer.protocol.identify import (
    Candidate,
    CandidateId,
    NonEmptyText,
    ProtocolVersion,
    SafeRelativePath,
    SafeRunId,
    StrictProtocolModel,
)


QuantitativeStatus = Literal["completed", "blocked", "failed"]
EvidenceLevel = Literal["Q0", "Q1", "Q2", "Q3", "Q4"]
EvidenceDirection = Literal[
    "supports", "mixed", "contradicts", "inconclusive", "not_tested"
]
MetricValue = str | int | FiniteFloat | bool | None


def _reject_duplicates(values: list[str], field_name: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


class QuantitativeRequest(StrictProtocolModel):
    protocol_version: ProtocolVersion
    run_id: SafeRunId
    message_id: NonEmptyText
    source: Literal["orchestrator"]
    target: Literal["quantitative"]
    message_type: Literal["quantitative_analysis_request"]
    in_reply_to: NonEmptyText
    candidate: Candidate
    task: AnalysisTask
    limitations: list[NonEmptyText] = Field(default_factory=list)

    @field_validator("limitations")
    @classmethod
    def validate_unique_limitations(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "limitations")


class QuantitativeResultPayload(StrictProtocolModel):
    task_id: NonEmptyText
    candidate_id: CandidateId
    hypothesis: NonEmptyText
    status: QuantitativeStatus
    analysis_type: NonEmptyText
    metrics: dict[NonEmptyText, MetricValue]
    evidence_level: EvidenceLevel
    evidence_direction: EvidenceDirection
    evidence_summary: NonEmptyText
    limitations: list[NonEmptyText]
    artifact_references: list[SafeRelativePath]
    recommended_followup: list[NonEmptyText]

    @field_validator(
        "limitations", "artifact_references", "recommended_followup"
    )
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)

    @model_validator(mode="after")
    def validate_status_evidence(self):
        if self.status == "completed":
            if self.evidence_level == "Q0":
                raise ValueError("completed quantitative work cannot use Q0")
            if self.evidence_direction == "not_tested":
                raise ValueError(
                    "completed quantitative work cannot be marked not_tested"
                )
        else:
            if self.evidence_level != "Q0":
                raise ValueError("blocked or failed quantitative work must use Q0")
            if self.evidence_direction != "not_tested":
                raise ValueError(
                    "blocked or failed quantitative work must be marked not_tested"
                )
            if self.metrics:
                raise ValueError(
                    "blocked or failed quantitative work cannot report metrics"
                )
            if not self.limitations:
                raise ValueError("blocked or failed work must explain a limitation")
        return self


class QuantitativeResult(StrictProtocolModel):
    protocol_version: ProtocolVersion
    run_id: SafeRunId
    message_id: NonEmptyText
    source: Literal["quantitative"]
    target: Literal["summarize"]
    message_type: Literal["quantitative_analysis_result"]
    in_reply_to: NonEmptyText
    payload: QuantitativeResultPayload
