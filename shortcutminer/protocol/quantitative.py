"""Strict messages for one Quantitative Agent task and result."""

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, StringConstraints, field_validator, model_validator

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
QuantitativeExecutionReason = Literal[
    "task_is_executable",
    "feasibility_not_confirmed",
    "run_experiment_unavailable",
]
EvidenceLevel = Literal["Q0", "Q1", "Q2", "Q3", "Q4"]
EvidenceDirection = Literal[
    "supports", "mixed", "contradicts", "inconclusive", "not_tested"
]
MetricValue = str | int | FiniteFloat | bool | None
EvidenceToolCallId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    ),
]


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


class QuantitativeExecutionRequirement(StrictProtocolModel):
    """Control-layer decision about whether measurement must be attempted."""

    must_attempt_run_experiment: bool
    reason: QuantitativeExecutionReason

    @model_validator(mode="after")
    def validate_reason(self):
        if self.must_attempt_run_experiment != (self.reason == "task_is_executable"):
            raise ValueError(
                "task_is_executable is the only reason that requires an experiment"
            )
        return self


class QuantitativeAgentDraft(StrictProtocolModel):
    """Provider-owned scientific interpretation without deterministic bookkeeping."""

    status: QuantitativeStatus
    analysis_type: NonEmptyText
    evidence_level: EvidenceLevel
    evidence_direction: EvidenceDirection
    evidence_summary: NonEmptyText
    evidence_tool_call_ids: list[EvidenceToolCallId]
    additional_limitations: list[NonEmptyText]
    recommended_followup: list[NonEmptyText]

    @field_validator(
        "evidence_tool_call_ids", "additional_limitations", "recommended_followup"
    )
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)

    @model_validator(mode="after")
    def validate_status_evidence(self):
        if self.status == "completed":
            if self.evidence_level == "Q0":
                raise ValueError("completed quantitative draft cannot use Q0")
            if self.evidence_direction == "not_tested":
                raise ValueError(
                    "completed quantitative draft cannot be marked not_tested"
                )
        else:
            if self.evidence_level != "Q0":
                raise ValueError("blocked or failed quantitative draft must use Q0")
            if self.evidence_direction != "not_tested":
                raise ValueError(
                    "blocked or failed quantitative draft must be marked not_tested"
                )
        return self


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
