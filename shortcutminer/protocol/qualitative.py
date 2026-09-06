from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator

from shortcutminer.protocol.executive import AnalysisTask, TaskId
from shortcutminer.protocol.identify import (
    Candidate,
    CandidateId,
    NonEmptyText,
    ProtocolVersion,
    SafeRelativePath,
    SafeRunId,
    StrictProtocolModel,
)


EvidenceStatus = Literal[
    "supported",
    "partially_supported",
    "inconclusive",
    "contradicted",
    "insufficient_evidence",
]
InterpretationConfidence = Literal["high", "medium", "low"]
SamplingStrategy = Literal["random", "group_balanced"]
InspectionQuality = Literal["sufficient", "limited", "unusable"]
FailureCategory = Literal[
    "missing_samples",
    "missing_metadata",
    "invalid_artifact_reference",
    "unsupported_file_format",
    "insufficient_group_coverage",
    "ambiguous_candidate_definition",
    "inspection_failure",
    "tool_failure",
]
ObservationId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^QO-C(?:00[1-9]|0[1-9][0-9]|[1-9][0-9]{2})-"
        r"(?:00[1-9]|0[1-9][0-9]|[1-9][0-9]{2})$",
    ),
]


def _reject_duplicates(values: list[str], field_name: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


class QualitativePreparationOptions(StrictProtocolModel):
    sampling_strategy: SamplingStrategy = "group_balanced"
    seed: int
    sample_id_column: NonEmptyText | None = None
    sample_path_column: NonEmptyText | None = None
    split_column: NonEmptyText | None = None
    group_columns: list[NonEmptyText] = Field(min_length=1)
    max_evidence_records: int = Field(default=40, ge=1)
    max_metadata_rows: int = Field(default=100_000, ge=1)

    @field_validator("group_columns")
    @classmethod
    def validate_unique_groups(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "group_columns")


class QualitativeRequest(StrictProtocolModel):
    protocol_version: ProtocolVersion
    run_id: SafeRunId
    message_id: NonEmptyText
    source: Literal["orchestrator"]
    target: Literal["qualitative"]
    message_type: Literal["qualitative_analysis_request"]
    in_reply_to: NonEmptyText

    candidate: Candidate
    task: AnalysisTask
    metadata_file: SafeRelativePath | None = None
    preparation: QualitativePreparationOptions
    limitations: list[NonEmptyText] = Field(default_factory=list)

    @field_validator("limitations")
    @classmethod
    def validate_unique_limitations(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "limitations")


class QualitativeSampling(StrictProtocolModel):
    strategy: SamplingStrategy
    seed: int
    groups: dict[NonEmptyText, int]
    sample_count: int = Field(ge=0)
    sample_ids: list[NonEmptyText]

    @field_validator("sample_ids")
    @classmethod
    def validate_unique_samples(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "sample_ids")


class QualitativeObservation(StrictProtocolModel):
    observation_id: ObservationId
    group: NonEmptyText
    sample_refs: list[NonEmptyText] = Field(min_length=1)
    observation: NonEmptyText
    inspection_quality: InspectionQuality

    @field_validator("sample_refs")
    @classmethod
    def validate_unique_samples(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "sample_refs")


class QualitativeInterpretation(StrictProtocolModel):
    interpretation: NonEmptyText
    supporting_observation_ids: list[ObservationId] = Field(min_length=1)
    confidence: InterpretationConfidence

    @field_validator("supporting_observation_ids")
    @classmethod
    def validate_unique_observations(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "supporting_observation_ids")


class QualitativeCounterevidence(StrictProtocolModel):
    sample_refs: list[NonEmptyText] = Field(min_length=1)
    description: NonEmptyText

    @field_validator("sample_refs")
    @classmethod
    def validate_unique_samples(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "sample_refs")


class QualitativeMetadataEvidence(StrictProtocolModel):
    field: NonEmptyText
    observation: NonEmptyText
    sample_refs: list[NonEmptyText] = Field(min_length=1)

    @field_validator("sample_refs")
    @classmethod
    def validate_unique_samples(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "sample_refs")


class QualitativeFailure(StrictProtocolModel):
    category: FailureCategory
    description: NonEmptyText
    sample_refs: list[NonEmptyText] = Field(default_factory=list)
    artifact_references: list[SafeRelativePath] = Field(default_factory=list)

    @field_validator("sample_refs", "artifact_references")
    @classmethod
    def validate_unique_references(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)


class QualitativeResultPayload(StrictProtocolModel):
    task_id: TaskId
    candidate_id: CandidateId
    candidate_name: NonEmptyText
    hypothesis: NonEmptyText
    evidence_status: EvidenceStatus
    evidence_summary: NonEmptyText
    sampling: QualitativeSampling
    observations: list[QualitativeObservation]
    interpretations: list[QualitativeInterpretation]
    counterevidence: list[QualitativeCounterevidence]
    metadata_evidence: list[QualitativeMetadataEvidence]
    failures: list[QualitativeFailure]
    limitations: list[NonEmptyText]
    artifact_references: list[SafeRelativePath]
    recommended_followup: list[NonEmptyText]

    @field_validator(
        "limitations",
        "artifact_references",
        "recommended_followup",
    )
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)


class QualitativeResult(StrictProtocolModel):
    protocol_version: ProtocolVersion
    run_id: SafeRunId
    message_id: NonEmptyText
    source: Literal["qualitative"]
    target: Literal["summarize"]
    message_type: Literal["qualitative_analysis_result"]
    in_reply_to: NonEmptyText
    payload: QualitativeResultPayload
