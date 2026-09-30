from typing import Annotated, Literal, TypeVar

from pydantic import Field, StringConstraints, field_validator, model_validator

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
InspectionStatus = Literal["completed", "limited", "failed", "not_applicable"]
ImageDescriptorName = Literal[
    "dimensions",
    "intensity",
    "rgb_stats",
    "entropy",
    "edge_density",
    "spatial_stats",
]
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
ComparisonId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^QC-C(?:00[1-9]|0[1-9][0-9]|[1-9][0-9]{2})-"
        r"(?:00[1-9]|0[1-9][0-9]|[1-9][0-9]{2})$",
    ),
]
ObservationIndex = Annotated[int, Field(ge=0)]
ComparisonIndex = Annotated[int, Field(ge=0)]
_ValueT = TypeVar("_ValueT")


def _reject_duplicates(values: list[_ValueT], field_name: str) -> list[_ValueT]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


class QualitativePreparationOptions(StrictProtocolModel):
    sampling_strategy: SamplingStrategy = "random"
    seed: int
    sample_id_column: NonEmptyText | None = None
    sample_path_column: NonEmptyText | None = None
    split_column: NonEmptyText | None = None
    group_columns: list[NonEmptyText] = Field(default_factory=list)
    requested_descriptors: list[ImageDescriptorName] = Field(
        default_factory=lambda: [
            "dimensions",
            "intensity",
            "rgb_stats",
            "entropy",
            "edge_density",
            "spatial_stats",
        ]
    )
    max_evidence_records: int = Field(default=40, ge=1, le=100)
    max_metadata_rows: int = Field(default=100_000, ge=1, le=1_000_000)
    max_groups: int = Field(default=100, ge=1, le=1_000)
    max_comparison_groups: int = Field(default=20, ge=2, le=50)
    max_sample_candidates: int = Field(default=10_000, ge=1, le=100_000)
    max_file_bytes: int = Field(default=25_000_000, ge=1, le=100_000_000)
    max_pixels: int = Field(default=20_000_000, ge=1, le=100_000_000)
    max_frames: int = Field(default=1, ge=1, le=100)
    max_text_bytes: int = Field(default=16_384, ge=1, le=1_000_000)
    max_total_inspection_bytes: int = Field(
        default=100_000_000,
        ge=1,
        le=500_000_000,
    )
    create_contact_sheet: bool = True
    max_contact_sheet_samples: int = Field(default=24, ge=1, le=40)
    thumbnail_size: int = Field(default=160, ge=32, le=512)

    @field_validator("group_columns")
    @classmethod
    def validate_unique_groups(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "group_columns")

    @field_validator("requested_descriptors")
    @classmethod
    def validate_unique_descriptors(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "requested_descriptors")

    @model_validator(mode="after")
    def validate_group_sampling(self):
        if self.sampling_strategy == "group_balanced" and not self.group_columns:
            raise ValueError(
                "group_balanced sampling requires at least one explicit group column"
            )
        return self


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


class QualitativeComparison(StrictProtocolModel):
    comparison_id: ComparisonId
    groups: list[NonEmptyText] = Field(min_length=2)
    sample_refs: list[NonEmptyText] = Field(min_length=1)
    comparison: NonEmptyText
    inspection_quality: InspectionQuality

    @field_validator("groups", "sample_refs")
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)


class QualitativeInterpretation(StrictProtocolModel):
    interpretation: NonEmptyText
    supporting_observation_ids: list[ObservationId]
    supporting_comparison_ids: list[ComparisonId]
    confidence: InterpretationConfidence

    @field_validator("supporting_observation_ids", "supporting_comparison_ids")
    @classmethod
    def validate_unique_evidence(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)

    @model_validator(mode="after")
    def validate_support(self):
        if not self.supporting_observation_ids and not self.supporting_comparison_ids:
            raise ValueError("interpretation requires supporting evidence")
        return self


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


class QualitativeDraftObservation(StrictProtocolModel):
    sample_ref: NonEmptyText
    observation: NonEmptyText
    inspection_quality: InspectionQuality


class QualitativeDraftComparison(StrictProtocolModel):
    sample_refs: list[NonEmptyText] = Field(min_length=2)
    comparison: NonEmptyText
    inspection_quality: InspectionQuality

    @field_validator("sample_refs")
    @classmethod
    def validate_unique_samples(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "sample_refs")


class QualitativeDraftInterpretation(StrictProtocolModel):
    interpretation: NonEmptyText
    supporting_observation_indexes: list[ObservationIndex]
    supporting_comparison_indexes: list[ComparisonIndex]
    confidence: InterpretationConfidence

    @field_validator(
        "supporting_observation_indexes",
        "supporting_comparison_indexes",
    )
    @classmethod
    def validate_unique_evidence(cls, values: list[int], info) -> list[int]:
        return _reject_duplicates(values, info.field_name)

    @model_validator(mode="after")
    def validate_support(self):
        if (
            not self.supporting_observation_indexes
            and not self.supporting_comparison_indexes
        ):
            raise ValueError("interpretation requires supporting evidence")
        return self


class QualitativeAgentDraft(StrictProtocolModel):
    """Provider-facing scientific analysis without deterministic bookkeeping."""

    sample_ids: list[NonEmptyText]
    evidence_status: EvidenceStatus
    evidence_summary: NonEmptyText
    observations: list[QualitativeDraftObservation]
    comparisons: list[QualitativeDraftComparison]
    interpretations: list[QualitativeDraftInterpretation]
    counterevidence: list[QualitativeCounterevidence]
    metadata_evidence: list[QualitativeMetadataEvidence]
    failures: list[QualitativeFailure]
    additional_limitations: list[NonEmptyText]
    artifact_references: list[SafeRelativePath]
    recommended_followup: list[NonEmptyText]

    @field_validator(
        "sample_ids",
        "additional_limitations",
        "artifact_references",
        "recommended_followup",
    )
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)

    @model_validator(mode="after")
    def validate_internal_references(self):
        for interpretation in self.interpretations:
            if any(index >= len(self.observations) for index in interpretation.supporting_observation_indexes):
                raise ValueError("interpretation references an unknown observation index")
            if any(index >= len(self.comparisons) for index in interpretation.supporting_comparison_indexes):
                raise ValueError("interpretation references an unknown comparison index")
        if self.evidence_status in {"supported", "partially_supported", "contradicted"} and not (
            self.observations or self.comparisons or self.metadata_evidence or self.counterevidence
        ):
            raise ValueError("substantive evidence status requires direct evidence")
        return self


class QualitativeResultPayload(StrictProtocolModel):
    task_id: TaskId
    candidate_id: CandidateId
    candidate_name: NonEmptyText
    hypothesis: NonEmptyText
    evidence_status: EvidenceStatus
    evidence_summary: NonEmptyText
    sampling: QualitativeSampling
    observations: list[QualitativeObservation]
    comparisons: list[QualitativeComparison]
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
