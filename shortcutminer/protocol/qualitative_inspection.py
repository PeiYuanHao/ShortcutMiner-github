"""Strict contracts for deterministic local qualitative evidence preparation."""

from typing import Any, Literal

from pydantic import Field, field_validator

from shortcutminer.protocol.executive import TaskId
from shortcutminer.protocol.identify import (
    CandidateId,
    NonEmptyText,
    SafeRelativePath,
    SafeRunId,
    StrictProtocolModel,
)
from shortcutminer.protocol.qualitative import InspectionStatus, SamplingStrategy


MediaType = Literal["image", "text", "tabular", "unknown"]


def _unique(values: list[str], name: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


class SampleEvidence(StrictProtocolModel):
    sample_id: NonEmptyText
    metadata_row_index: int | None = Field(default=None, ge=0)
    group: NonEmptyText
    group_values: dict[str, Any]
    sample_reference: SafeRelativePath | None = None
    media_type: MediaType
    metadata: dict[str, Any]
    selection_reason: SamplingStrategy
    sample_available: bool
    inspection_status: InspectionStatus
    visual_descriptors: dict[str, Any] = Field(default_factory=dict)
    text_descriptors: dict[str, Any] = Field(default_factory=dict)
    artifact_references: list[SafeRelativePath] = Field(default_factory=list)
    warnings: list[NonEmptyText] = Field(default_factory=list)
    limitations: list[NonEmptyText] = Field(default_factory=list)
    byte_size: int | None = Field(default=None, ge=0)
    bytes_consumed: int = Field(default=0, ge=0)

    @field_validator("artifact_references", "warnings", "limitations")
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _unique(values, info.field_name)


class GroupDescriptorSummary(StrictProtocolModel):
    group: NonEmptyText
    sample_count: int = Field(ge=0)
    inspected_count: int = Field(ge=0)
    descriptor_means: dict[NonEmptyText, float]


class DescriptorComparison(StrictProtocolModel):
    group_a: NonEmptyText
    group_b: NonEmptyText
    sample_count_a: int = Field(ge=0)
    sample_count_b: int = Field(ge=0)
    mean_differences_a_minus_b: dict[NonEmptyText, float]


class InspectionCoverage(StrictProtocolModel):
    requested_sample_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    available_count: int = Field(ge=0)
    inspected_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    group_coverage: dict[NonEmptyText, int]


class QualitativeInspectionArtifact(StrictProtocolModel):
    schema_version: Literal["1.0"]
    run_id: SafeRunId
    candidate_id: CandidateId
    task_id: TaskId
    backend: Literal["local_descriptor_v1"]
    request_fingerprint: NonEmptyText
    sampling_strategy: SamplingStrategy
    seed: int
    metadata_file: SafeRelativePath | None = None
    metadata_available: bool
    metadata_rows_read: int = Field(ge=0)
    preparation: dict[str, Any]
    request_limitations: list[NonEmptyText]
    coverage: InspectionCoverage
    selected_samples: list[SampleEvidence]
    group_summaries: list[GroupDescriptorSummary]
    descriptor_comparisons: list[DescriptorComparison]
    descriptor_definitions: dict[str, Any]
    warnings: list[NonEmptyText]
    limitations: list[NonEmptyText]
    artifact_references: list[SafeRelativePath]
    created_at: NonEmptyText

    @field_validator(
        "request_limitations", "warnings", "limitations", "artifact_references"
    )
    @classmethod
    def validate_unique_artifact_lists(cls, values: list[str], info) -> list[str]:
        return _unique(values, info.field_name)


__all__ = [
    "DescriptorComparison",
    "GroupDescriptorSummary",
    "InspectionCoverage",
    "MediaType",
    "QualitativeInspectionArtifact",
    "SampleEvidence",
]
