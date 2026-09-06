"""Strict, non-evaluative contract for bounded dataset discovery."""

from typing import Any, Literal

from pydantic import Field, model_validator

from shortcutminer.protocol.identify import (
    NonEmptyText,
    SafeRelativePath,
    StrictProtocolModel,
)


PrimitiveType = Literal["boolean", "integer", "number", "string", "unknown"]
DatasetModality = Literal["image", "tabular", "text", "mixed", "unknown"]
LabelValidity = Literal["valid", "missing_column", "source_unavailable"]


class DatasetProfileDataset(StrictProtocolModel):
    name: NonEmptyText
    root_reference: NonEmptyText
    metadata_file: SafeRelativePath | None = None
    metadata_present: bool


class DatasetProfileTask(StrictProtocolModel):
    task_type: NonEmptyText
    label_column: NonEmptyText
    intended_signal: NonEmptyText | None = None


class DatasetStructureProfile(StrictProtocolModel):
    files_scanned: int = Field(ge=0)
    directories_scanned: int = Field(ge=0)
    entries_scanned: int = Field(ge=0)
    max_entries: int = Field(ge=1)
    truncated: bool
    file_extension_distribution: dict[str, int]
    top_level_entries: list[str]
    max_depth_observed: int = Field(ge=0)
    modality: DatasetModality

    @model_validator(mode="after")
    def validate_counts(self):
        if self.entries_scanned > self.max_entries:
            raise ValueError("entries_scanned must not exceed max_entries")
        if any(count < 0 for count in self.file_extension_distribution.values()):
            raise ValueError("file extension counts must be non-negative")
        return self


class MetadataColumnProfile(StrictProtocolModel):
    name: NonEmptyText
    primitive_type: PrimitiveType
    source_type: NonEmptyText
    non_null_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    missing_rate: float = Field(ge=0.0, le=1.0)
    distinct_count: int = Field(ge=0)
    example_values: list[Any]


class DatasetMetadataProfile(StrictProtocolModel):
    available: bool
    file: SafeRelativePath | None = None
    row_count: int | None = Field(default=None, ge=0)
    rows_profiled: int = Field(ge=0)
    row_limit: int = Field(ge=1)
    truncated: bool
    column_names: list[str]
    columns: list[MetadataColumnProfile]

    @model_validator(mode="after")
    def validate_availability(self):
        if not self.available and (
            self.rows_profiled
            or self.column_names
            or self.columns
            or self.row_count is not None
        ):
            raise ValueError("unavailable metadata cannot expose profiled content")
        if self.rows_profiled > self.row_limit:
            raise ValueError("rows_profiled must not exceed row_limit")
        return self


class LabelProfile(StrictProtocolModel):
    column: NonEmptyText
    validity: LabelValidity
    distribution: dict[str, int]
    distinct_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_distribution(self):
        if any(count < 0 for count in self.distribution.values()):
            raise ValueError("label distribution counts must be non-negative")
        if self.validity != "valid" and (
            self.distribution or self.distinct_count or self.missing_count
        ):
            raise ValueError("unavailable labels cannot expose statistics")
        return self


class SplitProfile(StrictProtocolModel):
    detected_field: NonEmptyText | None = None
    distribution: dict[str, int]

    @model_validator(mode="after")
    def validate_distribution(self):
        if any(count < 0 for count in self.distribution.values()):
            raise ValueError("split distribution counts must be non-negative")
        if self.detected_field is None and self.distribution:
            raise ValueError("split values require a detected field")
        return self


class SampleReference(StrictProtocolModel):
    reference: SafeRelativePath
    kind: Literal["file"]
    extension: str


class DatasetInspectionBudget(StrictProtocolModel):
    max_dataset_entries: int = Field(ge=1)
    max_metadata_rows: int = Field(ge=1)
    max_samples: int = Field(ge=1)


class DatasetProfileMetadata(StrictProtocolModel):
    deterministic_seed: int
    inspection_budget: DatasetInspectionBudget


class DatasetProfile(StrictProtocolModel):
    """Observed dataset facts only; this schema cannot carry candidates."""

    schema_version: Literal["1.0"]
    dataset: DatasetProfileDataset
    task: DatasetProfileTask
    structure: DatasetStructureProfile
    metadata: DatasetMetadataProfile
    labels: LabelProfile
    splits: SplitProfile
    sample_references: list[SampleReference]
    warnings: list[NonEmptyText]
    limitations: list[NonEmptyText]
    profile_metadata: DatasetProfileMetadata

    @model_validator(mode="after")
    def validate_cross_field_contract(self):
        budget = self.profile_metadata.inspection_budget
        if self.structure.max_entries != budget.max_dataset_entries:
            raise ValueError("structure max_entries must match inspection budget")
        if self.metadata.row_limit != budget.max_metadata_rows:
            raise ValueError("metadata row_limit must match inspection budget")
        if len(self.sample_references) > budget.max_samples:
            raise ValueError("sample references exceed inspection budget")
        if self.dataset.metadata_present != self.metadata.available:
            raise ValueError("metadata presence fields disagree")
        if self.labels.validity == "valid" and not self.metadata.available:
            raise ValueError("valid labels require available metadata")
        return self


__all__ = [
    "DatasetInspectionBudget",
    "DatasetMetadataProfile",
    "DatasetProfile",
    "DatasetProfileDataset",
    "DatasetProfileMetadata",
    "DatasetProfileTask",
    "DatasetStructureProfile",
    "LabelProfile",
    "MetadataColumnProfile",
    "SampleReference",
    "SplitProfile",
]
