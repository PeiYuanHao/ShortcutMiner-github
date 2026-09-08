"""Strict parameter contracts for approved quantitative experiments."""

from typing import Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from shortcutminer.protocol.identify import (
    NonEmptyText,
    SafeRelativePath,
    StrictProtocolModel,
)


class EmptyExperimentParameters(StrictProtocolModel):
    """Compatibility shape used by mocks and explicit missing-input results."""


class UncertaintyOptions(StrictProtocolModel):
    confidence_level: float = Field(default=0.95, ge=0.8, lt=1.0)
    bootstrap_iterations: int = Field(default=200, ge=0, le=1_000)


class StatisticalCheckParameters(UncertaintyOptions):
    metadata_file: SafeRelativePath
    cue_field: NonEmptyText
    label_column: NonEmptyText
    variable_mode: Literal[
        "auto",
        "categorical_categorical",
        "numeric_categorical",
        "numeric_numeric",
    ] = "auto"
    max_rows: int = Field(default=100_000, ge=2, le=1_000_000)
    max_categories: int = Field(default=100, ge=2, le=1_000)
    min_group_size: int = Field(default=2, ge=1, le=100_000)
    n_permutations: int = Field(default=0, ge=0, le=1_000)
    seed: int = 0


class SubgroupEvaluationParameters(UncertaintyOptions):
    metadata_file: SafeRelativePath
    group_field: NonEmptyText
    label_column: NonEmptyText
    predictions_file: SafeRelativePath | None = None
    predictions_source: Literal["dataset", "run"] = "dataset"
    sample_id_column: NonEmptyText | None = None
    prediction_sample_id_column: NonEmptyText | None = None
    prediction_column: NonEmptyText = "prediction"
    max_rows: int = Field(default=100_000, ge=2, le=1_000_000)
    max_categories: int = Field(default=100, ge=2, le=1_000)
    min_group_size: int = Field(default=2, ge=1, le=100_000)
    seed: int = 0

    @model_validator(mode="after")
    def validate_prediction_join(self):
        if self.predictions_file is not None and bool(self.sample_id_column) != bool(
            self.prediction_sample_id_column
        ):
            raise ValueError(
                "sample_id_column and prediction_sample_id_column must be supplied together"
            )
        return self


class PredictiveProbeParameters(UncertaintyOptions):
    metadata_file: SafeRelativePath
    feature_fields: list[NonEmptyText] = Field(min_length=1)
    label_column: NonEmptyText
    split_column: NonEmptyText | None = None
    train_split_values: list[NonEmptyText] = Field(default_factory=list)
    test_split_values: list[NonEmptyText] = Field(default_factory=list)
    test_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)
    max_rows: int = Field(default=100_000, ge=4, le=1_000_000)
    max_categories: int = Field(default=100, ge=2, le=1_000)
    max_probe_features: int = Field(default=20, ge=1, le=100)
    max_probe_iterations: int = Field(default=500, ge=10, le=5_000)
    seed: int = 0

    @field_validator("feature_fields")
    @classmethod
    def validate_features(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("feature_fields must not contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_split(self):
        if self.label_column in self.feature_fields:
            raise ValueError("label_column cannot be used as a probe feature")
        if len(self.feature_fields) > self.max_probe_features:
            raise ValueError("feature_fields exceeds max_probe_features")
        supplied_values = bool(self.train_split_values or self.test_split_values)
        if supplied_values and (
            self.split_column is None
            or not self.train_split_values
            or not self.test_split_values
        ):
            raise ValueError(
                "explicit train/test values require split_column and both value lists"
            )
        if set(self.train_split_values) & set(self.test_split_values):
            raise ValueError("train and test split values must be disjoint")
        return self


class MetadataInterventionParameters(UncertaintyOptions):
    metadata_file: SafeRelativePath
    cue_field: NonEmptyText
    label_column: NonEmptyText
    intervention_type: Literal["shuffle", "within_split_shuffle"] = "shuffle"
    split_column: NonEmptyText | None = None
    max_rows: int = Field(default=100_000, ge=2, le=1_000_000)
    max_categories: int = Field(default=100, ge=2, le=1_000)
    min_group_size: int = Field(default=2, ge=1, le=100_000)
    seed: int = 0

    @model_validator(mode="after")
    def validate_intervention(self):
        if self.intervention_type == "within_split_shuffle" and self.split_column is None:
            raise ValueError("within_split_shuffle requires split_column")
        return self


class MetadataAblationParameters(PredictiveProbeParameters):
    ablate_field: NonEmptyText

    @model_validator(mode="after")
    def validate_ablation(self):
        if self.ablate_field not in self.feature_fields:
            raise ValueError("ablate_field must be one of feature_fields")
        if len(self.feature_fields) < 2:
            raise ValueError("metadata ablation requires at least two features")
        return self


class MetricComputationParameters(UncertaintyOptions):
    predictions_file: SafeRelativePath
    predictions_source: Literal["dataset", "run"] = "dataset"
    label_column: NonEmptyText
    prediction_column: NonEmptyText
    metric: Literal["accuracy", "balanced_accuracy"] = "accuracy"
    max_rows: int = Field(default=100_000, ge=1, le=1_000_000)
    seed: int = 0


ExperimentParameters: TypeAlias = (
    EmptyExperimentParameters
    | StatisticalCheckParameters
    | SubgroupEvaluationParameters
    | PredictiveProbeParameters
    | MetadataInterventionParameters
    | MetadataAblationParameters
    | MetricComputationParameters
)


__all__ = [
    "EmptyExperimentParameters",
    "ExperimentParameters",
    "MetadataAblationParameters",
    "MetadataInterventionParameters",
    "MetricComputationParameters",
    "PredictiveProbeParameters",
    "StatisticalCheckParameters",
    "SubgroupEvaluationParameters",
    "UncertaintyOptions",
]
