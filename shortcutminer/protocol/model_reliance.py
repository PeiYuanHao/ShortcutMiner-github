"""Strict contracts for audited task models and model-reliance evidence."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from shortcutminer.protocol.identify import (
    NonEmptyText,
    SafeRelativePath,
    SafeRunId,
    StrictProtocolModel,
)


TaskModelId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    ),
]
TaskModelKind = Literal["reference_image_linear", "prediction_artifact", "external"]


class TaskModelCapabilities(StrictProtocolModel):
    predictions: bool = True
    probabilities: bool = False
    logits: bool = False
    training: bool = False
    checkpoint: bool = False
    image_input: bool = False
    intervention_inference: bool = False


class TaskModelArtifact(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: SafeRunId
    model_id: TaskModelId
    model_kind: TaskModelKind
    backend: NonEmptyText
    backend_version: NonEmptyText
    feature_version: NonEmptyText | None = None
    model_hash: NonEmptyText
    training_fingerprint: NonEmptyText | None = None
    model_reference: SafeRelativePath
    weights_reference: SafeRelativePath | None = None
    capabilities: TaskModelCapabilities
    classes: list[str] = Field(default_factory=list)
    limitations: list[NonEmptyText] = Field(default_factory=list)

    @field_validator("classes", "limitations")
    @classmethod
    def unique_lists(cls, values: list[str], info):
        if len(values) != len(set(values)):
            raise ValueError(f"{info.field_name} must not contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_checkpoint(self):
        if self.capabilities.checkpoint and self.weights_reference is None:
            raise ValueError("checkpoint capability requires weights_reference")
        if self.model_kind == "prediction_artifact" and self.capabilities.intervention_inference:
            raise ValueError("static prediction artifacts cannot support intervention inference")
        return self


class TaskPrediction(StrictProtocolModel):
    sample_id: NonEmptyText
    sample_reference: SafeRelativePath
    label: str
    prediction: str
    probabilities: list[float] | None = None
    logits: list[float] | None = None

    @field_validator("probabilities", "logits")
    @classmethod
    def finite_vectors(cls, values: list[float] | None):
        if values is not None and not values:
            raise ValueError("prediction vectors cannot be empty")
        if values is not None and any(not math.isfinite(value) for value in values):
            raise ValueError("prediction vectors must contain only finite values")
        return values


class TaskModelEvaluationRecord(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    model_id: TaskModelId
    model_hash: NonEmptyText
    sample_count: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    accuracy: float | None = Field(default=None, ge=0, le=1)
    balanced_accuracy: float | None = Field(default=None, ge=0, le=1)
    predictions: list[TaskPrediction] = Field(default_factory=list)
    failures: list[NonEmptyText] = Field(default_factory=list)


__all__ = [
    "TaskModelArtifact",
    "TaskModelCapabilities",
    "TaskModelEvaluationRecord",
    "TaskModelId",
    "TaskModelKind",
    "TaskPrediction",
]
