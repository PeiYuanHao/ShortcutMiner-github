"""Strict contracts for audited task models and model-reliance evidence."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator, model_serializer

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
TaskModelKind = Literal["reference_image_linear", "prediction_artifact", "external", "torchvision_resnet_matrix"]


class AdaptivePairedScreeningOptions(StrictProtocolModel):
    """Caller-frozen budget decisions, never a shortcut verdict."""
    minimum_effect: float = Field(gt=0, lt=1)
    family_alpha: float = Field(gt=0, lt=1)
    candidate_budget: int = Field(ge=1, le=100)
    pair_checkpoints: list[int]
    probe_fraction: float = Field(gt=0, le=0.75)
    confidence_method: Literal["legacy-exact-v1", "paired-betting-cs-v1"] = "legacy-exact-v1"
    sensitivity_effects: list[float]

    @model_serializer(mode='wrap')
    def preserve_legacy_serialization(self, handler):
        result = handler(self)
        if 'confidence_method' not in self.model_fields_set:
            result.pop('confidence_method', None)
        return result

    @model_validator(mode='after')
    def valid_schedule(self):
        if self.confidence_method == 'legacy-exact-v1' and self.probe_fraction > .5:
            raise ValueError('legacy probe fraction must not exceed 0.5')
        if not self.pair_checkpoints or self.pair_checkpoints != sorted(set(self.pair_checkpoints)) or min(self.pair_checkpoints) < 2:
            raise ValueError('pair checkpoints must be increasing positive counts')
        if not self.sensitivity_effects or any(not 0 < x < 1 for x in self.sensitivity_effects):
            raise ValueError('sensitivity effects must lie in (0,1)')
        return self


class TorchvisionTrainingOptions(StrictProtocolModel):
    """Caller-frozen training template; providers cannot supply these fields."""
    architectures: list[Literal["resnet18", "resnet50"]] = Field(default_factory=lambda: ["resnet18", "resnet50"])
    seeds: list[int] = Field(default_factory=lambda: [42, 43, 44])
    epochs: int = Field(default=100, ge=1, le=1000)
    early_stopping_patience: int = Field(default=10, ge=1)
    minimum_epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=0.001, gt=0)
    momentum: float = Field(default=0.9, ge=0, lt=1)
    weight_decay: float = Field(default=0.0001, ge=0)
    workers: int = Field(default=4, ge=0, le=32)
    torch_threads: int = Field(default=24, ge=1, le=128)
    amp: bool = True
    checkpoint_interval: int = Field(default=5, ge=1)
    augmentation: Literal['natural_image', 'digit_preserving', 'spatial_preserving'] = 'natural_image'

    @model_validator(mode="after")
    def frozen_profiles(self):
        if not self.architectures or len(set(self.architectures)) != len(self.architectures):
            raise ValueError("architectures must be nonempty and unique")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs exceeds epochs")
        return self


class DatasetShiftSpec(StrictProtocolModel):
    """Different held-out samples, explicitly independent rather than paired."""
    id_split_values: list[NonEmptyText]
    ood_split_values: list[NonEmptyText]
    pilot_diagnostics_when_deferred: bool = False

    @model_validator(mode="after")
    def isolated(self):
        if not self.id_split_values or not self.ood_split_values:
            raise ValueError("ID and OOD splits must be explicit")
        if set(self.id_split_values) & set(self.ood_split_values):
            raise ValueError("ID and OOD splits overlap")
        return self


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
