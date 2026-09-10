"""Strict schemas for OOD construction and paired image interventions."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from shortcutminer.protocol.identify import (
    NonEmptyText,
    SafeRelativePath,
    StrictProtocolModel,
)
from shortcutminer.protocol.model_reliance import TaskModelId


LabelPreservationStatus = Literal[
    "guaranteed_by_construction",
    "supported_by_masked_intervention",
    "user_declared",
    "unverified",
    "failed",
]
ComparisonMode = Literal["paired", "independent"]
SpatialRegion = Literal[
    "top_left", "top_right", "bottom_left", "bottom_right", "center", "border", "custom"
]


class NormalizedBox(StrictProtocolModel):
    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError("normalized box must have positive area")
        return self


class GlobalColorTransform(StrictProtocolModel):
    type: Literal["global_color"]
    mode: Literal[
        "channel_scale", "swap_red_blue", "grayscale", "desaturate", "color_replace"
    ]
    channel_scale: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0], min_length=3, max_length=3)
    desaturation_factor: float = Field(default=0.5, ge=0, le=1)
    source_color: list[int] = Field(default_factory=lambda: [0, 0, 0], min_length=3, max_length=3)
    target_color: list[int] = Field(default_factory=lambda: [127, 127, 127], min_length=3, max_length=3)
    color_tolerance: int = Field(default=16, ge=0, le=255)

    @field_validator("channel_scale")
    @classmethod
    def bounded_scales(cls, values):
        if any(value < 0 or value > 4 for value in values):
            raise ValueError("channel scales must be within [0, 4]")
        return values

    @field_validator("source_color", "target_color")
    @classmethod
    def bounded_colors(cls, values):
        if any(value < 0 or value > 255 for value in values):
            raise ValueError("RGB values must be within [0, 255]")
        return values


class FactorTransform(StrictProtocolModel):
    type: Literal["brightness", "contrast"]
    factor: float = Field(ge=0.1, le=3.0)


class BlurTransform(StrictProtocolModel):
    type: Literal["blur"]
    radius: float = Field(ge=0, le=20)


class NoiseTransform(StrictProtocolModel):
    type: Literal["noise"]
    std: float = Field(gt=0, le=100)
    seed: int = 0


class CompressionTransform(StrictProtocolModel):
    type: Literal["compression"]
    quality: int = Field(ge=10, le=95)


class ResolutionTransform(StrictProtocolModel):
    type: Literal["resolution"]
    width: int = Field(ge=2, le=4096)
    height: int = Field(ge=2, le=4096)
    resampling: Literal["nearest", "bilinear", "bicubic", "lanczos"] = "bilinear"


class SpatialTransform(StrictProtocolModel):
    type: Literal["spatial_mask", "spatial_replace"]
    region: SpatialRegion
    custom_box: NormalizedBox | None = None
    fill: Literal["constant", "mean", "noise"] = "constant"
    fill_color: list[int] = Field(default_factory=lambda: [127, 127, 127], min_length=3, max_length=3)
    noise_std: float = Field(default=20, gt=0, le=100)
    seed: int = 0

    @model_validator(mode="after")
    def custom_region_requires_box(self):
        if (self.region == "custom") != (self.custom_box is not None):
            raise ValueError("custom region and custom_box must be supplied together")
        return self


    @field_validator("fill_color")
    @classmethod
    def bounded_fill_color(cls, values):
        if any(value < 0 or value > 255 for value in values):
            raise ValueError("RGB values must be within [0, 255]")
        return values


class SpatialRelocateTransform(StrictProtocolModel):
    type: Literal["spatial_relocate"]
    source_box: NormalizedBox
    target_box: NormalizedBox
    source_fill: Literal["constant", "mean"] = "mean"
    fill_color: list[int] = Field(default_factory=lambda: [127, 127, 127], min_length=3, max_length=3)

    @field_validator("fill_color")
    @classmethod
    def bounded_fill_color(cls, values):
        if any(value < 0 or value > 255 for value in values):
            raise ValueError("RGB values must be within [0, 255]")
        return values


class BackgroundReplaceTransform(StrictProtocolModel):
    type: Literal["background_replace"]
    mask_column: NonEmptyText
    replacement_mode: Literal["constant", "blurred", "sample"] = "constant"
    fill_color: list[int] = Field(default_factory=lambda: [127, 127, 127], min_length=3, max_length=3)
    blur_radius: float = Field(default=8.0, ge=0, le=30)
    replacement_sample_reference: SafeRelativePath | None = None

    @model_validator(mode="after")
    def sample_mode_requires_reference(self):
        if (self.replacement_mode == "sample") != (
            self.replacement_sample_reference is not None
        ):
            raise ValueError("sample replacement requires exactly one approved reference")
        return self

    @field_validator("fill_color")
    @classmethod
    def bounded_fill_color(cls, values):
        if any(value < 0 or value > 255 for value in values):
            raise ValueError("RGB values must be within [0, 255]")
        return values


ImageInterventionSpec: TypeAlias = Annotated[
    GlobalColorTransform
    | FactorTransform
    | BlurTransform
    | NoiseTransform
    | CompressionTransform
    | ResolutionTransform
    | SpatialTransform
    | SpatialRelocateTransform
    | BackgroundReplaceTransform,
    Field(discriminator="type"),
]


class InterventionQuality(StrictProtocolModel):
    decode_success: bool
    same_dimensions: bool
    mask_coverage: float | None = Field(default=None, ge=0, le=1)
    pixel_change_fraction: float | None = Field(default=None, ge=0, le=1)
    foreground_change_fraction: float | None = Field(default=None, ge=0, le=1)
    background_change_fraction: float | None = Field(default=None, ge=0, le=1)
    output_width: int | None = Field(default=None, ge=1)
    output_height: int | None = Field(default=None, ge=1)
    output_bytes: int | None = Field(default=None, ge=0)
    output_format: Literal["PNG", "JPEG"] | None = None


class CounterfactualPair(StrictProtocolModel):
    pair_id: NonEmptyText
    original_sample_id: NonEmptyText
    original_reference: SafeRelativePath
    counterfactual_reference: SafeRelativePath
    label: str
    transformation: ImageInterventionSpec
    target_cue: NonEmptyText
    intended_invariants: list[NonEmptyText]
    seed: int
    model_id: TaskModelId
    model_hash: NonEmptyText
    mask_reference: SafeRelativePath | None = None
    preservation_status: LabelPreservationStatus
    quality: InterventionQuality
    warnings: list[NonEmptyText] = Field(default_factory=list)
    limitations: list[NonEmptyText] = Field(default_factory=list)


class InterventionFailure(StrictProtocolModel):
    sample_id: NonEmptyText
    sample_reference: SafeRelativePath
    error_type: NonEmptyText
    message: NonEmptyText


class InterventionManifest(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: NonEmptyText
    candidate_id: NonEmptyText
    task_id: NonEmptyText
    intervention_id: NonEmptyText
    ood_id: NonEmptyText | None = None
    fingerprint: NonEmptyText
    target_cue: NonEmptyText
    intended_invariants: list[NonEmptyText]
    comparison_mode: ComparisonMode
    transformation: ImageInterventionSpec
    seed: int
    segmentation_provider: NonEmptyText | None = None
    segmentation_version: NonEmptyText | None = None
    model_id: TaskModelId
    model_hash: NonEmptyText
    selected_count: int = Field(ge=0)
    usable_pair_count: int = Field(ge=0)
    failed_pair_count: int = Field(ge=0)
    pair_coverage: float = Field(ge=0, le=1)
    pairs: list[CounterfactualPair]
    failures: list[InterventionFailure] = Field(default_factory=list)
    warnings: list[NonEmptyText] = Field(default_factory=list)
    limitations: list[NonEmptyText] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_pairs(self):
        ids = [pair.pair_id for pair in self.pairs]
        if len(ids) != len(set(ids)):
            raise ValueError("counterfactual pair IDs must be unique")
        if self.usable_pair_count != len(self.pairs) or self.failed_pair_count != len(self.failures):
            raise ValueError("intervention coverage counts must match pairs and failures")
        if self.selected_count != self.usable_pair_count + self.failed_pair_count:
            raise ValueError("selected intervention count must equal usable plus failed")
        return self


class SegmentationRequest(StrictProtocolModel):
    sample_id: NonEmptyText
    sample_reference: SafeRelativePath
    mask_reference: SafeRelativePath | None = None


class SegmentationResult(StrictProtocolModel):
    sample_id: NonEmptyText
    sample_reference: SafeRelativePath
    mask_reference: SafeRelativePath | None = None
    mask_coverage: float | None = Field(default=None, ge=0, le=1)
    provider: NonEmptyText
    version: NonEmptyText
    status: Literal["completed", "unavailable", "failed"]
    limitations: list[NonEmptyText] = Field(default_factory=list)


__all__ = [
    "BackgroundReplaceTransform",
    "ComparisonMode",
    "CounterfactualPair",
    "ImageInterventionSpec",
    "InterventionFailure",
    "InterventionManifest",
    "InterventionQuality",
    "LabelPreservationStatus",
    "NormalizedBox",
    "SegmentationRequest",
    "SegmentationResult",
]
