"""User input and bounded intake facts, never validation tasks or plans."""

from typing import Literal

from pydantic import Field, model_validator, model_serializer

from shortcutminer.protocol.identify import NonEmptyText, SafeRelativePath, StrictProtocolModel
from shortcutminer.protocol.model_reliance import TorchvisionTrainingOptions, DatasetShiftSpec, AdaptivePairedScreeningOptions
from shortcutminer.intervention.protocol import BackgroundReplaceTransform, GlobalColorTransform, CornerPatchTransform, ProvidedPairTransform, AutonomousImageTransform
from shortcutminer.protocol.ood_recipe import OODRecipeSelectionPolicy


class AuditExecutionOptions(StrictProtocolModel):
    """Explicit caller configuration, not inferred from candidate prose."""
    sample_id_column: NonEmptyText
    sample_path_column: NonEmptyText
    split_column: NonEmptyText
    model_metadata_file: SafeRelativePath
    train_split_values: list[NonEmptyText]
    validation_split_values: list[NonEmptyText]
    test_split_values: list[NonEmptyText]
    evaluation_split_values: list[NonEmptyText]
    model_backend: Literal["reference_image_linear", "torchvision_resnet_matrix"]
    feature_version: Literal["reference_image_features_v1", "torchvision_rgb224_v1"]
    training_options: TorchvisionTrainingOptions | None = None
    adaptive_screening: AdaptivePairedScreeningOptions | None = None
    max_train_samples: int = Field(ge=4, le=100000)
    max_samples: int = Field(ge=1, le=100000)
    max_intervention_samples: int = Field(default=1000,ge=1,le=10000)
    association_variable_mode: Literal["auto", "categorical_categorical", "numeric_categorical", "numeric_numeric"] = "auto"
    ood_specs: dict[NonEmptyText, DatasetShiftSpec] = Field(default_factory=dict)
    background_ood_specs: dict[NonEmptyText, BackgroundReplaceTransform] = Field(default_factory=dict)
    color_ood_specs: dict[NonEmptyText, GlobalColorTransform] = Field(default_factory=dict)
    patch_ood_specs: dict[NonEmptyText, CornerPatchTransform] = Field(default_factory=dict)
    ood_recipe_selection: OODRecipeSelectionPolicy | None = None
    excluded_scientific_fields: list[NonEmptyText] = Field(default_factory=list)
    visual_measurement_bindings: dict[NonEmptyText, NonEmptyText] = Field(default_factory=dict)
    visual_ood_specs: dict[NonEmptyText, ProvidedPairTransform | AutonomousImageTransform] = Field(default_factory=dict)

    @model_serializer(mode='wrap')
    def preserve_old_visual_contract(self, handler):
        result = handler(self)
        for name in ('visual_measurement_bindings', 'visual_ood_specs'):
            if name not in self.model_fields_set:
                result.pop(name, None)
        return result

    @model_validator(mode='after')
    def avoid_preselected_memory_recipe(self):
        if self.ood_recipe_selection and set(self.ood_recipe_selection.fields) & (set(self.color_ood_specs) | set(self.background_ood_specs) | set(self.patch_ood_specs)):
            raise ValueError('Memory-selected fields cannot also have a caller-preselected transformation')
        return self


class AuditIntakeBudget(StrictProtocolModel):
    max_entries: int = Field(default=10_000, ge=1)
    max_metadata_files: int = Field(default=8, ge=1)
    max_header_bytes: int = Field(default=65_536, ge=1)


class AuditLabelInformation(StrictProtocolModel):
    source: Literal["metadata_column", "class_directory", "split_class_directory"]
    column: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_column(self):
        if (self.source == "metadata_column") != (self.column is not None):
            raise ValueError("Only metadata_column labels require a column")
        return self


class AuditLabelSource(StrictProtocolModel):
    status: Literal["resolved", "unresolved"] = "unresolved"
    kind: Literal["metadata_column", "class_directory", "split_class_directory"] | None = None
    column: NonEmptyText | None = None
    classes: list[NonEmptyText] = Field(default_factory=list)
    possible_sources: list[NonEmptyText] = Field(default_factory=list)
    reason: NonEmptyText


class AuditDatasetInfo(StrictProtocolModel):
    root: NonEmptyText
    name: NonEmptyText
    dataset_type: Literal["image", "tabular", "text", "mixed", "unknown"]
    metadata_files: list[SafeRelativePath]
    selected_metadata_file: SafeRelativePath | None = None
    metadata_columns: list[NonEmptyText] = Field(default_factory=list)
    label_source: AuditLabelSource
    entries_scanned: int = Field(ge=0)
    truncated: bool
    limitations: list[NonEmptyText] = Field(default_factory=list)


class AuditRequest(StrictProtocolModel):
    """Dataset-only request; run/model settings belong to the RunRequest builder."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_path: NonEmptyText
    dataset_description: NonEmptyText | None = None
    label_information: AuditLabelInformation | None = None
    metadata_file: SafeRelativePath | None = None
    intake_budget: AuditIntakeBudget = Field(default_factory=AuditIntakeBudget)
    dataset_info: AuditDatasetInfo | None = None
    execution_options: AuditExecutionOptions | None = None
