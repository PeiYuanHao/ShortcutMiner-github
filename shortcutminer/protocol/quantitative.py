"""Strict messages for one Quantitative Agent task and result."""

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, StringConstraints, field_validator, model_validator, model_serializer

from shortcutminer.protocol.executive import AnalysisTask
from shortcutminer.protocol.model_reliance import TorchvisionTrainingOptions, DatasetShiftSpec, AdaptivePairedScreeningOptions
from shortcutminer.intervention.protocol import BackgroundReplaceTransform, GlobalColorTransform, CornerPatchTransform, ProvidedPairTransform, AutonomousImageTransform
from shortcutminer.protocol.ood_recipe import OODRecipeSelectionPolicy
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
    execution_resources: "QuantitativeExecutionResources | None" = None
    limitations: list[NonEmptyText] = Field(default_factory=list)

    @field_validator("limitations")
    @classmethod
    def validate_unique_limitations(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "limitations")


class QuantitativeExecutionRequirement(StrictProtocolModel):
    """Control-layer decision about whether measurement must be attempted."""

    must_attempt_run_experiment: bool
    reason: QuantitativeExecutionReason
    resolved_experiment_types: list[NonEmptyText] = Field(default_factory=list)
    available_now_experiment_types: list[NonEmptyText] = Field(default_factory=list)
    pending_dependency_experiment_types: list[NonEmptyText] = Field(default_factory=list)
    unsupported_experiment_types: list[NonEmptyText] = Field(default_factory=list)

    @field_validator(
        "resolved_experiment_types",
        "available_now_experiment_types",
        "pending_dependency_experiment_types",
        "unsupported_experiment_types",
    )
    @classmethod
    def validate_unique_resolved_types(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "resolved_experiment_types")

    @model_validator(mode="after")
    def validate_reason(self):
        if self.must_attempt_run_experiment != (self.reason == "task_is_executable"):
            raise ValueError(
                "task_is_executable is the only reason that requires an experiment"
            )
        if self.must_attempt_run_experiment and not self.resolved_experiment_types:
            raise ValueError(
                "executable quantitative work requires resolved experiment types"
            )
        classified = (
            self.available_now_experiment_types
            + self.pending_dependency_experiment_types
            + self.unsupported_experiment_types
        )
        if self.must_attempt_run_experiment and not classified:
            # Backward-compatible parse for persisted pre-plan requirements.
            # New control-layer construction always writes explicit availability.
            self.available_now_experiment_types = list(
                self.resolved_experiment_types
            )
            classified = list(self.available_now_experiment_types)
        if len(classified) != len(set(classified)):
            raise ValueError("experiment availability classes must not overlap")
        if not set(classified).issubset(self.resolved_experiment_types):
            raise ValueError("experiment availability must be resolved first")
        if self.must_attempt_run_experiment and not self.available_now_experiment_types:
            raise ValueError("executable quantitative work requires an available experiment")
        return self


class QuantitativeExecutionResources(StrictProtocolModel):
    """Authoritative resources injected by the control layer, never invented by an LLM."""

    protocol_version: ProtocolVersion = "1.0"
    dataset_name: NonEmptyText
    metadata_file: SafeRelativePath | None = None
    label_column: NonEmptyText
    available_metadata_fields: list[NonEmptyText] = Field(default_factory=list)
    allowed_scientific_metadata_fields: list[NonEmptyText] = Field(default_factory=list)
    sample_id_column: NonEmptyText | None = None
    sample_path_column: NonEmptyText | None = None
    split_column: NonEmptyText | None = None
    selected_split: NonEmptyText | None = None
    available_splits: list[NonEmptyText] = Field(default_factory=list)
    authorized_artifact_references: list[SafeRelativePath] = Field(default_factory=list)
    seed: int = 0
    max_rows: int = Field(default=100_000, ge=4, le=1_000_000)
    max_samples: int = Field(default=10_000, ge=1, le=100_000)
    max_categories: int = Field(default=100, ge=2, le=1_000)
    min_group_size: int = Field(default=2, ge=1, le=100_000)
    max_permutations: int = Field(default=100, ge=0, le=1_000)
    statistical_n_permutations: int = Field(default=0, ge=0, le=1_000)
    confidence_level: float = Field(default=0.95, ge=0.8, lt=1.0)
    bootstrap_iterations: int = Field(default=200, ge=0, le=1_000)
    max_probe_features: int = Field(default=20, ge=1, le=100)
    max_probe_iterations: int = Field(default=500, ge=10, le=5_000)
    max_model_iterations: int = Field(default=300, ge=10, le=5_000)
    training_class_weight: Literal["none", "balanced"] = "none"
    training_holdout_fraction: float = Field(default=0.25, gt=0.0, lt=0.5)
    max_intervention_samples: int = Field(default=1_000, ge=1, le=10_000)
    model_backend: Literal["reference_image_linear", "torchvision_resnet_matrix"] = "reference_image_linear"
    feature_version: Literal["reference_image_features_v1", "torchvision_rgb224_v1"] = "reference_image_features_v1"
    training_options: TorchvisionTrainingOptions | None = None
    adaptive_screening: AdaptivePairedScreeningOptions | None = None
    model_metadata_file: SafeRelativePath | None = None
    train_split_values: list[NonEmptyText] = Field(default_factory=list)
    validation_split_values: list[NonEmptyText] = Field(default_factory=list)
    test_split_values: list[NonEmptyText] = Field(default_factory=list)
    evaluation_split_values: list[NonEmptyText] = Field(default_factory=list)
    max_train_samples: int | None = Field(default=None, ge=4, le=100000)
    association_variable_mode: Literal["auto", "categorical_categorical", "numeric_categorical", "numeric_numeric"] = "auto"
    ood_specs: dict[NonEmptyText, DatasetShiftSpec] = Field(default_factory=dict)
    background_ood_specs: dict[NonEmptyText, BackgroundReplaceTransform] = Field(default_factory=dict)
    color_ood_specs: dict[NonEmptyText, GlobalColorTransform] = Field(default_factory=dict)
    patch_ood_specs: dict[NonEmptyText, CornerPatchTransform] = Field(default_factory=dict)
    ood_recipe_selection: OODRecipeSelectionPolicy | None = None
    artifact_policy: Literal["safe_json_npz"] = "safe_json_npz"
    visual_ood_specs: dict[NonEmptyText, ProvidedPairTransform | AutonomousImageTransform] = Field(default_factory=dict)

    @model_serializer(mode='wrap')
    def preserve_old_visual_contract(self, handler):
        result = handler(self)
        if 'visual_ood_specs' not in self.model_fields_set:
            result.pop('visual_ood_specs', None)
        return result

    @field_validator(
        "available_metadata_fields",
        "allowed_scientific_metadata_fields",
        "available_splits",
        "authorized_artifact_references",
    )
    @classmethod
    def validate_unique_resource_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)

    @model_validator(mode="after")
    def validate_resource_authority(self):
        available = set(self.available_metadata_fields)
        if not set(self.allowed_scientific_metadata_fields).issubset(available):
            raise ValueError("scientific metadata fields must be available")
        controlled = {
            self.label_column,
            self.sample_id_column,
            self.sample_path_column,
            self.split_column,
        } - {None}
        if set(self.allowed_scientific_metadata_fields) & controlled:
            raise ValueError("scientific fields cannot use control/addressing columns")
        if self.metadata_file is not None and self.label_column not in available:
            raise ValueError("label_column must be an available metadata field")
        for field in (
            self.sample_id_column,
            self.sample_path_column,
            self.split_column,
        ):
            if self.metadata_file is not None and field is not None and field not in available:
                raise ValueError("configured addressing columns must be available")
        return self


class QuantitativeAgentDraft(StrictProtocolModel):
    """Provider-owned scientific interpretation without deterministic bookkeeping."""

    status: QuantitativeStatus
    analysis_type: NonEmptyText
    evidence_direction: EvidenceDirection
    evidence_summary: NonEmptyText
    additional_limitations: list[NonEmptyText]
    recommended_followup: list[NonEmptyText]

    @field_validator(
        "additional_limitations", "recommended_followup"
    )
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)

    @model_validator(mode="after")
    def validate_status_evidence(self):
        if self.status == "completed":
            if self.evidence_direction == "not_tested":
                raise ValueError(
                    "completed quantitative draft cannot be marked not_tested"
                )
        else:
            if self.evidence_direction != "not_tested":
                raise ValueError(
                    "blocked or failed quantitative draft must be marked not_tested"
                )
        return self


class QuantitativeMeasurement(StrictProtocolModel):
    """One canonical plan step and one actual backend execution."""

    measurement_id: NonEmptyText
    run_id: SafeRunId
    candidate_id: CandidateId
    task_id: NonEmptyText
    plan_step_id: NonEmptyText
    planning_check: NonEmptyText
    experiment_type: NonEmptyText
    tool_call_id: EvidenceToolCallId
    backend_identity: NonEmptyText
    execution_outcome: Literal["completed", "inconclusive"]
    metrics: dict[NonEmptyText, MetricValue]
    artifact_references: list[SafeRelativePath]
    model_lineage: dict[str, object] = Field(default_factory=dict)
    design_strength: Literal["descriptive", "observational", "predictive", "interventional"]
    uncertainty_metadata: list[dict[str, object]] = Field(default_factory=list)


class QuantitativeResultPayload(StrictProtocolModel):
    task_id: NonEmptyText
    candidate_id: CandidateId
    hypothesis: NonEmptyText
    status: QuantitativeStatus
    analysis_type: NonEmptyText
    metrics: dict[NonEmptyText, MetricValue]
    measurements: list[QuantitativeMeasurement] = Field(default_factory=list)
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

    @model_validator(mode="after")
    def validate_measurement_provenance(self):
        seen_steps: set[str] = set()
        seen_calls: set[str] = set()
        seen_ids: set[str] = set()
        for measurement in self.payload.measurements:
            if (
                measurement.run_id != self.run_id
                or measurement.candidate_id != self.payload.candidate_id
                or measurement.task_id != self.payload.task_id
            ):
                raise ValueError("quantitative measurement correlation differs from result")
            if (
                measurement.plan_step_id in seen_steps
                or measurement.tool_call_id in seen_calls
                or measurement.measurement_id in seen_ids
            ):
                raise ValueError("duplicate quantitative measurement provenance")
            seen_steps.add(measurement.plan_step_id)
            seen_calls.add(measurement.tool_call_id)
            seen_ids.add(measurement.measurement_id)
        if self.payload.status == "completed" and not self.payload.measurements:
            raise ValueError("completed quantitative result requires measurements")
        return self
