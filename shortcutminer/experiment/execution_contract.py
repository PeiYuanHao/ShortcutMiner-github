"""Deterministic ownership boundary for provider-requested experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from shortcutminer.experiment.tool_executor import ExperimentType
from shortcutminer.intervention.protocol import (
    ComparisonMode,
    ImageInterventionSpec,
    LabelPreservationStatus,
)
from shortcutminer.protocol.identify import (
    NonEmptyText,
    SafeRelativePath,
    StrictProtocolModel,
)
from shortcutminer.protocol.quantitative import (
    QuantitativeExecutionRequirement,
    QuantitativeExecutionResources,
)
from shortcutminer.tooling.builtins import ExperimentToolInput
from shortcutminer.tooling.protocol import ToolResult


CONTROL_OWNED = "CONTROL_OWNED"
MODEL_OWNED = "MODEL_OWNED"
AUTHORIZED_RESOURCE_SELECTION = "AUTHORIZED_RESOURCE_SELECTION"


EXPERIMENT_PARAMETER_OWNERSHIP: dict[str, dict[str, tuple[str, ...]]] = {
    "statistical_check": {
        CONTROL_OWNED: (
            "metadata_file", "label_column", "max_rows", "max_categories",
            "min_group_size", "seed", "confidence_level", "bootstrap_iterations",
            "n_permutations",
        ),
        MODEL_OWNED: ("cue_field", "variable_mode"),
        AUTHORIZED_RESOURCE_SELECTION: (),
    },
    "subgroup_evaluation": {
        CONTROL_OWNED: (
            "metadata_file", "label_column", "sample_id_column", "max_rows",
            "max_categories", "min_group_size", "seed", "confidence_level",
            "bootstrap_iterations",
        ),
        MODEL_OWNED: (
            "group_field", "prediction_sample_id_column", "prediction_column",
        ),
        AUTHORIZED_RESOURCE_SELECTION: ("predictions_file", "predictions_source"),
    },
    "model_evaluation": {
        CONTROL_OWNED: (
            "metadata_file", "label_column", "split_column", "max_rows",
            "max_categories", "max_probe_features", "max_probe_iterations", "seed",
            "confidence_level", "bootstrap_iterations",
        ),
        MODEL_OWNED: (
            "feature_fields", "train_split_values", "test_split_values", "test_fraction",
        ),
        AUTHORIZED_RESOURCE_SELECTION: (),
    },
    "intervention": {
        CONTROL_OWNED: (
            "metadata_file", "label_column", "split_column", "max_rows",
            "max_categories", "min_group_size", "seed", "confidence_level",
            "bootstrap_iterations",
        ),
        MODEL_OWNED: ("cue_field", "intervention_type"),
        AUTHORIZED_RESOURCE_SELECTION: (),
    },
    "ablation": {
        CONTROL_OWNED: (
            "metadata_file", "label_column", "split_column", "max_rows",
            "max_categories", "max_probe_features", "max_probe_iterations", "seed",
            "confidence_level", "bootstrap_iterations",
        ),
        MODEL_OWNED: (
            "feature_fields", "ablate_field", "train_split_values",
            "test_split_values", "test_fraction",
        ),
        AUTHORIZED_RESOURCE_SELECTION: (),
    },
    "metric_computation": {
        CONTROL_OWNED: (
            "label_column", "max_rows", "seed", "confidence_level",
            "bootstrap_iterations",
        ),
        MODEL_OWNED: ("prediction_column", "metric"),
        AUTHORIZED_RESOURCE_SELECTION: ("predictions_file", "predictions_source"),
    },
    "task_model_training": {
        CONTROL_OWNED: (
            "metadata_file", "sample_id_column", "sample_path_column", "label_column",
            "split_column", "train_split_values", "validation_split_values",
            "test_split_values", "model_id", "model_backend", "feature_version",
            "max_train_samples", "max_eval_samples", "max_model_iterations", "seed",
            "artifact_policy", "confidence_level", "bootstrap_iterations",
            "class_weight", "holdout_fraction",
        ),
        MODEL_OWNED: (),
        AUTHORIZED_RESOURCE_SELECTION: (),
    },
    "task_model_evaluation": {
        CONTROL_OWNED: (
            "model_reference", "metadata_file", "sample_id_column", "sample_path_column",
            "label_column", "split_column", "split_values", "evaluation_partition",
            "max_eval_samples", "seed", "confidence_level", "bootstrap_iterations",
        ),
        MODEL_OWNED: (),
        AUTHORIZED_RESOURCE_SELECTION: (),
    },
    "ood_evaluation": {
        CONTROL_OWNED: (
            "model_reference", "metadata_file", "sample_id_column", "sample_path_column",
            "label_column", "split_column", "split_values", "evaluation_partition",
            "max_eval_samples", "max_intervention_samples", "seed", "confidence_level",
            "bootstrap_iterations",
        ),
        MODEL_OWNED: (
            "transformation", "target_cue", "intended_invariants", "comparison_mode",
            "preservation_status",
        ),
        AUTHORIZED_RESOURCE_SELECTION: (),
    },
    "counterfactual_intervention": {
        CONTROL_OWNED: (
            "model_reference", "metadata_file", "sample_id_column", "sample_path_column",
            "label_column", "split_column", "split_values", "evaluation_partition",
            "max_eval_samples", "max_intervention_samples", "seed", "confidence_level",
            "bootstrap_iterations",
        ),
        MODEL_OWNED: (
            "transformation", "target_cue", "intended_invariants", "preservation_status",
        ),
        AUTHORIZED_RESOURCE_SELECTION: (),
    },
}


class ExperimentContractError(ValueError):
    category = "experiment_contract_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ExperimentDraftInvalid(ExperimentContractError):
    category = "experiment_draft_invalid"


class AuthoritativeResourceMissing(ExperimentContractError):
    category = "authoritative_resource_missing"


class UnsafeResourcePath(ExperimentContractError):
    category = "unsafe_resource_path"


class UnauthorizedScientificField(ExperimentContractError):
    category = "unauthorized_scientific_field"


class ExperimentDependencyPending(ExperimentContractError):
    category = "experiment_dependency_pending"


class CanonicalExperimentInputInvalid(ExperimentContractError):
    category = "canonical_experiment_input_invalid"


class ToolCallDraftInvalid(ExperimentContractError):
    category = "tool_call_draft_invalid"


class ToolPermissionFailure(ExperimentContractError):
    category = "tool_permission_failure"


class BackendExecutionFailure(ExperimentContractError):
    category = "backend_execution_failure"


class StatisticalCheckDraft(StrictProtocolModel):
    cue_field: NonEmptyText
    variable_mode: Literal[
        "auto", "categorical_categorical", "numeric_categorical", "numeric_numeric"
    ] = "auto"
    n_permutations: int = Field(default=0, ge=0, le=1_000)


class SubgroupEvaluationDraft(StrictProtocolModel):
    group_field: NonEmptyText
    predictions_file: str | None = None
    predictions_source: Literal["dataset", "run"] = "dataset"
    prediction_sample_id_column: NonEmptyText | None = None
    prediction_column: NonEmptyText = "prediction"


class PredictiveProbeDraft(StrictProtocolModel):
    feature_fields: list[NonEmptyText] = Field(min_length=1)
    train_split_values: list[NonEmptyText] = Field(default_factory=list)
    test_split_values: list[NonEmptyText] = Field(default_factory=list)
    test_fraction: float = Field(default=0.25, gt=0, lt=1)


class MetadataInterventionDraft(StrictProtocolModel):
    cue_field: NonEmptyText
    intervention_type: Literal["shuffle", "within_split_shuffle"] = "shuffle"


class MetadataAblationDraft(PredictiveProbeDraft):
    ablate_field: NonEmptyText


class MetricComputationDraft(StrictProtocolModel):
    predictions_file: str
    predictions_source: Literal["dataset", "run"] = "dataset"
    prediction_column: NonEmptyText
    metric: Literal["accuracy", "balanced_accuracy"] = "accuracy"


class TaskModelTrainingDraft(StrictProtocolModel):
    class_weight: Literal["none", "balanced"] = "none"
    holdout_fraction: float = Field(default=0.25, gt=0, lt=0.5)


class TaskModelEvaluationDraft(StrictProtocolModel):
    pass


class OODEvaluationDraft(StrictProtocolModel):
    transformation: ImageInterventionSpec
    target_cue: NonEmptyText
    intended_invariants: list[NonEmptyText] = Field(
        default_factory=lambda: ["task label semantics"]
    )
    comparison_mode: ComparisonMode
    preservation_status: LabelPreservationStatus = "unverified"


class CounterfactualInterventionDraft(StrictProtocolModel):
    transformation: ImageInterventionSpec
    target_cue: NonEmptyText
    intended_invariants: list[NonEmptyText] = Field(
        default_factory=lambda: ["task label semantics"]
    )
    preservation_status: LabelPreservationStatus = "unverified"


DraftParameters = (
    StatisticalCheckDraft
    | SubgroupEvaluationDraft
    | PredictiveProbeDraft
    | MetadataInterventionDraft
    | MetadataAblationDraft
    | MetricComputationDraft
    | TaskModelTrainingDraft
    | TaskModelEvaluationDraft
    | OODEvaluationDraft
    | CounterfactualInterventionDraft
)


class RunExperimentDraft(StrictProtocolModel):
    experiment_type: ExperimentType
    parameters: DraftParameters

    @model_validator(mode="before")
    @classmethod
    def parse_parameters_for_type(cls, value):
        if not isinstance(value, dict):
            return value
        experiment_type = value.get("experiment_type")
        expected = {
            "statistical_check": StatisticalCheckDraft,
            "subgroup_evaluation": SubgroupEvaluationDraft,
            "model_evaluation": PredictiveProbeDraft,
            "intervention": MetadataInterventionDraft,
            "ablation": MetadataAblationDraft,
            "metric_computation": MetricComputationDraft,
            "task_model_training": TaskModelTrainingDraft,
            "task_model_evaluation": TaskModelEvaluationDraft,
            "ood_evaluation": OODEvaluationDraft,
            "counterfactual_intervention": CounterfactualInterventionDraft,
        }.get(experiment_type)
        if expected is None:
            return value
        return {
            **value,
            "parameters": expected.model_validate(value.get("parameters", {})),
        }

    @model_validator(mode="after")
    def validate_parameters_for_type(self):
        expected = {
            "statistical_check": StatisticalCheckDraft,
            "subgroup_evaluation": SubgroupEvaluationDraft,
            "model_evaluation": PredictiveProbeDraft,
            "intervention": MetadataInterventionDraft,
            "ablation": MetadataAblationDraft,
            "metric_computation": MetricComputationDraft,
            "task_model_training": TaskModelTrainingDraft,
            "task_model_evaluation": TaskModelEvaluationDraft,
            "ood_evaluation": OODEvaluationDraft,
            "counterfactual_intervention": CounterfactualInterventionDraft,
        }[self.experiment_type]
        if not isinstance(self.parameters, expected):
            raise ValueError("draft parameters do not match experiment_type")
        return self


DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "task_model_evaluation": ("task_model_training",),
    "ood_evaluation": ("task_model_training",),
    "counterfactual_intervention": ("task_model_training",),
}


def _task_results(
    history: list[ToolResult],
    *,
    run_id: str,
    candidate_id: str,
    task_id: str,
) -> list[ToolResult]:
    return [
        item for item in history
        if item.run_id == run_id
        and item.agent_name == "quantitative"
        and item.candidate_id == candidate_id
        and item.task_id == task_id
    ]


def _successful_types(
    history: list[ToolResult],
    *,
    run_id: str,
    candidate_id: str,
    task_id: str,
) -> set[str]:
    return {
        str(item.output.get("metadata", {}).get("experiment_type"))
        for item in _task_results(
            history,
            run_id=run_id,
            candidate_id=candidate_id,
            task_id=task_id,
        )
        if item.tool_name == "run_experiment"
        and item.status == "success"
        and item.output is not None
    }


def _model_references(
    *,
    history: list[ToolResult],
    explicit_artifacts: set[str],
    run_id: str,
    candidate_id: str,
    task_id: str,
) -> list[str]:
    values = {
        reference
        for item in _task_results(
            history,
            run_id=run_id,
            candidate_id=candidate_id,
            task_id=task_id,
        )
        if item.status == "success" and item.tool_name == "run_experiment"
        for reference in item.artifacts
        if reference.endswith("/model.json")
    }
    values.update(
        reference
        for reference in explicit_artifacts
        if reference.endswith("/model.json")
    )
    return sorted(values)


def classify_experiment_availability(
    resolved: list[str],
    *,
    history: list[ToolResult],
    explicit_artifacts: set[str],
    run_id: str,
    candidate_id: str,
    task_id: str,
) -> tuple[list[str], list[str], list[str]]:
    completed = _successful_types(
        history,
        run_id=run_id,
        candidate_id=candidate_id,
        task_id=task_id,
    )
    model_available = bool(_model_references(
        history=history,
        explicit_artifacts=explicit_artifacts,
        run_id=run_id,
        candidate_id=candidate_id,
        task_id=task_id,
    ))
    available: list[str] = []
    pending: list[str] = []
    unsupported: list[str] = []
    known = set(EXPERIMENT_PARAMETER_OWNERSHIP)
    for experiment_type in resolved:
        if experiment_type not in known:
            unsupported.append(experiment_type)
            continue
        if experiment_type in completed:
            continue
        dependencies = DEPENDENCIES.get(experiment_type, ())
        if dependencies and not model_available and not set(dependencies) & completed:
            pending.append(experiment_type)
        else:
            available.append(experiment_type)
    return available, pending, unsupported


def provider_run_experiment_schema() -> dict[str, Any]:
    return RunExperimentDraft.model_json_schema()


def _require_resource(value: Any, name: str) -> Any:
    if value is None or value == "":
        raise AuthoritativeResourceMissing(f"authoritative resource missing: {name}")
    return value


def _authorize_fields(values: list[str], resources: QuantitativeExecutionResources) -> list[str]:
    allowed = set(resources.allowed_scientific_metadata_fields)
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise UnauthorizedScientificField(
            "scientific field is not authorized: " + ", ".join(invalid)
        )
    return values


def _authorize_split_values(
    values: list[str], resources: QuantitativeExecutionResources
) -> list[str]:
    invalid = sorted(set(values) - set(resources.available_splits))
    if invalid:
        raise UnauthorizedScientificField(
            "split value is not authorized: " + ", ".join(invalid)
        )
    if values and resources.split_column is None:
        raise AuthoritativeResourceMissing(
            "split values require an authoritative split_column"
        )
    return values


def _authorize_transformation(
    transformation: ImageInterventionSpec,
    resources: QuantitativeExecutionResources,
    dataset_root: Path,
) -> ImageInterventionSpec:
    value = TypeAdapter(ImageInterventionSpec).dump_python(
        transformation, mode="json"
    )
    mask_column = value.get("mask_column")
    if mask_column is not None:
        _authorize_fields([mask_column], resources)
    reference = value.get("replacement_sample_reference")
    if reference is not None:
        value["replacement_sample_reference"] = _normalize_authorized_path(
            reference, root=dataset_root, authorized=None
        )
    return TypeAdapter(ImageInterventionSpec).validate_python(value)


def _normalize_authorized_path(
    value: str,
    *,
    root: Path,
    authorized: set[str] | None,
) -> str:
    if PureWindowsPath(value).is_absolute() and not Path(value).is_absolute():
        raise UnsafeResourcePath("Windows absolute path is outside the active approved root")
    root = root.resolve(strict=True)
    raw = Path(value)
    candidate = raw.resolve(strict=False) if raw.is_absolute() else (root / raw).resolve(strict=False)
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise UnsafeResourcePath("resource path resolves outside its approved root") from exc
    try:
        safe = TypeAdapter(SafeRelativePath).validate_python(relative)
    except ValidationError as exc:
        raise UnsafeResourcePath("resource path is not a safe relative path") from exc
    if not candidate.is_file():
        raise AuthoritativeResourceMissing("authorized resource file does not exist")
    if authorized is not None and safe not in authorized:
        raise UnsafeResourcePath("resource path is not in the authorized reference set")
    return safe


def _common(resources: QuantitativeExecutionResources) -> dict[str, Any]:
    return {
        "seed": resources.seed,
        "confidence_level": resources.confidence_level,
        "bootstrap_iterations": resources.bootstrap_iterations,
    }


def _model_id(run_id: str, task_id: str) -> str:
    digest = hashlib.sha256(f"{run_id}:{task_id}".encode("utf-8")).hexdigest()[:24]
    return f"TM-{digest}"


def finalize_run_experiment_draft(
    value: dict[str, Any],
    *,
    resources: QuantitativeExecutionResources,
    requirement: QuantitativeExecutionRequirement,
    run_id: str,
    candidate_id: str,
    task_id: str,
    dataset_root: Path,
    run_root: Path | None,
    approved_artifacts: set[str],
    explicit_artifacts: set[str],
    history: list[ToolResult],
) -> ExperimentToolInput:
    try:
        draft = RunExperimentDraft.model_validate(value)
    except ValidationError as exc:
        raise ExperimentDraftInvalid("provider experiment draft is invalid") from exc
    if draft.experiment_type not in requirement.resolved_experiment_types:
        raise ExperimentDraftInvalid("experiment type was not resolved from Executive checks")
    available, pending, _unsupported = classify_experiment_availability(
        requirement.resolved_experiment_types,
        history=history,
        explicit_artifacts=explicit_artifacts,
        run_id=run_id,
        candidate_id=candidate_id,
        task_id=task_id,
    )
    if draft.experiment_type in pending:
        raise ExperimentDependencyPending(
            f"experiment dependency pending: {draft.experiment_type}"
        )
    if draft.experiment_type not in available:
        raise ExperimentDraftInvalid("experiment type is not currently callable")

    metadata_file = resources.metadata_file
    if metadata_file is not None:
        _normalize_authorized_path(
            metadata_file,
            root=dataset_root,
            authorized=None,
        )
    common = _common(resources)
    parameters: dict[str, Any]
    choice = draft.parameters
    if isinstance(choice, StatisticalCheckDraft):
        _authorize_fields([choice.cue_field], resources)
        parameters = {
            **common,
            "metadata_file": _require_resource(metadata_file, "metadata_file"),
            "cue_field": choice.cue_field,
            "label_column": resources.label_column,
            "variable_mode": choice.variable_mode,
            "max_rows": resources.max_rows,
            "max_categories": resources.max_categories,
            "min_group_size": resources.min_group_size,
            "n_permutations": min(
                resources.statistical_n_permutations,
                resources.max_permutations,
            ),
        }
    elif isinstance(choice, SubgroupEvaluationDraft):
        _authorize_fields([choice.group_field], resources)
        predictions_file = None
        if choice.predictions_file is not None:
            source_root = dataset_root if choice.predictions_source == "dataset" else _require_resource(run_root, "run_root")
            predictions_file = _normalize_authorized_path(
                choice.predictions_file,
                root=source_root,
                authorized=(None if choice.predictions_source == "dataset" else approved_artifacts),
            )
        parameters = {
            **common,
            "metadata_file": _require_resource(metadata_file, "metadata_file"),
            "group_field": choice.group_field,
            "label_column": resources.label_column,
            "predictions_file": predictions_file,
            "predictions_source": choice.predictions_source,
            "sample_id_column": (
                resources.sample_id_column
                if predictions_file and choice.prediction_sample_id_column
                else None
            ),
            "prediction_sample_id_column": choice.prediction_sample_id_column if predictions_file else None,
            "prediction_column": choice.prediction_column,
            "max_rows": resources.max_rows,
            "max_categories": resources.max_categories,
            "min_group_size": resources.min_group_size,
        }
    elif isinstance(choice, MetadataAblationDraft):
        _authorize_fields([*choice.feature_fields, choice.ablate_field], resources)
        _authorize_split_values(
            [*choice.train_split_values, *choice.test_split_values], resources
        )
        parameters = {
            **common,
            "metadata_file": _require_resource(metadata_file, "metadata_file"),
            "feature_fields": choice.feature_fields,
            "ablate_field": choice.ablate_field,
            "label_column": resources.label_column,
            "split_column": resources.split_column if (choice.train_split_values or choice.test_split_values) else None,
            "train_split_values": choice.train_split_values,
            "test_split_values": choice.test_split_values,
            "test_fraction": choice.test_fraction,
            "max_rows": resources.max_rows,
            "max_categories": resources.max_categories,
            "max_probe_features": resources.max_probe_features,
            "max_probe_iterations": resources.max_probe_iterations,
        }
    elif isinstance(choice, PredictiveProbeDraft):
        _authorize_fields(choice.feature_fields, resources)
        _authorize_split_values(
            [*choice.train_split_values, *choice.test_split_values], resources
        )
        parameters = {
            **common,
            "metadata_file": _require_resource(metadata_file, "metadata_file"),
            "feature_fields": choice.feature_fields,
            "label_column": resources.label_column,
            "split_column": resources.split_column if (choice.train_split_values or choice.test_split_values) else None,
            "train_split_values": choice.train_split_values,
            "test_split_values": choice.test_split_values,
            "test_fraction": choice.test_fraction,
            "max_rows": resources.max_rows,
            "max_categories": resources.max_categories,
            "max_probe_features": resources.max_probe_features,
            "max_probe_iterations": resources.max_probe_iterations,
        }
    elif isinstance(choice, MetadataInterventionDraft):
        _authorize_fields([choice.cue_field], resources)
        parameters = {
            **common,
            "metadata_file": _require_resource(metadata_file, "metadata_file"),
            "cue_field": choice.cue_field,
            "label_column": resources.label_column,
            "intervention_type": choice.intervention_type,
            "split_column": resources.split_column if choice.intervention_type == "within_split_shuffle" else None,
            "max_rows": resources.max_rows,
            "max_categories": resources.max_categories,
            "min_group_size": resources.min_group_size,
        }
    elif isinstance(choice, MetricComputationDraft):
        source_root = dataset_root if choice.predictions_source == "dataset" else _require_resource(run_root, "run_root")
        predictions_file = _normalize_authorized_path(
            choice.predictions_file,
            root=source_root,
            authorized=(None if choice.predictions_source == "dataset" else approved_artifacts),
        )
        parameters = {
            **common,
            "predictions_file": predictions_file,
            "predictions_source": choice.predictions_source,
            "label_column": resources.label_column,
            "prediction_column": choice.prediction_column,
            "metric": choice.metric,
            "max_rows": resources.max_rows,
        }
    elif isinstance(choice, TaskModelTrainingDraft):
        parameters = {
            **common,
            "metadata_file": _require_resource(metadata_file, "metadata_file"),
            "sample_id_column": _require_resource(resources.sample_id_column, "sample_id_column"),
            "sample_path_column": _require_resource(resources.sample_path_column, "sample_path_column"),
            "label_column": resources.label_column,
            "split_column": None,
            "train_split_values": [],
            "validation_split_values": [],
            "test_split_values": [],
            "holdout_fraction": resources.training_holdout_fraction,
            "model_id": _model_id(run_id, task_id),
            "model_backend": resources.model_backend,
            "feature_version": resources.feature_version,
            "max_train_samples": max(4, resources.max_samples),
            "max_eval_samples": resources.max_samples,
            "max_model_iterations": resources.max_model_iterations,
            "class_weight": resources.training_class_weight,
            "artifact_policy": resources.artifact_policy,
        }
    else:
        model_references = _model_references(
            history=history,
            explicit_artifacts=explicit_artifacts,
            run_id=run_id,
            candidate_id=candidate_id,
            task_id=task_id,
        )
        if not model_references:
            raise ExperimentDependencyPending("model artifact dependency is not yet available")
        if len(model_references) != 1:
            raise AuthoritativeResourceMissing("model artifact authority is ambiguous")
        base = {
            **common,
            "model_reference": model_references[0],
            "metadata_file": _require_resource(metadata_file, "metadata_file"),
            "sample_id_column": _require_resource(resources.sample_id_column, "sample_id_column"),
            "sample_path_column": _require_resource(resources.sample_path_column, "sample_path_column"),
            "label_column": resources.label_column,
            "split_column": None,
            "split_values": [],
            "evaluation_partition": "training_manifest_test",
            "max_eval_samples": resources.max_samples,
        }
        if isinstance(choice, OODEvaluationDraft):
            parameters = {
                **base,
                "transformation": _authorize_transformation(
                    choice.transformation, resources, dataset_root
                ),
                "target_cue": choice.target_cue,
                "intended_invariants": choice.intended_invariants,
                "comparison_mode": choice.comparison_mode,
                "preservation_status": choice.preservation_status,
                "max_intervention_samples": resources.max_intervention_samples,
            }
        elif isinstance(choice, CounterfactualInterventionDraft):
            parameters = {
                **base,
                "transformation": _authorize_transformation(
                    choice.transformation, resources, dataset_root
                ),
                "target_cue": choice.target_cue,
                "intended_invariants": choice.intended_invariants,
                "preservation_status": choice.preservation_status,
                "max_intervention_samples": resources.max_intervention_samples,
            }
        else:
            parameters = base

    canonical = {
        "candidate_id": candidate_id,
        "task_id": task_id,
        "experiment_type": draft.experiment_type,
        "parameters": parameters,
    }
    try:
        return ExperimentToolInput.model_validate(canonical)
    except ValidationError as exc:
        raise CanonicalExperimentInputInvalid(
            "deterministic canonical ExperimentToolInput is invalid"
        ) from exc


def ownership_hash() -> str:
    return hashlib.sha256(
        json.dumps(
            EXPERIMENT_PARAMETER_OWNERSHIP,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


EXPERIMENT_PARAMETER_OWNERSHIP_HASH = ownership_hash()


__all__ = [
    "AUTHORIZED_RESOURCE_SELECTION",
    "AuthoritativeResourceMissing",
    "CONTROL_OWNED",
    "CanonicalExperimentInputInvalid",
    "ExperimentContractError",
    "ExperimentDependencyPending",
    "ExperimentDraftInvalid",
    "EXPERIMENT_PARAMETER_OWNERSHIP",
    "EXPERIMENT_PARAMETER_OWNERSHIP_HASH",
    "MODEL_OWNED",
    "RunExperimentDraft",
    "ToolCallDraftInvalid",
    "ToolPermissionFailure",
    "BackendExecutionFailure",
    "UnauthorizedScientificField",
    "UnsafeResourcePath",
    "classify_experiment_availability",
    "finalize_run_experiment_draft",
    "ownership_hash",
    "provider_run_experiment_schema",
]
