"""CPU-first, dataset-agnostic quantitative experiment execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import numpy as np
import pandas as pd
from pydantic import Field
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shortcutminer.experiment.tool_executor import (
    ExperimentExecutionOutput,
    ExperimentExecutionRequest,
    ExperimentOutcome,
    ExperimentType,
)
from shortcutminer.protocol.executive import TaskId
from shortcutminer.protocol.experiment import (
    EmptyExperimentParameters,
    MetadataAblationParameters,
    MetadataInterventionParameters,
    MetricComputationParameters,
    OODEvaluationParameters,
    PredictiveProbeParameters,
    StatisticalCheckParameters,
    SubgroupEvaluationParameters,
    TaskModelEvaluationParameters,
    TaskModelTrainingParameters,
    CounterfactualInterventionParameters,
)
from shortcutminer.experiment.model_reliance import (
    counterfactual_intervention_backend,
    ood_evaluation_backend,
    task_model_evaluation_backend,
    task_model_training_backend,
)
from shortcutminer.protocol.identify import (
    CandidateId,
    NonEmptyText,
    SafeRunId,
    StrictProtocolModel,
)
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.tooling.base import ToolExecutionContext
from shortcutminer.evaluation.calibration import bootstrap_interval
from shortcutminer.protocol.evidence import UncertaintyEstimate


class RealExperimentError(RuntimeError):
    """Execution/configuration failure, distinct from scientific uncertainty."""


class ExperimentDataError(RealExperimentError):
    pass


@dataclass
class BackendEvidence:
    outcome: str
    metrics: dict[str, str | int | float | bool | None]
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    input_references: list[str] = field(default_factory=list)
    uncertainty: list[UncertaintyEstimate] = field(default_factory=list)
    artifact_references: list[str] = field(default_factory=list)


class QuantitativeBackend(Protocol):
    def __call__(
        self,
        parameters: Any,
        access: "ExperimentAccess",
    ) -> BackendEvidence: ...


class RealExperimentArtifact(StrictProtocolModel):
    schema_version: Literal["1.0"]
    run_id: SafeRunId
    candidate_id: CandidateId
    task_id: TaskId
    experiment_type: ExperimentType
    backend: str
    parameters: dict[str, Any]
    seed: int | None
    outcome: ExperimentOutcome
    metrics: dict[str, str | int | float | bool | None]
    warnings: list[str]
    limitations: list[str]
    created_at: NonEmptyText
    input_references: list[str]
    details: dict[str, Any]
    uncertainty: list[UncertaintyEstimate] = Field(default_factory=list)
    output: ExperimentExecutionOutput


class ExperimentAccess:
    def __init__(self, context: ToolExecutionContext, *, max_rows: int, request=None) -> None:
        if context.agent_name != "quantitative":
            raise RealExperimentError("Real experiments require Quantitative context.")
        if context.dataset_root is None:
            raise RealExperimentError("No approved dataset root is configured.")
        self.context = context
        self.request = request
        self.max_rows = max_rows
        try:
            self.dataset_root = context.dataset_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RealExperimentError("Approved dataset root is unavailable.") from exc
        if not self.dataset_root.is_dir():
            raise RealExperimentError("Approved dataset root is not a directory.")

    @staticmethod
    def _beneath(root: Path, reference: str) -> Path:
        try:
            path = (root / reference).resolve(strict=True)
            path.relative_to(root)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise ExperimentDataError(
                f"Input file is unavailable or outside its approved root: {reference}"
            ) from exc
        if not path.is_file():
            raise ExperimentDataError(f"Input reference is not a file: {reference}")
        return path

    def dataset_file(self, reference: str) -> Path:
        return self._beneath(self.dataset_root, reference)

    def run_file(self, reference: str) -> Path:
        if self.context.runs_root is None:
            raise RealExperimentError("No approved runs root is configured.")
        if reference not in self.context.approved_artifacts:
            raise RealExperimentError("Run input is not an approved artifact reference.")
        root = (self.context.runs_root / self.context.run_id).resolve(strict=True)
        return self._beneath(root, reference)

    def load_csv(self, reference: str, *, source: str = "dataset") -> pd.DataFrame:
        path = (
            self.dataset_file(reference)
            if source == "dataset"
            else self.run_file(reference)
        )
        if path.suffix.casefold() != ".csv":
            raise ExperimentDataError("Real quantitative inputs must be CSV files.")
        try:
            frame = pd.read_csv(path, nrows=self.max_rows + 1)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ExperimentDataError(f"Unable to parse quantitative CSV: {reference}") from exc
        if len(frame) > self.max_rows:
            frame = frame.iloc[: self.max_rows].copy()
            frame.attrs["truncated"] = True
        else:
            frame.attrs["truncated"] = False
        return frame


def _require_columns(frame: pd.DataFrame, fields: list[str]) -> None:
    missing = [field for field in fields if field not in frame.columns]
    if missing:
        raise ExperimentDataError("Required columns are missing: " + ", ".join(missing))


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _counts(series: pd.Series) -> dict[str, int]:
    values = [
        (str(_json_value(value)), int(count))
        for value, count in series.value_counts(dropna=False, sort=False).items()
    ]
    return dict(sorted(values, key=lambda item: (item[0].casefold(), item[0])))


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _association_effect(frame: pd.DataFrame, cue: str, label: str, mode: str) -> float | None:
    """Compute only the effect statistic for bootstrap replicates."""
    if frame.empty:
        return None
    if mode == "categorical_categorical":
        table = pd.crosstab(frame[cue], frame[label])
        if min(table.shape) < 2:
            return None
        chi2 = float(stats.chi2_contingency(table, correction=False)[0])
        denominator = len(frame) * min(table.shape[0] - 1, table.shape[1] - 1)
        return float(np.sqrt(chi2 / denominator)) if denominator else None
    if mode == "numeric_categorical":
        values = pd.to_numeric(frame[cue], errors="coerce")
        usable = frame.assign(**{cue: values}).dropna(subset=[cue])
        if usable[label].nunique() < 2:
            return None
        array = usable[cue].to_numpy(dtype=float)
        total = float(np.sum((array - np.mean(array)) ** 2))
        if total == 0:
            return None
        between = sum(
            len(group) * (float(group[cue].mean()) - float(np.mean(array))) ** 2
            for _, group in usable.groupby(label, sort=True)
        )
        return float(between / total)
    numeric = frame[[cue, label]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(numeric) < 2 or numeric[cue].nunique() < 2 or numeric[label].nunique() < 2:
        return None
    return float(stats.pearsonr(numeric[cue], numeric[label]).statistic)


def _row_budget_warnings(
    frame: pd.DataFrame,
    *,
    max_rows: int,
    reference: str,
) -> list[str]:
    if not frame.attrs.get("truncated"):
        return []
    return [f"Input {reference} was truncated at max_rows={max_rows}."]


def _top_categories(
    frame: pd.DataFrame,
    field: str,
    max_categories: int,
) -> tuple[pd.DataFrame, list[str]]:
    counts = frame[field].value_counts(dropna=False)
    if len(counts) <= max_categories:
        return frame, []
    ordered = sorted(
        counts.items(),
        key=lambda item: (-int(item[1]), str(_json_value(item[0]))),
    )
    kept = {value for value, _ in ordered[:max_categories]}
    filtered = frame[frame[field].isin(kept)].copy()
    return filtered, [
        f"{field} exceeded max_categories={max_categories}; only deterministic top categories were retained."
    ]


def _infer_association_mode(
    frame: pd.DataFrame,
    parameters: StatisticalCheckParameters | MetadataInterventionParameters,
) -> str:
    configured = getattr(parameters, "variable_mode", "auto")
    if configured != "auto":
        return configured
    cue_numeric = pd.api.types.is_numeric_dtype(frame[parameters.cue_field])
    label_numeric = pd.api.types.is_numeric_dtype(frame[parameters.label_column])
    label_unique = frame[parameters.label_column].nunique(dropna=True)
    if cue_numeric and label_numeric and label_unique > parameters.max_categories:
        return "numeric_numeric"
    if cue_numeric:
        return "numeric_categorical"
    return "categorical_categorical"


def _association_evidence(
    frame: pd.DataFrame,
    parameters: StatisticalCheckParameters | MetadataInterventionParameters,
    *,
    include_tests: bool,
    estimate_uncertainty: bool = True,
) -> BackendEvidence:
    cue = parameters.cue_field
    label = parameters.label_column
    _require_columns(frame, [cue, label])
    missing_count = int(frame[[cue, label]].isna().any(axis=1).sum())
    valid = frame[[cue, label]].dropna().copy()
    base_metrics: dict[str, Any] = {
        "sample_count": int(len(frame)),
        "valid_count": int(len(valid)),
        "missing_count": missing_count,
    }
    if len(valid) < max(4, parameters.min_group_size * 2):
        return BackendEvidence(
            outcome="inconclusive",
            metrics=base_metrics,
            summary="Association evidence is scientifically inconclusive because too few valid rows are available.",
            limitations=["Insufficient valid sample size for association analysis."],
        )

    mode = _infer_association_mode(valid, parameters)
    warnings: list[str] = []
    details: dict[str, Any] = {"mode": mode}

    if mode == "categorical_categorical":
        valid, category_warnings = _top_categories(
            valid, cue, parameters.max_categories
        )
        warnings.extend(category_warnings)
        if valid[label].nunique() > parameters.max_categories:
            return BackendEvidence(
                outcome="inconclusive",
                metrics=base_metrics,
                summary="Association evidence is inconclusive because the label exceeds the category budget.",
                warnings=warnings,
                limitations=["Label cardinality exceeds max_categories."],
            )
        cue_counts = valid[cue].value_counts()
        rare = cue_counts[cue_counts < parameters.min_group_size]
        if not rare.empty:
            warnings.append(
                f"{len(rare)} cue categories have fewer than min_group_size={parameters.min_group_size} rows."
            )
        if valid[cue].nunique() < 2 or valid[label].nunique() < 2:
            return BackendEvidence(
                outcome="inconclusive",
                metrics=base_metrics,
                summary="Association evidence is inconclusive because one variable is constant or single-class.",
                warnings=warnings,
                limitations=["Both categorical variables require at least two observed values."],
            )
        table = pd.crosstab(valid[cue], valid[label])
        chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=False)
        denominator = len(valid) * min(table.shape[0] - 1, table.shape[1] - 1)
        cramers_v = float(np.sqrt(chi2 / denominator)) if denominator else 0.0
        metrics = {
            **base_metrics,
            "test_name": "chi_square",
            "statistic": float(chi2),
            "p_value": float(p_value) if include_tests else None,
            "effect_size": cramers_v,
            "effect_size_name": "cramers_v",
            "degrees_of_freedom": int(dof),
            "cue_category_count": int(table.shape[0]),
            "label_category_count": int(table.shape[1]),
        }
        details.update(
            contingency_table={
                str(row): {str(column): int(table.loc[row, column]) for column in table.columns}
                for row in table.index
            },
            expected_min=float(np.min(expected)),
            cue_counts=_counts(valid[cue]),
            label_counts=_counts(valid[label]),
        )
        if np.min(expected) < 5:
            warnings.append("Some expected contingency cells are below 5; asymptotic p-values may be unreliable.")

    elif mode == "numeric_categorical":
        numeric = pd.to_numeric(valid[cue], errors="coerce")
        valid = valid.assign(**{cue: numeric}).dropna(subset=[cue])
        groups = [
            (name, group[cue].to_numpy(dtype=float))
            for name, group in valid.groupby(label, sort=True)
        ]
        if len(groups) < 2 or any(len(values) < parameters.min_group_size for _, values in groups):
            return BackendEvidence(
                outcome="inconclusive",
                metrics={**base_metrics, "valid_count": int(len(valid))},
                summary="Numeric-group association is inconclusive because classes or group sizes are insufficient.",
                limitations=["At least two label groups meeting min_group_size are required."],
            )
        all_values = valid[cue].to_numpy(dtype=float)
        if np.allclose(all_values, all_values[0]):
            return BackendEvidence(
                outcome="inconclusive",
                metrics={**base_metrics, "valid_count": int(len(valid))},
                summary="Numeric-group association is inconclusive because the cue is constant.",
                limitations=["The numeric cue has zero variance."],
            )
        if len(groups) == 2:
            statistic, p_value = stats.ttest_ind(
                groups[0][1], groups[1][1], equal_var=False
            )
            test_name = "welch_t_test"
        else:
            statistic, p_value = stats.f_oneway(*(values for _, values in groups))
            test_name = "one_way_anova"
        grand_mean = float(np.mean(all_values))
        between = sum(
            len(values) * (float(np.mean(values)) - grand_mean) ** 2
            for _, values in groups
        )
        total = float(np.sum((all_values - grand_mean) ** 2))
        eta_squared = float(between / total) if total else 0.0
        metrics = {
            **base_metrics,
            "valid_count": int(len(valid)),
            "test_name": test_name,
            "statistic": _finite(float(statistic)),
            "p_value": _finite(float(p_value)) if include_tests else None,
            "effect_size": eta_squared,
            "effect_size_name": "eta_squared",
            "group_count": len(groups),
        }
        details["group_summaries"] = {
            str(name): {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
            for name, values in groups
        }

    else:
        x = pd.to_numeric(valid[cue], errors="coerce")
        y = pd.to_numeric(valid[label], errors="coerce")
        numeric = pd.DataFrame({cue: x, label: y}).dropna()
        if len(numeric) < max(4, parameters.min_group_size * 2):
            return BackendEvidence(
                outcome="inconclusive",
                metrics={**base_metrics, "valid_count": int(len(numeric))},
                summary="Numeric association is inconclusive because too few valid pairs are available.",
                limitations=["Insufficient numeric pairs."],
            )
        if numeric[cue].nunique() < 2 or numeric[label].nunique() < 2:
            return BackendEvidence(
                outcome="inconclusive",
                metrics={**base_metrics, "valid_count": int(len(numeric))},
                summary="Numeric association is inconclusive because one variable is constant.",
                limitations=["Pearson and Spearman correlation require non-constant variables."],
            )
        pearson = stats.pearsonr(numeric[cue], numeric[label])
        spearman = stats.spearmanr(numeric[cue], numeric[label])
        metrics = {
            **base_metrics,
            "valid_count": int(len(numeric)),
            "test_name": "pearson_correlation",
            "statistic": float(pearson.statistic),
            "p_value": float(pearson.pvalue) if include_tests else None,
            "effect_size": float(pearson.statistic),
            "effect_size_name": "pearson_r",
            "spearman_rho": float(spearman.statistic),
            "spearman_p_value": float(spearman.pvalue) if include_tests else None,
        }
        details["numeric_summary"] = {
            cue: {"mean": float(numeric[cue].mean()), "std": float(numeric[cue].std())},
            label: {"mean": float(numeric[label].mean()), "std": float(numeric[label].std())},
        }

    uncertainty = []
    if estimate_uncertainty:
        uncertainty.append(bootstrap_interval(
            metric="effect_size",
            sample_count=len(valid),
            statistic=lambda indices: _association_effect(valid.iloc[indices].reset_index(drop=True), cue, label, mode),
            confidence_level=parameters.confidence_level,
            iterations=parameters.bootstrap_iterations,
            seed=parameters.seed,
            resampling_unit="sample_row",
        ))
    return BackendEvidence(
        outcome="completed",
        metrics=metrics,
        summary="Association statistics and effect size were computed; they are evidence, not a shortcut judgment.",
        details=details,
        warnings=warnings,
        uncertainty=uncertainty,
    )


def _statistical_backend(parameters: StatisticalCheckParameters, access: ExperimentAccess) -> BackendEvidence:
    frame = access.load_csv(parameters.metadata_file)
    evidence = _association_evidence(frame, parameters, include_tests=True)
    evidence.input_references = [parameters.metadata_file]
    evidence.warnings.extend(
        _row_budget_warnings(
            frame,
            max_rows=parameters.max_rows,
            reference=parameters.metadata_file,
        )
    )
    if evidence.outcome == "completed" and parameters.n_permutations:
        valid = frame[[parameters.cue_field, parameters.label_column]].dropna().copy()
        observed = abs(float(evidence.metrics["effect_size"]))
        null_values = []
        random = np.random.default_rng(parameters.seed)
        for _ in range(parameters.n_permutations):
            permuted = valid.copy()
            permuted[parameters.label_column] = random.permutation(
                permuted[parameters.label_column].to_numpy()
            )
            null = _association_evidence(
                permuted,
                parameters,
                include_tests=False,
                estimate_uncertainty=False,
            )
            if null.outcome == "completed" and null.metrics["effect_size"] is not None:
                null_values.append(abs(float(null.metrics["effect_size"])))
        if null_values:
            evidence.metrics.update(
                permutation_observed=observed,
                permutation_null_mean=float(np.mean(null_values)),
                permutation_null_std=float(np.std(null_values)),
                permutation_p_value=float(
                    (1 + sum(value >= observed for value in null_values))
                    / (1 + len(null_values))
                ),
                n_permutations=len(null_values),
            )
    return evidence


def _subgroup_backend(parameters: SubgroupEvaluationParameters, access: ExperimentAccess) -> BackendEvidence:
    metadata = access.load_csv(parameters.metadata_file)
    _require_columns(metadata, [parameters.group_field, parameters.label_column])
    metadata, group_warnings = _top_categories(
        metadata, parameters.group_field, parameters.max_categories
    )
    group_warnings.extend(
        _row_budget_warnings(
            metadata,
            max_rows=parameters.max_rows,
            reference=parameters.metadata_file,
        )
    )
    reference = parameters.metadata_file
    if parameters.predictions_file is None:
        valid = metadata[[parameters.group_field, parameters.label_column]].dropna()
        groups = {
            str(name): {
                "sample_count": int(len(group)),
                "label_distribution": _counts(group[parameters.label_column]),
            }
            for name, group in valid.groupby(parameters.group_field, sort=True)
        }
        if len(groups) < 2:
            return BackendEvidence(
                outcome="inconclusive",
                metrics={"sample_count": int(len(valid)), "group_count": len(groups), "model_performance": False},
                summary="Subgroup label analysis is inconclusive because fewer than two groups are available.",
                limitations=["No prediction artifact was supplied; this is not a model performance audit."],
                warnings=group_warnings,
                input_references=[reference],
            )
        return BackendEvidence(
            outcome="completed",
            metrics={"sample_count": int(len(valid)), "group_count": len(groups), "model_performance": False},
            summary="Label distributions were summarized by subgroup; no model accuracy was computed.",
            details={"groups": groups},
            limitations=["No prediction artifact was supplied; this is not a model performance audit."],
            warnings=group_warnings,
            input_references=[reference],
        )

    reference = parameters.predictions_file
    if parameters.predictions_source == "dataset" and reference == parameters.metadata_file:
        joined = metadata
    else:
        predictions = access.load_csv(reference, source=parameters.predictions_source)
        group_warnings.extend(
            _row_budget_warnings(
                predictions,
                max_rows=parameters.max_rows,
                reference=reference,
            )
        )
        if not parameters.sample_id_column or not parameters.prediction_sample_id_column:
            raise ExperimentDataError("Separate predictions require explicit sample ID columns.")
        _require_columns(metadata, [parameters.sample_id_column])
        _require_columns(
            predictions,
            [parameters.prediction_sample_id_column, parameters.prediction_column],
        )
        if metadata[parameters.sample_id_column].duplicated().any() or predictions[
            parameters.prediction_sample_id_column
        ].duplicated().any():
            raise ExperimentDataError("Duplicate sample IDs prevent an unambiguous prediction join.")
        joined = metadata.merge(
            predictions[
                [parameters.prediction_sample_id_column, parameters.prediction_column]
            ],
            left_on=parameters.sample_id_column,
            right_on=parameters.prediction_sample_id_column,
            how="left",
        )
    _require_columns(
        joined,
        [parameters.group_field, parameters.label_column, parameters.prediction_column],
    )
    missing_predictions = int(joined[parameters.prediction_column].isna().sum())
    valid = joined[
        [parameters.group_field, parameters.label_column, parameters.prediction_column]
    ].dropna()
    if valid[parameters.group_field].nunique() < 2 or valid.empty:
        return BackendEvidence(
            outcome="inconclusive",
            metrics={"sample_count": int(len(valid)), "missing_prediction_count": missing_predictions},
            summary="Model subgroup evaluation is inconclusive because valid groups are insufficient.",
            limitations=["At least two subgroups with predictions are required."],
            warnings=group_warnings,
            input_references=[parameters.metadata_file, reference],
        )
    group_results = {}
    for name, group in valid.groupby(parameters.group_field, sort=True):
        group_results[str(name)] = {
            "sample_count": int(len(group)),
            "accuracy": float(
                accuracy_score(group[parameters.label_column], group[parameters.prediction_column])
            ),
        }
    accuracies = [item["accuracy"] for item in group_results.values()]
    warnings = list(group_warnings)
    tiny = sum(
        item["sample_count"] < parameters.min_group_size
        for item in group_results.values()
    )
    if tiny:
        warnings.append(f"{tiny} subgroups are smaller than min_group_size={parameters.min_group_size}.")
    labels = valid[parameters.label_column].to_numpy()
    predictions_array = valid[parameters.prediction_column].to_numpy()
    groups_array = valid[parameters.group_field].astype(str).to_numpy()

    def subgroup_stat(indices: np.ndarray, name: str) -> float | None:
        selected_groups = groups_array[indices]
        selected_labels = labels[indices]
        selected_predictions = predictions_array[indices]
        unique_groups = sorted(set(selected_groups))
        if name == "overall_accuracy":
            return float(accuracy_score(selected_labels, selected_predictions))
        values = [
            float(accuracy_score(selected_labels[selected_groups == group], selected_predictions[selected_groups == group]))
            for group in unique_groups
            if np.any(selected_groups == group)
        ]
        if len(values) < 2:
            return None
        return min(values) if name == "worst_group_accuracy" else max(values) - min(values)

    uncertainty = [
        bootstrap_interval(
            metric=name,
            sample_count=len(valid),
            statistic=lambda indices, metric=name: subgroup_stat(indices, metric),
            confidence_level=parameters.confidence_level,
            iterations=parameters.bootstrap_iterations,
            seed=parameters.seed + offset,
            resampling_unit="prediction_row_preserving_group_identity",
        )
        for offset, name in enumerate(("overall_accuracy", "worst_group_accuracy", "group_accuracy_gap"))
    ]
    return BackendEvidence(
        outcome="completed",
        metrics={
            "sample_count": int(len(valid)),
            "missing_prediction_count": missing_predictions,
            "overall_accuracy": float(
                accuracy_score(valid[parameters.label_column], valid[parameters.prediction_column])
            ),
            "worst_group_accuracy": float(min(accuracies)),
            "best_group_accuracy": float(max(accuracies)),
            "group_accuracy_gap": float(max(accuracies) - min(accuracies)),
            "group_count": len(group_results),
            "model_performance": True,
        },
        summary="Prediction accuracy was measured overall and by the explicitly resolved subgroup field.",
        details={"groups": group_results},
        warnings=warnings,
        input_references=[parameters.metadata_file, reference],
        uncertainty=uncertainty,
    )


def _split_probe_data(
    frame: pd.DataFrame,
    parameters: PredictiveProbeParameters,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    features = parameters.feature_fields
    label = parameters.label_column
    valid = frame[[*features, label, *([parameters.split_column] if parameters.split_column else [])]].dropna(subset=[label])
    if parameters.split_column and parameters.train_split_values:
        split = valid[parameters.split_column].astype(str)
        train_mask = split.isin(parameters.train_split_values)
        test_mask = split.isin(parameters.test_split_values)
        train = valid[train_mask]
        test = valid[test_mask]
        strategy = "explicit_split"
        if train.empty or test.empty:
            raise ValueError("Configured train/test split values produced an empty partition.")
        return train[features], test[features], train[label], test[label], strategy
    train_x, test_x, train_y, test_y = train_test_split(
        valid[features],
        valid[label],
        test_size=parameters.test_fraction,
        random_state=parameters.seed,
        stratify=valid[label],
    )
    return train_x, test_x, train_y, test_y, "deterministic_stratified_holdout"


def _probe_evidence(
    frame: pd.DataFrame,
    parameters: PredictiveProbeParameters,
    *,
    feature_fields: list[str] | None = None,
) -> BackendEvidence:
    features = feature_fields or parameters.feature_fields
    if parameters.label_column in features:
        raise RealExperimentError("Target label cannot be used as a probe feature.")
    _require_columns(
        frame,
        [*features, parameters.label_column]
        + ([parameters.split_column] if parameters.split_column else []),
    )
    if frame[parameters.label_column].dropna().nunique() < 2:
        return BackendEvidence(
            outcome="inconclusive",
            metrics={"sample_count": int(frame[parameters.label_column].notna().sum())},
            summary="Predictive probe is inconclusive because the target has one class.",
            limitations=["Out-of-sample classification requires at least two target classes."],
        )
    for feature in features:
        if frame[feature].nunique(dropna=True) > parameters.max_categories:
            return BackendEvidence(
                outcome="inconclusive",
                metrics={"sample_count": int(len(frame))},
                summary="Predictive probe is inconclusive because a feature exceeds the encoding category budget.",
                limitations=[
                    f"Feature {feature} exceeds max_categories={parameters.max_categories}."
                ],
            )
    try:
        configured = parameters.model_copy(update={"feature_fields": features})
        train_x, test_x, train_y, test_y, strategy = _split_probe_data(frame, configured)
    except ValueError as exc:
        return BackendEvidence(
            outcome="inconclusive",
            metrics={"sample_count": int(len(frame))},
            summary="Predictive probe could not form a valid out-of-sample split.",
            limitations=[str(exc)],
        )
    if train_y.nunique() < 2 or test_y.nunique() < 2:
        return BackendEvidence(
            outcome="inconclusive",
            metrics={"train_n": len(train_y), "test_n": len(test_y)},
            summary="Predictive probe split is scientifically inconclusive because a partition is single-class.",
            limitations=["Both train and test partitions require at least two classes."],
        )
    numeric = [field for field in features if pd.api.types.is_numeric_dtype(train_x[field])]
    categorical = [field for field in features if field not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    model = Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers)),
            (
                "model",
                LogisticRegression(
                    max_iter=parameters.max_probe_iterations,
                    random_state=parameters.seed,
                ),
            ),
        ]
    )
    try:
        model.fit(train_x, train_y)
        prediction = model.predict(test_x)
    except (TypeError, ValueError, FloatingPointError) as exc:
        return BackendEvidence(
            outcome="inconclusive",
            metrics={"train_n": len(train_y), "test_n": len(test_y)},
            summary="Predictive probe fitting was scientifically inconclusive.",
            limitations=[f"Probe fitting limitation: {exc}"],
        )
    majority = train_y.mode(dropna=True).iloc[0]
    baseline = np.full(len(test_y), majority, dtype=train_y.to_numpy().dtype)
    accuracy = float(accuracy_score(test_y, prediction))
    baseline_accuracy = float(accuracy_score(test_y, baseline))
    balanced = float(balanced_accuracy_score(test_y, prediction))
    labels = test_y.to_numpy()
    uncertainty = []
    for offset, (name, values) in enumerate((
        ("probe_accuracy", prediction),
        ("majority_baseline_accuracy", baseline),
    )):
        uncertainty.append(bootstrap_interval(
            metric=name,
            sample_count=len(labels),
            statistic=lambda indices, predicted=np.asarray(values): float(accuracy_score(labels[indices], predicted[indices])),
            confidence_level=parameters.confidence_level,
            iterations=parameters.bootstrap_iterations,
            seed=parameters.seed + offset,
            resampling_unit="held_out_prediction_row",
        ))
    uncertainty.append(bootstrap_interval(
        metric="improvement",
        sample_count=len(labels),
        statistic=lambda indices: float(
            accuracy_score(labels[indices], np.asarray(prediction)[indices])
            - accuracy_score(labels[indices], np.asarray(baseline)[indices])
        ),
        confidence_level=parameters.confidence_level,
        iterations=parameters.bootstrap_iterations,
        seed=parameters.seed + 2,
        resampling_unit="paired_held_out_prediction_row",
    ))
    uncertainty.append(bootstrap_interval(
        metric="balanced_accuracy",
        sample_count=len(labels),
        statistic=lambda indices: float(balanced_accuracy_score(labels[indices], np.asarray(prediction)[indices])),
        confidence_level=parameters.confidence_level,
        iterations=parameters.bootstrap_iterations,
        seed=parameters.seed + 3,
        resampling_unit="held_out_prediction_row",
    ))
    return BackendEvidence(
        outcome="completed",
        metrics={
            "sample_count": int(len(train_y) + len(test_y)),
            "train_n": int(len(train_y)),
            "test_n": int(len(test_y)),
            "probe_accuracy": accuracy,
            "balanced_accuracy": balanced,
            "majority_baseline_accuracy": baseline_accuracy,
            "improvement": float(accuracy - baseline_accuracy),
            "class_count": int(train_y.nunique()),
            "feature_count": len(features),
            "seed": parameters.seed,
        },
        summary="A bounded logistic-regression probe was evaluated out of sample; predictability alone does not confirm a shortcut.",
        details={
            "split_strategy": strategy,
            "feature_encoding": "median-scaled numeric plus one-hot categorical",
            "features": features,
            "train_class_distribution": _counts(train_y),
            "test_class_distribution": _counts(test_y),
        },
        uncertainty=uncertainty,
    )


def _probe_backend(parameters: PredictiveProbeParameters, access: ExperimentAccess) -> BackendEvidence:
    frame = access.load_csv(parameters.metadata_file)
    result = _probe_evidence(frame, parameters)
    result.input_references = [parameters.metadata_file]
    result.warnings.extend(
        _row_budget_warnings(
            frame,
            max_rows=parameters.max_rows,
            reference=parameters.metadata_file,
        )
    )
    return result


def _intervention_backend(parameters: MetadataInterventionParameters, access: ExperimentAccess) -> BackendEvidence:
    frame = access.load_csv(parameters.metadata_file)
    _require_columns(
        frame,
        [parameters.cue_field, parameters.label_column]
        + ([parameters.split_column] if parameters.split_column else []),
    )
    before = _association_evidence(frame, parameters, include_tests=False)
    before.warnings.extend(
        _row_budget_warnings(
            frame,
            max_rows=parameters.max_rows,
            reference=parameters.metadata_file,
        )
    )
    if before.outcome != "completed":
        before.summary = "Metadata intervention is inconclusive because the pre-intervention association is not estimable."
        before.input_references = [parameters.metadata_file]
        return before
    intervened = frame.copy()
    original = intervened[parameters.cue_field].copy()
    if parameters.intervention_type == "shuffle":
        intervened[parameters.cue_field] = original.sample(
            frac=1, random_state=parameters.seed
        ).to_numpy()
    else:
        shuffled = original.copy()
        for offset, (_, group) in enumerate(
            intervened.groupby(parameters.split_column, sort=True)
        ):
            shuffled.loc[group.index] = original.loc[group.index].sample(
                frac=1, random_state=(parameters.seed + offset) % (2**32)
            ).to_numpy()
        intervened[parameters.cue_field] = shuffled
    after = _association_evidence(intervened, parameters, include_tests=False)
    if after.outcome != "completed":
        return BackendEvidence(
            outcome="inconclusive",
            metrics=before.metrics,
            summary="Metadata intervention ran, but the post-intervention statistic was not estimable.",
            limitations=after.limitations,
            input_references=[parameters.metadata_file],
        )
    before_effect = float(before.metrics["effect_size"])
    after_effect = float(after.metrics["effect_size"])
    rows_affected = int((original != intervened[parameters.cue_field]).fillna(False).sum())
    before_valid = frame[[parameters.cue_field, parameters.label_column]].dropna()
    after_valid = intervened.loc[before_valid.index, [parameters.cue_field, parameters.label_column]]
    mode = _infer_association_mode(before_valid, parameters)

    def intervention_delta(indices: np.ndarray) -> float | None:
        before_value = _association_effect(before_valid.iloc[indices].reset_index(drop=True), parameters.cue_field, parameters.label_column, mode)
        after_value = _association_effect(after_valid.iloc[indices].reset_index(drop=True), parameters.cue_field, parameters.label_column, mode)
        return None if before_value is None or after_value is None else after_value - before_value

    uncertainty = [bootstrap_interval(
        metric="effect_size_delta",
        sample_count=len(before_valid),
        statistic=intervention_delta,
        confidence_level=parameters.confidence_level,
        iterations=parameters.bootstrap_iterations,
        seed=parameters.seed,
        resampling_unit="paired_sample_row",
    )]
    return BackendEvidence(
        outcome="completed",
        metrics={
            "sample_count": int(len(frame)),
            "rows_affected": rows_affected,
            "before_effect_size": before_effect,
            "after_effect_size": after_effect,
            "effect_size_delta": float(after_effect - before_effect),
            "effect_size_name": str(before.metrics["effect_size_name"]),
            "seed": parameters.seed,
        },
        summary="The resolved metadata cue was permuted and its association effect size recomputed; no final shortcut judgment was made.",
        details={
            "intervention_type": parameters.intervention_type,
            "before": before.details,
            "after": after.details,
        },
        warnings=[*before.warnings, *after.warnings],
        input_references=[parameters.metadata_file],
        uncertainty=uncertainty,
    )


def _ablation_backend(parameters: MetadataAblationParameters, access: ExperimentAccess) -> BackendEvidence:
    frame = access.load_csv(parameters.metadata_file)
    row_warnings = _row_budget_warnings(
        frame,
        max_rows=parameters.max_rows,
        reference=parameters.metadata_file,
    )
    full = _probe_evidence(frame, parameters)
    remaining = [field for field in parameters.feature_fields if field != parameters.ablate_field]
    ablated = _probe_evidence(frame, parameters, feature_fields=remaining)
    if full.outcome != "completed" or ablated.outcome != "completed":
        return BackendEvidence(
            outcome="inconclusive",
            metrics={},
            summary="Metadata feature ablation is inconclusive because one probe could not be evaluated.",
            limitations=[*full.limitations, *ablated.limitations],
            warnings=[*full.warnings, *ablated.warnings, *row_warnings],
            input_references=[parameters.metadata_file],
        )
    full_accuracy = float(full.metrics["probe_accuracy"])
    ablated_accuracy = float(ablated.metrics["probe_accuracy"])
    uncertainty = [
        estimate.model_copy(update={"metric": "full_probe_accuracy"})
        for estimate in full.uncertainty
        if estimate.metric == "probe_accuracy"
    ]
    uncertainty.extend(
        estimate.model_copy(update={"metric": "ablated_probe_accuracy"})
        for estimate in ablated.uncertainty
        if estimate.metric == "probe_accuracy"
    )
    uncertainty.append(UncertaintyEstimate(
        metric="accuracy_delta",
        available=False,
        reason_unavailable="Paired held-out predictions were not retained across both probe fits.",
    ))
    return BackendEvidence(
        outcome="completed",
        metrics={
            "full_probe_accuracy": full_accuracy,
            "ablated_probe_accuracy": ablated_accuracy,
            "accuracy_delta": float(ablated_accuracy - full_accuracy),
            "train_n": int(full.metrics["train_n"]),
            "test_n": int(full.metrics["test_n"]),
            "seed": parameters.seed,
        },
        summary="Out-of-sample probe accuracy was compared before and after removing one explicit metadata feature.",
        details={"ablate_field": parameters.ablate_field, "remaining_features": remaining},
        warnings=[*full.warnings, *ablated.warnings, *row_warnings],
        input_references=[parameters.metadata_file],
        uncertainty=uncertainty,
    )


def _metric_backend(parameters: MetricComputationParameters, access: ExperimentAccess) -> BackendEvidence:
    frame = access.load_csv(
        parameters.predictions_file, source=parameters.predictions_source
    )
    row_warnings = _row_budget_warnings(
        frame,
        max_rows=parameters.max_rows,
        reference=parameters.predictions_file,
    )
    _require_columns(frame, [parameters.label_column, parameters.prediction_column])
    valid = frame[[parameters.label_column, parameters.prediction_column]].dropna()
    if valid.empty:
        return BackendEvidence(
            outcome="inconclusive",
            metrics={"valid_count": 0},
            summary="Metric computation is inconclusive because no valid label/prediction pairs exist.",
            limitations=["No valid prediction rows."],
            warnings=row_warnings,
            input_references=[parameters.predictions_file],
        )
    metric_functions: dict[str, Callable[[Any, Any], float]] = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
    }
    value = float(
        metric_functions[parameters.metric](
            valid[parameters.label_column], valid[parameters.prediction_column]
        )
    )
    labels = valid[parameters.label_column].to_numpy()
    predicted = valid[parameters.prediction_column].to_numpy()
    uncertainty = [bootstrap_interval(
        metric="metric_value",
        sample_count=len(valid),
        statistic=lambda indices: float(metric_functions[parameters.metric](labels[indices], predicted[indices])),
        confidence_level=parameters.confidence_level,
        iterations=parameters.bootstrap_iterations,
        seed=parameters.seed,
        resampling_unit="prediction_row",
    )]
    return BackendEvidence(
        outcome="completed",
        metrics={
            "metric_name": parameters.metric,
            "metric_value": value,
            "valid_count": int(len(valid)),
            "missing_count": int(len(frame) - len(valid)),
        },
        summary="An allowlisted prediction metric was computed on valid bounded rows.",
        warnings=row_warnings,
        input_references=[parameters.predictions_file],
        uncertainty=uncertainty,
    )


BACKEND_REGISTRY: dict[ExperimentType, tuple[type, QuantitativeBackend]] = {
    "statistical_check": (StatisticalCheckParameters, _statistical_backend),
    "subgroup_evaluation": (SubgroupEvaluationParameters, _subgroup_backend),
    "model_evaluation": (PredictiveProbeParameters, _probe_backend),
    "intervention": (MetadataInterventionParameters, _intervention_backend),
    "ablation": (MetadataAblationParameters, _ablation_backend),
    "metric_computation": (MetricComputationParameters, _metric_backend),
    "task_model_training": (TaskModelTrainingParameters, task_model_training_backend),
    "task_model_evaluation": (TaskModelEvaluationParameters, task_model_evaluation_backend),
    "ood_evaluation": (OODEvaluationParameters, ood_evaluation_backend),
    "counterfactual_intervention": (
        CounterfactualInterventionParameters,
        counterfactual_intervention_backend,
    ),
}


def _seed(parameters: Any) -> int | None:
    value = getattr(parameters, "seed", None)
    return value if isinstance(value, int) else None


class RealExperimentExecutor:
    """Execute allowlisted statistical work and persist reusable artifacts."""

    backend_name = "real_cpu_v1"

    @property
    def supported_experiment_types(self) -> tuple[str, ...]:
        """Declare installed local backends; no experiment is executed."""
        return tuple(sorted(BACKEND_REGISTRY))

    def execute(
        self,
        request: ExperimentExecutionRequest,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ExperimentExecutionOutput:
        if context is None:
            raise RealExperimentError("RealExperimentExecutor requires ToolExecutionContext.")
        if context.run_id != request.run_id:
            raise RealExperimentError("Experiment request run_id does not match context.")
        artifact_reference, artifact_path = self._artifact_location(request, context)
        if artifact_path.is_file():
            return self._load_reusable(artifact_path, request)
        if isinstance(request.parameters, EmptyExperimentParameters):
            evidence = BackendEvidence(
                outcome="inconclusive",
                metrics={},
                summary="The approved experiment type requires explicit typed parameters.",
                limitations=["Required experiment parameters were not supplied."],
            )
        else:
            parameter_type, backend = BACKEND_REGISTRY[request.experiment_type]
            if not isinstance(request.parameters, parameter_type):
                raise RealExperimentError("Experiment parameter schema mismatch.")
            max_rows = int(getattr(
                request.parameters,
                "max_rows",
                max(
                    int(getattr(request.parameters, "max_train_samples", 0)),
                    int(getattr(request.parameters, "max_eval_samples", 0)),
                    100_000,
                ),
            ))
            access = ExperimentAccess(context, max_rows=max_rows, request=request)
            evidence = backend(request.parameters, access)
        output = ExperimentExecutionOutput(
            outcome=evidence.outcome,
            metrics=evidence.metrics,
            summary=evidence.summary,
            artifact_references=list(dict.fromkeys([
                artifact_reference, *getattr(evidence, "artifact_references", [])
            ])),
            metadata={
                "backend": evidence.details.get("execution_backend", self.backend_name),
                "experiment_type": request.experiment_type,
                "warnings": evidence.warnings,
                "limitations": evidence.limitations,
                "uncertainty": [
                    item.model_dump(mode="json") for item in evidence.uncertainty
                ],
                "reused": False,
                "details": evidence.details,
            },
        )
        artifact = RealExperimentArtifact(
            schema_version="1.0",
            run_id=request.run_id,
            candidate_id=request.candidate_id,
            task_id=request.task_id,
            experiment_type=request.experiment_type,
            backend=evidence.details.get("execution_backend", self.backend_name),
            parameters=request.parameters.model_dump(mode="json"),
            seed=_seed(request.parameters),
            outcome=output.outcome,
            metrics=output.metrics,
            warnings=evidence.warnings,
            limitations=evidence.limitations,
            created_at=datetime.now(timezone.utc).isoformat(),
            input_references=evidence.input_references,
            details=evidence.details,
            uncertainty=evidence.uncertainty,
            output=output,
        )
        atomic_write_json(
            artifact_path,
            artifact.model_dump(mode="json"),
            error_type=RealExperimentError,
        )
        return output

    def _artifact_location(
        self,
        request: ExperimentExecutionRequest,
        context: ToolExecutionContext,
    ) -> tuple[str, Path]:
        if context.runs_root is None:
            raise RealExperimentError("Real experiments require an approved runs root.")
        run_root = (context.runs_root / context.run_id).resolve(strict=False)
        configured_root = context.runs_root.resolve(strict=False)
        try:
            run_root.relative_to(configured_root)
        except ValueError as exc:
            raise RealExperimentError("Run artifact root escapes approved runs_root.") from exc
        identity: dict[str, Any] = {
            "parameters": request.parameters.model_dump(mode="json")
        }
        if request.experiment_type in {
            "task_model_training", "task_model_evaluation",
            "ood_evaluation", "counterfactual_intervention",
        }:
            identity["bounded_inputs"] = self._model_reliance_input_identity(
                request, context
            )
        encoded = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:12]
        filename = f"{request.experiment_type}-{digest}.json"
        reference = f"experiments/{request.task_id}/{filename}"
        artifact_path = (run_root / reference).resolve(strict=False)
        try:
            artifact_path.relative_to(run_root)
        except ValueError as exc:
            raise RealExperimentError(
                "Experiment artifact path escapes the approved run root."
            ) from exc
        return reference, artifact_path

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _model_reliance_input_identity(
        self,
        request: ExperimentExecutionRequest,
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        parameters = request.parameters
        identity: dict[str, Any] = {}
        dataset_root = context.dataset_root.resolve(strict=True) if context.dataset_root else None
        if dataset_root is None:
            raise RealExperimentError("Model-reliance identity requires dataset_root.")
        metadata_reference = getattr(parameters, "metadata_file", None)
        if metadata_reference:
            metadata_path = ExperimentAccess._beneath(dataset_root, metadata_reference)
            identity["metadata_sha256"] = self._hash_file(metadata_path)
            try:
                # Match ExperimentAccess' hard 100,000-row task-model ceiling so
                # every image that selection could reach participates in resume
                # identity, even when split filtering or ID sorting reorders rows.
                frame = pd.read_csv(metadata_path, nrows=100_001).iloc[:100_000]
                path_column = getattr(parameters, "sample_path_column", None)
                mask_column = getattr(getattr(parameters, "transformation", None), "mask_column", None)
                file_identities = []
                for column in (path_column, mask_column):
                    if column and column in frame:
                        for raw in frame[column].dropna().astype(str):
                            try:
                                path = ExperimentAccess._beneath(dataset_root, raw)
                                file_identities.append((column, raw, self._hash_file(path)))
                            except Exception as exc:
                                file_identities.append((column, raw, f"invalid:{type(exc).__name__}"))
                replacement = getattr(getattr(parameters, "transformation", None), "replacement_sample_reference", None)
                if replacement:
                    path = ExperimentAccess._beneath(dataset_root, replacement)
                    file_identities.append(("replacement", replacement, self._hash_file(path)))
                identity["sample_files"] = file_identities
            except (OSError, UnicodeError, pd.errors.ParserError) as exc:
                raise RealExperimentError("Model-reliance metadata could not be fingerprinted.") from exc
        model_reference = getattr(parameters, "model_reference", None)
        if model_reference:
            if model_reference not in context.approved_artifacts:
                raise RealExperimentError("Model reference is not an approved run artifact.")
            run_root = (context.runs_root / context.run_id).resolve(strict=True)
            model_path = ExperimentAccess._beneath(run_root, model_reference)
            identity["model_sha256"] = self._hash_file(model_path)
            try:
                model_data = json.loads(model_path.read_text(encoding="utf-8"))
                weights_reference = model_data.get("weights_reference")
                if isinstance(weights_reference, str):
                    weights_path = ExperimentAccess._beneath(run_root, weights_reference)
                    identity["weights_sha256"] = self._hash_file(weights_path)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RealExperimentError("Model artifact could not be fingerprinted.") from exc
        return identity

    def _load_reusable(
        self,
        path: Path,
        request: ExperimentExecutionRequest,
    ) -> ExperimentExecutionOutput:
        try:
            artifact = RealExperimentArtifact.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise RealExperimentError("Stored experiment artifact is incompatible.") from exc
        if (
            artifact.run_id != request.run_id
            or artifact.candidate_id != request.candidate_id
            or artifact.task_id != request.task_id
            or artifact.experiment_type != request.experiment_type
            or artifact.parameters != request.parameters.model_dump(mode="json")
        ):
            raise RealExperimentError("Stored experiment artifact does not match request.")
        return artifact.output.model_copy(
            update={
                "metadata": {
                    **artifact.output.metadata,
                    "reused": True,
                }
            }
        )


__all__ = [
    "BACKEND_REGISTRY",
    "ExperimentDataError",
    "RealExperimentError",
    "RealExperimentExecutor",
    "RealExperimentArtifact",
]
