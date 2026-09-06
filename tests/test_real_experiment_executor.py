from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from shortcutminer.experiment.real_executor import RealExperimentExecutor
from shortcutminer.experiment.tool_executor import ExperimentExecutionRequest
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolRuntime


def _runtime(tmp_path: Path, dataset: pd.DataFrame) -> tuple[ToolRuntime, Path]:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_root / "metadata.csv", index=False)
    runs_root = tmp_path / "runs"
    (runs_root / "real-experiment-run").mkdir(parents=True, exist_ok=True)
    registry = build_default_tool_registry(
        experiment_executor=RealExperimentExecutor()
    )
    return (
        ToolRuntime(
            registry,
            run_id="real-experiment-run",
            agent_name="quantitative",
            dataset_root=dataset_root,
            runs_root=runs_root,
        ),
        runs_root,
    )


def _execute(runtime: ToolRuntime, experiment_type: str, parameters: dict, call_id="call-1"):
    return runtime.execute(
        {
            "protocol_version": "1.0",
            "tool_call_id": call_id,
            "agent_name": "quantitative",
            "run_id": "real-experiment-run",
            "tool_name": "run_experiment",
            "candidate_id": "C001",
            "task_id": "TASK-C001-QT",
            "arguments": {
                "candidate_id": "C001",
                "task_id": "TASK-C001-QT",
                "experiment_type": experiment_type,
                "parameters": parameters,
            },
        }
    )


def _association_parameters(**updates):
    value = {
        "metadata_file": "metadata.csv",
        "cue_field": "cue",
        "label_column": "label",
        "max_rows": 1000,
        "max_categories": 20,
        "min_group_size": 2,
        "n_permutations": 0,
        "seed": 11,
    }
    value.update(updates)
    return value


def test_strong_and_independent_categorical_association(tmp_path: Path):
    strong = pd.DataFrame(
        {"cue": ["a"] * 50 + ["b"] * 50, "label": [0] * 50 + [1] * 50}
    )
    strong_runtime, _ = _runtime(tmp_path / "strong", strong)
    strong_result = _execute(
        strong_runtime, "statistical_check", _association_parameters()
    )
    assert strong_result.status == "success"
    assert strong_result.output["outcome"] == "completed"
    assert strong_result.output["metrics"]["effect_size_name"] == "cramers_v"
    assert strong_result.output["metrics"]["effect_size"] > 0.95
    assert strong_result.output["metrics"]["p_value"] is not None

    independent = pd.DataFrame(
        {
            "cue": ["a", "a", "b", "b"] * 25,
            "label": [0, 1, 0, 1] * 25,
        }
    )
    independent_runtime, _ = _runtime(tmp_path / "independent", independent)
    independent_result = _execute(
        independent_runtime, "statistical_check", _association_parameters()
    )
    assert independent_result.output["metrics"]["effect_size"] < 0.01

    partial = pd.DataFrame(
        {
            "cue": ["a"] * 40 + ["b"] * 40,
            "label": [0] * 30 + [1] * 10 + [0] * 10 + [1] * 30,
        }
    )
    partial_runtime, _ = _runtime(tmp_path / "partial", partial)
    partial_result = _execute(
        partial_runtime,
        "statistical_check",
        _association_parameters(n_permutations=25),
    )
    assert 0.2 < partial_result.output["metrics"]["effect_size"] < 0.8
    assert partial_result.output["metrics"]["n_permutations"] == 25
    assert 0.0 < partial_result.output["metrics"]["permutation_p_value"] <= 1.0


def test_numeric_association_modes_and_missing_values(tmp_path: Path):
    numeric_group = pd.DataFrame(
        {
            "cue": [*range(20), *range(100, 120), np.nan],
            "label": [0] * 20 + [1] * 20 + [1],
        }
    )
    runtime, _ = _runtime(tmp_path / "groups", numeric_group)
    grouped = _execute(
        runtime,
        "statistical_check",
        _association_parameters(variable_mode="numeric_categorical"),
    )
    assert grouped.output["metrics"]["effect_size_name"] == "eta_squared"
    assert grouped.output["metrics"]["effect_size"] > 0.8
    assert grouped.output["metrics"]["missing_count"] == 1

    numeric = pd.DataFrame({"cue": range(40), "label": [value * 3 for value in range(40)]})
    numeric_runtime, _ = _runtime(tmp_path / "numeric", numeric)
    correlated = _execute(
        numeric_runtime,
        "statistical_check",
        _association_parameters(variable_mode="numeric_numeric"),
    )
    assert correlated.output["metrics"]["effect_size_name"] == "pearson_r"
    assert correlated.output["metrics"]["effect_size"] > 0.99


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"cue": ["a"] * 20, "label": [0, 1] * 10}),
        pd.DataFrame({"cue": ["a", "b"], "label": [0, 1]}),
        pd.DataFrame({"cue": ["a", "b", "c", "d"], "label": [1] * 4}),
    ],
)
def test_constant_tiny_and_single_class_are_scientifically_inconclusive(
    tmp_path: Path, frame: pd.DataFrame
):
    runtime, _ = _runtime(tmp_path, frame)
    result = _execute(runtime, "statistical_check", _association_parameters())
    assert result.status == "success"
    assert result.output["outcome"] == "inconclusive"
    assert result.output["metadata"]["limitations"]


def test_high_cardinality_is_bounded_and_traceable(tmp_path: Path):
    frame = pd.DataFrame(
        {"cue": [f"g-{i}" for i in range(80)], "label": [i % 2 for i in range(80)]}
    )
    runtime, _ = _runtime(tmp_path, frame)
    result = _execute(
        runtime,
        "statistical_check",
        _association_parameters(max_categories=5, min_group_size=1),
    )
    assert result.output["metrics"]["cue_category_count"] == 5
    assert any("max_categories=5" in item for item in result.output["metadata"]["warnings"])


def test_row_budget_truncation_is_traceable(tmp_path: Path):
    frame = pd.DataFrame(
        {"cue": ["a", "b"] * 10, "label": [0, 1] * 10}
    )
    runtime, _ = _runtime(tmp_path, frame)
    result = _execute(
        runtime,
        "statistical_check",
        _association_parameters(max_rows=6, min_group_size=1),
    )
    assert result.output["metrics"]["sample_count"] == 6
    assert any(
        "truncated at max_rows=6" in item
        for item in result.output["metadata"]["warnings"]
    )


def test_subgroup_prediction_accuracy_and_label_only_mode(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "group": ["a"] * 10 + ["b"] * 10,
            "label": [0, 1] * 10,
            "prediction": [0, 1] * 5 + [0] * 10,
        }
    )
    runtime, _ = _runtime(tmp_path / "prediction", frame)
    result = _execute(
        runtime,
        "subgroup_evaluation",
        {
            "metadata_file": "metadata.csv",
            "group_field": "group",
            "label_column": "label",
            "predictions_file": "metadata.csv",
            "prediction_column": "prediction",
            "max_rows": 100,
        },
    )
    metrics = result.output["metrics"]
    assert metrics["overall_accuracy"] == pytest.approx(0.75)
    assert metrics["worst_group_accuracy"] == pytest.approx(0.5)
    assert metrics["best_group_accuracy"] == pytest.approx(1.0)
    assert metrics["group_accuracy_gap"] == pytest.approx(0.5)

    label_runtime, _ = _runtime(tmp_path / "labels", frame.drop(columns="prediction"))
    label_only = _execute(
        label_runtime,
        "subgroup_evaluation",
        {
            "metadata_file": "metadata.csv",
            "group_field": "group",
            "label_column": "label",
            "max_rows": 100,
        },
    )
    assert label_only.output["metrics"]["model_performance"] is False
    assert "not a model performance audit" in label_only.output["metadata"]["limitations"][0]


def test_subgroup_separate_predictions_reject_duplicates_and_count_missing(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "group": ["a", "a", "b", "b"],
            "label": [0, 1, 0, 1],
        }
    )
    runtime, _ = _runtime(tmp_path / "missing", frame)
    dataset_root = tmp_path / "missing" / "dataset"
    pd.DataFrame(
        {
            "prediction_id": ["s1", "s2", "s3"],
            "prediction": [0, 1, 1],
        }
    ).to_csv(dataset_root / "predictions.csv", index=False)
    parameters = {
        "metadata_file": "metadata.csv",
        "group_field": "group",
        "label_column": "label",
        "predictions_file": "predictions.csv",
        "sample_id_column": "sample_id",
        "prediction_sample_id_column": "prediction_id",
        "prediction_column": "prediction",
        "max_rows": 100,
        "min_group_size": 1,
    }
    missing = _execute(runtime, "subgroup_evaluation", parameters)
    assert missing.status == "success"
    assert missing.output["outcome"] == "completed"
    assert missing.output["metrics"]["sample_count"] == 3
    assert missing.output["metrics"]["missing_prediction_count"] == 1

    duplicate_runtime, _ = _runtime(tmp_path / "duplicate", frame)
    pd.DataFrame(
        {
            "prediction_id": ["s1", "s1", "s2", "s3"],
            "prediction": [0, 1, 1, 0],
        }
    ).to_csv(tmp_path / "duplicate" / "dataset" / "predictions.csv", index=False)
    duplicate = _execute(duplicate_runtime, "subgroup_evaluation", parameters)
    assert duplicate.status == "failed"
    assert "Duplicate sample IDs" in duplicate.error.message


def _probe_frame(strong: bool) -> pd.DataFrame:
    labels = np.array([0, 1] * 100)
    raw_cue = labels.copy() if strong else np.array(([0, 0, 1, 1] * 50))
    cue = np.where(raw_cue == 0, "x", "y")
    return pd.DataFrame({"cue": cue, "label": labels})


def _probe_parameters(features=None, **updates):
    value = {
        "metadata_file": "metadata.csv",
        "feature_fields": features or ["cue"],
        "label_column": "label",
        "test_fraction": 0.25,
        "max_rows": 1000,
        "max_probe_features": 5,
        "max_probe_iterations": 200,
        "seed": 23,
    }
    value.update(updates)
    return value


def test_probe_is_out_of_sample_deterministic_and_has_majority_baseline(tmp_path: Path):
    strong_runtime, _ = _runtime(tmp_path / "strong", _probe_frame(True))
    first = _execute(strong_runtime, "model_evaluation", _probe_parameters())
    repeat_runtime, _ = _runtime(tmp_path / "repeat", _probe_frame(True))
    second = _execute(repeat_runtime, "model_evaluation", _probe_parameters())
    assert first.status == "success", first.error
    assert second.status == "success", second.error
    assert first.output["metrics"] == second.output["metrics"]
    assert first.output["metrics"]["probe_accuracy"] > 0.95
    assert first.output["metrics"]["improvement"] > 0.4

    independent_runtime, _ = _runtime(tmp_path / "independent", _probe_frame(False))
    independent = _execute(
        independent_runtime, "model_evaluation", _probe_parameters()
    )
    assert abs(independent.output["metrics"]["improvement"]) < 0.2


def test_probe_rejects_label_leakage_and_arbitrary_code_parameters():
    with pytest.raises(ValidationError, match="label_column"):
        ExperimentExecutionRequest.model_validate(
            {
                "run_id": "real-experiment-run",
                "candidate_id": "C001",
                "task_id": "TASK-C001-QT",
                "experiment_type": "model_evaluation",
                "parameters": _probe_parameters(features=["label"]),
            }
        )
    with pytest.raises(ValidationError):
        ExperimentExecutionRequest.model_validate(
            {
                "run_id": "real-experiment-run",
                "candidate_id": "C001",
                "task_id": "TASK-C001-QT",
                "experiment_type": "statistical_check",
                "parameters": {"python": "import os"},
            }
        )


def test_shuffle_intervention_changes_effect_and_is_seeded(tmp_path: Path):
    frame = _probe_frame(True)
    parameters = {
        "metadata_file": "metadata.csv",
        "cue_field": "cue",
        "label_column": "label",
        "intervention_type": "shuffle",
        "max_rows": 1000,
        "max_categories": 20,
        "min_group_size": 2,
        "seed": 7,
    }
    runtime, _ = _runtime(tmp_path / "one", frame)
    first = _execute(runtime, "intervention", parameters)
    repeat_runtime, _ = _runtime(tmp_path / "two", frame)
    repeat = _execute(repeat_runtime, "intervention", parameters)
    assert first.output["metrics"] == repeat.output["metrics"]
    assert first.output["metrics"]["before_effect_size"] > 0.95
    assert first.output["metrics"]["after_effect_size"] < 0.3

    different_runtime, _ = _runtime(tmp_path / "three", frame)
    different = _execute(
        different_runtime, "intervention", {**parameters, "seed": 8}
    )
    assert different.output["metrics"]["rows_affected"] != first.output["metrics"]["rows_affected"]


def test_metadata_ablation_and_metric_allowlist(tmp_path: Path):
    frame = _probe_frame(True)
    frame["noise"] = ([0, 0, 1, 1] * 50)
    runtime, _ = _runtime(tmp_path / "ablation", frame)
    ablation = _execute(
        runtime,
        "ablation",
        {
            **_probe_parameters(features=["cue", "noise"]),
            "ablate_field": "cue",
        },
    )
    assert ablation.status == "success", ablation.error
    assert ablation.output["metrics"]["full_probe_accuracy"] > 0.95
    assert ablation.output["metrics"]["ablated_probe_accuracy"] < 0.7

    metric_frame = pd.DataFrame({"label": [0, 1, 0, 1], "prediction": [0, 1, 1, 1]})
    metric_runtime, _ = _runtime(tmp_path / "metric", metric_frame)
    metric = _execute(
        metric_runtime,
        "metric_computation",
        {
            "predictions_file": "metadata.csv",
            "label_column": "label",
            "prediction_column": "prediction",
            "metric": "accuracy",
            "max_rows": 20,
        },
    )
    assert metric.output["metrics"]["metric_value"] == pytest.approx(0.75)


def test_artifact_is_machine_readable_and_reused(tmp_path: Path):
    runtime, runs_root = _runtime(tmp_path, _probe_frame(True))
    first = _execute(runtime, "statistical_check", _association_parameters())
    reference = first.output["artifact_references"][0]
    artifact = runs_root / "real-experiment-run" / reference
    assert artifact.is_file()
    original_mtime = artifact.stat().st_mtime_ns
    readable = runtime.execute(
        {
            "tool_call_id": "read-generated-artifact",
            "tool_name": "read_artifact",
            "agent_name": "quantitative",
            "run_id": "real-experiment-run",
            "candidate_id": "C001",
            "task_id": "TASK-C001-QT",
            "arguments": {"artifact": reference},
        }
    )
    assert readable.status == "success"
    assert readable.output["content"]["backend"] == "real_cpu_v1"

    second_runtime, _ = _runtime(tmp_path, _probe_frame(True))
    second = _execute(
        second_runtime,
        "statistical_check",
        _association_parameters(),
        call_id="call-2",
    )
    assert second.output["metadata"]["reused"] is True
    assert artifact.stat().st_mtime_ns == original_mtime


def test_parse_failure_is_execution_failure_not_inconclusive(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "metadata.csv").write_text(
        'cue,label\na,"unterminated\n', encoding="utf-8"
    )
    runs_root = tmp_path / "runs"
    (runs_root / "real-experiment-run").mkdir(parents=True)
    runtime = ToolRuntime(
        build_default_tool_registry(experiment_executor=RealExperimentExecutor()),
        run_id="real-experiment-run",
        agent_name="quantitative",
        dataset_root=dataset_root,
        runs_root=runs_root,
    )
    result = _execute(runtime, "statistical_check", _association_parameters())
    assert result.status == "failed"
    assert result.error.error_type == "ToolExecutionError"
