import ast
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from shortcutminer.experiment.real_executor import (
    BACKEND_REGISTRY,
    RealExperimentExecutor,
)
from shortcutminer.experiment.tool_executor import ExperimentExecutionRequest
from shortcutminer.protocol.experiment import PredictiveProbeParameters
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolRuntime


BACKEND_PATH = Path("shortcutminer/experiment/real_executor.py")


def test_registry_is_static_complete_and_has_no_dynamic_imports_or_processes():
    assert set(BACKEND_REGISTRY) == {
        "statistical_check",
        "subgroup_evaluation",
        "model_evaluation",
        "intervention",
        "ablation",
        "metric_computation",
    }
    tree = ast.parse(BACKEND_PATH.read_text(encoding="utf-8"))
    forbidden_imports = {"subprocess", "importlib"}
    forbidden_calls = {"eval", "exec", "compile", "__import__"}
    assert not {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    } & forbidden_imports
    assert not {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } & forbidden_imports
    assert not {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } & forbidden_calls


def test_backend_has_no_threshold_based_shortcut_judgment():
    tree = ast.parse(BACKEND_PATH.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        identifiers = {
            child.id.casefold()
            for child in ast.walk(node.test)
            if isinstance(child, ast.Name)
        }
        attributes = {
            child.attr.casefold()
            for child in ast.walk(node.test)
            if isinstance(child, ast.Attribute)
        }
        decision_names = identifiers | attributes
        if decision_names & {
            "shortcut",
            "is_shortcut",
            "supports",
            "evidence_direction",
            "confirmed",
        }:
            violations.append(node.lineno)
    assert violations == []


@pytest.mark.parametrize("field", ["python", "command", "script"])
def test_arbitrary_code_parameters_are_forbidden(field: str):
    with pytest.raises(ValidationError):
        ExperimentExecutionRequest.model_validate(
            {
                "run_id": "security-run",
                "candidate_id": "C001",
                "task_id": "TASK-C001-QT",
                "experiment_type": "statistical_check",
                "parameters": {field: "do something"},
            }
        )


def test_budget_and_metric_allowlists_are_schema_enforced():
    with pytest.raises(ValidationError):
        ExperimentExecutionRequest.model_validate(
            {
                "run_id": "security-run",
                "candidate_id": "C001",
                "task_id": "TASK-C001-QT",
                "experiment_type": "statistical_check",
                "parameters": {
                    "metadata_file": "metadata.csv",
                    "cue_field": "cue",
                    "label_column": "label",
                    "n_permutations": 1001,
                },
            }
        )
    with pytest.raises(ValidationError):
        ExperimentExecutionRequest.model_validate(
            {
                "run_id": "security-run",
                "candidate_id": "C001",
                "task_id": "TASK-C001-QT",
                "experiment_type": "metric_computation",
                "parameters": {
                    "predictions_file": "predictions.csv",
                    "label_column": "label",
                    "prediction_column": "prediction",
                    "metric": "eval_user_metric",
                },
            }
        )
    with pytest.raises(ValidationError):
        PredictiveProbeParameters(
            metadata_file="metadata.csv",
            feature_fields=[f"f{i}" for i in range(3)],
            label_column="label",
            max_probe_features=2,
        )


def test_candidate_name_is_not_fuzzily_resolved_to_a_metadata_field(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    pd.DataFrame(
        {"device_model": ["a", "b"] * 5, "label": [0, 1] * 5}
    ).to_csv(dataset_root / "metadata.csv", index=False)
    runs_root = tmp_path / "runs"
    (runs_root / "security-run").mkdir(parents=True)
    runtime = ToolRuntime(
        build_default_tool_registry(experiment_executor=RealExperimentExecutor()),
        run_id="security-run",
        agent_name="quantitative",
        dataset_root=dataset_root,
        runs_root=runs_root,
    )
    result = runtime.execute(
        {
            "tool_call_id": "missing-field",
            "tool_name": "run_experiment",
            "agent_name": "quantitative",
            "run_id": "security-run",
            "candidate_id": "C001",
            "task_id": "TASK-C001-QT",
            "arguments": {
                "candidate_id": "C001",
                "task_id": "TASK-C001-QT",
                "experiment_type": "statistical_check",
                "parameters": {
                    "metadata_file": "metadata.csv",
                    "cue_field": "scanner characteristics",
                    "label_column": "label",
                },
            },
        }
    )
    assert result.status == "failed"
    assert "scanner characteristics" in result.error.message


def test_run_prediction_inputs_must_be_explicitly_approved(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    pd.DataFrame({"label": [0, 1]}).to_csv(
        dataset_root / "metadata.csv", index=False
    )
    runs_root = tmp_path / "runs"
    run_root = runs_root / "security-run"
    run_root.mkdir(parents=True)
    pd.DataFrame({"label": [0, 1], "prediction": [0, 1]}).to_csv(
        run_root / "predictions.csv", index=False
    )
    runtime = ToolRuntime(
        build_default_tool_registry(experiment_executor=RealExperimentExecutor()),
        run_id="security-run",
        agent_name="quantitative",
        dataset_root=dataset_root,
        runs_root=runs_root,
    )
    result = runtime.execute(
        {
            "tool_call_id": "unapproved-predictions",
            "tool_name": "run_experiment",
            "agent_name": "quantitative",
            "run_id": "security-run",
            "candidate_id": "C001",
            "task_id": "TASK-C001-QT",
            "arguments": {
                "candidate_id": "C001",
                "task_id": "TASK-C001-QT",
                "experiment_type": "metric_computation",
                "parameters": {
                    "predictions_file": "predictions.csv",
                    "predictions_source": "run",
                    "label_column": "label",
                    "prediction_column": "prediction",
                },
            },
        }
    )
    assert result.status == "failed"
    assert "approved artifact" in result.error.message


def test_experiment_artifact_symlink_escape_is_rejected(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    pd.DataFrame({"cue": ["a", "b"] * 3, "label": [0, 1] * 3}).to_csv(
        dataset_root / "metadata.csv", index=False
    )
    runs_root = tmp_path / "runs"
    run_root = runs_root / "security-run"
    run_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (run_root / "experiments").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")
    runtime = ToolRuntime(
        build_default_tool_registry(experiment_executor=RealExperimentExecutor()),
        run_id="security-run",
        agent_name="quantitative",
        dataset_root=dataset_root,
        runs_root=runs_root,
    )
    result = runtime.execute(
        {
            "tool_call_id": "artifact-escape",
            "tool_name": "run_experiment",
            "agent_name": "quantitative",
            "run_id": "security-run",
            "candidate_id": "C001",
            "task_id": "TASK-C001-QT",
            "arguments": {
                "candidate_id": "C001",
                "task_id": "TASK-C001-QT",
                "experiment_type": "statistical_check",
                "parameters": {
                    "metadata_file": "metadata.csv",
                    "cue_field": "cue",
                    "label_column": "label",
                },
            },
        }
    )
    assert result.status == "failed"
    assert "artifact path escapes" in result.error.message
