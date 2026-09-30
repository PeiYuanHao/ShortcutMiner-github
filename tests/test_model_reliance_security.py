import ast
from pathlib import Path

from PIL import Image
import pytest
from pydantic import ValidationError

from shortcutminer.experiment.tool_executor import ExperimentExecutionRequest
from shortcutminer.task_models.prediction_artifact import PredictionArtifactAdapter
from shortcutminer.task_models.reference import TaskModelError
from shortcutminer.task_models.registry import TASK_MODEL_REGISTRY
from test_model_reliance import (
    execute,
    evaluation_parameters,
    make_shape_dataset,
    runtime_for,
    train,
    training_parameters,
)


SOURCE_PATHS = [
    Path("shortcutminer/experiment/model_reliance.py"),
    Path("shortcutminer/intervention/engine.py"),
    Path("shortcutminer/intervention/segmentation.py"),
    Path("shortcutminer/task_models/reference.py"),
    Path("shortcutminer/task_models/registry.py"),
]


def test_model_and_intervention_sources_have_no_dynamic_execution_or_downloads():
    forbidden_imports = {"subprocess", "importlib", "requests", "torch"}
    forbidden_calls = {"eval", "exec", "compile", "__import__"}
    for path in SOURCE_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            node.names[0].name.split(".")[0]
            for node in ast.walk(tree) if isinstance(node, ast.Import)
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
        }
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not imports & forbidden_imports
        assert not calls & forbidden_calls
        assert "pickle.load" not in path.read_text(encoding="utf-8")
    assert set(TASK_MODEL_REGISTRY) == {"reference_image_linear"}


def test_model_reliance_backend_has_no_magic_verdict_thresholds():
    source = Path("shortcutminer/experiment/model_reliance.py").read_text(encoding="utf-8")
    assert "shortcut_present" not in source
    assert "model_relies" not in source
    assert 'metrics["supports"]' not in source
    assert "causal_validity" not in source


@pytest.mark.parametrize(
    "field",
    ["python_file", "module_path", "entrypoint", "command", "shell", "checkpoint_url"],
)
def test_arbitrary_model_configuration_is_rejected(field: str):
    parameters = training_parameters()
    parameters[field] = "malicious"
    with pytest.raises(ValidationError):
        ExperimentExecutionRequest.model_validate({
            "run_id": "phase15-run", "candidate_id": "C001",
            "task_id": "TASK-C001-QT", "experiment_type": "task_model_training",
            "parameters": parameters,
        })


def test_prediction_only_adapter_cannot_train_or_infer_interventions():
    adapter = PredictionArtifactAdapter("static", {})
    assert adapter.capabilities.predictions
    assert not adapter.capabilities.training
    assert not adapter.capabilities.checkpoint
    assert not adapter.capabilities.intervention_inference
    with pytest.raises(TaskModelError, match="cannot be retrained"):
        adapter.fit([], [], seed=0, max_iterations=10)


def test_corrupted_safe_weights_fail_without_retraining(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path)
    runtime, runs = runtime_for(tmp_path, dataset)
    training, model_reference = train(runtime)
    weights_reference = next(
        ref for ref in training.output["artifact_references"] if ref.endswith("weights.npz")
    )
    (runs / "phase15-run" / weights_reference).write_bytes(b"not-an-npz")
    result = execute(runtime, "task_model_evaluation", evaluation_parameters(model_reference), "corrupt-model")
    assert result.status == "failed"
    assert "weights hash" in result.error.message or "fingerprint" in result.error.message


def test_changed_training_image_cannot_reuse_old_model_identity(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path)
    runtime, _ = runtime_for(tmp_path, dataset)
    train(runtime)
    Image.new("RGB", (24, 24), (0, 0, 0)).save(dataset / "images/s000.png")
    result = execute(runtime, "task_model_training", training_parameters(), "changed-input")
    assert result.status == "failed"
    assert "different training identity" in result.error.message


def test_model_artifact_directory_symlink_escape_is_rejected(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path)
    runtime, runs = runtime_for(tmp_path, dataset)
    outside = tmp_path / "outside"
    outside.mkdir()
    models = runs / "phase15-run" / "models"
    try:
        models.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")
    result = execute(runtime, "task_model_training", training_parameters(), "escape-model")
    assert result.status == "failed"
    assert "escapes" in result.error.message
