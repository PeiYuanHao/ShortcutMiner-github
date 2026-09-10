import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import pytest

from shortcutminer.evaluation.blind.downstream import downstream_impact_from_output
from shortcutminer.experiment.real_executor import RealExperimentExecutor
from shortcutminer.task_models.reference import ReferenceImageLinearClassifier, TaskModelError
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolBudget, ToolRuntime


def _shape_image(path: Path, *, label: int, background: tuple[int, int, int], patch=None):
    image = Image.new("RGB", (24, 24), background)
    draw = ImageDraw.Draw(image)
    if label == 0:
        draw.ellipse((8, 8, 15, 15), fill=(245, 245, 245))
    else:
        draw.rectangle((8, 8, 15, 15), fill=(245, 245, 245))
    if patch is not None:
        draw.rectangle((0, 0, 5, 5), fill=patch)
    image.save(path)


def make_shape_dataset(root: Path, *, cue: str = "color", clean: bool = False, count=60):
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    rows = []
    for index in range(count):
        label = index % 2
        split = "train" if index < 40 else "id"
        if cue == "color":
            if clean:
                background = (220, 35, 35) if (index // 2) % 2 == 0 else (35, 35, 220)
            else:
                background = (220, 35, 35) if label == 0 else (35, 35, 220)
            patch = None
        elif cue == "spatial":
            background = (120, 120, 120)
            patch = (220, 35, 35) if label == 0 else (35, 35, 220)
        else:
            base = 55 if label == 0 else 195
            background = (base, base, base)
            patch = None
        reference = f"images/s{index:03d}.png"
        path = dataset / reference
        path.parent.mkdir(exist_ok=True)
        _shape_image(path, label=label, background=background, patch=patch)
        mask_reference = f"masks/s{index:03d}.png"
        mask_path = dataset / mask_reference
        mask_path.parent.mkdir(exist_ok=True)
        mask = Image.new("L", (24, 24), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle((7, 7, 16, 16), fill=255)
        mask.save(mask_path)
        rows.append({
            "sample_id": f"s{index:03d}", "path": reference, "label": label,
            "split": split, "mask": mask_reference,
        })
    pd.DataFrame(rows).to_csv(dataset / "metadata.csv", index=False)
    return dataset


def runtime_for(root: Path, dataset: Path, *, constraints=(), capabilities=None):
    runs = root / "runs"
    (runs / "phase15-run").mkdir(parents=True)
    return ToolRuntime(
        build_default_tool_registry(experiment_executor=RealExperimentExecutor()),
        run_id="phase15-run", agent_name="quantitative", dataset_root=dataset,
        runs_root=runs, runtime_constraints=set(constraints),
        model_capabilities=set(capabilities or {
            "predictions", "logits", "checkpoint", "retraining", "intervention_inference"
        }),
        budget=ToolBudget(max_tool_calls=20, timeout_seconds=30),
    ), runs


def execute(runtime, kind, parameters, call_id):
    return runtime.execute({
        "tool_call_id": call_id, "tool_name": "run_experiment",
        "agent_name": "quantitative", "run_id": "phase15-run",
        "candidate_id": "C001", "task_id": "TASK-C001-QT",
        "arguments": {
            "candidate_id": "C001", "task_id": "TASK-C001-QT",
            "experiment_type": kind, "parameters": parameters,
        },
    })


def training_parameters(model_id="reference-color", **updates):
    values = {
        "metadata_file": "metadata.csv", "sample_id_column": "sample_id",
        "sample_path_column": "path", "label_column": "label", "split_column": "split",
        "train_split_values": ["train"], "validation_split_values": [],
        "test_split_values": ["id"], "model_id": model_id,
        "model_backend": "reference_image_linear",
        "feature_version": "reference_image_features_v1", "max_train_samples": 100,
        "max_eval_samples": 100, "max_model_iterations": 200, "seed": 17,
        "class_weight": "none", "artifact_policy": "safe_json_npz",
        "bootstrap_iterations": 30,
    }
    values.update(updates)
    return values


def evaluation_parameters(model_reference, **updates):
    values = {
        "model_reference": model_reference, "metadata_file": "metadata.csv",
        "sample_id_column": "sample_id", "sample_path_column": "path",
        "label_column": "label", "split_column": "split", "split_values": ["id"],
        "max_eval_samples": 100, "seed": 17, "bootstrap_iterations": 30,
    }
    values.update(updates)
    return values


def train(runtime, *, model_id="reference-color", call_id="train"):
    result = execute(runtime, "task_model_training", training_parameters(model_id), call_id)
    assert result.status == "success", result.error
    assert result.output["outcome"] == "completed"
    model_reference = next(ref for ref in result.output["artifact_references"] if ref.endswith("/model.json"))
    return result, model_reference


def test_reference_training_is_real_deterministic_safe_and_reloadable(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path / "first")
    runtime, runs = runtime_for(tmp_path / "first", dataset)
    result, reference = train(runtime)
    assert result.output["metrics"]["train_count"] == 40
    assert result.output["metrics"]["test_count"] == 20
    assert result.output["metrics"]["training_accuracy"] > 0.95
    model_path = runs / "phase15-run" / reference
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    assert payload["feature_version"] == "reference_image_features_v1"
    assert payload["feature_specification"]["resize"] == [16, 16]
    assert payload["weights_reference"].endswith("weights.npz")
    assert not any(path.suffix in {".pkl", ".pickle", ".joblib", ".pt"} for path in model_path.parent.iterdir())
    model, artifact = ReferenceImageLinearClassifier.load_safe_artifact(
        model_path, run_root=runs / "phase15-run", expected_run_id="phase15-run"
    )
    assert artifact.model_hash == result.output["metadata"]["details"]["model_hash"]
    assert len(model.predict([dataset / "images/s040.png"])) == 1

    dataset2 = make_shape_dataset(tmp_path / "second")
    runtime2, _ = runtime_for(tmp_path / "second", dataset2)
    result2, _ = train(runtime2)
    assert result2.output["metadata"]["details"]["model_hash"] == artifact.model_hash


def test_training_resume_does_not_fit_again(tmp_path: Path, monkeypatch):
    dataset = make_shape_dataset(tmp_path)
    runtime, _ = runtime_for(tmp_path, dataset)
    first, _ = train(runtime)
    monkeypatch.setattr(
        ReferenceImageLinearClassifier,
        "fit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fit repeated")),
    )
    second = execute(runtime, "task_model_training", training_parameters(), "train-again")
    assert second.status == "success"
    assert second.output["metadata"]["reused"] is True
    assert first.output["artifact_references"] == second.output["artifact_references"]


def test_training_without_split_uses_deterministic_stratified_holdout(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path)
    frame = pd.read_csv(dataset / "metadata.csv").drop(columns="split")
    frame.to_csv(dataset / "metadata.csv", index=False)
    runtime, runs = runtime_for(tmp_path, dataset)
    params = training_parameters(
        model_id="generated-split", split_column=None, train_split_values=[], test_split_values=[]
    )
    result = execute(runtime, "task_model_training", params, "generated")
    assert result.status == "success"
    manifest_ref = next(ref for ref in result.output["artifact_references"] if ref.endswith("training_manifest.json"))
    manifest = json.loads((runs / "phase15-run" / manifest_ref).read_text(encoding="utf-8"))
    assert manifest["split_generated_by_runtime"] is True
    assert set(manifest["train_sample_ids"]).isdisjoint(manifest["test_sample_ids"])


@pytest.mark.parametrize("mode", ["few", "single", "corrupt"])
def test_training_failure_semantics_are_inconclusive(tmp_path: Path, mode: str):
    dataset = make_shape_dataset(tmp_path, count=8)
    frame = pd.read_csv(dataset / "metadata.csv")
    frame.loc[:5, "split"] = "train"
    frame.loc[6:, "split"] = "id"
    if mode == "few":
        frame = frame.iloc[:3]
    elif mode == "single":
        frame["label"] = 0
    else:
        for reference in frame["path"]:
            (dataset / reference).write_bytes(b"corrupt")
    frame.to_csv(dataset / "metadata.csv", index=False)
    runtime, _ = runtime_for(tmp_path, dataset)
    result = execute(runtime, "task_model_training", training_parameters(), "failure")
    assert result.status == "success"
    assert result.output["outcome"] == "inconclusive"


def test_task_model_id_and_ood_counterfactual_use_same_model(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path)
    runtime, _ = runtime_for(tmp_path, dataset)
    training, model_reference = train(runtime)
    model_hash = training.output["metadata"]["details"]["model_hash"]
    evaluation = execute(runtime, "task_model_evaluation", evaluation_parameters(model_reference), "id-eval")
    assert evaluation.status == "success"
    assert evaluation.output["metrics"]["accuracy"] > 0.95
    assert evaluation.output["metrics"]["model_hash"] == model_hash

    base = evaluation_parameters(model_reference)
    ood = execute(runtime, "ood_evaluation", {
        **base, "transformation": {"type": "global_color", "mode": "swap_red_blue"},
        "target_cue": "background color", "comparison_mode": "paired",
        "preservation_status": "guaranteed_by_construction", "max_intervention_samples": 100,
    }, "ood")
    assert ood.status == "success", ood.error
    assert ood.output["metrics"]["model_hash"] == model_hash
    assert ood.output["metrics"]["ood_id"].startswith("OOD-")
    assert ood.output["metrics"]["id_accuracy"] > ood.output["metrics"]["ood_accuracy"]
    assert ood.output["metrics"]["accuracy_delta"] < 0
    assert ood.output["metadata"]["uncertainty"][0]["resampling_unit"] == "counterfactual_pair"
    assert "supports" not in json.dumps(ood.output).casefold()

    counterfactual = execute(runtime, "counterfactual_intervention", {
        **base, "transformation": {"type": "global_color", "mode": "swap_red_blue"},
        "target_cue": "background color",
        "preservation_status": "guaranteed_by_construction", "max_intervention_samples": 100,
    }, "counterfactual")
    metrics = counterfactual.output["metrics"]
    assert counterfactual.status == "success"
    assert metrics["model_hash"] == model_hash
    assert metrics["pair_count"] == 20
    assert metrics["prediction_flip_rate"] > 0.9
    assert metrics["true_class_confidence_delta"] < 0
    assert metrics["true_class_logit_delta"] < 0
    assert metrics["logit_margin_delta"] < 0
    impact = downstream_impact_from_output(counterfactual.output)
    assert impact.available and impact.metric_before == metrics["original_accuracy"]
    assert impact.model_hash == model_hash
    assert impact.intervention_id == metrics["intervention_id"]
    assert impact.comparison_mode == "paired"
    assert impact.pair_coverage == 1
    assert impact.uncertainty[0]["resampling_unit"] == "counterfactual_pair"


def test_independent_ood_records_independent_bootstrap(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path)
    runtime, _ = runtime_for(tmp_path, dataset)
    _, model_reference = train(runtime)
    result = execute(runtime, "ood_evaluation", {
        **evaluation_parameters(model_reference),
        "transformation": {"type": "brightness", "factor": 0.6},
        "target_cue": "brightness", "comparison_mode": "independent",
        "preservation_status": "unverified", "max_intervention_samples": 10,
    }, "independent")
    assert result.status == "success"
    assert result.output["metadata"]["uncertainty"][0]["method"] == "independent_percentile_bootstrap"


def test_ood_never_fits_and_resume_skips_construction_and_inference(tmp_path: Path, monkeypatch):
    dataset = make_shape_dataset(tmp_path)
    runtime, _ = runtime_for(tmp_path, dataset)
    _, model_reference = train(runtime)
    parameters = {
        **evaluation_parameters(model_reference),
        "transformation": {"type": "brightness", "factor": 0.65},
        "target_cue": "brightness", "comparison_mode": "paired",
        "preservation_status": "unverified", "max_intervention_samples": 10,
    }
    monkeypatch.setattr(
        ReferenceImageLinearClassifier,
        "fit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OOD retrained")),
    )
    first = execute(runtime, "ood_evaluation", parameters, "ood-no-fit")
    assert first.status == "success"
    monkeypatch.setattr(
        "shortcutminer.intervention.engine.InterventionEngine.apply",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OOD construction repeated")),
    )
    monkeypatch.setattr(
        ReferenceImageLinearClassifier,
        "predict",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OOD inference repeated")),
    )
    second = execute(runtime, "ood_evaluation", parameters, "ood-resume")
    assert second.status == "success"
    assert second.output["metadata"]["reused"] is True


def test_evaluation_rejects_training_partition_leakage(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path)
    runtime, _ = runtime_for(tmp_path, dataset)
    _, model_reference = train(runtime)
    result = execute(runtime, "task_model_evaluation", {
        **evaluation_parameters(model_reference), "split_values": ["train"],
    }, "leaked-evaluation")
    assert result.status == "failed"
    assert "overlap" in result.error.message


def test_no_retraining_and_missing_capabilities_are_enforced(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path / "blocked")
    blocked, _ = runtime_for(
        tmp_path / "blocked", dataset, constraints={"no_retraining"}, capabilities={"retraining"}
    )
    denied = execute(blocked, "task_model_training", training_parameters(), "denied")
    assert denied.status == "failed"
    assert "no_retraining" in denied.error.message

    prediction_only, _ = runtime_for(
        tmp_path / "prediction", make_shape_dataset(tmp_path / "prediction"),
        capabilities={"predictions"},
    )
    denied_cf = execute(prediction_only, "counterfactual_intervention", {
        **evaluation_parameters("models/missing/model.json"),
        "transformation": {"type": "brightness", "factor": 0.8},
        "target_cue": "brightness", "preservation_status": "unverified",
        "max_intervention_samples": 10,
    }, "denied-cf")
    assert denied_cf.status == "failed"
    assert "intervention_inference" in denied_cf.error.message


@pytest.mark.parametrize(
    ("cue", "transformation", "target"),
    [
        (
            "spatial",
            {"type": "spatial_replace", "region": "top_left", "fill": "mean"},
            "top-left corner patch",
        ),
        ("acquisition", {"type": "brightness", "factor": 0.55}, "brightness"),
    ],
)
def test_spatial_and_acquisition_scientific_smokes(tmp_path: Path, cue, transformation, target):
    dataset = make_shape_dataset(tmp_path, cue=cue)
    runtime, _ = runtime_for(tmp_path, dataset)
    _, model_reference = train(runtime, model_id=f"reference-{cue}")
    result = execute(runtime, "counterfactual_intervention", {
        **evaluation_parameters(model_reference), "transformation": transformation,
        "target_cue": target, "preservation_status": "guaranteed_by_construction",
        "max_intervention_samples": 20,
    }, f"{cue}-counterfactual")
    assert result.status == "success"
    assert result.output["outcome"] == "completed"
    assert result.output["metrics"]["pair_count"] == 20
    assert "supports" not in json.dumps(result.output).casefold()


def test_spatial_ood_mask_and_relocate_run_as_typed_pairs(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path, cue="spatial")
    runtime, _ = runtime_for(tmp_path, dataset)
    _, model_reference = train(runtime, model_id="spatial-ood")
    transformations = [
        {"type": "spatial_mask", "region": "top_left", "fill": "mean"},
        {
            "type": "spatial_relocate",
            "source_box": {"x0": 0, "y0": 0, "x1": 0.25, "y1": 0.25},
            "target_box": {"x0": 0.75, "y0": 0.75, "x1": 1, "y1": 1},
        },
    ]
    for index, transformation in enumerate(transformations):
        result = execute(runtime, "ood_evaluation", {
            **evaluation_parameters(model_reference), "transformation": transformation,
            "target_cue": "corner patch", "comparison_mode": "paired",
            "preservation_status": "guaranteed_by_construction",
            "max_intervention_samples": 10,
        }, f"spatial-ood-{index}")
        assert result.status == "success"
        assert result.output["metrics"]["pair_count"] == 10


def test_acquisition_blur_is_measured_without_backend_verdict(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path, cue="acquisition")
    runtime, _ = runtime_for(tmp_path, dataset)
    _, model_reference = train(runtime, model_id="acquisition-blur")
    result = execute(runtime, "counterfactual_intervention", {
        **evaluation_parameters(model_reference),
        "transformation": {"type": "blur", "radius": 3},
        "target_cue": "acquisition blur", "preservation_status": "unverified",
        "max_intervention_samples": 10,
    }, "acquisition-blur")
    assert result.status == "success"
    assert result.output["metrics"]["pair_count"] == 10
    assert "verdict" not in json.dumps(result.output["metrics"]).casefold()


def test_three_seed_runtime_robustness_has_traceable_identity_and_resume(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path)
    runtime, _ = runtime_for(tmp_path, dataset)
    model_hashes = []
    for seed in (3, 17, 29):
        model_id = f"seed-{seed}"
        training = execute(runtime, "task_model_training", training_parameters(
            model_id=model_id, seed=seed
        ), f"train-{seed}")
        assert training.status == "success"
        model_reference = next(
            ref for ref in training.output["artifact_references"] if ref.endswith("/model.json")
        )
        model_hashes.append(training.output["metadata"]["details"]["model_hash"])
        parameters = {
            **evaluation_parameters(model_reference, seed=seed),
            "transformation": {"type": "global_color", "mode": "swap_red_blue"},
            "target_cue": "background color", "preservation_status": "unverified",
            "max_intervention_samples": 10,
        }
        first = execute(runtime, "counterfactual_intervention", parameters, f"cf-{seed}")
        resumed = execute(runtime, "counterfactual_intervention", parameters, f"cf-resume-{seed}")
        assert first.status == resumed.status == "success"
        assert resumed.output["metadata"]["reused"] is True
        assert first.output["metrics"] == resumed.output["metrics"]
    assert len(set(model_hashes)) == 3


def test_clean_control_never_receives_backend_verdict(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path, clean=True)
    runtime, _ = runtime_for(tmp_path, dataset)
    _, model_reference = train(runtime, model_id="clean-control")
    result = execute(runtime, "counterfactual_intervention", {
        **evaluation_parameters(model_reference),
        "transformation": {"type": "global_color", "mode": "swap_red_blue"},
        "target_cue": "background color", "preservation_status": "guaranteed_by_construction",
        "max_intervention_samples": 20,
    }, "clean")
    assert result.status == "success"
    assert not {"supports", "shortcut_present", "model_relies", "verdict"}.intersection(
        result.output["metrics"]
    )


def test_provided_mask_background_scientific_smoke(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path)
    runtime, runs = runtime_for(tmp_path, dataset)
    _, model_reference = train(runtime, model_id="background-model")
    result = execute(runtime, "counterfactual_intervention", {
        **evaluation_parameters(model_reference),
        "transformation": {
            "type": "background_replace", "mask_column": "mask",
            "replacement_mode": "constant", "fill_color": [0, 180, 0],
        },
        "target_cue": "background context",
        "preservation_status": "supported_by_masked_intervention",
        "max_intervention_samples": 10,
    }, "background")
    assert result.status == "success", result.error
    manifest_ref = next(ref for ref in result.output["artifact_references"] if ref.endswith("intervention_manifest.json"))
    manifest = json.loads((runs / "phase15-run" / manifest_ref).read_text(encoding="utf-8"))
    assert manifest["segmentation_provider"] == "provided_mask"
    assert manifest["pairs"][0]["quality"]["foreground_change_fraction"] == 0
    assert manifest["pairs"][0]["quality"]["background_change_fraction"] > 0


def test_counterfactual_resume_skips_construction_and_inference(tmp_path: Path, monkeypatch):
    dataset = make_shape_dataset(tmp_path)
    runtime, _ = runtime_for(tmp_path, dataset)
    _, model_reference = train(runtime)
    parameters = {
        **evaluation_parameters(model_reference),
        "transformation": {"type": "brightness", "factor": 0.7},
        "target_cue": "brightness", "preservation_status": "unverified",
        "max_intervention_samples": 10,
    }
    first = execute(runtime, "counterfactual_intervention", parameters, "first-cf")
    assert first.status == "success"
    monkeypatch.setattr(
        "shortcutminer.intervention.engine.InterventionEngine.apply",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("construction repeated")),
    )
    monkeypatch.setattr(
        ReferenceImageLinearClassifier,
        "predict",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("inference repeated")),
    )
    second = execute(runtime, "counterfactual_intervention", parameters, "second-cf")
    assert second.status == "success"
    assert second.output["metadata"]["reused"] is True


def test_corrupt_counterfactual_pair_is_retained_as_failure(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path)
    runtime, _ = runtime_for(tmp_path, dataset)
    _, model_reference = train(runtime)
    (dataset / "images/s040.png").write_bytes(b"corrupt")
    result = execute(runtime, "counterfactual_intervention", {
        **evaluation_parameters(model_reference),
        "transformation": {"type": "brightness", "factor": 0.8},
        "target_cue": "brightness", "preservation_status": "unverified",
        "max_intervention_samples": 20,
    }, "corrupt-pair")
    assert result.status == "success"
    assert result.output["metrics"]["pair_count"] == 19
    assert result.output["metrics"]["failed_pair_count"] == 1


def test_intervention_mask_path_cannot_escape_dataset(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path)
    runtime, _ = runtime_for(tmp_path, dataset)
    _, model_reference = train(runtime)
    frame = pd.read_csv(dataset / "metadata.csv")
    frame.loc[frame["split"] == "id", "mask"] = "../outside.png"
    frame.to_csv(dataset / "metadata.csv", index=False)
    Image.new("L", (24, 24), 255).save(tmp_path / "outside.png")
    result = execute(runtime, "counterfactual_intervention", {
        **evaluation_parameters(model_reference),
        "transformation": {"type": "background_replace", "mask_column": "mask"},
        "target_cue": "background", "preservation_status": "unverified",
        "max_intervention_samples": 20,
    }, "escape")
    assert result.status == "success"
    assert result.output["outcome"] == "inconclusive"
    assert result.output["metrics"]["failed_pair_count"] == 20


def test_intervention_mask_symlink_cannot_escape_dataset(tmp_path: Path):
    dataset = make_shape_dataset(tmp_path)
    runtime, _ = runtime_for(tmp_path, dataset)
    _, model_reference = train(runtime)
    outside = tmp_path / "outside-mask.png"
    Image.new("L", (24, 24), 255).save(outside)
    link = dataset / "masks" / "escape.png"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"File symlinks are unavailable: {exc}")
    frame = pd.read_csv(dataset / "metadata.csv")
    frame.loc[frame["split"] == "id", "mask"] = "masks/escape.png"
    frame.to_csv(dataset / "metadata.csv", index=False)
    result = execute(runtime, "counterfactual_intervention", {
        **evaluation_parameters(model_reference),
        "transformation": {"type": "background_replace", "mask_column": "mask"},
        "target_cue": "background", "preservation_status": "unverified",
        "max_intervention_samples": 20,
    }, "symlink-escape")
    assert result.status == "success"
    assert result.output["outcome"] == "inconclusive"
    assert result.output["metrics"]["failed_pair_count"] == 20
