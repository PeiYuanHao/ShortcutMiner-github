"""Paired, supported-mask background validation of the frozen model matrix.

Uses the unchanged InterventionEngine and ProvidedMaskSegmentationProvider.
This records model responses, not a shortcut verdict or an Agent assessment.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

from shortcutminer.experiment.waterbirds_paper import (
    archive_source, build_model, identity, initialization, inspect_dataset, loader,
    load_config, load_state, metrics, predict, sha, source_identity, torch_runtime, write,
)
from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.intervention.engine import InterventionEngine
from shortcutminer.intervention.protocol import BackgroundReplaceTransform
from shortcutminer.intervention.segmentation import ProvidedMaskSegmentationProvider


ARCHIVE_SHA256 = "dc77f6cffea0cbe2e41d4201115c8f29a6320ecb04fffd2444f51b8066e4b84f"
PROTOCOL = {"version": "paired-gray-background-supported-mask-v1", "fill_color": [127, 127, 127],
            "foreground_support": "all_nonzero_official_mask_values", "protection_margin_pixels": 8,
            "analysis_partition": "official_test", "label_preservation": "supported_by_masked_intervention",
            "model_selection": "reuse_frozen_validation_selected_checkpoints_no_refit",
            "limitation": "One synthetic background condition; not automatic shortcut confirmation or a natural-background causal generalization."}


def protect_mask(rgb, shape, *, margin=8):
    if rgb.ndim != 3 or rgb.shape != tuple(shape) or rgb.shape[2] != 3:
        raise ValueError("Official mask/image dimensions mismatch")
    if not np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) or not np.array_equal(rgb[:, :, 0], rgb[:, :, 2]):
        raise ValueError("Official foreground annotation must be grayscale")
    support = rgb[:, :, 0] > 0
    if not support.any():
        raise ValueError("Empty official foreground annotation")
    return maximum_filter(support, size=2*margin+1, mode="constant", cval=0).astype(bool)


def paired_metrics(y, groups, before, after, *, seed, iterations=1000):
    y, groups, before, after = [np.asarray(v, dtype=np.int64) for v in (y, groups, before, after)]
    if not len(y) or not all(v.shape == y.shape for v in (groups, before, after)):
        raise ValueError("Paired rows must have identical aligned shapes")
    if any(np.any(~np.isin(v, [0, 1])) for v in (y, groups, before, after)):
        raise ValueError("Binary task labels/groups/predictions required")
    correct_before, correct_after = before == y, after == y
    rng = np.random.default_rng(seed)
    output, boot_delta, boot_flip, weights = {}, [], [], []
    for label in (0, 1):
        for group in (0, 1):
            selected = (y == label) & (groups == group)
            n = int(selected.sum())
            if not n:
                raise ValueError("Missing evaluation group")
            joint = correct_before[selected].astype(int)*2 + correct_after[selected].astype(int)
            counts = np.bincount(joint, minlength=4)
            draws = rng.multinomial(n, counts/n, size=iterations)
            delta = (draws[:, 1] - draws[:, 2]) / n
            flips = (draws[:, 1] + draws[:, 2]) / n
            output[f"{label}:{group}"] = {"count": n,
                "accuracy_before": float(correct_before[selected].mean()),
                "accuracy_after": float(correct_after[selected].mean()),
                "accuracy_delta": float((correct_after[selected].astype(float)-correct_before[selected]).mean()),
                "prediction_flip_rate": float((before[selected] != after[selected]).mean()),
                "accuracy_delta_ci": np.quantile(delta, [0.025, 0.975]).tolist()}
            boot_delta.append(delta); boot_flip.append(flips); weights.append(n/len(y))
    return {"sample_count": len(y), "accuracy_delta": float(correct_after.mean()-correct_before.mean()),
            "prediction_flip_rate": float((before != after).mean()),
            "correct_to_wrong": int((correct_before & ~correct_after).sum()),
            "wrong_to_correct": int((~correct_before & correct_after).sum()),
            "accuracy_delta_ci": np.quantile(np.column_stack(boot_delta) @ weights, [0.025, 0.975]).tolist(),
            "prediction_flip_rate_ci": np.quantile(np.column_stack(boot_flip) @ weights, [0.025, 0.975]).tolist(),
            "groups": output, "bootstrap": {"method": "paired_joint_correctness_within_group", "iterations": iterations,
                                          "confidence_level": 0.95, "seed": seed}}


def prepare(config, matrix, assets, output):
    rows, dataset = inspect_dataset(config)
    parent = json.loads((matrix / "manifest.json").read_text(encoding="utf-8"))
    if parent["data_sha256"] != dataset["data_sha256"]:
        raise RuntimeError("Data changed after model training")
    archive = assets / "segmentations.tgz"
    if sha(archive) != ARCHIVE_SHA256:
        raise RuntimeError("Official segmentation archive changed")
    test = [r for r in rows if r["split"] == "2"]
    wanted = {"segmentations/" + Path(r["img_filename"]).with_suffix(".png").as_posix(): r for r in test}
    fingerprint = identity({"protocol": PROTOCOL, "data_sha256": dataset["data_sha256"], "archive_sha256": ARCHIVE_SHA256})
    manifest_path = output / "intervention_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["input_hash"] != fingerprint or len(manifest["samples"]) != len(test):
            raise RuntimeError("Intervention inputs changed")
        for record in manifest["samples"]:
            if sha(output / record["modified_reference"]) != record["modified_sha256"] or sha(output / record["mask_reference"]) != record["mask_sha256"]:
                raise RuntimeError("Persisted intervention changed")
        return rows, test, manifest
    output.mkdir(parents=True, exist_ok=True)

    def create(row, member_name, raw):
        sample_id = row["img_id"]
        if not sample_id.isascii() or not sample_id.isdecimal():
            raise ValueError("Invalid sample identifier")
        source = (Path(config["dataset_root"]) / row["img_filename"]).resolve(strict=True)
        original = load_rgb_image_file(source).rgb
        with Image.open(io.BytesIO(raw)) as image:
            mask = np.asarray(image.convert("RGB"))
        protected = protect_mask(mask, original.shape)
        mask_reference, modified_reference = f"protected_masks/{sample_id}.png", f"images/{sample_id}.png"
        mask_path = output / mask_reference
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(protected.astype(np.uint8)*255).save(mask_path)
        quality = InterventionEngine(segmentation_provider=ProvidedMaskSegmentationProvider()).apply(
            source_path=source, output_path=output / modified_reference, mask_path=mask_path,
            transformation=BackgroundReplaceTransform(type="background_replace", mask_column="protected_mask", fill_color=PROTOCOL["fill_color"]))
        if not quality.same_dimensions or not quality.decode_success or quality.foreground_change_fraction != 0:
            raise RuntimeError("Foreground-preservation check failed; no inference permitted")
        return {"img_id": sample_id, "original_reference": row["img_filename"], "source_sha256": sha(source),
                "official_mask_member": member_name, "official_mask_sha256": hashlib.sha256(raw).hexdigest(),
                "mask_reference": mask_reference, "mask_sha256": sha(mask_path),
                "modified_reference": modified_reference, "modified_sha256": sha(output / modified_reference),
                "quality": quality.model_dump(mode="json")}

    futures = []
    seen = set()
    with ThreadPoolExecutor(max_workers=4) as pool, tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.name not in wanted:
                continue
            if member.name in seen or not member.isfile() or member.size > 25_000_000:
                raise ValueError("Invalid/duplicate segmentation archive entry")
            seen.add(member.name)
            with tar.extractfile(member) as stream:
                raw = stream.read()
            futures.append(pool.submit(create, wanted[member.name], member.name, raw))
        if seen != set(wanted):
            raise RuntimeError("Official test masks missing; no samples silently dropped")
        records = [future.result() for future in futures]
    by_id = {r["img_id"]: r for r in records}
    manifest = {"input_hash": fingerprint, "protocol": PROTOCOL, "data_sha256": dataset["data_sha256"],
                "official_archive_sha256": ARCHIVE_SHA256, "samples": [by_id[r["img_id"]] for r in test],
                "sample_count": len(test), "failed_samples": 0, "foreground_preservation_passed": len(test)}
    write(manifest_path, manifest)
    print(json.dumps({"phase": "intervention_preparation", "status": "PASS", "samples": len(test)}), flush=True)
    return rows, test, manifest


def run(config_path, matrix_path, assets_path):
    config = load_config(config_path)
    matrix, assets = Path(matrix_path).resolve(strict=True), Path(assets_path).resolve(strict=True)
    parent = json.loads((matrix / "paper_model_results.json").read_text(encoding="utf-8"))
    if parent["status"] != "COMPLETED" or parent["completed_training_runs"] != 6:
        raise RuntimeError("All six frozen model runs must complete first")
    output = matrix / "background_validation"
    rows, test, intervention = prepare(config, matrix, assets, output)
    torch = torch_runtime(config)
    current_source = source_identity(require_clean=False)
    write(output / "execution_manifest.json", {"protocol": PROTOCOL, "source": current_source,
        "source_archive_sha256": archive_source(output / "source_snapshot.zip", current_source["source_tree_sha256"]),
        "parent_model_input_hash": parent["input_hash"], "intervention_input_hash": intervention["input_hash"]})
    train_counts = {f"{y}:{g}": sum(r["split"] == "0" and r["y"] == str(y) and r["place"] == str(g) for r in rows) for y in (0, 1) for g in (0, 1)}
    modified = [{**row, "img_filename": record["modified_reference"]} for row, record in zip(test, intervention["samples"])]
    results = []
    for original_result in parent["results"]:
        architecture, seed = original_result["architecture"], original_result["seed"]
        directory = matrix / f"{architecture}-seed{seed}"
        if sha(directory / "test_predictions.csv") != original_result["predictions_sha256"]:
            raise RuntimeError("Original predictions changed")
        base = list(csv.DictReader((directory / "test_predictions.csv").open(encoding="utf-8", newline="")))
        if [r["img_id"] for r in base] != [r["img_id"] for r in test]:
            raise RuntimeError("Sample identity alignment failed")
        weight_path, _ = initialization(architecture)
        model = build_model(architecture, weight_path)
        load_state(directory / "best.npz", original_result["checkpoint_sha256"], model)
        # Verify the frozen checkpoint reproduces its recorded predictions before treatment.
        _, _, control, control_probs, _ = predict(model, loader(config, test[:32], training=False, seed=seed))
        if control != [int(r["prediction"]) for r in base[:32]] or not np.allclose(control_probs,
            [[float(r["probability_0"]), float(r["probability_1"])] for r in base[:32]], atol=1e-6):
            raise RuntimeError("Checkpoint/control predictions disagree")
        derived_config = {**config, "dataset_root": str(output)}
        y, groups, predictions, probs, indices = predict(model, loader(derived_config, modified, training=False, seed=seed))
        if indices != list(range(len(test))) or y != [int(r["y"]) for r in base] or groups != [int(r["place"]) for r in base]:
            raise RuntimeError("Paired prediction alignment changed")
        destination = output / f"{architecture}-seed{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        prediction_file = destination / "paired_predictions.csv"
        with prediction_file.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(["img_id", "y", "place", "prediction_before", "prediction_after", "probability_after_0", "probability_after_1"])
            for row, before, after, probability in zip(test, base, predictions, probs):
                writer.writerow([row["img_id"], row["y"], row["place"], before["prediction"], after, *probability])
        comparison = paired_metrics(y, groups, [int(r["prediction"]) for r in base], predictions, seed=seed, iterations=config["bootstrap_iterations"])
        result = {"architecture": architecture, "seed": seed, "status": "COMPLETED", "checkpoint_sha256": original_result["checkpoint_sha256"],
                  "paired_predictions_sha256": sha(prediction_file), "control_prediction_check": "PASS",
                  "label_preservation": "supported_by_masked_intervention", "comparison": comparison,
                  "modified_metrics": metrics(y, groups, predictions, train_group_counts=train_counts, seed=seed, iterations=config["bootstrap_iterations"])}
        write(destination / "result.json", result); results.append(result)
        print(json.dumps({"model": architecture, "seed": seed, "samples": len(y), "flip_rate": comparison["prediction_flip_rate"], "accuracy_delta": comparison["accuracy_delta"]}), flush=True)
        del model
        torch.cuda.empty_cache()
    summary = {architecture: {key: {"mean": float(np.mean([r["comparison"][key] for r in results if r["architecture"] == architecture])),
                                    "std": float(np.std([r["comparison"][key] for r in results if r["architecture"] == architecture], ddof=1))}
                              for key in ("prediction_flip_rate", "accuracy_delta")} for architecture in config["models"]}
    write(output / "results.json", {"status": "COMPLETED", "protocol": PROTOCOL, "sample_count_per_model": len(test),
        "completed_model_evaluations": 6, "foreground_preservation_passed": intervention["foreground_preservation_passed"],
        "parent_model_input_hash": parent["input_hash"], "results": results, "summary": summary})
