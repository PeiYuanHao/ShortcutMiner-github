"""Real CPU task-model, OOD, and paired counterfactual experiment backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import TypeAdapter
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from shortcutminer.evaluation.calibration import (
    bootstrap_interval,
    independent_bootstrap_interval,
)
from shortcutminer.inspection.image import SafeImageDecodeError, load_rgb_image_file
from shortcutminer.intervention.engine import InterventionEngine, InterventionError
from shortcutminer.intervention.protocol import (
    BackgroundReplaceTransform,
    CompressionTransform,
    CounterfactualPair,
    InterventionFailure,
    InterventionManifest,
)
from shortcutminer.intervention.segmentation import ProvidedMaskSegmentationProvider
from shortcutminer.protocol.evidence import UncertaintyEstimate
from shortcutminer.protocol.experiment import (
    CounterfactualInterventionParameters,
    OODEvaluationParameters,
    TaskModelEvaluationParameters,
    TaskModelTrainingParameters,
)
from shortcutminer.protocol.identify import SafeRelativePath
from shortcutminer.protocol.model_reliance import TaskModelEvaluationRecord, TaskPrediction
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.task_models.reference import (
    FEATURE_VERSION,
    ReferenceImageLinearClassifier,
    TaskModelError,
)
from shortcutminer.task_models.registry import TASK_MODEL_REGISTRY


@dataclass
class ModelRelianceEvidence:
    outcome: str
    metrics: dict[str, str | int | float | bool | None]
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    input_references: list[str] = field(default_factory=list)
    uncertainty: list[UncertaintyEstimate] = field(default_factory=list)
    artifact_references: list[str] = field(default_factory=list)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_root(access) -> Path:
    if access.context.runs_root is None:
        raise TaskModelError("Model-reliance experiments require runs_root.")
    root = (access.context.runs_root / access.context.run_id).resolve(strict=True)
    root.relative_to(access.context.runs_root.resolve(strict=True))
    return root


def _run_output(access, reference: str) -> Path:
    validated = TypeAdapter(SafeRelativePath).validate_python(reference)
    root = _run_root(access)
    path = (root / validated).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TaskModelError("Output artifact escaped the run root.") from exc
    return path


def _required_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise TaskModelError("Required model-reliance columns are missing: " + ", ".join(missing))


def _safe_reference(value: Any) -> str:
    return TypeAdapter(SafeRelativePath).validate_python(str(value))


def _base_rows(parameters, access) -> pd.DataFrame:
    frame = access.load_csv(parameters.metadata_file)
    columns = [parameters.sample_id_column, parameters.sample_path_column, parameters.label_column]
    if parameters.split_column:
        columns.append(parameters.split_column)
    transformation = getattr(parameters, "transformation", None)
    if isinstance(transformation, BackgroundReplaceTransform):
        columns.append(transformation.mask_column)
    _required_columns(frame, columns)
    rows = frame.dropna(subset=columns).copy()
    rows[parameters.sample_id_column] = rows[parameters.sample_id_column].astype(str)
    rows[parameters.sample_path_column] = rows[parameters.sample_path_column].astype(str)
    rows[parameters.label_column] = rows[parameters.label_column].astype(str)
    if rows[parameters.sample_id_column].duplicated().any():
        raise TaskModelError("Task-model sample IDs must be unique.")
    return rows.sort_values(parameters.sample_id_column, kind="stable").reset_index(drop=True)


def _filter_evaluation_rows(parameters, access, artifact=None) -> pd.DataFrame:
    rows = _base_rows(parameters, access)
    training_ids: set[str] = set()
    if artifact is not None:
        try:
            payload = json.loads(access.run_file(parameters.model_reference).read_text(encoding="utf-8"))
            training_ids = set(str(value) for value in payload["training"]["train_sample_ids"])
        except Exception as exc:
            raise TaskModelError("Model artifact has no valid frozen training partition.") from exc
    if parameters.evaluation_partition == "training_manifest_test":
        try:
            test_ids = set(str(value) for value in payload["training"]["test_sample_ids"])
        except Exception as exc:
            raise TaskModelError("Model training manifest has no valid test partition.") from exc
        rows = rows[rows[parameters.sample_id_column].isin(test_ids)]
    if parameters.split_column and parameters.split_values:
        rows = rows[rows[parameters.split_column].astype(str).isin(parameters.split_values)]
    selected_ids = set(rows[parameters.sample_id_column].astype(str))
    overlap = selected_ids & training_ids
    if overlap:
        raise TaskModelError(
            f"Evaluation samples overlap the frozen training partition ({len(overlap)} sample IDs)."
        )
    return rows.iloc[: parameters.max_eval_samples].copy()


def _resolve_valid_rows(rows: pd.DataFrame, parameters, access) -> tuple[pd.DataFrame, list[Path], list[str]]:
    valid_indices: list[int] = []
    paths: list[Path] = []
    warnings: list[str] = []
    for index, row in rows.iterrows():
        sample_id = str(row[parameters.sample_id_column])
        try:
            reference = _safe_reference(row[parameters.sample_path_column])
            path = access.dataset_file(reference)
            load_rgb_image_file(path, max_file_bytes=25_000_000, max_pixels=20_000_000)
        except Exception as exc:
            warnings.append(f"Sample {sample_id} was excluded after safe image decoding failed: {type(exc).__name__}.")
            continue
        valid_indices.append(index)
        paths.append(path)
    return rows.loc[valid_indices].reset_index(drop=True), paths, warnings


def _split_training_rows(parameters: TaskModelTrainingParameters, access):
    rows = _base_rows(parameters, access)
    generated = parameters.split_column is None
    if not generated:
        split = rows[parameters.split_column].astype(str)
        train = rows[split.isin(parameters.train_split_values)].copy()
        validation = rows[split.isin(parameters.validation_split_values)].copy()
        test = rows[split.isin(parameters.test_split_values)].copy()
    else:
        if len(rows) < 6 or rows[parameters.label_column].nunique() < 2:
            return rows.iloc[0:0], rows.iloc[0:0], rows.iloc[0:0], True
        indices = np.arange(len(rows))
        try:
            train_indices, test_indices = train_test_split(
                indices,
                test_size=parameters.holdout_fraction,
                random_state=parameters.seed,
                stratify=rows[parameters.label_column],
            )
        except ValueError:
            return rows.iloc[0:0], rows.iloc[0:0], rows.iloc[0:0], True
        train = rows.iloc[sorted(train_indices)].copy()
        validation = rows.iloc[0:0].copy()
        test = rows.iloc[sorted(test_indices)].copy()
    return (
        train.iloc[: parameters.max_train_samples].copy(),
        validation.iloc[: parameters.max_eval_samples].copy(),
        test.iloc[: parameters.max_eval_samples].copy(),
        generated,
    )


def task_model_training_backend(parameters: TaskModelTrainingParameters, access) -> ModelRelianceEvidence:
    if parameters.model_backend not in TASK_MODEL_REGISTRY:
        raise TaskModelError("Task-model backend is not statically allowlisted.")
    train, validation, test, generated = _split_training_rows(parameters, access)
    if test.empty:
        return ModelRelianceEvidence(
            outcome="inconclusive",
            metrics={"train_count": len(train), "test_count": 0},
            summary="Reference task-model training was scientifically inconclusive.",
            limitations=["No isolated test samples were available."],
            input_references=[parameters.metadata_file],
        )
    train, paths, decode_warnings = _resolve_valid_rows(train, parameters, access)
    labels = train[parameters.label_column].astype(str).tolist()
    if len(train) < 4 or len(set(labels)) < 2 or min(pd.Series(labels).value_counts()) < 2:
        return ModelRelianceEvidence(
            outcome="inconclusive",
            metrics={"train_count": len(train), "class_count": len(set(labels))},
            summary="Reference task-model training was scientifically inconclusive.",
            warnings=decode_warnings,
            limitations=["Too few safely decoded training samples or fewer than two viable classes."],
            input_references=[parameters.metadata_file],
        )
    sample_ids = train[parameters.sample_id_column].astype(str).tolist()
    validation_ids = validation[parameters.sample_id_column].astype(str).tolist()
    test_ids = test[parameters.sample_id_column].astype(str).tolist()
    if set(sample_ids) & (set(validation_ids) | set(test_ids)):
        raise TaskModelError("Training split overlaps validation or test samples.")
    hashes = [_file_hash(path) for path in paths]
    fingerprint = _canonical_hash({
        "dataset_hash": _file_hash(access.dataset_file(parameters.metadata_file)),
        "train_samples": list(zip(sample_ids, hashes, strict=True)),
        "backend": parameters.model_backend,
        "feature_version": parameters.feature_version,
        "seed": parameters.seed,
        "max_iterations": parameters.max_model_iterations,
        "class_weight": parameters.class_weight,
    })
    root = _run_root(access)
    directory_reference = f"models/{parameters.model_id}"
    model_reference = f"{directory_reference}/model.json"
    weights_reference = f"{directory_reference}/weights.npz"
    manifest_reference = f"{directory_reference}/training_manifest.json"
    model_path = _run_output(access, model_reference)
    if model_path.is_file():
        model, artifact = ReferenceImageLinearClassifier.load_safe_artifact(
            model_path, run_root=root, expected_run_id=access.context.run_id
        )
        if artifact.training_fingerprint != fingerprint:
            raise TaskModelError("Existing model_id has a different training identity.")
        return ModelRelianceEvidence(
            outcome="completed",
            metrics={
                "train_count": len(train), "validation_count": len(validation),
                "test_count": len(test), "class_count": len(artifact.classes),
                "model_reused": True,
            },
            summary="The matching safe reference model artifact was reused without retraining.",
            warnings=decode_warnings,
            limitations=list(artifact.limitations),
            input_references=[parameters.metadata_file],
            artifact_references=[model_reference, weights_reference, manifest_reference],
            details={"model_id": artifact.model_id, "model_hash": artifact.model_hash},
        )
    model = ReferenceImageLinearClassifier(parameters.model_id, class_weight=parameters.class_weight)
    model.fit(paths, labels, seed=parameters.seed, max_iterations=parameters.max_model_iterations)
    predictions = model.predict(paths)
    train_accuracy = float(accuracy_score(labels, predictions))
    train_balanced = float(balanced_accuracy_score(labels, predictions))
    training_manifest = {
        "schema_version": "1.0",
        "run_id": access.context.run_id,
        "model_id": parameters.model_id,
        "training_fingerprint": fingerprint,
        "dataset_hash": _file_hash(access.dataset_file(parameters.metadata_file)),
        "split_generated_by_runtime": generated,
        "seed": parameters.seed,
        "train_sample_ids": sample_ids,
        "validation_sample_ids": validation_ids,
        "test_sample_ids": test_ids,
        "feature_version": FEATURE_VERSION,
        "model_backend": parameters.model_backend,
        "max_iterations": parameters.max_model_iterations,
        "converged": bool(model.converged_),
        "training_rows": len(train),
        "classes": sorted(set(labels)),
        "metrics": {"training_accuracy": train_accuracy, "training_balanced_accuracy": train_balanced},
        "limitations": ["Training metrics are descriptive and do not establish generalization."],
    }
    artifact = model.save_safe_artifact(
        run_id=access.context.run_id,
        run_root=root,
        model_directory=_run_output(access, directory_reference + "/placeholder").parent,
        model_reference=model_reference,
        weights_reference=weights_reference,
        training_fingerprint=fingerprint,
        training_manifest=training_manifest,
    )
    atomic_write_json(
        _run_output(access, manifest_reference), training_manifest, error_type=TaskModelError
    )
    return ModelRelianceEvidence(
        outcome="completed",
        metrics={
            "train_count": len(train), "validation_count": len(validation), "test_count": len(test),
            "class_count": len(artifact.classes), "training_accuracy": train_accuracy,
            "training_balanced_accuracy": train_balanced, "converged": bool(model.converged_),
            "model_reused": False,
        },
        summary="A bounded CPU reference image classifier was trained on the isolated train split.",
        details={"model_id": artifact.model_id, "model_hash": artifact.model_hash, "training_fingerprint": fingerprint},
        warnings=decode_warnings,
        limitations=list(artifact.limitations),
        input_references=[parameters.metadata_file],
        artifact_references=[model_reference, weights_reference, manifest_reference],
    )


def _load_model(parameters, access):
    model_path = access.run_file(parameters.model_reference)
    return ReferenceImageLinearClassifier.load_safe_artifact(
        model_path, run_root=_run_root(access), expected_run_id=access.context.run_id
    )


def _evaluate(model, artifact, rows: pd.DataFrame, paths: list[Path], parameters):
    labels = rows[parameters.label_column].astype(str).tolist()
    predictions = model.predict(paths)
    probabilities = model.predict_proba(paths)
    logits = model.predict_logits(paths)
    class_index = {str(value): index for index, value in enumerate(model.classes_)}
    records = []
    for index, (_, row) in enumerate(rows.iterrows()):
        records.append(TaskPrediction(
            sample_id=str(row[parameters.sample_id_column]),
            sample_reference=_safe_reference(row[parameters.sample_path_column]),
            label=labels[index], prediction=predictions[index],
            probabilities=[float(value) for value in probabilities[index]],
            logits=[float(value) for value in logits[index]],
        ))
    correct = np.asarray([prediction == label for prediction, label in zip(predictions, labels, strict=True)], dtype=float)
    accuracy = float(np.mean(correct)) if len(correct) else None
    balanced = float(balanced_accuracy_score(labels, predictions)) if labels else None
    return labels, predictions, probabilities, logits, class_index, correct, TaskModelEvaluationRecord(
        model_id=artifact.model_id, model_hash=artifact.model_hash,
        sample_count=len(labels), correct_count=int(np.sum(correct)),
        accuracy=accuracy, balanced_accuracy=balanced, predictions=records,
    )


def task_model_evaluation_backend(parameters: TaskModelEvaluationParameters, access) -> ModelRelianceEvidence:
    model, artifact = _load_model(parameters, access)
    rows, paths, warnings = _resolve_valid_rows(
        _filter_evaluation_rows(parameters, access, artifact), parameters, access
    )
    if not paths:
        return ModelRelianceEvidence(
            outcome="inconclusive", metrics={"sample_count": 0},
            summary="Task-model evaluation had no safely decodable samples.", warnings=warnings,
            limitations=["No valid evaluation samples were available."],
            input_references=[parameters.model_reference, parameters.metadata_file],
        )
    *_, correct, evaluation = _evaluate(model, artifact, rows, paths, parameters)
    uncertainty = [bootstrap_interval(
        metric="accuracy", sample_count=len(correct),
        statistic=lambda indices: float(np.mean(correct[indices])),
        confidence_level=parameters.confidence_level,
        iterations=parameters.bootstrap_iterations, seed=parameters.seed,
        resampling_unit="prediction_row",
    )]
    reference = f"models/evaluations/{access.request.task_id}/{artifact.model_hash[:12]}.json"
    payload = {
        "schema_version": "1.0", "run_id": access.context.run_id,
        "candidate_id": access.request.candidate_id, "task_id": access.request.task_id,
        "evaluation": evaluation.model_dump(mode="json"),
        "uncertainty": [item.model_dump(mode="json") for item in uncertainty],
    }
    atomic_write_json(_run_output(access, reference), payload, error_type=TaskModelError)
    return ModelRelianceEvidence(
        outcome="completed",
        metrics={
            "sample_count": evaluation.sample_count, "correct_count": evaluation.correct_count,
            "accuracy": evaluation.accuracy, "balanced_accuracy": evaluation.balanced_accuracy,
            "model_id": artifact.model_id, "model_hash": artifact.model_hash,
        },
        summary="The safe task model was evaluated on the requested ID split; no shortcut verdict was inferred.",
        details={"model_id": artifact.model_id, "model_hash": artifact.model_hash},
        warnings=warnings, limitations=list(artifact.limitations),
        input_references=[parameters.model_reference, parameters.metadata_file],
        uncertainty=uncertainty, artifact_references=[reference],
    )


def _construct_and_measure(parameters, access, *, artifact_kind: str) -> ModelRelianceEvidence:
    model, artifact = _load_model(parameters, access)
    selected = _filter_evaluation_rows(parameters, access, artifact).iloc[: parameters.max_intervention_samples].copy()
    root = _run_root(access)
    provider = ProvidedMaskSegmentationProvider() if isinstance(parameters.transformation, BackgroundReplaceTransform) else None
    engine = InterventionEngine(segmentation_provider=provider)
    input_identity = []
    for _, row in selected.iterrows():
        try:
            reference = _safe_reference(row[parameters.sample_path_column])
            input_identity.append((str(row[parameters.sample_id_column]), reference, _file_hash(access.dataset_file(reference))))
            if isinstance(parameters.transformation, BackgroundReplaceTransform):
                mask_reference = _safe_reference(row[parameters.transformation.mask_column])
                input_identity.append((str(row[parameters.sample_id_column]) + ":mask", mask_reference, _file_hash(access.dataset_file(mask_reference))))
        except Exception:
            input_identity.append((str(row[parameters.sample_id_column]), "invalid", "invalid"))
    fingerprint = _canonical_hash({
        "parameters": parameters.model_dump(mode="json"), "model_hash": artifact.model_hash,
        "inputs": input_identity, "engine": engine.engine_version,
        "segmentation": provider.provider_version if provider else None,
    })
    base_reference = f"interventions/{access.request.task_id}/{artifact_kind}/{fingerprint[:12]}"
    pairs: list[CounterfactualPair] = []
    failures: list[InterventionFailure] = []
    original_paths: list[Path] = []
    modified_paths: list[Path] = []
    usable_rows: list[pd.Series] = []
    replacement_path = None
    if isinstance(parameters.transformation, BackgroundReplaceTransform) and parameters.transformation.replacement_sample_reference:
        replacement_path = access.dataset_file(parameters.transformation.replacement_sample_reference)
    for _, row in selected.iterrows():
        sample_id = str(row[parameters.sample_id_column])
        raw_reference = str(row[parameters.sample_path_column])
        safe_failure_reference = "invalid-reference"
        try:
            reference = _safe_reference(raw_reference)
            safe_failure_reference = reference
            source_path = access.dataset_file(reference)
            suffix = ".jpg" if isinstance(parameters.transformation, CompressionTransform) else ".png"
            output_reference = f"{base_reference}/{hashlib.sha256(sample_id.encode()).hexdigest()[:16]}{suffix}"
            output_path = _run_output(access, output_reference)
            mask_reference = None
            mask_path = None
            if isinstance(parameters.transformation, BackgroundReplaceTransform):
                mask_reference = _safe_reference(row[parameters.transformation.mask_column])
                mask_path = access.dataset_file(mask_reference)
            quality = engine.apply(
                source_path=source_path, output_path=output_path,
                transformation=parameters.transformation, mask_path=mask_path,
                replacement_path=replacement_path,
            )
            pair = CounterfactualPair(
                pair_id=f"PAIR-{hashlib.sha256((sample_id + fingerprint).encode()).hexdigest()[:16]}",
                original_sample_id=sample_id, original_reference=reference,
                counterfactual_reference=output_reference,
                label=str(row[parameters.label_column]), transformation=parameters.transformation,
                target_cue=parameters.target_cue,
                intended_invariants=parameters.intended_invariants,
                seed=parameters.seed,
                model_id=artifact.model_id, model_hash=artifact.model_hash,
                mask_reference=mask_reference,
                preservation_status=parameters.preservation_status,
                quality=quality,
                limitations=[
                    "Label preservation is an explicit intervention-design assumption, not an automatic causal guarantee."
                ],
            )
            pairs.append(pair)
            original_paths.append(source_path)
            modified_paths.append(output_path)
            usable_rows.append(row)
        except Exception as exc:
            failures.append(InterventionFailure(
                sample_id=sample_id, sample_reference=safe_failure_reference,
                error_type=type(exc).__name__, message=str(exc) or type(exc).__name__,
            ))
    manifest_reference = f"{base_reference}/intervention_manifest.json"
    ood_id = f"OOD-{fingerprint[:16]}" if artifact_kind == "ood" else None
    manifest = InterventionManifest(
        run_id=access.context.run_id, candidate_id=access.request.candidate_id,
        task_id=access.request.task_id, intervention_id=f"INT-{fingerprint[:16]}",
        ood_id=ood_id,
        fingerprint=fingerprint, target_cue=parameters.target_cue,
        intended_invariants=parameters.intended_invariants,
        comparison_mode=getattr(parameters, "comparison_mode", "paired"),
        transformation=parameters.transformation, seed=parameters.seed,
        segmentation_provider=provider.provider_name if provider else None,
        segmentation_version=provider.provider_version if provider else None,
        model_id=artifact.model_id, model_hash=artifact.model_hash,
        selected_count=len(selected), usable_pair_count=len(pairs),
        failed_pair_count=len(failures),
        pair_coverage=len(pairs) / max(1, len(selected)),
        pairs=pairs, failures=failures,
        limitations=[
            "Intervention measurements describe model response and do not automatically establish shortcut reliance or causality."
        ],
    )
    atomic_write_json(_run_output(access, manifest_reference), manifest.model_dump(mode="json"), error_type=InterventionError)
    if not pairs:
        return ModelRelianceEvidence(
            outcome="inconclusive", metrics={"pair_count": 0, "failed_pair_count": len(failures)},
            summary="No usable intervention pairs could be constructed.",
            limitations=["Intervention construction failed or was blocked for every selected sample."],
            input_references=[parameters.model_reference, parameters.metadata_file],
            artifact_references=[manifest_reference],
            details={
                "model_id": artifact.model_id, "model_hash": artifact.model_hash,
                "intervention_id": manifest.intervention_id, "ood_id": manifest.ood_id,
            },
        )
    rows = pd.DataFrame(usable_rows).reset_index(drop=True)
    labels = rows[parameters.label_column].astype(str).tolist()
    original_pred = model.predict(original_paths)
    modified_pred = model.predict(modified_paths)
    original_proba = model.predict_proba(original_paths)
    modified_proba = model.predict_proba(modified_paths)
    original_logits = model.predict_logits(original_paths)
    modified_logits = model.predict_logits(modified_paths)
    class_index = {str(value): index for index, value in enumerate(model.classes_)}
    original_correct = np.asarray([p == y for p, y in zip(original_pred, labels, strict=True)], dtype=float)
    modified_correct = np.asarray([p == y for p, y in zip(modified_pred, labels, strict=True)], dtype=float)
    flips = np.asarray([a != b for a, b in zip(original_pred, modified_pred, strict=True)], dtype=float)
    confidence_delta = np.asarray([
        modified_proba[i, class_index[label]] - original_proba[i, class_index[label]]
        for i, label in enumerate(labels)
    ])
    logit_delta = np.asarray([
        modified_logits[i, class_index[label]] - original_logits[i, class_index[label]]
        for i, label in enumerate(labels)
    ])
    original_margin = np.asarray([
        original_logits[i, class_index[label]] - np.max(np.delete(original_logits[i], class_index[label]))
        for i, label in enumerate(labels)
    ])
    modified_margin = np.asarray([
        modified_logits[i, class_index[label]] - np.max(np.delete(modified_logits[i], class_index[label]))
        for i, label in enumerate(labels)
    ])
    accuracy_delta = float(np.mean(modified_correct) - np.mean(original_correct))
    if getattr(parameters, "comparison_mode", "paired") == "independent":
        uncertainty = [independent_bootstrap_interval(
            metric="accuracy_delta", first_count=len(original_correct), second_count=len(modified_correct),
            statistic=lambda first, second: float(np.mean(modified_correct[second]) - np.mean(original_correct[first])),
            confidence_level=parameters.confidence_level,
            iterations=parameters.bootstrap_iterations, seed=parameters.seed,
        )]
    else:
        uncertainty = [bootstrap_interval(
            metric="accuracy_delta", sample_count=len(pairs),
            statistic=lambda indices: float(np.mean(modified_correct[indices] - original_correct[indices])),
            confidence_level=parameters.confidence_level,
            iterations=parameters.bootstrap_iterations, seed=parameters.seed,
            resampling_unit="counterfactual_pair",
        )]
    measurement_reference = f"{base_reference}/measurement.json"
    measurement = {
        "schema_version": "1.0", "run_id": access.context.run_id,
        "candidate_id": access.request.candidate_id, "task_id": access.request.task_id,
        "model_id": artifact.model_id, "model_hash": artifact.model_hash,
        "intervention_id": manifest.intervention_id,
        "ood_id": manifest.ood_id,
        "rows": [
            {
                "pair_id": pair.pair_id, "label": labels[index],
                "prediction_original": original_pred[index],
                "prediction_counterfactual": modified_pred[index],
                "probability_original": [float(v) for v in original_proba[index]],
                "probability_counterfactual": [float(v) for v in modified_proba[index]],
                "logits_original": [float(v) for v in original_logits[index]],
                "logits_counterfactual": [float(v) for v in modified_logits[index]],
            }
            for index, pair in enumerate(pairs)
        ],
        "uncertainty": [item.model_dump(mode="json") for item in uncertainty],
    }
    atomic_write_json(_run_output(access, measurement_reference), measurement, error_type=TaskModelError)
    metrics = {
        "sample_count": len(pairs), "pair_count": len(pairs), "usable_pair_count": len(pairs),
        "failed_pair_count": len(failures), "prediction_flip_count": int(np.sum(flips)),
        "prediction_flip_rate": float(np.mean(flips)),
        "id_accuracy": float(np.mean(original_correct)),
        "original_accuracy": float(np.mean(original_correct)),
        "ood_accuracy": float(np.mean(modified_correct)),
        "counterfactual_accuracy": float(np.mean(modified_correct)),
        "accuracy_delta": accuracy_delta,
        "id_balanced_accuracy": float(balanced_accuracy_score(labels, original_pred)),
        "ood_balanced_accuracy": float(balanced_accuracy_score(labels, modified_pred)),
        "balanced_accuracy_delta": float(balanced_accuracy_score(labels, modified_pred) - balanced_accuracy_score(labels, original_pred)),
        "true_class_confidence_delta": float(np.mean(confidence_delta)),
        "true_class_logit_delta": float(np.mean(logit_delta)),
        "logit_margin_delta": float(np.mean(modified_margin - original_margin)),
        "model_id": artifact.model_id, "model_hash": artifact.model_hash,
        "intervention_id": manifest.intervention_id,
        "ood_id": manifest.ood_id,
    }
    return ModelRelianceEvidence(
        outcome="completed", metrics=metrics,
        summary="The same task model was measured before and after a deterministic cue intervention; no verdict threshold was applied.",
        details={
            "model_id": artifact.model_id, "model_hash": artifact.model_hash,
            "intervention_id": manifest.intervention_id, "ood_id": manifest.ood_id,
            "comparison_mode": manifest.comparison_mode,
            "pair_coverage": len(pairs) / max(1, len(selected)),
        },
        warnings=[], limitations=list(manifest.limitations),
        input_references=[parameters.model_reference, parameters.metadata_file],
        uncertainty=uncertainty,
        artifact_references=[parameters.model_reference, manifest_reference, measurement_reference, *[pair.counterfactual_reference for pair in pairs]],
    )


def ood_evaluation_backend(parameters: OODEvaluationParameters, access) -> ModelRelianceEvidence:
    return _construct_and_measure(parameters, access, artifact_kind="ood")


def counterfactual_intervention_backend(parameters: CounterfactualInterventionParameters, access) -> ModelRelianceEvidence:
    return _construct_and_measure(parameters, access, artifact_kind="counterfactual")


__all__ = [
    "counterfactual_intervention_backend",
    "ModelRelianceEvidence", "ood_evaluation_backend",
    "task_model_evaluation_backend", "task_model_training_backend",
]
