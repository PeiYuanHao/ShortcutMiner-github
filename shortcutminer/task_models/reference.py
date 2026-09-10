"""Bounded CPU reference image classifier with non-executable persistence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.protocol.model_reliance import (
    TaskModelArtifact,
    TaskModelCapabilities,
)
from shortcutminer.storage.json_artifacts import atomic_write_json


FEATURE_VERSION = "reference_image_features_v1"
BACKEND_VERSION = "reference_image_linear_v1"
FEATURE_SIZE = (16, 16)


class TaskModelError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class ReferenceImageLinearClassifier:
    """Logistic-regression fixture backend, not a paper model or deep backbone."""

    model_kind: Literal["reference_image_linear"] = "reference_image_linear"
    capabilities = TaskModelCapabilities(
        predictions=True,
        probabilities=True,
        logits=True,
        training=True,
        checkpoint=True,
        image_input=True,
        intervention_inference=True,
    )

    def __init__(self, model_id: str, *, class_weight: str = "none") -> None:
        self.model_id = model_id
        self.class_weight = class_weight
        self.classes_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.converged_: bool | None = None

    @staticmethod
    def _feature(path: Path) -> np.ndarray:
        decoded = load_rgb_image_file(path, max_file_bytes=25_000_000, max_pixels=20_000_000)
        resized = Image.fromarray(decoded.rgb, mode="RGB").resize(
            FEATURE_SIZE, Image.Resampling.BILINEAR
        )
        return np.asarray(resized, dtype=np.float64).reshape(-1) / 255.0

    @classmethod
    def extract_features(cls, paths: list[Path]) -> np.ndarray:
        if not paths:
            raise TaskModelError("Task-model input is empty.")
        return np.vstack([cls._feature(path) for path in paths])

    def fit(
        self,
        paths: list[Path],
        labels: list[str],
        *,
        seed: int,
        max_iterations: int,
    ) -> None:
        if len(paths) != len(labels) or len(paths) < 4:
            raise TaskModelError("Reference training requires at least four aligned samples.")
        classes, counts = np.unique(np.asarray(labels, dtype=str), return_counts=True)
        if len(classes) < 2 or np.min(counts) < 2:
            raise TaskModelError("Reference training requires at least two classes and two samples per class.")
        features = self.extract_features(paths)
        mean = np.mean(features, axis=0)
        scale = np.std(features, axis=0)
        scale[scale == 0] = 1.0
        normalized = (features - mean) / scale
        classifier = LogisticRegression(
            max_iter=max_iterations,
            random_state=seed,
            class_weight=None if self.class_weight == "none" else "balanced",
            solver="lbfgs",
        )
        classifier.fit(normalized, np.asarray(labels, dtype=str))
        self.classes_ = np.asarray(classifier.classes_, dtype=str)
        self.coef_ = np.asarray(classifier.coef_, dtype=np.float64)
        self.intercept_ = np.asarray(classifier.intercept_, dtype=np.float64)
        self.feature_mean_ = mean
        self.feature_scale_ = scale
        self.converged_ = bool(np.all(classifier.n_iter_ < max_iterations))

    def _require_fitted(self) -> None:
        values = (self.classes_, self.coef_, self.intercept_, self.feature_mean_, self.feature_scale_)
        if any(value is None for value in values):
            raise TaskModelError("Reference model is not fitted.")

    def predict_logits(self, paths: list[Path]) -> np.ndarray:
        self._require_fitted()
        features = (self.extract_features(paths) - self.feature_mean_) / self.feature_scale_
        scores = features @ self.coef_.T + self.intercept_
        if len(self.classes_) == 2 and scores.shape[1] == 1:
            scores = np.column_stack([np.zeros(len(scores)), scores[:, 0]])
        return np.asarray(scores, dtype=np.float64)

    def predict_proba(self, paths: list[Path]) -> np.ndarray:
        logits = self.predict_logits(paths)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / np.sum(exponent, axis=1, keepdims=True)

    def predict(self, paths: list[Path]) -> list[str]:
        probabilities = self.predict_proba(paths)
        return [str(self.classes_[index]) for index in np.argmax(probabilities, axis=1)]

    def save_safe_artifact(
        self,
        *,
        run_id: str,
        run_root: Path,
        model_directory: Path,
        model_reference: str,
        weights_reference: str,
        training_fingerprint: str,
        training_manifest: dict[str, Any],
    ) -> TaskModelArtifact:
        self._require_fitted()
        model_directory.mkdir(parents=True, exist_ok=True)
        weights_path = run_root / weights_reference
        model_path = run_root / model_reference
        for path in (weights_path, model_path):
            try:
                path.resolve(strict=False).relative_to(run_root.resolve(strict=False))
            except ValueError as exc:
                raise TaskModelError("Model artifact path escaped the run root.") from exc
        temporary = weights_path.with_name(f".{weights_path.name}.{uuid4().hex}.npz")
        try:
            np.savez_compressed(
                temporary,
                classes=self.classes_, coef=self.coef_, intercept=self.intercept_,
                feature_mean=self.feature_mean_, feature_scale=self.feature_scale_,
            )
            temporary.replace(weights_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        weights_hash = _sha256(weights_path)
        model_hash = _canonical_hash({
            "backend_version": BACKEND_VERSION,
            "feature_version": FEATURE_VERSION,
            "training_fingerprint": training_fingerprint,
            "weights_sha256": weights_hash,
        })
        artifact = TaskModelArtifact(
            run_id=run_id,
            model_id=self.model_id,
            model_kind=self.model_kind,
            backend="reference_image_linear",
            backend_version=BACKEND_VERSION,
            feature_version=FEATURE_VERSION,
            model_hash=model_hash,
            training_fingerprint=training_fingerprint,
            model_reference=model_reference,
            weights_reference=weights_reference,
            capabilities=self.capabilities,
            classes=[str(value) for value in self.classes_],
            limitations=[
                "This bounded linear image classifier is a runtime reference model, not a paper baseline or deep backbone."
            ],
        )
        atomic_write_json(
            model_path,
            {
                **artifact.model_dump(mode="json"),
                "weights_sha256": weights_hash,
                "feature_specification": {
                    "resize": list(FEATURE_SIZE),
                    "mode": "RGB",
                    "representation": "flattened_float_rgb_0_1",
                    "standardization": "training_mean_and_population_std",
                    "resampling": "bilinear",
                },
                "training": training_manifest,
                "converged": self.converged_,
            },
            error_type=TaskModelError,
        )
        return artifact

    @classmethod
    def load_safe_artifact(
        cls,
        model_path: Path,
        *,
        run_root: Path,
        expected_run_id: str,
    ) -> tuple["ReferenceImageLinearClassifier", TaskModelArtifact]:
        try:
            resolved_model = model_path.resolve(strict=True)
            resolved_model.relative_to(run_root.resolve(strict=True))
            payload = json.loads(resolved_model.read_text(encoding="utf-8"))
            artifact = TaskModelArtifact.model_validate({
                name: payload[name]
                for name in TaskModelArtifact.model_fields
                if name in payload
            })
        except Exception as exc:
            raise TaskModelError("Safe model metadata is unavailable or invalid.") from exc
        if artifact.run_id != expected_run_id:
            raise TaskModelError("Cross-run model artifact reuse is not allowed.")
        if artifact.backend != "reference_image_linear" or artifact.backend_version != BACKEND_VERSION:
            raise TaskModelError("Reference model backend/version mismatch.")
        if artifact.feature_version != FEATURE_VERSION or artifact.weights_reference is None:
            raise TaskModelError("Reference model feature/checkpoint metadata mismatch.")
        weights_path = (run_root / artifact.weights_reference).resolve(strict=True)
        try:
            weights_path.relative_to(run_root.resolve(strict=True))
        except ValueError as exc:
            raise TaskModelError("Weights escaped the approved run root.") from exc
        if _sha256(weights_path) != payload.get("weights_sha256"):
            raise TaskModelError("Safe model weights hash mismatch.")
        expected_model_hash = _canonical_hash({
            "backend_version": BACKEND_VERSION,
            "feature_version": FEATURE_VERSION,
            "training_fingerprint": artifact.training_fingerprint,
            "weights_sha256": payload.get("weights_sha256"),
        })
        if artifact.model_hash != expected_model_hash:
            raise TaskModelError("Safe model identity hash mismatch.")
        try:
            with np.load(weights_path, allow_pickle=False) as values:
                arrays = {name: np.asarray(values[name]) for name in (
                    "classes", "coef", "intercept", "feature_mean", "feature_scale"
                )}
        except Exception as exc:
            raise TaskModelError("Safe NPZ model weights are invalid.") from exc
        if arrays["classes"].ndim != 1 or len(arrays["classes"]) < 2:
            raise TaskModelError("Safe model classes are invalid.")
        feature_count = FEATURE_SIZE[0] * FEATURE_SIZE[1] * 3
        if arrays["coef"].ndim != 2 or arrays["coef"].shape[1] != feature_count:
            raise TaskModelError("Safe model coefficient shape is invalid.")
        expected_rows = 1 if len(arrays["classes"]) == 2 else len(arrays["classes"])
        if arrays["coef"].shape[0] != expected_rows or arrays["intercept"].shape != (expected_rows,):
            raise TaskModelError("Safe model class/weight shapes are inconsistent.")
        if arrays["feature_mean"].shape != (feature_count,) or arrays["feature_scale"].shape != (feature_count,):
            raise TaskModelError("Safe model feature scaling shape is invalid.")
        numeric = [arrays[name] for name in ("coef", "intercept", "feature_mean", "feature_scale")]
        if any(not np.all(np.isfinite(value)) for value in numeric) or np.any(arrays["feature_scale"] <= 0):
            raise TaskModelError("Safe model weights contain invalid numeric values.")
        model = cls(artifact.model_id)
        model.classes_ = arrays["classes"].astype(str)
        model.coef_ = arrays["coef"].astype(np.float64)
        model.intercept_ = arrays["intercept"].astype(np.float64)
        model.feature_mean_ = arrays["feature_mean"].astype(np.float64)
        model.feature_scale_ = arrays["feature_scale"].astype(np.float64)
        model.converged_ = bool(payload.get("converged"))
        return model, artifact


__all__ = [
    "BACKEND_VERSION", "FEATURE_SIZE", "FEATURE_VERSION",
    "ReferenceImageLinearClassifier", "TaskModelError",
]
