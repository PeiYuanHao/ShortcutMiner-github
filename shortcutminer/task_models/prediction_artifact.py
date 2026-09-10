"""Prediction-only adapter that never claims retraining or intervention inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shortcutminer.protocol.model_reliance import TaskModelCapabilities
from shortcutminer.task_models.reference import TaskModelError


class PredictionArtifactAdapter:
    model_kind = "prediction_artifact"
    capabilities = TaskModelCapabilities(
        predictions=True,
        probabilities=False,
        logits=False,
        training=False,
        checkpoint=False,
        image_input=False,
        intervention_inference=False,
    )

    def __init__(self, model_id: str, predictions: dict[str, str]) -> None:
        self.model_id = model_id
        self._predictions = dict(predictions)

    def fit(self, *_args, **_kwargs) -> None:
        raise TaskModelError("Prediction-only artifacts cannot be retrained.")

    def predict(self, paths: list[Path]) -> list[str]:
        try:
            return [self._predictions[path.as_posix()] for path in paths]
        except KeyError as exc:
            raise TaskModelError("Prediction artifact has no row for the requested sample.") from exc

    def predict_proba(self, _paths: list[Path]) -> np.ndarray:
        raise TaskModelError("Prediction artifact has no probability capability.")

    def predict_logits(self, _paths: list[Path]) -> np.ndarray:
        raise TaskModelError("Prediction artifact has no logits capability.")


__all__ = ["PredictionArtifactAdapter"]
