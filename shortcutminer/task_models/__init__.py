"""Allowlisted task-model adapters used to audit model behavior."""

from shortcutminer.task_models.base import ExternalTaskModelAdapter, TaskModelAdapter
from shortcutminer.task_models.prediction_artifact import PredictionArtifactAdapter
from shortcutminer.task_models.reference import ReferenceImageLinearClassifier
from shortcutminer.task_models.registry import TASK_MODEL_REGISTRY

__all__ = [
    "ExternalTaskModelAdapter",
    "PredictionArtifactAdapter",
    "ReferenceImageLinearClassifier",
    "TASK_MODEL_REGISTRY",
    "TaskModelAdapter",
]
