"""Static task-model allowlist. No module path or user entrypoint is accepted."""

from shortcutminer.task_models.reference import ReferenceImageLinearClassifier


TASK_MODEL_REGISTRY = {
    "reference_image_linear": ReferenceImageLinearClassifier,
}


__all__ = ["TASK_MODEL_REGISTRY"]
