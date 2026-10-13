"""Static task-model allowlist. No module path or user entrypoint is accepted."""

from shortcutminer.task_models.reference import ReferenceImageLinearClassifier
from shortcutminer.task_models.torchvision_matrix import TorchvisionResNetMatrix


TASK_MODEL_REGISTRY = {
    "reference_image_linear": ReferenceImageLinearClassifier,
    "torchvision_resnet_matrix": TorchvisionResNetMatrix,
}


__all__ = ["TASK_MODEL_REGISTRY"]
