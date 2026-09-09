"""Shared, bounded inspection infrastructure used by production stages."""

from shortcutminer.inspection.image import (
    DESCRIPTOR_DEFINITION_VERSION,
    DESCRIPTOR_DEFINITIONS,
    SUPPORTED_IMAGE_FORMATS,
    ImageInspectionResult,
    flatten_image_descriptors,
    inspect_image_file,
)

__all__ = [
    "DESCRIPTOR_DEFINITION_VERSION",
    "DESCRIPTOR_DEFINITIONS",
    "SUPPORTED_IMAGE_FORMATS",
    "ImageInspectionResult",
    "flatten_image_descriptors",
    "inspect_image_file",
]
