"""Bounded segmentation-provider contracts; no shortcut detection occurs here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from shortcutminer.inspection.image import load_rgb_image_file


class SegmentationError(ValueError):
    pass


@dataclass(frozen=True)
class SegmentationMask:
    mask: np.ndarray
    coverage: float
    provider: str
    version: str


class SegmentationProvider(Protocol):
    provider_name: str
    provider_version: str
    available: bool

    def segment(
        self,
        *,
        image_path: Path,
        mask_path: Path | None,
        max_file_bytes: int,
        max_pixels: int,
    ) -> SegmentationMask: ...


class ProvidedMaskSegmentationProvider:
    provider_name = "provided_mask"
    provider_version = "provided_mask_v1"
    available = True

    def segment(
        self,
        *,
        image_path: Path,
        mask_path: Path | None,
        max_file_bytes: int,
        max_pixels: int,
    ) -> SegmentationMask:
        if mask_path is None:
            raise SegmentationError("Provided-mask segmentation requires a mask.")
        image = load_rgb_image_file(
            image_path, max_file_bytes=max_file_bytes, max_pixels=max_pixels
        )
        decoded = load_rgb_image_file(
            mask_path, max_file_bytes=max_file_bytes, max_pixels=max_pixels
        )
        if (decoded.width, decoded.height) != (image.width, image.height):
            raise SegmentationError("Provided mask dimensions do not match the image.")
        channel = decoded.rgb[:, :, 0]
        if not np.array_equal(decoded.rgb[:, :, 0], decoded.rgb[:, :, 1]) or not np.array_equal(
            decoded.rgb[:, :, 0], decoded.rgb[:, :, 2]
        ):
            raise SegmentationError("Provided mask must be grayscale/binary.")
        values = set(int(value) for value in np.unique(channel))
        if not values.issubset({0, 255}) or not values:
            raise SegmentationError("Provided mask values must be binary 0/255.")
        mask = channel == 255
        return SegmentationMask(
            mask=mask,
            coverage=float(np.mean(mask)),
            provider=self.provider_name,
            version=self.provider_version,
        )


class SAMCompatibleSegmentationProvider:
    """Future local-only adapter boundary; it never installs or downloads SAM."""

    provider_name = "sam_compatible_local"
    provider_version = "contract_v1"

    def __init__(self, *, checkpoint: Path | None = None) -> None:
        self.checkpoint = checkpoint
        self.available = False

    def segment(self, **_kwargs) -> SegmentationMask:
        raise SegmentationError(
            "SAM-compatible segmentation is unavailable without an explicit allowlisted local implementation and checkpoint."
        )


__all__ = [
    "ProvidedMaskSegmentationProvider",
    "SAMCompatibleSegmentationProvider",
    "SegmentationError",
    "SegmentationMask",
    "SegmentationProvider",
]
