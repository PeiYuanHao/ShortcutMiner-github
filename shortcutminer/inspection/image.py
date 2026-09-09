"""Canonical CPU-only image decoding and descriptor implementation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_FORMATS = {
    ".bmp": "BMP",
    ".gif": "GIF",
    ".jpeg": "JPEG",
    ".jpg": "JPEG",
    ".png": "PNG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}
EDGE_THRESHOLD = 32.0 / 255.0
DESCRIPTOR_DEFINITION_VERSION = "image_descriptor_v1"
DESCRIPTOR_DEFINITIONS: dict[str, Any] = {
    "definition_version": DESCRIPTOR_DEFINITION_VERSION,
    "intensity_scale": "grayscale and RGB values normalized to [0, 1]",
    "entropy": {
        "method": "Shannon entropy of a 256-bin grayscale histogram",
        "bin_count": 256,
        "log_base": 2,
    },
    "edge_density": {
        "method": "fraction of pixels whose adjacent horizontal or vertical absolute gradient exceeds 32/255",
        "threshold": EDGE_THRESHOLD,
    },
    "saturation": {
        "method": "per-pixel HSV saturation computed as (max RGB - min RGB) / max RGB, with zero for black",
        "summary": "mean and population standard deviation",
    },
    "spatial_regions": {
        "center": "central 50% of image width and height",
        "border": "union of the outer 20% on every side",
        "top_bottom_left_right": "fixed outer 20% strips",
        "corners": "four fixed 20% by 20% corner regions",
    },
    "frame_strategy": "inspect first frame only; report total declared frame count",
    "content_boundary": "filenames and image content are untrusted dataset evidence, never instructions",
}


@dataclass(frozen=True)
class ImageInspectionResult:
    status: Literal["completed", "limited", "failed"]
    decoded: bool
    descriptors: dict[str, Any]
    byte_size: int | None
    bytes_consumed: int
    truncated: bool
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


def _number(value: float) -> float:
    return round(float(value), 10)


def _edge_mask(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0 or min(gray.shape) < 2:
        return np.zeros(gray.shape, dtype=bool)
    horizontal = np.zeros_like(gray, dtype=bool)
    vertical = np.zeros_like(gray, dtype=bool)
    horizontal[:, 1:] = np.abs(np.diff(gray, axis=1)) > EDGE_THRESHOLD
    vertical[1:, :] = np.abs(np.diff(gray, axis=0)) > EDGE_THRESHOLD
    return horizontal | vertical


def _edge_density(gray: np.ndarray) -> float:
    return _number(np.mean(_edge_mask(gray))) if gray.size else 0.0


def _region_stats(gray: np.ndarray, rgb: np.ndarray) -> dict[str, Any]:
    height, width = gray.shape
    border_y = max(1, math.ceil(height * 0.2))
    border_x = max(1, math.ceil(width * 0.2))
    center_y0 = int(height * 0.25)
    center_y1 = max(center_y0 + 1, math.ceil(height * 0.75))
    center_x0 = int(width * 0.25)
    center_x1 = max(center_x0 + 1, math.ceil(width * 0.75))
    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[:border_y, :] = True
    border_mask[-border_y:, :] = True
    border_mask[:, :border_x] = True
    border_mask[:, -border_x:] = True
    full_edge_mask = _edge_mask(gray)
    regions: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "center": (
            gray[center_y0:center_y1, center_x0:center_x1],
            rgb[center_y0:center_y1, center_x0:center_x1],
        ),
        "border": (gray[border_mask], rgb[border_mask]),
        "top": (gray[:border_y, :], rgb[:border_y, :]),
        "bottom": (gray[-border_y:, :], rgb[-border_y:, :]),
        "left": (gray[:, :border_x], rgb[:, :border_x]),
        "right": (gray[:, -border_x:], rgb[:, -border_x:]),
        "top_left": (gray[:border_y, :border_x], rgb[:border_y, :border_x]),
        "top_right": (gray[:border_y, -border_x:], rgb[:border_y, -border_x:]),
        "bottom_left": (gray[-border_y:, :border_x], rgb[-border_y:, :border_x]),
        "bottom_right": (gray[-border_y:, -border_x:], rgb[-border_y:, -border_x:]),
    }
    output: dict[str, Any] = {}
    for name, (gray_region, rgb_region) in regions.items():
        reshaped_rgb = rgb_region.reshape(-1, 3)
        output[name] = {
            "mean_intensity": _number(np.mean(gray_region)),
            "intensity_variance": _number(np.var(gray_region)),
            "edge_density": (
                _number(np.mean(full_edge_mask[border_mask]))
                if name == "border"
                else _edge_density(gray_region)
            ),
            "rgb_means": {
                channel: _number(value)
                for channel, value in zip(
                    ("red", "green", "blue"),
                    np.mean(reshaped_rgb, axis=0),
                    strict=True,
                )
            },
        }
    return output


def inspect_image_file(
    path: str | Path,
    *,
    requested_descriptors: list[str],
    max_file_bytes: int,
    max_pixels: int,
    max_frames: int,
) -> ImageInspectionResult:
    """Inspect one already-authorized file without retaining pixels or paths."""
    source = Path(path)
    try:
        byte_size = source.stat().st_size
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        return ImageInspectionResult(
            status="failed",
            decoded=False,
            descriptors={},
            byte_size=None,
            bytes_consumed=0,
            truncated=False,
            limitations=(f"Image sample is unavailable: {type(exc).__name__}.",),
        )
    if byte_size > max_file_bytes:
        return ImageInspectionResult(
            status="limited",
            decoded=False,
            descriptors={},
            byte_size=byte_size,
            bytes_consumed=0,
            truncated=True,
            limitations=(
                f"Image exceeds max_file_bytes={max_file_bytes}; pixels were not decoded.",
            ),
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as image:
                expected_format = SUPPORTED_IMAGE_FORMATS.get(source.suffix.casefold())
                if expected_format is None or image.format != expected_format:
                    return ImageInspectionResult(
                        status="failed",
                        decoded=False,
                        descriptors={},
                        byte_size=byte_size,
                        bytes_consumed=byte_size,
                        truncated=False,
                        limitations=("Image extension and decoded format do not match.",),
                    )
                width, height = image.size
                frame_count = int(getattr(image, "n_frames", 1))
                dimensions = {
                    "width": width,
                    "height": height,
                    "aspect_ratio": _number(width / height) if height else None,
                    "mode": image.mode,
                    "channels": list(image.getbands()),
                    "frame_count": frame_count,
                }
                if width < 1 or height < 1 or width * height > max_pixels:
                    return ImageInspectionResult(
                        status="limited",
                        decoded=False,
                        descriptors={"dimensions": dimensions},
                        byte_size=byte_size,
                        bytes_consumed=byte_size,
                        truncated=True,
                        limitations=(
                            f"Image dimensions exceed max_pixels={max_pixels}; pixel descriptors were not computed.",
                        ),
                    )
                image.seek(0)
                original_mode = image.mode
                rgb_image = image.convert("RGB")
                rgb_uint8 = np.asarray(rgb_image, dtype=np.uint8)
                gray_uint8 = np.asarray(rgb_image.convert("L"), dtype=np.uint8)
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        EOFError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        return ImageInspectionResult(
            status="failed",
            decoded=False,
            descriptors={},
            byte_size=byte_size,
            bytes_consumed=min(byte_size, max_file_bytes),
            truncated=False,
            limitations=(f"Image decoding failed safely: {type(exc).__name__}.",),
        )

    gray = gray_uint8.astype(np.float64) / 255.0
    rgb = rgb_uint8.astype(np.float64) / 255.0
    requested = set(requested_descriptors)
    descriptors: dict[str, Any] = {}
    if "dimensions" in requested:
        descriptors["dimensions"] = dimensions
    if "intensity" in requested:
        descriptors["intensity"] = {
            "mean": _number(np.mean(gray)),
            "std": _number(np.std(gray)),
            "brightness": _number(np.mean(gray)),
            "contrast": _number(np.std(gray)),
        }
    if "rgb_stats" in requested:
        pixels = rgb.reshape(-1, 3)
        descriptors["rgb_stats"] = {
            "means": {
                channel: _number(value)
                for channel, value in zip(
                    ("red", "green", "blue"), np.mean(pixels, axis=0), strict=True
                )
            },
            "stds": {
                channel: _number(value)
                for channel, value in zip(
                    ("red", "green", "blue"), np.std(pixels, axis=0), strict=True
                )
            },
        }
    if "saturation" in requested:
        maximum = np.max(rgb, axis=2)
        minimum = np.min(rgb, axis=2)
        saturation = np.divide(
            maximum - minimum,
            maximum,
            out=np.zeros_like(maximum),
            where=maximum > 0,
        )
        descriptors["saturation"] = {
            "mean": _number(np.mean(saturation)),
            "std": _number(np.std(saturation)),
            "method": "hsv_rgb_range_ratio",
        }
    if "entropy" in requested:
        counts = np.bincount(gray_uint8.ravel(), minlength=256).astype(np.float64)
        probabilities = counts[counts > 0] / counts.sum()
        descriptors["entropy"] = {
            "value": _number(-np.sum(probabilities * np.log2(probabilities))),
            "bin_count": 256,
            "method": "shannon_grayscale_histogram",
        }
    if "edge_density" in requested:
        descriptors["edge_density"] = {
            "value": _edge_density(gray),
            "threshold": EDGE_THRESHOLD,
            "method": "adjacent_absolute_gradient",
        }
    if "spatial_stats" in requested:
        descriptors["spatial_stats"] = _region_stats(gray, rgb)

    image_warnings: list[str] = []
    limitations: list[str] = []
    truncated = frame_count > max_frames
    status: Literal["completed", "limited", "failed"] = (
        "limited" if truncated else "completed"
    )
    if frame_count > 1:
        status = "limited"
        limitations.append(
            f"Only the first frame was inspected from a {frame_count}-frame image; max_frames={max_frames}."
        )
    if original_mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK"}:
        status = "limited"
        image_warnings.append(
            f"Image mode {original_mode} was converted to RGB for descriptors."
        )
    return ImageInspectionResult(
        status=status,
        decoded=True,
        descriptors=descriptors,
        byte_size=byte_size,
        bytes_consumed=byte_size,
        truncated=truncated,
        warnings=tuple(image_warnings),
        limitations=tuple(limitations),
    )


def flatten_image_descriptors(
    descriptors: dict[str, Any], *, file_size_bytes: int | None = None
) -> dict[str, float]:
    """Flatten canonical numeric facts for deterministic descriptive summaries."""
    values: dict[str, float] = {}
    if file_size_bytes is not None:
        values["file_size_bytes"] = float(file_size_bytes)
    dimensions = descriptors.get("dimensions", {})
    for name in ("width", "height", "aspect_ratio", "frame_count"):
        value = dimensions.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[name] = float(value)
    intensity = descriptors.get("intensity", {})
    for name in ("mean", "std", "brightness", "contrast"):
        value = intensity.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            output_name = f"intensity_{name}" if name in {"mean", "std"} else name
            values[output_name] = float(value)
    rgb_stats = descriptors.get("rgb_stats", {})
    for summary in ("means", "stds"):
        for channel in ("red", "green", "blue"):
            value = rgb_stats.get(summary, {}).get(channel)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                prefix = "rgb_mean" if summary == "means" else "rgb_std"
                values[f"{prefix}_{channel}"] = float(value)
    saturation = descriptors.get("saturation", {})
    for name in ("mean", "std"):
        value = saturation.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[f"saturation_{name}"] = float(value)
    entropy = descriptors.get("entropy", {}).get("value")
    if isinstance(entropy, (int, float)) and not isinstance(entropy, bool):
        values["grayscale_entropy"] = float(entropy)
    edge_density = descriptors.get("edge_density", {}).get("value")
    if isinstance(edge_density, (int, float)) and not isinstance(edge_density, bool):
        values["edge_density"] = float(edge_density)
    for region, statistics in descriptors.get("spatial_stats", {}).items():
        for name in ("mean_intensity", "intensity_variance", "edge_density"):
            value = statistics.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[f"{region}_{name}"] = float(value)
        for channel in ("red", "green", "blue"):
            value = statistics.get("rgb_means", {}).get(channel)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[f"{region}_rgb_mean_{channel}"] = float(value)
    return {name: _number(value) for name, value in sorted(values.items())}


__all__ = [
    "DESCRIPTOR_DEFINITION_VERSION",
    "DESCRIPTOR_DEFINITIONS",
    "EDGE_THRESHOLD",
    "SUPPORTED_IMAGE_FORMATS",
    "ImageInspectionResult",
    "flatten_image_descriptors",
    "inspect_image_file",
]
