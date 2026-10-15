"""CPU-only allowlisted image interventions with bounded artifact output."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from shortcutminer.inspection.image import load_rgb_image_file
from shortcutminer.intervention.protocol import (
    BackgroundReplaceTransform,
    BlurTransform,
    CompressionTransform,
    FactorTransform,
    LuminanceTransform,
    ChannelRatioTransform,
    RotationTransform,
    FrequencyTransform,
    GlobalColorTransform,
    ImageInterventionSpec,
    InterventionQuality,
    ResolutionTransform,
    SpatialRelocateTransform,
    SpatialTransform,
    CornerPatchTransform,
    ProvidedPairTransform,
)
from shortcutminer.intervention.segmentation import SegmentationProvider


class InterventionError(ValueError):
    pass


_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def _clip(array: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def _box(spec, width: int, height: int) -> tuple[int, int, int, int]:
    if getattr(spec, "region", None) in {"custom", "custom_exterior"}:
        value = spec.custom_box
        assert value is not None
        return (
            int(round(value.x0 * width)), int(round(value.y0 * height)),
            max(1, int(round(value.x1 * width))), max(1, int(round(value.y1 * height))),
        )
    fractions = {
        "top_left": (0, 0, .25, .25), "top_right": (.75, 0, 1, .25),
        "bottom_left": (0, .75, .25, 1), "bottom_right": (.75, .75, 1, 1),
        "center": (.25, .25, .75, .75),
    }
    x0, y0, x1, y1 = fractions[spec.region]
    return int(x0 * width), int(y0 * height), max(1, int(x1 * width)), max(1, int(y1 * height))


def _region_mask(spec: SpatialTransform, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if spec.region == "border":
        border_x, border_y = max(1, int(round(width * .2))), max(1, int(round(height * .2)))
        mask[:border_y, :] = True
        mask[-border_y:, :] = True
        mask[:, :border_x] = True
        mask[:, -border_x:] = True
        return mask
    x0, y0, x1, y1 = _box(spec, width, height)
    mask[y0:y1, x0:x1] = True
    if spec.region == 'custom_exterior':
        mask = ~mask
    return mask


def _fill(array: np.ndarray, mask: np.ndarray, spec: SpatialTransform) -> np.ndarray:
    output = array.copy()
    if spec.fill == "constant":
        output[mask] = np.asarray(spec.fill_color, dtype=np.uint8)
    elif spec.fill == "mean":
        output[mask] = np.rint(array.reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    elif spec.fill == 'color_complement':
        output[mask] = 255 - array[mask]
    else:
        rng = np.random.default_rng(spec.seed)
        values = rng.normal(127.5, spec.noise_std, size=(int(mask.sum()), 3))
        output[mask] = _clip(values)
    return output


def _transform_basic(array: np.ndarray, spec: ImageInterventionSpec) -> np.ndarray:
    if spec.type == 'bounded_channel_ratio':
        from shortcutminer.intervention.bounded_color import transform
        return transform(array,spec)
    if isinstance(spec, ChannelRatioTransform):
        from shortcutminer.intervention.channel_ratio import transform
        return transform(array,spec)
    if isinstance(spec, CornerPatchTransform):
        from shortcutminer.intervention.corner_patch import transform
        return transform(array, spec)
    image = Image.fromarray(array, mode="RGB")
    if isinstance(spec, LuminanceTransform):
        channels = np.asarray(image.convert('YCbCr')).copy()
        y = channels[:,:,0].astype(np.float64)
        mean = float(y.mean())
        desired = spec.target_mean if spec.mode == 'normalize_mean' else (
            2 * spec.pivot_mean - mean if spec.mode == 'reflect_mean' else mean + spec.offset)
        channels[:,:,0] = _clip(y + desired - mean)
        return np.asarray(Image.frombytes('YCbCr', image.size, channels.tobytes()).convert('RGB'))
    if isinstance(spec, RotationTransform):
        height,width = array.shape[:2]
        pad = max(height,width)
        padded = np.pad(array, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
        rotated = Image.fromarray(padded).rotate(spec.angle_degrees, Image.Resampling.BICUBIC)
        return np.asarray(rotated.crop((pad,pad,pad+width,pad+height)))
    if isinstance(spec, FrequencyTransform):
        height,width = array.shape[:2]
        radius = np.hypot(np.fft.fftfreq(height)[:,None], np.fft.fftfreq(width)[None,:])
        mask = radius <= spec.cutoff
        if spec.mode == 'high_pass': mask = ~mask
        spectrum = np.fft.fft2(array.astype(float), axes=(0,1))
        filtered = np.fft.ifft2(spectrum * mask[:,:,None], axes=(0,1)).real
        if spec.mode == 'high_pass': filtered += array.mean(axis=(0,1),keepdims=True)
        return _clip(filtered)
    if isinstance(spec, GlobalColorTransform):
        if spec.mode == "channel_scale":
            return _clip(array.astype(np.float64) * np.asarray(spec.channel_scale))
        if spec.mode == "swap_red_blue":
            return array[:, :, [2, 1, 0]].copy()
        if spec.mode == "swap_red_green":
            return array[:, :, [1, 0, 2]].copy()
        if spec.mode == "cycle_rgb":
            return array[:, :, [2, 0, 1]].copy()
        if spec.mode == "grayscale_max":
            return np.repeat(array.max(axis=2, keepdims=True), 3, axis=2)
        if spec.mode == "grayscale":
            return np.asarray(image.convert("L").convert("RGB"), dtype=np.uint8)
        if spec.mode == "desaturate":
            gray = np.asarray(image.convert("L").convert("RGB"), dtype=np.float64)
            return _clip(array * spec.desaturation_factor + gray * (1 - spec.desaturation_factor))
        distance = np.max(np.abs(array.astype(int) - np.asarray(spec.source_color)), axis=2)
        output = array.copy()
        output[distance <= spec.color_tolerance] = np.asarray(spec.target_color, dtype=np.uint8)
        return output
    if isinstance(spec, FactorTransform):
        transformed = (
            ImageEnhance.Brightness(image).enhance(spec.factor)
            if spec.type == "brightness"
            else ImageEnhance.Contrast(image).enhance(spec.factor)
        )
        return np.asarray(transformed, dtype=np.uint8)
    if isinstance(spec, BlurTransform):
        return np.asarray(image.filter(ImageFilter.GaussianBlur(spec.radius)), dtype=np.uint8)
    if spec.type == "noise":
        rng = np.random.default_rng(spec.seed)
        return _clip(array.astype(np.float64) + rng.normal(0, spec.std, array.shape))
    if isinstance(spec, CompressionTransform):
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=spec.quality, optimize=False, progressive=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    if isinstance(spec, ResolutionTransform):
        down = image.resize((spec.width, spec.height), _RESAMPLING[spec.resampling])
        return np.asarray(down.resize(image.size, _RESAMPLING[spec.resampling]), dtype=np.uint8)
    if isinstance(spec, SpatialTransform):
        return _fill(array, _region_mask(spec, array.shape[1], array.shape[0]), spec)
    if isinstance(spec, SpatialRelocateTransform):
        output = array.copy()
        source = _absolute_box(spec.source_box, array.shape[1], array.shape[0])
        target = _absolute_box(spec.target_box, array.shape[1], array.shape[0])
        sx0, sy0, sx1, sy1 = source
        tx0, ty0, tx1, ty1 = target
        patch = Image.fromarray(array[sy0:sy1, sx0:sx1], mode="RGB").resize(
            (tx1 - tx0, ty1 - ty0), Image.Resampling.BILINEAR
        )
        if spec.source_fill == "constant":
            output[sy0:sy1, sx0:sx1] = np.asarray(spec.fill_color, dtype=np.uint8)
        else:
            output[sy0:sy1, sx0:sx1] = np.rint(array.reshape(-1, 3).mean(axis=0)).astype(np.uint8)
        output[ty0:ty1, tx0:tx1] = np.asarray(patch, dtype=np.uint8)
        return output
    raise InterventionError(f"Unsupported intervention type: {spec.type}")


def _absolute_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0 = int(round(box.x0 * width)), int(round(box.y0 * height))
    x1, y1 = int(round(box.x1 * width)), int(round(box.y1 * height))
    return min(x0, width - 1), min(y0, height - 1), max(x0 + 1, min(x1, width)), max(y0 + 1, min(y1, height))


class InterventionEngine:
    engine_version = "image_intervention_v1"

    def __init__(
        self,
        *,
        segmentation_provider: SegmentationProvider | None = None,
        max_file_bytes: int = 25_000_000,
        max_pixels: int = 20_000_000,
        max_output_bytes: int = 25_000_000,
    ) -> None:
        self.segmentation_provider = segmentation_provider
        self.max_file_bytes = max_file_bytes
        self.max_pixels = max_pixels
        self.max_output_bytes = max_output_bytes

    def apply(
        self,
        *,
        source_path: Path,
        output_path: Path,
        transformation: ImageInterventionSpec,
        mask_path: Path | None = None,
        replacement_path: Path | None = None,
        dataset_root: Path | None = None,
    ) -> InterventionQuality:
        decoded = load_rgb_image_file(
            source_path,
            max_file_bytes=self.max_file_bytes,
            max_pixels=self.max_pixels,
        )
        original = decoded.rgb
        mask = None
        if isinstance(transformation, ProvidedPairTransform):
            from shortcutminer.intervention.provided_pairs import resolve_pair
            transformed, _ = resolve_pair(dataset_root, source_path, transformation)
        elif isinstance(transformation, BackgroundReplaceTransform):
            if self.segmentation_provider is None or not self.segmentation_provider.available:
                raise InterventionError("Background replacement requires available segmentation.")
            segmented = self.segmentation_provider.segment(
                image_path=source_path,
                mask_path=mask_path,
                max_file_bytes=self.max_file_bytes,
                max_pixels=self.max_pixels,
            )
            mask = segmented.mask
            if transformation.replacement_mode == "constant":
                replacement = np.empty_like(original)
                replacement[:] = np.asarray(transformation.fill_color, dtype=np.uint8)
            elif transformation.replacement_mode == "blurred":
                replacement = np.asarray(
                    Image.fromarray(original).filter(ImageFilter.GaussianBlur(transformation.blur_radius)),
                    dtype=np.uint8,
                )
            else:
                if replacement_path is None:
                    raise InterventionError("Sample background replacement needs an approved source.")
                other = load_rgb_image_file(
                    replacement_path,
                    max_file_bytes=self.max_file_bytes,
                    max_pixels=self.max_pixels,
                )
                replacement = np.asarray(
                    Image.fromarray(other.rgb).resize(
                        (decoded.width, decoded.height), Image.Resampling.BILINEAR
                    ), dtype=np.uint8,
                )
            transformed = np.where(mask[:, :, None], original, replacement)
        else:
            transformed = _transform_basic(original, transformation)

        if transformed.shape != original.shape:
            raise InterventionError("Intervention changed output dimensions unexpectedly.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = ".jpg" if isinstance(transformation, CompressionTransform) else ".png"
        if output_path.suffix.casefold() != suffix:
            raise InterventionError(f"Intervention output must use {suffix} format.")
        temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp{suffix}")
        try:
            Image.fromarray(transformed, mode="RGB").save(
                temporary,
                format="JPEG" if suffix == ".jpg" else "PNG",
                **({"quality": transformation.quality} if isinstance(transformation, CompressionTransform) else {}),
            )
            if temporary.stat().st_size > self.max_output_bytes:
                raise InterventionError("Intervention output exceeds max_output_bytes.")
            os.replace(temporary, output_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        persisted = load_rgb_image_file(
            output_path,
            max_file_bytes=self.max_output_bytes,
            max_pixels=self.max_pixels,
        )
        changed = np.any(original != persisted.rgb, axis=2)
        foreground_change = background_change = coverage = None
        if mask is not None:
            coverage = float(np.mean(mask))
            foreground_change = float(np.mean(changed[mask])) if np.any(mask) else 0.0
            background_change = float(np.mean(changed[~mask])) if np.any(~mask) else 0.0
        return InterventionQuality(
            decode_success=True,
            same_dimensions=(persisted.width, persisted.height) == (decoded.width, decoded.height),
            mask_coverage=coverage,
            pixel_change_fraction=float(np.mean(changed)),
            foreground_change_fraction=foreground_change,
            background_change_fraction=background_change,
            output_width=persisted.width,
            output_height=persisted.height,
            output_bytes=persisted.byte_size,
            output_format="JPEG" if suffix == ".jpg" else "PNG",
        )


__all__ = ["InterventionEngine", "InterventionError"]
