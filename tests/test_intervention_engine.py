from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from shortcutminer.intervention.engine import InterventionEngine, InterventionError
from shortcutminer.intervention.protocol import (
    BackgroundReplaceTransform,
    BlurTransform,
    CompressionTransform,
    FactorTransform,
    GlobalColorTransform,
    NoiseTransform,
    NormalizedBox,
    ResolutionTransform,
    SpatialRelocateTransform,
    SpatialTransform,
)
from shortcutminer.intervention.segmentation import (
    ProvidedMaskSegmentationProvider,
    SAMCompatibleSegmentationProvider,
    SegmentationError,
)


def _source(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    array = np.zeros((20, 24, 3), dtype=np.uint8)
    array[:] = (30, 90, 180)
    array[6:14, 8:16] = (240, 240, 240)
    path = root / "source.png"
    Image.fromarray(array).save(path)
    return path, array


@pytest.mark.parametrize(
    "spec",
    [
        GlobalColorTransform(type="global_color", mode="channel_scale", channel_scale=[1, .5, 1]),
        GlobalColorTransform(type="global_color", mode="swap_red_blue"),
        GlobalColorTransform(type="global_color", mode="grayscale"),
        GlobalColorTransform(type="global_color", mode="desaturate", desaturation_factor=.3),
        GlobalColorTransform(type="global_color", mode="color_replace", source_color=[30, 90, 180], target_color=[10, 20, 30]),
        FactorTransform(type="brightness", factor=.7),
        FactorTransform(type="contrast", factor=1.4),
        BlurTransform(type="blur", radius=2),
        NoiseTransform(type="noise", std=10, seed=23),
        CompressionTransform(type="compression", quality=50),
        ResolutionTransform(type="resolution", width=6, height=5, resampling="bilinear"),
        SpatialTransform(type="spatial_mask", region="top_left", fill="constant", fill_color=[0, 0, 0]),
        SpatialTransform(type="spatial_replace", region="center", fill="mean"),
        SpatialTransform(type="spatial_replace", region="custom", custom_box=NormalizedBox(x0=.1, y0=.1, x1=.4, y1=.4), fill="noise", seed=5),
        SpatialRelocateTransform(
            type="spatial_relocate",
            source_box=NormalizedBox(x0=0, y0=0, x1=.25, y1=.25),
            target_box=NormalizedBox(x0=.75, y0=.75, x1=1, y1=1),
        ),
    ],
)
def test_allowlisted_transformations_are_bounded_and_deterministic(tmp_path: Path, spec):
    source, _ = _source(tmp_path / "input")
    suffix = ".jpg" if spec.type == "compression" else ".png"
    first = tmp_path / f"first{suffix}"
    second = tmp_path / f"second{suffix}"
    engine = InterventionEngine()
    quality = engine.apply(source_path=source, output_path=first, transformation=spec)
    repeated = engine.apply(source_path=source, output_path=second, transformation=spec)
    assert first.read_bytes() == second.read_bytes()
    assert quality.same_dimensions and quality.output_width == 24 and quality.output_height == 20
    assert quality.output_bytes <= 25_000_000
    assert repeated.pixel_change_fraction == quality.pixel_change_fraction


def test_provided_mask_background_replacement_preserves_foreground(tmp_path: Path):
    source, original = _source(tmp_path / "input")
    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[6:14, 8:16] = 255
    mask_path = tmp_path / "input" / "mask.png"
    Image.fromarray(mask).save(mask_path)
    engine = InterventionEngine(segmentation_provider=ProvidedMaskSegmentationProvider())
    output = tmp_path / "background.png"
    quality = engine.apply(
        source_path=source, output_path=output,
        transformation=BackgroundReplaceTransform(
            type="background_replace", mask_column="mask", replacement_mode="constant",
            fill_color=[0, 180, 0],
        ),
        mask_path=mask_path,
    )
    changed = np.asarray(Image.open(output).convert("RGB"))
    assert np.array_equal(changed[6:14, 8:16], original[6:14, 8:16])
    assert quality.mask_coverage == pytest.approx(64 / 480)
    assert quality.foreground_change_fraction == 0
    assert quality.background_change_fraction > 0.99


def test_provided_mask_rejects_wrong_dimensions_and_nonbinary_values(tmp_path: Path):
    source, _ = _source(tmp_path / "input")
    provider = ProvidedMaskSegmentationProvider()
    wrong = tmp_path / "input" / "wrong.png"
    Image.new("L", (4, 4), 255).save(wrong)
    with pytest.raises(SegmentationError, match="dimensions"):
        provider.segment(image_path=source, mask_path=wrong, max_file_bytes=100_000, max_pixels=10_000)
    invalid = tmp_path / "input" / "invalid.png"
    Image.new("L", (24, 20), 128).save(invalid)
    with pytest.raises(SegmentationError, match="binary"):
        provider.segment(image_path=source, mask_path=invalid, max_file_bytes=100_000, max_pixels=10_000)


def test_background_replacement_requires_segmentation(tmp_path: Path):
    source, _ = _source(tmp_path / "input")
    with pytest.raises(InterventionError, match="segmentation"):
        InterventionEngine().apply(
            source_path=source, output_path=tmp_path / "out.png",
            transformation=BackgroundReplaceTransform(
                type="background_replace", mask_column="mask", replacement_mode="constant"
            ),
        )


def test_sam_compatible_adapter_is_explicitly_unavailable():
    adapter = SAMCompatibleSegmentationProvider()
    assert adapter.available is False
    with pytest.raises(SegmentationError, match="unavailable"):
        adapter.segment()
