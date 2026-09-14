"""Deterministic, evaluation-only generators for small blind challenges."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from shortcutminer.evaluation.blind.protocol import (
    BlindChallenge,
    BlindChallengeConfig,
    CleanControlSpec,
    HiddenShortcutSpec,
    HiddenShortcutTruth,
    PublicChallenge,
)
from shortcutminer.protocol.identify import TaskSpec
from shortcutminer.storage.json_artifacts import atomic_write_json


GENERATOR_VERSION = "blind_synthetic_v1"


class BlindGenerationError(RuntimeError):
    """Raised when an isolated challenge cannot be generated safely."""


@dataclass(frozen=True)
class GeneratedBlindChallenge:
    challenge_root: Path
    public_dataset_root: Path
    hidden_truth_path: Path
    challenge: BlindChallenge


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_public_dataset(dataset_root: Path) -> str:
    """Hash public relative names and bytes, never filesystem metadata."""
    digest = hashlib.sha256()
    for path in sorted(
        (item for item in dataset_root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(dataset_root).as_posix(),
    ):
        relative = path.relative_to(dataset_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _image_bytes(
    *,
    label: int,
    index: int,
    rng: np.random.Generator,
    spatial_patch: bool,
    acquisition_shift: bool,
    semantic_background: str | None,
    clean_size: int,
) -> bytes:
    size = clean_size
    noise = rng.integers(0, 20, size=(size, size, 3), dtype=np.uint8)
    array = np.clip(noise + 45, 0, 255).astype(np.uint8)
    if semantic_background == "outdoor":
        array[: size // 2, :, :] = np.array([80, 160, 225], dtype=np.uint8)
        array[size // 2 :, :, :] = np.array([65, 150, 70], dtype=np.uint8)
    elif semantic_background == "indoor":
        array[: size // 2, :, :] = np.array([185, 170, 145], dtype=np.uint8)
        array[size // 2 :, :, :] = np.array([115, 95, 80], dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(image)
    if label == 0:
        draw.ellipse((size // 3, size // 3, 2 * size // 3, 2 * size // 3), fill=(90, 150, 210))
    else:
        draw.rectangle((size // 3, size // 3, 2 * size // 3, 2 * size // 3), fill=(210, 150, 90))
    if acquisition_shift:
        shifted = np.asarray(image, dtype=np.int16) + 65
        image = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8), mode="RGB")
    elif index % 7 == 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.35))
    if spatial_patch:
        draw = ImageDraw.Draw(image)
        patch = max(3, size // 5)
        draw.rectangle((0, 0, patch - 1, patch - 1), fill=(245, 245, 35))
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=6)
    return stream.getvalue()


def _strength_probability(name: str) -> float:
    return {"weak": 0.65, "moderate": 0.8, "strong": 0.95}[name]


class BlindChallengeGenerator:
    """Create physically separated public datasets and private truth files."""

    def generate(
        self,
        config: BlindChallengeConfig,
        output_root: str | Path,
    ) -> GeneratedBlindChallenge:
        root = (Path(output_root).resolve(strict=False) / config.challenge_id).resolve(
            strict=False
        )
        base = Path(output_root).resolve(strict=False)
        try:
            root.relative_to(base)
        except ValueError as exc:
            raise BlindGenerationError("challenge path escapes output root") from exc
        if root.exists():
            if not root.is_dir() or any(root.iterdir()):
                raise BlindGenerationError("challenge directory must be absent or empty")
        dataset_root = root / "public" / "dataset"
        private_root = root / "private"
        image_root = dataset_root / "images"
        image_root.mkdir(parents=True, exist_ok=True)
        private_root.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(config.challenge_generation_seed)
        count = config.sample_count
        if config.family == "clean_control":
            zero_count = int(round(count * 0.6))
            labels = np.array([0] * zero_count + [1] * (count - zero_count))
        else:
            labels = np.array([index % 2 for index in range(count)])
        rng.shuffle(labels)
        per_label_offsets = {0: 0, 1: 0}
        probability = _strength_probability(config.injection_strength)
        rows: list[dict[str, str]] = []
        affected: list[str] = []

        for index, label_value in enumerate(labels, 1):
            label = int(label_value)
            offset = per_label_offsets[label]
            per_label_offsets[label] += 1
            sample_id = f"sample-{index:04d}"
            if config.family == "clean_control":
                site = ("A", "B")[offset % 2]
                device = ("D1", "D2")[offset % 2]
                batch = ("K1", "K2", "K3", "K4")[offset % 4]
                brightness_code = offset % 2
                size = (20, 24)[offset % 2]
                measurement = "" if offset % 5 == 0 else f"{(offset % 7) / 7:.3f}"
                spatial_patch = False
                acquisition_shift = bool(brightness_code)
                semantic_background = None
            else:
                site = ("A", "B")[index % 2]
                batch = ("K1", "K2", "K3")[index % 3]
                device = ("D1", "D2")[index % 2]
                measurement = "" if index % 11 == 0 else f"{(index % 7) / 7:.3f}"
                size = 24
                spatial_patch = (
                    config.family == "spatial_visual"
                    and label == 1
                    and bool(rng.random() < probability)
                )
                acquisition_shift = (
                    config.family == "acquisition_quality"
                    and label == 1
                    and bool(rng.random() < probability)
                )
                semantic_background = None
                if config.family == "semantic_background":
                    aligned = bool(rng.random() < probability)
                    context_index = label if aligned else 1 - label
                    semantic_background = ("outdoor", "indoor")[context_index]
                    if aligned:
                        affected.append(sample_id)
                if config.family in {"metadata_association", "mixed"}:
                    aligned = bool(rng.random() < probability)
                    device = ("D1", "D2")[label if aligned else 1 - label]
                    if aligned:
                        affected.append(sample_id)
                if spatial_patch or acquisition_shift:
                    affected.append(sample_id)
                if config.family == "mixed":
                    size = (20, 24)[index % 2]
                    acquisition_shift = bool(index % 4 == 0)

            relative_image = f"images/item-{index:04d}.png"
            (dataset_root / relative_image).write_bytes(
                _image_bytes(
                    label=label,
                    index=index,
                    rng=rng,
                    spatial_patch=spatial_patch,
                    acquisition_shift=acquisition_shift,
                    semantic_background=semantic_background,
                    clean_size=size,
                )
            )
            rows.append(
                {
                    "id": sample_id,
                    "label": str(label),
                    "site": site,
                    "device": device,
                    "batch": batch,
                    "measure": measurement,
                    "path": relative_image,
                }
            )

        metadata_path = dataset_root / "metadata.csv"
        with metadata_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["id", "label", "site", "device", "batch", "measure", "path"],
            )
            writer.writeheader()
            writer.writerows(rows)
        public_hash = hash_public_dataset(dataset_root)

        shortcuts: list[HiddenShortcutSpec] = []
        clean_control = None
        if config.family in {"metadata_association", "mixed"}:
            shortcuts.append(
                HiddenShortcutSpec(
                    truth_id="T001",
                    family="metadata",
                    target_field="device",
                    injection_mechanism="Label-correlated assignment of a neutral acquisition identifier.",
                    affected_sample_ids=affected,
                    injection_strength=config.injection_strength,
                    strength_parameters={"alignment_probability": probability},
                )
            )
        elif config.family == "spatial_visual":
            shortcuts.append(
                HiddenShortcutSpec(
                    truth_id="T001",
                    family="spatial",
                    spatial_region="corner",
                    injection_mechanism="A fixed high-contrast pattern is added to one label group.",
                    affected_sample_ids=affected,
                    injection_strength=config.injection_strength,
                    strength_parameters={"patch_fraction": 0.2},
                )
            )
        elif config.family == "acquisition_quality":
            shortcuts.append(
                HiddenShortcutSpec(
                    truth_id="T001",
                    family="acquisition",
                    acquisition_family="brightness",
                    injection_mechanism="A global capture-intensity transformation is associated with one label group.",
                    affected_sample_ids=affected,
                    injection_strength=config.injection_strength,
                    strength_parameters={"channel_shift": 65},
                )
            )
        elif config.family == "semantic_background":
            shortcuts.append(
                HiddenShortcutSpec(
                    truth_id="T001",
                    family="semantic",
                    target_field="background context",
                    injection_mechanism="Outdoor-like versus indoor-like scene context is associated with the central-shape label.",
                    affected_sample_ids=affected,
                    injection_strength=config.injection_strength,
                    strength_parameters={"alignment_probability": probability},
                )
            )
        else:
            clean_control = CleanControlSpec(
                independent_variations=[
                    "balanced site variation within labels",
                    "balanced device variation within labels",
                    "four batch values",
                    "brightness and image-size variation balanced within labels",
                    "class imbalance and missing numeric metadata",
                ],
                construction_notes=[
                    "Variations are assigned within each label stratum and are not designated shortcuts."
                ],
            )

        truth = HiddenShortcutTruth(
            challenge_id=config.challenge_id,
            generator_version=GENERATOR_VERSION,
            challenge_family=config.family,
            challenge_generation_seed=config.challenge_generation_seed,
            shortcut_present=bool(shortcuts),
            shortcuts=shortcuts,
            clean_control=clean_control,
            public_dataset_hash=public_hash,
        )
        truth_data = truth.model_dump(mode="json")
        truth_hash = _canonical_hash(truth_data)
        truth_path = private_root / "hidden_truth.json"
        atomic_write_json(truth_path, truth_data, error_type=BlindGenerationError)

        public = PublicChallenge(
            challenge_id=config.challenge_id,
            generator_version=GENERATOR_VERSION,
            dataset_reference="public/dataset",
            metadata_file="metadata.csv",
            task=TaskSpec(
                type="classification",
                label_column="label",
                description="Classify the intended central shape.",
                intended_signal="central shape",
            ),
            runtime_sampling_seed=config.challenge_generation_seed + 10_000,
            public_dataset_hash=public_hash,
        )
        challenge = BlindChallenge(
            challenge_id=config.challenge_id,
            public=public,
            hidden_truth_reference="private/hidden_truth.json",
            private_truth_hash=truth_hash,
        )
        atomic_write_json(
            root / "challenge.json",
            challenge.model_dump(mode="json"),
            error_type=BlindGenerationError,
        )
        return GeneratedBlindChallenge(
            challenge_root=root,
            public_dataset_root=dataset_root,
            hidden_truth_path=truth_path,
            challenge=challenge,
        )


__all__ = [
    "BlindChallengeGenerator",
    "BlindGenerationError",
    "GENERATOR_VERSION",
    "GeneratedBlindChallenge",
    "hash_public_dataset",
]
