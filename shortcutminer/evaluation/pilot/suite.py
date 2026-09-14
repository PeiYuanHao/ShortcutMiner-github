"""Deterministic six-challenge Phase 16 suite construction."""

from __future__ import annotations

import json
from pathlib import Path

from shortcutminer.evaluation.blind.generator import BlindChallengeGenerator, GeneratedBlindChallenge
from shortcutminer.evaluation.blind.protocol import BlindChallenge, BlindChallengeConfig
from shortcutminer.evaluation.pilot.protocol import PilotConfig
from shortcutminer.storage.json_artifacts import atomic_write_json


class PilotSuiteError(RuntimeError):
    """Raised when generated public challenge identity is not reproducible."""


def build_suite(config: PilotConfig, pilot_root: str | Path) -> list[GeneratedBlindChallenge]:
    root = Path(pilot_root) / "challenges"
    root.mkdir(parents=True, exist_ok=True)
    generated: list[GeneratedBlindChallenge] = []
    generator = BlindChallengeGenerator()
    public_manifest: list[dict[str, str]] = []
    for item in config.challenges:
        challenge_root = root / item.challenge_id
        challenge_path = challenge_root / "challenge.json"
        if challenge_path.is_file():
            challenge = BlindChallenge.model_validate_json(challenge_path.read_text(encoding="utf-8"))
            value = GeneratedBlindChallenge(
                challenge_root=challenge_root,
                public_dataset_root=challenge_root / challenge.public.dataset_reference,
                hidden_truth_path=challenge_root / challenge.hidden_truth_reference,
                challenge=challenge,
            )
        else:
            value = generator.generate(
                BlindChallengeConfig(
                    challenge_id=item.challenge_id,
                    family=item.family,
                    challenge_generation_seed=item.generation_seed,
                    sample_count=item.sample_count,
                    injection_strength=item.injection_strength,
                ),
                root,
            )
        if value.challenge.public.challenge_id != item.challenge_id:
            raise PilotSuiteError("stored challenge ID differs from frozen suite")
        generated.append(value)
        public_manifest.append({
            "challenge_id": item.challenge_id,
            "kind": item.kind,
            "public_dataset_hash": value.challenge.public.public_dataset_hash,
            "dataset_reference": f"challenges/{item.challenge_id}/public/dataset",
        })
    atomic_write_json(
        Path(pilot_root) / "suite_public.json",
        {"schema_version": "1.0", "challenges": public_manifest},
        error_type=PilotSuiteError,
    )
    return generated


def load_pilot_config(path: str | Path) -> PilotConfig:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
        if source.suffix.casefold() in {".yaml", ".yml"}:
            import yaml
            value = yaml.safe_load(text)
        else:
            value = json.loads(text)
        return PilotConfig.model_validate(value)
    except Exception as exc:
        raise PilotSuiteError(f"invalid pilot configuration: {source}") from exc


__all__ = ["PilotSuiteError", "build_suite", "load_pilot_config"]
