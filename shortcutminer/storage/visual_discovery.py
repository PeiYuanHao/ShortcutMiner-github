"""Persistence for the Identify visual-discovery artifact."""

from __future__ import annotations

from pathlib import Path

from shortcutminer.protocol.visual_discovery import VisualDiscoveryProfile
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.storage.paths import RunPathError, get_run_dir


class VisualDiscoveryArtifactError(RuntimeError):
    """Raised when a visual-discovery artifact cannot be safely reused."""


VISUAL_DISCOVERY_REFERENCE = "identify/visual_discovery.json"


def visual_discovery_path(runs_root: str | Path, run_id: str) -> Path:
    try:
        return get_run_dir(runs_root=runs_root, run_id=run_id) / VISUAL_DISCOVERY_REFERENCE
    except RunPathError as exc:
        raise VisualDiscoveryArtifactError(str(exc)) from exc


def load_visual_discovery_profile(
    runs_root: str | Path,
    run_id: str,
    *,
    expected_fingerprint: str,
) -> VisualDiscoveryProfile | None:
    path = visual_discovery_path(runs_root, run_id)
    if not path.is_file():
        return None
    try:
        profile = VisualDiscoveryProfile.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise VisualDiscoveryArtifactError(
            "Stored visual discovery artifact is invalid."
        ) from exc
    if profile.run_id != run_id or profile.request_fingerprint != expected_fingerprint:
        raise VisualDiscoveryArtifactError(
            "Stored visual discovery artifact does not match the current dataset, task, seed, budgets, descriptors, or semantic capability."
        )
    return profile


def save_visual_discovery_profile(
    runs_root: str | Path,
    profile: VisualDiscoveryProfile,
) -> Path:
    path = visual_discovery_path(runs_root, profile.run_id)
    if path.is_file():
        existing = load_visual_discovery_profile(
            runs_root,
            profile.run_id,
            expected_fingerprint=profile.request_fingerprint,
        )
        if existing != profile:
            raise VisualDiscoveryArtifactError(
                "Stored visual discovery artifact differs despite an identical fingerprint."
            )
        return path
    atomic_write_json(
        path,
        profile.model_dump(mode="json"),
        error_type=VisualDiscoveryArtifactError,
    )
    return path


__all__ = [
    "VISUAL_DISCOVERY_REFERENCE",
    "VisualDiscoveryArtifactError",
    "load_visual_discovery_profile",
    "save_visual_discovery_profile",
    "visual_discovery_path",
]
