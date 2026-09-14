"""Secret-free, content-addressed pilot snapshot construction and verification."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from shortcutminer.evaluation.pilot.protocol import FrozenWorkspace, PilotConfig, PilotSnapshot
from shortcutminer.runtime.openai_compatible import build_runtime_snapshot
from shortcutminer.storage.json_artifacts import atomic_write_json


class PilotSnapshotError(RuntimeError):
    """Raised when a frozen pilot identity cannot be built or verified."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        if not item.is_file() or "__pycache__" in item.parts or item.suffix in {".pyc", ".pyo"}:
            continue
        relative = item.relative_to(path).as_posix().encode("utf-8")
        data = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _git(repository_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=repository_root, check=True, capture_output=True,
            text=True, encoding="utf-8",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotSnapshotError(f"git {' '.join(args)} failed") from exc


def workspace_hashes(repository_root: Path) -> list[FrozenWorkspace]:
    return [
        FrozenWorkspace(
            agent_name=agent,
            sha256=tree_hash(repository_root / "agents" / agent / "workspace"),
        )
        for agent in ("identify", "executive", "quantitative", "qualitative", "summarize")
    ]


def build_pilot_snapshot(
    config: PilotConfig,
    *,
    created_at: str | None = None,
    git_commit: str | None = None,
) -> PilotSnapshot:
    root = Path(config.repository_root).resolve(strict=True)
    explicit_commit = git_commit or config.frozen_git_commit
    resolved_commit = explicit_commit or _git(root, "rev-parse", "HEAD")
    try:
        repository_dirty = bool(_git(root, "status", "--porcelain"))
    except PilotSnapshotError:
        if explicit_commit is None:
            raise
        repository_dirty = False
    text = build_runtime_snapshot(config.text_runtime)
    semantic = build_runtime_snapshot(config.semantic_runtime)
    combination = canonical_hash({
        "text_runtime_snapshot": text.snapshot_id,
        "semantic_runtime_snapshot": semantic.snapshot_id,
    })
    challenge_hash = canonical_hash([item.model_dump(mode="json") for item in config.challenges])
    config_hash = canonical_hash(config.model_dump(mode="json"))
    workspaces = workspace_hashes(root)
    protocol_paths = (
        "shortcutminer/protocol", "shortcutminer/evaluation/blind",
        "shortcutminer/evaluation/pilot",
    )
    protocol_hashes = {
        value: tree_hash(root / value) if (root / value).is_dir() else file_hash(root / value)
        for value in protocol_paths
    }
    backend_paths = (
        "shortcutminer/experiment/real_executor.py",
        "shortcutminer/experiment/qualitative_backend.py",
        "shortcutminer/protocol/evidence.py",
    )
    backend_hashes = {
        value: file_hash(root / value) for value in backend_paths if (root / value).is_file()
    }
    identity = {
        "pilot_id": config.pilot_id,
        "git_commit": resolved_commit,
        "workspace_hashes": [item.model_dump(mode="json") for item in workspaces],
        "protocol_hashes": protocol_hashes,
        "backend_hashes": backend_hashes,
        "text_runtime": text.model_dump(mode="json"),
        "semantic_runtime": semantic.model_dump(mode="json"),
        "provider_combination_id": combination,
        "config_hash": config_hash,
        "challenge_suite_hash": challenge_hash,
    }
    snapshot_id = f"pilot-snapshot-{canonical_hash(identity)[:20]}"
    return PilotSnapshot(
        snapshot_id=snapshot_id,
        pilot_id=config.pilot_id,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        git_commit=identity["git_commit"],
        repository_dirty=repository_dirty,
        workspace_hashes=workspaces,
        protocol_hashes=protocol_hashes,
        backend_hashes=backend_hashes,
        text_runtime=text,
        semantic_runtime=semantic,
        provider_combination_id=combination,
        config_hash=config_hash,
        challenge_suite_hash=challenge_hash,
        candidate_limit=config.candidate_limit,
        validation_depth=config.validation_depth,
        visual_sample_budget=config.visual_sample_budget,
        semantic_sample_budget=config.semantic_sample_budget,
        runtime_seeds=config.runtime_seeds,
        bootstrap_seeds=config.bootstrap_seeds,
        budget=config.budget,
    )


def save_pilot_snapshot(snapshot: PilotSnapshot, pilot_root: str | Path) -> Path:
    path = Path(pilot_root) / "snapshot.json"
    if path.is_file():
        existing = PilotSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        if existing != snapshot:
            raise PilotSnapshotError("pilot snapshot is immutable and does not match")
        return path
    atomic_write_json(path, snapshot.model_dump(mode="json"), error_type=PilotSnapshotError)
    return path


def verify_pilot_snapshot(config: PilotConfig, snapshot: PilotSnapshot) -> None:
    current = build_pilot_snapshot(
        config, created_at=snapshot.created_at,
    )
    if current.snapshot_id != snapshot.snapshot_id or current != snapshot:
        raise PilotSnapshotError("workspace, protocol, backend, runtime, or budget freeze changed")


__all__ = [
    "PilotSnapshotError", "build_pilot_snapshot", "canonical_hash", "file_hash",
    "save_pilot_snapshot", "tree_hash", "verify_pilot_snapshot", "workspace_hashes",
]
