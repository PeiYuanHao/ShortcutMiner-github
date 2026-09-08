"""Freeze production-facing blind evaluation settings without private truth."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePath
import re
from typing import Any

from shortcutminer.evaluation.blind.protocol import EvaluationSnapshot
from shortcutminer.storage.json_artifacts import atomic_write_json


WORKSPACE_FILES = (
    "AGENTS.md", "IDENTITY.md", "SOUL.md", "USER.md", "TOOLS.md", "MEMORY.md", "HEARTBEAT.md"
)
PROTOCOL_FILES = (
    "shortcutminer/protocol/identify.py",
    "shortcutminer/protocol/executive.py",
    "shortcutminer/protocol/quantitative.py",
    "shortcutminer/protocol/qualitative.py",
    "shortcutminer/protocol/evidence.py",
    "shortcutminer/protocol/summarize.py",
)


class SnapshotError(RuntimeError):
    """Raised when a reproducible production snapshot cannot be created."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    if not path.is_file():
        raise SnapshotError(f"snapshot input is missing: {path.name}")
    return _sha256(path.read_bytes())


def _git_commit(repository_root: Path) -> str:
    git_entry = repository_root / ".git"
    if git_entry.is_file():
        text = git_entry.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir: "):
            raise SnapshotError("unsupported .git indirection")
        git_dir = (repository_root / text.removeprefix("gitdir: ")).resolve()
    else:
        git_dir = git_entry.resolve()
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference = head.removeprefix("ref: ")
    loose = git_dir / reference
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == reference:
                    return commit
    raise SnapshotError("unable to resolve git HEAD without executing git")


def _validated_git_commit(value: str) -> str:
    commit = value.strip().casefold()
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
        raise SnapshotError("git_commit must be a full 40- or 64-character hexadecimal hash")
    return commit


def _sanitize(value: Any, *, key: str = "") -> Any:
    lowered = key.casefold()
    if any(token in lowered for token in ("api_key", "token", "secret", "password")):
        return None
    if isinstance(value, dict):
        sanitized = {}
        for item_key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            clean = _sanitize(item, key=str(item_key))
            if clean is not None:
                sanitized[str(item_key)] = clean
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, PurePath):
        return "<path-omitted>" if value.is_absolute() else value.as_posix()
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute() or (len(value) >= 2 and value[1] == ":"):
            return "<absolute-path-omitted>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise SnapshotError(f"runtime config contains a non-JSON value: {type(value).__name__}")


def _canonical_hash(value: Any) -> str:
    return _sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def build_evaluation_snapshot(
    *,
    repository_root: str | Path,
    git_commit: str | None = None,
    runtime_config: dict[str, Any],
    model_config: dict[str, Any] | None,
    challenge_public_hash: str,
    runtime_sampling_seed: int,
    evaluation_bootstrap_seed: int,
    candidate_limit: int,
    validation_depth: str,
    budget_settings: dict[str, Any],
) -> EvaluationSnapshot:
    root = Path(repository_root).resolve(strict=True)
    workspace_hashes = {}
    for agent in ("identify", "executive", "quantitative", "qualitative", "summarize"):
        for filename in WORKSPACE_FILES:
            relative = f"agents/{agent}/workspace/{filename}"
            workspace_hashes[relative] = _file_hash(root / relative)
    protocol_hashes = {
        relative: _file_hash(root / relative) for relative in PROTOCOL_FILES
    }
    sanitized_runtime = _sanitize(runtime_config)
    sanitized_model = _sanitize(model_config) if model_config is not None else None
    sanitized_budgets = _sanitize(budget_settings)
    payload = {
        "git_commit": _validated_git_commit(
            _git_commit(root) if git_commit is None else git_commit
        ),
        "workspace_hashes": workspace_hashes,
        "protocol_hashes": protocol_hashes,
        "experiment_backend_version": "real_cpu_v1",
        "qualitative_backend_version": "local_descriptor_v1",
        "evidence_calibration_version": "evidence_ledger_v1",
        "runtime_config_hash": _canonical_hash(sanitized_runtime),
        "sanitized_model_config": sanitized_model,
        "challenge_public_hash": challenge_public_hash,
        "runtime_sampling_seed": runtime_sampling_seed,
        "evaluation_bootstrap_seed": evaluation_bootstrap_seed,
        "candidate_limit": candidate_limit,
        "validation_depth": validation_depth,
        "budget_settings": sanitized_budgets,
    }
    identity_payload = dict(payload)
    # The snapshot ID freezes production configuration across a batch. Each
    # per-run snapshot still records its own public challenge hash.
    identity_payload.pop("challenge_public_hash")
    snapshot_id = f"snapshot-{_canonical_hash(identity_payload)[:20]}"
    return EvaluationSnapshot(snapshot_id=snapshot_id, **payload)


def save_evaluation_snapshot(
    snapshot: EvaluationSnapshot,
    *,
    run_root: str | Path,
) -> str:
    reference = f"evaluation/snapshots/{snapshot.snapshot_id}.json"
    root = Path(run_root).resolve(strict=False)
    path = (root / reference).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SnapshotError("snapshot artifact escaped run root") from exc
    if path.is_file():
        existing = EvaluationSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        if existing != snapshot:
            raise SnapshotError("snapshot ID collision")
        return reference
    atomic_write_json(path, snapshot.model_dump(mode="json"), error_type=SnapshotError)
    return reference


__all__ = ["SnapshotError", "build_evaluation_snapshot", "save_evaluation_snapshot"]
