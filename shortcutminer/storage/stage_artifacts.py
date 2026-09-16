"""Small shared persistence helper for newly integrated workflow stages."""

from pathlib import Path

from pydantic import BaseModel

from shortcutminer.storage.json_artifacts import (
    JsonObjectInput,
    atomic_write_json,
    model_to_json_object,
)
from shortcutminer.storage.paths import RunPathError, get_run_dir


class StageArtifactError(RuntimeError):
    """Raised when a stage artifact cannot be persisted."""


def save_stage_json(
    runs_root: str | Path,
    *,
    run_id: str,
    stage: str,
    filename: str,
    payload: JsonObjectInput | BaseModel,
    scope: str | None = None,
) -> Path:
    """Persist JSON below a fixed workflow stage and optional task scope."""
    if stage not in {"quantitative", "summarize"}:
        raise StageArtifactError(f"Unsupported stage artifact scope: {stage}")
    if filename not in {
        "input.json",
        "raw_output.json",
        "draft.json",
        "result.json",
        "execution_obligation.json",
    }:
        raise StageArtifactError(f"Unsupported stage artifact filename: {filename}")
    if scope is not None and (
        not scope or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for character in scope)
    ):
        raise StageArtifactError("Artifact scope contains unsafe characters")
    try:
        directory = get_run_dir(runs_root=runs_root, run_id=run_id) / stage
    except RunPathError as exc:
        raise StageArtifactError(str(exc)) from exc
    if scope is not None:
        directory /= scope
    data = model_to_json_object(payload) if isinstance(payload, BaseModel) else payload
    path = directory / filename
    atomic_write_json(path, data, error_type=StageArtifactError)
    return path
