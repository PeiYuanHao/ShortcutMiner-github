from pathlib import Path

from shortcutminer.protocol.qualitative import QualitativeResult
from shortcutminer.storage.json_artifacts import (
    JsonObjectInput,
    JsonValue,
    atomic_write_json,
    model_to_json_object,
)
from shortcutminer.storage.paths import RunPathError, get_run_dir


class QualitativeArtifactError(RuntimeError):
    """Raised when Qualitative artifacts cannot be persisted."""


def get_qualitative_run_dir(
    runs_root: str | Path,
    run_id: str,
    scope: str | None = None,
) -> Path:
    try:
        run_dir = get_run_dir(runs_root=runs_root, run_id=run_id)
    except RunPathError as exc:
        raise QualitativeArtifactError(str(exc)) from exc
    directory = run_dir / "qualitative"
    if scope is not None:
        if not scope or any(
            character
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in scope
        ):
            raise QualitativeArtifactError(
                "Qualitative artifact scope contains unsafe characters."
            )
        directory /= scope
    return directory


def save_qualitative_input(
    runs_root: str | Path,
    *,
    run_id: str,
    payload: JsonObjectInput,
    scope: str | None = None,
) -> Path:
    path = get_qualitative_run_dir(runs_root, run_id, scope) / "input.json"
    atomic_write_json(path, payload, error_type=QualitativeArtifactError)
    return path


def save_qualitative_raw_output(
    runs_root: str | Path,
    *,
    run_id: str,
    raw_output: JsonValue,
    scope: str | None = None,
) -> Path:
    path = get_qualitative_run_dir(runs_root, run_id, scope) / "raw_output.json"
    atomic_write_json(path, raw_output, error_type=QualitativeArtifactError)
    return path


def save_qualitative_draft_invariant_telemetry(
    runs_root: str | Path,
    *,
    run_id: str,
    telemetry: JsonObjectInput,
    scope: str | None = None,
) -> Path:
    path = (
        get_qualitative_run_dir(runs_root, run_id, scope)
        / "draft_invariant_telemetry.json"
    )
    atomic_write_json(path, telemetry, error_type=QualitativeArtifactError)
    return path


def save_qualitative_result(
    runs_root: str | Path,
    result: QualitativeResult,
    *,
    scope: str | None = None,
) -> Path:
    path = (
        get_qualitative_run_dir(runs_root, result.run_id, scope)
        / "result.json"
    )
    atomic_write_json(
        path,
        model_to_json_object(result),
        error_type=QualitativeArtifactError,
    )
    return path
