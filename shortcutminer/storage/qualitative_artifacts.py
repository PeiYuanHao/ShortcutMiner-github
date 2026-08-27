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


def get_qualitative_run_dir(runs_root: str | Path, run_id: str) -> Path:
    try:
        run_dir = get_run_dir(runs_root=runs_root, run_id=run_id)
    except RunPathError as exc:
        raise QualitativeArtifactError(str(exc)) from exc
    return run_dir / "qualitative"


def save_qualitative_input(
    runs_root: str | Path,
    *,
    run_id: str,
    payload: JsonObjectInput,
) -> Path:
    path = get_qualitative_run_dir(runs_root, run_id) / "input.json"
    atomic_write_json(path, payload, error_type=QualitativeArtifactError)
    return path


def save_qualitative_raw_output(
    runs_root: str | Path,
    *,
    run_id: str,
    raw_output: JsonValue,
) -> Path:
    path = get_qualitative_run_dir(runs_root, run_id) / "raw_output.json"
    atomic_write_json(path, raw_output, error_type=QualitativeArtifactError)
    return path


def save_qualitative_result(
    runs_root: str | Path,
    result: QualitativeResult,
) -> Path:
    path = get_qualitative_run_dir(runs_root, result.run_id) / "result.json"
    atomic_write_json(
        path,
        model_to_json_object(result),
        error_type=QualitativeArtifactError,
    )
    return path
