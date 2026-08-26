from pathlib import Path

from shortcutminer.protocol.executive import RoutingPlan
from shortcutminer.storage.json_artifacts import (
    JsonObjectInput,
    atomic_write_json,
    model_to_json_object,
)
from shortcutminer.storage.paths import RunPathError, get_run_dir


class ExecutiveArtifactError(RuntimeError):
    """Raised when Executive artifacts cannot be persisted."""


def get_executive_run_dir(
    runs_root: str | Path,
    run_id: str,
) -> Path:
    try:
        run_dir = get_run_dir(runs_root=runs_root, run_id=run_id)
    except RunPathError as exc:
        raise ExecutiveArtifactError(str(exc)) from exc

    return run_dir / "executive"


def save_executive_input(
    runs_root: str | Path,
    *,
    run_id: str,
    payload: JsonObjectInput,
) -> Path:
    output_path = get_executive_run_dir(runs_root, run_id) / "input.json"
    atomic_write_json(
        output_path,
        payload,
        error_type=ExecutiveArtifactError,
    )
    return output_path


def save_executive_raw_output(
    runs_root: str | Path,
    *,
    run_id: str,
    raw_output: JsonObjectInput,
) -> Path:
    output_path = (
        get_executive_run_dir(runs_root, run_id) / "raw_output.json"
    )
    atomic_write_json(
        output_path,
        raw_output,
        error_type=ExecutiveArtifactError,
    )
    return output_path


def save_executive_routing(
    runs_root: str | Path,
    plan: RoutingPlan,
) -> Path:
    output_path = (
        get_executive_run_dir(runs_root, plan.run_id) / "routing.json"
    )
    atomic_write_json(
        output_path,
        model_to_json_object(plan),
        error_type=ExecutiveArtifactError,
    )
    return output_path
