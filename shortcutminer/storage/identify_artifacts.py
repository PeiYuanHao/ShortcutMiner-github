from pathlib import Path

from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.storage.json_artifacts import (
    JsonObject,
    JsonObjectInput,
    atomic_write_json,
    model_to_json_object,
)
from shortcutminer.storage.paths import (
    RunPathError,
    get_run_dir,
)


class IdentifyArtifactError(RuntimeError):
    """Raised when Identify artifacts cannot be persisted."""


def get_identify_run_dir(
    runs_root: str | Path,
    run_id: str,
) -> Path:
    """
    Return the Identify artifact directory for one run.
    """

    try:
        run_dir = get_run_dir(
            runs_root=runs_root,
            run_id=run_id,
        )

    except RunPathError as exc:
        raise IdentifyArtifactError(
            str(exc)
        ) from exc

    return run_dir / "identify"


def save_identify_input(
    runs_root: str | Path,
    *,
    run_id: str,
    payload: JsonObjectInput,
) -> Path:
    """
    Persist the exact JSON payload sent to the Identify Agent.

    ``payload`` must already be JSON-compatible. Pydantic models should be
    converted with ``model_dump(mode="json")`` before calling this function.
    """

    identify_dir = get_identify_run_dir(
        runs_root=runs_root,
        run_id=run_id,
    )

    output_path = (
        identify_dir
        / "input.json"
    )

    atomic_write_json(
        output_path,
        payload,
        error_type=IdentifyArtifactError,
    )

    return output_path


def save_identify_raw_output(
    runs_root: str | Path,
    *,
    run_id: str,
    raw_output: JsonObjectInput,
) -> Path:
    """Persist the JSON object returned by the Identify Agent verbatim."""

    identify_dir = get_identify_run_dir(
        runs_root=runs_root,
        run_id=run_id,
    )

    output_path = identify_dir / "raw_output.json"

    atomic_write_json(
        output_path,
        raw_output,
        error_type=IdentifyArtifactError,
    )

    return output_path


def save_identify_output(
    runs_root: str | Path,
    report: CandidateReport,
) -> Path:
    """
    Persist the final CandidateReport produced by Identify.
    """

    identify_dir = get_identify_run_dir(
        runs_root=runs_root,
        run_id=report.run_id,
    )

    output_path = (
        identify_dir
        / "output.json"
    )

    atomic_write_json(
        output_path,
        model_to_json_object(report),
        error_type=IdentifyArtifactError,
    )

    return output_path
