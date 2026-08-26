from collections.abc import Mapping
import json
import os
import tempfile
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel

from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.storage.paths import (
    RunPathError,
    get_run_dir,
)


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = (
    JsonScalar
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]
JsonObjectInput: TypeAlias = Mapping[str, JsonValue]


class IdentifyArtifactError(RuntimeError):
    """Raised when Identify artifacts cannot be persisted."""


def _atomic_write_json(
    path: Path,
    data: JsonObjectInput,
) -> None:
    """
    Write JSON atomically.

    The temporary file is written in the same directory and
    replaced only after the complete JSON document is ready.
    """

    temp_path: Path | None = None

    try:
        serialized = json.dumps(
            dict(data),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            file.write(serialized)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temp_path,
            path,
        )

    except Exception as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(
                    missing_ok=True
                )

            except OSError:
                pass

        raise IdentifyArtifactError(
            f"Failed to write artifact: {path}"
        ) from exc


def _model_to_dict(
    model: BaseModel,
) -> JsonObject:
    data = model.model_dump(
        mode="json"
    )

    return data


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

    _atomic_write_json(
        output_path,
        payload,
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

    _atomic_write_json(
        output_path,
        raw_output,
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

    _atomic_write_json(
        output_path,
        _model_to_dict(report),
    )

    return output_path
