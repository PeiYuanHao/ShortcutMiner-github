import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from shortcutminer.protocol.identify import (
    CandidateReport,
    IdentifyRequest,
)


class IdentifyArtifactError(RuntimeError):
    """Raised when Identify artifacts cannot be persisted."""


def _atomic_write_json(
    path: Path,
    data: dict,
) -> None:
    """
    Write JSON atomically.

    The temporary file is written in the same directory and
    replaced only after the complete JSON document is ready.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

            temp_path = Path(file.name)

        os.replace(
            temp_path,
            path,
        )

    except Exception as exc:
        if (
            temp_path is not None
            and temp_path.exists()
        ):
            temp_path.unlink(
                missing_ok=True
            )

        raise IdentifyArtifactError(
            f"Failed to write artifact: {path}"
        ) from exc


def _model_to_dict(
    model: BaseModel,
) -> dict:
    return model.model_dump(
        mode="json"
    )


def get_identify_run_dir(
    runs_root: str | Path,
    run_id: str,
) -> Path:
    """
    Return the Identify artifact directory for one run.
    """

    if not run_id.strip():
        raise IdentifyArtifactError(
            "run_id must not be empty."
        )

    return (
        Path(runs_root)
        / run_id
        / "identify"
    )


def save_identify_input(
    runs_root: str | Path,
    request: IdentifyRequest,
) -> Path:
    """
    Persist the exact IdentifyRequest used by the run.
    """

    identify_dir = get_identify_run_dir(
        runs_root=runs_root,
        run_id=request.run_id,
    )

    output_path = (
        identify_dir
        / "input.json"
    )

    _atomic_write_json(
        output_path,
        _model_to_dict(request),
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