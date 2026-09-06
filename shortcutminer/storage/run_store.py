"""Durable JSON records for one formal ShortcutMiner run."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from shortcutminer.protocol.run import RunManifest, RunRecord
from shortcutminer.protocol.summarize import SummaryReport
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.storage.paths import RunPathError, get_run_dir


class RunStoreError(RuntimeError):
    """Raised when formal run metadata cannot be stored or loaded."""


def _run_dir(runs_root: str | Path, run_id: str) -> Path:
    try:
        return get_run_dir(runs_root, run_id)
    except RunPathError as exc:
        raise RunStoreError(str(exc)) from exc


def run_record_exists(runs_root: str | Path, run_id: str) -> bool:
    return (_run_dir(runs_root, run_id) / "run.json").is_file()


def _write_model(path: Path, model: RunRecord | RunManifest) -> Path:
    atomic_write_json(
        path,
        model.model_dump(mode="json"),
        error_type=RunStoreError,
    )
    return path


def create_run_records(
    runs_root: str | Path,
    *,
    record: RunRecord,
    manifest: RunManifest,
    run_input: dict[str, Any],
) -> Path:
    """Create the durable run envelope without overwriting an existing run."""
    run_dir = _run_dir(runs_root, record.run_id)
    if (run_dir / "run.json").exists():
        raise RunStoreError(f"Run already exists: {record.run_id}")
    _write_model(run_dir / "run.json", record)
    try:
        _write_model(run_dir / "manifest.json", manifest)
        atomic_write_json(
            run_dir / "input.json",
            run_input,
            error_type=RunStoreError,
        )
    except Exception:
        # The existing run.json deliberately prevents a later invocation from
        # silently treating this partially initialized run as a fresh run.
        raise
    return run_dir


def save_run_record(
    runs_root: str | Path,
    record: RunRecord,
) -> Path:
    return _write_model(_run_dir(runs_root, record.run_id) / "run.json", record)


def save_run_manifest(
    runs_root: str | Path,
    manifest: RunManifest,
) -> Path:
    return _write_model(
        _run_dir(runs_root, manifest.run_id) / "manifest.json",
        manifest,
    )


def save_workflow_state(
    runs_root: str | Path,
    run_id: str,
    state: dict[str, Any],
) -> Path:
    path = _run_dir(runs_root, run_id) / "workflow_state.json"
    atomic_write_json(path, state, error_type=RunStoreError)
    return path


def save_final_result(
    runs_root: str | Path,
    run_id: str,
    result: dict[str, Any],
) -> Path:
    path = _run_dir(runs_root, run_id) / "final_result.json"
    try:
        report = SummaryReport.model_validate(result)
    except ValidationError as exc:
        raise RunStoreError("Final result is not a valid SummaryReport") from exc
    if report.run_id != run_id:
        raise RunStoreError("Final result run_id does not match its run directory")
    atomic_write_json(
        path,
        report.model_dump(mode="json"),
        error_type=RunStoreError,
    )
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunStoreError(f"Unable to read run record: {path}") from exc
    if not isinstance(value, dict):
        raise RunStoreError(f"Run record is not a JSON object: {path}")
    return value


def load_run_record(runs_root: str | Path, run_id: str) -> RunRecord:
    path = _run_dir(runs_root, run_id) / "run.json"
    try:
        return RunRecord.model_validate(_read_json(path))
    except ValidationError as exc:
        raise RunStoreError(f"Invalid run record: {path}") from exc


def load_run_manifest(runs_root: str | Path, run_id: str) -> RunManifest:
    path = _run_dir(runs_root, run_id) / "manifest.json"
    try:
        return RunManifest.model_validate(_read_json(path))
    except ValidationError as exc:
        raise RunStoreError(f"Invalid run manifest: {path}") from exc


def load_workflow_state(
    runs_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    return _read_json(_run_dir(runs_root, run_id) / "workflow_state.json")


def load_final_result(
    runs_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    path = _run_dir(runs_root, run_id) / "final_result.json"
    try:
        report = SummaryReport.model_validate(_read_json(path))
    except ValidationError as exc:
        raise RunStoreError(f"Invalid final result: {path}") from exc
    if report.run_id != run_id:
        raise RunStoreError("Final result run_id does not match its run directory")
    return report.model_dump(mode="json")
