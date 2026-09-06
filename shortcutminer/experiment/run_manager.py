"""Minimal formal run lifecycle built on the existing storage primitives."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from shortcutminer.experiment.manifest import (
    build_run_manifest,
    update_manifest_status,
)
from shortcutminer.protocol.run import (
    CheckpointReference,
    RunError,
    RunManifest,
    RunRecord,
    RunStatus,
)
from shortcutminer.storage.run_store import (
    create_run_records,
    load_run_manifest,
    load_run_record,
    save_run_manifest,
    save_run_record,
)
from shortcutminer.workflow.state import ShortcutMinerState


class RunLifecycleError(RuntimeError):
    """Raised when an invalid formal run status transition is attempted."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_run_id() -> str:
    """Generate a portable unique run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"run-{timestamp}-{uuid4().hex[:12]}"


def initialize_run(
    state: ShortcutMinerState,
    *,
    runs_root: str | Path,
    checkpoint: CheckpointReference,
) -> tuple[RunRecord, RunManifest]:
    """Persist the PENDING envelope before workflow execution begins."""
    timestamp = utc_now()
    record = RunRecord(
        schema_version="1.0",
        run_id=state["run_id"],
        status="PENDING",
        created_at=timestamp,
        updated_at=timestamp,
        attempt_count=0,
        checkpoint=checkpoint,
    )
    manifest = build_run_manifest(
        state,
        checkpoint=checkpoint,
        timestamp=timestamp,
    )
    create_run_records(
        runs_root,
        record=record,
        manifest=manifest,
        run_input={"initial_state": dict(state)},
    )
    return record, manifest


def _updated_record(
    record: RunRecord,
    *,
    status: RunStatus,
    timestamp: str,
    resume: bool = False,
    error: RunError | None = None,
) -> RunRecord:
    allowed = {
        "PENDING": {"RUNNING"},
        "RUNNING": {"COMPLETED", "FAILED"},
        "FAILED": {"RUNNING"},
        "COMPLETED": set(),
    }
    resuming_interrupted = (
        resume and record.status == "RUNNING" and status == "RUNNING"
    )
    if status not in allowed[record.status] and not resuming_interrupted:
        raise RunLifecycleError(
            f"Invalid run transition: {record.status} -> {status}"
        )
    data = record.model_dump(mode="json")
    data.update(status=status, updated_at=timestamp, error=error)
    if status == "RUNNING":
        data["attempt_count"] = record.attempt_count + 1
        data["started_at"] = record.started_at or timestamp
        data["completed_at"] = None
        data["failed_at"] = None
        if resume:
            data["resumed_at"] = timestamp
    elif status == "COMPLETED":
        data["completed_at"] = timestamp
    elif status == "FAILED":
        data["failed_at"] = timestamp
    return RunRecord.model_validate(data)


def transition_run(
    runs_root: str | Path,
    run_id: str,
    *,
    status: RunStatus,
    resume: bool = False,
    error: RunError | None = None,
) -> tuple[RunRecord, RunManifest]:
    """Apply one validated transition to both record and manifest."""
    record = load_run_record(runs_root, run_id)
    manifest = load_run_manifest(runs_root, run_id)
    timestamp = utc_now()
    updated_record = _updated_record(
        record,
        status=status,
        timestamp=timestamp,
        resume=resume,
        error=error,
    )
    updated_manifest = update_manifest_status(
        manifest,
        status=status,
        timestamp=timestamp,
        error=error,
    )
    save_run_record(runs_root, updated_record)
    save_run_manifest(runs_root, updated_manifest)
    return updated_record, updated_manifest
