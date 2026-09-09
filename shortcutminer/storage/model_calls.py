"""File-backed, secret-free model invocation telemetry."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from shortcutminer.protocol.model_runtime import (
    AgentModelTelemetry,
    ModelInvocationRecord,
    ModelRuntimeSnapshot,
    RunModelTelemetry,
)
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.storage.paths import get_run_dir


class ModelTelemetryError(RuntimeError):
    """Raised when model telemetry cannot be persisted safely."""


_LOCK = Lock()
MODEL_CALLS_DIR = "model_calls"
TELEMETRY_REFERENCE = "model_calls/telemetry.json"


def _invocation_reference(invocation_id: str) -> str:
    if not invocation_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in invocation_id):
        raise ModelTelemetryError("invocation_id is unsafe for artifact storage")
    return f"{MODEL_CALLS_DIR}/{invocation_id}.json"


def _read_records(run_dir: Path) -> list[ModelInvocationRecord]:
    directory = run_dir / MODEL_CALLS_DIR
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "telemetry.json" or path.name.startswith("runtime_snapshot-"):
            continue
        try:
            records.append(
                ModelInvocationRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            )
        except Exception as exc:
            raise ModelTelemetryError(
                f"Stored model invocation is invalid: {path.name}"
            ) from exc
    return records


def _runtime_snapshot_ids(run_dir: Path) -> list[str]:
    directory = run_dir / MODEL_CALLS_DIR
    prefix = "runtime_snapshot-"
    suffix = ".json"
    return sorted(
        path.name[len(prefix) : -len(suffix)]
        for path in directory.glob(f"{prefix}*{suffix}")
        if path.name.startswith(prefix) and path.name.endswith(suffix)
    )


def _agent_summary(agent_name: str, records: list[ModelInvocationRecord]) -> AgentModelTelemetry:
    selected = [record for record in records if record.agent_name == agent_name]
    usage_complete = bool(selected) and all(record.usage.available for record in selected)
    return AgentModelTelemetry(
        agent_name=agent_name,
        total_calls=len(selected),
        successful_calls=sum(record.status == "completed" for record in selected),
        failed_calls=sum(record.status != "completed" for record in selected),
        retry_calls=sum(record.transport_retry for record in selected),
        schema_retry_calls=sum(record.schema_retry for record in selected),
        rate_limit_count=sum(record.status == "rate_limited" for record in selected),
        timeout_count=sum(record.status == "timed_out" for record in selected),
        provider_5xx_count=sum(record.status == "provider_5xx" for record in selected),
        invalid_json_count=sum(record.status == "invalid_json" for record in selected),
        protocol_schema_failure_count=sum(
            record.status == "protocol_schema_failed" for record in selected
        ),
        budget_exceeded_count=sum(record.status == "budget_exceeded" for record in selected),
        prompt_tokens=(
            sum(record.usage.prompt_tokens or 0 for record in selected)
            if usage_complete
            else None
        ),
        completion_tokens=(
            sum(record.usage.completion_tokens or 0 for record in selected)
            if usage_complete
            else None
        ),
        total_tokens=(
            sum(record.usage.total_tokens or 0 for record in selected)
            if usage_complete
            else None
        ),
        usage_complete=usage_complete,
        total_model_latency_ms=round(sum(record.latency_ms for record in selected), 6),
    )


def build_run_model_telemetry(
    run_id: str,
    records: list[ModelInvocationRecord],
    *,
    runtime_snapshot_ids: list[str],
) -> RunModelTelemetry:
    agent_names = (
        "identify",
        "executive",
        "quantitative",
        "qualitative",
        "summarize",
        "semantic_visual",
    )
    per_agent = [_agent_summary(name, records) for name in agent_names]
    usage_complete = bool(records) and all(record.usage.available for record in records)
    limitations = []
    if records and not usage_complete:
        limitations.append(
            "Token usage is unavailable for at least one provider invocation; aggregate token totals are unavailable and were not guessed."
        )
    return RunModelTelemetry(
        run_id=run_id,
        runtime_snapshot_ids=list(dict.fromkeys(runtime_snapshot_ids)),
        total_calls=len(records),
        successful_calls=sum(record.status == "completed" for record in records),
        failed_calls=sum(record.status != "completed" for record in records),
        retry_calls=sum(record.transport_retry for record in records),
        schema_retry_calls=sum(record.schema_retry for record in records),
        rate_limit_count=sum(record.status == "rate_limited" for record in records),
        timeout_count=sum(record.status == "timed_out" for record in records),
        provider_5xx_count=sum(record.status == "provider_5xx" for record in records),
        invalid_json_count=sum(record.status == "invalid_json" for record in records),
        protocol_schema_failure_count=sum(
            record.status == "protocol_schema_failed" for record in records
        ),
        budget_exceeded_count=sum(record.status == "budget_exceeded" for record in records),
        prompt_tokens=(
            sum(record.usage.prompt_tokens or 0 for record in records)
            if usage_complete
            else None
        ),
        completion_tokens=(
            sum(record.usage.completion_tokens or 0 for record in records)
            if usage_complete
            else None
        ),
        total_tokens=(
            sum(record.usage.total_tokens or 0 for record in records)
            if usage_complete
            else None
        ),
        usage_complete=usage_complete,
        total_model_latency_ms=round(sum(record.latency_ms for record in records), 6),
        per_agent=per_agent,
        invocation_references=[record.artifact_reference for record in records if record.artifact_reference],
        limitations=limitations,
    )


def save_runtime_snapshot(
    runs_root: str | Path,
    run_id: str,
    snapshot: ModelRuntimeSnapshot,
) -> str:
    reference = f"{MODEL_CALLS_DIR}/runtime_snapshot-{snapshot.snapshot_id}.json"
    path = get_run_dir(runs_root, run_id) / reference
    if path.is_file():
        existing = ModelRuntimeSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        if existing != snapshot:
            raise ModelTelemetryError("model runtime snapshot ID collision")
        return reference
    atomic_write_json(path, snapshot.model_dump(mode="json"), error_type=ModelTelemetryError)
    return reference


def save_model_invocation(
    runs_root: str | Path,
    record: ModelInvocationRecord,
    *,
    runtime_snapshot_ids: list[str],
) -> ModelInvocationRecord:
    reference = _invocation_reference(record.invocation_id)
    persisted = record.model_copy(update={"artifact_reference": reference})
    run_dir = get_run_dir(runs_root, record.run_id)
    path = run_dir / reference
    with _LOCK:
        if path.exists():
            raise ModelTelemetryError("model invocation ID already exists")
        atomic_write_json(
            path,
            persisted.model_dump(mode="json"),
            error_type=ModelTelemetryError,
        )
        records = _read_records(run_dir)
        stored_snapshot_ids = _runtime_snapshot_ids(run_dir)
        telemetry = build_run_model_telemetry(
            record.run_id,
            records,
            runtime_snapshot_ids=[*stored_snapshot_ids, *runtime_snapshot_ids],
        )
        atomic_write_json(
            run_dir / TELEMETRY_REFERENCE,
            telemetry.model_dump(mode="json"),
            error_type=ModelTelemetryError,
        )
    return persisted


def load_run_model_telemetry(
    runs_root: str | Path, run_id: str
) -> RunModelTelemetry | None:
    path = get_run_dir(runs_root, run_id) / TELEMETRY_REFERENCE
    if not path.is_file():
        return None
    try:
        return RunModelTelemetry.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ModelTelemetryError("stored model telemetry is invalid") from exc


__all__ = [
    "MODEL_CALLS_DIR",
    "ModelTelemetryError",
    "TELEMETRY_REFERENCE",
    "build_run_model_telemetry",
    "load_run_model_telemetry",
    "save_model_invocation",
    "save_runtime_snapshot",
]
