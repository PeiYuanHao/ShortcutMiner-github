"""Construction of the reproducibility manifest for a formal run."""

from typing import Any

from shortcutminer.protocol.identify import DatasetSpec, TaskSpec
from shortcutminer.protocol.run import (
    CheckpointReference,
    RunError,
    RunManifest,
    RunStatus,
)
from shortcutminer.workflow.state import ShortcutMinerState


def build_run_manifest(
    state: ShortcutMinerState,
    *,
    checkpoint: CheckpointReference,
    timestamp: str,
) -> RunManifest:
    """Capture the initial reproducibility inputs and durable artifact paths."""
    run_id = state["run_id"]
    config: dict[str, Any] = {
        name: state.get(name, {})
        for name in (
            "identify_config",
            "executive_config",
            "quantitative_config",
            "qualitative_config",
            "summarize_config",
        )
    }
    if state.get("model_runtime") is not None:
        config["model_runtime"] = state["model_runtime"]
    return RunManifest(
        schema_version="1.0",
        run_id=run_id,
        status="PENDING",
        dataset_info=DatasetSpec.model_validate(state["dataset_info"]),
        task_input=TaskSpec.model_validate(state["task_input"]),
        seed=state["identify_config"]["seed"],
        config=config,
        metadata=state.get("metadata", {}),
        checkpoint=checkpoint,
        artifacts={
            "run_record": "run.json",
            "input": "input.json",
            "events": "events.jsonl",
            "workflow_state": "workflow_state.json",
            "final_result": "final_result.json",
            "manifest": "manifest.json",
        },
        created_at=timestamp,
        updated_at=timestamp,
        error=None,
    )


def update_manifest_status(
    manifest: RunManifest,
    *,
    status: RunStatus,
    timestamp: str,
    error: RunError | None,
) -> RunManifest:
    """Return a validated manifest status update without changing its schema."""
    data = manifest.model_dump(mode="json")
    data.update(status=status, updated_at=timestamp, error=error)
    return RunManifest.model_validate(data)
