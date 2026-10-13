"""Adapt dataset intake to the existing Runner; no workflow or task generation."""

from pathlib import Path
import json

from shortcutminer.experiment.audit_intake import intake_dataset
from shortcutminer.experiment.run_manager import generate_run_id
from shortcutminer.protocol.audit import AuditRequest
from shortcutminer.protocol.run import RunRequest
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.storage.paths import get_run_dir


UNAVAILABLE_LABEL_COLUMN = "__shortcutminer_unavailable_label__"


def build_audit_run_request(
    request: AuditRequest, *, runs_root: str | Path = "runs", run_id: str | None = None,
    model_runtime=None,
) -> RunRequest:
    """Reinspect inputs and produce only the existing run-level contract."""
    request = intake_dataset(request)
    info = request.dataset_info
    label = info.label_source
    limitations = list(info.limitations)
    metadata_file = None
    label_column = UNAVAILABLE_LABEL_COLUMN
    if label.status == "resolved" and label.kind == "metadata_column":
        metadata_file = info.selected_metadata_file
        label_column = label.column
    elif label.status == "resolved":
        limitations.append(
            "directory_labels_not_bound: intake records class labels, but the existing "
            "profiler requires CSV labels. No label table is synthesized in Step1."
        )
    if info.selected_metadata_file and metadata_file is None:
        limitations.append(
            "metadata_not_bound: existing profiler requires a resolved label column; "
            "the discovered table remains recorded in AuditRequest without guessing labels."
        )
    # TaskSpec's required label field is an unavailable marker, never a guessed column.
    description = request.dataset_description or "Dataset-driven ShortcutMiner audit."
    if limitations:
        description += "\nIntake limitations:\n" + "\n".join(limitations)
    return RunRequest(
        run_id=run_id or generate_run_id(), runs_root=str(runs_root), seed=42,
        dataset_info={"name": info.name, "root": info.root, "metadata_file": metadata_file},
        task_input={"type": "dataset_audit", "label_column": label_column, "description": description},
        model_runtime=model_runtime,
        executive_config={"limitations": limitations},
        metadata={"entrypoint": "audit", "audit_request": request.model_dump(mode="json"),
                  "audit_limitations": limitations},
    )


def reserve_audit_intake(request: AuditRequest, *, runs_root, run_id) -> Path:
    """Exclusively reserve a CLI audit before intake so lifecycle events can persist."""
    run_dir = get_run_dir(runs_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    path = run_dir / "audit_request.json"
    atomic_write_json(path, request.model_dump(mode="json"), error_type=ValueError)
    return path


def persist_audit_request(request: RunRequest, *, reserved_intake: AuditRequest | None = None) -> Path:
    """Reserve a fresh run directory and persist intake before existing Runner execution."""
    run_dir = get_run_dir(request.runs_root, request.run_id)
    if reserved_intake is None:
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        pending = AuditRequest.model_validate(json.loads((run_dir / "audit_request.json").read_text(encoding="utf-8")))
        completed = AuditRequest.model_validate(request.metadata["audit_request"])
        if (pending != reserved_intake or completed.model_copy(update={"dataset_info": None}) != reserved_intake
                or (run_dir / "run.json").exists()):
            raise FileExistsError("Audit intake reservation does not match this fresh request")
    path = run_dir / "audit_request.json"
    atomic_write_json(path, request.metadata["audit_request"], error_type=ValueError)
    return path
