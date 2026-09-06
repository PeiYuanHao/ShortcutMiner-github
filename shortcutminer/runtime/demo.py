"""Small synthetic five-agent demo; no model, credentials, or test imports."""

from pathlib import Path

from pydantic import TypeAdapter

from shortcutminer.protocol.identify import SafeRunId
from shortcutminer.protocol.run import RunRequest
from shortcutminer.runtime.mock import MockAgentClient
from shortcutminer.workflow.state import ShortcutMinerState


def build_demo_request(runs_root: str | Path, run_id: str) -> RunRequest:
    """Create only a four-row synthetic input, outside all agent workspaces."""
    run_id = TypeAdapter(SafeRunId).validate_python(run_id)
    runs_root = Path(runs_root).resolve()
    dataset_root = (runs_root / ".mock-data" / run_id).resolve()
    if not dataset_root.is_relative_to(runs_root):
        raise ValueError("Mock dataset must remain within runs_root.")
    dataset_root.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_root / "metadata.csv"
    if metadata_path.is_symlink():
        raise ValueError("Mock metadata must not be a symlink.")
    content = "id,label,site\n1,0,A\n2,0,A\n3,1,B\n4,1,B\n"
    # Never overwrite pre-existing input, even for a demonstration.
    try:
        with metadata_path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except FileExistsError:
        if metadata_path.read_text(encoding="utf-8") != content:
            raise ValueError("Existing mock dataset differs from the demo fixture.")
    state = ShortcutMinerState(
        run_id=run_id,
        dataset_info={
            "name": "mock_dataset",
            "root": str(dataset_root),
            "metadata_file": "metadata.csv",
        },
        task_input={
            "type": "classification",
            "label_column": "label",
            "description": "Exercise the complete offline workflow.",
        },
        identify_config={
            "seed": 42,
            "runs_root": str(runs_root),
            "budget": {
                "max_samples": 4,
                "max_candidates": 1,
                "max_dataset_entries": 20,
                "max_metadata_rows": 20,
            },
        },
        executive_config={
            "audit_options": {
                "validation_depth": "standard",
                "max_tasks": 2,
                "max_samples_per_task": 4,
                "allowed_targets": ["quantitative", "qualitative"],
                "runtime_constraints": ["no_external_services"],
            },
            "capabilities": {
                "available_targets": ["quantitative", "qualitative"],
                "available_metadata_fields": ["id", "label", "site"],
                "quantitative_checks": ["association_test"],
                "qualitative_checks": ["metadata_inspection"],
                "sample_access": False,
                "external_services": False,
                "available_output_types": ["metrics", "observations"],
            },
        },
        qualitative_config={
            "preparation": {
                "seed": 42,
                "sample_id_column": "id",
                "group_columns": ["label"],
                "max_evidence_records": 4,
                "max_metadata_rows": 20,
            }
        },
        stage_status={},
        errors=[],
        metadata={"runtime": "mock", "synthetic_demo": True},
    )

    return RunRequest(
        run_id=run_id,
        runs_root=str(runs_root),
        dataset_info=state["dataset_info"],
        task_input=state["task_input"],
        seed=42,
        identify_config=state["identify_config"],
        executive_config=state["executive_config"],
        qualitative_config=state["qualitative_config"],
        metadata=state["metadata"],
    )


class DemoAgentClient(MockAgentClient):
    """Canned protocol outputs for smoke tests, not scientific measurements."""

    def __init__(self, *, fail_at: str | None = None) -> None:
        self.fail_at = fail_at
        super().__init__({
            name: self._factory(name)
            for name in ("identify", "executive", "quantitative", "qualitative", "summarize")
        })

    def _factory(self, name: str):
        def respond(payload: dict, system_prompt: str | None) -> dict:
            if name == self.fail_at:
                raise RuntimeError(f"Planned mock failure at {name}.")
            return getattr(self, f"_{name}")(payload)
        return respond

    def _identify(self, payload: dict) -> dict:
        request = payload["request"]
        context = payload["discovery_context"]
        return {
            "protocol_version": "1.0",
            "run_id": request["run_id"],
            "message_id": f"{request['run_id']}:identify:report",
            "source": "identify",
            "target": "executive",
            "message_type": "candidate_report",
            "in_reply_to": request["message_id"],
            "artifact_references": [],
            "dataset_summary": {
                "dataset": request["dataset"]["name"],
                "samples_inspected": context["samples_inspected"],
                "metadata_used": context["metadata_available"],
            },
            "candidates": [
                {
                    "candidate_id": "C001",
                    "cue": {"family": "metadata", "name": "site"},
                    "description": "Collection site may encode the label.",
                    "observation": "Site and label co-vary in metadata.",
                    "hypothesis": "Site may be spuriously associated with label.",
                    "evidence_sources": ["metadata"],
                    "priority": "high",
                    "status": "unverified",
                }
            ],
            "limitations": context["limitations"],
        }

    def _executive(self, payload: dict) -> dict:
        request = payload["request"]
        candidate = request["candidate_report"]["candidates"][0]
        context = request["routing_context"]
        common_inputs = {
            "dataset": context["dataset"]["name"],
            "dataset_reference": context["dataset"]["root"],
            "label_column": context["task"]["label_column"],
            "split": None,
            "metadata_fields": ["site", "id"],
            "artifact_references": [],
            "sample_limit": 4,
            "requires_sample_access": False,
            "requires_external_services": False,
            "runtime_constraints": ["no_external_services"],
            "required_model_capabilities": [],
        }
        tasks = []
        for suffix, target, check, output in (
            ("QT", "quantitative", "association_test", "metrics"),
            ("QL", "qualitative", "metadata_inspection", "observations"),
        ):
            tasks.append(
                {
                    "protocol_version": "1.0",
                    "run_id": request["run_id"],
                    "message_id": f"{request['run_id']}:executive:{suffix}:task",
                    "source": "executive",
                    "target": target,
                    "message_type": "analysis_task",
                    "in_reply_to": f"{request['run_id']}:executive:plan",
                    "task_id": f"TASK-C001-{suffix}",
                    "candidate_id": "C001",
                    "hypothesis": candidate["hypothesis"],
                    "objective": f"Validate the candidate with {target} evidence.",
                    "inputs": common_inputs,
                    "requested_checks": [check],
                    "expected_outputs": [output],
                    "priority": candidate["priority"],
                    "dependencies": [],
                    "status": "planned",
                }
            )
        return {
            "protocol_version": "1.0",
            "run_id": request["run_id"],
            "message_id": f"{request['run_id']}:executive:plan",
            "source": "executive",
            "target": "orchestrator",
            "message_type": "routing_plan",
            "in_reply_to": request["message_id"],
            "artifact_references": [],
            "decisions": [
                {
                    "candidate": candidate,
                    "route": "both",
                    "routing_reason": "Both evidence modes are available.",
                    "task_ids": ["TASK-C001-QT", "TASK-C001-QL"],
                    "priority": candidate["priority"],
                    "status": "planned",
                    "missing_requirements": [],
                    "limitations": [],
                }
            ],
            "tasks": tasks,
            "limitations": context["limitations"],
        }

    def _quantitative(self, payload: dict) -> dict:
        request = payload["request"]
        return {
            "protocol_version": "1.0",
            "run_id": request["run_id"],
            "message_id": f"{request['run_id']}:quantitative:result",
            "source": "quantitative",
            "target": "summarize",
            "message_type": "quantitative_analysis_result",
            "in_reply_to": request["message_id"],
            "payload": {
                "task_id": request["task"]["task_id"],
                "candidate_id": request["candidate"]["candidate_id"],
                "hypothesis": request["candidate"]["hypothesis"],
                "status": "completed",
                "analysis_type": "dataset_association",
                "metrics": {"association_score": 0.75},
                "evidence_level": "Q2",
                "evidence_direction": "supports",
                "evidence_summary": "The mock association is strong.",
                "limitations": request["limitations"],
                "artifact_references": [],
                "recommended_followup": [],
            },
        }

    def _qualitative(self, payload: dict) -> dict:
        request = payload["request"]
        context = payload["evidence_context"]
        sample_ids = [item["sample_id"] for item in context["selected_samples"]]
        groups: dict[str, int] = {}
        for item in context["selected_samples"]:
            groups[item["group"]] = groups.get(item["group"], 0) + 1
        return {
            "protocol_version": "1.0",
            "run_id": request["run_id"],
            "message_id": f"{request['run_id']}:qualitative:result",
            "source": "qualitative",
            "target": "summarize",
            "message_type": "qualitative_analysis_result",
            "in_reply_to": request["message_id"],
            "payload": {
                "task_id": request["task"]["task_id"],
                "candidate_id": request["candidate"]["candidate_id"],
                "candidate_name": request["candidate"]["cue"]["name"],
                "hypothesis": request["candidate"]["hypothesis"],
                "evidence_status": "supported",
                "evidence_summary": "Metadata rows reproduce the site pattern.",
                "sampling": {
                    "strategy": context["sampling_strategy"],
                    "seed": context["seed"],
                    "groups": groups,
                    "sample_count": len(sample_ids),
                    "sample_ids": sample_ids,
                },
                "observations": [],
                "interpretations": [],
                "counterevidence": [],
                "metadata_evidence": [
                    {
                        "field": "site",
                        "observation": "Site differs across selected label groups.",
                        "sample_refs": sample_ids,
                    }
                ],
                "failures": [],
                "limitations": context["limitations"],
                "artifact_references": [],
                "recommended_followup": [],
            },
        }

    def _summarize(self, payload: dict) -> dict:
        request = payload["request"]
        quantitative = request["quantitative_result"]["results"][0]
        qualitative = request["qualitative_result"]["results"][0]
        return {
            "protocol_version": "1.0",
            "run_id": request["run_id"],
            "message_id": f"{request['run_id']}:summarize:report",
            "source": "summarize",
            "target": "orchestrator",
            "message_type": "summary_report",
            "in_reply_to": request["message_id"],
            "payload": {
                "overview": "Both mock evidence branches support the candidate.",
                "findings": [
                    {
                        "candidate_id": "C001",
                        "assessment": "supported",
                        "summary": "Site is a plausible shortcut.",
                        "quantitative_result_ids": [quantitative["message_id"]],
                        "qualitative_result_ids": [qualitative["message_id"]],
                        "limitations": [],
                        "recommendations": ["Validate on a held-out site."],
                    }
                ],
                "limitations": [],
                "artifact_references": [],
            },
        }
