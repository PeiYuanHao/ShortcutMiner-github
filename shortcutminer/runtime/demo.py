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
            "candidates": [
                {
                    "cue": {"family": "metadata", "name": "site"},
                    "description": "Collection site may encode the label.",
                    "observation": "Site and label co-vary in metadata.",
                    "hypothesis": "Site may be spuriously associated with label.",
                    "evidence_sources": ["metadata"],
                    "priority": "high",
                    "metadata_field_ref": "site",
                    "additional_limitations": [],
                }
            ],
            "additional_limitations": [],
        }

    def _executive(self, payload: dict) -> dict:
        decisions = []
        for candidate, executable in zip(
            payload["request"]["candidate_report"]["candidates"],
            payload["executable_candidate_contexts"],
            strict=True,
        ):
            quantitative = (
                {
                    "objective": "Measure the candidate with authorized quantitative evidence.",
                    "requested_checks": [
                        executable["allowed_quantitative_checks"][0]
                    ],
                }
                if executable["allowed_quantitative_checks"] else None
            )
            qualitative = (
                {
                    "objective": "Inspect the candidate with authorized qualitative evidence.",
                    "requested_checks": [
                        executable["allowed_qualitative_checks"][0]
                    ],
                }
                if executable["allowed_qualitative_checks"] else None
            )
            route = (
                "both" if quantitative and qualitative
                else "quantitative" if quantitative
                else "qualitative" if qualitative
                else "unresolved"
            )
            decisions.append({
                "candidate_id": candidate["candidate_id"],
                "route": route,
                "routing_reason": "Use only candidate-specific executable checks.",
                "quantitative": quantitative,
                "qualitative": qualitative,
                "missing_requirements": (
                    [] if route != "unresolved"
                    else ["No candidate-specific executable check is available."]
                ),
                "limitations": [],
            })
        return {"decisions": decisions, "limitations": []}

    def _quantitative(self, payload: dict) -> dict:
        runtime = payload.get("tool_runtime")
        results = runtime.get("results", []) if isinstance(runtime, dict) else []
        selected = [
            item["tool_call_id"]
            for item in results
            if item.get("status") == "success"
            and item.get("tool_name") == "run_experiment"
        ]
        completed = bool(selected)
        return {
            "status": "completed" if completed else "blocked",
            "analysis_type": "dataset_association",
            "evidence_direction": "supports" if completed else "not_tested",
            "evidence_summary": (
                "The selected experiment result supports the interpretation."
                if completed
                else "No successful experiment result is available."
            ),
            "additional_limitations": (
                [] if completed else ["No successful experiment result is available."]
            ),
            "recommended_followup": [],
        }

    def _qualitative(self, payload: dict) -> dict:
        context = payload["evidence_context"]
        sample_ids = [item["sample_id"] for item in context["selected_samples"]]
        return {
            "sample_ids": sample_ids,
            "evidence_status": "supported",
            "evidence_summary": "Metadata rows reproduce the site pattern.",
            "observations": [],
            "comparisons": [],
            "interpretations": [],
            "counterevidence": [],
            "metadata_evidence": [{
                "field": "site",
                "observation": "Site differs across selected label groups.",
                "sample_refs": sample_ids,
            }],
            "failures": [],
            "additional_limitations": [],
            "artifact_references": [],
            "recommended_followup": [],
        }

    def _summarize(self, payload: dict) -> dict:
        request = payload["request"]
        quantitative_results = request["quantitative_result"]["results"]
        qualitative_results = request["qualitative_result"]["results"]
        ledger_entry = request.get("evidence_ledger", {}).get("candidates", [{}])[0]
        profile = ledger_entry.get("profile", {})
        consistency = profile.get("consistency")
        assessment = {
            "mixed": "partially_supported",
            "inconclusive": "inconclusive",
            "no_usable_evidence": "insufficient_evidence",
            "consistent_contradiction": "contradicted",
        }.get(consistency, "supported")
        return {
            "overview": "Both mock evidence branches support the candidate.",
            "findings": [
                    {
                        "candidate_ref": "C001",
                        "assessment": assessment,
                        "summary": (
                            "Supporting and contradictory evidence are both retained."
                            if consistency == "mixed"
                            else "Site is a plausible shortcut."
                        ),
                        "additional_limitations": [],
                        "recommendations": ["Validate on a held-out site."],
                    }
                ],
            "additional_limitations": [],
        }
