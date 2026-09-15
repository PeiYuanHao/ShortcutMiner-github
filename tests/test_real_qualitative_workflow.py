"""Complete workflow smoke with both real evidence preparation branches."""

import json
from pathlib import Path

from PIL import Image

from shortcutminer.experiment.real_executor import RealExperimentExecutor
from shortcutminer.runner import resume_run, run_workflow
from shortcutminer.runtime.demo import DemoAgentClient
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolRuntime
from test_pipeline_mock import _initial_state


class RealEvidenceClient(DemoAgentClient):
    """Mock reasoning over real CPU quantitative and qualitative evidence."""

    def invoke_agent(self, agent_name, payload, *, system_prompt=None):
        self.calls.append(
            {"agent_name": agent_name, "payload": payload, "system_prompt": system_prompt}
        )
        runtime = payload.get("tool_runtime")
        if agent_name == "quantitative" and runtime and not runtime["results"]:
            request = payload["request"]
            return {
                "tool_calls": [
                    {
                        "protocol_version": "1.0",
                        "tool_call_id": "full-real-association",
                        "agent_name": "quantitative",
                        "run_id": request["run_id"],
                        "tool_name": "run_experiment",
                        "candidate_id": "C001",
                        "task_id": request["task"]["task_id"],
                        "arguments": {
                            "candidate_id": "C001",
                            "task_id": request["task"]["task_id"],
                            "experiment_type": "statistical_check",
                            "parameters": {
                                "metadata_file": "metadata.csv",
                                "cue_field": "site",
                                "label_column": "label",
                                "max_rows": 20,
                                "max_categories": 10,
                                "min_group_size": 1,
                                "n_permutations": 10,
                                "seed": 42,
                            },
                        },
                    }
                ]
            }
        if agent_name == "executive":
            output = super()._executive(payload)
            for task in output["tasks"]:
                task["inputs"]["metadata_fields"] = ["site", "id", "path"]
                if task["target"] == "qualitative":
                    task["inputs"]["requires_sample_access"] = True
                    task["requested_checks"] = ["sample_inspection"]
            return output
        if agent_name == "quantitative" and runtime:
            output = super()._quantitative(payload)
            experiment = runtime["results"][0]
            assert experiment["output"]["metadata"]["backend"] == "real_cpu_v1"
            output["payload"]["metrics"] = experiment["output"]["metrics"]
            output["payload"]["artifact_references"] = experiment["artifacts"]
            return output
        if agent_name == "qualitative":
            return self._real_qualitative(payload)
        return getattr(self, f"_{agent_name}")(payload)

    def _real_qualitative(self, payload: dict) -> dict:
        context = payload["evidence_context"]
        samples = context["selected_samples"]
        assert context["samples_inspected"] == 4
        assert context["descriptor_comparisons"]
        assert all(sample["visual_descriptors"] for sample in samples)
        return {
            "sample_ids": [sample["sample_id"] for sample in samples],
            "evidence_status": "partially_supported",
            "evidence_summary": "Local descriptors differ across sampled groups, with counterevidence retained.",
            "observations": [
                {
                    "sample_refs": [samples[0]["sample_id"]],
                    "observation": "The bounded intensity descriptor was computed from the selected image.",
                    "inspection_quality": "sufficient",
                }
            ],
            "interpretations": [
                {
                    "interpretation": "The difference is descriptive and does not establish causality.",
                    "supporting_observation_indexes": [0],
                    "confidence": "low",
                }
            ],
            "counterevidence": [
                {
                    "sample_refs": [samples[-1]["sample_id"]],
                    "description": "A retained comparison sample prevents a universal claim.",
                }
            ],
            "metadata_evidence": [],
            "failures": [],
            "additional_limitations": [],
            "artifact_references": [],
            "recommended_followup": [],
        }


def test_full_workflow_has_real_quantitative_and_qualitative_evidence(tmp_path: Path):
    state = _initial_state(tmp_path)
    state["run_id"] = "real-dual-evidence-workflow"
    dataset = Path(state["dataset_info"]["root"])
    (dataset / "images").mkdir()
    rows = ["id,label,site,path"]
    for index, (label, site, value) in enumerate(
        ((0, "A", 16), (0, "A", 48), (1, "B", 208), (1, "B", 240)), 1
    ):
        reference = f"images/s{index}.png"
        Image.new("RGB", (12, 12), (value, value, value)).save(dataset / reference)
        rows.append(f"{index},{label},{site},{reference}")
    (dataset / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    state["executive_config"]["capabilities"]["available_metadata_fields"] = [
        "id", "label", "site", "path"
    ]
    state["executive_config"]["capabilities"]["sample_access"] = True
    state["executive_config"]["capabilities"]["qualitative_checks"] = [
        "sample_inspection"
    ]
    state["qualitative_config"]["preparation"].update(
        {
            "sampling_strategy": "group_balanced",
            "sample_path_column": "path",
            "group_columns": ["label"],
        }
    )
    registry = build_default_tool_registry(experiment_executor=RealExperimentExecutor())
    runtimes = {
        agent: ToolRuntime(
            registry,
            run_id=state["run_id"],
            agent_name=agent,
            dataset_root=dataset,
            runs_root=state["identify_config"]["runs_root"],
        )
        for agent in ("quantitative", "qualitative")
    }
    client = RealEvidenceClient()

    result = run_workflow(state, agent_client=client, tool_runtimes=runtimes)
    quantitative = result["quantitative_result"]["results"][0]["payload"]
    qualitative = result["qualitative_result"]["results"][0]["payload"]
    run_dir = Path(state["identify_config"]["runs_root"]) / state["run_id"]
    inspection = run_dir / "qualitative" / "TASK-C001-QL" / "inspection.json"
    sheet = run_dir / "qualitative" / "TASK-C001-QL" / "contact_sheet.png"
    ledger = run_dir / "evidence" / "ledger.json"
    mtimes = (inspection.stat().st_mtime_ns, sheet.stat().st_mtime_ns)
    ledger_mtime = ledger.stat().st_mtime_ns
    inspection_data = json.loads(inspection.read_text(encoding="utf-8"))

    assert quantitative["metrics"]["effect_size_name"] == "cramers_v"
    assert any(ref.startswith("experiments/") for ref in quantitative["artifact_references"])
    assert inspection_data["backend"] == "local_descriptor_v1"
    assert inspection_data["coverage"]["inspected_count"] == 4
    assert inspection_data["descriptor_comparisons"]
    assert qualitative["counterevidence"]
    assert "qualitative/TASK-C001-QL/inspection.json" in qualitative["artifact_references"]
    assert result["final_result"]["message_type"] == "summary_report"
    assert result["summarize_request"]["evidence_ledger"]["candidates"][0]["profile"]["consistency"] == "mixed"
    assert json.loads(ledger.read_text(encoding="utf-8"))["fingerprint"]

    resumed = resume_run(
        state["run_id"],
        runs_root=state["identify_config"]["runs_root"],
        agent_client=client,
        tool_runtimes=runtimes,
    )
    assert resumed.status == "COMPLETED"
    assert (inspection.stat().st_mtime_ns, sheet.stat().st_mtime_ns) == mtimes
    assert ledger.stat().st_mtime_ns == ledger_mtime
    assert len(runtimes["quantitative"].history) == 1
