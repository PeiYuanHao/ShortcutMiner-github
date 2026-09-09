"""Complete workflow coverage for image-derived Identify context."""

import json
from pathlib import Path

from PIL import Image

from shortcutminer.experiment.real_executor import RealExperimentExecutor
from shortcutminer.runner import run_workflow
from shortcutminer.runtime.demo import DemoAgentClient
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolRuntime
from test_pipeline_mock import _initial_state
from test_real_qualitative_workflow import RealEvidenceClient


class VisualEvidenceClient(RealEvidenceClient):
    """Controlled reasoning that cites only public visual-discovery facts."""

    def _identify(self, payload: dict) -> dict:
        request = payload["request"]
        context = payload["discovery_context"]
        visual = context["visual_discovery"]
        assert visual["coverage"]["decoded_samples"] == 4
        if context["metadata_available"]:
            assert visual["descriptor_differences"]
        else:
            assert visual["sample_observations"]
        return {
            "protocol_version": "1.0",
            "run_id": request["run_id"],
            "message_id": f"{request['run_id']}:identify:visual-report",
            "source": "identify",
            "target": "executive",
            "message_type": "candidate_report",
            "in_reply_to": request["message_id"],
            "artifact_references": [],
            "dataset_summary": {
                "dataset": request["dataset"]["name"],
                "samples_inspected": visual["coverage"]["decoded_samples"],
                "metadata_used": True,
            },
            "candidates": [
                {
                    "candidate_id": "C001",
                    "cue": {"family": "appearance", "name": "brightness"},
                    "description": "Brightness differs across selected label groups.",
                    "observation": "Real decoded descriptors expose different group means.",
                    "hypothesis": "Brightness may be spuriously associated with label.",
                    "evidence_sources": [
                        "sample_inspection",
                        "image_statistics",
                    ],
                    "priority": "high",
                    "status": "unverified",
                }
            ],
            "limitations": context["limitations"],
        }


class MetadataFreeVisualClient(VisualEvidenceClient):
    def _identify(self, payload: dict) -> dict:
        output = super()._identify(payload)
        output["dataset_summary"]["metadata_used"] = False
        return output

    def invoke_agent(self, agent_name, payload, *, system_prompt=None):
        if agent_name != "executive":
            return super().invoke_agent(
                agent_name, payload, system_prompt=system_prompt
            )
        self.calls.append(
            {
                "agent_name": agent_name,
                "payload": payload,
                "system_prompt": system_prompt,
            }
        )
        output = DemoAgentClient._executive(self, payload)
        for task in output["tasks"]:
            task["inputs"]["metadata_fields"] = []
            task["inputs"]["requires_sample_access"] = task["target"] == "qualitative"
            if task["target"] == "qualitative":
                task["requested_checks"] = ["sample_inspection"]
        return output

    def _real_qualitative(self, payload: dict) -> dict:
        request = payload["request"]
        context = payload["evidence_context"]
        samples = context["selected_samples"]
        assert len(samples) == context["samples_inspected"] == 4
        assert not context["descriptor_comparisons"]
        return {
            "protocol_version": "1.0",
            "run_id": request["run_id"],
            "message_id": f"{request['run_id']}:qualitative:visual-result",
            "source": "qualitative",
            "target": "summarize",
            "message_type": "qualitative_analysis_result",
            "in_reply_to": request["message_id"],
            "payload": {
                "task_id": request["task"]["task_id"],
                "candidate_id": "C001",
                "candidate_name": request["candidate"]["cue"]["name"],
                "hypothesis": request["candidate"]["hypothesis"],
                "evidence_status": "inconclusive",
                "evidence_summary": "Real image descriptors were inspected without a label grouping.",
                "sampling": {
                    "strategy": context["sampling_strategy"],
                    "seed": context["seed"],
                    "groups": {"ungrouped": 4},
                    "sample_count": 4,
                    "sample_ids": [item["sample_id"] for item in samples],
                },
                "observations": [
                    {
                        "observation_id": "QO-C001-001",
                        "group": "ungrouped",
                        "sample_refs": [samples[0]["sample_id"]],
                        "observation": "A bounded image descriptor was computed.",
                        "inspection_quality": "sufficient",
                    }
                ],
                "interpretations": [],
                "counterevidence": [],
                "metadata_evidence": [],
                "failures": [],
                "limitations": context["limitations"],
                "artifact_references": [],
                "recommended_followup": [],
            },
        }


def _add_images_and_metadata(state: dict) -> Path:
    dataset = Path(state["dataset_info"]["root"])
    images = dataset / "images"
    images.mkdir()
    rows = ["id,label,site,path"]
    for index, (label, site, value) in enumerate(
        ((0, "A", 16), (0, "A", 48), (1, "B", 208), (1, "B", 240)), 1
    ):
        reference = f"images/s{index}.png"
        Image.new("RGB", (12, 12), (value, value, value)).save(
            dataset / reference
        )
        rows.append(f"{index},{label},{site},{reference}")
    (dataset / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return dataset


def _enable_visual_state(state: dict) -> None:
    state["identify_config"]["visual_discovery"] = {
        "sample_id_column": "id",
        "sample_path_column": "path",
        "max_samples": 4,
    }
    state["executive_config"]["capabilities"].update(
        {
            "available_metadata_fields": ["id", "label", "site", "path"],
            "sample_access": True,
            "qualitative_checks": ["sample_inspection"],
        }
    )
    state["qualitative_config"]["preparation"].update(
        {
            "sampling_strategy": "group_balanced",
            "sample_path_column": "path",
            "group_columns": ["label"],
        }
    )


def test_full_metadata_visual_workflow_uses_real_evidence_backends(tmp_path: Path):
    state = _initial_state(tmp_path)
    state["run_id"] = "full-metadata-visual-workflow"
    dataset = _add_images_and_metadata(state)
    _enable_visual_state(state)
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

    result = run_workflow(
        state,
        agent_client=VisualEvidenceClient(),
        tool_runtimes=runtimes,
    )
    run_dir = Path(state["identify_config"]["runs_root"]) / state["run_id"]
    visual_path = run_dir / "identify" / "visual_discovery.json"
    visual = json.loads(visual_path.read_text(encoding="utf-8"))

    assert visual["coverage"]["decoded_samples"] == 4
    assert result["identify_result"]["candidates"][0]["evidence_sources"] == [
        "sample_inspection",
        "image_statistics",
    ]
    assert result["quantitative_result"]["results"][0]["payload"]["metrics"][
        "effect_size_name"
    ] == "cramers_v"
    assert (
        run_dir / "qualitative" / "TASK-C001-QL" / "inspection.json"
    ).is_file()
    assert (run_dir / "evidence" / "ledger.json").is_file()
    assert result["final_result"]["message_type"] == "summary_report"


def test_full_metadata_free_visual_workflow_is_legal_and_honest(tmp_path: Path):
    state = _initial_state(tmp_path)
    state["run_id"] = "full-metadata-free-visual-workflow"
    dataset = Path(state["dataset_info"]["root"])
    (dataset / "metadata.csv").unlink()
    state["dataset_info"]["metadata_file"] = None
    for index, value in enumerate((0, 30, 220, 255), 1):
        Image.new("RGB", (12, 12), (value, value, value)).save(
            dataset / f"sample-{index}.png"
        )
    state["identify_config"]["visual_discovery"] = {"max_samples": 4}
    state["executive_config"]["capabilities"].update(
        {
            "available_metadata_fields": [],
            "sample_access": True,
            "qualitative_checks": ["sample_inspection"],
        }
    )
    state["qualitative_config"]["preparation"] = {
        "seed": 42,
        "sampling_strategy": "random",
        "max_evidence_records": 4,
        "max_sample_candidates": 20,
    }

    result = run_workflow(state, agent_client=MetadataFreeVisualClient())
    visual = result["identify_context"]["visual_discovery"]

    assert visual["coverage"]["decoded_samples"] == 4
    assert any(
        "label_grouping_unavailable" in limitation
        for limitation in visual["limitations"]
    )
    assert result["identify_result"]["dataset_summary"]["metadata_used"] is False
    assert result["final_result"]["message_type"] == "summary_report"
