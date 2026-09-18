import inspect
import json
from pathlib import Path

import pytest

from shortcutminer.experiment.identify_executor import IdentifyExecutionError
from shortcutminer.experiment.identify_workspace import (
    IDENTIFY_WORKSPACE_FILES,
    load_identify_workspace_prompt,
)
from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.runner import run_workflow
from shortcutminer.workflow import nodes as nodes_module
from shortcutminer.workflow.nodes import (
    build_identify_request_from_state,
    identify_node,
)
from shortcutminer.workflow.state import ShortcutMinerState


class FakeAgentClient:
    def __init__(self, output: dict):
        self.output = output
        self.calls: list[dict] = []

    def invoke_agent(
        self,
        agent_name: str,
        payload: dict,
        *,
        system_prompt: str | None = None,
    ) -> dict:
        self.calls.append(
            {
                "agent_name": agent_name,
                "payload": payload,
                "system_prompt": system_prompt,
            }
        )
        return self.output


class FailingAgentClient:
    def invoke_agent(
        self,
        agent_name: str,
        payload: dict,
        *,
        system_prompt: str | None = None,
    ) -> dict:
        raise RuntimeError("deterministic Identify failure")


class WorkflowAgentClient(FakeAgentClient):
    """Keep the Phase 2 Identify fixture valid inside the Phase 3 graph."""

    def invoke_agent(
        self,
        agent_name: str,
        payload: dict,
        *,
        system_prompt: str | None = None,
    ) -> dict:
        if agent_name == "identify":
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
        request = payload["request"]
        if agent_name == "executive":
            candidate = request["candidate_report"]["candidates"][0]
            return {
                "decisions": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "route": "unresolved",
                        "routing_reason": "No downstream capabilities configured.",
                        "quantitative": None,
                        "qualitative": None,
                        "missing_requirements": [
                            "Downstream capabilities were not configured."
                        ],
                        "limitations": [],
                    }
                ],
                "limitations": [],
            }
        if agent_name == "summarize":
            candidate = request["identify_result"]["candidates"][0]
            return {
                "protocol_version": "1.0",
                "run_id": request["run_id"],
                "message_id": "identify_workflow_run:summarize:report",
                "source": "summarize",
                "target": "orchestrator",
                "message_type": "summary_report",
                "in_reply_to": request["message_id"],
                "payload": {
                    "overview": "Candidate validation remains unresolved.",
                    "findings": [
                        {
                            "candidate_id": candidate["candidate_id"],
                            "assessment": "insufficient_evidence",
                            "summary": "No validation task was planned.",
                            "quantitative_result_ids": [],
                            "qualitative_result_ids": [],
                            "limitations": ["No downstream capabilities."],
                            "recommendations": ["Configure validation capabilities."],
                        }
                    ],
                    "limitations": ["No downstream capabilities."],
                    "artifact_references": [],
                },
            }
        raise AssertionError(f"Unexpected agent call: {agent_name}")


def _create_dataset(root: Path) -> None:
    root.mkdir()
    train_dir = root / "train"
    train_dir.mkdir()
    for sample_id in range(1, 5):
        (train_dir / f"sample_{sample_id}.jpg").write_bytes(b"test")

    (root / "metadata.csv").write_text(
        "id,label,site\n"
        "1,0,A\n"
        "2,0,A\n"
        "3,1,B\n"
        "4,1,B\n",
        encoding="utf-8",
    )


def _state(tmp_path: Path) -> ShortcutMinerState:
    dataset_root = tmp_path / "dataset"
    _create_dataset(dataset_root)
    return ShortcutMinerState(
        run_id="identify_workflow_run",
        dataset_info={
            "name": "workflow_dataset",
            "root": str(dataset_root),
            "metadata_file": "metadata.csv",
        },
        task_input={
            "type": "classification",
            "label_column": "label",
            "description": "Identify workflow integration test.",
        },
        identify_config={
            "seed": 17,
            "message_id": "identify_workflow_run:identify:request",
            "budget": {
                "max_samples": 4,
                "max_candidates": 2,
                "max_dataset_entries": 100,
                "max_metadata_rows": 100,
            },
            "runs_root": str(tmp_path / "runs"),
        },
        errors=[],
        metadata={"phase": 2},
    )


def _agent_output() -> dict:
    return {
        "protocol_version": "1.0",
        "run_id": "identify_workflow_run",
        "message_id": "identify_workflow_run:identify:report",
        "source": "identify",
        "target": "executive",
        "message_type": "candidate_report",
        "in_reply_to": "identify_workflow_run:identify:request",
        "artifact_references": [],
        "dataset_summary": {
            "dataset": "workflow_dataset",
            "samples_inspected": 0,
            "metadata_used": True,
        },
        "candidates": [
            {
                "candidate_id": "C001",
                "cue": {"family": "metadata", "name": "site"},
                "description": "Collection site may be an unintended cue.",
                "observation": "Prepared metadata contains multiple sites.",
                "hypothesis": "Site may be spuriously associated with label.",
                "evidence_sources": ["metadata"],
                "priority": "medium",
                "status": "unverified",
            }
        ],
        "limitations": [
            "Selected records are metadata rows only; no dataset sample "
            "content was inspected."
        ],
    }


def test_identify_node_maps_workflow_state_to_existing_request(tmp_path: Path):
    state = _state(tmp_path)

    request = build_identify_request_from_state(state)

    assert request.run_id == state["run_id"]
    assert request.dataset.name == state["dataset_info"]["name"]
    assert request.task.description == state["task_input"]["description"]
    assert request.budget.max_samples == 4


def test_identify_node_calls_existing_executor_and_returns_minimal_update(
    tmp_path: Path,
):
    state = _state(tmp_path)
    client = FakeAgentClient(_agent_output())

    update = identify_node(state, client=client)

    assert set(update) == {
        "dataset_profile",
        "identify_request",
        "identify_context",
        "identify_result",
        "stage_status",
    }
    report = CandidateReport.model_validate(update["identify_result"])
    assert report.run_id == state["run_id"]
    assert report.candidates[0].candidate_id == "C001"
    assert client.calls[0]["agent_name"] == "identify"
    assert client.calls[0]["payload"]["discovery_context"]["seed"] == 17


def test_workflow_invokes_real_identify_and_preserves_state(
    tmp_path: Path,
):
    state = _state(tmp_path)
    client = WorkflowAgentClient(_agent_output())

    result = run_workflow(state, agent_client=client)

    report = CandidateReport.model_validate(result["identify_result"])
    assert report.message_type == "candidate_report"
    assert result["executive_result"]["message_type"] == "routing_plan"
    assert result["quantitative_result"]["results"] == []
    assert result["qualitative_result"]["results"] == []
    assert result["summarize_result"]["message_type"] == "summary_report"
    assert result["metadata"] == {"phase": 2}


def test_identify_workflow_persists_artifacts_and_events(tmp_path: Path):
    state = _state(tmp_path)
    client = WorkflowAgentClient(_agent_output())

    run_workflow(state, agent_client=client)

    run_dir = tmp_path / "runs" / state["run_id"]
    assert (run_dir / "identify" / "input.json").exists()
    assert (run_dir / "identify" / "raw_output.json").exists()
    output_path = run_dir / "identify" / "output.json"
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["run_id"] == (
        state["run_id"]
    )

    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    identify_events = [
        event["event"] for event in events if event.get("stage") == "identify"
    ]
    assert identify_events == [
        "IDENTIFY_STARTED",
        "IDENTIFY_COMPLETED",
    ]
    assert all(event["run_id"] == state["run_id"] for event in events)


def test_identify_workflow_failure_is_not_swallowed(tmp_path: Path):
    state = _state(tmp_path)

    with pytest.raises(
        IdentifyExecutionError,
        match="deterministic Identify failure",
    ):
        run_workflow(state, identify_client=FailingAgentClient())

    events_path = tmp_path / "runs" / state["run_id"] / "events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events[-2:]] == [
        "AGENT_FAILED",
        "RUN_FAILED",
    ]


def test_identify_workspace_prompt_preserves_all_documents_in_order():
    workspace_dir = Path(__file__).parents[1] / "agents" / "identify" / "workspace"
    prompt = load_identify_workspace_prompt(workspace_dir)
    positions = []

    for filename in IDENTIFY_WORKSPACE_FILES:
        content = (workspace_dir / filename).read_text(encoding="utf-8")
        assert content in prompt
        positions.append(prompt.index(content))

    assert positions == sorted(positions)


def test_identify_node_passes_workspace_prompt_to_agent_client(tmp_path: Path):
    state = _state(tmp_path)
    client = FakeAgentClient(_agent_output())

    identify_node(state, client=client)

    prompt = client.calls[0]["system_prompt"]
    assert "Identify Agent Operational Specification" in prompt
    assert "# Identify Agent Identity" in prompt


def test_identify_node_has_no_retired_platform_dependency():
    source = inspect.getsource(nodes_module).lower()

    assert "open" + "claw" not in source
