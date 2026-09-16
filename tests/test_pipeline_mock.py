"""Deterministic, network-free coverage for the complete five-agent graph."""

import json
from pathlib import Path
from threading import Barrier, Lock

import pytest

from shortcutminer.runner import run_workflow
from shortcutminer.runtime.demo import DemoAgentClient
from shortcutminer.runtime.mock import MockAgentClient
from shortcutminer.workflow.nodes import (
    DownstreamPlanFeasibilityError,
    ExecutiveNodeError,
    SummarizeNodeError,
    build_executive_request_from_state,
    build_summarize_request_from_state,
)
from shortcutminer.workflow.state import ShortcutMinerState


def _dataset(root: Path) -> None:
    root.mkdir()
    (root / "metadata.csv").write_text(
        "id,label,site\n1,0,A\n2,0,A\n3,1,B\n4,1,B\n",
        encoding="utf-8",
    )


def _initial_state(tmp_path: Path) -> ShortcutMinerState:
    dataset_root = tmp_path / "dataset"
    _dataset(dataset_root)
    return ShortcutMinerState(
        run_id="phase3_mock_run",
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
            "runs_root": str(tmp_path / "runs"),
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
        metadata={"phase": 3},
    )


class DeterministicAgentClient(DemoAgentClient):
    """Return protocol-valid outputs and require branch-time concurrency."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.completed: set[str] = set()
        self._lock = Lock()
        self._branch_barrier = Barrier(2, timeout=5)

    def invoke_agent(
        self,
        agent_name: str,
        payload: dict,
        *,
        system_prompt: str | None = None,
    ) -> dict:
        with self._lock:
            self.calls.append(
                {
                    "agent_name": agent_name,
                    "payload": payload,
                    "system_prompt": system_prompt,
                }
            )
        if agent_name in {"quantitative", "qualitative"}:
            self._branch_barrier.wait()

        output = getattr(self, f"_{agent_name}")(payload)
        with self._lock:
            self.completed.add(agent_name)
        return output

    def _summarize(self, payload: dict) -> dict:
        assert {"quantitative", "qualitative"}.issubset(self.completed)
        return super()._summarize(payload)


def test_complete_mock_workflow_is_parallel_typed_and_network_free(tmp_path: Path):
    state = _initial_state(tmp_path)
    client = DeterministicAgentClient()

    result = run_workflow(state, agent_client=client)

    assert result["final_result"] == result["summarize_result"]
    assert result["final_result"]["payload"]["findings"][0]["candidate_id"] == "C001"
    assert result["stage_status"] == {
        "identify": "completed",
        "executive": "completed",
        "feasibility": "completed",
        "quantitative": "completed",
        "qualitative": "completed",
        "summarize": "completed",
    }
    assert client.calls[0]["agent_name"] == "identify"
    assert client.calls[1]["agent_name"] == "executive"
    assert {call["agent_name"] for call in client.calls[2:4]} == {
        "quantitative",
        "qualitative",
    }
    assert client.calls[-1]["agent_name"] == "summarize"
    assert all(call["system_prompt"] for call in client.calls)
    json.dumps(result, allow_nan=False)
    run_dir = tmp_path / "runs" / state["run_id"]
    assert (
        run_dir / "quantitative" / "TASK-C001-QT" / "result.json"
    ).is_file()
    assert (
        run_dir / "qualitative" / "TASK-C001-QL" / "result.json"
    ).is_file()
    assert (run_dir / "summarize" / "result.json").is_file()


def test_packaged_mock_runtime_executes_complete_five_agent_workflow(
    tmp_path: Path,
):
    scripted = DeterministicAgentClient()

    def response_factory(agent_name: str):
        def respond(payload: dict, _system_prompt: str | None) -> dict:
            output = getattr(scripted, f"_{agent_name}")(payload)
            with scripted._lock:
                scripted.completed.add(agent_name)
            return output

        return respond

    client = MockAgentClient(
        {
            agent_name: response_factory(agent_name)
            for agent_name in (
                "identify",
                "executive",
                "quantitative",
                "qualitative",
                "summarize",
            )
        }
    )

    result = run_workflow(_initial_state(tmp_path), agent_client=client)

    assert result["final_result"]["message_type"] == "summary_report"
    assert {call["agent_name"] for call in client.calls} == {
        "identify",
        "executive",
        "quantitative",
        "qualitative",
        "summarize",
    }
    assert all(call["system_prompt"] for call in client.calls)


def test_both_branches_receive_the_executive_plan(tmp_path: Path):
    client = DeterministicAgentClient()
    run_workflow(_initial_state(tmp_path), agent_client=client)
    calls = {call["agent_name"]: call for call in client.calls}

    assert calls["quantitative"]["payload"]["request"]["task"]["task_id"] == (
        "TASK-C001-QT"
    )
    assert calls["qualitative"]["payload"]["request"]["task"]["task_id"] == (
        "TASK-C001-QL"
    )
    summary_request = calls["summarize"]["payload"]["request"]
    assert summary_request["identify_result"]["candidates"][0]["candidate_id"] == (
        "C001"
    )
    assert summary_request["executive_result"]["message_type"] == "routing_plan"
    assert len(summary_request["quantitative_result"]["results"]) == 1
    assert len(summary_request["qualitative_result"]["results"]) == 1


def test_executive_cannot_run_without_identify_state(tmp_path: Path):
    with pytest.raises(ExecutiveNodeError, match="Invalid Executive workflow state"):
        build_executive_request_from_state(_initial_state(tmp_path))


def test_summarize_requires_both_branch_results(tmp_path: Path):
    state = _initial_state(tmp_path)
    state["identify_result"] = {}
    state["executive_result"] = {}
    state["quantitative_result"] = {}

    with pytest.raises(SummarizeNodeError, match="complete branch state"):
        build_summarize_request_from_state(state)


def test_feasibility_gate_blocks_both_downstream_models(tmp_path: Path):
    state = _initial_state(tmp_path)
    state["qualitative_config"]["preparation"]["sample_id_column"] = "unknown_id"
    client = DeterministicAgentClient()

    with pytest.raises(DownstreamPlanFeasibilityError, match="unknown_id"):
        run_workflow(state, agent_client=client)

    assert [call["agent_name"] for call in client.calls] == [
        "identify", "executive"
    ]
