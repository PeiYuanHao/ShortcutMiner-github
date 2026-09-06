"""Formal Runner, lifecycle storage, checkpoint, and resume integration."""

import json
from collections import Counter
from pathlib import Path
from threading import Event

import pytest

import shortcutminer.orchestrator as orchestrator_module
import shortcutminer.runner as runner_module
from shortcutminer.experiment.qualitative_executor import QualitativeExecutionError
from shortcutminer.orchestrator import execute_workflow_state
from shortcutminer.runner import resume_run, run
from shortcutminer.storage.run_store import load_run_manifest, load_run_record
from shortcutminer.workflow.checkpoint import (
    checkpoint_config,
    checkpoint_path,
    open_sqlite_checkpointer,
)
from test_pipeline_mock import DeterministicAgentClient, _initial_state


def _request(tmp_path: Path, run_id: str = "phase4_formal_run") -> dict:
    state = _initial_state(tmp_path)
    return {
        "run_id": run_id,
        "runs_root": state["identify_config"]["runs_root"],
        "dataset_info": state["dataset_info"],
        "task_input": state["task_input"],
        "seed": state["identify_config"]["seed"],
        "identify_config": {
            key: value
            for key, value in state["identify_config"].items()
            if key not in {"seed", "runs_root"}
        },
        "executive_config": state["executive_config"],
        "quantitative_config": state.get("quantitative_config", {}),
        "qualitative_config": state["qualitative_config"],
        "summarize_config": state.get("summarize_config", {}),
        "metadata": {"phase": 4},
    }


def _events(runs_root: Path, run_id: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (runs_root / run_id / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_runner_validates_request_generates_run_id_and_calls_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    request = _request(tmp_path)
    request["run_id"] = None
    captured = {}
    marker = object()

    def fake_execute_run(parsed, **clients):
        captured["request"] = parsed
        captured["clients"] = clients
        return marker

    monkeypatch.setattr(runner_module, "execute_run", fake_execute_run)

    assert runner_module.run(request) is marker
    assert captured["request"].run_id.startswith("run-")
    assert captured["request"].dataset_info.name == "mock_dataset"


def test_runner_loads_and_validates_yaml_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    request = _request(tmp_path, run_id="phase4_yaml_run")
    config_path = tmp_path / "run.yaml"
    import yaml

    config_path.write_text(
        yaml.safe_dump(request, sort_keys=False),
        encoding="utf-8",
    )
    captured = {}

    def fake_execute_run(parsed, **_clients):
        captured["request"] = parsed
        return object()

    monkeypatch.setattr(runner_module, "execute_run", fake_execute_run)

    runner_module.run(config_path)

    assert captured["request"].run_id == "phase4_yaml_run"
    assert captured["request"].seed == 42


def test_orchestrator_compiles_and_invokes_langgraph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    state = _initial_state(tmp_path)
    state["run_id"] = "orchestrator_graph_bridge"
    captured = {}

    class FakeWorkflow:
        def invoke(self, graph_input, *, config):
            captured["input"] = graph_input
            captured["config"] = config
            final_result = {
                "protocol_version": "1.0",
                "run_id": graph_input["run_id"],
                "message_id": "orchestrator_graph_bridge:summarize:report",
                "source": "summarize",
                "target": "orchestrator",
                "message_type": "summary_report",
                "in_reply_to": "orchestrator_graph_bridge:summarize:request",
                "payload": {
                    "overview": "The fake graph completed.",
                    "findings": [],
                    "limitations": [],
                    "artifact_references": [],
                },
            }
            return {**graph_input, "final_result": final_result}

    def fake_build_workflow(*, checkpointer, **clients):
        captured["checkpointer"] = checkpointer
        captured["clients"] = clients
        return FakeWorkflow()

    monkeypatch.setattr(
        orchestrator_module,
        "build_workflow",
        fake_build_workflow,
    )

    result = execute_workflow_state(
        state,
        run_id=state["run_id"],
        runs_root=state["identify_config"]["runs_root"],
        resume=False,
    )

    assert result.status == "COMPLETED"
    assert captured["input"] == state
    assert captured["config"] == checkpoint_config(state["run_id"])
    assert captured["checkpointer"] is not None


def test_formal_runner_executes_five_agents_and_persists_run(tmp_path: Path):
    request = _request(tmp_path)
    client = DeterministicAgentClient()

    result = run(request, agent_client=client)

    runs_root = Path(request["runs_root"])
    run_dir = runs_root / request["run_id"]
    assert result.status == "COMPLETED"
    assert result.final_result["message_type"] == "summary_report"
    assert [call["agent_name"] for call in client.calls[:2]] == [
        "identify",
        "executive",
    ]
    assert {call["agent_name"] for call in client.calls[2:4]} == {
        "quantitative",
        "qualitative",
    }
    assert client.calls[-1]["agent_name"] == "summarize"
    assert load_run_record(runs_root, request["run_id"]).status == "COMPLETED"
    assert load_run_manifest(runs_root, request["run_id"]).status == "COMPLETED"
    for relative_path in (
        "run.json",
        "input.json",
        "manifest.json",
        "events.jsonl",
        "workflow_state.json",
        "final_result.json",
        "checkpoints.sqlite",
        "identify/output.json",
        "executive/routing.json",
        "quantitative/TASK-C001-QT/result.json",
        "qualitative/TASK-C001-QL/result.json",
        "summarize/result.json",
    ):
        assert (run_dir / relative_path).is_file(), relative_path
    event_names = [event["event"] for event in _events(runs_root, request["run_id"])]
    assert event_names[0] == "RUN_STARTED"
    assert event_names[-1] == "RUN_COMPLETED"


class FailQualitativeOnceClient(DeterministicAgentClient):
    """Fail after Quantitative succeeds, then allow checkpoint resume."""

    def __init__(self) -> None:
        super().__init__()
        self._quantitative_completed = Event()
        self._failed_once = False

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
        if agent_name == "quantitative":
            output = self._quantitative(payload)
            with self._lock:
                self.completed.add(agent_name)
            self._quantitative_completed.set()
            return output
        if agent_name == "qualitative" and not self._failed_once:
            assert self._quantitative_completed.wait(timeout=5)
            self._failed_once = True
            raise RuntimeError("planned qualitative interruption")
        output = getattr(self, f"_{agent_name}")(payload)
        with self._lock:
            self.completed.add(agent_name)
        return output


def test_failure_is_persisted_and_resume_reuses_successful_nodes(tmp_path: Path):
    request = _request(tmp_path, run_id="phase4_resume_run")
    client = FailQualitativeOnceClient()
    runs_root = Path(request["runs_root"])

    with pytest.raises(
        QualitativeExecutionError,
        match="planned qualitative interruption",
    ):
        run(request, agent_client=client)

    failed_record = load_run_record(runs_root, request["run_id"])
    assert failed_record.status == "FAILED"
    assert failed_record.error.error_type == "QualitativeExecutionError"
    failed_manifest = load_run_manifest(runs_root, request["run_id"])
    assert failed_manifest.status == "FAILED"
    assert failed_manifest.error.error_type == "QualitativeExecutionError"
    run_dir = runs_root / request["run_id"]
    assert (run_dir / "identify" / "output.json").is_file()
    assert (run_dir / "workflow_state.json").is_file()
    assert (
        run_dir / "quantitative" / "TASK-C001-QT" / "result.json"
    ).is_file()
    database_path = checkpoint_path(runs_root, request["run_id"])
    assert database_path.is_file()
    with open_sqlite_checkpointer(database_path) as saver:
        assert saver.get_tuple(checkpoint_config(request["run_id"])) is not None

    resumed = resume_run(
        request["run_id"],
        runs_root=runs_root,
        agent_client=client,
    )

    assert resumed.status == "COMPLETED"
    counts = Counter(call["agent_name"] for call in client.calls)
    assert counts == {
        "identify": 1,
        "executive": 1,
        "quantitative": 1,
        "qualitative": 2,
        "summarize": 1,
    }
    completed_record = load_run_record(runs_root, request["run_id"])
    assert completed_record.status == "COMPLETED"
    assert completed_record.attempt_count == 2
    assert completed_record.error is None
    event_names = [event["event"] for event in _events(runs_root, request["run_id"])]
    assert "RUN_FAILED" in event_names
    assert "RUN_RESUMED" in event_names
    assert event_names[-1] == "RUN_COMPLETED"

    resume_run(
        request["run_id"],
        runs_root=runs_root,
        agent_client=client,
    )
    assert Counter(call["agent_name"] for call in client.calls) == counts
