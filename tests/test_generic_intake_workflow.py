from pathlib import Path

import pytest

import shortcutminer.orchestrator as orchestrator_module
from shortcutminer.experiment.identify_executor import IdentifyExecutionError
from shortcutminer.runner import resume_run, run
from shortcutminer.runtime.mock import MockAgentClient
from shortcutminer.storage.dataset_profiles import load_dataset_profile
from shortcutminer.workflow.nodes import identify_node
from test_pipeline_mock import DeterministicAgentClient, _initial_state


def _generic_request(tmp_path: Path, run_id: str) -> dict:
    state = _initial_state(tmp_path)
    state["dataset_info"]["name"] = "totally_unseen_dataset_2026"
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
        "qualitative_config": state["qualitative_config"],
    }


def test_unknown_dataset_completes_full_five_agent_mock_workflow(tmp_path: Path):
    request = _generic_request(tmp_path, "generic-e2e-run")
    client = DeterministicAgentClient()

    result = run(request, agent_client=client)

    assert result.status == "COMPLETED"
    assert result.final_result["message_type"] == "summary_report"
    assert result.workflow_state["dataset_profile"]["dataset"]["name"] == (
        "totally_unseen_dataset_2026"
    )
    run_dir = Path(request["runs_root"]) / request["run_id"]
    assert (run_dir / "dataset" / "profile.json").is_file()
    identify_payload = client.calls[0]["payload"]
    assert identify_payload["request"]["dataset"]["name"] == (
        "totally_unseen_dataset_2026"
    )
    assert identify_payload["discovery_context"]["dataset_profile"] == (
        result.workflow_state["dataset_profile"]
    )
    assert all(
        candidate["status"] == "unverified"
        for candidate in result.workflow_state["identify_result"]["candidates"]
    )


class FailIdentifyOnceClient(DeterministicAgentClient):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def invoke_agent(self, agent_name, payload, *, system_prompt=None):
        if agent_name == "identify" and not self.failed:
            self.failed = True
            raise RuntimeError("planned identify interruption")
        return super().invoke_agent(
            agent_name, payload, system_prompt=system_prompt
        )


def test_resume_reuses_profile_without_rescanning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    request = _generic_request(tmp_path, "generic-profile-resume")
    client = FailIdentifyOnceClient()
    original = orchestrator_module.build_dataset_profile
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(orchestrator_module, "build_dataset_profile", counted)
    with pytest.raises(IdentifyExecutionError, match="planned identify interruption"):
        run(request, agent_client=client)
    assert calls == 1
    stored = load_dataset_profile(request["runs_root"], request["run_id"])
    assert stored.dataset.name == "totally_unseen_dataset_2026"

    result = resume_run(
        request["run_id"],
        runs_root=request["runs_root"],
        agent_client=client,
    )
    assert result.status == "COMPLETED"
    assert calls == 1


def test_metadata_free_profile_reaches_identify_without_crashing(tmp_path: Path):
    dataset_root = Path(__file__).parent / "fixtures" / "generic_dataset_no_metadata"
    state = _initial_state(tmp_path)
    state["run_id"] = "metadata-free-identify"
    state["dataset_info"] = {
        "name": "unseen_without_metadata",
        "root": str(dataset_root),
        "metadata_file": None,
    }

    def response(payload, _system_prompt):
        context = payload["discovery_context"]
        assert context["dataset_profile"]["metadata"]["available"] is False
        return {
            "candidates": [],
            "additional_limitations": [],
        }

    update = identify_node(
        state,
        client=MockAgentClient({"identify": response}),
    )
    assert update["identify_result"]["candidates"] == []
    assert any(
        "label_source_unavailable" in limitation
        for limitation in update["identify_result"]["limitations"]
    )
