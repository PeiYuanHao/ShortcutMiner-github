"""Existing audit CLI/graph with fake providers, real backends and replay monitoring."""
import io
import json
from pathlib import Path
import pytest
from rich.console import Console
from shortcutminer import cli
from shortcutminer.audit import build_audit_run_request
from shortcutminer.experiment.candidate_binding import resolve_candidate_binding
from shortcutminer.monitoring.event_listener import EventListener
from shortcutminer.monitoring.renderer import render_dashboard
from shortcutminer.protocol.audit import AuditRequest
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.runner import run, resume_run, build_local_tool_runtimes
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient
from test_phase16_three_candidate_chain import ThreeCandidateReasoning


class AuditReasoning(ThreeCandidateReasoning):
    def _identify(self, payload):
        result = super()._identify(payload)
        result["candidates"] = result["candidates"][:1] + [{
            "cue": {"family": "other", "name": "background context"},
            "description": "Potential background cue", "observation": "Dataset structure contains sample entries",
            "hypothesis": "Background might encode the label", "evidence_sources": ["dataset_structure"],
            "priority": "low", "metadata_field_ref": None, "additional_limitations": []}]
        return result


class QuantitativeOnly(AuditReasoning):
    def _executive(self, payload):
        result = super()._executive(payload)
        for decision in result["decisions"]:
            decision["route"], decision["qualitative"] = "quantitative", None
            decision["quantitative"]["requested_checks"] = ["association_test", "model_evaluation"]
        return result


class Transport:
    def __init__(self, reasoning=None, fail_summary=False):
        self.reasoning = reasoning or AuditReasoning()
        self.fail_summary = fail_summary
        self.calls = []

    def post_json(self, _url, *, headers, body, timeout):
        payload = next(json.loads(message["content"]) for message in body["messages"]
                       if message["role"] == "user" and message["content"].lstrip().startswith("{"))
        target = payload["request"]["target"]
        self.calls.append(target)
        if target == "summarize" and self.fail_summary:
            raise RuntimeError("Authorization: Bearer SECRET; raw Provider response")
        output = getattr(self.reasoning, f"_{target}")(payload)
        return {"choices": [{"message": {"content": json.dumps(output)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}}


def client(root, transport):
    return OpenAICompatibleClient(OpenAICompatibleModelConfig(model="fake", base_url="http://offline.invalid/v1",
        api_key_env="OFFLINE_KEY", model_call_budget={"max_total_calls": 50, "max_calls_per_agent": 25}),
        api_key="offline", transport=transport, runs_root=root)


def request_and_runtimes(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "metadata.csv").write_text("id,label,site\n1,0,A\n2,0,A\n3,1,B\n4,1,B\n", encoding="utf-8")
    request = build_audit_run_request(AuditRequest(dataset_path=str(dataset)), run_id="audit-progress", runs_root=tmp_path / "runs")
    request = request.model_copy(update={"identify_config": {**request.identify_config,
        "candidate_screening": {"signals": {"confidence_by_candidate": {"C002": 0.1}}}}})
    runtimes = build_local_tool_runtimes(run_id=request.run_id, dataset_root=dataset, runs_root=request.runs_root)
    return request, runtimes


def test_actual_audit_candidates_tasks_steps_and_provider_telemetry(tmp_path):
    request, runtimes = request_and_runtimes(tmp_path)
    transport = Transport()
    result = run(request, agent_client=client(request.runs_root, transport), tool_runtimes=runtimes)
    state = EventListener(request.runs_root, request.run_id).poll()
    assert state.status == "completed"
    assert len(state.candidates) == 2 and len(state.tasks) == 2
    assert state.candidates["C002"].decision == "screened_out"
    assert "weak_signal" in state.candidates["C002"].reasons
    assert all(task.status == "completed" for task in state.tasks.values())
    assert len(state.tasks["TASK-C001-QT"].steps) == 1
    assert state.stages["visual_discovery"] == "skipped"
    assert state.stages["evidence_calibration"] == state.stages["audit_report"] == "completed"
    assert state.provider_calls["text_agent"] == 5 and state.total_tokens == 90
    assert len(state.tools) == 1
    assert result.workflow_state["identify_result"]["candidates"][1]["status"] == "unverified"


def test_multiple_plan_steps_with_no_qualitative_tasks(tmp_path):
    request, runtimes = request_and_runtimes(tmp_path)
    # Explicit fixture-only promotion rule; production default has no thresholds.
    request = request.model_copy(update={"metadata": {**request.metadata, "promotion_policy": {
        "policy_version": "progress-fixture-v1", "calibration_status": "test_only", "minimum_samples": 4,
        "metric_rules": [{"experiment_type": "statistical_check", "metric": "effect_size", "operator": "ge", "threshold": 0.2}]}}})
    run(request, agent_client=client(request.runs_root, Transport(QuantitativeOnly())), tool_runtimes=runtimes)
    state = EventListener(request.runs_root, request.run_id).poll()
    assert state.stages["qualitative"] == state.agents["qualitative"] == "skipped"
    task = state.tasks["TASK-C001-QT"]
    assert len(task.steps) == 2 and {step.experiment_type for step in task.steps.values()} == {"statistical_check", "model_evaluation"}
    assert all(step.status == "completed" for step in task.steps.values())
    assert state.agent_calls.get("qualitative") is None


def test_actual_checkpoint_resume_reconstruction_and_readonly_database(tmp_path):
    request, runtimes = request_and_runtimes(tmp_path)
    with pytest.raises(Exception):
        run(request, agent_client=client(request.runs_root, Transport(fail_summary=True)), tool_runtimes=runtimes)
    root = Path(request.runs_root) / request.run_id
    # Force the monitor to recover committed graph authority instead of exported JSON.
    (root / "workflow_state.json").unlink()
    (root / "executive" / "routing.json").unlink()
    database = root / "checkpoints.sqlite"
    original = database.read_bytes()
    state = EventListener(request.runs_root, request.run_id).poll()
    assert state.planning_complete and len(state.tasks) == 2
    assert all(task.status == "completed" for task in state.tasks.values())
    assert state.stages["identify"] == "completed" and state.agents["summarize"] == "failed"
    assert database.read_bytes() == original
    output = io.StringIO()
    Console(file=output, width=180).print(render_dashboard(state))
    assert "SECRET" not in output.getvalue() and "Authorization" not in output.getvalue()
    recovery = Transport()
    result = resume_run(request.run_id, runs_root=request.runs_root, agent_client=client(request.runs_root, recovery), tool_runtimes=runtimes)
    assert result.status == "COMPLETED" and recovery.calls == ["summarize"]
    restored = EventListener(request.runs_root, request.run_id).poll()
    assert restored.status == "completed" and restored.total_tokens is None
    assert restored.provider_calls["text_agent"] >= 6  # Failed calls had no token usage.


@pytest.mark.parametrize("enabled", [True, False])
def test_audit_cli_defaults_live_and_no_progress_preserves_science(tmp_path, monkeypatch, capsys, enabled):
    config = tmp_path / "model.json"
    config.write_text(json.dumps({"model": "fake", "base_url": "http://offline.invalid/v1", "api_key_env": "OFFLINE_KEY"}), encoding="utf-8")
    monkeypatch.setenv("SHORTCUTMINER_MODEL_CONFIG", str(config))
    transport = Transport()
    monkeypatch.setattr(cli, "_text_client_arguments", lambda config, *, runs_root: {"agent_client": client(runs_root, transport)})
    args = ["audit", "tests/fixtures/generic_dataset_a", "--run-id", "cli-audit-monitor", "--runs-root", str(tmp_path / "runs")]
    if not enabled:
        args.append("--no-progress")
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert ("[ShortcutMiner]" in captured.err) == enabled
    assert "\x1b" not in captured.err
    assert "SECRET" not in captured.err
    assert [json.loads(line)["event"] for line in captured.out.splitlines()] == ["started", "completed"]
    state = EventListener(tmp_path / "runs", "cli-audit-monitor").poll()
    assert state.stages["audit_intake"] == "completed" and state.status == "completed"
    assert state.total_tokens == 90 and state.provider_calls["text_agent"] == 5
    assert "device" not in {field for task in state.tasks.values() for field in task.requested_checks}


def test_audit_cli_failure_is_sanitized_and_artifacts_preserved(tmp_path, monkeypatch, capsys):
    request, _ = request_and_runtimes(tmp_path)
    config = tmp_path / "model.json"
    config.write_text(json.dumps({"model": "fake", "base_url": "http://offline.invalid/v1", "api_key_env": "OFFLINE_KEY"}), encoding="utf-8")
    monkeypatch.setenv("SHORTCUTMINER_MODEL_CONFIG", str(config))
    monkeypatch.setattr(cli, "_text_client_arguments", lambda config, *, runs_root: {"agent_client": client(runs_root, Transport(fail_summary=True))})
    assert cli.main(["audit", request.dataset_info.root, "--runs-root", request.runs_root, "--run-id", request.run_id]) == 1
    output = capsys.readouterr().err
    assert "SECRET" not in output and "Authorization" not in output and "Traceback" not in output
    assert "Artifacts preserved" in output
    assert (Path(request.runs_root) / request.run_id / "checkpoints.sqlite").is_file()


@pytest.mark.parametrize("field", ["id", "label", "path", "split"])
def test_raw_available_fields_never_authorize_control_selector(field):
    from shortcutminer.protocol.identify import Candidate
    candidate = Candidate(candidate_id="C001", cue={"family": "metadata", "name": field}, metadata_field_ref=field,
        description="Candidate", observation="Public control field", hypothesis="Unvalidated", evidence_sources=["metadata"],
        priority="low", status="unverified")
    binding = resolve_candidate_binding(candidate, available_metadata_fields=["id", "label", "path", "split", "site"],
                                        label_column="label", addressing_fields=["id", "path", "split"])
    assert binding.status == "unresolved" and binding.metadata_fields == []
