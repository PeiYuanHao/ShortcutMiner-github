import json

import pytest

from shortcutminer.experiment.audit_report import build_audit_report, export_audit_report
from shortcutminer.runner import resume_run, run
from shortcutminer.storage.run_store import load_run_record
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient
from test_audit_report import completed_audit
from test_audit_progress import request_and_runtimes, Transport, client


def test_safe_actual_runtime_snapshot_and_cost(completed_audit):
    request, _, _, _ = completed_audit
    report = build_audit_report(request.runs_root, request.run_id)
    snapshot = report.runtime["snapshots"][0]
    assert snapshot["model"] == "fake" and snapshot["provider"] == "openai_compatible"
    assert snapshot["endpoint"]["host"] == "offline.invalid"
    assert snapshot["temperature"] == 0.0 and snapshot["max_tokens"] == 4096
    assert snapshot["model_seed"] is None and snapshot["reasoning"] is None
    assert snapshot["model_call_budget"]["max_total_calls"] == 50
    assert snapshot["retry_policy"]["max_attempts"] == 3
    assert "api_key_env" not in snapshot and "offline" != snapshot.get("api_key")
    assert report.cost["text_provider_calls"] == 5 and report.cost["total_tokens"] == 90
    assert report.cost["tool_calls"] == 1 and report.cost["vlm_calls"] == 0


def test_real_seeds_budgets_and_plan_identity(completed_audit):
    request, result, _, _ = completed_audit
    report = build_audit_report(request.runs_root, request.run_id)
    repro = report.reproducibility
    assert repro["seed"] == request.seed == 42
    assert repro["profile_sampling"] == result.workflow_state["dataset_profile"]["profile_metadata"]
    assert repro["intake_budget"] == request.metadata["audit_request"]["intake_budget"]
    assert repro["evidence_fingerprint"] == result.workflow_state["evidence_ledger"]["fingerprint"]
    assert repro["capability_hash"] == result.workflow_state["audit_capabilities"]["result_hash"]
    assert report.validation["tools"][0]["parameters"]["seed"] == 42
    assert report.validation["tools"][0]["timeout_seconds"] == 30.0
    assert report.qualitative[0]["payload"]["sampling"]["seed"] == 42
    assert repro["qualitative_preparation"]["TASK-C001-QL"]["max_evidence_records"] == 40


def test_export_is_idempotent_and_readonly_except_outputs(completed_audit, monkeypatch):
    request, _, root, transport = completed_audit
    before = {name: (root / name).read_bytes() for name in ("audit_report.json", "audit_report.md", "events.jsonl", "checkpoints.sqlite")}
    monkeypatch.setenv("UNRELATED_SECRET", "SENTINEL_CHANGED_ENV")
    for _ in range(2):
        export_audit_report(request.runs_root, request.run_id)
    assert all((root / name).read_bytes() == content for name, content in before.items())
    assert len(transport.calls) == 5


def test_completed_resume_rebuilds_missing_report_without_work(completed_audit):
    request, _, root, _ = completed_audit
    before = (root / "audit_report.json").read_bytes(), (root / "audit_report.md").read_bytes()
    (root / "audit_report.json").unlink()
    (root / "audit_report.md").unlink()
    transport = Transport()
    resume_run(request.run_id, runs_root=request.runs_root, agent_client=client(request.runs_root, transport))
    assert transport.calls == []
    assert before == ((root / "audit_report.json").read_bytes(), (root / "audit_report.md").read_bytes())


def test_failed_then_resumed_audit_uses_persisted_upstream(tmp_path):
    request, runtimes = request_and_runtimes(tmp_path)
    with pytest.raises(Exception):
        run(request, agent_client=client(request.runs_root, Transport(fail_summary=True)), tool_runtimes=runtimes)
    transport = Transport()
    result = resume_run(request.run_id, runs_root=request.runs_root, agent_client=client(request.runs_root, transport), tool_runtimes=runtimes)
    assert transport.calls == ["summarize"]
    report = build_audit_report(request.runs_root, request.run_id)
    assert report.findings == result.final_result["payload"]
    assert report.cost["total_tokens"] is None  # Failed invocation usage was not recorded.
    assert report.cost["text_provider_calls"] >= 6
    assert "SECRET" not in report.model_dump_json()


def test_export_io_failure_preserves_scientific_completion(tmp_path, monkeypatch):
    request, runtimes = request_and_runtimes(tmp_path)
    from shortcutminer import orchestrator
    original = orchestrator.export_audit_report
    def fail_export(*args, **kwargs):
        raise OSError("report output unavailable")
    monkeypatch.setattr(orchestrator, "export_audit_report", fail_export)
    with pytest.raises(OSError, match="report output unavailable"):
        run(request, agent_client=client(request.runs_root, Transport()), tool_runtimes=runtimes)
    assert load_run_record(request.runs_root, request.run_id).status == "COMPLETED"
    monkeypatch.setattr(orchestrator, "export_audit_report", original)
    transport = Transport()
    resume_run(request.run_id, runs_root=request.runs_root, agent_client=client(request.runs_root, transport))
    assert transport.calls == []
    assert build_audit_report(request.runs_root, request.run_id).run["status"] == "COMPLETED"


def test_model_seed_and_reasoning_settings_are_recorded_without_secret(tmp_path):
    request, runtimes = request_and_runtimes(tmp_path)
    config = OpenAICompatibleModelConfig(model="seeded-fake", base_url="http://offline.invalid/v1",
        api_key_env="UNREAD_SECRET_ENV", temperature=0.2, max_tokens=512, model_seed=123,
        capabilities={"supports_model_seed": True, "supports_reasoning_config": True},
        reasoning={"mode": "enabled", "effort": "high"})
    provider = OpenAICompatibleClient(config, api_key="SENTINEL_KEY_VALUE", transport=Transport(), runs_root=request.runs_root)
    run(request, agent_client=provider, tool_runtimes=runtimes)
    report = build_audit_report(request.runs_root, request.run_id)
    snapshot = report.runtime["snapshots"][0]
    assert snapshot["temperature"] == 0.2 and snapshot["max_tokens"] == 512 and snapshot["model_seed"] == 123
    assert snapshot["reasoning"] == {"mode": "enabled", "effort": "high", "wire_format": "thinking"}
    assert "SENTINEL_KEY_VALUE" not in report.model_dump_json() and "UNREAD_SECRET_ENV" not in report.model_dump_json()
