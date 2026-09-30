"""No-network immutable Full Real Canary lifecycle tests."""

import json
from pathlib import Path

import pytest

from shortcutminer.evaluation.pilot.protocol import CanaryAttemptRecord, PilotRunRecord
from shortcutminer.evaluation.pilot.runner import PilotRunner, PilotRunnerError
from test_real_blind_pilot import _config


def _ready_runner(tmp_path: Path, monkeypatch, *, pilot_id="phase16-test"):
    config = _config(tmp_path).model_copy(update={"pilot_id": pilot_id})
    runner = PilotRunner(config)
    runner.dry_run()
    snapshot = runner._snapshot().model_copy(update={"repository_dirty": False})
    monkeypatch.setattr(runner, "_snapshot", lambda: snapshot)
    gates = {
        "legacy_task_negative_replay.json": {
            "status": "passed", "semantic_gate": "rejected", "source_pilot_id": "source",
        },
        "historical_candidate_binding_replay.json": {
            "status": "passed", "provider_call_count": 0,
            "source_pilot_id": "source", "site_device_batch_result": "unresolved",
            "fuzzy_fallback_count": 0, "generated_task_count": 0,
        },
        "current_contract_task_offline.json": {
            "status": "passed", "provider_call_count": 0,
            "binding_status": "resolved", "ready_step_count": 1,
            "mandatory_backend_attempt_count": 1,
            "successful_tool_result_count": 1, "real_backend": "real_cpu_v1",
            "quantitative_finalization": "passed",
            "task": {"metadata_fields": ["site"], "requested_checks": ["association_test"]},
        },
    }
    for name, payload in gates.items():
        (runner.pilot_root / name).write_text(json.dumps(payload), encoding="utf-8")
    return runner, snapshot


def test_failed_canary_is_immutable_and_same_snapshot_refused(tmp_path: Path, monkeypatch):
    runner, snapshot = _ready_runner(tmp_path, monkeypatch)
    attempted = []

    def fails(*args, **kwargs):
        directory = runner.pilot_root / "canary_attempts"
        running = list(directory.glob("*.running.json"))
        assert len(running) == 1
        attempted.append(running[0].read_bytes())
        raise RuntimeError("offline canary failure")

    monkeypatch.setattr(runner, "_execute", fails)
    with pytest.raises(RuntimeError, match="offline canary failure"):
        runner.canary()
    directory = runner.pilot_root / "canary_attempts"
    running_path, = directory.glob("*.running.json")
    failed_path, = directory.glob("*.failed.json")
    running = CanaryAttemptRecord.model_validate_json(running_path.read_text(encoding="utf-8"))
    failed = CanaryAttemptRecord.model_validate_json(failed_path.read_text(encoding="utf-8"))
    assert running_path.read_bytes() == attempted[0]
    assert running.status == "running"
    assert failed.status == "failed"
    assert failed.snapshot_id == snapshot.snapshot_id
    assert failed.failure_stage == "execution"
    assert failed.failure_category == "RuntimeError"
    assert not (runner.pilot_root / "canary.json").exists()
    failed_bytes = failed_path.read_bytes()
    with pytest.raises(PilotRunnerError, match="already attempted"):
        runner.canary()
    assert failed_path.read_bytes() == failed_bytes


def test_new_pilot_snapshot_can_record_a_new_completed_attempt(tmp_path: Path, monkeypatch):
    old, old_snapshot = _ready_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(old, "_execute", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("old failure")))
    with pytest.raises(RuntimeError):
        old.canary()
    old_failed, = (old.pilot_root / "canary_attempts").glob("*.failed.json")
    old_bytes = old_failed.read_bytes()

    new, snapshot = _ready_runner(tmp_path, monkeypatch, pilot_id="phase16-test-next")
    assert snapshot.snapshot_id != old_snapshot.snapshot_id
    monkeypatch.setattr(new, "_save_canary_quantitative_audit", lambda run_id: None)
    monkeypatch.setattr(new, "_execute", lambda generated, *, run_id, seed, bootstrap_seed, snapshot: PilotRunRecord(
        challenge_id=generated.challenge.challenge_id, challenge_kind="positive",
        repeat_index=0, run_id=run_id, runtime_seed=seed,
        bootstrap_seed=bootstrap_seed, status="completed",
        snapshot_id=snapshot.snapshot_id,
        public_challenge_hash=generated.challenge.public.public_dataset_hash,
    ))
    record = new.canary()
    assert record.status == "completed"
    assert (new.pilot_root / "canary.json").is_file()
    completed, = (new.pilot_root / "canary_attempts").glob("*.completed.json")
    assert CanaryAttemptRecord.model_validate_json(completed.read_text(encoding="utf-8")).status == "completed"
    assert old_failed.read_bytes() == old_bytes
