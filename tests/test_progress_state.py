"""Read-only progress projections from actual IDs and persisted public facts."""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from shortcutminer.monitoring.event_listener import EventListener
from shortcutminer.monitoring.progress import AuditStage, ProgressState, TaskStatus, PlanStepStatus
from shortcutminer.storage.events import append_event
from test_monitoring import event, invocation


def write(root, relative, payload):
    path = root / "monitor" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def public_artifacts(root):
    write(root, "identify/output.json", {"run_id": "monitor", "candidates": [
        {"candidate_id": "C001", "cue": {"name": "collection site"}},
        {"candidate_id": "C002", "cue": {"name": "device bias"}}]})
    write(root, "screening/result.json", {"run_id": "monitor", "results": [
        {"candidate_id": "C001", "decision": "promote", "reason": ["sufficient_samples"]},
        {"candidate_id": "C002", "decision": "screened_out", "reason": ["weak_signal"]}]})
    write(root, "capabilities/result.json", {"run_id": "monitor", "capabilities": {"available_metadata_fields": ["site"]}})
    write(root, "executive/routing.json", {"run_id": "monitor", "tasks": [{"task_id": "TASK-C001-QT",
        "candidate_id": "C001", "target": "quantitative", "requested_checks": ["association_test", "model_evaluation"]}]})
    write(root, "quantitative/TASK-C001-QT/experiment_plan.json", {"task_id": "TASK-C001-QT", "steps": [
        {"step_id": "QSTEP-a", "experiment_type": "statistical_check", "planning_check": "association_test", "execution_state": "ready"},
        {"step_id": "QSTEP-b", "experiment_type": "model_evaluation", "planning_check": "model_evaluation", "execution_state": "ready"}]})


def test_fresh_audit_has_thirteen_pending_stages_and_five_waiting_agents():
    state = ProgressState("monitor")
    assert set(state.stages) == set(AuditStage)
    assert set(state.stages.values()) == {"pending"}
    assert len(state.agents) == 5 and set(state.agents.values()) == {"waiting"}
    assert state.progress == 0 and state.total_tokens is None


def test_identify_completion_and_agent_call_duration():
    state = ProgressState.reconstruct("monitor", [event("IDENTIFY_STARTED", "identify"),
        event("MODEL_INVOCATION_STARTED", "identify", {"runtime_kind": "text_agent"}),
        event("IDENTIFY_COMPLETED", "identify", second=9)])
    assert state.stages["identify"] == state.agents["identify"] == "completed"
    assert state.agent_calls["identify"] == 1 and state.agent_seconds("identify") == 9
    assert state.agents["executive"] == "waiting"


def test_all_candidates_subset_tasks_and_skipped_qualitative(tmp_path):
    public_artifacts(tmp_path)
    state = EventListener(tmp_path, "monitor").poll()
    assert len(state.candidates) == 2 and len(state.tasks) == 1
    assert state.candidates["C002"].decision == "screened_out"
    assert state.candidates["C002"].reasons == ["weak_signal"]
    assert state.stages["qualitative"] == state.agents["qualitative"] == "skipped"
    assert state.capability_available


def test_quantitative_steps_follow_ids_not_order_and_failure_category(tmp_path):
    public_artifacts(tmp_path)
    listener = EventListener(tmp_path, "monitor")
    listener.poll()
    details = {"candidate_id": "C001", "task_id": "TASK-C001-QT", "plan_step_id": "QSTEP-b",
               "tool_call_id": "second", "tool": "run_experiment", "experiment_type": "model_evaluation"}
    append_event(tmp_path, "monitor", "TASK_STARTED", stage="quantitative", details=details)
    append_event(tmp_path, "monitor", "TOOL_STARTED", stage="tool:quantitative", details=details)
    state = listener.poll()
    assert state.tasks["TASK-C001-QT"].steps["QSTEP-b"].status == "running"
    assert state.tasks["TASK-C001-QT"].steps["QSTEP-a"].status == "pending"
    assert "QSTEP-b" in state.current_activity
    append_event(tmp_path, "monitor", "TOOL_FAILED", stage="tool:quantitative", details=details)
    append_event(tmp_path, "monitor", "TASK_FAILED", stage="quantitative", details={**details, "reason": "backend_execution_failed", "message": "SECRET"})
    state = listener.poll()
    assert state.tasks["TASK-C001-QT"].status == "failed"
    assert state.tasks["TASK-C001-QT"].reason == "backend_execution_failed"


def test_weights_dynamic_for_new_tasks_and_parallel_branches():
    state = ProgressState("monitor")
    state.set_plan([{"task_id": "TASK-C001-QT", "candidate_id": "C001", "target": "quantitative", "requested_checks": []}])
    task = state.tasks["TASK-C001-QT"]
    task.steps = {"a": PlanStepStatus("a", "statistical_check", status="completed"), "b": PlanStepStatus("b", "model_evaluation")}
    assert state.progress == 0.175
    state.set_plan([{"task_id": "TASK-C002-QT", "candidate_id": "C002", "target": "quantitative", "requested_checks": []}])
    assert state.progress == 0.0875
    state.tasks["TASK-C002-QT"].status = "completed"
    assert state.progress == 0.2625


def test_evidence_summary_and_report_milestones():
    state = ProgressState.reconstruct("monitor", [
        event("STAGE_COMPLETED", "evidence_calibration"), event("SUMMARIZE_COMPLETED", "summarize"),
        event("STAGE_COMPLETED", "audit_report"), event("RUN_COMPLETED")])
    assert all(state.stages[key] == "completed" for key in ("evidence_calibration", "summarize", "audit_report"))
    assert state.progress == 1.0


def test_reconstruction_toolresults_checkpoint_and_monitor_does_not_mutate(tmp_path):
    public_artifacts(tmp_path)
    snapshot = {"run_id": "monitor", "stage_status": {"identify": "completed", "screening": "completed",
                "executive": "completed", "feasibility": "completed"}, "future_result": {"hidden_truth": "NEVER_RENDER"}}
    path = write(tmp_path, "workflow_state.json", snapshot)
    original = path.read_bytes()
    details = {"task_id": "TASK-C001-QT", "candidate_id": "C001"}
    append_event(tmp_path, "monitor", "RUN_STARTED")
    append_event(tmp_path, "monitor", "TASK_STARTED", stage="quantitative", details=details)
    write(tmp_path, "tools/quantitative/qt.json", {"run_id": "monitor", "tool_call_id": "qt", "agent_name": "quantitative",
        "tool_name": "run_experiment", "candidate_id": "C001", "task_id": "TASK-C001-QT", "plan_step_id": "QSTEP-a",
        "experiment_type": "statistical_check", "status": "success", "output": {"reasoning": "SECRET"},
        "completed_at": datetime.now(timezone.utc).isoformat(), "duration_ms": 20})
    first = EventListener(tmp_path, "monitor").poll()
    second = EventListener(tmp_path, "monitor").poll()
    assert first.tasks == second.tasks and first.progress == second.progress
    assert second.tasks["TASK-C001-QT"].steps["QSTEP-a"].status == "completed"
    assert second.tasks["TASK-C001-QT"].steps["QSTEP-b"].status == "pending"
    assert second.stages["feasibility"] == "completed"
    assert path.read_bytes() == original and json.loads(original) == snapshot


def test_missing_usage_stays_na_and_complete_usage_is_not_double_counted():
    state = ProgressState("monitor")
    state.update_telemetry([invocation("missing", available=False)])
    assert state.total_tokens is None
    record = invocation("complete")
    state.update_telemetry([record, record])
    assert state.total_tokens == 10 and state.provider_calls["text_agent"] == 1
    assert state.vlm_images is None


def test_old_toolresult_cannot_revert_current_retry():
    state = ProgressState("monitor")
    state.tasks["TASK-C001-QT"] = TaskStatus("TASK-C001-QT", "C001", "quantitative")
    common = {"task_id": "TASK-C001-QT", "plan_step_id": "QSTEP-a", "tool": "run_experiment"}
    state.apply(event("TOOL_STARTED", "tool:quantitative", {**common, "tool_call_id": "new"}, second=10))
    state.apply(event("TOOL_FAILED", "tool:quantitative", {**common, "tool_call_id": "old"}, second=1))
    assert state.tasks["TASK-C001-QT"].steps["QSTEP-a"].status == "running"


def test_telemetry_written_ahead_of_event_poll_cannot_double_count_calls():
    state = ProgressState("monitor")
    state.update_telemetry([invocation("one")])
    state.apply(event("MODEL_INVOCATION_STARTED", "identify", {"runtime_kind": "text_agent"}))
    state.update_telemetry([invocation("one")])
    assert state.provider_calls["text_agent"] == state.agent_calls["identify"] == 1
