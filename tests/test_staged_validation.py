"""Real CPU evidence and fake providers through the single production graph."""
import json
from pathlib import Path
from collections import Counter
from types import SimpleNamespace

import pytest

from shortcutminer.audit import build_audit_run_request
from shortcutminer.protocol.audit import AuditRequest
from shortcutminer.runner import run, resume_run
from shortcutminer.experiment.real_executor import RealExperimentExecutor
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolRuntime, ToolBudget, RunToolBudgetState
from shortcutminer.tooling.protocol import ToolCall
from shortcutminer.workflow.staged_validation import guard_runtime
from shortcutminer.monitoring.event_listener import EventListener
from shortcutminer.monitoring.renderer import projection
from test_audit_progress import Transport, client
from test_phase16_three_candidate_chain import ThreeCandidateReasoning
from test_evidence_promotion import fixture_policy


class TwoCandidates(ThreeCandidateReasoning):
    def _identify(self, payload):
        result = super()._identify(payload)
        result["candidates"] = result["candidates"][:2]
        return result

    def _executive(self, payload):
        result = super()._executive(payload)
        for decision in result["decisions"]:
            decision["quantitative"]["requested_checks"] = ["association_test", "model_evaluation", "intervention"]
        return result


class SpyCPU(RealExperimentExecutor):
    def __init__(self):
        self.calls = []

    def execute(self, request, *, context=None):
        self.calls.append((request.candidate_id, request.experiment_type))
        return super().execute(request, context=context)


def staged_request(tmp_path, budget=8, configured=True):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rows = [f"{index},{index % 2},{(index // 2) % 2},{index % 2}" for index in range(64)]
    (dataset / "metadata.csv").write_text("id,label,site,device\n" + "\n".join(rows) + "\n")
    request = build_audit_run_request(AuditRequest(dataset_path=str(dataset)), runs_root=tmp_path / "runs", run_id="staged-audit")
    if configured:
        request = request.model_copy(update={"metadata": {**request.metadata, "promotion_policy": fixture_policy().model_dump(mode="json")}})
    executor = SpyCPU()
    registry = build_default_tool_registry(experiment_executor=executor)
    shared = RunToolBudgetState(budget)
    runtimes = {agent: ToolRuntime(registry, run_id=request.run_id, agent_name=agent, dataset_root=dataset,
        runs_root=request.runs_root, budget=ToolBudget(max_tool_calls=budget), run_budget_state=shared,
        restore_persisted_task_results=True) for agent in ("quantitative", "qualitative")}
    return request, runtimes, executor


@pytest.fixture
def completed_staged(tmp_path):
    request, runtimes, executor = staged_request(tmp_path)
    transport = Transport(TwoCandidates())
    result = run(request, agent_client=client(request.runs_root, transport), tool_runtimes=runtimes)
    return request, runtimes, executor, transport, result


def test_weak_and_strong_evidence_control_actual_expensive_calls(completed_staged):
    request, _, executor, transport, result = completed_staged
    assert result.workflow_state["screening_result"]["promoted"] == 2
    decisions = {item["candidate_id"]: item["decision"] for item in result.workflow_state["promotion_result"]["results"]}
    assert decisions == {"C001": "terminated_low_signal", "C002": "promoted"}
    assert Counter(executor.calls)[("C001", "statistical_check")] == Counter(executor.calls)[("C002", "statistical_check")] == 1
    for expensive in ("model_evaluation", "task_model_training", "intervention", "task_model_evaluation", "ood_evaluation", "counterfactual_intervention"):
        assert Counter(executor.calls)[("C001", expensive)] == 0
    assert Counter(executor.calls)[("C002", "model_evaluation")] == 1
    assert Counter(executor.calls)[("C002", "intervention")] == 1
    assert set(transport.calls) == {"identify", "executive", "quantitative", "qualitative", "summarize"}
    from shortcutminer.monitoring.progress import AGENTS
    assert len(AGENTS) == 5
    events = [json.loads(line) for line in (Path(request.runs_root) / request.run_id / "events.jsonl").read_text().splitlines()]
    gate = next(index for index, event in enumerate(events) if event["event"] == "PROMOTION_COMPLETED")
    assert all(index > gate for index, event in enumerate(events) if event["event"] == "TOOL_STARTED" and event["details"].get("experiment_type") in {"model_evaluation", "intervention"})


def test_stage1_runtime_rejects_all_expensive_tool_calls():
    calls = []
    runtime = SimpleNamespace(execute=lambda call: calls.append(call))
    guard_runtime(runtime, phase="stage1", step_ids={"CHEAP"})
    for expensive in ("task_model_training", "model_evaluation", "intervention", "ood_evaluation", "counterfactual_intervention"):
        call = ToolCall(tool_call_id="attempt-" + expensive, tool_name="run_experiment", agent_name="quantitative", run_id="gate",
            candidate_id="C001", task_id="TASK-C001-QT", plan_step_id="EXPENSIVE", experiment_type=expensive,
            arguments={"experiment_type": expensive})
        with pytest.raises(ValueError, match="forbidden"):
            runtime.execute(call)
    assert calls == []


def test_resume_after_promotion_does_not_repeat_stage1(tmp_path, monkeypatch):
    request, runtimes, executor = staged_request(tmp_path)
    import shortcutminer.workflow.staged_validation as controllers
    original = controllers.load_or_evaluate_promotion
    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("fixture interruption after frozen promotion")
    monkeypatch.setattr(controllers, "load_or_evaluate_promotion", interrupted)
    initial = Transport(TwoCandidates())
    with pytest.raises(Exception, match="fixture interruption"):
        run(request, agent_client=client(request.runs_root, initial), tool_runtimes=runtimes)
    root = Path(request.runs_root) / request.run_id
    frozen = (root / "promotion/result.json").read_bytes()
    assert len(executor.calls) == 2
    monkeypatch.setattr(controllers, "load_or_evaluate_promotion", original)
    recovery = Transport(TwoCandidates())
    result = resume_run(request.run_id, runs_root=request.runs_root, agent_client=client(request.runs_root, recovery), tool_runtimes=runtimes)
    assert recovery.calls == ["quantitative", "summarize"]
    assert frozen == (root / "promotion/result.json").read_bytes()
    assert sum(kind == "statistical_check" for _, kind in executor.calls) == 2
    assert result.status == "COMPLETED"


def test_budget_stays_enforced_after_strong_promotion(tmp_path):
    request, runtimes, executor = staged_request(tmp_path, budget=2)
    with pytest.raises(Exception):
        run(request, agent_client=client(request.runs_root, Transport(TwoCandidates())), tool_runtimes=runtimes)
    assert len(executor.calls) == 2 and all(kind == "statistical_check" for _, kind in executor.calls)
    gate = json.loads((Path(request.runs_root) / request.run_id / "promotion/result.json").read_text())
    assert gate["results"][1]["decision"] == "promoted"
    assert runtimes["quantitative"].run_budget_state.used == 2
    assert json.loads((Path(request.runs_root) / request.run_id / "run.json").read_text())["status"] == "FAILED"


def test_report_and_progress_keep_both_gates_and_science_separate(completed_staged):
    request, _, _, _, result = completed_staged
    root = Path(request.runs_root) / request.run_id
    report = json.loads((root / "audit_report.json").read_text())
    assert report["screening"]["results"][0]["decision"] == "promote"
    assert report["evidence_promotion"]["report"]["results"][0]["decision"] == "terminated_low_signal"
    assert report["findings"] == result.final_result["payload"]
    assert all(item["candidate_id"] == "C002" for item in report["evidence_promotion"]["stage2"]["executions"])
    assert report["reproducibility"]["promotion_policy_hash"] == result.workflow_state["validation_stage_plan"]["policy_hash"]
    markdown = (root / "audit_report.md").read_text(encoding="utf-8")
    assert "## Candidate Screening" in markdown and "## Evidence Promotion" in markdown
    assert "not experimental contradiction" in markdown
    progress = EventListener(request.runs_root, request.run_id).poll()
    assert progress.candidates["C001"].decision == "promote" and progress.candidates["C001"].promotion == "terminated_low_signal"
    assert all(progress.stages[stage] == "completed" for stage in ("stage1_validation", "evidence_promotion", "stage2_validation"))
    lines = "\n".join(projection(progress).values())
    assert "Evidence-promoted=1" in lines and "Early terminated before expensive validation=1" in lines
    assert "saved dollars" not in lines


def test_unconfigured_dataset_only_audit_defers_expensive_validation(tmp_path):
    request, runtimes, executor = staged_request(tmp_path, configured=False)
    result = run(request, agent_client=client(request.runs_root, Transport(TwoCandidates())), tool_runtimes=runtimes)
    assert all(item["decision"] == "deferred_insufficient_evidence" for item in result.workflow_state["promotion_result"]["results"])
    assert all(kind == "statistical_check" for _, kind in executor.calls)
    assert result.workflow_state["stage2_validation"]["status"] == "skipped"


def test_direct_stage2_cannot_execute_terminated_candidate(completed_staged):
    request, runtimes, executor, _, result = completed_staged
    from shortcutminer.workflow.nodes import quantitative_node
    before = list(executor.calls)
    transport = Transport(TwoCandidates())
    with pytest.raises(ValueError, match="not evidence-promoted"):
        quantitative_node(result.workflow_state, client=client(request.runs_root, transport),
            tool_runtime=runtimes["quantitative"], validation_phase="stage2", selected_task_ids={"TASK-C001-QT"})
    assert executor.calls == before and transport.calls == []


def test_missing_stage1_provenance_fails_closed_without_duplicate_work(completed_staged):
    request, runtimes, executor, _, result = completed_staged
    from shortcutminer.workflow.nodes import quantitative_node
    root = Path(request.runs_root) / request.run_id
    measurement = result.workflow_state["stage1_quantitative_result"]["results"][1]["payload"]["measurements"][0]
    (root / "tools" / "quantitative" / f"{measurement['tool_call_id']}.json").unlink()
    before, transport = list(executor.calls), Transport(TwoCandidates())
    with pytest.raises(Exception, match="Stage-1 tool provenance unavailable"):
        quantitative_node(result.workflow_state, client=client(request.runs_root, transport),
            tool_runtime=runtimes["quantitative"], validation_phase="stage2", selected_task_ids={"TASK-C002-QT"})
    assert executor.calls == before and transport.calls == []


def test_private_truth_is_never_read_or_hashed(completed_staged, monkeypatch):
    request, _, _, _, result = completed_staged
    from shortcutminer.experiment.evidence_promotion import stage1_projection, stable_hash, load_or_evaluate_promotion
    from shortcutminer.protocol.promotion import PromotionPolicy
    root = Path(request.runs_root) / request.run_id
    (root / "hidden_truth.json").write_text("PRIVATE_SENTINEL")
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        assert path.name != "hidden_truth.json"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    qt = result.workflow_state["stage1_quantitative_result"]
    ql = result.workflow_state["stage1_qualitative_result"]
    projection = stage1_projection(["C001", "C002"], qt, ql, run_id=request.run_id)
    changed = json.loads(json.dumps(qt))
    changed["results"][0]["payload"]["measurements"][0]["metrics"]["hidden_truth"] = 999
    assert stable_hash(stage1_projection(["C001", "C002"], changed, ql)) == stable_hash(projection)
    frozen = load_or_evaluate_promotion(request.runs_root, request.run_id, PromotionPolicy.model_validate(request.metadata["promotion_policy"]), projection)
    assert frozen.model_dump(mode="json") == result.workflow_state["promotion_result"]


def test_actual_insufficient_coverage_defers_all_stage2(tmp_path):
    request, runtimes, executor = staged_request(tmp_path)
    policy = fixture_policy().model_copy(update={"minimum_samples": 1000})
    request = request.model_copy(update={"metadata": {**request.metadata, "promotion_policy": policy.model_dump(mode="json")}})
    result = run(request, agent_client=client(request.runs_root, Transport(TwoCandidates())), tool_runtimes=runtimes)
    assert all(item["decision"] == "deferred_insufficient_evidence" for item in result.workflow_state["promotion_result"]["results"])
    assert len(executor.calls) == 2 and all(kind == "statistical_check" for _, kind in executor.calls)
