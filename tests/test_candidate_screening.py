"""Screening controller regression tests with no provider or experiment calls."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import shortcutminer.experiment.candidate_screening as screening_module
import shortcutminer.workflow.graph as graph_module
from shortcutminer.experiment.candidate_screening import (
    build_screening_report, load_or_screen_candidates, screen_candidates,
)
from shortcutminer.experiment.dataset_profiler import build_dataset_profile
from shortcutminer.protocol.identify import CandidateReport, IdentifyRequest
from shortcutminer.protocol.screening import (
    CandidateScreeningPolicy, CandidateScreeningResult, CandidateScreeningSignals,
)
from shortcutminer.runner import run, resume_run
from shortcutminer.runtime.demo import build_demo_request
from shortcutminer.runtime.mock import MockAgentClient


def inputs(tmp_path):
    base = build_demo_request(tmp_path / "runs", "screening-test")
    dataset = Path(base.dataset_info.root)
    (dataset / "metadata.csv").write_text(
        "id,label,site,device,batch\n1,0,A,X,P\n2,0,A,X,P\n3,1,B,Y,Q\n4,1,B,Y,Q\n", encoding="utf-8")
    request = IdentifyRequest(protocol_version="1.0", run_id=base.run_id, message_id="identify:request",
                             source="orchestrator", target="identify", message_type="identify_request",
                             dataset=base.dataset_info, task=base.task_input)
    profile = build_dataset_profile(request, seed=42)
    candidates = [{"candidate_id": f"C{index:03d}", "cue": {"family": "metadata", "name": field},
                   "metadata_field_ref": field, "description": f"Potential {field} cue.",
                   "observation": f"Public metadata contains {field}.",
                   "hypothesis": f"{field} may differ across labels.", "evidence_sources": ["metadata"],
                   "priority": "medium", "status": "unverified"}
                  for index, field in enumerate(["site", "device", "batch"], 1)]
    report = CandidateReport(protocol_version="1.0", run_id=base.run_id, message_id="identify:report",
                             source="identify", target="executive", message_type="candidate_report",
                             in_reply_to=request.message_id, artifact_references=[],
                             dataset_summary={"dataset": base.dataset_info.name, "samples_inspected": 0,
                                              "metadata_used": True}, candidates=candidates,
                             limitations=[])
    return base, report, profile


def test_multiple_candidates_low_confidence_and_no_premature_verdict(tmp_path):
    _, report, profile = inputs(tmp_path)
    signals = CandidateScreeningSignals(confidence_by_candidate={"C001": 0.82, "C002": 0.2})
    results = screen_candidates(report, profile, signals=signals)
    assert [item.decision for item in results] == ["promote", "screened_out", "promote"]
    assert results[1].reason == ["weak_signal"]
    assert results[2].confidence is None  # No priority-to-confidence fabrication.
    assert all(candidate.status == "unverified" for candidate in report.candidates)
    for verdict in ["confirmed", "contradicted", "rejected"]:
        with pytest.raises(ValidationError):
            CandidateScreeningResult.model_validate({**results[0].model_dump(), "decision": verdict})


def test_cost_and_count_caps_choose_deterministically(tmp_path):
    _, report, profile = inputs(tmp_path)
    signals = CandidateScreeningSignals(confidence_by_candidate={"C001": 0.6, "C002": 0.9, "C003": 0.8},
                                        cost_by_candidate={"C001": 1.0, "C002": 2.0, "C003": 100.0})
    results = screen_candidates(report, profile, signals=signals,
                                policy=CandidateScreeningPolicy(max_promoted=1, max_candidate_cost=10.0))
    assert [item.decision for item in results] == ["screened_out", "promote", "screened_out"]
    assert results[0].reason == results[2].reason == ["resource_budget"]
    results = screen_candidates(report, profile, signals=signals,
                                policy=CandidateScreeningPolicy(max_total_cost=2.0))
    assert [item.candidate_id for item in results if item.decision == "promote"] == ["C002"]


def test_resource_unavailability_and_empty_report(tmp_path):
    _, report, profile = inputs(tmp_path)
    profile = profile.model_copy(update={"metadata": profile.metadata.model_copy(update={
        "available": False, "rows_profiled": 0, "column_names": [], "columns": [], "row_count": None})})
    results = screen_candidates(report, profile)
    assert all(item.decision == "screened_out" for item in results)
    assert all("insufficient_metadata" in item.reason and "unsupported_validation" in item.reason for item in results)
    empty = build_screening_report(report.model_copy(update={"candidates": []}), profile)
    assert empty.candidates_generated == empty.promoted == empty.screened_out == 0


def test_insufficient_samples(tmp_path):
    _, report, profile = inputs(tmp_path)
    results = screen_candidates(report, profile, policy=CandidateScreeningPolicy(min_samples=5))
    assert all(item.decision == "screened_out" and "insufficient_samples" in item.reason for item in results)


def test_no_hidden_truth_future_results_models_or_dataset_reads(tmp_path, monkeypatch):
    _, report, profile = inputs(tmp_path)
    private = tmp_path / "private" / "truth.json"
    private.parent.mkdir()
    private.write_text('{"expected_shortcut":"device"}', encoding="utf-8")
    def forbid(*args, **kwargs):
        pytest.fail("Pure screening must not read files or invoke providers/tools")
    monkeypatch.setattr(Path, "open", forbid)
    monkeypatch.setattr(Path, "read_text", forbid)
    monkeypatch.setattr("shortcutminer.runtime.openai_compatible.OpenAICompatibleClient.invoke_agent", forbid)
    monkeypatch.setattr("shortcutminer.tooling.runtime.ToolRuntime.execute", forbid)
    result = build_screening_report(report, profile)
    assert result.promoted == 3
    with pytest.raises(ValidationError):
        CandidateScreeningSignals.model_validate({"quantitative_results": {}, "hidden_truth": "device"})


def test_reuse_persisted_decisions_and_deterministic_hash(tmp_path, monkeypatch):
    base, report, profile = inputs(tmp_path)
    left = CandidateScreeningSignals(confidence_by_candidate={"C001": 0.8, "C002": 0.2})
    right = CandidateScreeningSignals(confidence_by_candidate={"C002": 0.2, "C001": 0.8})
    first = load_or_screen_candidates(report, profile, signals=left, runs_root=base.runs_root)
    assert first == build_screening_report(report, profile, signals=right)
    def forbid(*args, **kwargs):
        pytest.fail("A matching screening artifact must be reused")
    monkeypatch.setattr(screening_module, "screen_candidates", forbid)
    second = load_or_screen_candidates(report, profile, signals=right, runs_root=base.runs_root)
    assert first == second
    with pytest.raises(ValueError, match="inputs changed"):
        load_or_screen_candidates(report, profile, signals=left,
                                  policy=CandidateScreeningPolicy(min_confidence=0.9), runs_root=base.runs_root)


def test_foreign_signals_and_tampered_cache_are_rejected(tmp_path):
    base, report, profile = inputs(tmp_path)
    with pytest.raises(ValueError, match="unknown candidate"):
        screen_candidates(report, profile, signals=CandidateScreeningSignals(confidence_by_candidate={"C004": 0.1}))
    load_or_screen_candidates(report, profile, runs_root=base.runs_root)
    path = Path(base.runs_root) / base.run_id / "screening" / "result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["results"][0]["reason"] = ["tampered"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="result hash"):
        load_or_screen_candidates(report, profile, runs_root=base.runs_root)


def provider(report):
    def identify(_payload, _prompt):
        return {"candidates": [{key: value for key, value in candidate.model_dump(mode="json").items()
                                if key not in {"candidate_id", "status"}}
                               for candidate in report.candidates], "additional_limitations": []}
    def executive(payload, _prompt):
        return {"decisions": [{"candidate_id": item["candidate_id"], "route": "unresolved",
                "routing_reason": "No configured validation capability for this test.",
                "quantitative": None, "qualitative": None, "missing_requirements": ["test capability missing"],
                "limitations": []} for item in payload["request"]["candidate_report"]["candidates"]],
                "limitations": []}
    def summarize(payload, _prompt):
        return {"overview": "Unvalidated hypotheses retained with their screening history.",
                "findings": [{"candidate_ref": item["candidate_id"], "assessment": "insufficient_evidence",
                              "summary": "No scientific verdict is available.", "additional_limitations": [],
                              "recommendations": []} for item in payload["request"]["identify_result"]["candidates"]],
                "additional_limitations": []}
    return MockAgentClient({"identify": identify, "executive": executive, "summarize": summarize})


def request_for_gate(base, signals):
    return base.model_copy(update={"identify_config": {
        **base.identify_config, "budget": {"max_samples": 4, "max_candidates": 3},
        "candidate_screening": {"signals": signals.model_dump(mode="json")}},
        "executive_config": {}, "qualitative_config": {}})


def test_screened_out_candidates_never_enter_executive_and_remain_in_report(tmp_path):
    base, report, _ = inputs(tmp_path)
    signals = CandidateScreeningSignals(confidence_by_candidate={"C001": 0.1, "C002": 0.82, "C003": 0.2})
    client = provider(report)
    result = run(request_for_gate(base, signals), agent_client=client)
    executive = next(item["payload"] for item in client.calls if item["agent_name"] == "executive")
    assert [item["candidate_id"] for item in executive["request"]["candidate_report"]["candidates"]] == ["C002"]
    assert [item["candidate_id"] for item in executive["executable_candidate_contexts"]] == ["C002"]
    assert result.workflow_state["executive_result"]["tasks"] == []
    assert [item["candidate_id"] for item in result.workflow_state["identify_result"]["candidates"]] == ["C001", "C002", "C003"]
    section = result.final_result["payload"]["candidate_screening"]
    assert section["candidates_generated"] == 3 and section["promoted"] == 1 and section["screened_out"] == 2
    assert section["results"][0]["reason"] == ["weak_signal"]
    assert all(finding["assessment"] == "insufficient_evidence" for finding in result.final_result["payload"]["findings"])
    saved = json.loads((Path(base.runs_root) / base.run_id / "final_result.json").read_text(encoding="utf-8"))
    assert saved == result.final_result
    from shortcutminer.protocol.summarize import SummaryReport, SummarizeRequest
    from shortcutminer.experiment.summarize_executor import SummarizeExecutionError, validate_summary_report
    summary = SummaryReport.model_validate(result.final_result)
    changed = summary.model_copy(update={"payload": summary.payload.model_copy(update={"candidate_screening": None})})
    with pytest.raises(SummarizeExecutionError, match="screening section differs"):
        validate_summary_report(changed, SummarizeRequest.model_validate(result.workflow_state["summarize_request"]))


def test_all_screened_out_skips_executive_and_validation_providers(tmp_path):
    base, report, _ = inputs(tmp_path)
    signals = CandidateScreeningSignals(confidence_by_candidate={item.candidate_id: 0.1 for item in report.candidates})
    client = provider(report)
    result = run(request_for_gate(base, signals), agent_client=client)
    assert [item["agent_name"] for item in client.calls] == ["identify", "summarize"]
    assert result.workflow_state["executive_result"]["decisions"] == []
    assert result.final_result["payload"]["candidate_screening"]["screened_out"] == 3
    assert all(not item["record_ids"] for item in result.workflow_state["evidence_ledger"]["candidates"])


def test_only_promoted_original_id_executes_real_backend_evidence(tmp_path):
    from test_phase16_three_candidate_chain import FakeTransport
    from shortcutminer.protocol.model import OpenAICompatibleModelConfig
    from shortcutminer.protocol.run import RunRequest
    from shortcutminer.runner import build_local_tool_runtimes
    from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient
    base, _, _ = inputs(tmp_path)
    data = base.model_dump(mode="json")
    data["identify_config"]["budget"]["max_candidates"] = 3
    data["identify_config"]["candidate_screening"] = {"signals": {
        "confidence_by_candidate": {"C001": 0.1, "C002": 0.82, "C003": 0.2}}}
    data["executive_config"]["capabilities"]["available_metadata_fields"] = ["id", "label", "site", "device", "batch"]
    request = RunRequest.model_validate(data)
    transport = FakeTransport()
    client = OpenAICompatibleClient(OpenAICompatibleModelConfig(
        model="offline", base_url="https://offline.invalid/v1", api_key_env="OFFLINE_KEY"),
        api_key="offline", transport=transport, runs_root=base.runs_root)
    runtimes = build_local_tool_runtimes(run_id=base.run_id, dataset_root=base.dataset_info.root,
                                         runs_root=base.runs_root)
    result = run(request, agent_client=client, tool_runtimes=runtimes)
    state = result.workflow_state
    assert [task["candidate_id"] for task in state["executive_result"]["tasks"]] == ["C002", "C002"]
    assert [item["payload"]["candidate_id"] for item in state["quantitative_result"]["results"]] == ["C002"]
    assert [item["payload"]["candidate_id"] for item in state["qualitative_result"]["results"]] == ["C002"]
    entries = state["evidence_ledger"]["candidates"]
    assert [item["candidate_id"] for item in entries] == ["C001", "C002", "C003"]
    assert entries[0]["records"] == entries[2]["records"] == []
    assert entries[1]["records"]
    assert all(record["provenance"]["candidate_id"] == "C002" for record in entries[1]["records"])
    measurement = state["quantitative_result"]["results"][0]["payload"]["measurements"][0]
    assert measurement["tool_call_id"] and measurement["plan_step_id"]
    before = len(transport.calls)
    completed = resume_run(base.run_id, runs_root=base.runs_root, agent_client=client, tool_runtimes=runtimes)
    assert completed.final_result == result.final_result
    assert len(transport.calls) == before


def test_screening_node_ignores_private_metadata_and_future_state(tmp_path):
    from shortcutminer.workflow.nodes import candidate_screening_node
    base, report, profile = inputs(tmp_path)
    class Unreadable:
        def __iter__(self):
            pytest.fail("Future/private state must not be traversed")
        def __repr__(self):
            pytest.fail("Future/private state must not be serialized")
    state = {"identify_config": {}, "identify_result": report.model_dump(mode="json"),
             "dataset_profile": profile.model_dump(mode="json"),
             "metadata": {"hidden_truth": Unreadable()}, "quantitative_result": Unreadable(),
             "qualitative_result": Unreadable()}
    result = candidate_screening_node(state)
    assert result["screening_result"]["promoted"] == 3


def test_legacy_executive_boundary_runs_same_gate(tmp_path, monkeypatch):
    from shortcutminer.workflow.nodes import executive_node
    base, report, _ = inputs(tmp_path)
    request = request_for_gate(base, CandidateScreeningSignals(confidence_by_candidate={"C001": 0.1}))
    completed = run(request, agent_client=provider(report))
    state = dict(completed.workflow_state)
    state.pop("screening_result")
    def forbid(*args, **kwargs):
        pytest.fail("The same artifact must be reused at a legacy Executive boundary")
    monkeypatch.setattr(screening_module, "screen_candidates", forbid)
    client = provider(report)
    update = executive_node(state, client=client)
    assert update["screening_result"] == completed.workflow_state["screening_result"]
    assert [item.candidate_id for item in CandidateReport.model_validate(
        update["executive_request"]["candidate_report"]).candidates] == ["C002", "C003"]


@pytest.mark.parametrize("boundary", ["screening", "executive"])
def test_checkpoint_resume_reuses_screening_results(tmp_path, monkeypatch, boundary):
    base, report, _ = inputs(tmp_path)
    request = request_for_gate(base, CandidateScreeningSignals(confidence_by_candidate={"C001": 0.1}))
    client = provider(report)
    original = getattr(graph_module, f"{'candidate_screening' if boundary == 'screening' else boundary}_node")
    def interrupt(state, **kwargs):
        if boundary == "screening":
            original(state, **kwargs)  # Simulate persistence before a missing checkpoint.
        raise RuntimeError("planned screening boundary interruption")
    name = "candidate_screening_node" if boundary == "screening" else "executive_node"
    monkeypatch.setattr(graph_module, name, interrupt)
    with pytest.raises(RuntimeError, match="planned screening boundary"):
        run(request, agent_client=client)
    path = Path(base.runs_root) / base.run_id / "screening" / "result.json"
    prior = path.read_bytes()
    monkeypatch.setattr(graph_module, name, original)
    def forbid(*args, **kwargs):
        pytest.fail("Resume must not re-screen or invoke Identify")
    monkeypatch.setattr(screening_module, "screen_candidates", forbid)
    result = resume_run(base.run_id, runs_root=base.runs_root, agent_client=client)
    assert result.status == "COMPLETED"
    assert path.read_bytes() == prior
    assert sum(item["agent_name"] == "identify" for item in client.calls) == 1
