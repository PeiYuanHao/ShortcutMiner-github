"""Scientific calibration fixtures for the threshold-free Evidence Ledger."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from shortcutminer.evaluation.calibration import benjamini_hochberg, bootstrap_interval
from shortcutminer.evaluation.evidence_ledger import EvidenceLedgerBuilder
from shortcutminer.experiment.summarize_executor import SummarizeExecutionError, validate_summary_report
from shortcutminer.protocol.batches import QualitativeResultBatch, QuantitativeResultBatch
from shortcutminer.protocol.executive import RoutingPlan
from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.protocol.summarize import SummarizeRequest, SummaryReport
from shortcutminer.runner import run
from shortcutminer.runtime.demo import DemoAgentClient, build_demo_request


def _completed_run(tmp_path: Path):
    request = build_demo_request(tmp_path / "runs", "phase11-calibration")
    return run(request, agent_client=DemoAgentClient())


def _inputs(result):
    state = result.workflow_state
    return (
        CandidateReport.model_validate(state["identify_result"]),
        RoutingPlan.model_validate(state["executive_result"]),
        QuantitativeResultBatch.model_validate(state["quantitative_result"]),
        QualitativeResultBatch.model_validate(state["qualitative_result"]),
    )


def _replace_quantitative(batch, **payload_updates):
    data = batch.model_dump(mode="json")
    if payload_updates.get("status") not in {"failed", "blocked"} and (
        "metrics" in payload_updates or "evidence_direction" in payload_updates
    ):
        payload_updates.setdefault("status", "completed")
        payload_updates.setdefault("evidence_level", "Q2")
        if payload_updates.get("evidence_direction") is None:
            payload_updates["evidence_direction"] = "inconclusive"
    data["results"][0]["payload"].update(payload_updates)
    payload = data["results"][0]["payload"]
    if payload["status"] == "completed":
        payload["measurements"] = [{
            "measurement_id": "M-calibration-fixture",
            "run_id": data["results"][0]["run_id"],
            "candidate_id": payload["candidate_id"],
            "task_id": payload["task_id"],
            "plan_step_id": "STEP-calibration-fixture",
            "planning_check": "association_test",
            "experiment_type": "statistical_check",
            "tool_call_id": "TC-calibration-fixture",
            "backend_identity": "fixture",
            "execution_outcome": "completed",
            "metrics": payload["metrics"],
            "artifact_references": payload["artifact_references"],
            "model_lineage": {},
            "design_strength": "observational",
            "uncertainty_metadata": [],
        }]
    else:
        payload["measurements"] = []
    return QuantitativeResultBatch.model_validate(data)


def _replace_qualitative(batch, **payload_updates):
    data = batch.model_dump(mode="json")
    data["results"][0]["payload"].update(payload_updates)
    return QualitativeResultBatch.model_validate(data)


def test_ledger_is_persisted_traceable_and_candidate_complete(tmp_path: Path):
    result = _completed_run(tmp_path)
    ledger = result.workflow_state["evidence_ledger"]
    path = tmp_path / "runs" / result.run_id / "evidence" / "ledger.json"

    assert path.is_file()
    assert json.loads(path.read_text("utf-8")) == ledger
    assert [item["candidate_id"] for item in ledger["candidates"]] == ["C001"]
    entry = ledger["candidates"][0]
    assert entry["profile"]["consistency"] == "consistent_support"
    assert entry["profile"]["shared_source_groups"]
    assert all(record["provenance"]["result_id"] for record in entry["records"])
    assert all("uncertainty" in record for record in entry["records"])


def test_ledger_fingerprint_reuses_matching_and_rebuilds_changed_inputs(tmp_path: Path):
    result = _completed_run(tmp_path)
    identify, plan, quant, qual = _inputs(result)
    runs_root = tmp_path / "runs"
    path = runs_root / result.run_id / "evidence" / "ledger.json"
    builder = EvidenceLedgerBuilder()
    first = builder.build(identify, plan, quant, qual, runs_root=runs_root)
    first_mtime = path.stat().st_mtime_ns
    reused = builder.build(identify, plan, quant, qual, runs_root=runs_root)
    assert reused.fingerprint == first.fingerprint
    assert path.stat().st_mtime_ns == first_mtime

    changed_quant = _replace_quantitative(
        quant,
        metrics={"association_score": 0.5, "p_value": 0.2},
    )
    rebuilt = builder.build(
        identify, plan, changed_quant, qual, runs_root=runs_root
    )
    assert rebuilt.fingerprint != first.fingerprint
    assert json.loads(path.read_text("utf-8"))["fingerprint"] == rebuilt.fingerprint


def test_ledger_retains_observation_comparison_and_counterevidence(tmp_path: Path):
    result = _completed_run(tmp_path)
    identify, plan, quant, qual = _inputs(result)
    qualitative = _replace_qualitative(
        qual,
        evidence_status="partially_supported",
        sampling={
            "strategy": "group_balanced",
            "seed": 42,
            "groups": {"label=A": 1, "label=B": 1},
            "sample_count": 2,
            "sample_ids": ["S1", "S2"],
        },
        observations=[{
            "observation_id": "QO-C001-001",
            "group": "label=A",
            "sample_refs": ["S1"],
            "observation": "A direct within-group property was observed.",
            "inspection_quality": "sufficient",
        }],
        comparisons=[{
            "comparison_id": "QC-C001-001",
            "groups": ["label=A", "label=B"],
            "sample_refs": ["S1", "S2"],
            "comparison": "The inspected groups differ in the measured property.",
            "inspection_quality": "limited",
        }],
        interpretations=[],
        counterevidence=[{
            "sample_refs": ["S2"],
            "description": "A counterexample limits the comparison.",
        }],
        metadata_evidence=[],
    )

    ledger = EvidenceLedgerBuilder().build(identify, plan, quant, qualitative)
    records = [
        item for item in ledger.candidates[0].records
        if item.source_branch == "qualitative"
    ]

    assert [item.evidence_kind for item in records] == [
        "qualitative_visual",
        "qualitative_group_comparison",
        "qualitative_counterevidence",
        "qualitative_assessment",
    ]
    comparison = records[1]
    assert comparison.metrics == {
        "groups": ["label=A", "label=B"],
        "comparison": "The inspected groups differ in the measured property.",
        "sample_refs": ["S1", "S2"],
        "inspection_quality": "limited",
    }
    assert comparison.execution_state == "usable"
    assert comparison.direction == "inconclusive"
    assert records[-1].direction == "mixed"


def test_unusable_comparison_is_scientifically_inconclusive(tmp_path: Path):
    result = _completed_run(tmp_path)
    identify, plan, quant, qual = _inputs(result)
    qualitative = _replace_qualitative(
        qual,
        comparisons=[{
            "comparison_id": "QC-C001-001",
            "groups": ["label=A", "label=B"],
            "sample_refs": ["S1", "S2"],
            "comparison": "Inspection quality prevents a usable comparison.",
            "inspection_quality": "unusable",
        }],
    )

    ledger = EvidenceLedgerBuilder().build(identify, plan, quant, qualitative)
    comparison = next(
        item for item in ledger.candidates[0].records
        if item.evidence_kind == "qualitative_group_comparison"
    )
    assert comparison.execution_state == "scientifically_inconclusive"
    assert comparison.direction == "inconclusive"


@pytest.mark.parametrize(
    "quant_updates,qual_updates,expected",
    [
        ({"evidence_direction": "supports"}, {"evidence_status": "supported"}, "consistent_support"),
        ({"evidence_direction": "supports"}, {"evidence_status": "contradicted"}, "mixed"),
        ({"evidence_direction": "contradicts"}, {"evidence_status": "supported"}, "mixed"),
        ({"evidence_direction": "inconclusive"}, {"evidence_status": "inconclusive"}, "inconclusive"),
        (
            {"status": "failed", "evidence_level": "Q0", "evidence_direction": "not_tested", "metrics": {}, "limitations": ["execution failed"]},
            {"evidence_status": "insufficient_evidence", "metadata_evidence": [], "observations": [], "counterevidence": [], "limitations": ["inspection blocked"]},
            "no_usable_evidence",
        ),
    ],
)
def test_calibration_combination_fixtures(tmp_path: Path, quant_updates, qual_updates, expected):
    result = _completed_run(tmp_path)
    identify, plan, quant, qual = _inputs(result)
    ledger = EvidenceLedgerBuilder().build(
        identify,
        plan,
        _replace_quantitative(quant, **quant_updates),
        _replace_qualitative(qual, **qual_updates),
    )
    profile = ledger.candidates[0].profile
    assert profile.consistency == expected
    if quant_updates.get("status") == "failed":
        record = next(item for item in ledger.candidates[0].records if item.source_branch == "quantitative")
        assert record.execution_state == "failed"
        assert record.direction == "not_tested"


def test_bh_fdr_is_monotone_bounded_deterministic_and_preserves_raw_values():
    raw = [0.04, 0.001, 0.03, 0.2, 1.0]
    first = benjamini_hochberg(raw)
    second = benjamini_hochberg(raw)
    assert first == second
    assert raw == [0.04, 0.001, 0.03, 0.2, 1.0]
    assert all(0 <= value <= 1 for value in first)
    ordered = sorted(zip(raw, first))
    assert [item[1] for item in ordered] == sorted(item[1] for item in ordered)


def test_ledger_records_bh_family_without_overwriting_raw_p_values(tmp_path: Path):
    result = _completed_run(tmp_path)
    identify, plan, quant, qual = _inputs(result)
    quant = _replace_quantitative(
        quant,
        metrics={"effect_size": 0.4, "p_value": 0.04, "spearman_p_value": 0.01},
    )
    ledger = EvidenceLedgerBuilder().build(identify, plan, quant, qual)
    tests = ledger.candidates[0].records[0].statistical_tests
    assert [item.raw_p_value for item in tests] == [0.04, 0.01]
    assert all(item.adjusted_p_value is not None for item in tests)
    assert {item.family_id for item in tests} == {"C001:related_statistical_tests"}
    assert {item.family_size for item in tests} == {2}


def test_bootstrap_is_seeded_bounded_and_invalid_data_is_unavailable():
    values = np.arange(40, dtype=float)
    kwargs = dict(
        metric="mean",
        sample_count=len(values),
        statistic=lambda indices: float(values[indices].mean()),
        confidence_level=0.95,
        iterations=80,
        seed=17,
        resampling_unit="sample_row",
    )
    first = bootstrap_interval(**kwargs)
    assert first == bootstrap_interval(**kwargs)
    assert first.available and first.lower <= values.mean() <= first.upper
    invalid = bootstrap_interval(
        metric="constant_invalid",
        sample_count=5,
        statistic=lambda _indices: None,
        confidence_level=0.95,
        iterations=20,
        seed=1,
        resampling_unit="sample_row",
    )
    assert not invalid.available
    assert invalid.reason_unavailable


def test_summary_overclaim_guard_rejects_all_failed_support(tmp_path: Path):
    result = _completed_run(tmp_path)
    identify, plan, quant, qual = _inputs(result)
    quant = _replace_quantitative(
        quant,
        status="failed",
        evidence_level="Q0",
        evidence_direction="not_tested",
        metrics={},
        limitations=["execution failed"],
    )
    qual = _replace_qualitative(
        qual,
        evidence_status="insufficient_evidence",
        metadata_evidence=[],
        observations=[],
        counterevidence=[],
        limitations=["inspection unavailable"],
    )
    ledger = EvidenceLedgerBuilder().build(identify, plan, quant, qual)
    request = SummarizeRequest(
        protocol_version="1.0",
        run_id=result.run_id,
        message_id=f"{result.run_id}:guard:request",
        source="orchestrator",
        target="summarize",
        message_type="summarize_request",
        in_reply_to=plan.message_id,
        identify_result=identify,
        executive_result=plan,
        quantitative_result=quant,
        qualitative_result=qual,
        evidence_ledger=ledger,
    )
    report_data = result.final_result.copy()
    report_data["in_reply_to"] = request.message_id
    report_data["payload"]["findings"][0]["assessment"] = "supported"
    report_data["payload"]["artifact_references"] = []
    report_data["payload"]["evidence_ledger_reference"] = None
    report = SummaryReport.model_validate(report_data)
    with pytest.raises(SummarizeExecutionError, match="no supporting evidence"):
        validate_summary_report(report, request)


def test_production_has_no_metric_threshold_to_shortcut_verdict_mapping():
    roots = [Path("shortcutminer/experiment"), Path("shortcutminer/evaluation"), Path("shortcutminer/workflow"), Path("shortcutminer/protocol")]
    suspicious_metrics = {"p_value", "effect_size", "cramers_v", "accuracy_gap", "score"}
    verdict_words = {"supported", "confirmed", "shortcut", "risk"}
    violations = []
    for root in roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text("utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.If):
                    continue
                condition = ast.unparse(node.test).casefold()
                body = " ".join(ast.unparse(item).casefold() for item in node.body)
                if any(metric in condition for metric in suspicious_metrics) and any(word in body for word in verdict_words):
                    violations.append(f"{path}:{node.lineno}")
    assert not violations
