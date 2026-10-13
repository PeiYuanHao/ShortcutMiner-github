"""Canonical exports use finalized authority, never private evaluation inputs."""
import json
from pathlib import Path

import pytest

from shortcutminer.experiment.audit_report import build_audit_report, export_audit_report, AuditReportError
from shortcutminer.protocol.audit_report import AuditReport
from shortcutminer.runner import run
from shortcutminer.audit import build_audit_run_request
from shortcutminer.protocol.audit import AuditRequest
from test_audit_progress import request_and_runtimes, Transport, client
from test_audit_entry import audit_client


@pytest.fixture
def completed_audit(tmp_path):
    request, runtimes = request_and_runtimes(tmp_path)
    transport = Transport()
    result = run(request, agent_client=client(request.runs_root, transport), tool_runtimes=runtimes)
    return request, result, Path(request.runs_root) / request.run_id, transport


def test_completed_audit_exports_both_formats(completed_audit):
    request, _, root, _ = completed_audit
    report = AuditReport.model_validate_json((root / "audit_report.json").read_text(encoding="utf-8"))
    assert report == build_audit_report(request.runs_root, request.run_id)
    assert (root / "audit_report.md").read_text(encoding="utf-8").startswith("# ShortcutMiner Audit Report")
    assert report.run["status"] == "COMPLETED"
    assert report.dataset["sample_count"] == 4
    assert report.dataset["label_source"]["status"] == "resolved"
    assert report.dataset["dataset_type"] == request.metadata["audit_request"]["dataset_info"]["dataset_type"]
    assert report.dataset["description"] == request.metadata["audit_request"]["dataset_description"]
    assert report.dataset["public_metadata_fields"] == ["id", "label", "site"]


def test_all_hypotheses_and_original_confidence_preserved(completed_audit):
    _, result, root, _ = completed_audit
    report = json.loads((root / "audit_report.json").read_text())
    actual = report["discovery"]["candidates"]
    assert len(actual) == len(result.workflow_state["identify_result"]["candidates"]) == 2
    assert all(candidate["status"] == "unverified" and candidate["identify_confidence"] is None for candidate in actual)
    assert report["screening"]["results"][1]["confidence"] == 0.1


def test_screened_out_is_separate_from_scientific_findings(completed_audit):
    _, result, root, _ = completed_audit
    report = json.loads((root / "audit_report.json").read_text())
    assert report["screening"]["promoted"] == report["screening"]["screened_out"] == 1
    assert report["screening"]["results"][1]["decision"] == "screened_out"
    assert "weak_signal" in report["screening"]["results"][1]["reason"]
    assert report["findings"] == result.final_result["payload"]
    finding = next(f for f in report["findings"]["findings"] if f["candidate_id"] == "C002")
    assert finding["assessment"] == "insufficient_evidence"
    assert all(task["candidate_id"] != "C002" for task in report["validation"]["routing"]["tasks"])


def test_candidate_task_measurement_evidence_provenance(completed_audit):
    _, result, root, _ = completed_audit
    report = json.loads((root / "audit_report.json").read_text())
    assert report["evidence"] == result.workflow_state["evidence_ledger"]
    assert report["quantitative"] == result.workflow_state["quantitative_result"]["results"]
    assert report["qualitative"] == result.workflow_state["qualitative_result"]["results"]
    evidence = next(record for entry in report["evidence"]["candidates"] for record in entry["records"]
                    if record["source_branch"] == "quantitative")
    measurement = report["quantitative"][0]["payload"]["measurements"][0]
    assert evidence["candidate_id"] == measurement["candidate_id"] == "C001"
    assert evidence["task_id"] == measurement["task_id"]
    assert evidence["provenance"]["measurement_id"] == measurement["measurement_id"]
    assert evidence["provenance"]["tool_call_id"] == measurement["tool_call_id"]
    assert evidence["provenance"]["plan_step_id"] == measurement["plan_step_id"]
    assert measurement["backend_identity"] == "real_cpu_v1"


def test_limitations_deduplicate_with_source_attribution(completed_audit):
    _, _, root, _ = completed_audit
    limitations = json.loads((root / "audit_report.json").read_text())["limitations"]
    assert len({item["text"] for item in limitations}) == len(limitations)
    assert all(item["sources"] for item in limitations)
    assert any(len(item["sources"]) > 1 for item in limitations)


def test_artifact_index_is_portable_and_does_not_read_private_files(completed_audit, monkeypatch):
    request, _, root, _ = completed_audit
    (root / "hidden_truth.json").write_text('{"secret":"PRIVATE_SENTINEL"}')
    (root / "model_calls" / "raw_provider.json").write_text('INVALID_RAW_PROVIDER_BODY')
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        assert path.name not in {"hidden_truth.json", "raw_provider.json"}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    report = export_audit_report(request.runs_root, request.run_id)
    assert "PRIVATE_SENTINEL" not in report.model_dump_json()
    assert all(not Path(ref).is_absolute() and ".." not in Path(ref).parts and "\\" not in ref
               for ref in report.artifacts["references"])


def test_secrets_and_unsafe_arbitrary_metrics_are_withheld(completed_audit):
    request, _, root, _ = completed_audit
    path = root / "workflow_state.json"
    state = json.loads(path.read_text())
    state["quantitative_result"]["results"][0]["payload"]["metrics"].update({
        "api_key": "SENTINEL_API", "secret": "SENTINEL_ENV", "hidden_truth": "SENTINEL_TRUTH",
        "raw_provider_body": "SENTINEL_RAW", "reasoning_content": "SENTINEL_REASON",
        "image_blob": "data:image/png;base64,SENTINEL_IMAGE", "unsafe_text": "Authorization: Bearer SENTINEL_AUTH"})
    state["metadata"]["hidden_truth"] = "SENTINEL_STATE"
    path.write_text(json.dumps(state), encoding="utf-8")
    export_audit_report(request.runs_root, request.run_id)
    for filename in ("audit_report.json", "audit_report.md"):
        content = (root / filename).read_text(encoding="utf-8")
        assert "SENTINEL" not in content and "Authorization" not in content and "base64" not in content
        assert "hidden_truth" not in content and "api_key" not in content and "reasoning_content" not in content


def test_missing_telemetry_is_unavailable(completed_audit):
    request, _, root, _ = completed_audit
    (root / "model_calls/telemetry.json").unlink()
    report = build_audit_report(request.runs_root, request.run_id)
    assert report.cost["text_provider_calls"] is None
    assert report.cost["total_tokens"] is None
    assert report.runtime["snapshots"] == []


def test_cross_run_authority_fails_closed(completed_audit):
    request, _, root, _ = completed_audit
    path = root / "screening/result.json"
    value = json.loads(path.read_text())
    value["run_id"] = "other-run"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(AuditReportError, match="another run"):
        build_audit_report(request.runs_root, request.run_id)


@pytest.mark.parametrize("metadata", ["id,device\n1,A\n2,B\n", None])
def test_dataset_only_unresolved_label_and_empty_validation(tmp_path, metadata):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    if metadata:
        (dataset / "metadata.csv").write_text(metadata)
    request = build_audit_run_request(AuditRequest(dataset_path=str(dataset)), run_id="unresolved", runs_root=tmp_path / "runs")
    result = run(request, agent_client=audit_client())
    report = build_audit_report(request.runs_root, request.run_id)
    assert report.dataset["label_source"]["status"] == "unresolved"
    assert report.quantitative == report.qualitative == []
    assert report.findings == result.final_result["payload"]
    assert report.cost["text_provider_calls"] is None
    # Intake knows headers, but unresolved-label metadata was not profiled/bound.
    assert report.dataset["sample_count"] is None
    assert report.dataset["public_metadata_fields"] == (["id", "device"] if metadata else [])


def test_bounded_rows_are_not_reported_as_complete_sample_count(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "metadata.csv").write_text("id,label,site\n1,0,A\n2,1,B\n3,0,A\n")
    request = build_audit_run_request(AuditRequest(dataset_path=str(dataset)), run_id="bounded", runs_root=tmp_path / "runs")
    request = request.model_copy(update={"identify_config": {"budget": {"max_metadata_rows": 2}}})
    run(request, agent_client=audit_client())
    report = build_audit_report(request.runs_root, request.run_id)
    assert report.dataset["metadata_truncated"] and report.dataset["rows_profiled"] == 2
    assert report.dataset["sample_count"] is None


def test_unsafe_artifact_references_excluded(completed_audit):
    request, _, root, _ = completed_audit
    path = root / "final_result.json"
    summary = json.loads(path.read_text())
    summary["payload"]["artifact_references"].extend(["hidden_truth/evaluator.json", "raw_response/body.json"])
    path.write_text(json.dumps(summary), encoding="utf-8")
    report = build_audit_report(request.runs_root, request.run_id)
    assert all("hidden_truth" not in value and "raw_response" not in value for value in report.artifacts["references"])
    assert "hidden_truth" not in report.model_dump_json() and "raw_response" not in report.model_dump_json()
