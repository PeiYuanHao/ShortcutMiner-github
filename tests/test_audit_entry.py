import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from shortcutminer.audit import UNAVAILABLE_LABEL_COLUMN, build_audit_run_request, persist_audit_request
from shortcutminer.protocol.audit import AuditRequest
from shortcutminer.runner import resume_run, run
from shortcutminer.runtime.mock import MockAgentClient
from shortcutminer.storage.run_store import load_run_manifest


def audit_client(*, fail_summary=False):
    def identify(payload, _prompt):
        # The test provider discovers a candidate after the entry builder returns.
        return {"candidates": [{"cue": {"family": "other", "name": "dataset structure"},
                "description": "A structural cue might be present.", "observation": "A bounded profile is available.",
                "hypothesis": "Dataset structure might encode an unintended signal.",
                "evidence_sources": ["dataset_structure"], "priority": "low",
                "metadata_field_ref": None, "additional_limitations": []}], "additional_limitations": []}

    def executive(payload, _prompt):
        return {"decisions": [{"candidate_id": candidate["candidate_id"], "route": "unresolved",
                "routing_reason": "Step1 does not auto-configure validation capabilities.",
                "quantitative": None, "qualitative": None,
                "missing_requirements": ["No validation capabilities were configured."], "limitations": []}
                for candidate in payload["request"]["candidate_report"]["candidates"]], "limitations": []}

    def summarize(payload, _prompt):
        if fail_summary:
            raise RuntimeError("planned summary failure")
        return {"overview": "No shortcut conclusion is available from this entry-only run.",
                "findings": [{"candidate_ref": "C001", "assessment": "insufficient_evidence",
                              "summary": "The hypothesis was not validated.",
                              "additional_limitations": [], "recommendations": []}],
                "additional_limitations": []}

    return MockAgentClient({"identify": identify, "executive": executive, "summarize": summarize})


@pytest.mark.parametrize("labelled", [True, False])
def test_builder_enters_existing_graph_without_creating_tasks(tmp_path, labelled):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "metadata.csv").write_text(
        "id,label,device\n1,0,A\n2,1,B\n" if labelled else "id,device\n1,A\n2,B\n",
        encoding="utf-8")
    request = build_audit_run_request(AuditRequest(dataset_path=str(dataset)),
                                      runs_root=tmp_path / "runs", run_id="audit-entry")
    assert request.executive_config.keys() == {"limitations"}
    assert request.quantitative_config == request.qualitative_config == {}
    assert request.task_input.label_column == ("label" if labelled else UNAVAILABLE_LABEL_COLUMN)
    saved = persist_audit_request(request)
    client = audit_client()
    result = run(request, agent_client=client)
    assert result.status == "COMPLETED"
    assert [call["agent_name"] for call in client.calls] == ["identify", "executive", "summarize"]
    assert result.workflow_state["stage_status"] == {
        name: "completed" for name in ["identify", "screening", "capabilities", "executive", "feasibility", "quantitative",
                                       "qualitative", "evidence_calibration", "summarize"]}
    assert result.workflow_state["executive_result"]["tasks"] == []
    assert result.workflow_state["identify_result"]["candidates"][0]["status"] == "unverified"
    assert (saved.parent / "checkpoints.sqlite").is_file()
    assert (saved.parent / "final_result.json").is_file()
    assert json.loads(saved.read_text(encoding="utf-8")) == request.metadata["audit_request"]
    assert load_run_manifest(request.runs_root, request.run_id).metadata["audit_request"] == request.metadata["audit_request"]
    if not labelled:
        assert result.workflow_state["dataset_profile"]["labels"]["validity"] == "source_unavailable"
        assert "label_source_unresolved" in client.calls[0]["payload"]["request"]["task"]["description"]


def test_directory_label_intake_keeps_existing_profiler_boundary(tmp_path):
    dataset = tmp_path / "dataset"
    for label in ["cats", "dogs"]:
        (dataset / label).mkdir(parents=True)
        Image.new("RGB", (4, 4)).save(dataset / label / "sample.png")
    before = sorted(path.relative_to(dataset).as_posix() for path in dataset.rglob("*"))
    request = build_audit_run_request(AuditRequest(dataset_path=str(dataset)),
                                      runs_root=tmp_path / "runs", run_id="directory-audit")
    assert request.metadata["audit_request"]["dataset_info"]["label_source"]["status"] == "resolved"
    assert request.dataset_info.metadata_file is None
    assert any("directory_labels_not_bound" in item for item in request.metadata["audit_limitations"])
    result = run(request, agent_client=audit_client())
    assert result.status == "COMPLETED"
    assert result.workflow_state["dataset_profile"]["structure"]["modality"] == "image"
    assert before == sorted(path.relative_to(dataset).as_posix() for path in dataset.rglob("*"))


def test_existing_resume_uses_persisted_audit_inputs(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    request = build_audit_run_request(AuditRequest(dataset_path=str(dataset)),
                                      runs_root=tmp_path / "runs", run_id="audit-resume")
    saved = persist_audit_request(request)
    original = saved.read_bytes()
    with pytest.raises(Exception, match="planned summary failure"):
        run(request, agent_client=audit_client(fail_summary=True))
    monkeypatch.setattr("shortcutminer.audit.intake_dataset", lambda _: pytest.fail("Resume must not re-run intake"))
    client = audit_client()
    result = resume_run(request.run_id, runs_root=request.runs_root, agent_client=client)
    assert result.status == "COMPLETED"
    assert [call["agent_name"] for call in client.calls] == ["summarize"]
    assert saved.read_bytes() == original


def test_audit_contract_rejects_task_fields_and_preserves_existing_artifacts(tmp_path):
    with pytest.raises(ValidationError):
        AuditRequest(dataset_path=str(tmp_path), experiment_plan={})
    with pytest.raises(ValidationError):
        build_audit_run_request(AuditRequest(dataset_path=str(tmp_path)), run_id="../escape")
    request = build_audit_run_request(AuditRequest(dataset_path=str(tmp_path)),
                                      runs_root=tmp_path / "runs", run_id="same-run")
    path = persist_audit_request(request)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        persist_audit_request(request)
    assert path.read_bytes() == original
