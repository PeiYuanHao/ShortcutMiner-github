"""Automatic audit authority, deterministic recovery, and real planning regression."""

import json
from pathlib import Path

import pytest
from PIL import Image

import shortcutminer.experiment.audit_capabilities as capabilities_module
from shortcutminer.audit import build_audit_run_request
from shortcutminer.experiment.audit_capabilities import build_audit_capabilities, load_or_build_audit_capabilities
from shortcutminer.experiment.dataset_profiler import build_dataset_profile
from shortcutminer.experiment.check_requirements import build_candidate_contexts
from shortcutminer.protocol.audit import AuditRequest
from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.runner import build_local_tool_runtimes, run, resume_run
from shortcutminer.runtime.demo import DemoAgentClient
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolRuntime
from shortcutminer.workflow.nodes import build_identify_request_from_state, build_executive_request_from_state


def inputs(tmp_path, *, header="id,label,site,device", rows="1,0,A,X\n2,0,A,X\n3,1,B,Y\n4,1,B,Y\n", image=False):
    root = tmp_path / "dataset"
    root.mkdir()
    if header is not None:
        (root / "metadata.csv").write_text(header + "\n" + rows, encoding="utf-8")
    if image:
        Image.new("RGB", (8, 8), color="red").save(root / "sample.png")
    request = build_audit_run_request(AuditRequest(dataset_path=str(root)),
                                      run_id="auto-capabilities", runs_root=tmp_path / "runs")
    audit = AuditRequest.model_validate(request.metadata["audit_request"])
    state = {"run_id": request.run_id, "dataset_info": request.dataset_info.model_dump(mode="json"),
             "task_input": request.task_input.model_dump(mode="json"), "identify_config": {"seed": 42}}
    identify = build_identify_request_from_state(state)
    profile = build_dataset_profile(identify, seed=42)
    report = CandidateReport(protocol_version="1.0", run_id=request.run_id, message_id="identify:report",
        source="identify", target="executive", message_type="candidate_report", in_reply_to=identify.message_id,
        artifact_references=[], dataset_summary={"dataset": audit.dataset_info.name,
            "samples_inspected": 1 if image else 0, "metadata_used": profile.metadata.available},
        candidates=[{"candidate_id": "C001", "cue": {"family": "metadata" if header else "other",
            "name": "site" if header else "background"}, "metadata_field_ref": "site" if header else None,
            "description": "Potential cue", "observation": "Observed public data", "hypothesis": "Potential shortcut",
            "evidence_sources": ["metadata"] if header else ["sample_inspection"], "priority": "high",
            "status": "unverified"}], limitations=[])
    runtimes = build_local_tool_runtimes(run_id=request.run_id, dataset_root=root, runs_root=request.runs_root)
    kwargs = {"quantitative_runtime": runtimes["quantitative"], "qualitative_runtime": runtimes["qualitative"]}
    return request, audit, profile, report, runtimes, kwargs


def test_metadata_discovered_with_exact_control_and_scientific_fields(tmp_path):
    _, audit, profile, report, _, kwargs = inputs(tmp_path)
    result = build_audit_capabilities(audit, profile, report, **kwargs)
    assert result.capabilities.available_metadata_fields == ["device", "id", "label", "site"]
    assert result.capabilities.addressing_fields == ["label", "id"]
    assert result.execution_resources == {"sample_id_column": "id"}
    assert result.capabilities.quantitative_checks == ["association_test", "subgroup_evaluation", "model_evaluation", "intervention"]
    assert result.capabilities.qualitative_checks == ["metadata_inspection"]
    assert "statistical_check" in result.capabilities.supported_experiment_types
    assert not result.capabilities.sample_access
    assert report.candidates[0].status == "unverified"


def test_image_without_metadata_advertises_existing_sample_inspection(tmp_path):
    _, audit, profile, report, _, kwargs = inputs(tmp_path, header=None, image=True)
    result = build_audit_capabilities(audit, profile, report, **kwargs)
    assert result.capabilities.image_access and result.capabilities.sample_access
    assert result.capabilities.available_metadata_fields == []
    assert result.capabilities.quantitative_checks == []
    assert result.capabilities.qualitative_checks == ["sample_inspection", "counterexample_inspection"]
    assert result.qualitative_preparation == {"sampling_strategy": "random"}
    assert "executable_label_source_unavailable" in result.capabilities.limitations


def test_no_metadata_or_samples_retains_limitations(tmp_path):
    _, audit, profile, report, _, kwargs = inputs(tmp_path, header=None)
    result = build_audit_capabilities(audit, profile, report, **kwargs)
    assert result.capabilities.available_targets == []
    assert not result.capabilities.sample_access
    assert "raw_sample_inspection_unavailable" in result.capabilities.limitations


def test_only_intake_confirmed_fields_no_truth_or_fuzzy_binding(tmp_path):
    _, audit, profile, report, _, kwargs = inputs(tmp_path,
        header="id,label,site,hidden_truth,unapproved", rows="1,0,A,true,Z\n2,1,B,false,Z\n")
    audit = audit.model_copy(update={"dataset_info": audit.dataset_info.model_copy(update={
        "metadata_columns": ["id", "label", "site", "hidden_truth"]})})
    result = build_audit_capabilities(audit, profile, report, **kwargs)
    assert result.capabilities.available_metadata_fields == ["id", "label", "site"]
    assert result.dataset_profile.metadata.column_names == ["id", "label", "site"]
    assert "hidden_truth" not in [column.name for column in result.dataset_profile.metadata.columns]
    assert "site_device_batch" not in result.capabilities.available_metadata_fields
    assert "unapproved" not in result.capabilities.available_metadata_fields


def test_builder_reads_no_source_files_and_calls_no_models_or_tools(tmp_path, monkeypatch):
    _, audit, profile, report, runtimes, kwargs = inputs(tmp_path)
    expected = build_audit_capabilities(audit, profile, report, **kwargs)
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: pytest.fail("No data/private file reads"))
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("No source file reads"))
    for runtime in runtimes.values():
        monkeypatch.setattr(runtime.registry, "execute", lambda *args, **kwargs: pytest.fail("No tool invocation"))
        runtime.history.append({"hidden_truth": "poison", "future_result": "supported"})
    assert build_audit_capabilities(audit, profile, report, **kwargs) == expected


def test_registered_tool_without_executor_is_not_available_backend(tmp_path):
    request, audit, profile, report, _, kwargs = inputs(tmp_path)
    kwargs["quantitative_runtime"] = ToolRuntime(build_default_tool_registry(), run_id=request.run_id,
        agent_name="quantitative", dataset_root=request.dataset_info.root)
    result = build_audit_capabilities(audit, profile, report, **kwargs)
    assert result.capabilities.quantitative_checks == []
    assert result.capabilities.supported_experiment_types == []
    assert "quantitative_backend_unavailable" in result.capabilities.limitations


def test_hash_stable_and_persisted_context_reused(tmp_path, monkeypatch):
    request, audit, profile, report, _, kwargs = inputs(tmp_path)
    result = load_or_build_audit_capabilities(audit, profile, report, runs_root=request.runs_root, **kwargs)
    assert build_audit_capabilities(audit, profile, report, **kwargs).input_hash == result.input_hash
    reordered = audit.model_copy(update={"dataset_info": audit.dataset_info.model_copy(update={
        "metadata_columns": list(reversed(audit.dataset_info.metadata_columns))})})
    assert build_audit_capabilities(reordered, profile, report, **kwargs).result_hash == result.result_hash
    monkeypatch.setattr(capabilities_module, "build_audit_capabilities", lambda *args, **kwargs: pytest.fail("Must reuse"))
    assert load_or_build_audit_capabilities(audit, profile, report, runs_root=request.runs_root, **kwargs) == result
    assert load_or_build_audit_capabilities(audit, profile, report, persisted=result.model_dump(mode="json"), **kwargs) == result
    with pytest.raises(ValueError, match="authority changed"):
        load_or_build_audit_capabilities(audit, profile, report, runs_root=request.runs_root)
    tampered = result.model_dump(mode="json")
    tampered["capabilities"]["available_metadata_fields"].append("unauthorized")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_or_build_audit_capabilities(audit, profile, report, persisted=tampered, **kwargs)


def test_ambiguous_sample_addressing_not_guessed(tmp_path):
    _, audit, profile, report, _, kwargs = inputs(tmp_path, header="id,sample_id,label,site",
                                                rows="1,A,0,X\n2,B,1,Y\n")
    result = build_audit_capabilities(audit, profile, report, **kwargs)
    assert "sample_id_column" not in result.execution_resources
    assert {"id", "sample_id"}.issubset(result.capabilities.addressing_fields)
    assert "ambiguous_sample_id_column" in result.capabilities.limitations


def test_foreign_root_backend_and_metadata_source_rejected(tmp_path):
    _, audit, profile, report, runtimes, kwargs = inputs(tmp_path)
    foreign = profile.model_copy(update={"dataset": profile.dataset.model_copy(update={"root_reference": str(tmp_path)})})
    with pytest.raises(ValueError, match="different datasets"):
        build_audit_capabilities(audit, foreign, report, **kwargs)
    foreign = profile.model_copy(update={"metadata": profile.metadata.model_copy(update={"file": "private.csv"})})
    with pytest.raises(ValueError, match="not authorized"):
        build_audit_capabilities(audit, foreign, report, **kwargs)
    kwargs["quantitative_runtime"] = runtimes["executive"]
    with pytest.raises(ValueError, match="backend authority"):
        build_audit_capabilities(audit, profile, report, **kwargs)


def test_candidate_cannot_fuzzy_bind_or_authorize_private_fields(tmp_path):
    _, audit, profile, report, _, kwargs = inputs(tmp_path)
    for field in ("site_device_batch", "hidden_truth", "id", "label"):
        candidate = report.candidates[0].model_copy(update={"metadata_field_ref": field})
        unbound = report.model_copy(update={"candidates": [candidate]})
        result = build_audit_capabilities(audit, profile, unbound, **kwargs)
        assert result.capabilities.available_validation_methods == []
        assert "candidate_binding_unresolved:C001" in result.capabilities.limitations


def test_direct_csv_dataset_path_supported(tmp_path):
    _, audit, profile, report, _, kwargs = inputs(tmp_path)
    audit = audit.model_copy(update={"dataset_path": str(Path(audit.dataset_info.root) / "metadata.csv")})
    assert build_audit_capabilities(audit, profile, report, **kwargs).capabilities.quantitative_checks


def test_dataset_only_request_generates_plan_via_executive_and_existing_finalizer(tmp_path):
    request, _, _, _, runtimes, _ = inputs(tmp_path)
    assert request.executive_config.keys() == {"limitations"}
    assert request.quantitative_config == request.qualitative_config == {}
    client = DemoAgentClient()
    result = run(request, agent_client=client, tool_runtimes=runtimes)
    state = result.workflow_state
    assert result.status == "COMPLETED"
    names = [call["agent_name"] for call in client.calls]
    assert names[:2] == ["identify", "executive"] and names[-1] == "summarize"
    assert sorted(names[2:-1]) == ["qualitative", "quantitative"]
    assert state["executive_request"]["routing_context"]["dataset_profile"] is not None
    assert {task["task_id"] for task in state["executive_result"]["tasks"]} == {"TASK-C001-QT", "TASK-C001-QL"}
    assert all(task["inputs"]["metadata_fields"] == ["site"] for task in state["executive_result"]["tasks"])
    assert state["quantitative_result"]["results"][0]["payload"]["measurements"]
    contexts = build_candidate_contexts(build_executive_request_from_state(state))
    assert contexts[0].bound_metadata_fields == ["site"]
    assert (Path(request.runs_root) / request.run_id / "capabilities" / "result.json").is_file()


def test_resume_at_executive_reuses_capabilities_without_intake_or_rebuild(tmp_path, monkeypatch):
    request, _, _, _, runtimes, _ = inputs(tmp_path)
    with pytest.raises(Exception, match="Planned mock failure"):
        run(request, agent_client=DemoAgentClient(fail_at="executive"), tool_runtimes=runtimes)
    artifact = Path(request.runs_root) / request.run_id / "capabilities" / "result.json"
    previous = artifact.read_bytes()
    monkeypatch.setattr(capabilities_module, "build_audit_capabilities", lambda *args, **kwargs: pytest.fail("Must reuse"))
    monkeypatch.setattr("shortcutminer.audit.intake_dataset", lambda *args: pytest.fail("No repeated intake"))
    client = DemoAgentClient()
    result = resume_run(request.run_id, runs_root=request.runs_root, agent_client=client, tool_runtimes=runtimes)
    assert result.status == "COMPLETED"
    assert client.calls[0]["agent_name"] == "executive"
    assert artifact.read_bytes() == previous
    assert result.workflow_state["audit_capabilities"] == json.loads(previous)
