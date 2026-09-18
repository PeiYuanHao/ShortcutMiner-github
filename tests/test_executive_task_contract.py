import hashlib
import json
from pathlib import Path

import pytest

from shortcutminer.experiment.candidate_binding import resolve_candidate_binding
from shortcutminer.experiment.check_requirements import (
    AnalysisTaskSemanticError,
    build_candidate_contexts,
    build_executable_candidate_context,
    validate_analysis_task_semantics,
)
from shortcutminer.experiment.executive_executor import (
    ExecutiveDraftSemanticError,
    build_executive_agent_payload,
    finalize_executive_draft,
    validate_executive_draft,
    validate_routing_plan,
)
from shortcutminer.protocol.executive import (
    ExecutiveAgentDraft,
    ExecutiveAuditOptions,
    ExecutiveCapabilities,
    ExecutiveRequest,
    ExecutiveRoutingContext,
)
from shortcutminer.protocol.identify import (
    Candidate,
    CandidateReport,
    Cue,
    DatasetInspectionSummary,
    DatasetSpec,
    TaskSpec,
)


def _candidate(*, family: str = "metadata", name: str = "device") -> Candidate:
    return Candidate(
        candidate_id="C001",
        cue=Cue(family=family, name=name),
        description="A public candidate description.",
        observation="A bounded public observation.",
        hypothesis="The candidate may be associated with the label.",
        evidence_sources=["metadata" if family == "metadata" else "samples"],
        priority="high",
        status="unverified",
    )


def _request(tmp_path: Path, *, candidate: Candidate | None = None) -> ExecutiveRequest:
    candidate = candidate or _candidate()
    report = CandidateReport(
        protocol_version="1.0",
        run_id="contract_test",
        message_id="contract_test:identify:report",
        source="identify",
        target="executive",
        message_type="candidate_report",
        in_reply_to="contract_test:identify:request",
        dataset_summary=DatasetInspectionSummary(
            dataset="dataset", samples_inspected=4, metadata_used=True
        ),
        candidates=[candidate],
        artifact_references=[],
        limitations=[],
    )
    return ExecutiveRequest(
        protocol_version="1.0",
        run_id="contract_test",
        message_id="contract_test:executive:request",
        source="orchestrator",
        target="executive",
        message_type="executive_request",
        in_reply_to=report.message_id,
        candidate_report=report,
        routing_context=ExecutiveRoutingContext(
            protocol_version="1.0",
            run_id="contract_test",
            dataset=DatasetSpec(
                name="dataset", root=str(tmp_path), metadata_file="metadata.csv"
            ),
            task=TaskSpec(type="classification", label_column="label"),
            audit_options=ExecutiveAuditOptions(
                max_tasks=2,
                max_samples_per_task=8,
                allowed_targets=["quantitative", "qualitative"],
                requested_outputs=[],
                runtime_constraints=[],
            ),
            capabilities=ExecutiveCapabilities(
                available_targets=["quantitative", "qualitative"],
                available_metadata_fields=["id", "label", "site", "device", "path", "split"],
                addressing_fields=["id", "path", "split"],
                quantitative_checks=[
                    "association_test", "subgroup_evaluation", "model_evaluation",
                    "intervention", "ablation", "metric_computation",
                    "task_model_training", "task_model_evaluation", "ood_evaluation",
                    "counterfactual_intervention",
                ],
                qualitative_checks=[
                    "metadata_inspection", "sample_inspection",
                    "counterexample_inspection",
                ],
                model_capabilities=["retraining", "checkpoint", "intervention_inference"],
                sample_access=True,
                available_output_types=["metrics", "observations"],
            ),
            artifact_references=[],
            limitations=[],
        ),
    )


def _draft(*, route="quantitative", quantitative_checks=None) -> ExecutiveAgentDraft:
    quantitative = None
    qualitative = None
    if route in {"quantitative", "both"}:
        quantitative = {
            "objective": "Test the candidate with authorized measurements.",
            "requested_checks": quantitative_checks or ["association_test"],
        }
    if route in {"qualitative", "both"}:
        qualitative = {
            "objective": "Inspect authorized public samples.",
            "requested_checks": ["sample_inspection"],
        }
    return ExecutiveAgentDraft.model_validate({
        "decisions": [{
            "candidate_id": "C001",
            "route": route,
            "routing_reason": "Use candidate-specific executable checks.",
            "quantitative": quantitative,
            "qualitative": qualitative,
            "missing_requirements": ["No executable check."] if route == "unresolved" else [],
            "limitations": [],
        }],
        "limitations": [],
    })


def _hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_candidate_binding_is_exact_and_has_addressing_firewall():
    allowed = ["id", "label", "site", "device", "path", "split"]
    exact = resolve_candidate_binding(
        _candidate(), available_metadata_fields=allowed, label_column="label",
        addressing_fields=["id", "path", "split"],
    )
    assert exact.status == "resolved"
    assert exact.metadata_fields == ["device"]
    fuzzy = resolve_candidate_binding(
        _candidate(name="device information"), available_metadata_fields=allowed,
        label_column="label", addressing_fields=["id", "path", "split"],
    )
    assert fuzzy.status == "unresolved"
    for field in ("id", "label", "path", "split"):
        binding = resolve_candidate_binding(
            _candidate(name=field), available_metadata_fields=allowed,
            label_column="label", addressing_fields=["id", "path", "split"],
        )
        assert binding.status == "unresolved"


def test_requirement_matrix_is_candidate_specific(tmp_path: Path):
    request = _request(tmp_path)
    context = build_candidate_contexts(request)[0]
    assert "association_test" in context.allowed_quantitative_checks
    assert "counterfactual_intervention" not in context.allowed_quantitative_checks
    assert context.unavailable_checks_with_reasons["counterfactual_intervention"] == (
        "requires_structured_scientific_spec"
    )

    unresolved_request = _request(tmp_path, candidate=_candidate(name="device information"))
    unresolved = build_candidate_contexts(unresolved_request)[0]
    assert "association_test" not in unresolved.allowed_quantitative_checks

    request.routing_context.capabilities.model_capabilities = ["checkpoint"]
    missing_training = build_candidate_contexts(request)[0]
    assert "task_model_training" not in missing_training.allowed_quantitative_checks
    request.routing_context.capabilities.model_capabilities = ["retraining"]
    request.routing_context.audit_options.runtime_constraints = ["no_retraining"]
    no_retraining = build_candidate_contexts(request)[0]
    assert "task_model_training" not in no_retraining.allowed_quantitative_checks

    request.routing_context.audit_options.runtime_constraints = []
    request.routing_context.capabilities.model_capabilities = [
        "checkpoint", "intervention_inference"
    ]
    binding = resolve_candidate_binding(
        request.candidate_report.candidates[0],
        available_metadata_fields=request.routing_context.capabilities.available_metadata_fields,
        label_column="label", addressing_fields=["id", "path", "split"],
    )
    with_spec = build_executable_candidate_context(
        binding, capabilities=request.routing_context.capabilities,
        runtime_constraints=[], structured_spec_available=True,
    )
    assert "counterfactual_intervention" in with_spec.allowed_quantitative_checks


def test_draft_finalizer_and_semantic_gate(tmp_path: Path):
    request = _request(tmp_path)
    contexts = build_candidate_contexts(request)
    unavailable = _draft(quantitative_checks=["counterfactual_intervention"])
    with pytest.raises(ExecutiveDraftSemanticError, match="unavailable"):
        validate_executive_draft(unavailable, request, contexts)

    plan = finalize_executive_draft(request, contexts, _draft(route="both"))
    assert [task.task_id for task in plan.tasks] == ["TASK-C001-QT", "TASK-C001-QL"]
    quantitative = plan.tasks[0]
    assert quantitative.inputs.metadata_fields == ["device"]
    assert quantitative.inputs.required_model_capabilities == []
    assert quantitative.inputs.requires_sample_access is False
    validate_routing_plan(plan, request, candidate_contexts=contexts)

    unresolved = finalize_executive_draft(
        _request(tmp_path, candidate=_candidate(name="unknown field")),
        build_candidate_contexts(_request(tmp_path, candidate=_candidate(name="unknown field"))),
        _draft(route="unresolved"),
    )
    assert unresolved.tasks == []


def test_legacy_association_without_metadata_is_rejected(tmp_path: Path):
    request = _request(tmp_path)
    contexts = build_candidate_contexts(request)
    plan = finalize_executive_draft(request, contexts, _draft())
    malformed = plan.tasks[0].model_copy(
        update={"inputs": plan.tasks[0].inputs.model_copy(update={"metadata_fields": []})}
    )
    with pytest.raises(AnalysisTaskSemanticError, match="missing_scientific"):
        validate_analysis_task_semantics(
            malformed, contexts[0], request.routing_context.capabilities
        )


def test_hidden_truth_cannot_change_binding_payload_or_final_plan(tmp_path: Path):
    request = _request(tmp_path)
    variants = [None, {"target_field": "device"}, {"target_field": "site"}]
    hashes = []
    for private_truth in variants:
        envelope = {"request": request.model_dump(mode="json")}
        if private_truth is not None:
            envelope["private_hidden_truth"] = private_truth
        production_request = ExecutiveRequest.model_validate(envelope["request"])
        contexts = build_candidate_contexts(production_request)
        plan = finalize_executive_draft(production_request, contexts, _draft())
        hashes.append(_hash({
            "binding": contexts[0].model_dump(mode="json"),
            "payload": build_executive_agent_payload(production_request, contexts),
            "plan": plan.model_dump(mode="json"),
        }))
    assert len(set(hashes)) == 1


def test_executive_workspace_stays_compact():
    prompt = Path("agents/executive/workspace/AGENTS.md").read_text(encoding="utf-8")
    assert len(prompt.encode("utf-8")) <= 20_000
    assert "ExecutiveAgentDraft" in prompt
