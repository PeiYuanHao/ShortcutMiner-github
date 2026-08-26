from pathlib import Path

import pytest

from shortcutminer.experiment.executive_context import (
    ExecutiveContextError,
    build_executive_request,
)
from shortcutminer.experiment.identify_context import build_identify_context
from shortcutminer.protocol.executive import (
    ExecutiveAuditOptions,
    ExecutiveCapabilities,
)
from shortcutminer.protocol.identify import (
    Candidate,
    CandidateReport,
    Cue,
    DatasetInspectionSummary,
    DatasetSpec,
    IdentifyBudget,
    IdentifyRequest,
    TaskSpec,
)


def _identify_state(tmp_path: Path):
    (tmp_path / "metadata.csv").write_text(
        "id,label,site\n1,0,A\n2,1,B\n",
        encoding="utf-8",
    )
    request = IdentifyRequest(
        protocol_version="1.0",
        run_id="executive_context_run",
        message_id="identify:request:001",
        source="orchestrator",
        target="identify",
        message_type="identify_request",
        dataset=DatasetSpec(
            name="test_dataset",
            root=str(tmp_path),
            metadata_file="metadata.csv",
        ),
        task=TaskSpec(type="classification", label_column="label"),
        budget=IdentifyBudget(max_samples=2, max_candidates=1),
    )
    context = build_identify_context(request=request, seed=42)
    report = CandidateReport(
        protocol_version="1.0",
        run_id=request.run_id,
        message_id="identify:report:001",
        source="identify",
        target="executive",
        message_type="candidate_report",
        in_reply_to=request.message_id,
        artifact_references=[],
        dataset_summary=DatasetInspectionSummary(
            dataset="test_dataset",
            samples_inspected=0,
            metadata_used=True,
        ),
        candidates=[
            Candidate(
                candidate_id="C001",
                cue=Cue(family="metadata", name="site"),
                description="Site may be an unintended cue.",
                observation="The metadata contains site values.",
                hypothesis="Site may be associated with the label.",
                evidence_sources=["metadata"],
                priority="medium",
                status="unverified",
            )
        ],
        limitations=list(context.limitations),
    )
    return request, context, report


def _capabilities() -> ExecutiveCapabilities:
    return ExecutiveCapabilities(
        available_targets=["quantitative"],
        available_metadata_fields=["label", "site"],
        available_splits=[],
        quantitative_checks=["association_test"],
        qualitative_checks=[],
        model_capabilities=[],
        sample_access=False,
        available_output_types=["metrics"],
    )


def test_build_executive_request_preserves_identify_state(tmp_path: Path):
    identify_request, identify_context, report = _identify_state(tmp_path)

    request = build_executive_request(
        identify_request=identify_request,
        identify_context=identify_context,
        candidate_report=report,
        message_id="executive:request:001",
        audit_options=ExecutiveAuditOptions(max_tasks=1),
        capabilities=_capabilities(),
        artifact_references=["identify/output.json"],
    )

    assert request.in_reply_to == report.message_id
    assert request.candidate_report.candidates[0].candidate_id == "C001"
    assert request.routing_context.dataset == identify_request.dataset
    assert "identify/output.json" in request.routing_context.artifact_references
    assert request.routing_context.limitations == identify_context.limitations


def test_context_rejects_mixed_runs(tmp_path: Path):
    identify_request, identify_context, report = _identify_state(tmp_path)
    report.run_id = "another_run"

    with pytest.raises(ExecutiveContextError, match="run_id"):
        build_executive_request(
            identify_request=identify_request,
            identify_context=identify_context,
            candidate_report=report,
            message_id="executive:request:001",
            audit_options=ExecutiveAuditOptions(),
            capabilities=_capabilities(),
        )


def test_context_rejects_unprofiled_metadata_capability(tmp_path: Path):
    identify_request, identify_context, report = _identify_state(tmp_path)
    capabilities = _capabilities()
    capabilities.available_metadata_fields.append("hospital_id")

    with pytest.raises(ExecutiveContextError, match="absent"):
        build_executive_request(
            identify_request=identify_request,
            identify_context=identify_context,
            candidate_report=report,
            message_id="executive:request:001",
            audit_options=ExecutiveAuditOptions(),
            capabilities=capabilities,
        )


def test_context_rejects_false_identify_sample_count(tmp_path: Path):
    identify_request, identify_context, report = _identify_state(tmp_path)
    report.dataset_summary.samples_inspected = 1

    with pytest.raises(ExecutiveContextError, match="samples_inspected"):
        build_executive_request(
            identify_request=identify_request,
            identify_context=identify_context,
            candidate_report=report,
            message_id="executive:request:001",
            audit_options=ExecutiveAuditOptions(),
            capabilities=_capabilities(),
        )
