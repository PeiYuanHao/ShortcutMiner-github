import pytest
from pydantic import ValidationError

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


def test_identify_request():
    request = IdentifyRequest(
        run_id="waterbirds__model-a__seed42__r01",
        dataset=DatasetSpec(
            name="waterbirds",
            root="/datasets/waterbirds",
            metadata_file="metadata.csv",
        ),
        task=TaskSpec(
            type="classification",
            label_column="y",
        ),
        budget=IdentifyBudget(
            max_samples=40,
            max_candidates=5,
        ),
    )

    assert request.protocol_version == "1.0"
    assert request.dataset.name == "waterbirds"
    assert request.budget.max_candidates == 5


def test_candidate():
    candidate = Candidate(
        candidate_id="C001",
        cue=Cue(
            family="context",
            name="background_scene",
        ),
        description=(
            "Background scene appears associated "
            "with the target class."
        ),
        observation=(
            "Different classes frequently appear "
            "with different background types."
        ),
        hypothesis=(
            "Background scene may be spuriously "
            "correlated with the target label."
        ),
        evidence_sources=[
            "sample_inspection",
            "metadata",
        ],
        priority="high",
    )

    assert candidate.status == "unverified"
    assert candidate.cue.family == "context"


def test_candidate_report():
    report = CandidateReport(
        run_id="waterbirds__model-a__seed42__r01",
        dataset_summary=DatasetInspectionSummary(
            dataset="waterbirds",
            samples_inspected=40,
            metadata_used=True,
        ),
        candidates=[
            Candidate(
                candidate_id="C001",
                cue=Cue(
                    family="context",
                    name="background_scene",
                ),
                description="Potential background cue.",
                observation="Background differs across classes.",
                hypothesis=(
                    "Background may be spuriously "
                    "associated with the label."
                ),
                evidence_sources=["sample_inspection"],
                priority="high",
            )
        ],
        limitations=[
            "Candidates have not been validated."
        ],
    )

    assert report.source == "identify"
    assert report.message_type == "candidate_report"
    assert len(report.candidates) == 1
    assert report.candidates[0].status == "unverified"


def test_invalid_cue_family():
    with pytest.raises(ValidationError):
        Cue(
            family="random_invalid_family",
            name="something",
        )


def test_invalid_candidate_status():
    with pytest.raises(ValidationError):
        Candidate(
            candidate_id="C001",
            cue=Cue(
                family="context",
                name="background_scene",
            ),
            description="Potential shortcut.",
            observation="Observed pattern.",
            hypothesis="Possible shortcut hypothesis.",
            priority="high",
            status="confirmed",
        )