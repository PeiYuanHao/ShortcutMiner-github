from copy import deepcopy

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


def _request_data() -> dict:
    return {
        "protocol_version": "1.0",
        "run_id": "waterbirds__model-a__seed42__r01",
        "message_id": "msg_identify_0001",
        "source": "orchestrator",
        "target": "identify",
        "message_type": "identify_request",
        "dataset": {
            "name": "waterbirds",
            "root": "/datasets/waterbirds",
            "metadata_file": "metadata/metadata.csv",
        },
        "task": {
            "type": "classification",
            "label_column": "y",
            "description": "Classify the bird category.",
            "intended_signal": "Bird morphology.",
        },
        "budget": {
            "max_samples": 40,
            "max_candidates": 5,
            "max_dataset_entries": 10_000,
            "max_metadata_rows": 100_000,
        },
    }


def _candidate_data() -> dict:
    return {
        "candidate_id": "C001",
        "cue": {
            "family": "context",
            "name": "background_scene",
        },
        "description": "Potential background cue.",
        "observation": "Background differs across selected rows.",
        "hypothesis": (
            "Background may be spuriously associated with the label."
        ),
        "evidence_sources": ["metadata"],
        "priority": "high",
        "status": "unverified",
    }


def _report_data() -> dict:
    return {
        "protocol_version": "1.0",
        "run_id": "waterbirds__model-a__seed42__r01",
        "message_id": "msg_candidate_report_0001",
        "source": "identify",
        "target": "executive",
        "message_type": "candidate_report",
        "in_reply_to": "msg_identify_0001",
        "artifact_references": [
            "identify/input.json",
            "identify/context.json",
        ],
        "dataset_summary": {
            "dataset": "waterbirds",
            "samples_inspected": 0,
            "metadata_used": True,
        },
        "candidates": [_candidate_data()],
        "limitations": [
            "Selected metadata rows were not decoded as samples."
        ],
    }


def test_identify_request_accepts_complete_contract_and_strips_text():
    data = _request_data()
    data["message_id"] = "  msg_identify_0001  "
    data["dataset"]["name"] = "  waterbirds  "

    request = IdentifyRequest.model_validate(data)

    assert request.protocol_version == "1.0"
    assert request.message_id == "msg_identify_0001"
    assert request.dataset.name == "waterbirds"
    assert request.task.description == "Classify the bird category."
    assert request.task.intended_signal == "Bird morphology."
    assert request.budget.max_dataset_entries == 10_000
    assert request.budget.max_metadata_rows == 100_000


def test_task_context_fields_remain_optional():
    task = TaskSpec(
        type="classification",
        label_column="y",
    )

    assert task.description is None
    assert task.intended_signal is None


def test_candidate_report_accepts_complete_contract():
    report = CandidateReport.model_validate(_report_data())

    assert report.source == "identify"
    assert report.target == "executive"
    assert report.in_reply_to == "msg_identify_0001"
    assert report.candidates[0].status == "unverified"
    assert report.candidates[0].evidence_sources == ["metadata"]


@pytest.mark.parametrize(
    "field",
    [
        "protocol_version",
        "message_id",
        "source",
        "target",
        "message_type",
    ],
)
def test_request_control_fields_must_be_explicit(field: str):
    data = _request_data()
    del data[field]

    with pytest.raises(ValidationError):
        IdentifyRequest.model_validate(data)


@pytest.mark.parametrize(
    "field",
    [
        "protocol_version",
        "message_id",
        "source",
        "target",
        "message_type",
        "in_reply_to",
        "artifact_references",
    ],
)
def test_report_control_fields_must_be_explicit(field: str):
    data = _report_data()
    del data[field]

    with pytest.raises(ValidationError):
        CandidateReport.model_validate(data)


def test_candidate_status_must_be_explicit():
    data = _candidate_data()
    del data["status"]

    with pytest.raises(ValidationError):
        Candidate.model_validate(data)


@pytest.mark.parametrize("version", ["1", "1.0.0", "2.0", 1.0])
def test_protocol_version_is_exact(version: object):
    request_data = _request_data()
    request_data["protocol_version"] = version
    report_data = _report_data()
    report_data["protocol_version"] = version

    with pytest.raises(ValidationError):
        IdentifyRequest.model_validate(request_data)

    with pytest.raises(ValidationError):
        CandidateReport.model_validate(report_data)


@pytest.mark.parametrize(
    "run_id",
    [
        "../escape",
        "run/child",
        r"run\child",
        ".hidden",
        "CON",
        "run.",
        "run name",
        "运行",
    ],
)
def test_run_id_must_be_a_safe_single_path_identifier(run_id: str):
    data = _request_data()
    data["run_id"] = run_id

    with pytest.raises(ValidationError):
        IdentifyRequest.model_validate(data)


@pytest.mark.parametrize(
    "metadata_file",
    [
        "",
        "   ",
        "/etc/passwd",
        r"C:\secret.csv",
        "../secret.csv",
        "metadata/../secret.csv",
        "metadata//secret.csv",
    ],
)
def test_metadata_file_must_be_a_safe_relative_path(metadata_file: str):
    with pytest.raises(ValidationError):
        DatasetSpec(
            name="waterbirds",
            root="/datasets/waterbirds",
            metadata_file=metadata_file,
        )


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (
            DatasetSpec,
            {"name": "waterbirds", "root": "   "},
        ),
        (
            TaskSpec,
            {"type": "classification", "label_column": "   "},
        ),
        (
            Cue,
            {"family": "context", "name": "   "},
        ),
    ],
)
def test_required_text_rejects_whitespace_only(model, kwargs: dict):
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_optional_task_text_rejects_whitespace_when_supplied():
    with pytest.raises(ValidationError):
        TaskSpec(
            type="classification",
            label_column="y",
            intended_signal="   ",
        )


@pytest.mark.parametrize(
    "candidate_id",
    ["C000", "C01", "C0001", "c001", "candidate-1"],
)
def test_candidate_id_uses_c001_format(candidate_id: str):
    data = _candidate_data()
    data["candidate_id"] = candidate_id

    with pytest.raises(ValidationError):
        Candidate.model_validate(data)


def test_invalid_cue_family_is_rejected():
    with pytest.raises(ValidationError):
        Cue(
            family="random_invalid_family",
            name="something",
        )


@pytest.mark.parametrize(
    "evidence_sources",
    [[], ["model_guess"], ["metadata", "unsupported_source"]],
)
def test_evidence_sources_use_the_canonical_enum(
    evidence_sources: list[str],
):
    data = _candidate_data()
    data["evidence_sources"] = evidence_sources

    with pytest.raises(ValidationError):
        Candidate.model_validate(data)


def test_invalid_candidate_status_is_rejected():
    data = _candidate_data()
    data["status"] = "confirmed"

    with pytest.raises(ValidationError):
        Candidate.model_validate(data)


@pytest.mark.parametrize(
    "artifact_reference",
    [
        "",
        "   ",
        "/runs/output.json",
        r"C:\runs\output.json",
        "identify/../output.json",
    ],
)
def test_artifact_references_are_safe_run_relative_paths(
    artifact_reference: str,
):
    data = _report_data()
    data["artifact_references"] = [artifact_reference]

    with pytest.raises(ValidationError):
        CandidateReport.model_validate(data)


def test_extra_fields_are_forbidden_at_every_level():
    data = _request_data()
    data["dataset"]["unexpected"] = "value"

    with pytest.raises(ValidationError):
        IdentifyRequest.model_validate(data)

    report_data = _report_data()
    report_data["candidates"][0]["unexpected"] = "value"

    with pytest.raises(ValidationError):
        CandidateReport.model_validate(report_data)


def test_strict_types_are_not_coerced():
    data = _request_data()
    data["budget"]["max_samples"] = "40"

    with pytest.raises(ValidationError):
        IdentifyRequest.model_validate(data)

    report_data = deepcopy(_report_data())
    report_data["dataset_summary"]["metadata_used"] = 1

    with pytest.raises(ValidationError):
        CandidateReport.model_validate(report_data)


def test_budget_limits_must_be_positive():
    with pytest.raises(ValidationError):
        IdentifyBudget(max_dataset_entries=0)

    with pytest.raises(ValidationError):
        IdentifyBudget(max_metadata_rows=0)


def test_candidate_budget_matches_candidate_id_space():
    with pytest.raises(ValidationError):
        IdentifyBudget(max_candidates=1000)


def test_whitespace_only_limitations_are_rejected():
    data = _report_data()
    data["limitations"] = ["   "]

    with pytest.raises(ValidationError):
        CandidateReport.model_validate(data)


def test_dataset_summary_requires_strict_non_negative_counts():
    with pytest.raises(ValidationError):
        DatasetInspectionSummary(
            dataset="waterbirds",
            samples_inspected=-1,
            metadata_used=True,
        )
