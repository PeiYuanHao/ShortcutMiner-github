import json
from pathlib import Path

import pytest

from shortcutminer.experiment.qualitative_context import build_qualitative_context
from shortcutminer.experiment.qualitative_executor import (
    QualitativeExecutionError,
    execute_qualitative,
    finalize_qualitative_draft,
    validate_qualitative_result,
)
from test_qualitative_context import _dataset, _request
from shortcutminer.protocol.qualitative import QualitativeAgentDraft, QualitativeResult


class MockQualitativeClient:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def invoke_agent(self, agent_name: str, payload: dict):
        self.calls.append({"agent_name": agent_name, "payload": payload})
        return self.output


def _valid_draft(request, context) -> dict:
    samples = context.selected_samples
    sample_ids = [sample.sample_id for sample in samples]
    return {
        "sample_ids": sample_ids,
        "evidence_status": "partially_supported",
        "evidence_summary": "The inspected samples contain mixed evidence.",
        "observations": [{
            "sample_refs": [samples[0].sample_id],
            "observation": "A border is visible in this inspected sample.",
            "inspection_quality": "sufficient",
        }],
        "comparisons": [],
        "interpretations": [{
            "interpretation": "The border may reflect preprocessing.",
            "supporting_observation_indexes": [0],
            "supporting_comparison_indexes": [],
            "confidence": "low",
        }],
        "counterevidence": [{
            "sample_refs": [samples[-1].sample_id],
            "description": "A comparison sample does not fit the pattern.",
        }],
        "metadata_evidence": [{
            "field": "label",
            "observation": "Both configured label groups were sampled.",
            "sample_refs": sample_ids,
        }],
        "failures": [],
        "additional_limitations": [],
        "artifact_references": [],
        "recommended_followup": ["Inspect a larger bounded sample."],
    }


def _valid_output(request, context) -> dict:
    draft = QualitativeAgentDraft.model_validate(_valid_draft(request, context))
    return finalize_qualitative_draft(draft, request, context).model_dump(mode="json")


def test_execute_qualitative_persists_evidence_and_events(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _dataset(dataset_root)
    request = _request(dataset_root)
    context = build_qualitative_context(request)
    output = _valid_draft(request, context)
    runs_root = tmp_path / "runs"

    result = execute_qualitative(
        request,
        MockQualitativeClient(output),
        runs_root=runs_root,
    )

    assert result.payload.evidence_status == "partially_supported"
    stage_dir = runs_root / request.run_id / "qualitative"
    assert json.loads((stage_dir / "input.json").read_text("utf-8"))[
        "evidence_context"
    ]["samples_selected"] == 4
    assert (stage_dir / "raw_output.json").exists()
    assert (stage_dir / "result.json").exists()
    events = [
        json.loads(line)["event"]
        for line in (runs_root / request.run_id / "events.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    assert events == ["QUALITATIVE_STARTED", "QUALITATIVE_COMPLETED"]


def test_unselected_sample_reference_is_rejected(tmp_path: Path):
    _dataset(tmp_path)
    request = _request(tmp_path)
    context = build_qualitative_context(request)
    output = _valid_output(request, context)
    output["payload"]["observations"][0]["sample_refs"] = ["invented"]
    result = QualitativeResult.model_validate(output)

    with pytest.raises(QualitativeExecutionError, match="uninspected"):
        validate_qualitative_result(result, request, context)


def test_observation_group_must_match_referenced_samples(tmp_path: Path):
    _dataset(tmp_path)
    request = _request(tmp_path)
    context = build_qualitative_context(request)
    output = _valid_output(request, context)
    output["payload"]["observations"][0]["group"] = "label=invented"
    result = QualitativeResult.model_validate(output)

    with pytest.raises(QualitativeExecutionError, match="group does not match"):
        validate_qualitative_result(result, request, context)


def test_unknown_interpretation_dependency_is_rejected(tmp_path: Path):
    _dataset(tmp_path)
    request = _request(tmp_path)
    context = build_qualitative_context(request)
    output = _valid_output(request, context)
    output["payload"]["interpretations"][0]["supporting_observation_ids"] = [
        "QO-C001-002"
    ]
    result = QualitativeResult.model_validate(output)

    with pytest.raises(QualitativeExecutionError, match="unknown observation"):
        validate_qualitative_result(result, request, context)


def test_context_limitations_must_be_preserved(tmp_path: Path):
    _dataset(tmp_path)
    request = _request(tmp_path)
    context = build_qualitative_context(request)
    output = _valid_output(request, context)
    output["payload"]["limitations"] = []
    result = QualitativeResult.model_validate(output)

    with pytest.raises(QualitativeExecutionError, match="omitted"):
        validate_qualitative_result(result, request, context)


def test_invalid_raw_output_is_not_persisted_and_failure_is_logged(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _dataset(dataset_root)
    request = _request(dataset_root)
    runs_root = tmp_path / "runs"

    with pytest.raises(QualitativeExecutionError, match="invalid QualitativeAgentDraft"):
        execute_qualitative(
            request,
            MockQualitativeClient({"invalid": "output"}),
            runs_root=runs_root,
        )

    stage_dir = runs_root / request.run_id / "qualitative"
    assert not (stage_dir / "raw_output.json").exists()
    assert not (stage_dir / "result.json").exists()
    assert "AGENT_FAILED" in (
        runs_root / request.run_id / "events.jsonl"
    ).read_text("utf-8")


def test_non_object_raw_output_is_not_persisted_and_failure_is_logged(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _dataset(dataset_root)
    request = _request(dataset_root)
    runs_root = tmp_path / "runs"

    with pytest.raises(QualitativeExecutionError, match="non-object"):
        execute_qualitative(
            request,
            MockQualitativeClient(["invalid", "output"]),
            runs_root=runs_root,
        )

    assert not (
        runs_root / request.run_id / "qualitative" / "raw_output.json"
    ).exists()
    assert "AGENT_FAILED" in (
        runs_root / request.run_id / "events.jsonl"
    ).read_text("utf-8")
