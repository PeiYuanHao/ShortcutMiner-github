from copy import deepcopy

import pytest
from pydantic import ValidationError

from shortcutminer.protocol.qualitative import (
    QualitativePreparationOptions,
    QualitativeRequest,
    QualitativeResult,
)


def _candidate() -> dict:
    return {
        "candidate_id": "C001",
        "cue": {"family": "appearance", "name": "dark_border"},
        "description": "A border may differ by label.",
        "observation": "Upstream evidence suggests a border candidate.",
        "hypothesis": "A dark border may be associated with the label.",
        "evidence_sources": ["metadata"],
        "priority": "high",
        "status": "unverified",
    }


def _task() -> dict:
    candidate = _candidate()
    return {
        "protocol_version": "1.0",
        "run_id": "qualitative_test_run",
        "message_id": "executive:task:C001:QL",
        "source": "executive",
        "target": "qualitative",
        "message_type": "analysis_task",
        "in_reply_to": "executive:plan:001",
        "task_id": "TASK-C001-QL",
        "candidate_id": "C001",
        "hypothesis": candidate["hypothesis"],
        "objective": "Inspect border appearance across labels.",
        "inputs": {
            "dataset": "test_dataset",
            "dataset_reference": "/datasets/test",
            "label_column": "label",
            "split": "train",
            "metadata_fields": ["sample_id", "filename", "label"],
            "artifact_references": [],
            "sample_limit": 4,
            "requires_sample_access": True,
            "requires_external_services": False,
            "runtime_constraints": ["no_retraining"],
            "required_model_capabilities": [],
        },
        "requested_checks": ["sample_inspection", "counterexample_inspection"],
        "expected_outputs": ["observations", "representative_examples"],
        "priority": "high",
        "dependencies": [],
        "status": "planned",
    }


def _request_data() -> dict:
    return {
        "protocol_version": "1.0",
        "run_id": "qualitative_test_run",
        "message_id": "qualitative:request:C001",
        "source": "orchestrator",
        "target": "qualitative",
        "message_type": "qualitative_analysis_request",
        "in_reply_to": "executive:task:C001:QL",
        "candidate": _candidate(),
        "task": _task(),
        "metadata_file": "metadata.csv",
        "preparation": {
            "sampling_strategy": "group_balanced",
            "seed": 42,
            "sample_id_column": "sample_id",
            "sample_path_column": "filename",
            "split_column": "split",
            "group_columns": ["label"],
            "max_evidence_records": 4,
            "max_metadata_rows": 100,
        },
        "limitations": [],
    }


def _result_data() -> dict:
    candidate = _candidate()
    return {
        "protocol_version": "1.0",
        "run_id": "qualitative_test_run",
        "message_id": "qualitative:result:C001",
        "source": "qualitative",
        "target": "summarize",
        "message_type": "qualitative_analysis_result",
        "in_reply_to": "qualitative:request:C001",
        "payload": {
            "task_id": "TASK-C001-QL",
            "candidate_id": "C001",
            "candidate_name": "dark_border",
            "hypothesis": candidate["hypothesis"],
            "evidence_status": "partially_supported",
            "evidence_summary": "A border was visible, with exceptions.",
            "sampling": {
                "strategy": "group_balanced",
                "seed": 42,
                "groups": {"label=A": 1},
                "sample_count": 1,
                "sample_ids": ["S1"],
            },
            "observations": [
                {
                    "observation_id": "QO-C001-001",
                    "group": "label=A",
                    "sample_refs": ["S1"],
                    "observation": "A dark border is visible.",
                    "inspection_quality": "sufficient",
                }
            ],
            "interpretations": [
                {
                    "interpretation": "The border may reflect preprocessing.",
                    "supporting_observation_ids": ["QO-C001-001"],
                    "confidence": "low",
                }
            ],
            "counterevidence": [],
            "metadata_evidence": [],
            "failures": [],
            "limitations": [],
            "artifact_references": [],
            "recommended_followup": ["Inspect more comparison samples."],
        },
    }


def test_qualitative_request_accepts_complete_contract():
    request = QualitativeRequest.model_validate(_request_data())
    assert request.task.task_id == "TASK-C001-QL"
    assert request.preparation.seed == 42


def test_qualitative_result_accepts_separated_evidence_contract():
    result = QualitativeResult.model_validate(_result_data())
    assert result.payload.task_id == "TASK-C001-QL"
    assert result.payload.observations[0].observation_id == "QO-C001-001"
    assert result.payload.interpretations[0].confidence == "low"


def test_unknown_fields_are_rejected():
    data = _result_data()
    data["payload"]["unsupported"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        QualitativeResult.model_validate(data)


def test_observation_ids_are_candidate_scoped():
    data = deepcopy(_result_data())
    data["payload"]["observations"][0]["observation_id"] = "QO-001"
    with pytest.raises(ValidationError, match="observation_id"):
        QualitativeResult.model_validate(data)


def test_evidence_status_is_canonical():
    data = deepcopy(_result_data())
    data["payload"]["evidence_status"] = "likely"
    with pytest.raises(ValidationError, match="evidence_status"):
        QualitativeResult.model_validate(data)


def test_random_sampling_does_not_require_group_columns():
    options = QualitativePreparationOptions(seed=42)
    assert options.sampling_strategy == "random"
    assert options.group_columns == []


def test_group_balanced_requires_explicit_group_columns():
    with pytest.raises(ValidationError, match="requires at least one explicit"):
        QualitativePreparationOptions(seed=42, sampling_strategy="group_balanced")
