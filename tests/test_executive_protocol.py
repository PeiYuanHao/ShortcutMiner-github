from copy import deepcopy

import pytest
from pydantic import ValidationError

from shortcutminer.protocol.executive import (
    ExecutiveRequest,
    RoutingPlan,
)


def _candidate() -> dict:
    return {
        "candidate_id": "C001",
        "cue": {"family": "metadata", "name": "site"},
        "description": "Site identity may provide an unintended cue.",
        "observation": "The metadata contains a site field.",
        "hypothesis": "Site identity may be associated with the label.",
        "evidence_sources": ["metadata"],
        "priority": "high",
        "status": "unverified",
    }


def _request_data() -> dict:
    return {
        "protocol_version": "1.0",
        "run_id": "executive_test_run",
        "message_id": "executive:request:001",
        "source": "orchestrator",
        "target": "executive",
        "message_type": "executive_request",
        "in_reply_to": "identify:report:001",
        "candidate_report": {
            "protocol_version": "1.0",
            "run_id": "executive_test_run",
            "message_id": "identify:report:001",
            "source": "identify",
            "target": "executive",
            "message_type": "candidate_report",
            "in_reply_to": "identify:request:001",
            "artifact_references": [],
            "dataset_summary": {
                "dataset": "test_dataset",
                "samples_inspected": 0,
                "metadata_used": True,
            },
            "candidates": [_candidate()],
            "limitations": ["Samples were not inspected."],
        },
        "routing_context": {
            "protocol_version": "1.0",
            "run_id": "executive_test_run",
            "dataset": {
                "name": "test_dataset",
                "root": "/datasets/test",
                "metadata_file": "metadata.csv",
            },
            "task": {
                "type": "classification",
                "label_column": "label",
            },
            "audit_options": {
                "validation_depth": "standard",
                "max_tasks": 2,
                "max_samples_per_task": 20,
                "allowed_targets": ["quantitative", "qualitative"],
                "requested_outputs": ["metrics", "observations"],
                "runtime_constraints": ["no_retraining"],
            },
            "capabilities": {
                "available_targets": ["quantitative", "qualitative"],
                "available_metadata_fields": ["site", "label"],
                "available_splits": ["train"],
                "quantitative_checks": ["association_test"],
                "qualitative_checks": ["sample_inspection"],
                "model_capabilities": ["predictions"],
                "sample_access": True,
                "available_output_types": [
                    "metrics",
                    "tables",
                    "observations",
                    "representative_examples",
                ],
            },
            "artifact_references": [],
            "limitations": ["Samples were not inspected."],
        },
    }


def _plan_data() -> dict:
    candidate = _candidate()
    return {
        "protocol_version": "1.0",
        "run_id": "executive_test_run",
        "message_id": "executive:plan:001",
        "source": "executive",
        "target": "orchestrator",
        "message_type": "routing_plan",
        "in_reply_to": "executive:request:001",
        "artifact_references": [],
        "decisions": [
            {
                "candidate": candidate,
                "route": "both",
                "routing_reason": "Complementary evidence is useful.",
                "task_ids": ["TASK-C001-QT", "TASK-C001-QL"],
                "priority": "high",
                "status": "planned",
                "missing_requirements": [],
                "limitations": [],
            }
        ],
        "tasks": [
            {
                "protocol_version": "1.0",
                "run_id": "executive_test_run",
                "message_id": "executive:task:C001:QT",
                "source": "executive",
                "target": "quantitative",
                "message_type": "analysis_task",
                "in_reply_to": "executive:plan:001",
                "task_id": "TASK-C001-QT",
                "candidate_id": "C001",
                "hypothesis": candidate["hypothesis"],
                "objective": "Measure the site-label association.",
                "inputs": {
                    "dataset": "test_dataset",
                    "dataset_reference": "/datasets/test",
                    "label_column": "label",
                    "split": "train",
                    "metadata_fields": ["site", "label"],
                    "artifact_references": [],
                    "sample_limit": None,
                    "requires_sample_access": False,
                    "required_model_capabilities": [],
                },
                "requested_checks": ["association_test"],
                "expected_outputs": ["metrics", "tables"],
                "priority": "high",
                "dependencies": [],
                "status": "planned",
            },
            {
                "protocol_version": "1.0",
                "run_id": "executive_test_run",
                "message_id": "executive:task:C001:QL",
                "source": "executive",
                "target": "qualitative",
                "message_type": "analysis_task",
                "in_reply_to": "executive:plan:001",
                "task_id": "TASK-C001-QL",
                "candidate_id": "C001",
                "hypothesis": candidate["hypothesis"],
                "objective": "Inspect site-related visual context.",
                "inputs": {
                    "dataset": "test_dataset",
                    "dataset_reference": "/datasets/test",
                    "label_column": "label",
                    "split": "train",
                    "metadata_fields": ["site"],
                    "artifact_references": [],
                    "sample_limit": 20,
                    "requires_sample_access": True,
                    "required_model_capabilities": [],
                },
                "requested_checks": ["sample_inspection"],
                "expected_outputs": [
                    "observations",
                    "representative_examples",
                ],
                "priority": "high",
                "dependencies": [],
                "status": "planned",
            },
        ],
        "limitations": ["Samples were not inspected."],
    }


def test_executive_request_schema_accepts_complete_request():
    request = ExecutiveRequest.model_validate(_request_data())
    assert request.candidate_report.candidates[0].candidate_id == "C001"
    assert request.routing_context.audit_options.max_tasks == 2


def test_routing_plan_schema_accepts_complete_plan():
    plan = RoutingPlan.model_validate(_plan_data())
    assert plan.decisions[0].route == "both"
    assert [task.target for task in plan.tasks] == [
        "quantitative",
        "qualitative",
    ]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("source",), "identify"),
        (("target",), "quantitative"),
        (("message_type",), "routing_request"),
        (("protocol_version",), "2.0"),
    ],
)
def test_executive_request_rejects_invalid_envelope(path, value):
    data = _request_data()
    data[path[0]] = value
    with pytest.raises(ValidationError):
        ExecutiveRequest.model_validate(data)


def test_unknown_fields_are_rejected_recursively():
    data = _request_data()
    data["routing_context"]["capabilities"]["unknown"] = True
    with pytest.raises(ValidationError):
        ExecutiveRequest.model_validate(data)


def test_strict_types_are_not_coerced():
    data = _request_data()
    data["routing_context"]["audit_options"]["max_tasks"] = "2"
    with pytest.raises(ValidationError):
        ExecutiveRequest.model_validate(data)


def test_duplicate_capabilities_are_rejected():
    data = _request_data()
    data["routing_context"]["capabilities"]["available_targets"] = [
        "quantitative",
        "quantitative",
    ]
    with pytest.raises(ValidationError):
        ExecutiveRequest.model_validate(data)


@pytest.mark.parametrize(
    "task_id",
    ["TASK-SC-001-QT", "TASK-C000-QT", "TASK-C001-XX", "C001-QT"],
)
def test_task_id_format_is_strict(task_id: str):
    data = _plan_data()
    data["tasks"][0]["task_id"] = task_id
    with pytest.raises(ValidationError):
        RoutingPlan.model_validate(data)


def test_artifact_references_must_be_safe_relative_paths():
    data = deepcopy(_plan_data())
    data["artifact_references"] = ["../another-run/output.json"]
    with pytest.raises(ValidationError):
        RoutingPlan.model_validate(data)
