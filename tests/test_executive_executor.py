import json
from pathlib import Path

import pytest

from shortcutminer.experiment.executive_executor import (
    ExecutiveExecutionError,
    execute_executive,
    validate_executive_request,
    validate_routing_plan,
)
from shortcutminer.protocol.executive import (
    ExecutiveAuditOptions,
    ExecutiveCapabilities,
    ExecutiveRequest,
    ExecutiveRoutingContext,
    RoutingPlan,
)
from shortcutminer.protocol.identify import (
    Candidate,
    CandidateReport,
    Cue,
    DatasetInspectionSummary,
    DatasetSpec,
    TaskSpec,
)


class MockExecutiveClient:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def invoke_agent(self, agent_name: str, payload: dict):
        self.calls.append({"agent_name": agent_name, "payload": payload})
        return self.output


def _candidate() -> Candidate:
    return Candidate(
        candidate_id="C001",
        cue=Cue(family="metadata", name="site"),
        description="Site identity may provide an unintended cue.",
        observation="The metadata contains a site field.",
        hypothesis="Site identity may be associated with the label.",
        evidence_sources=["metadata"],
        priority="high",
        status="unverified",
    )


def _request(tmp_path: Path) -> ExecutiveRequest:
    report = CandidateReport(
        protocol_version="1.0",
        run_id="executive_test_run",
        message_id="identify:report:001",
        source="identify",
        target="executive",
        message_type="candidate_report",
        in_reply_to="identify:request:001",
        artifact_references=[],
        dataset_summary=DatasetInspectionSummary(
            dataset="test_dataset",
            samples_inspected=0,
            metadata_used=True,
        ),
        candidates=[_candidate()],
        limitations=["Samples were not inspected."],
    )
    context = ExecutiveRoutingContext(
        protocol_version="1.0",
        run_id="executive_test_run",
        dataset=DatasetSpec(
            name="test_dataset",
            root=str(tmp_path),
            metadata_file="metadata.csv",
        ),
        task=TaskSpec(type="classification", label_column="label"),
        audit_options=ExecutiveAuditOptions(
            validation_depth="standard",
            max_tasks=2,
            max_samples_per_task=20,
            allowed_targets=["quantitative", "qualitative"],
            requested_outputs=["metrics", "observations"],
            runtime_constraints=["no_retraining"],
        ),
        capabilities=ExecutiveCapabilities(
            available_targets=["quantitative", "qualitative"],
            available_metadata_fields=["site", "label"],
            available_splits=["train"],
            quantitative_checks=["association_test"],
            qualitative_checks=["sample_inspection"],
            model_capabilities=["predictions"],
            sample_access=True,
            available_output_types=[
                "metrics",
                "tables",
                "observations",
                "representative_examples",
            ],
        ),
        artifact_references=[],
        limitations=["Samples were not inspected."],
    )
    return ExecutiveRequest(
        protocol_version="1.0",
        run_id="executive_test_run",
        message_id="executive:request:001",
        source="orchestrator",
        target="executive",
        message_type="executive_request",
        in_reply_to=report.message_id,
        candidate_report=report,
        routing_context=context,
    )


def _valid_output(request: ExecutiveRequest) -> dict:
    candidate = request.candidate_report.candidates[0].model_dump(mode="json")
    root = request.routing_context.dataset.root
    hypothesis = candidate["hypothesis"]
    return {
        "protocol_version": "1.0",
        "run_id": request.run_id,
        "message_id": "executive:plan:001",
        "source": "executive",
        "target": "orchestrator",
        "message_type": "routing_plan",
        "in_reply_to": request.message_id,
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
                "run_id": request.run_id,
                "message_id": "executive:task:C001:QT",
                "source": "executive",
                "target": "quantitative",
                "message_type": "analysis_task",
                "in_reply_to": "executive:plan:001",
                "task_id": "TASK-C001-QT",
                "candidate_id": "C001",
                "hypothesis": hypothesis,
                "objective": "Measure the site-label association.",
                "inputs": {
                    "dataset": "test_dataset",
                    "dataset_reference": root,
                    "label_column": "label",
                    "split": "train",
                    "metadata_fields": ["site", "label"],
                    "artifact_references": [],
                    "sample_limit": None,
                    "requires_sample_access": False,
                    "requires_external_services": False,
                    "runtime_constraints": ["no_retraining"],
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
                "run_id": request.run_id,
                "message_id": "executive:task:C001:QL",
                "source": "executive",
                "target": "qualitative",
                "message_type": "analysis_task",
                "in_reply_to": "executive:plan:001",
                "task_id": "TASK-C001-QL",
                "candidate_id": "C001",
                "hypothesis": hypothesis,
                "objective": "Inspect site-related visual context.",
                "inputs": {
                    "dataset": "test_dataset",
                    "dataset_reference": root,
                    "label_column": "label",
                    "split": "train",
                    "metadata_fields": ["site"],
                    "artifact_references": [],
                    "sample_limit": 20,
                    "requires_sample_access": True,
                    "requires_external_services": False,
                    "runtime_constraints": ["no_retraining"],
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


def test_execute_executive_returns_valid_plan(tmp_path: Path):
    request = _request(tmp_path)
    client = MockExecutiveClient(_valid_output(request))

    plan = execute_executive(request=request, client=client)

    assert plan.decisions[0].route == "both"
    assert [task.task_id for task in plan.tasks] == [
        "TASK-C001-QT",
        "TASK-C001-QL",
    ]
    assert client.calls[0]["agent_name"] == "executive"
    assert client.calls[0]["payload"]["request"]["candidate_report"]


def test_request_must_correlate_to_candidate_report(tmp_path: Path):
    request = _request(tmp_path)
    request.in_reply_to = "another-message"
    with pytest.raises(ExecutiveExecutionError, match="in_reply_to"):
        validate_executive_request(request)


def test_plan_must_preserve_candidate(tmp_path: Path):
    request = _request(tmp_path)
    output = _valid_output(request)
    output["decisions"][0]["candidate"]["description"] = "Changed"
    plan = RoutingPlan.model_validate(output)
    with pytest.raises(ExecutiveExecutionError, match="preserve"):
        validate_routing_plan(plan, request)


def test_route_and_task_ids_must_agree(tmp_path: Path):
    request = _request(tmp_path)
    output = _valid_output(request)
    output["decisions"][0]["route"] = "quantitative"
    plan = RoutingPlan.model_validate(output)
    with pytest.raises(ExecutiveExecutionError, match="task IDs"):
        validate_routing_plan(plan, request)


def test_unavailable_check_is_rejected(tmp_path: Path):
    request = _request(tmp_path)
    output = _valid_output(request)
    output["tasks"][0]["requested_checks"] = ["mutual_information"]
    plan = RoutingPlan.model_validate(output)
    with pytest.raises(ExecutiveExecutionError, match="unavailable check"):
        validate_routing_plan(plan, request)


def test_unavailable_target_is_rejected(tmp_path: Path):
    request = _request(tmp_path)
    request.routing_context.capabilities.available_targets = ["quantitative"]
    plan = RoutingPlan.model_validate(_valid_output(request))

    with pytest.raises(ExecutiveExecutionError, match="unavailable target"):
        validate_routing_plan(plan, request)


def test_sample_budget_is_enforced(tmp_path: Path):
    request = _request(tmp_path)
    output = _valid_output(request)
    output["tasks"][1]["inputs"]["sample_limit"] = 21
    plan = RoutingPlan.model_validate(output)
    with pytest.raises(ExecutiveExecutionError, match="sample budget"):
        validate_routing_plan(plan, request)


def test_sample_access_requires_explicit_limit(tmp_path: Path):
    request = _request(tmp_path)
    output = _valid_output(request)
    output["tasks"][1]["inputs"]["sample_limit"] = None
    plan = RoutingPlan.model_validate(output)

    with pytest.raises(ExecutiveExecutionError, match="without a sample_limit"):
        validate_routing_plan(plan, request)


def test_runtime_constraints_must_be_preserved(tmp_path: Path):
    request = _request(tmp_path)
    output = _valid_output(request)
    output["tasks"][0]["inputs"]["runtime_constraints"] = []
    plan = RoutingPlan.model_validate(output)

    with pytest.raises(ExecutiveExecutionError, match="runtime constraints"):
        validate_routing_plan(plan, request)


def test_no_retraining_constraint_is_enforced(tmp_path: Path):
    request = _request(tmp_path)
    request.routing_context.capabilities.model_capabilities.append(
        "retraining"
    )
    output = _valid_output(request)
    output["tasks"][0]["inputs"]["required_model_capabilities"] = [
        "retraining"
    ]
    plan = RoutingPlan.model_validate(output)

    with pytest.raises(ExecutiveExecutionError, match="no_retraining"):
        validate_routing_plan(plan, request)


def test_no_external_services_constraint_is_enforced(tmp_path: Path):
    request = _request(tmp_path)
    request.routing_context.audit_options.runtime_constraints = [
        "no_external_services"
    ]
    request.routing_context.capabilities.external_services = True
    output = _valid_output(request)
    for task in output["tasks"]:
        task["inputs"]["runtime_constraints"] = ["no_external_services"]
    output["tasks"][0]["inputs"]["requires_external_services"] = True
    plan = RoutingPlan.model_validate(output)

    with pytest.raises(
        ExecutiveExecutionError,
        match="no_external_services",
    ):
        validate_routing_plan(plan, request)


def test_unavailable_external_services_are_rejected(tmp_path: Path):
    request = _request(tmp_path)
    request.routing_context.audit_options.runtime_constraints = []
    output = _valid_output(request)
    for task in output["tasks"]:
        task["inputs"]["runtime_constraints"] = []
    output["tasks"][0]["inputs"]["requires_external_services"] = True
    plan = RoutingPlan.model_validate(output)

    with pytest.raises(
        ExecutiveExecutionError,
        match="unavailable external services",
    ):
        validate_routing_plan(plan, request)


def test_user_requested_outputs_must_be_covered(tmp_path: Path):
    request = _request(tmp_path)
    request.routing_context.audit_options.requested_outputs.append("plots")
    request.routing_context.capabilities.available_output_types.append("plots")
    plan = RoutingPlan.model_validate(_valid_output(request))

    with pytest.raises(ExecutiveExecutionError, match="requested outputs"):
        validate_routing_plan(plan, request)


def test_task_dependencies_must_be_acyclic(tmp_path: Path):
    request = _request(tmp_path)
    output = _valid_output(request)
    output["tasks"][0]["dependencies"] = ["TASK-C001-QL"]
    output["tasks"][1]["dependencies"] = ["TASK-C001-QT"]
    plan = RoutingPlan.model_validate(output)

    with pytest.raises(ExecutiveExecutionError, match="cyclic"):
        validate_routing_plan(plan, request)


def test_task_cannot_invent_artifact_reference(tmp_path: Path):
    request = _request(tmp_path)
    output = _valid_output(request)
    output["tasks"][0]["inputs"]["artifact_references"] = [
        "identify/output.json"
    ]
    plan = RoutingPlan.model_validate(output)

    with pytest.raises(ExecutiveExecutionError, match="invented"):
        validate_routing_plan(plan, request)


def test_unresolved_candidate_is_preserved_without_tasks(tmp_path: Path):
    request = _request(tmp_path)
    candidate = request.candidate_report.candidates[0].model_dump(mode="json")
    output = _valid_output(request)
    output["decisions"] = [
        {
            "candidate": candidate,
            "route": "unresolved",
            "routing_reason": "The required tool is unavailable.",
            "task_ids": [],
            "priority": "high",
            "status": "unresolved",
            "missing_requirements": ["supported validation tool"],
            "limitations": [],
        }
    ]
    output["tasks"] = []

    plan = RoutingPlan.model_validate(output)
    validate_routing_plan(plan, request)


def test_artifacts_and_events_are_persisted(tmp_path: Path):
    request = _request(tmp_path / "dataset")
    runs_root = tmp_path / "runs"
    output = _valid_output(request)

    execute_executive(
        request=request,
        client=MockExecutiveClient(output),
        runs_root=runs_root,
    )

    executive_dir = runs_root / request.run_id / "executive"
    assert json.loads((executive_dir / "input.json").read_text("utf-8"))[
        "request"
    ]["run_id"] == request.run_id
    assert json.loads(
        (executive_dir / "raw_output.json").read_text("utf-8")
    ) == output
    assert json.loads((executive_dir / "routing.json").read_text("utf-8"))[
        "message_type"
    ] == "routing_plan"

    events = [
        json.loads(line)
        for line in (
            runs_root / request.run_id / "events.jsonl"
        ).read_text("utf-8").splitlines()
    ]
    assert [item["event"] for item in events] == [
        "EXECUTIVE_STARTED",
        "EXECUTIVE_COMPLETED",
    ]


def test_invalid_raw_output_is_retained_and_failure_is_logged(tmp_path: Path):
    request = _request(tmp_path / "dataset")
    runs_root = tmp_path / "runs"

    with pytest.raises(ExecutiveExecutionError, match="invalid RoutingPlan"):
        execute_executive(
            request=request,
            client=MockExecutiveClient({"invalid": "output"}),
            runs_root=runs_root,
        )

    executive_dir = runs_root / request.run_id / "executive"
    assert (executive_dir / "raw_output.json").exists()
    assert not (executive_dir / "routing.json").exists()
    events = (
        runs_root / request.run_id / "events.jsonl"
    ).read_text("utf-8")
    assert "AGENT_FAILED" in events
