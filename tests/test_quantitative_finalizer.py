import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from shortcutminer.experiment.quantitative_context import build_quantitative_request
from shortcutminer.evaluation.evidence_ledger import _quantitative_records
from shortcutminer.experiment.quantitative_executor import (
    QuantitativeDraftInvariantError,
    QuantitativeExecutionError,
    execute_quantitative,
    finalize_quantitative_result,
)
from shortcutminer.protocol.executive import RoutingPlan
from shortcutminer.protocol.quantitative import (
    QuantitativeAgentDraft,
    QuantitativeResultPayload,
)
from shortcutminer.tooling.protocol import ToolErrorInfo, ToolResult
from test_executive_executor import _request as _executive_request
from test_executive_executor import _valid_output as _executive_output


def _request(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "metadata.csv").write_text(
        "site,label\nA,0\nA,0\nB,1\nB,1\n",
        encoding="utf-8",
    )
    executive_request = _executive_request(tmp_path)
    plan = RoutingPlan.model_validate(_executive_output(executive_request))
    return build_quantitative_request(
        executive_request=executive_request,
        routing_plan=plan,
        task_id="TASK-C001-QT",
        message_id="quantitative:request:C001",
        limitations=["Preserve this upstream limitation."],
    )


def _draft(**updates):
    values = {
        "status": "completed",
        "analysis_type": "association_test",
        "evidence_direction": "inconclusive",
        "evidence_summary": "The measured association remains inconclusive.",
        "additional_limitations": ["Interpretation is observational."],
        "recommended_followup": [],
    }
    values.update(updates)
    values.pop("evidence_level", None)
    values.pop("evidence_tool_call_ids", None)
    return QuantitativeAgentDraft.model_validate(values)


def _tool_result(
    *,
    call_id="experiment-001",
    task_id="TASK-C001-QT",
    status="success",
    metrics=None,
    artifacts=None,
    experiment_type="statistical_check",
    planning_check="association_test",
):
    common = {
        "protocol_version": "1.0",
        "tool_call_id": call_id,
        "tool_name": "run_experiment",
        "agent_name": "quantitative",
        "run_id": "executive_test_run",
        "candidate_id": "C001",
        "task_id": task_id,
        "plan_step_id": f"STEP-{call_id}",
        "planning_check": planning_check,
        "experiment_type": experiment_type,
        "artifacts": artifacts or [f"experiments/{call_id}/result.json"],
        "metadata": {},
        "started_at": "2026-01-01T00:00:00Z",
        "completed_at": "2026-01-01T00:00:01Z",
        "duration_ms": 1.0,
    }
    if status == "success":
        return ToolResult(
            **common,
            status="success",
            output={
                "outcome": "completed",
                "metrics": metrics or {"effect_size": 0.25, "sample_count": 48},
                "summary": "Typed backend result.",
                "artifact_references": [],
                "metadata": {"limitations": ["Backend limitation."]},
            },
            error=None,
        )
    return ToolResult(
        **common,
        status="failed",
        output=None,
        error=ToolErrorInfo(
            error_type="ToolExecutionError", message="bounded failure"
        ),
    )


def _hash(result) -> str:
    payload = json.dumps(
        result.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def test_finalizer_derives_identity_metrics_artifacts_and_limitations(tmp_path: Path):
    request = _request(tmp_path)
    tool = _tool_result()

    result = finalize_quantitative_result(_draft(), request, tool_results=(tool,))

    assert result.protocol_version == request.protocol_version
    assert result.run_id == request.run_id
    assert result.in_reply_to == request.message_id
    assert result.payload.task_id == request.task.task_id
    assert result.payload.candidate_id == request.candidate.candidate_id
    assert result.payload.hypothesis == request.candidate.hypothesis
    assert result.payload.metrics == {"effect_size": 0.25, "sample_count": 48}
    assert result.payload.artifact_references == tool.artifacts
    assert result.payload.limitations == [
        *request.limitations,
        "Interpretation is observational.",
        "Backend limitation.",
    ]


@pytest.mark.parametrize("extra", [
    {"metrics": {}}, {"artifact_references": []},
    {"evidence_level": "Q4"}, {"evidence_tool_call_ids": ["invented"]},
])
def test_draft_rejects_python_owned_fields(extra):
    values = _draft().model_dump(mode="json") | extra
    with pytest.raises(ValidationError, match="extra_forbidden"):
        QuantitativeAgentDraft.model_validate(values)


def test_unknown_failed_and_other_task_evidence_are_rejected(tmp_path: Path):
    request = _request(tmp_path)
    with pytest.raises(QuantitativeDraftInvariantError, match="requires successful"):
        finalize_quantitative_result(_draft(), request)
    with pytest.raises(QuantitativeDraftInvariantError, match="requires successful"):
        finalize_quantitative_result(
            _draft(), request, tool_results=(_tool_result(status="failed"),)
        )
    with pytest.raises(QuantitativeDraftInvariantError, match="another task"):
        finalize_quantitative_result(
            _draft(),
            request,
            tool_results=(_tool_result(task_id="TASK-C001-QL"),),
        )


def test_completed_requires_selected_successful_result(tmp_path: Path):
    request = _request(tmp_path)
    draft = _draft(evidence_tool_call_ids=[])
    with pytest.raises(QuantitativeDraftInvariantError, match="requires successful"):
        finalize_quantitative_result(draft, request)


def test_blocked_draft_finalizes_without_metrics(tmp_path: Path):
    request = _request(tmp_path)
    draft = _draft(
        status="blocked",
        evidence_level="Q0",
        evidence_direction="not_tested",
        evidence_tool_call_ids=[],
        additional_limitations=["Required measurement is unavailable."],
    )
    result = finalize_quantitative_result(draft, request)
    assert result.payload.status == "blocked"
    assert result.payload.metrics == {}


def test_canonical_blocked_payload_rejects_metrics():
    with pytest.raises(ValidationError, match="cannot report metrics"):
        QuantitativeResultPayload(
            task_id="TASK-C001-QT",
            candidate_id="C001",
            hypothesis="A measured hypothesis.",
            status="blocked",
            analysis_type="association_test",
            metrics={"forbidden": 1},
            evidence_level="Q0",
            evidence_direction="not_tested",
            evidence_summary="No valid test completed.",
            limitations=["Unavailable."],
            artifact_references=[],
            recommended_followup=[],
        )


def test_metric_collisions_are_namespaced_and_artifacts_are_stable(tmp_path: Path):
    request = _request(tmp_path)
    first = _tool_result(
        metrics={"effect_size": 0.25, "first_only": True},
        artifacts=["experiments/shared.json", "experiments/first.json"],
    )
    second = _tool_result(
        call_id="experiment-002",
        metrics={"effect_size": 0.5, "second_only": None},
        artifacts=["experiments/shared.json", "experiments/second.json"],
    )
    draft = _draft(evidence_tool_call_ids=["experiment-001", "experiment-002"])

    result = finalize_quantitative_result(draft, request, tool_results=(first, second))

    assert result.payload.metrics == {
        "experiment-001.effect_size": 0.25,
        "first_only": True,
        "experiment-002.effect_size": 0.5,
        "second_only": None,
    }
    assert result.payload.artifact_references == [
        "experiments/shared.json",
        "experiments/first.json",
        "experiments/second.json",
    ]


def test_training_and_evaluation_remain_separate_canonical_measurements(tmp_path: Path):
    request = _request(tmp_path)
    training = _tool_result(
        call_id="training-001", experiment_type="task_model_training",
        planning_check="task_model_training", metrics={"train_n": 30},
    )
    evaluation = _tool_result(
        call_id="evaluation-001", experiment_type="task_model_evaluation",
        planning_check="task_model_evaluation", metrics={"test_n": 10},
    )
    result = finalize_quantitative_result(
        _draft(evidence_direction="supports"), request,
        tool_results=(training, evaluation),
    )
    assert len(result.payload.measurements) == 2
    assert [item.plan_step_id for item in result.payload.measurements] == [
        "STEP-training-001", "STEP-evaluation-001",
    ]
    assert [item.tool_call_id for item in result.payload.measurements] == [
        "training-001", "evaluation-001",
    ]
    assert result.payload.evidence_level == "Q3"
    records = _quantitative_records(result, request.task, None)
    assert len(records) == 2
    assert len({item.evidence_id for item in records}) == 2
    assert [item.provenance.plan_step_id for item in records] == [
        "STEP-training-001", "STEP-evaluation-001",
    ]
    assert records[0].direction == "inconclusive"
    assert records[1].direction == "supports"


def test_missing_and_duplicate_measurement_provenance_are_rejected(tmp_path: Path):
    request = _request(tmp_path)
    tool = _tool_result()
    with pytest.raises(QuantitativeDraftInvariantError, match="plan provenance"):
        finalize_quantitative_result(
            _draft(), request,
            tool_results=(tool.model_copy(update={"plan_step_id": None}),),
        )
    with pytest.raises(QuantitativeDraftInvariantError, match="duplicate quantitative measurement"):
        finalize_quantitative_result(_draft(), request, tool_results=(tool, tool))


def test_backend_experiment_type_cannot_override_canonical_plan(tmp_path: Path):
    request = _request(tmp_path)
    tool = _tool_result()
    output = dict(tool.output)
    output["metadata"] = {"experiment_type": "task_model_training"}
    with pytest.raises(QuantitativeDraftInvariantError, match="backend experiment type"):
        finalize_quantitative_result(
            _draft(), request, tool_results=(tool.model_copy(update={"output": output}),),
        )


def test_inconclusive_backend_cannot_be_promoted_to_support(tmp_path: Path):
    request = _request(tmp_path)
    tool = _tool_result()
    output = dict(tool.output)
    output["outcome"] = "inconclusive"
    tool = tool.model_copy(update={"output": output})
    with pytest.raises(QuantitativeDraftInvariantError, match="inconclusive backend"):
        finalize_quantitative_result(
            _draft(evidence_direction="supports"), request, tool_results=(tool,),
        )


def test_private_truth_cannot_change_final_result(tmp_path: Path):
    request = _request(tmp_path)
    tool = _tool_result()
    draft = _draft()
    private_truth_variants = [
        None,
        {"hidden_truth": "candidate supported"},
        {"hidden_truth": "candidate contradicted"},
    ]

    hashes = {
        _hash(finalize_quantitative_result(draft, request, tool_results=(tool,)))
        for _private_truth in private_truth_variants
    }

    assert len(hashes) == 1


class _FixedClient:
    def __init__(self, output):
        self.output = output

    def invoke_agent(self, agent_name, payload, *, system_prompt=None):
        assert agent_name == "quantitative"
        return self.output


def test_executor_persists_validated_draft_not_raw_output(tmp_path: Path):
    request = _request(tmp_path)
    draft = _draft(
        status="blocked",
        evidence_level="Q0",
        evidence_direction="not_tested",
        evidence_tool_call_ids=[],
        additional_limitations=["No tool runtime was supplied."],
    )

    execute_quantitative(
        request,
        _FixedClient(draft.model_dump(mode="json")),
        runs_root=tmp_path / "runs",
    )

    scope = (
        tmp_path / "runs" / request.run_id / "quantitative"
        / request.task.task_id
    )
    assert (scope / "input.json").is_file()
    assert (scope / "draft.json").is_file()
    assert (scope / "result.json").is_file()
    assert not (scope / "raw_output.json").exists()


def test_schema_invalid_quantitative_response_is_not_persisted(tmp_path: Path):
    request = _request(tmp_path)

    with pytest.raises(QuantitativeExecutionError, match="required draft schema"):
        execute_quantitative(
            request,
            _FixedClient({"metrics": {"not": "provider-owned"}}),
            runs_root=tmp_path / "runs",
        )

    scope = (
        tmp_path / "runs" / request.run_id / "quantitative"
        / request.task.task_id
    )
    assert (scope / "input.json").is_file()
    assert not (scope / "draft.json").exists()
    assert not (scope / "raw_output.json").exists()
    assert not (scope / "result.json").exists()
