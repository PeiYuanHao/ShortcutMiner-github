"""Cross-agent contract tests for the complete validated protocol chain."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from shortcutminer.experiment.summarize_executor import (
    SummarizeExecutionError,
    validate_summary_report,
)
from shortcutminer.protocol.batches import (
    QualitativeResultBatch,
    QuantitativeResultBatch,
)
from shortcutminer.protocol.executive import ExecutiveRequest, RoutingPlan
from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.protocol.qualitative import QualitativeResult
from shortcutminer.protocol.quantitative import QuantitativeResult
from shortcutminer.protocol.summarize import SummarizeRequest, SummaryReport
from shortcutminer.runner import run_workflow
from shortcutminer.storage.run_store import (
    RunStoreError,
    load_final_result,
    load_workflow_state,
    save_final_result,
)
from shortcutminer.workflow.checkpoint import (
    checkpoint_config,
    checkpoint_path,
    open_sqlite_checkpointer,
)
from shortcutminer.workflow.graph import build_workflow
from test_pipeline_mock import DeterministicAgentClient, _initial_state


@pytest.fixture(scope="module")
def contract_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("schema_contract")
    initial = _initial_state(root)
    initial["run_id"] = "schema_contract_run"
    client = DeterministicAgentClient()
    result = run_workflow(initial, agent_client=client)
    return root, client, result


def _validated_state(state: dict) -> dict:
    """Validate every canonical protocol object stored in graph state."""
    return {
        "identify": CandidateReport.model_validate(state["identify_result"]),
        "executive_request": ExecutiveRequest.model_validate(
            state["executive_request"]
        ),
        "executive": RoutingPlan.model_validate(state["executive_result"]),
        "quantitative": QuantitativeResultBatch.model_validate(
            state["quantitative_result"]
        ),
        "qualitative": QualitativeResultBatch.model_validate(
            state["qualitative_result"]
        ),
        "summarize_request": SummarizeRequest.model_validate(
            state["summarize_request"]
        ),
        "summary": SummaryReport.model_validate(state["summarize_result"]),
    }


def test_complete_executor_chain_uses_the_formal_schemas(contract_run):
    _root, client, state = contract_run
    parsed = _validated_state(state)

    assert parsed["executive_request"].candidate_report == parsed["identify"]
    assert parsed["summarize_request"].identify_result == parsed["identify"]
    assert parsed["summarize_request"].executive_result == parsed["executive"]
    assert parsed["summarize_request"].quantitative_result == parsed["quantitative"]
    assert parsed["summarize_request"].qualitative_result == parsed["qualitative"]
    assert state["final_result"] == state["summarize_result"]
    assert [call["agent_name"] for call in client.calls[:2]] == [
        "identify",
        "executive",
    ]


def test_routing_tasks_and_results_preserve_the_id_chain(contract_run):
    _root, _client, state = contract_run
    parsed = _validated_state(state)
    plan = parsed["executive"]
    quant = parsed["quantitative"].results[0]
    qual = parsed["qualitative"].results[0]
    summary = parsed["summary"]

    assert {task.task_id for task in plan.tasks} == {
        quant.payload.task_id,
        qual.payload.task_id,
    }
    assert {
        parsed["identify"].candidates[0].candidate_id,
        *[task.candidate_id for task in plan.tasks],
        quant.payload.candidate_id,
        qual.payload.candidate_id,
        summary.payload.findings[0].candidate_id,
    } == {"C001"}
    assert {
        parsed["identify"].run_id,
        plan.run_id,
        quant.run_id,
        qual.run_id,
        summary.run_id,
    } == {"schema_contract_run"}
    assert summary.payload.findings[0].quantitative_result_ids == [
        quant.message_id
    ]
    assert summary.payload.findings[0].qualitative_result_ids == [qual.message_id]


def test_parallel_branch_state_is_independent_and_merged(contract_run):
    _root, _client, state = contract_run

    assert state["quantitative_result"]["message_type"] == (
        "quantitative_result_batch"
    )
    assert state["qualitative_result"]["message_type"] == (
        "qualitative_result_batch"
    )
    assert state["stage_status"]["quantitative"] == "completed"
    assert state["stage_status"]["qualitative"] == "completed"


def test_result_batches_reject_cross_run_or_duplicate_results(contract_run):
    _root, _client, state = contract_run
    quant = deepcopy(state["quantitative_result"])
    quant["results"][0]["run_id"] = "another_run"
    with pytest.raises(ValidationError, match="run_id differs from batch"):
        QuantitativeResultBatch.model_validate(quant)

    qual = deepcopy(state["qualitative_result"])
    qual["results"].append(deepcopy(qual["results"][0]))
    qual["results"][1]["message_id"] = "schema_contract_run:qualitative:other"
    with pytest.raises(ValidationError, match="task_ids must be unique"):
        QualitativeResultBatch.model_validate(qual)


def test_status_and_failure_contracts_are_canonical_and_serializable(contract_run):
    _root, _client, state = contract_run
    failed = deepcopy(state["quantitative_result"]["results"][0])
    failed["payload"].update(
        status="failed",
        metrics={},
        evidence_level="Q0",
        evidence_direction="not_tested",
        limitations=["The analysis tool failed before producing evidence."],
    )
    parsed = QuantitativeResult.model_validate(failed)
    assert json.loads(parsed.model_dump_json())["payload"]["status"] == "failed"

    inconsistent = deepcopy(failed)
    inconsistent["payload"]["evidence_direction"] = "supports"
    with pytest.raises(ValidationError, match="must be marked not_tested"):
        QuantitativeResult.model_validate(inconsistent)

    alias = deepcopy(failed)
    alias["payload"]["status"] = "success"
    with pytest.raises(ValidationError, match="status"):
        QuantitativeResult.model_validate(alias)

    qualitative_alias = deepcopy(state["qualitative_result"]["results"][0])
    qualitative_alias["payload"]["evidence_status"] = "success"
    with pytest.raises(ValidationError, match="evidence_status"):
        QualitativeResult.model_validate(qualitative_alias)


def test_summary_artifacts_must_trace_to_validated_upstream(contract_run):
    _root, _client, state = contract_run
    request_data = deepcopy(state["summarize_request"])
    reference = "quantitative/TASK-C001-QT/metrics.json"
    request_data["quantitative_result"]["results"][0]["payload"][
        "artifact_references"
    ] = [reference]
    request = SummarizeRequest.model_validate(request_data)

    report_data = deepcopy(state["summarize_result"])
    report_data["payload"]["artifact_references"] = [reference]
    report = SummaryReport.model_validate(report_data)
    validate_summary_report(report, request)
    assert json.loads(report.model_dump_json())["payload"][
        "artifact_references"
    ] == [reference]

    report_data["payload"]["artifact_references"] = ["invented/report.json"]
    invented = SummaryReport.model_validate(report_data)
    with pytest.raises(SummarizeExecutionError, match="absent from validated"):
        validate_summary_report(invented, request)


def test_storage_round_trip_preserves_every_schema(contract_run):
    root, _client, state = contract_run
    runs_root = root / "runs"
    loaded_state = load_workflow_state(runs_root, state["run_id"])
    parsed = _validated_state(loaded_state)
    loaded_final = load_final_result(runs_root, state["run_id"])

    assert loaded_state == state
    assert SummaryReport.model_validate(loaded_final) == parsed["summary"]


def test_final_result_storage_rejects_non_schema_objects(tmp_path: Path):
    with pytest.raises(RunStoreError, match="valid SummaryReport"):
        save_final_result(tmp_path, "schema_storage_run", {"ok": True})


def test_checkpoint_round_trip_preserves_every_schema(contract_run):
    root, _client, state = contract_run
    database = checkpoint_path(root / "runs", state["run_id"])
    with open_sqlite_checkpointer(database) as saver:
        workflow = build_workflow(
            checkpointer=saver,
            agent_client=DeterministicAgentClient(),
        )
        restored = dict(workflow.get_state(checkpoint_config(state["run_id"])).values)

    _validated_state(restored)
    assert restored["quantitative_result"] == state["quantitative_result"]
    assert restored["qualitative_result"] == state["qualitative_result"]
    assert restored["summarize_result"] == state["summarize_result"]
