from collections import deque

import pytest

from shortcutminer.experiment.quantitative_executor import execute_quantitative
from shortcutminer.experiment.quantitative_executor import QuantitativeDraftInvariantError
from shortcutminer.experiment.tool_executor import (
    ExperimentExecutionOutput,
    MockExperimentExecutor,
)
from shortcutminer.protocol.quantitative import QuantitativeExecutionRequirement
from shortcutminer.tooling.agent_loop import (
    RequiredExperimentNotAttempted,
    invoke_agent_with_tools,
)
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolRuntime
from test_quantitative_finalizer import _request


def _requirement(required: bool = True) -> QuantitativeExecutionRequirement:
    return QuantitativeExecutionRequirement(
        must_attempt_run_experiment=required,
        reason="task_is_executable" if required else "feasibility_not_confirmed",
        resolved_experiment_types=["statistical_check"] if required else [],
    )


def _blocked():
    return {
        "status": "blocked",
        "analysis_type": "association_test",
        "evidence_level": "Q0",
        "evidence_direction": "not_tested",
        "evidence_summary": "The required measurement could not be completed.",
        "evidence_tool_call_ids": [],
        "additional_limitations": ["The experiment backend failed."],
        "recommended_followup": [],
    }


def _completed(call_id: str, *, direction: str = "inconclusive"):
    return {
        "status": "completed",
        "analysis_type": "association_test",
        "evidence_level": "Q2",
        "evidence_direction": direction,
        "evidence_summary": "The executed measurement is scientifically inconclusive.",
        "evidence_tool_call_ids": [call_id],
        "additional_limitations": ["The analysis is observational."],
        "recommended_followup": [],
    }


def _call(request):
    return {
        "tool_calls": [{
            "name": "run_experiment",
            "arguments": {
                "experiment_type": "statistical_check",
                "parameters": {"cue_field": "site"},
            },
        }]
    }


class _ScriptedClient:
    def __init__(self, *outputs):
        self.outputs = deque(outputs)
        self.payloads = []

    def invoke_agent(self, agent_name, payload, *, system_prompt=None):
        self.payloads.append(payload)
        output = self.outputs.popleft()
        return output(payload) if callable(output) else output


def _completed_from_payload(payload, *, direction="inconclusive"):
    return _completed(
        payload["tool_runtime"]["results"][0]["tool_call_id"],
        direction=direction,
    )


def _runtime(request, executor):
    return ToolRuntime(
        build_default_tool_registry(experiment_executor=executor),
        run_id=request.run_id,
        agent_name="quantitative",
        dataset_root=request.task.inputs.dataset_reference,
    )


def _payload(request, requirement=None):
    return {
        "request": request.model_dump(
            mode="json", exclude={"execution_resources"}
        ),
        "execution_requirement": (requirement or _requirement()).model_dump(
            mode="json"
        ),
        "execution_resources": request.execution_resources.model_dump(mode="json"),
    }


def test_required_empty_history_rejects_blocked_after_one_correction(tmp_path):
    request = _request(tmp_path)
    client = _ScriptedClient(_blocked(), _blocked())

    with pytest.raises(RequiredExperimentNotAttempted) as raised:
        invoke_agent_with_tools(
            client,
            "quantitative",
            _payload(request),
            system_prompt="policy",
            tool_runtime=_runtime(request, MockExperimentExecutor()),
        )

    assert raised.value.category == "required_experiment_not_attempted"
    assert raised.value.retry_count == 1
    assert len(client.payloads) == 2
    assert "execution_obligation_correction" not in client.payloads[0]
    assert client.payloads[1]["execution_obligation_correction"]["category"] == (
        "required_experiment_not_attempted"
    )


def test_required_tool_call_executes_and_completed_draft_selects_result(tmp_path):
    request = _request(tmp_path)
    backend = MockExperimentExecutor(metrics={"effect_size": 0.0})
    runtime = _runtime(request, backend)
    client = _ScriptedClient(_call(request), _completed_from_payload)

    result = execute_quantitative(
        request,
        client,
        tool_runtime=runtime,
        execution_requirement=_requirement(),
    )

    assert len(backend.calls) == 1
    assert len(runtime.history) == 1
    assert runtime.history[0].status == "success"
    assert result.payload.status == "completed"
    assert result.payload.evidence_direction == "inconclusive"
    assert result.payload.metrics == {"effect_size": 0.0}


class _FailingExecutor:
    def execute(self, request, *, context=None):
        raise RuntimeError("bounded backend failure")


def test_failed_required_experiment_allows_blocked_q0(tmp_path):
    request = _request(tmp_path)
    runtime = _runtime(request, _FailingExecutor())
    client = _ScriptedClient(_call(request), _blocked())

    result = execute_quantitative(
        request,
        client,
        tool_runtime=runtime,
        execution_requirement=_requirement(),
    )

    assert runtime.history[0].status == "failed"
    assert result.payload.status == "blocked"
    assert result.payload.evidence_level == "Q0"
    assert result.payload.metrics == {}


def test_non_required_task_may_block_without_tool(tmp_path):
    request = _request(tmp_path)
    client = _ScriptedClient(_blocked())

    result = execute_quantitative(
        request,
        client,
        execution_requirement=_requirement(False),
    )

    assert result.payload.status == "blocked"
    assert len(client.payloads) == 1


def test_successful_scientifically_inconclusive_result_is_completed(tmp_path):
    request = _request(tmp_path)
    backend = MockExperimentExecutor(outcome="inconclusive", metrics={"sample_count": 4})
    runtime = _runtime(request, backend)
    client = _ScriptedClient(_call(request), _completed_from_payload)

    result = execute_quantitative(
        request,
        client,
        tool_runtime=runtime,
        execution_requirement=_requirement(),
    )

    assert runtime.history[0].output["outcome"] == "inconclusive"
    assert result.payload.status == "completed"
    assert result.payload.evidence_level == "Q2"
    assert result.payload.evidence_direction == "inconclusive"


def test_successful_experiment_cannot_be_downgraded_to_blocked(tmp_path):
    request = _request(tmp_path)
    runtime = _runtime(request, MockExperimentExecutor(outcome="inconclusive"))
    client = _ScriptedClient(_call(request), _blocked())

    with pytest.raises(QuantitativeDraftInvariantError, match="successful"):
        execute_quantitative(
            request,
            client,
            tool_runtime=runtime,
            execution_requirement=_requirement(),
        )


class _SchemaRetryClient(_ScriptedClient):
    def __init__(self, request):
        super().__init__(_blocked(), _call(request), _completed_from_payload)
        self.schema_retry_count = 0

    def invoke_agent_validated(
        self, agent_name, payload, *, system_prompt=None, validator
    ):
        if not self.payloads:
            try:
                validator({"unexpected": True})
            except Exception:
                self.schema_retry_count += 1
        output = self.invoke_agent(agent_name, payload, system_prompt=system_prompt)
        validator(output)
        return output


def test_execution_obligation_retry_is_independent_of_schema_retry(tmp_path):
    request = _request(tmp_path)
    client = _SchemaRetryClient(request)

    invocation = invoke_agent_with_tools(
        client,
        "quantitative",
        _payload(request),
        system_prompt="policy",
        tool_runtime=_runtime(request, MockExperimentExecutor()),
    )

    assert client.schema_retry_count == 1
    assert invocation.execution_obligation_retry_count == 1
    assert invocation.output["status"] == "completed"


def test_task_c002_fake_full_path_calls_tool_then_finalizes(tmp_path):
    base = _request(tmp_path)
    candidate = base.candidate.model_copy(update={"candidate_id": "C002"})
    task = base.task.model_copy(update={
        "candidate_id": "C002",
        "task_id": "TASK-C002-QT",
        "message_id": f"{base.run_id}:executive:TASK-C002-QT:task",
    })
    request = base.model_copy(update={
        "candidate": candidate,
        "task": task,
        "message_id": f"{base.run_id}:quantitative:TASK-C002-QT:request",
        "in_reply_to": task.message_id,
    })
    backend = MockExperimentExecutor(metrics={"association": 0.25})
    runtime = _runtime(request, backend)
    client = _ScriptedClient(
        _call(request),
        lambda payload: _completed_from_payload(payload, direction="supports"),
    )

    result = execute_quantitative(
        request,
        client,
        tool_runtime=runtime,
        execution_requirement=_requirement(),
    )

    assert result.payload.task_id == "TASK-C002-QT"
    assert result.payload.candidate_id == "C002"
    assert result.payload.status == "completed"
    assert result.payload.metrics == {"association": 0.25}
    assert len(backend.calls) == 1
    assert runtime.history[0].agent_name == "quantitative"
    assert runtime.history[0].run_id == request.run_id
    assert runtime.history[0].candidate_id == "C002"
    assert runtime.history[0].task_id == "TASK-C002-QT"
    assert runtime.history[0].tool_call_id.startswith("TC-")
