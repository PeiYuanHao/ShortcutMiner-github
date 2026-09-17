import re

import pytest
from pydantic import ValidationError

from shortcutminer.experiment.tool_executor import MockExperimentExecutor
from shortcutminer.tooling.agent_loop import (
    _validate_provider_output,
    finalize_tool_call_draft,
)
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.errors import ToolInputValidationError, ToolNotFound
from shortcutminer.tooling.protocol import ToolCallDraft
from shortcutminer.tooling.runtime import ToolRuntime


def _payload():
    return {
        "request": {
            "run_id": "pilot_001-diagnostic-quantitative",
            "candidate": {"candidate_id": "C002"},
            "task": {"task_id": "TASK-C002-QT"},
        }
    }


def _arguments(experiment_type="statistical_check"):
    return {
        "candidate_id": "C002",
        "task_id": "TASK-C002-QT",
        "experiment_type": experiment_type,
    }


def _runtime():
    runtime = ToolRuntime(
        build_default_tool_registry(
            experiment_executor=MockExperimentExecutor(metrics={"association": 0.25})
        ),
        run_id="pilot_001-diagnostic-quantitative",
        agent_name="quantitative",
    )
    runtime.bind_analysis_task(
        runtime_constraints=["no_external_services"],
        requested_checks=["association_test"],
    )
    return runtime


def _finalize(runtime, *, arguments=None, call_index=0):
    return finalize_tool_call_draft(
        {"name": "run_experiment", "arguments": arguments or _arguments()},
        agent_name="quantitative",
        base_payload=_payload(),
        tool_runtime=runtime,
        iteration=1,
        call_index=call_index,
    )


def test_provider_draft_finalizes_authoritative_identity_and_executes():
    runtime = _runtime()
    provider_output = {
        "tool_calls": [{"name": "run_experiment", "arguments": _arguments()}]
    }

    _validate_provider_output("quantitative", provider_output)
    call = _finalize(runtime)
    result = runtime.execute(call)

    assert call.agent_name == "quantitative"
    assert call.run_id == "pilot_001-diagnostic-quantitative"
    assert call.candidate_id == "C002"
    assert call.task_id == "TASK-C002-QT"
    assert re.fullmatch(r"TC-[0-9a-f]{32}", call.tool_call_id)
    assert result.status == "success"


def test_call_id_is_deterministic_and_changes_with_index_or_arguments():
    runtime = _runtime()
    first = _finalize(runtime)
    same = _finalize(runtime)
    other_index = _finalize(runtime, call_index=1)
    other_arguments = _finalize(
        runtime,
        arguments={**_arguments(), "parameters": {}},
    )

    assert first.tool_call_id == same.tool_call_id
    assert first.tool_call_id != other_index.tool_call_id
    assert first.tool_call_id != other_arguments.tool_call_id


def test_draft_rejects_unknown_name_and_canonical_provider_fields():
    runtime = _runtime()
    with pytest.raises(ToolNotFound):
        finalize_tool_call_draft(
            {"name": "unknown_tool", "arguments": {}},
            agent_name="quantitative",
            base_payload=_payload(),
            tool_runtime=runtime,
            iteration=1,
            call_index=0,
        )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ToolCallDraft.model_validate({
            "name": "run_experiment",
            "arguments": _arguments(),
            "tool_call_id": "provider-owned-id",
        })


def test_live_input_schema_and_resolved_vocabulary_remain_authoritative():
    invalid_runtime = _runtime()
    invalid_call = _finalize(invalid_runtime, arguments={"unexpected": True})
    with pytest.raises(ToolInputValidationError):
        invalid_runtime.execute(invalid_call)

    unresolved_runtime = _runtime()
    unresolved_call = _finalize(
        unresolved_runtime,
        arguments=_arguments("association_test"),
    )
    with pytest.raises(ToolInputValidationError):
        unresolved_runtime.execute(unresolved_call)

    resolved_runtime = _runtime()
    resolved = resolved_runtime.execute(_finalize(resolved_runtime))
    assert resolved.status == "success"
