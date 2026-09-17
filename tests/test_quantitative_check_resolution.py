from typing import get_args

import pytest

from shortcutminer.experiment.check_resolution import (
    EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES,
    QUANTITATIVE_CHECK_RESOLUTION_HASH,
    QUANTITATIVE_CHECK_RESOLUTION_VERSION,
    QUANTITATIVE_EXPERIMENT_TYPES,
    UnsupportedQuantitativeCheck,
    resolve_quantitative_checks,
)
from shortcutminer.experiment.quantitative_executor import (
    determine_quantitative_execution_requirement,
)
from shortcutminer.experiment.real_executor import BACKEND_REGISTRY
from shortcutminer.experiment.tool_executor import ExperimentType, MockExperimentExecutor
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.errors import ToolInputValidationError
from shortcutminer.tooling.protocol import ToolCall
from shortcutminer.tooling.runtime import ToolRuntime
from test_quantitative_finalizer import _request


EXPECTED_PLANNING_VOCABULARY = {
    "association_test": ("statistical_check",),
    "task_model_training": ("task_model_training",),
    "task_model_evaluation": ("task_model_evaluation",),
    "ood_evaluation": ("ood_evaluation",),
    "counterfactual_intervention": ("counterfactual_intervention",),
}


def _runtime(request):
    return ToolRuntime(
        build_default_tool_registry(
            experiment_executor=MockExperimentExecutor()
        ),
        run_id=request.run_id,
        agent_name="quantitative",
        dataset_root=request.task.inputs.dataset_reference,
        model_capabilities={
            "retraining", "checkpoint", "intervention_inference"
        },
    )


def test_current_executive_quantitative_vocabulary_is_fully_mapped():
    assert dict(EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES) == (
        EXPECTED_PLANNING_VOCABULARY
    )
    assert QUANTITATIVE_CHECK_RESOLUTION_VERSION == (
        "quantitative_check_resolution_v1"
    )
    assert len(QUANTITATIVE_CHECK_RESOLUTION_HASH) == 64


@pytest.mark.parametrize(
    ("planning_check", "runtime_types"),
    EXPECTED_PLANNING_VOCABULARY.items(),
)
def test_each_declared_check_resolves_to_existing_authorized_backend(
    tmp_path, planning_check, runtime_types
):
    request = _request(tmp_path)
    runtime = _runtime(request)

    assert resolve_quantitative_checks([planning_check]) == runtime_types
    assert all(item in BACKEND_REGISTRY for item in runtime_types)
    runtime.bind_analysis_task(
        runtime_constraints=request.task.inputs.runtime_constraints,
        requested_checks=[planning_check],
    )
    assert runtime.context.allowed_experiment_types == frozenset(runtime_types)
    assert any(spec.name == "run_experiment" for spec in runtime.tool_specs())


def test_resolver_is_stable_deduplicated_and_fails_closed():
    checks = [
        "association_test",
        "task_model_evaluation",
        "association_test",
    ]
    assert resolve_quantitative_checks(checks) == (
        "statistical_check",
        "task_model_evaluation",
    )
    with pytest.raises(
        UnsupportedQuantitativeCheck,
        match="unknown_quantitative_check",
    ):
        resolve_quantitative_checks(["unknown_quantitative_check"])


def test_runtime_backend_and_protocol_experiment_vocabularies_match():
    declared = set(get_args(ExperimentType))
    assert set(QUANTITATIVE_EXPERIMENT_TYPES) == declared
    assert set(BACKEND_REGISTRY) == declared
    assert {
        target
        for targets in EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES.values()
        for target in targets
    }.issubset(declared)


def test_task_c002_association_requirement_resolves_and_binds(tmp_path):
    base = _request(tmp_path)
    candidate = base.candidate.model_copy(update={"candidate_id": "C002"})
    inputs = base.task.inputs.model_copy(update={
        "requires_sample_access": True,
        "required_model_capabilities": [
            "retraining", "checkpoint", "intervention_inference"
        ],
        "runtime_constraints": ["no_external_services"],
    })
    task = base.task.model_copy(update={
        "candidate_id": "C002",
        "task_id": "TASK-C002-QT",
        "message_id": f"{base.run_id}:executive:TASK-C002-QT:task",
        "requested_checks": [
            "association_test",
            "task_model_training",
            "task_model_evaluation",
            "ood_evaluation",
            "counterfactual_intervention",
        ],
        "inputs": inputs,
    })
    request = base.model_copy(update={
        "candidate": candidate,
        "task": task,
        "message_id": f"{base.run_id}:quantitative:TASK-C002-QT:request",
        "in_reply_to": task.message_id,
    })
    runtime = _runtime(request)

    requirement = determine_quantitative_execution_requirement(
        request,
        runtime,
        feasibility_confirmed=True,
    )
    runtime.bind_analysis_task(
        runtime_constraints=task.inputs.runtime_constraints,
        requested_checks=task.requested_checks,
    )

    assert requirement.must_attempt_run_experiment is True
    assert requirement.reason == "task_is_executable"
    assert requirement.resolved_experiment_types == [
        "statistical_check",
        "task_model_training",
        "task_model_evaluation",
        "ood_evaluation",
        "counterfactual_intervention",
    ]
    assert "statistical_check" in runtime.context.allowed_experiment_types
    assert "association_test" not in runtime.context.allowed_experiment_types


def test_tool_boundary_rejects_planning_alias_without_rewriting(tmp_path):
    request = _request(tmp_path)
    runtime = _runtime(request)
    runtime.bind_analysis_task(
        runtime_constraints=request.task.inputs.runtime_constraints,
        requested_checks=["association_test"],
    )
    call = ToolCall(
        tool_call_id="wrong-planning-alias",
        tool_name="run_experiment",
        agent_name="quantitative",
        run_id=request.run_id,
        candidate_id=request.candidate.candidate_id,
        task_id=request.task.task_id,
        arguments={
            "candidate_id": request.candidate.candidate_id,
            "task_id": request.task.task_id,
            "experiment_type": "association_test",
        },
    )

    with pytest.raises(ToolInputValidationError, match="association_test"):
        runtime.execute(call)
    assert runtime.history == []
