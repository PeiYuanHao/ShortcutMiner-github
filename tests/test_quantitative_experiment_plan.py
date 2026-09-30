from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from shortcutminer.experiment.quantitative_plan import (
    PlanStepAlreadyCompleted,
    PlanStepDependencyPending,
    QuantitativePlanUnmaterializable,
    UnknownPlanStep,
    compile_quantitative_experiment_plan,
    primary_mandatory_steps,
)
from shortcutminer.experiment.real_executor import RealExperimentExecutor
from shortcutminer.experiment.quantitative_executor import (
    QuantitativeDraftInvariantError, finalize_quantitative_result,
)
from shortcutminer.experiment.tool_executor import ExperimentType, MockExperimentExecutor
from shortcutminer.protocol.quantitative import QuantitativeExecutionRequirement
from shortcutminer.protocol.quantitative_plan import (
    QuantitativeExperimentPlan,
    RunExperimentPlanStepDraft,
)
from shortcutminer.tooling.agent_loop import finalize_tool_call_draft
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolBudget, ToolRuntime
from test_quantitative_finalizer import _draft, _request, _tool_result


def test_wrong_plan_step_and_skipped_required_step_are_rejected(tmp_path: Path):
    request = _request(tmp_path)
    plan = compile_quantitative_experiment_plan(
        request, _requirement(["statistical_check"]), _runtime(request),
    )
    step = next(item for item in plan.steps if item.required)
    matching = _tool_result().model_copy(update={"plan_step_id": step.step_id})
    wrong = matching.model_copy(update={"plan_step_id": "QSTEP-forged"})
    with pytest.raises(QuantitativeDraftInvariantError, match="plan provenance"):
        finalize_quantitative_result(
            _draft(), request, tool_results=(wrong,), experiment_plan=plan,
        )
    with pytest.raises(QuantitativeDraftInvariantError, match="required plan step is unhandled"):
        finalize_quantitative_result(
            _draft(), request, tool_results=(matching,), experiment_plan=plan,
        )


ALL_TYPES = list(get_args(ExperimentType))


def _runtime(request, *, real=False, runs_root=None):
    return ToolRuntime(
        build_default_tool_registry(
            experiment_executor=(
                RealExperimentExecutor() if real else MockExperimentExecutor()
            )
        ),
        run_id=request.run_id,
        agent_name="quantitative",
        dataset_root=request.task.inputs.dataset_reference,
        runs_root=runs_root,
        model_capabilities={
            "predictions", "logits", "checkpoint", "retraining",
            "intervention_inference",
        },
        budget=ToolBudget(max_tool_calls=20, timeout_seconds=30),
    )


def _requirement(types):
    pending = [
        item for item in types
        if item in {"task_model_evaluation", "ood_evaluation", "counterfactual_intervention"}
    ]
    return QuantitativeExecutionRequirement(
        must_attempt_run_experiment=True,
        reason="task_is_executable",
        resolved_experiment_types=list(types),
        available_now_experiment_types=[item for item in types if item not in pending],
        pending_dependency_experiment_types=pending,
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"plan_step_id": "QSTEP-valid", "cue_field": "path"},
        {"plan_step_id": "QSTEP-valid", "cue_field": "id"},
        {"plan_step_id": "QSTEP-valid", "cue_field": "label"},
        {"plan_step_id": "QSTEP-valid", "cue_field": "unknown"},
        {"plan_step_id": "QSTEP-valid", "metadata_file": "C:\\data\\metadata.csv"},
        {"plan_step_id": "QSTEP-valid", "model_reference": "other/model.json"},
        {"plan_step_id": "QSTEP-valid", "feature_fields": ["arbitrary"]},
        {"plan_step_id": "QSTEP-valid", "ablate_field": "unknown"},
        {
            "experiment_type": "statistical_check",
            "parameters": {"cue_field": "site"},
        },
    ],
)
def test_live_provider_schema_has_no_raw_parameter_surface(arguments):
    with pytest.raises(ValidationError, match="extra_forbidden|Field required"):
        RunExperimentPlanStepDraft.model_validate(arguments)


def test_empty_executive_metadata_scope_does_not_fall_back_to_dataset_fields(tmp_path: Path):
    request = _request(tmp_path)
    task = request.task.model_copy(update={
        "inputs": request.task.inputs.model_copy(update={"metadata_fields": []})
    })
    resources = request.execution_resources.model_copy(update={
        "allowed_scientific_metadata_fields": []
    })
    request = request.model_copy(update={"task": task, "execution_resources": resources})
    runtime = _runtime(request)

    with pytest.raises(
        QuantitativePlanUnmaterializable,
        match="association_test_requires_scientific_metadata_field",
    ):
        compile_quantitative_experiment_plan(
            request, _requirement(["statistical_check"]), runtime
        )


def test_multiple_authorized_fields_compile_one_bounded_step_each(tmp_path: Path):
    request = _request(tmp_path)
    task = request.task.model_copy(update={
        "inputs": request.task.inputs.model_copy(
            update={"metadata_fields": ["site", "device"]}
        )
    })
    resources = request.execution_resources.model_copy(update={
        "available_metadata_fields": ["site", "device", "label"],
        "allowed_scientific_metadata_fields": ["site", "device"],
    })
    request = request.model_copy(update={"task": task, "execution_resources": resources})
    plan = compile_quantitative_experiment_plan(
        request, _requirement(["statistical_check"]), _runtime(request)
    )

    assert [step.scientific_scope for step in plan.steps] == [["site"], ["device"]]
    assert all(step.execution_state == "ready" for step in plan.steps)
    assert [
        step.canonical_input.parameters.cue_field for step in plan.steps
    ] == ["site", "device"]
    assert len(primary_mandatory_steps(plan, request.task.requested_checks)) == 2


def test_all_ten_experiment_types_are_classified_without_provider_parameters(tmp_path: Path):
    request = _request(tmp_path)
    checks = [
        "association_test", "subgroup_evaluation", "model_evaluation",
        "intervention", "ablation", "metric_computation",
        "task_model_training", "task_model_evaluation",
        "ood_evaluation", "counterfactual_intervention",
    ]
    task = request.task.model_copy(update={
        "requested_checks": checks,
        "inputs": request.task.inputs.model_copy(update={
            "metadata_fields": ["site", "device"],
            "required_model_capabilities": [
                "retraining", "checkpoint", "intervention_inference"
            ],
        }),
    })
    resources = request.execution_resources.model_copy(update={
        "available_metadata_fields": ["id", "path", "label", "site", "device"],
        "allowed_scientific_metadata_fields": ["site", "device"],
        "sample_id_column": "id",
        "sample_path_column": "path",
    })
    request = request.model_copy(update={"task": task, "execution_resources": resources})
    plan = compile_quantitative_experiment_plan(
        request, _requirement(ALL_TYPES), _runtime(request)
    )

    by_type = {}
    for step in plan.steps:
        by_type.setdefault(step.experiment_type, set()).add(step.execution_state)
        if step.canonical_input is not None:
            assert set(step.scientific_scope).issubset(task.inputs.metadata_fields)
    assert set(by_type) == set(ALL_TYPES)
    assert by_type["statistical_check"] == {"ready"}
    assert by_type["subgroup_evaluation"] == {"ready"}
    assert by_type["model_evaluation"] == {"ready"}
    assert by_type["intervention"] == {"ready"}
    assert by_type["ablation"] == {"ready"}
    assert by_type["task_model_training"] == {"ready"}
    assert by_type["task_model_evaluation"] == {"pending_dependency"}
    assert by_type["metric_computation"] == {"needs_scientific_spec"}
    assert by_type["ood_evaluation"] == {"needs_scientific_spec"}
    assert by_type["counterfactual_intervention"] == {"needs_scientific_spec"}


def test_plan_step_selection_rejects_unknown_completed_and_pending(tmp_path: Path):
    request = _request(tmp_path)
    runtime = _runtime(request)
    requirement = _requirement(["statistical_check"])
    plan = compile_quantitative_experiment_plan(request, requirement, runtime)
    payload = {
        "request": request.model_dump(mode="json", exclude={"execution_resources"}),
        "execution_requirement": requirement.model_dump(mode="json"),
    }

    def finalize(step_id, selected_plan=plan):
        return finalize_tool_call_draft(
            {"name": "run_experiment", "arguments": {"plan_step_id": step_id}},
            agent_name="quantitative",
            base_payload=payload,
            tool_runtime=runtime,
            iteration=1,
            call_index=0,
            experiment_plan=selected_plan,
        )

    with pytest.raises(UnknownPlanStep):
        finalize("QSTEP-unknown")
    completed = plan.model_copy(update={
        "steps": [plan.steps[0].model_copy(update={"execution_state": "completed"})]
    })
    with pytest.raises(PlanStepAlreadyCompleted):
        finalize(plan.steps[0].step_id, completed)
    pending = plan.model_copy(update={
        "steps": [
            plan.steps[0].model_copy(
                update={"execution_state": "pending_dependency", "canonical_input": None}
            )
        ]
    })
    with pytest.raises(PlanStepDependencyPending):
        finalize(plan.steps[0].step_id, pending)


def test_plan_protocol_rejects_missing_and_cyclic_dependencies(tmp_path: Path):
    request = _request(tmp_path)
    plan = compile_quantitative_experiment_plan(
        request, _requirement(["statistical_check"]), _runtime(request)
    )
    base = plan.steps[0].model_copy(update={
        "execution_state": "pending_dependency",
        "canonical_input": None,
    })
    missing = plan.model_dump(mode="json")
    missing["steps"] = [
        base.model_copy(update={
            "step_id": "QSTEP-a",
            "dependency_step_ids": ["QSTEP-missing"],
        }).model_dump(mode="json")
    ]
    with pytest.raises(ValidationError, match="dependency is missing"):
        QuantitativeExperimentPlan.model_validate(missing)

    cyclic = plan.model_dump(mode="json")
    cyclic["steps"] = [
        base.model_copy(update={
            "step_id": "QSTEP-a", "dependency_step_ids": ["QSTEP-b"]
        }).model_dump(mode="json"),
        base.model_copy(update={
            "step_id": "QSTEP-b", "dependency_step_ids": ["QSTEP-a"]
        }).model_dump(mode="json"),
    ]
    with pytest.raises(ValidationError, match="dependency cycle"):
        QuantitativeExperimentPlan.model_validate(cyclic)


def test_mandatory_ready_step_executes_real_backend_before_any_agent(tmp_path: Path):
    request = _request(tmp_path / "dataset")
    runs = tmp_path / "runs"
    runtime = _runtime(request, real=True, runs_root=runs)
    requirement = _requirement(["statistical_check"])
    plan = compile_quantitative_experiment_plan(request, requirement, runtime)
    step = primary_mandatory_steps(plan, request.task.requested_checks)[0]
    payload = {
        "request": request.model_dump(mode="json", exclude={"execution_resources"}),
        "execution_requirement": requirement.model_dump(mode="json"),
    }
    call = finalize_tool_call_draft(
        {"name": "run_experiment", "arguments": {"plan_step_id": step.step_id}},
        agent_name="quantitative",
        base_payload=payload,
        tool_runtime=runtime,
        iteration=0,
        call_index=0,
        experiment_plan=plan,
    )
    result = runtime.execute(call)
    assert result.status == "success"
    assert result.output["metadata"]["backend"] == "real_cpu_v1"
