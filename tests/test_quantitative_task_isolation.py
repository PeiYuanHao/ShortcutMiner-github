"""Task-scoped Quantitative runtime authority and real-backend regressions."""

from pathlib import Path

from PIL import Image
import pytest

from shortcutminer.experiment.quantitative_executor import (
    determine_quantitative_execution_requirement,
    execute_quantitative,
)
from shortcutminer.experiment.quantitative_plan import (
    QuantitativePlanCompilerBug,
    compile_quantitative_experiment_plan,
)
from shortcutminer.experiment.real_executor import RealExperimentExecutor
from shortcutminer.tooling.agent_loop import finalize_tool_call_draft
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.protocol import ToolResult
from shortcutminer.tooling.runtime import ToolBudget, ToolRuntime
from test_quantitative_finalizer import _request


def _dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    images = root / "images"
    images.mkdir()
    rows = ["id,label,site,path"]
    for index in range(12):
        label = index % 2
        relative = f"images/item-{index:02d}.png"
        Image.new(
            "RGB",
            (16, 16),
            (220, 40 + label * 120, 30 + (index % 3) * 20),
        ).save(root / relative)
        rows.append(f"item-{index:02d},{label},site-{index % 3},{relative}")
    (root / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def _task_request(
    root: Path,
    *,
    candidate_id: str,
    requested_checks: list[str],
    metadata_fields: list[str] | None = None,
):
    dataset = _dataset(root) if not (root / "metadata.csv").exists() else root
    base = _request(root / "request-template")
    hypothesis = f"Candidate {candidate_id} requires isolated measurement."
    candidate = base.candidate.model_copy(update={
        "candidate_id": candidate_id,
        "hypothesis": hypothesis,
    })
    task_id = f"TASK-{candidate_id}-QT"
    fields = metadata_fields or []
    inputs = base.task.inputs.model_copy(update={
        "dataset": "task-isolation-fixture",
        "dataset_reference": str(dataset),
        "metadata_fields": fields,
        "requires_sample_access": "task_model_training" in requested_checks,
        "runtime_constraints": [],
        "required_model_capabilities": (
            ["retraining", "checkpoint"]
            if "task_model_training" in requested_checks else []
        ),
    })
    task = base.task.model_copy(update={
        "run_id": "task-isolation-run",
        "message_id": f"executive:task:{candidate_id}:QT",
        "task_id": task_id,
        "candidate_id": candidate_id,
        "hypothesis": hypothesis,
        "inputs": inputs,
        "requested_checks": requested_checks,
    })
    resources = base.execution_resources.model_copy(update={
        "dataset_name": "task-isolation-fixture",
        "available_metadata_fields": ["id", "label", "site", "path"],
        "allowed_scientific_metadata_fields": fields,
        "sample_id_column": "id",
        "sample_path_column": "path",
        "max_samples": 12,
        "bootstrap_iterations": 10,
        "max_model_iterations": 50,
    })
    return base.model_copy(update={
        "run_id": "task-isolation-run",
        "message_id": f"quantitative:request:{candidate_id}",
        "in_reply_to": task.message_id,
        "candidate": candidate,
        "task": task,
        "execution_resources": resources,
    })


def _base_runtime(dataset: Path, runs_root: Path, **updates) -> ToolRuntime:
    return ToolRuntime(
        build_default_tool_registry(experiment_executor=RealExperimentExecutor()),
        run_id="task-isolation-run",
        agent_name="quantitative",
        dataset_root=dataset,
        runs_root=runs_root,
        model_capabilities={"retraining", "checkpoint", "predictions", "logits"},
        budget=ToolBudget(max_tool_calls=12, timeout_seconds=30),
        **updates,
    )


def _fork(base: ToolRuntime, request) -> ToolRuntime:
    return base.fork_for_task(
        candidate_id=request.candidate.candidate_id,
        task_id=request.task.task_id,
        runtime_constraints=request.task.inputs.runtime_constraints,
        requested_checks=request.task.requested_checks,
        explicitly_authorized_artifacts=request.task.inputs.artifact_references,
    )


def _requirement(request, runtime):
    return determine_quantitative_execution_requirement(
        request, runtime, feasibility_confirmed=True
    )


def _execute_step(request, runtime, requirement, plan, step):
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
        call_index=len(runtime.history),
        experiment_plan=plan,
    )
    return call, runtime.execute(call)


def _execute_model_task(request, runtime):
    requirement = _requirement(request, runtime)
    initial = compile_quantitative_experiment_plan(request, requirement, runtime)
    by_type = {step.experiment_type: step for step in initial.steps}
    assert by_type["task_model_training"].execution_state == "ready"
    assert by_type["task_model_evaluation"].execution_state == "pending_dependency"
    training_call, training_result = _execute_step(
        request, runtime, requirement, initial, by_type["task_model_training"]
    )
    refreshed = compile_quantitative_experiment_plan(request, requirement, runtime)
    evaluation = next(
        step for step in refreshed.steps
        if step.experiment_type == "task_model_evaluation"
    )
    assert evaluation.execution_state == "ready"
    evaluation_call, evaluation_result = _execute_step(
        request, runtime, requirement, refreshed, evaluation
    )
    return {
        "initial": initial,
        "training_call": training_call,
        "training_result": training_result,
        "evaluation_call": evaluation_call,
        "evaluation_result": evaluation_result,
        "model_reference": evaluation.canonical_input.parameters.model_reference,
    }


def test_mid_dag_failed_evaluation_resumes_without_duplicate_training(tmp_path: Path):
    dataset = _dataset(tmp_path / "dataset")
    runs_root = tmp_path / "runs"
    request = _task_request(
        dataset, candidate_id="C001",
        requested_checks=["task_model_training", "task_model_evaluation"],
    )

    class FailEvaluationOnce(RealExperimentExecutor):
        def execute(self, execution_request, *, context=None):
            if execution_request.experiment_type == "task_model_evaluation":
                raise RuntimeError("planned evaluation interruption")
            return super().execute(execution_request, context=context)

    first_base = ToolRuntime(
        build_default_tool_registry(experiment_executor=FailEvaluationOnce()),
        run_id=request.run_id, agent_name="quantitative", dataset_root=dataset,
        runs_root=runs_root,
        model_capabilities={"retraining", "checkpoint", "predictions", "logits"},
        budget=ToolBudget(max_tool_calls=12, timeout_seconds=30),
    )
    first = _fork(first_base, request)
    requirement = _requirement(request, first)
    plan = compile_quantitative_experiment_plan(request, requirement, first)
    training = next(step for step in plan.steps if step.experiment_type == "task_model_training")
    _, trained = _execute_step(request, first, requirement, plan, training)
    assert trained.status == "success"
    refreshed = compile_quantitative_experiment_plan(request, requirement, first)
    evaluation = next(step for step in refreshed.steps if step.experiment_type == "task_model_evaluation")
    _, interrupted = _execute_step(request, first, requirement, refreshed, evaluation)
    assert interrupted.status == "failed"

    resumed_base = _base_runtime(
        dataset, runs_root, restore_persisted_task_results=True,
    )
    resumed = _fork(resumed_base, request)
    assert len(resumed.history) == 2
    from shortcutminer.runtime.demo import DemoAgentClient
    result = execute_quantitative(
        request, DemoAgentClient(), runs_root=runs_root,
        tool_runtime=resumed, execution_requirement=_requirement(request, resumed),
    )
    assert result.payload.status == "completed"
    assert len(result.payload.measurements) == 2
    assert len({item.plan_step_id for item in result.payload.measurements}) == 2
    assert sum(item.experiment_type == "task_model_training" for item in resumed.history) == 1


def test_sibling_training_does_not_satisfy_dependency_and_current_training_does(
    tmp_path: Path,
):
    dataset = _dataset(tmp_path / "dataset")
    base = _base_runtime(dataset, tmp_path / "runs")
    c1 = _task_request(dataset, candidate_id="C001", requested_checks=[
        "task_model_training", "task_model_evaluation"
    ])
    c2 = _task_request(dataset, candidate_id="C002", requested_checks=[
        "task_model_training", "task_model_evaluation"
    ])

    c1_trace = _execute_model_task(c1, _fork(base, c1))
    c2_runtime = _fork(base, c2)
    c2_initial = compile_quantitative_experiment_plan(
        c2, _requirement(c2, c2_runtime), c2_runtime
    )
    assert [step.execution_state for step in c2_initial.steps] == [
        "ready", "pending_dependency"
    ]
    c2_trace = _execute_model_task(c2, c2_runtime)

    assert c1_trace["model_reference"] in c1_trace["training_result"].artifacts
    assert c2_trace["model_reference"] in c2_trace["training_result"].artifacts
    assert c1_trace["model_reference"] != c2_trace["model_reference"]
    assert all(result.task_id == "TASK-C002-QT" for result in c2_runtime.history)


def test_approved_sibling_artifact_is_not_selected_and_history_hash_is_task_local(
    tmp_path: Path,
):
    dataset = _dataset(tmp_path / "dataset")
    base = _base_runtime(
        dataset,
        tmp_path / "runs",
        approved_artifacts={"models/another-task/model.json"},
    )
    request = _task_request(dataset, candidate_id="C002", requested_checks=[
        "task_model_training", "task_model_evaluation"
    ])
    first = _fork(base, request)
    plan = compile_quantitative_experiment_plan(request, _requirement(request, first), first)
    call_a, _ = _execute_step(request, first, _requirement(request, first), plan, plan.steps[0])
    second = _fork(base, request)
    plan_b = compile_quantitative_experiment_plan(
        request, _requirement(request, second), second
    )
    call_b = finalize_tool_call_draft(
        {"name": "run_experiment", "arguments": {"plan_step_id": plan_b.steps[0].step_id}},
        agent_name="quantitative",
        base_payload={
            "request": request.model_dump(mode="json", exclude={"execution_resources"}),
            "execution_requirement": _requirement(request, second).model_dump(mode="json"),
        },
        tool_runtime=second,
        iteration=0,
        call_index=0,
        experiment_plan=plan_b,
    )
    assert call_a.tool_call_id == call_b.tool_call_id
    assert "models/another-task/model.json" not in second.context.approved_artifacts


def test_same_task_ambiguous_models_still_reject(tmp_path: Path):
    dataset = _dataset(tmp_path / "dataset")
    runs = tmp_path / "runs"
    request = _task_request(dataset, candidate_id="C001", requested_checks=[
        "task_model_training", "task_model_evaluation"
    ])
    runtime = _fork(_base_runtime(dataset, runs), request)
    trace = _execute_model_task(request, runtime)
    duplicate = trace["training_result"].model_copy(update={
        "tool_call_id": "TC-same-task-second-model",
        "artifacts": ["models/same-task-second/model.json"],
    })
    runtime.history.append(duplicate)
    with pytest.raises(QuantitativePlanCompilerBug, match="authoritative_resource_missing"):
        compile_quantitative_experiment_plan(
            request, _requirement(request, runtime), runtime
        )


def test_provider_tool_runtime_results_are_task_local(tmp_path: Path):
    dataset = _dataset(tmp_path / "dataset")
    request = _task_request(dataset, candidate_id="C002", requested_checks=[
        "task_model_training", "task_model_evaluation"
    ])
    runtime = _fork(_base_runtime(dataset, tmp_path / "runs"), request)

    class LocalInterpretationClient:
        def __init__(self):
            self.payloads = []

        def invoke_agent(self, _agent_name, payload, *, system_prompt=None):
            self.payloads.append(payload)
            results = payload["tool_runtime"]["results"]
            if len(results) == 1:
                evaluation = next(
                    step for step in payload["experiment_plan_summary"]["steps"]
                    if step["experiment_type"] == "task_model_evaluation"
                    and step["execution_state"] == "ready"
                )
                return {"tool_calls": [{
                    "name": "run_experiment",
                    "arguments": {"plan_step_id": evaluation["step_id"]},
                }]}
            return {
                "status": "completed",
                "analysis_type": "offline_task_isolation",
                "evidence_direction": "inconclusive",
                "evidence_summary": "Only current-task real ToolResults were interpreted.",
                "additional_limitations": ["Offline engineering verification."],
                "recommended_followup": [],
            }

    client = LocalInterpretationClient()
    execute_quantitative(
        request,
        client,
        tool_runtime=runtime,
        execution_requirement=_requirement(request, runtime),
    )
    assert client.payloads
    assert all(
        result["candidate_id"] == "C002"
        and result["task_id"] == "TASK-C002-QT"
        for payload in client.payloads
        for result in payload["tool_runtime"]["results"]
    )


def test_three_task_real_backend_and_resume_history_are_isolated(tmp_path: Path):
    dataset = _dataset(tmp_path / "dataset")
    runs = tmp_path / "runs"
    base = _base_runtime(dataset, runs)
    requests = [
        _task_request(dataset, candidate_id="C001", requested_checks=[
            "task_model_training", "task_model_evaluation"
        ]),
        _task_request(dataset, candidate_id="C002", requested_checks=[
            "task_model_training", "task_model_evaluation"
        ]),
        _task_request(
            dataset,
            candidate_id="C003",
            requested_checks=["association_test"],
            metadata_fields=["site"],
        ),
    ]
    traces = [_execute_model_task(request, _fork(base, request)) for request in requests[:2]]
    c3_runtime = _fork(base, requests[2])
    c3_requirement = _requirement(requests[2], c3_runtime)
    c3_plan = compile_quantitative_experiment_plan(
        requests[2], c3_requirement, c3_runtime
    )
    _execute_step(requests[2], c3_runtime, c3_requirement, c3_plan, c3_plan.steps[0])
    assert len(c3_runtime.history) == 1
    assert not any(
        reference.endswith("/model.json")
        for result in c3_runtime.history
        for reference in result.artifacts
    )

    resumed_base = _base_runtime(
        dataset, runs, restore_persisted_task_results=True
    )
    restored = [_fork(resumed_base, request) for request in requests]
    assert [len(runtime.history) for runtime in restored] == [2, 2, 1]
    for request, runtime in zip(requests, restored, strict=True):
        assert all(
            result.candidate_id == request.candidate.candidate_id
            and result.task_id == request.task.task_id
            for result in runtime.history
        )
    assert traces[0]["model_reference"] != traces[1]["model_reference"]
