"""Zero-provider Phase 16 Quantitative task-isolation replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from shortcutminer.experiment.quantitative_executor import (
    determine_quantitative_execution_requirement,
)
from shortcutminer.experiment.quantitative_plan import (
    compile_quantitative_experiment_plan,
)
from shortcutminer.experiment.real_executor import RealExperimentExecutor
from shortcutminer.protocol.quantitative import QuantitativeRequest
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.tooling.agent_loop import finalize_tool_call_draft
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolBudget, ToolRuntime


RUN_ID = "phase16-task-isolation-offline"


def _fixture(root: Path) -> Path:
    dataset = root / "public" / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = ["id,label,scientific_cue,path"]
    for index in range(12):
        label = index % 2
        relative = f"images/sample-{index:02d}.png"
        Image.new(
            "RGB", (16, 16),
            (30 + label * 170, 60 + (index % 3) * 40, 180 - label * 80),
        ).save(dataset / relative)
        rows.append(
            f"sample-{index:02d},{label},group-{index % 3},{relative}"
        )
    (dataset / "metadata.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    return dataset


def _request(
    dataset: Path,
    *,
    candidate_id: str,
    family: str,
    cue_name: str,
    requested_checks: list[str],
    metadata_fields: list[str],
) -> QuantitativeRequest:
    task_id = f"TASK-{candidate_id}-QT"
    hypothesis = f"Candidate {candidate_id} requires isolated offline measurement."
    return QuantitativeRequest.model_validate({
        "protocol_version": "1.0",
        "run_id": RUN_ID,
        "message_id": f"quantitative:request:{candidate_id}",
        "source": "orchestrator",
        "target": "quantitative",
        "message_type": "quantitative_analysis_request",
        "in_reply_to": f"executive:task:{candidate_id}:QT",
        "candidate": {
            "candidate_id": candidate_id,
            "cue": {"family": family, "name": cue_name},
            "description": "Generic public engineering contract candidate.",
            "observation": "The public fixture exposes a bounded candidate.",
            "hypothesis": hypothesis,
            "evidence_sources": ["metadata"],
            "priority": "high",
            "status": "unverified",
        },
        "task": {
            "protocol_version": "1.0",
            "run_id": RUN_ID,
            "message_id": f"executive:task:{candidate_id}:QT",
            "source": "executive",
            "target": "quantitative",
            "message_type": "analysis_task",
            "in_reply_to": "executive:plan:task-isolation",
            "task_id": task_id,
            "candidate_id": candidate_id,
            "hypothesis": hypothesis,
            "objective": "Verify task-scoped RealExperimentExecutor authority.",
            "inputs": {
                "dataset": "task-isolation-public-fixture",
                "dataset_reference": str(dataset),
                "label_column": "label",
                "metadata_fields": metadata_fields,
                "artifact_references": [],
                "requires_sample_access": "task_model_training" in requested_checks,
                "requires_external_services": False,
                "runtime_constraints": [],
                "required_model_capabilities": (
                    ["retraining", "checkpoint"]
                    if "task_model_training" in requested_checks else []
                ),
            },
            "requested_checks": requested_checks,
            "expected_outputs": ["metrics"],
            "priority": "high",
            "dependencies": [],
            "status": "planned",
        },
        "execution_resources": {
            "dataset_name": "task-isolation-public-fixture",
            "metadata_file": "metadata.csv",
            "label_column": "label",
            "available_metadata_fields": [
                "id", "label", "scientific_cue", "path"
            ],
            "allowed_scientific_metadata_fields": metadata_fields,
            "sample_id_column": "id",
            "sample_path_column": "path",
            "authorized_artifact_references": [],
            "seed": 1603,
            "max_rows": 100,
            "max_samples": 12,
            "bootstrap_iterations": 10,
            "max_model_iterations": 50,
        },
        "limitations": ["Offline engineering contract fixture only."],
    })


def _fork(base: ToolRuntime, request: QuantitativeRequest) -> ToolRuntime:
    return base.fork_for_task(
        candidate_id=request.candidate.candidate_id,
        task_id=request.task.task_id,
        runtime_constraints=request.task.inputs.runtime_constraints,
        requested_checks=request.task.requested_checks,
        explicitly_authorized_artifacts=request.task.inputs.artifact_references,
    )


def _call(request, runtime, requirement, plan, step):
    call = finalize_tool_call_draft(
        {"name": "run_experiment", "arguments": {"plan_step_id": step.step_id}},
        agent_name="quantitative",
        base_payload={
            "request": request.model_dump(
                mode="json", exclude={"execution_resources"}
            ),
            "execution_requirement": requirement.model_dump(mode="json"),
        },
        tool_runtime=runtime,
        iteration=0,
        call_index=len(runtime.history),
        experiment_plan=plan,
    )
    return call, runtime.execute(call)


def _model_task(request: QuantitativeRequest, runtime: ToolRuntime) -> dict:
    requirement = determine_quantitative_execution_requirement(
        request, runtime, feasibility_confirmed=True
    )
    initial = compile_quantitative_experiment_plan(request, requirement, runtime)
    states = {
        step.experiment_type: step.execution_state for step in initial.steps
    }
    if states != {
        "task_model_training": "ready",
        "task_model_evaluation": "pending_dependency",
    }:
        raise RuntimeError(f"invalid initial task availability: {states}")
    training = next(
        step for step in initial.steps
        if step.experiment_type == "task_model_training"
    )
    training_call, training_result = _call(
        request, runtime, requirement, initial, training
    )
    refreshed = compile_quantitative_experiment_plan(
        request, requirement, runtime
    )
    evaluation = next(
        step for step in refreshed.steps
        if step.experiment_type == "task_model_evaluation"
    )
    if evaluation.execution_state != "ready" or evaluation.canonical_input is None:
        raise RuntimeError("current-task training did not satisfy evaluation dependency")
    model_reference = evaluation.canonical_input.parameters.model_reference
    if model_reference not in training_result.artifacts:
        raise RuntimeError("evaluation model lineage is not current-task training")
    evaluation_call, evaluation_result = _call(
        request, runtime, requirement, refreshed, evaluation
    )
    if not all(result.status == "success" for result in runtime.history):
        raise RuntimeError("real model task backend did not succeed")
    return {
        "task_id": request.task.task_id,
        "initial_availability": states,
        "backend_attempts": len(runtime.history),
        "successful_results": sum(
            result.status == "success" for result in runtime.history
        ),
        "training_tool_call_id": training_call.tool_call_id,
        "model_artifact_reference": model_reference,
        "evaluation_tool_call_id": evaluation_call.tool_call_id,
        "evaluation_status": evaluation_result.status,
    }


def run_replay(root: Path) -> dict:
    dataset = _fixture(root)
    runs_root = root / "runs"
    base = ToolRuntime(
        build_default_tool_registry(experiment_executor=RealExperimentExecutor()),
        run_id=RUN_ID,
        agent_name="quantitative",
        dataset_root=dataset,
        runs_root=runs_root,
        model_capabilities={"retraining", "checkpoint", "predictions", "logits"},
        budget=ToolBudget(max_tool_calls=12, timeout_seconds=30),
    )
    c1 = _request(
        dataset, candidate_id="C001", family="appearance",
        cue_name="fixture_color", requested_checks=[
            "task_model_training", "task_model_evaluation"
        ], metadata_fields=[],
    )
    c2 = _request(
        dataset, candidate_id="C002", family="geometry",
        cue_name="fixture_geometry", requested_checks=[
            "task_model_training", "task_model_evaluation"
        ], metadata_fields=[],
    )
    c3 = _request(
        dataset, candidate_id="C003", family="metadata",
        cue_name="scientific_cue", requested_checks=["association_test"],
        metadata_fields=["scientific_cue"],
    )
    c1_runtime = _fork(base, c1)
    c1_trace = _model_task(c1, c1_runtime)
    c2_runtime = _fork(base, c2)
    c2_trace = _model_task(c2, c2_runtime)
    c3_runtime = _fork(base, c3)
    c3_requirement = determine_quantitative_execution_requirement(
        c3, c3_runtime, feasibility_confirmed=True
    )
    c3_plan = compile_quantitative_experiment_plan(
        c3, c3_requirement, c3_runtime
    )
    _call(c3, c3_runtime, c3_requirement, c3_plan, c3_plan.steps[0])
    runtimes = [c1_runtime, c2_runtime, c3_runtime]
    cross_task_results = sum(
        result.candidate_id != request.candidate.candidate_id
        or result.task_id != request.task.task_id
        for request, runtime in zip([c1, c2, c3], runtimes, strict=True)
        for result in runtime.history
    )
    c3_model_references = [
        reference
        for result in c3_runtime.history
        for reference in result.artifacts
        if reference.endswith("/model.json")
    ]
    if cross_task_results or c3_model_references:
        raise RuntimeError("cross-task evidence leaked into isolated runtime")
    report = {
        "schema_version": "1.0",
        "status": "passed",
        "provider_call_count": 0,
        "real_backend": "real_cpu_v1",
        "provider_visible_cross_task_result_count": 0,
        "tasks": [
            c1_trace,
            c2_trace,
            {
                "task_id": c3.task.task_id,
                "initial_availability": {
                    step.experiment_type: step.execution_state
                    for step in c3_plan.steps
                },
                "backend_attempts": len(c3_runtime.history),
                "successful_results": sum(
                    result.status == "success" for result in c3_runtime.history
                ),
                "model_artifact_references": c3_model_references,
            },
        ],
    }
    atomic_write_json(
        root / "task_isolation_offline.json",
        report,
        error_type=RuntimeError,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_replay(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
