"""Compile Executive-authorized quantitative work before any provider call."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from shortcutminer.experiment.check_resolution import (
    EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES,
)
from shortcutminer.experiment.execution_contract import (
    ExperimentContractError,
    ExperimentDependencyPending,
    finalize_run_experiment_draft,
)
from shortcutminer.protocol.quantitative import (
    QuantitativeExecutionRequirement,
    QuantitativeRequest,
)
from shortcutminer.protocol.quantitative_plan import (
    QUANTITATIVE_EXPERIMENT_PLAN_VERSION,
    QuantitativeExperimentPlan,
    QuantitativeExperimentPlanStep,
)
from shortcutminer.tooling.protocol import ToolResult
from shortcutminer.tooling.runtime import ToolRuntime


class QuantitativePlanError(ValueError):
    category = "quantitative_plan_error"


class QuantitativePlanUnmaterializable(QuantitativePlanError):
    category = "quantitative_plan_unmaterializable"


class QuantitativePlanCompilerBug(QuantitativePlanError):
    category = "quantitative_plan_compiler_bug"


class UnknownPlanStep(QuantitativePlanError):
    category = "unknown_plan_step"


class PlanStepAlreadyCompleted(QuantitativePlanError):
    category = "plan_step_already_completed"


class PlanStepDependencyPending(QuantitativePlanError):
    category = "plan_step_dependency_pending"


def _step_id(
    *, task_id: str, planning_check: str, experiment_type: str,
    scientific_scope: list[str], ordinal: int,
) -> str:
    digest = hashlib.sha256(json.dumps({
        "version": QUANTITATIVE_EXPERIMENT_PLAN_VERSION,
        "task_id": task_id,
        "planning_check": planning_check,
        "experiment_type": experiment_type,
        "scientific_scope": scientific_scope,
        "ordinal": ordinal,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]
    return f"QSTEP-{digest}"


def _model_reference_available(history: list[ToolResult]) -> bool:
    return any(
        result.status == "success"
        and result.tool_name == "run_experiment"
        and any(reference.endswith("/model.json") for reference in result.artifacts)
        for result in history
    )


def _canonical(
    request: QuantitativeRequest,
    requirement: QuantitativeExecutionRequirement,
    runtime: ToolRuntime,
    experiment_type: str,
    parameters: dict[str, Any],
):
    resources = request.execution_resources
    if resources is None:
        raise QuantitativePlanUnmaterializable("authoritative_execution_resources_missing")
    dataset_root = runtime.context.dataset_root
    if dataset_root is None:
        raise QuantitativePlanUnmaterializable("approved_dataset_root_missing")
    runs_root = runtime.context.runs_root
    dependency_history = [
        result for result in runtime.history
        if not (
            result.status == "success"
            and result.tool_name == "run_experiment"
            and result.output is not None
            and result.output.get("metadata", {}).get("experiment_type")
            == experiment_type
        )
    ]
    try:
        return finalize_run_experiment_draft(
            {"experiment_type": experiment_type, "parameters": parameters},
            resources=resources,
            requirement=requirement,
            run_id=request.run_id,
            candidate_id=request.candidate.candidate_id,
            task_id=request.task.task_id,
            dataset_root=dataset_root,
            run_root=(runs_root / request.run_id if runs_root is not None else None),
            approved_artifacts=set(runtime.context.approved_artifacts),
            history=dependency_history,
        )
    except ExperimentDependencyPending:
        raise
    except ExperimentContractError as exc:
        raise QuantitativePlanCompilerBug(exc.category) from exc


def compile_quantitative_experiment_plan(
    request: QuantitativeRequest,
    requirement: QuantitativeExecutionRequirement,
    runtime: ToolRuntime,
) -> QuantitativeExperimentPlan:
    """Compile bounded canonical steps exclusively from public authoritative state."""

    resources = request.execution_resources
    if resources is None:
        raise QuantitativePlanUnmaterializable("authoritative_execution_resources_missing")
    allowed = list(resources.allowed_scientific_metadata_fields)
    explicit = list(request.task.inputs.metadata_fields)
    controlled = {
        resources.label_column,
        resources.sample_id_column,
        resources.sample_path_column,
        resources.split_column,
    } - {None}
    if allowed != [field for field in explicit if field not in controlled]:
        raise QuantitativePlanCompilerBug("scientific_scope_differs_from_executive_task")

    training_ids: list[str] = []
    steps: list[QuantitativeExperimentPlanStep] = []
    ordinal = 0

    def add(
        planning_check: str,
        experiment_type: str,
        scope: list[str],
        state: str,
        canonical_input,
        *,
        dependencies: list[str] | None = None,
        provenance: str = "executive_task",
        description: str,
    ) -> str:
        nonlocal ordinal
        ordinal += 1
        step_id = _step_id(
            task_id=request.task.task_id,
            planning_check=planning_check,
            experiment_type=experiment_type,
            scientific_scope=scope,
            ordinal=ordinal,
        )
        steps.append(QuantitativeExperimentPlanStep(
            step_id=step_id,
            planning_check=planning_check,
            experiment_type=experiment_type,
            dependency_step_ids=dependencies or [],
            execution_state=state,
            canonical_input=canonical_input,
            scientific_scope=scope,
            required=True,
            description=description,
            provenance=provenance,
        ))
        return step_id

    for planning_check in request.task.requested_checks:
        targets = EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES.get(planning_check, ())
        for experiment_type in targets:
            if experiment_type not in requirement.resolved_experiment_types:
                raise QuantitativePlanCompilerBug("experiment_type_not_resolved")
            if experiment_type == "statistical_check":
                if not allowed:
                    raise QuantitativePlanUnmaterializable(
                        "association_test_requires_scientific_metadata_field"
                    )
                for field in allowed:
                    canonical = _canonical(
                        request, requirement, runtime, experiment_type,
                        {"cue_field": field, "variable_mode": "auto"},
                    )
                    add(
                        planning_check, experiment_type, [field], "ready", canonical,
                        description=f"Statistical association between authorized field '{field}' and task label.",
                    )
            elif experiment_type == "subgroup_evaluation":
                if not allowed:
                    raise QuantitativePlanUnmaterializable(
                        "subgroup_evaluation_requires_scientific_metadata_field"
                    )
                for field in allowed:
                    canonical = _canonical(
                        request, requirement, runtime, experiment_type,
                        {"group_field": field},
                    )
                    add(
                        planning_check, experiment_type, [field], "ready", canonical,
                        description=f"Subgroup evaluation for authorized field '{field}'.",
                    )
            elif experiment_type == "model_evaluation":
                if not allowed:
                    raise QuantitativePlanUnmaterializable(
                        "model_evaluation_requires_scientific_metadata_field"
                    )
                canonical = _canonical(
                    request, requirement, runtime, experiment_type,
                    {"feature_fields": allowed},
                )
                add(
                    planning_check, experiment_type, allowed, "ready", canonical,
                    description="Predictive probe over the Executive-authorized metadata field set.",
                )
            elif experiment_type == "intervention":
                if not allowed:
                    raise QuantitativePlanUnmaterializable(
                        "intervention_requires_scientific_metadata_field"
                    )
                for field in allowed:
                    canonical = _canonical(
                        request, requirement, runtime, experiment_type,
                        {"cue_field": field, "intervention_type": "shuffle"},
                    )
                    add(
                        planning_check, experiment_type, [field], "ready", canonical,
                        description=f"Deterministic shuffle intervention for authorized field '{field}'.",
                    )
            elif experiment_type == "ablation":
                if len(allowed) < 2:
                    add(
                        planning_check, experiment_type, allowed,
                        "needs_scientific_spec", None,
                        provenance="explicit_scientific_spec_required",
                        description="Ablation requires at least two explicitly authorized feature fields.",
                    )
                else:
                    for field in allowed:
                        canonical = _canonical(
                            request, requirement, runtime, experiment_type,
                            {"feature_fields": allowed, "ablate_field": field},
                        )
                        add(
                            planning_check, experiment_type, allowed, "ready", canonical,
                            description=f"Ablate authorized field '{field}' from the fixed feature set.",
                        )
            elif experiment_type == "metric_computation":
                add(
                    planning_check, experiment_type, [], "needs_scientific_spec", None,
                    provenance="explicit_scientific_spec_required",
                    description="Metric computation requires a typed authoritative predictions artifact schema.",
                )
            elif experiment_type == "task_model_training":
                canonical = _canonical(
                    request, requirement, runtime, experiment_type,
                    {"class_weight": resources.training_class_weight,
                     "holdout_fraction": resources.training_holdout_fraction},
                )
                step_id = add(
                    planning_check, experiment_type, [], "ready", canonical,
                    provenance="execution_resources",
                    description="Train the frozen reference task model from authoritative dataset resources.",
                )
                training_ids.append(step_id)
            elif experiment_type == "task_model_evaluation":
                if not training_ids and not _model_reference_available(runtime.history):
                    raise QuantitativePlanUnmaterializable(
                        "task_model_evaluation_requires_training_step_or_model_artifact"
                    )
                try:
                    canonical = _canonical(
                        request, requirement, runtime, experiment_type, {}
                    )
                except ExperimentDependencyPending:
                    canonical = None
                add(
                    planning_check, experiment_type, [],
                    "ready" if canonical is not None else "pending_dependency",
                    canonical,
                    dependencies=training_ids,
                    provenance="runtime_dependency",
                    description="Evaluate the approved model artifact produced by its training dependency.",
                )
            elif experiment_type in {"ood_evaluation", "counterfactual_intervention"}:
                add(
                    planning_check, experiment_type, allowed,
                    "needs_scientific_spec", None,
                    dependencies=training_ids,
                    provenance="explicit_scientific_spec_required",
                    description=(
                        "A deterministic transformation specification is required before this step can execute."
                    ),
                )
            else:
                add(
                    planning_check, experiment_type, [], "unsupported", None,
                    description="No deterministic Phase 16 compiler is available for this experiment type.",
                )

    return QuantitativeExperimentPlan(
        run_id=request.run_id,
        task_id=request.task.task_id,
        candidate_id=request.candidate.candidate_id,
        steps=steps,
    )


def primary_mandatory_steps(
    plan: QuantitativeExperimentPlan,
    requested_checks: list[str],
) -> list[QuantitativeExperimentPlanStep]:
    for planning_check in requested_checks:
        ready = [
            step for step in plan.steps
            if step.planning_check == planning_check
            and step.required
            and step.execution_state == "ready"
        ]
        if ready:
            return ready
    return []


def plan_summary(plan: QuantitativeExperimentPlan) -> dict[str, Any]:
    return {
        "plan_version": plan.plan_version,
        "ready_step_ids": [s.step_id for s in plan.steps if s.execution_state == "ready"],
        "completed_step_ids": [s.step_id for s in plan.steps if s.execution_state == "completed"],
        "pending_step_ids": [
            s.step_id for s in plan.steps
            if s.execution_state in {"pending_dependency", "needs_scientific_spec"}
        ],
        "steps": [
            {
                "step_id": s.step_id,
                "planning_check": s.planning_check,
                "experiment_type": s.experiment_type,
                "execution_state": s.execution_state,
                "description": s.description,
                "dependency_step_ids": s.dependency_step_ids,
            }
            for s in plan.steps
        ],
    }


def plan_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


QUANTITATIVE_EXPERIMENT_PLAN_COMPILER_HASH = plan_hash()


__all__ = [
    "PlanStepAlreadyCompleted",
    "PlanStepDependencyPending",
    "QUANTITATIVE_EXPERIMENT_PLAN_COMPILER_HASH",
    "QuantitativePlanCompilerBug",
    "QuantitativePlanError",
    "QuantitativePlanUnmaterializable",
    "UnknownPlanStep",
    "compile_quantitative_experiment_plan",
    "plan_summary",
    "primary_mandatory_steps",
]
