"""Strict deterministic execution plans for Quantitative experiments."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from shortcutminer.experiment.tool_executor import ExperimentType
from shortcutminer.protocol.executive import TaskId
from shortcutminer.protocol.identify import (
    CandidateId,
    NonEmptyText,
    ProtocolVersion,
    SafeRunId,
    StrictProtocolModel,
)
from shortcutminer.tooling.builtins import ExperimentToolInput


QUANTITATIVE_EXPERIMENT_PLAN_VERSION = "quantitative_experiment_plan_v1"

PlanExecutionState = Literal[
    "ready",
    "pending_dependency",
    "needs_scientific_spec",
    "unsupported",
    "completed",
    "failed",
]


class QuantitativeExperimentPlanStep(StrictProtocolModel):
    step_id: NonEmptyText
    planning_check: NonEmptyText
    experiment_type: ExperimentType
    dependency_step_ids: list[NonEmptyText] = Field(default_factory=list)
    execution_state: PlanExecutionState
    canonical_input: ExperimentToolInput | None = None
    scientific_scope: list[NonEmptyText] = Field(default_factory=list)
    required: bool
    description: NonEmptyText
    provenance: Literal[
        "executive_task",
        "execution_resources",
        "runtime_dependency",
        "explicit_scientific_spec_required",
    ]

    @model_validator(mode="after")
    def validate_materialization(self):
        if len(self.dependency_step_ids) != len(set(self.dependency_step_ids)):
            raise ValueError("plan step dependencies must be unique")
        if len(self.scientific_scope) != len(set(self.scientific_scope)):
            raise ValueError("plan step scientific scope must be unique")
        if self.execution_state in {"ready", "completed", "failed"}:
            if self.canonical_input is None:
                raise ValueError("materialized plan step requires canonical_input")
        elif self.canonical_input is not None:
            raise ValueError("non-ready plan step must not contain canonical_input")
        if (
            self.canonical_input is not None
            and self.canonical_input.experiment_type != self.experiment_type
        ):
            raise ValueError("plan step experiment_type differs from canonical input")
        return self


class QuantitativeExperimentPlan(StrictProtocolModel):
    protocol_version: ProtocolVersion = "1.0"
    run_id: SafeRunId
    task_id: TaskId
    candidate_id: CandidateId
    plan_version: Literal["quantitative_experiment_plan_v1"] = (
        QUANTITATIVE_EXPERIMENT_PLAN_VERSION
    )
    steps: list[QuantitativeExperimentPlanStep]

    @model_validator(mode="after")
    def validate_plan(self):
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("quantitative plan step IDs must be unique")
        known = set(ids)
        for step in self.steps:
            if step.step_id in step.dependency_step_ids:
                raise ValueError("plan step cannot depend on itself")
            if not set(step.dependency_step_ids).issubset(known):
                raise ValueError("plan step dependency is missing")
            if step.canonical_input is not None:
                if step.canonical_input.task_id != self.task_id:
                    raise ValueError("canonical input task identity differs from plan")
                if step.canonical_input.candidate_id != self.candidate_id:
                    raise ValueError("canonical input candidate differs from plan")
        graph = {step.step_id: step.dependency_step_ids for step in self.steps}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("quantitative plan dependency cycle")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in graph[step_id]:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in ids:
            visit(step_id)
        return self


class RunExperimentPlanStepDraft(StrictProtocolModel):
    """The complete real-provider input surface for run_experiment."""

    plan_step_id: NonEmptyText


__all__ = [
    "PlanExecutionState",
    "QUANTITATIVE_EXPERIMENT_PLAN_VERSION",
    "QuantitativeExperimentPlan",
    "QuantitativeExperimentPlanStep",
    "RunExperimentPlanStepDraft",
]
