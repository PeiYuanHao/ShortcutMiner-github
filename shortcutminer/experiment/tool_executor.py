"""Controlled experiment backend contract used by the Experiment Tool.

No unrestricted code or shell execution is exposed here. Concrete scientific
backends can implement this interface in later phases; Phase 7 tests use the
deterministic mock implementation below.
"""

from typing import Any, Literal, Protocol

from pydantic import Field

from shortcutminer.protocol.executive import TaskId
from shortcutminer.protocol.identify import (
    CandidateId,
    NonEmptyText,
    SafeRelativePath,
    SafeRunId,
    StrictProtocolModel,
)


ExperimentType = Literal[
    "statistical_check",
    "model_evaluation",
    "intervention",
    "ablation",
    "subgroup_evaluation",
    "metric_computation",
]
ExperimentOutcome = Literal["completed", "inconclusive"]


class ExperimentExecutionRequest(StrictProtocolModel):
    run_id: SafeRunId
    candidate_id: CandidateId
    task_id: TaskId
    experiment_type: ExperimentType
    parameters: dict[str, Any] = Field(default_factory=dict)


class ExperimentExecutionOutput(StrictProtocolModel):
    outcome: ExperimentOutcome
    metrics: dict[str, str | int | float | bool | None]
    summary: NonEmptyText
    artifact_references: list[SafeRelativePath] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentExecutor(Protocol):
    def execute(
        self,
        request: ExperimentExecutionRequest,
    ) -> ExperimentExecutionOutput: ...


class MockExperimentExecutor:
    """Deterministic schema-real backend for offline runtime tests."""

    def __init__(
        self,
        *,
        outcome: ExperimentOutcome = "completed",
        metrics: dict[str, str | int | float | bool | None] | None = None,
    ) -> None:
        self.outcome = outcome
        self.metrics = dict(metrics or {"mock_score": 0.5})
        self.calls: list[ExperimentExecutionRequest] = []

    def execute(
        self,
        request: ExperimentExecutionRequest,
    ) -> ExperimentExecutionOutput:
        self.calls.append(request)
        return ExperimentExecutionOutput(
            outcome=self.outcome,
            metrics=self.metrics,
            summary=(
                "Deterministic mock experiment completed."
                if self.outcome == "completed"
                else "Deterministic mock experiment was scientifically inconclusive."
            ),
            artifact_references=[],
            metadata={"backend": "mock", "experiment_type": request.experiment_type},
        )
