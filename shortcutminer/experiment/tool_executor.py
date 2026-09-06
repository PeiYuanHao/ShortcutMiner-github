"""Controlled experiment backend contract used by the Experiment Tool.

No unrestricted code or shell execution is exposed here. Concrete scientific
backends can implement this interface in later phases; Phase 7 tests use the
deterministic mock implementation below.
"""

from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from shortcutminer.protocol.experiment import (
    EmptyExperimentParameters,
    ExperimentParameters,
    MetadataAblationParameters,
    MetadataInterventionParameters,
    MetricComputationParameters,
    PredictiveProbeParameters,
    StatisticalCheckParameters,
    SubgroupEvaluationParameters,
)
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
    parameters: ExperimentParameters = Field(
        default_factory=EmptyExperimentParameters
    )

    @model_validator(mode="after")
    def validate_parameters_for_type(self):
        expected = {
            "statistical_check": StatisticalCheckParameters,
            "subgroup_evaluation": SubgroupEvaluationParameters,
            "model_evaluation": PredictiveProbeParameters,
            "intervention": MetadataInterventionParameters,
            "ablation": MetadataAblationParameters,
            "metric_computation": MetricComputationParameters,
        }[self.experiment_type]
        if not isinstance(self.parameters, (expected, EmptyExperimentParameters)):
            raise ValueError(
                f"parameters do not match experiment_type={self.experiment_type}"
            )
        return self


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
        *,
        context=None,
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
        *,
        context=None,
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
