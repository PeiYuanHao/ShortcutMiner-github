"""Formal Runner request and persisted run lifecycle contracts."""

from typing import Any, Literal

from pydantic import Field, model_validator

from shortcutminer.protocol.identify import (
    DatasetSpec,
    NonEmptyText,
    SafeRunId,
    StrictProtocolModel,
    TaskSpec,
)
from shortcutminer.protocol.model import (
    ModelRuntimeConfigSet,
    OpenAICompatibleModelConfig,
)


RunStatus = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"]


class RunRequest(StrictProtocolModel):
    """Validated input accepted by the formal Python Runner API."""

    run_id: SafeRunId | None = None
    runs_root: NonEmptyText = "runs"
    dataset_info: DatasetSpec
    task_input: TaskSpec
    seed: int
    model_runtime: OpenAICompatibleModelConfig | ModelRuntimeConfigSet | None = None
    identify_config: dict[str, Any] = Field(default_factory=dict)
    executive_config: dict[str, Any] = Field(default_factory=dict)
    quantitative_config: dict[str, Any] = Field(default_factory=dict)
    qualitative_config: dict[str, Any] = Field(default_factory=dict)
    summarize_config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_managed_identify_fields(self):
        configured_seed = self.identify_config.get("seed")
        if configured_seed is not None and configured_seed != self.seed:
            raise ValueError("identify_config.seed must match RunRequest.seed")
        configured_root = self.identify_config.get("runs_root")
        if configured_root is not None and configured_root != self.runs_root:
            raise ValueError(
                "identify_config.runs_root must match RunRequest.runs_root"
            )
        return self


class RunError(StrictProtocolModel):
    error_type: NonEmptyText
    message: NonEmptyText


class CheckpointReference(StrictProtocolModel):
    backend: Literal["sqlite"]
    path: NonEmptyText
    thread_id: SafeRunId


class RunRecord(StrictProtocolModel):
    schema_version: Literal["1.0"]
    run_id: SafeRunId
    status: RunStatus
    created_at: NonEmptyText
    updated_at: NonEmptyText
    started_at: NonEmptyText | None = None
    completed_at: NonEmptyText | None = None
    failed_at: NonEmptyText | None = None
    resumed_at: NonEmptyText | None = None
    attempt_count: int = Field(ge=0)
    checkpoint: CheckpointReference
    error: RunError | None = None


class RunManifest(StrictProtocolModel):
    schema_version: Literal["1.0"]
    run_id: SafeRunId
    status: RunStatus
    dataset_info: DatasetSpec
    task_input: TaskSpec
    seed: int
    config: dict[str, Any]
    metadata: dict[str, Any]
    checkpoint: CheckpointReference
    artifacts: dict[NonEmptyText, NonEmptyText]
    created_at: NonEmptyText
    updated_at: NonEmptyText
    error: RunError | None = None


class RunExecutionResult(StrictProtocolModel):
    run_id: SafeRunId
    status: Literal["COMPLETED"]
    final_result: dict[str, Any]
    workflow_state: dict[str, Any]
    manifest: RunManifest
