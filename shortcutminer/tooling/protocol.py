"""Serializable provider-neutral ToolCall and ToolResult contracts."""

import json

from typing import Annotated, Any, Literal

from pydantic import Field, FiniteFloat, StringConstraints, field_validator, model_validator

from shortcutminer.protocol.executive import TaskId
from shortcutminer.protocol.identify import (
    CandidateId,
    NonEmptyText,
    ProtocolVersion,
    SafeRelativePath,
    SafeRunId,
    StrictProtocolModel,
)


AgentName = Literal[
    "identify",
    "executive",
    "quantitative",
    "qualitative",
    "summarize",
]
ToolStatus = Literal["success", "failed"]
ToolErrorType = Literal[
    "ToolExecutionError",
    "ToolTimeoutError",
]
ToolName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[a-z][a-z0-9_]{1,63}$"),
]
ToolCallId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    ),
]


def _unique(values: list[str], field_name: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


class ToolSpec(StrictProtocolModel):
    name: ToolName
    description: NonEmptyText
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


class ToolCallDraft(StrictProtocolModel):
    """Provider-owned tool selection and arguments, without control identity."""

    name: ToolName
    arguments: dict[str, Any]

    @model_validator(mode="after")
    def validate_strict_json_arguments(self):
        try:
            json.dumps(self.arguments, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("ToolCallDraft arguments must be strict JSON") from exc
        return self


class ToolCall(StrictProtocolModel):
    protocol_version: ProtocolVersion = "1.0"
    tool_call_id: ToolCallId
    tool_name: ToolName
    agent_name: AgentName
    run_id: SafeRunId
    candidate_id: CandidateId | None = None
    task_id: TaskId | None = None
    arguments: dict[str, Any]

    @model_validator(mode="after")
    def validate_task_candidate_correlation(self):
        if self.task_id is not None and self.candidate_id is not None:
            task_candidate = self.task_id.removeprefix("TASK-").rsplit("-", 1)[0]
            if task_candidate != self.candidate_id:
                raise ValueError("task_id and candidate_id identify different candidates")
        try:
            json.dumps(self.arguments, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("ToolCall arguments must be strict JSON") from exc
        return self


class ToolErrorInfo(StrictProtocolModel):
    error_type: ToolErrorType
    message: NonEmptyText


class ToolResult(StrictProtocolModel):
    protocol_version: ProtocolVersion = "1.0"
    tool_call_id: ToolCallId
    tool_name: ToolName
    agent_name: AgentName
    run_id: SafeRunId
    candidate_id: CandidateId | None = None
    task_id: TaskId | None = None
    status: ToolStatus
    output: dict[str, Any] | None = None
    error: ToolErrorInfo | None = None
    artifacts: list[SafeRelativePath] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: NonEmptyText
    completed_at: NonEmptyText
    duration_ms: FiniteFloat = Field(ge=0)

    @field_validator("artifacts")
    @classmethod
    def validate_unique_artifacts(cls, values: list[str]) -> list[str]:
        return _unique(values, "artifacts")

    @model_validator(mode="after")
    def validate_status_contract(self):
        if self.status == "success" and (self.output is None or self.error is not None):
            raise ValueError("successful tool results require output and no error")
        if self.status == "failed" and (self.output is not None or self.error is None):
            raise ValueError("failed tool results require error and no output")
        try:
            json.dumps(
                {"output": self.output, "metadata": self.metadata},
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("ToolResult payload must be strict JSON") from exc
        return self
