"""Strict provider-neutral contracts for model-call accounting and freezing."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, FiniteFloat, field_validator, model_validator

from shortcutminer.protocol.identify import (
    NonEmptyText,
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
    "semantic_visual",
]
RuntimeKind = Literal["text_agent", "semantic_vlm"]
InvocationStatus = Literal[
    "completed",
    "authentication_failed",
    "authorization_failed",
    "rate_limited",
    "timed_out",
    "transport_failed",
    "provider_5xx",
    "provider_rejected",
    "invalid_response_envelope",
    "invalid_json",
    "protocol_schema_failed",
    "budget_exceeded",
    "cancelled",
]
FailureCategory = Literal[
    "authentication_failed",
    "authorization_failed",
    "rate_limited",
    "timed_out",
    "transport_failed",
    "provider_5xx",
    "provider_rejected",
    "invalid_response_envelope",
    "invalid_json",
    "protocol_schema_failed",
    "budget_exceeded",
    "cancelled",
]


class ModelUsage(StrictProtocolModel):
    available: bool
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_availability(self):
        core = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        if self.available and any(value is None for value in core):
            raise ValueError("available usage requires prompt, completion, and total tokens")
        if not self.available and any(
            value is not None
            for value in (*core, self.cached_tokens, self.reasoning_tokens)
        ):
            raise ValueError("unavailable usage cannot contain token counts")
        if (
            self.available
            and self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens is not None
            and self.prompt_tokens + self.completion_tokens > self.total_tokens
        ):
            raise ValueError("total_tokens cannot be less than prompt plus completion")
        return self


class ModelFailure(StrictProtocolModel):
    category: FailureCategory
    message: NonEmptyText
    http_status: int | None = Field(default=None, ge=100, le=599)
    retryable: bool


class ModelRetryPolicy(StrictProtocolModel):
    max_attempts: int = Field(default=3, ge=1, le=5)
    initial_backoff_seconds: FiniteFloat = Field(default=0.5, ge=0, le=60)
    backoff_multiplier: FiniteFloat = Field(default=2.0, ge=1, le=10)
    max_backoff_seconds: FiniteFloat = Field(default=8.0, ge=0, le=300)
    retry_on_rate_limit: bool = True
    retry_on_timeout: bool = True
    retry_on_5xx: bool = True


class ModelCallBudget(StrictProtocolModel):
    max_total_calls: int = Field(default=100, ge=1, le=10_000)
    max_calls_per_agent: int = Field(default=25, ge=1, le=1_000)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_schema_retries: int = Field(default=1, ge=0, le=5)
    max_transport_retries: int = Field(default=2, ge=0, le=4)


class ModelCapabilities(StrictProtocolModel):
    supports_text: bool = True
    supports_image_input: bool = False
    supports_json_object: bool = True
    supports_model_seed: bool = False
    supports_reasoning_config: bool = False


class ModelReasoningConfig(StrictProtocolModel):
    mode: Literal["enabled", "disabled"]
    wire_format: Literal["thinking", "enable_thinking"] = "thinking"
    effort: Literal["low", "high", "max"] | None = None

    @model_validator(mode="after")
    def validate_effort(self):
        if self.mode == "disabled" and self.effort is not None:
            raise ValueError("disabled reasoning mode must not set an effort")
        return self


class ModelResponseDiagnostics(StrictProtocolModel):
    finish_reason: Literal[
        "stop", "length", "content_filter", "tool_calls", "other", "missing"
    ]
    content_present: bool
    content_length: int = Field(ge=0)
    content_whitespace_only: bool
    content_first_token_class: Literal[
        "object_open", "array_open", "markdown_fence", "other", "empty"
    ]
    content_last_token_class: Literal[
        "object_close", "array_close", "markdown_fence", "other", "empty"
    ]
    json_parse_status: Literal[
        "valid_object", "valid_non_object", "invalid_json",
        "content_missing", "content_not_string",
    ]


class ModelSchemaIssue(StrictProtocolModel):
    """A value-free Pydantic issue safe to persist or return to a model."""

    field_path: NonEmptyText
    error_type: NonEmptyText


class ModelSchemaDiagnostics(StrictProtocolModel):
    """Bounded schema metadata which never contains rejected input values."""

    total_schema_issue_count: int = Field(ge=0)
    issues: list[ModelSchemaIssue] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_issue_count(self):
        if len(self.issues) > self.total_schema_issue_count:
            raise ValueError("saved schema issues cannot exceed the total issue count")
        return self


class ModelEndpointIdentity(StrictProtocolModel):
    scheme: Literal["http", "https"]
    host: NonEmptyText
    path: NonEmptyText


class ModelRuntimeSnapshot(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    snapshot_id: NonEmptyText
    provider: NonEmptyText
    model: NonEmptyText
    endpoint: ModelEndpointIdentity
    api_key_env: NonEmptyText
    temperature: FiniteFloat
    max_tokens: int = Field(ge=1)
    timeout_seconds: FiniteFloat = Field(gt=0)
    retry_policy: ModelRetryPolicy
    model_call_budget: ModelCallBudget
    model_seed: int | None = None
    capabilities: ModelCapabilities
    reasoning: ModelReasoningConfig | None = None
    structured_output_mode: Literal["json_object", "none"] = "none"
    raw_response_persistence: Literal[False] = False


class ModelImageInput(StrictProtocolModel):
    sample_id: NonEmptyText
    sample_reference: SafeRelativePath
    image_sha256: NonEmptyText
    mime_type: NonEmptyText
    width: int = Field(ge=1)
    height: int = Field(ge=1)

    @field_validator("image_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("image_sha256 must be lowercase hexadecimal SHA-256")
        return value


class ModelInvocationRecord(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    invocation_id: NonEmptyText
    run_id: SafeRunId
    agent_name: AgentName
    runtime_kind: RuntimeKind
    provider: NonEmptyText
    model: NonEmptyText
    attempt: int = Field(ge=1)
    request_hash: NonEmptyText
    system_prompt_hash: NonEmptyText
    payload_hash: NonEmptyText
    response_hash: NonEmptyText | None = None
    started_at: NonEmptyText
    completed_at: NonEmptyText
    latency_ms: FiniteFloat = Field(ge=0)
    status: InvocationStatus
    failure_category: FailureCategory | None = None
    http_status: int | None = Field(default=None, ge=100, le=599)
    retryable: bool
    usage: ModelUsage
    schema_valid: bool | None = None
    transport_retry: bool = False
    schema_retry: bool = False
    structured_output_mode: Literal["json_object", "none"] = "none"
    response_kind: Literal["final_result", "tool_call", "unknown"] = "unknown"
    response_diagnostics: ModelResponseDiagnostics | None = None
    schema_diagnostics: ModelSchemaDiagnostics | None = None
    image_inputs: list[ModelImageInput] = Field(default_factory=list)
    limitations: list[NonEmptyText] = Field(default_factory=list)
    artifact_reference: SafeRelativePath | None = None

    @model_validator(mode="after")
    def validate_status(self):
        if self.status == "completed":
            if self.failure_category is not None or self.retryable:
                raise ValueError("completed invocation cannot contain failure state")
            if self.schema_valid is not True:
                raise ValueError("completed invocation must be schema-valid")
        elif self.failure_category != self.status:
            raise ValueError("failed invocation category must match status")
        if len(self.limitations) != len(set(self.limitations)):
            raise ValueError("invocation limitations must be unique")
        return self


class AgentModelTelemetry(StrictProtocolModel):
    agent_name: AgentName
    total_calls: int = Field(ge=0)
    successful_calls: int = Field(ge=0)
    failed_calls: int = Field(ge=0)
    retry_calls: int = Field(ge=0)
    schema_retry_calls: int = Field(ge=0)
    rate_limit_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    provider_5xx_count: int = Field(ge=0)
    invalid_json_count: int = Field(ge=0)
    protocol_schema_failure_count: int = Field(ge=0)
    budget_exceeded_count: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    usage_complete: bool
    total_model_latency_ms: FiniteFloat = Field(ge=0)


class RunModelTelemetry(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: SafeRunId
    runtime_snapshot_ids: list[NonEmptyText]
    total_calls: int = Field(ge=0)
    successful_calls: int = Field(ge=0)
    failed_calls: int = Field(ge=0)
    retry_calls: int = Field(ge=0)
    schema_retry_calls: int = Field(ge=0)
    rate_limit_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    provider_5xx_count: int = Field(ge=0)
    invalid_json_count: int = Field(ge=0)
    protocol_schema_failure_count: int = Field(ge=0)
    budget_exceeded_count: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    usage_complete: bool
    total_model_latency_ms: FiniteFloat = Field(ge=0)
    per_agent: list[AgentModelTelemetry]
    invocation_references: list[SafeRelativePath]
    limitations: list[NonEmptyText]


__all__ = [
    "AgentModelTelemetry",
    "AgentName",
    "FailureCategory",
    "InvocationStatus",
    "ModelCallBudget",
    "ModelCapabilities",
    "ModelEndpointIdentity",
    "ModelFailure",
    "ModelInvocationRecord",
    "ModelImageInput",
    "ModelReasoningConfig",
    "ModelResponseDiagnostics",
    "ModelSchemaDiagnostics",
    "ModelSchemaIssue",
    "ModelRetryPolicy",
    "ModelRuntimeSnapshot",
    "ModelUsage",
    "RunModelTelemetry",
    "RuntimeKind",
]
