"""Provider-neutral configuration for ShortcutMiner model runtimes."""

from typing import Literal
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, field_validator, model_validator

from shortcutminer.protocol.identify import NonEmptyText, StrictProtocolModel
from shortcutminer.protocol.model_runtime import (
    ModelCallBudget,
    ModelCapabilities,
    ModelReasoningConfig,
    ModelRetryPolicy,
)


TextAgentName = Literal[
    "identify", "executive", "quantitative", "qualitative", "summarize"
]


class OpenAICompatibleModelConfig(StrictProtocolModel):
    """Safe, serializable settings for an OpenAI-compatible chat endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    provider: Literal["openai_compatible"] = "openai_compatible"
    model: NonEmptyText
    base_url: NonEmptyText
    api_key_env: NonEmptyText
    timeout: float = Field(default=60.0, gt=0)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=4096, gt=0)
    model_seed: int | None = None
    reasoning: ModelReasoningConfig | None = None
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    retry_policy: ModelRetryPolicy = Field(default_factory=ModelRetryPolicy)
    model_call_budget: ModelCallBudget = Field(default_factory=ModelCallBudget)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain endpoint credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        return value.rstrip("/")

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_env(cls, value: str) -> str:
        if not value.replace("_", "a").isalnum() or value[0].isdigit():
            raise ValueError("api_key_env must be an environment variable name")
        return value

    @model_validator(mode="after")
    def validate_optional_capabilities(self):
        if self.reasoning is not None and not self.capabilities.supports_reasoning_config:
            raise ValueError(
                "reasoning configuration requires supports_reasoning_config=true"
            )
        return self


class ModelRuntimeConfigSet(StrictProtocolModel):
    """One shared model config with optional explicit per-Agent overrides."""

    shared: OpenAICompatibleModelConfig
    per_agent: dict[TextAgentName, OpenAICompatibleModelConfig] = Field(
        default_factory=dict
    )

    def resolve(self, agent_name: TextAgentName) -> OpenAICompatibleModelConfig:
        return self.per_agent.get(agent_name, self.shared)


__all__ = [
    "ModelRuntimeConfigSet",
    "OpenAICompatibleModelConfig",
    "TextAgentName",
]
