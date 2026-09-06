"""Provider-neutral configuration for ShortcutMiner model runtimes."""

from typing import Literal
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, field_validator

from shortcutminer.protocol.identify import NonEmptyText, StrictProtocolModel


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

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        return value.rstrip("/")

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_env(cls, value: str) -> str:
        if not value.replace("_", "a").isalnum() or value[0].isdigit():
            raise ValueError("api_key_env must be an environment variable name")
        return value


__all__ = ["OpenAICompatibleModelConfig"]
