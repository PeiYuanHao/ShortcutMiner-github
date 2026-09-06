"""Provider-neutral AgentClient for OpenAI-compatible chat APIs."""

import json
import os
from collections.abc import Mapping
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from shortcutminer.protocol.model import OpenAICompatibleModelConfig


class ModelRuntimeError(RuntimeError):
    """Raised when a configured model runtime cannot return structured output."""


class JsonHttpTransport(Protocol):
    """Small injectable HTTP boundary used to keep tests offline."""

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout: float,
    ) -> object: ...


class UrllibJsonTransport:
    """Standard-library JSON POST transport with bounded request time."""

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout: float,
    ) -> object:
        request = Request(
            url,
            data=json.dumps(body, allow_nan=False).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            raise ModelRuntimeError(
                f"Model endpoint returned HTTP {exc.code}."
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ModelRuntimeError("Model endpoint request failed.") from exc
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelRuntimeError(
                "Model endpoint returned invalid JSON."
            ) from exc


class OpenAICompatibleClient:
    """Invoke any configured OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        config: OpenAICompatibleModelConfig,
        *,
        api_key: str | None = None,
        transport: JsonHttpTransport | None = None,
    ) -> None:
        self.config = config
        self._explicit_api_key = api_key
        self._transport = transport or UrllibJsonTransport()

    def _api_key(self) -> str:
        key = self._explicit_api_key
        if key is None:
            key = os.environ.get(self.config.api_key_env)
        if not key or not key.strip():
            raise ModelRuntimeError(
                f"Model credential environment variable is not set: "
                f"{self.config.api_key_env}"
            )
        return key.strip()

    def invoke_agent(
        self,
        agent_name: str,
        payload: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> object:
        if not agent_name.strip():
            raise ModelRuntimeError("agent_name must not be empty.")
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
        )
        response = self._transport.post_json(
            f"{self.config.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ShortcutMiner/0.1",
            },
            body={
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=self.config.timeout,
        )
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: object) -> dict[str, Any]:
        try:
            content = response["choices"][0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelRuntimeError(
                "Model response does not contain choices[0].message.content."
            ) from exc
        if not isinstance(content, str):
            raise ModelRuntimeError("Model response content must be a JSON string.")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ModelRuntimeError(
                "Model response content is not valid JSON."
            ) from exc
        if not isinstance(result, dict):
            raise ModelRuntimeError(
                "Model response content must decode to an object."
            )
        return result


__all__ = [
    "JsonHttpTransport",
    "ModelRuntimeError",
    "OpenAICompatibleClient",
    "UrllibJsonTransport",
]
