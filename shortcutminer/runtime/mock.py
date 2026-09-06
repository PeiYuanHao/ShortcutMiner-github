"""Deterministic, network-free AgentClient implementation."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from threading import Lock
from typing import Any


MockResponse = dict[str, Any]
MockResponseFactory = Callable[[dict[str, Any], str | None], MockResponse]


class MockAgentClientError(RuntimeError):
    """Raised when a scripted mock response is unavailable or invalid."""


class MockAgentClient:
    """Return per-agent scripted results without credentials or network access.

    Each response is copied before it is returned so callers cannot mutate the
    script. Factories can derive protocol identifiers from the structured input.
    The per-agent invocation count is stable even when graph branches run in
    parallel.
    """

    def __init__(
        self,
        responses: Mapping[str, MockResponse | MockResponseFactory],
    ) -> None:
        self._responses = dict(responses)
        self._counts: dict[str, int] = {}
        self._lock = Lock()
        self.calls: list[dict[str, Any]] = []

    def invoke_agent(
        self,
        agent_name: str,
        payload: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> object:
        with self._lock:
            if agent_name not in self._responses:
                raise MockAgentClientError(
                    f"No mock response configured for agent: {agent_name}"
                )
            self._counts[agent_name] = self._counts.get(agent_name, 0) + 1
            self.calls.append(
                {
                    "agent_name": agent_name,
                    "payload": deepcopy(payload),
                    "system_prompt": system_prompt,
                    "invocation": self._counts[agent_name],
                }
            )
            scripted = self._responses[agent_name]

        response = (
            scripted(deepcopy(payload), system_prompt)
            if callable(scripted)
            else deepcopy(scripted)
        )
        if not isinstance(response, dict):
            raise MockAgentClientError(
                f"Mock response for {agent_name} must be an object."
            )
        return response


__all__ = ["MockAgentClient", "MockAgentClientError"]
