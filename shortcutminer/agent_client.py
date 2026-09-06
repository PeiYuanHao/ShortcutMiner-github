"""Runtime-agnostic client contract for ShortcutMiner agent calls."""

from inspect import Parameter, signature
from typing import Any, Protocol


class AgentClient(Protocol):
    """Minimal interface implemented by a concrete model or agent backend."""

    def invoke_agent(
        self,
        agent_name: str,
        payload: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> object:
        """Invoke one named ShortcutMiner agent with a structured payload."""
        ...


def invoke_agent_client(
    client: AgentClient,
    agent_name: str,
    payload: dict[str, Any],
    *,
    system_prompt: str | None = None,
) -> object:
    """Invoke a client while accepting small two-argument test doubles."""
    parameters = signature(client.invoke_agent).parameters.values()
    accepts_system_prompt = any(
        parameter.name == "system_prompt"
        or parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_system_prompt:
        return client.invoke_agent(
            agent_name,
            payload,
            system_prompt=system_prompt,
        )
    return client.invoke_agent(agent_name, payload)
