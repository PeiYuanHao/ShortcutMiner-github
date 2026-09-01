"""Runtime-agnostic client contract for ShortcutMiner agent calls."""

from typing import Any, Protocol


class AgentClient(Protocol):
    """Minimal interface implemented by a concrete model or agent backend."""

    def invoke_agent(
        self,
        agent_name: str,
        payload: dict[str, Any],
    ) -> object:
        """Invoke one named ShortcutMiner agent with a structured payload."""
        ...
