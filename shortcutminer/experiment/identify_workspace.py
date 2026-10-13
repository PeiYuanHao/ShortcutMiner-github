"""Compatibility entry points for the unified agent workspace loader."""

from pathlib import Path

from shortcutminer.experiment.agent_workspace import (
    WORKSPACE_FILES,
    AgentWorkspaceError,
    default_agent_workspace_dir,
    load_agent_workspace_prompt,
)
from shortcutminer.runtime.identify_output_contract import identify_output_contract


IDENTIFY_WORKSPACE_FILES = WORKSPACE_FILES


class IdentifyWorkspaceError(RuntimeError):
    """Raised when the Identify workspace cannot form a complete prompt."""


def default_identify_workspace_dir() -> Path:
    """Return the repository's preserved Identify workspace directory."""
    return default_agent_workspace_dir("identify")


def load_identify_workspace_prompt(
    workspace_dir: str | Path | None = None,
) -> str:
    """Load all Identify workspace documents in their established order."""
    try:
        workspace = load_agent_workspace_prompt("identify", workspace_dir)
        return workspace + "\n\n" + identify_output_contract()
    except AgentWorkspaceError as exc:
        raise IdentifyWorkspaceError(str(exc)) from exc
