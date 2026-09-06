"""Load preserved agent workspace documents as explicit instructions."""

from pathlib import Path


REQUIRED_WORKSPACE_FILES = (
    "AGENTS.md",
    "IDENTITY.md",
    "SOUL.md",
)

OPTIONAL_WORKSPACE_FILES = (
    "USER.md",
    "TOOLS.md",
    "MEMORY.md",
    "HEARTBEAT.md",
)

WORKSPACE_FILES = REQUIRED_WORKSPACE_FILES + OPTIONAL_WORKSPACE_FILES

AGENT_NAMES = frozenset(
    {"identify", "executive", "quantitative", "qualitative", "summarize"}
)


class AgentWorkspaceError(RuntimeError):
    """Raised when preserved workspace instructions cannot be loaded."""


def default_agent_workspace_dir(agent_name: str) -> Path:
    """Return the repository workspace directory for one known agent."""
    if agent_name not in AGENT_NAMES:
        raise AgentWorkspaceError(f"Unknown ShortcutMiner agent: {agent_name}")
    return (
        Path(__file__).resolve().parents[2]
        / "agents"
        / agent_name
        / "workspace"
    )


def load_agent_workspace_prompt(
    agent_name: str,
    workspace_dir: str | Path | None = None,
) -> str:
    """Combine workspace files in the established, deterministic order.

    The role-defining documents are required. Optional policy and context
    documents are included when present, which lets a deliberately minimal
    workspace remain valid without relying on directory enumeration order.
    """
    root = (
        Path(workspace_dir)
        if workspace_dir is not None
        else default_agent_workspace_dir(agent_name)
    )
    contents: list[str] = []
    for filename in REQUIRED_WORKSPACE_FILES:
        path = root / filename
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise AgentWorkspaceError(
                f"Unable to load {agent_name} workspace document {path}: {exc}"
            ) from exc
        if not content.strip():
            raise AgentWorkspaceError(
                f"{agent_name} workspace document is empty: {path}"
            )
        contents.append(content)
    for filename in OPTIONAL_WORKSPACE_FILES:
        path = root / filename
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError) as exc:
            raise AgentWorkspaceError(
                f"Unable to load {agent_name} workspace document {path}: {exc}"
            ) from exc
        if content.strip():
            contents.append(content)
    return "\n\n".join(contents)
