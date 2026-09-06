"""Persistence for serializable ToolResult records."""

from pathlib import Path
from typing import TYPE_CHECKING

from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.storage.paths import RunPathError, get_run_dir

if TYPE_CHECKING:
    from shortcutminer.tooling.protocol import ToolResult


class ToolArtifactError(RuntimeError):
    """Raised when a tool trace cannot be persisted safely."""


def tool_result_reference(result: "ToolResult") -> str:
    return f"tools/{result.agent_name}/{result.tool_call_id}.json"


def save_tool_result(
    runs_root: str | Path,
    result: "ToolResult",
) -> Path:
    try:
        run_dir = get_run_dir(runs_root, result.run_id)
    except RunPathError as exc:
        raise ToolArtifactError(str(exc)) from exc
    path = run_dir / tool_result_reference(result)
    atomic_write_json(
        path,
        result.model_dump(mode="json"),
        error_type=ToolArtifactError,
    )
    return path
