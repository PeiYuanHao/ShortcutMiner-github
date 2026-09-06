"""Provider-independent tools managed by ShortcutMiner."""

from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.errors import (
    ToolExecutionError,
    ToolInputValidationError,
    ToolNotFound,
    ToolPermissionError,
    ToolTimeoutError,
)
from shortcutminer.tooling.protocol import (
    AgentName,
    ToolCall,
    ToolErrorInfo,
    ToolResult,
    ToolSpec,
)
from shortcutminer.tooling.registry import ToolRegistry
from shortcutminer.tooling.runtime import ToolBudget, ToolRuntime

__all__ = [
    "AgentName",
    "ToolBudget",
    "ToolCall",
    "ToolErrorInfo",
    "ToolExecutionError",
    "ToolInputValidationError",
    "ToolNotFound",
    "ToolPermissionError",
    "ToolRegistry",
    "ToolResult",
    "ToolRuntime",
    "ToolSpec",
    "ToolTimeoutError",
    "build_default_tool_registry",
]
