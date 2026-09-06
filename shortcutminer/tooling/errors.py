"""Explicit failure classes for the ShortcutMiner tool boundary."""


class ToolRuntimeError(RuntimeError):
    """Base class for tool protocol and execution failures."""


class ToolNotFound(ToolRuntimeError):
    """Raised when a call names an unregistered tool."""


class ToolPermissionError(ToolRuntimeError):
    """Raised when an Agent is not permitted to invoke a tool."""


class ToolInputValidationError(ToolRuntimeError):
    """Raised when ToolCall arguments fail the tool input schema."""


class ToolExecutionError(ToolRuntimeError):
    """Represents a failure inside an otherwise valid tool invocation."""


class ToolTimeoutError(ToolExecutionError):
    """Represents a tool invocation that exceeded its time budget."""


class ToolBudgetExceeded(ToolRuntimeError):
    """Raised before execution when an invocation exhausted its tool budget."""
