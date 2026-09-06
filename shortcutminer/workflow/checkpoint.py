"""Thin integration with LangGraph's official SQLite checkpointer."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from shortcutminer.storage.paths import get_run_dir


class CheckpointBackendUnavailable(RuntimeError):
    """Raised when the optional official SQLite backend is not installed."""


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be opened or located."""


def checkpoint_path(runs_root: str | Path, run_id: str) -> Path:
    return get_run_dir(runs_root, run_id) / "checkpoints.sqlite"


def checkpoint_config(run_id: str) -> dict[str, dict[str, str]]:
    """Use the stable run ID as LangGraph's durable thread identifier."""
    return {"configurable": {"thread_id": run_id}}


@contextmanager
def open_sqlite_checkpointer(path: str | Path) -> Iterator[Any]:
    """Open the local synchronous saver for one bounded graph invocation."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ModuleNotFoundError as exc:
        raise CheckpointBackendUnavailable(
            "SQLite checkpoint support requires langgraph-checkpoint-sqlite."
        ) from exc

    resolved = Path(path).resolve(strict=False)
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(resolved), check_same_thread=False)
    except (OSError, sqlite3.Error) as exc:
        raise CheckpointError(
            f"Unable to open checkpoint database: {resolved}"
        ) from exc

    try:
        serializer = JsonPlusSerializer(allowed_msgpack_modules=None)
        saver = SqliteSaver(connection, serde=serializer)
        saver.setup()
        yield saver
    except sqlite3.Error as exc:
        raise CheckpointError(
            f"SQLite checkpoint operation failed: {resolved}"
        ) from exc
    finally:
        connection.close()
