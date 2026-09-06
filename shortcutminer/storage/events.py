import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from shortcutminer.storage.paths import (
    RunPathError,
    get_run_dir,
)


class EventLogError(RuntimeError):
    """Raised when an experiment event cannot be recorded."""


_EVENT_LOCKS_GUARD = Lock()
_EVENT_LOCKS: dict[str, Lock] = {}


def _event_lock(path: Path) -> Lock:
    key = str(path.resolve(strict=False))
    with _EVENT_LOCKS_GUARD:
        return _EVENT_LOCKS.setdefault(key, Lock())


def append_event(
    runs_root: str | Path,
    run_id: str,
    event: str,
    *,
    stage: str | None = None,
    details: dict[str, Any] | None = None,
) -> Path:
    """
    Append one structured event to runs/<run_id>/events.jsonl.
    """

    if not isinstance(event, str) or not event.strip():
        raise EventLogError(
            "event must not be empty."
        )

    try:
        run_dir = get_run_dir(
            runs_root=runs_root,
            run_id=run_id,
        )

    except RunPathError as exc:
        raise EventLogError(
            str(exc)
        ) from exc

    event_path = run_dir / "events.jsonl"

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "run_id": run_id,
        "event": event,
    }

    if stage is not None:
        record["stage"] = stage

    if details:
        record["details"] = details

    try:
        serialized = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with _event_lock(event_path):
            with event_path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(f"{serialized}\n")

    except Exception as exc:
        raise EventLogError(
            f"Failed to append event: {event_path}"
        ) from exc

    return event_path
