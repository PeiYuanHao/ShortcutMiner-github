import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLogError(RuntimeError):
    """Raised when an experiment event cannot be recorded."""


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

    if not run_id.strip():
        raise EventLogError(
            "run_id must not be empty."
        )

    if not event.strip():
        raise EventLogError(
            "event must not be empty."
        )

    run_dir = (
        Path(runs_root)
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    event_path = (
        run_dir
        / "events.jsonl"
    )

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
        with event_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")

    except OSError as exc:
        raise EventLogError(
            f"Failed to append event: {event_path}"
        ) from exc

    return event_path