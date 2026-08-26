import json
from pathlib import Path

import pytest

from shortcutminer.storage.events import (
    EventLogError,
    append_event,
)


def test_append_event(
    tmp_path: Path,
):
    path = append_event(
        runs_root=tmp_path,
        run_id="test_run",
        event="IDENTIFY_STARTED",
        stage="identify",
    )

    assert path.exists()

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1

    record = json.loads(
        lines[0]
    )

    assert record["run_id"] == "test_run"
    assert record["event"] == "IDENTIFY_STARTED"
    assert record["stage"] == "identify"
    assert "timestamp" in record


def test_multiple_events(
    tmp_path: Path,
):
    append_event(
        runs_root=tmp_path,
        run_id="test_run",
        event="IDENTIFY_STARTED",
        stage="identify",
    )

    append_event(
        runs_root=tmp_path,
        run_id="test_run",
        event="IDENTIFY_COMPLETED",
        stage="identify",
    )

    path = (
        tmp_path
        / "test_run"
        / "events.jsonl"
    )

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert (
        first["event"]
        == "IDENTIFY_STARTED"
    )

    assert (
        second["event"]
        == "IDENTIFY_COMPLETED"
    )


def test_event_details(
    tmp_path: Path,
):
    path = append_event(
        runs_root=tmp_path,
        run_id="test_run",
        event="AGENT_FAILED",
        stage="identify",
        details={
            "error_type": "RuntimeError",
            "message": "gateway unavailable",
        },
    )

    record = json.loads(
        path.read_text(
            encoding="utf-8"
        ).strip()
    )

    assert (
        record["details"]["error_type"]
        == "RuntimeError"
    )


def test_empty_run_id_rejected(
    tmp_path: Path,
):
    with pytest.raises(
        EventLogError,
        match="run_id",
    ):
        append_event(
            runs_root=tmp_path,
            run_id="",
            event="IDENTIFY_STARTED",
        )


def test_empty_event_rejected(
    tmp_path: Path,
):
    with pytest.raises(
        EventLogError,
        match="event",
    ):
        append_event(
            runs_root=tmp_path,
            run_id="test_run",
            event="",
        )