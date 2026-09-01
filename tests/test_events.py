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
            "message": "runtime unavailable",
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


@pytest.mark.parametrize(
    "run_id",
    [
        "..",
        "../escape",
        "nested/run",
        r"nested\run",
        ".hidden",
        "CON",
    ],
)
def test_unsafe_run_id_rejected(
    tmp_path: Path,
    run_id: str,
):
    with pytest.raises(
        EventLogError,
        match="run_id",
    ):
        append_event(
            runs_root=tmp_path,
            run_id=run_id,
            event="IDENTIFY_STARTED",
        )


def test_absolute_run_id_rejected(
    tmp_path: Path,
):
    with pytest.raises(
        EventLogError,
        match="run_id",
    ):
        append_event(
            runs_root=tmp_path / "runs",
            run_id=str(
                (tmp_path / "escape").resolve()
            ),
            event="IDENTIFY_STARTED",
        )


def test_repeated_dots_inside_run_id_are_allowed(
    tmp_path: Path,
):
    path = append_event(
        runs_root=tmp_path,
        run_id="run..variant",
        event="IDENTIFY_STARTED",
    )

    assert path.parent.name == "run..variant"


def test_run_directory_cannot_escape_through_symlink(
    tmp_path: Path,
):
    runs_root = tmp_path / "runs"
    outside = tmp_path / "outside"
    runs_root.mkdir()
    outside.mkdir()

    try:
        (runs_root / "linked_run").symlink_to(
            outside,
            target_is_directory=True,
        )

    except OSError:
        pytest.skip(
            "Directory symlinks are not available."
        )

    with pytest.raises(
        EventLogError,
        match="runs_root",
    ):
        append_event(
            runs_root=runs_root,
            run_id="linked_run",
            event="IDENTIFY_STARTED",
        )


def test_directory_creation_failure_is_wrapped(
    tmp_path: Path,
):
    runs_root = tmp_path / "runs"
    runs_root.write_text(
        "not a directory",
        encoding="utf-8",
    )

    with pytest.raises(
        EventLogError,
        match="Failed to append event",
    ) as exc_info:
        append_event(
            runs_root=runs_root,
            run_id="test_run",
            event="IDENTIFY_STARTED",
        )

    assert isinstance(
        exc_info.value.__cause__,
        OSError,
    )


def test_event_write_failure_is_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_open(
        *_args,
        **_kwargs,
    ):
        raise OSError(
            "write unavailable"
        )

    monkeypatch.setattr(
        Path,
        "open",
        fail_open,
    )

    with pytest.raises(
        EventLogError,
        match="Failed to append event",
    ) as exc_info:
        append_event(
            runs_root=tmp_path / "runs",
            run_id="test_run",
            event="IDENTIFY_STARTED",
        )

    assert isinstance(
        exc_info.value.__cause__,
        OSError,
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


def test_non_finite_event_detail_is_rejected_without_partial_line(
    tmp_path: Path,
):
    with pytest.raises(
        EventLogError,
        match="Failed to append event",
    ):
        append_event(
            runs_root=tmp_path,
            run_id="test_run",
            event="AGENT_FAILED",
            details={"metric": float("nan")},
        )

    event_path = tmp_path / "test_run" / "events.jsonl"
    assert not event_path.exists()
