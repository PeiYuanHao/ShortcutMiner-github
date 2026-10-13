"""Monitoring replays real logging envelopes without controlling execution."""

import io
import json
from datetime import datetime, timedelta, timezone
from threading import Event
from types import SimpleNamespace

import pytest
from rich.console import Console

from shortcutminer import cli
from shortcutminer.monitoring.event_listener import EventListener, parse_event
from shortcutminer.monitoring.progress import AGENTS, ProgressState
from shortcutminer.monitoring.renderer import ProgressMonitor, render_dashboard
from shortcutminer.protocol.model_runtime import ModelInvocationRecord
from shortcutminer.storage.events import append_event


def event(name, stage=None, details=None, *, second=0, run_id="monitor"):
    return parse_event(json.dumps({
        "timestamp": (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=second)).isoformat(),
        "run_id": run_id, "event": name, "stage": stage, "details": details or {},
    }))


def invocation(identifier, kind="text_agent", *, available=True, status="completed"):
    return ModelInvocationRecord(
        invocation_id=identifier, run_id="monitor",
        agent_name="semantic_visual" if kind == "semantic_vlm" else "identify",
        runtime_kind=kind, provider="test", model="test", attempt=1,
        request_hash="hash", system_prompt_hash="hash", payload_hash="hash",
        started_at="2026-01-01T00:00:00+00:00", completed_at="2026-01-01T00:00:01+00:00",
        latency_ms=1000.0, status=status, failure_category=None if status == "completed" else status,
        retryable=False, schema_valid=True if status == "completed" else None,
        usage={"available": True, "prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
        if available else {"available": False},
    )


def test_parse_real_append_event_envelope(tmp_path):
    path = append_event(tmp_path, "monitor", "IDENTIFY_STARTED", stage="identify", details={"future": "field"})
    record = parse_event(path.read_bytes())
    assert record.event == "IDENTIFY_STARTED"
    assert record.stage == "identify"
    assert record.timestamp.tzinfo is not None
    assert record.details == {"future": "field"}
    assert event("FUTURE_EVENT").event == "FUTURE_EVENT"


@pytest.mark.parametrize("line", [
    "{", "[]", "null", b"\xff", '{}',
    '{"timestamp":"bad","run_id":"monitor","event":"X"}',
    '{"timestamp":"2026-01-01","run_id":"monitor","event":"X"}',
    '{"timestamp":"2026-01-01T00:00:00Z","run_id":"monitor","event":"X","details":[]}',
    '{"timestamp":"2026-01-01T00:00:00Z","run_id":"monitor","event":"X","stage":5}',
])
def test_malformed_events_are_rejected(line):
    with pytest.raises(ValueError, match="invalid progress event"):
        parse_event(line)


def test_progress_reconstruction_parallel_tasks_and_terminal_failure():
    records = [event("RUN_STARTED")]
    for agent in AGENTS[:2]:
        records.extend([event(f"{agent.upper()}_STARTED", agent), event(f"{agent.upper()}_COMPLETED", agent)])
    records.extend([
        event("QUANTITATIVE_STARTED", "quantitative", {"task_id": "one"}),
        event("QUANTITATIVE_STARTED", "quantitative", {"task_id": "two"}),
        event("QUANTITATIVE_COMPLETED", "quantitative", {"task_id": "one"}),
        event("QUALITATIVE_STARTED", "qualitative"),
        event("QUALITATIVE_COMPLETED", "qualitative", {"candidate_id": "candidate"}),
        event("TOOL_STARTED", "tool:quantitative", {"tool_call_id": "tool-one", "tool": "inspect"}),
    ])
    state = ProgressState.reconstruct("monitor", records)
    assert state.agents["quantitative"] == "running"
    assert state.agents["qualitative"] == "completed"
    assert state.completed_agents == 3
    assert state.progress == 0.25  # Identify and Executive milestones; no planned task count yet.
    assert state.current_stage == "quantitative"
    state.apply(event("AGENT_FAILED", "quantitative", {"task_id": "two"}))
    state.apply(event("RUN_FAILED", "run", second=9))
    assert state.status == "failed"
    assert state.agents["quantitative"] == "failed"
    assert state.tools["quantitative", "tool-one"].status == "failed"
    assert state.elapsed_seconds(datetime(2027, 1, 1, tzinfo=timezone.utc)) == 9


def test_replay_preserves_completed_agents_on_resume():
    state = ProgressState.reconstruct("monitor", [
        event("RUN_STARTED"), event("IDENTIFY_STARTED", "identify"),
        event("IDENTIFY_COMPLETED", "identify"),
        event("EXECUTIVE_STARTED", "executive", second=2),
    ], interrupted=True)
    assert state.status == "interrupted"
    assert state.agents["identify"] == "completed"
    assert state.agents["executive"] == "interrupted"
    assert state.elapsed_seconds() == 2
    state.apply(event("RUN_RESUMED", "run", second=20))
    state.apply(event("EXECUTIVE_STARTED", "executive", second=21))
    state.apply(event("EXECUTIVE_COMPLETED", "executive", second=22))
    assert state.agents["executive"] == "completed"
    assert state.agents["identify"] == "completed"
    assert state.status == "running"
    assert state.elapsed_seconds(datetime(2026, 1, 1, 0, 0, 23, tzinfo=timezone.utc)) == 23
    state.apply(event("RUN_COMPLETED", "run", second=24))
    assert state.progress == 1.0
    assert state.elapsed_seconds() == 24


def test_listener_handles_partial_utf8_and_corrupt_lines_without_duplicates(tmp_path):
    listener = EventListener(tmp_path, "monitor")
    assert listener.poll().status == "waiting"
    path = append_event(tmp_path, "monitor", "RUN_STARTED", details={"message": "中文"})
    content = path.read_bytes()
    split = content.index("中文".encode()) + 1
    path.write_bytes(content[:split])
    assert listener.poll().status == "waiting"
    with path.open("ab") as stream:
        stream.write(content[split:] + b"bad json\n")
    assert listener.poll().status == "running"
    assert listener.state.ignored_records == 1
    assert listener.poll().ignored_records == 1
    append_event(tmp_path, "other", "RUN_COMPLETED")
    # Foreign run IDs must not influence this monitor even if copied here.
    with path.open("ab") as stream:
        stream.write((tmp_path / "other" / "events.jsonl").read_bytes())
    assert listener.poll().status == "running"
    path.write_bytes(b"")
    assert listener.poll().status == "waiting"


def test_provider_telemetry_retries_and_unavailable_usage(tmp_path):
    listener = EventListener(tmp_path, "monitor")
    for kind in ("text_agent", "text_agent", "semantic_vlm"):
        append_event(tmp_path, "monitor", "MODEL_INVOCATION_STARTED", stage="identify", details={"runtime_kind": kind})
    records = [invocation("one"), invocation("two", status="timed_out", available=False), invocation("vlm", "semantic_vlm")]
    directory = tmp_path / "monitor" / "model_calls"
    directory.mkdir()
    for record in records:
        (directory / f"{record.invocation_id}.json").write_text(record.model_dump_json(), encoding="utf-8")
    (directory / "telemetry.json").write_text("{}", encoding="utf-8")
    (directory / "runtime_snapshot-test.json").write_text("{}", encoding="utf-8")
    (directory / "broken.json").write_text("{", encoding="utf-8")
    state = listener.poll()
    assert state.provider_calls == {"text_agent": 2, "semantic_vlm": 1}
    assert state.total_tokens is None
    assert state.latency_ms == 3000
    assert listener.poll().latency_ms == 3000
    state.update_telemetry([records[0], records[0], records[2], invocation("budget", status="budget_exceeded", available=False)])
    assert state.total_tokens == 20
    assert state.prompt_tokens == 8
    assert state.completion_tokens == 12
    assert state.latency_ms == 2000


def test_tools_and_markup_are_rendered_as_plain_text():
    state = ProgressState.reconstruct("monitor", [
        event("TOOL_STARTED", "tool:identify", {"tool_call_id": "one", "tool": "[red]inspect"}),
        event("TOOL_REJECTED", "tool:identify", {"tool_call_id": "one", "tool": "[red]inspect"}),
        event("TOOL_COMPLETED", "tool:quantitative", {"tool_call_id": "two", "tool": "measure", "duration_ms": 12}),
    ])
    output = io.StringIO()
    Console(file=output, width=120).print(render_dashboard(state))
    rendered = output.getvalue()
    for text in ("Identify", "Executive", "Quantitative", "Qualitative", "Summarize", "Text API calls", "VLM calls", "N/A", "Elapsed", "rejected", "12ms", "[red]inspect"):
        assert text in rendered


def test_cli_interrupted_run_recovery_with_real_events(tmp_path, capsys):
    args = ["mock", "--runs-root", str(tmp_path), "--run-id", "monitor", "--progress", "--fail-at", "summarize"]
    assert cli.main(args) == 1
    assert "ShortcutMiner progress" in capsys.readouterr().err
    listener = EventListener(tmp_path, "monitor")
    state = listener.poll()
    assert state.status == "failed"
    assert state.completed_agents == 4
    assert state.agents["summarize"] == "failed"
    assert cli.main(["resume", "monitor", "--runs-root", str(tmp_path), "--runtime", "mock", "--progress"]) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out.splitlines()[-1])
    assert summary["mock_agents_invoked"] == ["summarize"]
    assert listener.poll().status == "completed"
    recovered = EventListener(tmp_path, "monitor").poll()
    assert recovered.agents == dict.fromkeys(AGENTS, "completed")
    assert recovered.progress == 1.0
    assert recovered.started_at == state.started_at


def test_monitor_background_failure_cannot_change_execution(tmp_path, monkeypatch):
    monitor = ProgressMonitor(tmp_path, "monitor", console=Console(file=io.StringIO()))
    monkeypatch.setattr(monitor.listener, "poll", lambda: (_ for _ in ()).throw(OSError("unavailable")))
    with monitor:
        pass
    with pytest.raises(RuntimeError, match="execution failure"):
        with monitor:
            raise RuntimeError("execution failure")


def test_live_monitor_follows_appended_events_and_closes_thread(tmp_path, monkeypatch):
    ready = Event()
    monitor = ProgressMonitor(tmp_path, "monitor", console=Console(file=io.StringIO()), interval=0.01)
    original_poll = monitor.listener.poll

    def observe():
        state = original_poll()
        if state.agents["identify"] == "running":
            ready.set()
        return state

    monkeypatch.setattr(monitor.listener, "poll", observe)
    with monitor:
        append_event(tmp_path, "monitor", "RUN_STARTED")
        append_event(tmp_path, "monitor", "IDENTIFY_STARTED", stage="identify")
        assert ready.wait(3), "background monitor did not consume events"
    assert not monitor._thread.is_alive()
    assert monitor.listener.state.status == "interrupted"
    assert monitor.listener.state.agents["identify"] == "interrupted"
    # Monitoring must never append its own progress or interruption events.
    path = tmp_path / "monitor" / "events.jsonl"
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_cli_api_run_progress_keeps_json_output_and_run_arguments(tmp_path, monkeypatch, capsys):
    from shortcutminer.protocol.model import OpenAICompatibleModelConfig
    from shortcutminer.runtime.demo import build_demo_request

    model = OpenAICompatibleModelConfig(model="test", base_url="https://example.test/v1", api_key_env="TEST_KEY")
    request = build_demo_request(tmp_path, "monitor").model_copy(update={"model_runtime": model})
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    client, runtimes = object(), object()
    monkeypatch.setattr(cli, "_text_client_arguments", lambda *args, **kwargs: {"agent_client": client})
    monkeypatch.setattr(cli, "build_local_tool_runtimes", lambda **kwargs: runtimes)

    def execute(actual_request, **kwargs):
        assert actual_request == request
        assert kwargs == {"agent_client": client, "tool_runtimes": runtimes}
        append_event(tmp_path, "monitor", "RUN_STARTED")
        append_event(tmp_path, "monitor", "RUN_COMPLETED")
        return SimpleNamespace(run_id="monitor", status="COMPLETED")

    monkeypatch.setattr(cli, "run", execute)
    assert cli.main(["run", "--config", str(request_path), "--progress"]) == 0
    output = capsys.readouterr()
    assert [json.loads(line)["event"] for line in output.out.splitlines()] == ["started", "completed"]
    assert "ShortcutMiner progress" in output.err
    assert "100%" in output.err
