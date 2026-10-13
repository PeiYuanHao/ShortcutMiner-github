import io
from rich.console import Console
from shortcutminer.monitoring.event_listener import EventListener
from shortcutminer.monitoring.renderer import ProgressMonitor, render_dashboard, projection
from shortcutminer.storage.events import append_event
from test_progress_state import public_artifacts


def test_candidates_tasks_steps_and_screening_reasons_visible(tmp_path):
    public_artifacts(tmp_path)
    state = EventListener(tmp_path, "monitor").poll()
    output = io.StringIO()
    Console(file=output, width=160).print(render_dashboard(state))
    text = output.getvalue()
    for fragment in ("collection site", "device bias", "screened_out", "weak_signal", "TASK-C001-QT",
                     "statistical_check", "model_evaluation", "QSTEP-a", "skipped", "N/A", "Early termination active"):
        assert fragment in text


def test_non_tty_plain_lines_no_ansi_and_no_duplicate_unchanged_updates(tmp_path):
    public_artifacts(tmp_path)
    output = io.StringIO()
    monitor = ProgressMonitor(tmp_path, "monitor", console=Console(file=output, width=200, force_terminal=False))
    monitor._refresh()
    before = output.getvalue()
    monitor._refresh()
    assert output.getvalue() == before
    assert "[ShortcutMiner]" in before and "\x1b" not in before
    assert "promoted=1 screened_out=1" in before


def test_no_secrets_raw_provider_payload_or_error_messages_rendered(tmp_path):
    public_artifacts(tmp_path)
    append_event(tmp_path, "monitor", "RUN_FAILED", details={"message": "Authorization: Bearer SECRET_API_KEY",
        "response": {"reasoning_content": "PRIVATE_REASONING"}, "base64": "LONG_BASE64"})
    append_event(tmp_path, "monitor", "TOOL_FAILED", stage="tool:quantitative", details={
        "tool": "run_experiment", "tool_call_id": "call", "arguments": {"key": "RAW_ARGUMENT"},
        "error": {"message": "SECRET_API_KEY"}, "output": {"payload": "RAW_PROVIDER"}})
    state = EventListener(tmp_path, "monitor").poll()
    output = io.StringIO()
    Console(file=output, width=180).print(render_dashboard(state))
    text = output.getvalue() + "\n".join(projection(state).values())
    for secret in ("SECRET_API_KEY", "PRIVATE_REASONING", "LONG_BASE64", "RAW_ARGUMENT", "RAW_PROVIDER", "Authorization"):
        assert secret not in text
    assert "Artifacts preserved" in text


def test_tty_selects_live_dashboard_and_keeps_failure_snapshot(tmp_path):
    output = io.StringIO()
    monitor = ProgressMonitor(tmp_path, "monitor", console=Console(file=output, force_terminal=True, width=180))
    with monitor:
        assert monitor._live is not None
        append_event(tmp_path, "monitor", "RUN_STARTED")
        append_event(tmp_path, "monitor", "RUN_FAILED")
    assert monitor.listener.state.status == "failed"
    assert "ShortcutMiner Audit" in output.getvalue()
