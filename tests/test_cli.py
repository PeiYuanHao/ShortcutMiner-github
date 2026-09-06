"""CLI tests call Python only; no Docker daemon or network is required."""

import json
from pathlib import Path

import pytest

import shortcutminer.cli as cli
from shortcutminer.storage.run_store import load_run_record


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "resume" in capsys.readouterr().out


def test_cli_mock_and_resume_in_new_client(tmp_path: Path, capsys):
    root = str(tmp_path / "runs")
    assert cli.main(["mock", "--run-id", "cli-failure", "--runs-root", root,
                     "--fail-at", "summarize"]) == 1
    assert load_run_record(root, "cli-failure").status == "FAILED"
    assert "Planned mock failure" in capsys.readouterr().err
    assert cli.main(["resume", "cli-failure", "--runs-root", root,
                     "--runtime", "mock"]) == 0
    output = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert output["mock_agents_invoked"] == ["summarize"]
    assert load_run_record(root, "cli-failure").attempt_count == 2
    assert cli.main(["resume", "cli-failure", "--runs-root", root,
                     "--runtime", "mock"]) == 0
    output = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert output["mock_agents_invoked"] == []


def test_cli_mock_invokes_all_five_agents_without_credentials(tmp_path: Path, capsys,
                                                           monkeypatch):
    def prohibit_live_client(*args, **kwargs):
        raise AssertionError("Mock must not construct a live client")
    monkeypatch.setattr(cli, "OpenAICompatibleClient", prohibit_live_client)
    assert cli.main(["mock", "--runs-root", str(tmp_path), "--run-id", "cli-mock"]) == 0
    output = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert set(output["mock_agents_invoked"]) == {
        "identify", "executive", "quantitative", "qualitative", "summarize"}


def test_cli_passes_model_configuration_without_calling_api(tmp_path: Path, monkeypatch):
    from shortcutminer.runtime.demo import build_demo_request
    request = build_demo_request(tmp_path / "runs", "api-config-test")
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    model_path = tmp_path / "model.json"
    settings = {"provider": "openai_compatible", "model": "test-model",
                "base_url": "https://endpoint.example.test/v1",
                "api_key_env": "SHORTCUTMINER_TEST_KEY", "temperature": 0.0,
                "timeout": 12.0, "max_tokens": 256}
    model_path.write_text(json.dumps(settings), encoding="utf-8")
    captured = {}
    def client(config):
        captured["config"] = config
        return object()
    def run(request, *, agent_client, tool_runtimes):
        captured["request"] = request
        captured["tool_runtimes"] = tool_runtimes
        from types import SimpleNamespace
        return SimpleNamespace(run_id=request.run_id, status="COMPLETED")
    monkeypatch.setattr(cli, "OpenAICompatibleClient", client)
    monkeypatch.setattr(cli, "run", run)
    assert cli.main(["run", "--config", str(request_path),
                     "--model-config", str(model_path)]) == 0
    assert captured["config"].model_dump() == settings
    assert captured["request"].model_runtime == captured["config"]
    assert captured["tool_runtimes"]["quantitative"].agent_name == "quantitative"


def test_mock_rejects_unsafe_run_id_before_creating_data(tmp_path: Path):
    assert cli.main(["mock", "--run-id", "../escape", "--runs-root", str(tmp_path)]) == 1
    assert list(tmp_path.iterdir()) == []


def test_api_resume_requires_explicit_model_config(tmp_path: Path):
    assert cli.main(["resume", "missing", "--runs-root", str(tmp_path),
                     "--runtime", "openai-compatible"]) == 1
    assert list(tmp_path.iterdir()) == []
