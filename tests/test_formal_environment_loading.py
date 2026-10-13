"""Credential loading checks use dummy values and never make provider requests."""
import os
from types import SimpleNamespace

import pytest

from shortcutminer.runtime import formal_environment as environment


def enable(monkeypatch):
    monkeypatch.delenv("SHORTCUTMINER_DISABLE_FORMAL_ENV", raising=False)
    for name in environment.CREDENTIAL_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_bom_quotes_comments_and_context_cleanup(tmp_path, monkeypatch):
    enable(monkeypatch)
    path = tmp_path / ".env.formal"
    path.write_text('\ufeff# local\nSHORTCUTMINER_TEXT_API_KEY="dummy-text#literal" # comment\n'
                    "SHORTCUTMINER_VLM_API_KEY='dummy-vlm'\nSHORTCUTMINER_REAL_TEXT_TEST=1\n", encoding="utf-8")
    with environment.formal_credentials(env_file=path):
        assert os.environ[environment.CREDENTIAL_NAMES[0]] == "dummy-text#literal"
        assert os.environ[environment.CREDENTIAL_NAMES[1]] == "dummy-vlm"
        assert "SHORTCUTMINER_REAL_TEXT_TEST" not in os.environ
    assert all(name not in os.environ for name in environment.CREDENTIAL_NAMES)


def test_explicit_fake_credential_is_preserved_even_on_failure(tmp_path, monkeypatch):
    enable(monkeypatch)
    monkeypatch.setenv(environment.CREDENTIAL_NAMES[0], "dummy-explicit")
    path = tmp_path / ".env.formal"
    path.write_text("SHORTCUTMINER_TEXT_API_KEY=dummy-file\nSHORTCUTMINER_VLM_API_KEY=dummy-vlm\n")
    with pytest.raises(RuntimeError):
        with environment.formal_credentials(env_file=path):
            assert os.environ[environment.CREDENTIAL_NAMES[0]] == "dummy-explicit"
            raise RuntimeError("dummy failure")
    assert os.environ[environment.CREDENTIAL_NAMES[0]] == "dummy-explicit"
    assert environment.CREDENTIAL_NAMES[1] not in os.environ


def test_disabled_loading_does_not_even_read_the_file(tmp_path, monkeypatch):
    path = tmp_path / ".env.formal"
    path.write_text("invalid dummy content")
    with environment.formal_credentials(env_file=path):
        pass


def test_empty_process_value_uses_file_then_restores_empty(tmp_path, monkeypatch):
    enable(monkeypatch)
    name = environment.CREDENTIAL_NAMES[0]
    monkeypatch.setenv(name, "")
    path = tmp_path / ".env.formal"
    path.write_text(f"{name}=dummy-text\n")
    with environment.formal_credentials(env_file=path):
        assert os.environ[name] == "dummy-text"
    assert os.environ[name] == ""


def test_parser_error_does_not_expose_value_or_partially_load(tmp_path, monkeypatch):
    enable(monkeypatch)
    path = tmp_path / ".env.formal"
    path.write_text('SHORTCUTMINER_TEXT_API_KEY=dummy-text\nSHORTCUTMINER_VLM_API_KEY="dummy-unclosed\n')
    with pytest.raises(ValueError) as exc:
        with environment.formal_credentials(env_file=path):
            pass
    assert "dummy" not in str(exc.value)
    assert all(name not in os.environ for name in environment.CREDENTIAL_NAMES)


def test_cli_loads_repo_file_and_cleans_up_without_provider_calls(tmp_path, monkeypatch):
    from shortcutminer import cli
    enable(monkeypatch)
    path = tmp_path / ".env.formal"
    path.write_text("SHORTCUTMINER_TEXT_API_KEY=dummy-text\n")
    monkeypatch.setattr(environment, "FORMAL_ENV_FILE", path)
    def audit(_args):
        assert os.environ[environment.CREDENTIAL_NAMES[0]] == "dummy-text"
        return 0
    monkeypatch.setattr(cli, "_audit_command", audit)
    assert cli.main(["audit", "dummy-dataset"]) == 0
    assert environment.CREDENTIAL_NAMES[0] not in os.environ


@pytest.mark.parametrize("command,fields,expected", [
    ("mock", {}, False), ("audit", {}, True), ("run", {}, True),
    ("resume", {"runtime": "mock"}, False),
    ("resume", {"runtime": "openai-compatible"}, True),
    ("pilot", {"pilot_command": "dry-run"}, False),
    ("pilot", {"pilot_command": "credentials"}, True),
])
def test_only_api_commands_and_presence_checks_load_credentials(command, fields, expected):
    assert environment.command_uses_formal_credentials(SimpleNamespace(command=command, **fields)) is expected
