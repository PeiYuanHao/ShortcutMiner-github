"""Offline regressions for the Phase 16 Docker credential boundary."""

import json
from pathlib import Path
import subprocess

import pytest
import yaml
from pydantic import ValidationError

import shortcutminer.cli as cli
from shortcutminer.evaluation.pilot.credentials import credential_readiness
from shortcutminer.evaluation.pilot.protocol import PilotConfig


ROOT = Path(__file__).resolve().parents[1]
PILOT_CONFIG = ROOT / "configs" / "experiments" / "real_blind_pilot.example.yaml"
host_repository_only = pytest.mark.skipif(
    not (ROOT / "docker-compose.yml").is_file(),
    reason="source-control and Compose files are intentionally absent from the runtime image",
)


def _config() -> PilotConfig:
    return PilotConfig.model_validate(yaml.safe_load(PILOT_CONFIG.read_text(encoding="utf-8")))


@host_repository_only
def test_local_phase16_secret_file_is_gitignored_but_public_template_is_tracked():
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", ".env.phase16"], cwd=ROOT,
        check=False,
    )
    public = subprocess.run(
        ["git", "check-ignore", "--quiet", ".env.phase16.example"], cwd=ROOT,
        check=False,
    )
    assert ignored.returncode == 0
    assert public.returncode != 0


@host_repository_only
def test_default_compose_is_offline_and_does_not_load_phase16_secrets():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    service = compose["services"]["shortcutminer"]
    assert service["network_mode"] == "${SHORTCUTMINER_NETWORK_MODE:-none}"
    assert "env_file" not in service


@host_repository_only
def test_api_override_is_minimal_explicit_bridge_and_local_env_file():
    override = yaml.safe_load((ROOT / "docker-compose.api.yml").read_text(encoding="utf-8"))
    assert override == {
        "services": {
            "shortcutminer": {
                "env_file": [".env.phase16"],
                "network_mode": "bridge",
            }
        }
    }


@host_repository_only
def test_secret_file_is_excluded_from_build_context_and_dockerfile_copy_allowlist():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in dockerignore
    assert ".env.*" in dockerignore
    dockerfile = (ROOT / "infrastructure" / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert ".env.phase16" not in dockerfile
    copy_lines = [line.strip() for line in dockerfile.splitlines() if line.strip().startswith("COPY ")]
    assert all(not line.startswith("COPY . ") and not line.startswith("COPY ./ ") for line in copy_lines)


@host_repository_only
def test_committed_compose_and_runtime_configs_contain_no_raw_credential_field():
    for path in (ROOT / "configs").rglob("*.yaml"):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        serialized = json.dumps(value, sort_keys=True).casefold()
        assert '"api_key":' not in serialized
        assert '"authorization":' not in serialized
        assert '"bearer":' not in serialized
    for path in (ROOT / "docker-compose.yml", ROOT / "docker-compose.api.yml"):
        text = path.read_text(encoding="utf-8")
        assert "SHORTCUTMINER_TEXT_API_KEY=" not in text
        assert "SHORTCUTMINER_VLM_API_KEY=" not in text


def test_pilot_config_and_snapshot_contract_reject_raw_credentials():
    value = _config().model_dump(mode="json")
    value["text_runtime"]["api_key"] = "offline-test-marker"
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(value)


def test_readiness_reports_only_set_or_missing(monkeypatch, capsys):
    config = _config()
    marker = "offline-test-marker-not-for-output"
    monkeypatch.setenv("SHORTCUTMINER_TEXT_API_KEY", marker)
    monkeypatch.delenv("SHORTCUTMINER_VLM_API_KEY", raising=False)
    assert credential_readiness(config) == {
        "SHORTCUTMINER_TEXT_API_KEY": "SET",
        "SHORTCUTMINER_VLM_API_KEY": "MISSING",
    }
    assert cli.main(["pilot", "credentials", "--config", str(PILOT_CONFIG)]) == 0
    output = capsys.readouterr().out
    assert marker not in output
    assert '"SHORTCUTMINER_TEXT_API_KEY": "SET"' in output
    assert '"SHORTCUTMINER_VLM_API_KEY": "MISSING"' in output


@host_repository_only
def test_public_environment_templates_contain_only_empty_values():
    phase16 = (ROOT / ".env.phase16.example").read_text(encoding="utf-8")
    assignments = [line for line in phase16.splitlines() if line and not line.startswith("#")]
    assert assignments == [
        "SHORTCUTMINER_TEXT_API_KEY=",
        "SHORTCUTMINER_VLM_API_KEY=",
    ]
