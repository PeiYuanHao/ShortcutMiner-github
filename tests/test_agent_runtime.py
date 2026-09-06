"""Phase 5 coverage for workspaces, runtime isolation, and model config."""

import ast
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from shortcutminer.experiment.agent_workspace import (
    AGENT_NAMES,
    OPTIONAL_WORKSPACE_FILES,
    REQUIRED_WORKSPACE_FILES,
    AgentWorkspaceError,
    load_agent_workspace_prompt,
)
from shortcutminer.experiment.loader import load_model_runtime_config
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.protocol.run import RunRequest
from shortcutminer.runtime.mock import MockAgentClient, MockAgentClientError
from shortcutminer.runtime.openai_compatible import (
    ModelRuntimeError,
    OpenAICompatibleClient,
)


REPOSITORY_ROOT = Path(__file__).parents[1]


def test_all_five_agent_workspaces_are_usable_and_deterministic():
    assert AGENT_NAMES == {
        "identify",
        "executive",
        "quantitative",
        "qualitative",
        "summarize",
    }
    retired_terms = ("open" + "claw", "sessions" + "_send", "gate" + "way")
    for agent_name in sorted(AGENT_NAMES):
        first = load_agent_workspace_prompt(agent_name)
        second = load_agent_workspace_prompt(agent_name)
        assert first == second
        assert len(first.strip()) > 100
        assert not any(term in first.lower() for term in retired_terms)


def test_workspace_loader_requires_core_files_but_allows_missing_optional_files(
    tmp_path: Path,
):
    for filename in REQUIRED_WORKSPACE_FILES:
        (tmp_path / filename).write_text(
            f"# {filename}\nRequired role instructions.",
            encoding="utf-8",
        )

    prompt = load_agent_workspace_prompt("identify", tmp_path)
    assert all(filename in prompt for filename in REQUIRED_WORKSPACE_FILES)
    assert all(filename not in prompt for filename in OPTIONAL_WORKSPACE_FILES)

    (tmp_path / REQUIRED_WORKSPACE_FILES[0]).unlink()
    with pytest.raises(AgentWorkspaceError, match="Unable to load"):
        load_agent_workspace_prompt("identify", tmp_path)


def test_mock_agent_client_is_repeatable_and_network_free():
    client = MockAgentClient({"identify": {"result": [1, 2, 3]}})

    first = client.invoke_agent("identify", {"run_id": "run-1"})
    first["result"].append(4)
    second = client.invoke_agent("identify", {"run_id": "run-1"})

    assert second == {"result": [1, 2, 3]}
    assert [call["invocation"] for call in client.calls] == [1, 2]
    with pytest.raises(MockAgentClientError, match="No mock response"):
        client.invoke_agent("summarize", {})


class RecordingTransport:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict] = []

    def post_json(self, url, *, headers, body, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "body": body,
                "timeout": timeout,
            }
        )
        return self.response


def _model_config() -> OpenAICompatibleModelConfig:
    return OpenAICompatibleModelConfig(
        provider="openai_compatible",
        model="test-model",
        base_url="https://models.example.test/v1/",
        api_key_env="SHORTCUTMINER_TEST_API_KEY",
        timeout=12.5,
        temperature=0.0,
        max_tokens=512,
    )


def test_openai_compatible_runtime_uses_injected_transport_and_structured_json():
    transport = RecordingTransport(
        {"choices": [{"message": {"content": '{"status":"ok"}'}}]}
    )
    client = OpenAICompatibleClient(
        _model_config(),
        api_key="unit-test-placeholder",
        transport=transport,
    )

    result = client.invoke_agent(
        "identify",
        {"z": 2, "a": 1},
        system_prompt="Follow the schema.",
    )

    assert result == {"status": "ok"}
    call = transport.calls[0]
    assert call["url"] == "https://models.example.test/v1/chat/completions"
    assert call["timeout"] == 12.5
    assert call["body"]["model"] == "test-model"
    assert call["body"]["messages"] == [
        {"role": "system", "content": "Follow the schema."},
        {"role": "user", "content": '{"a":1,"z":2}'},
    ]
    assert call["body"]["response_format"] == {"type": "json_object"}


def test_openai_compatible_runtime_never_calls_transport_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
):
    config = _model_config()
    monkeypatch.delenv(config.api_key_env, raising=False)
    transport = RecordingTransport({})
    client = OpenAICompatibleClient(config, transport=transport)

    with pytest.raises(ModelRuntimeError, match="is not set"):
        client.invoke_agent("identify", {})
    assert transport.calls == []


def test_model_yaml_examples_validate_without_containing_credentials():
    for path in sorted((REPOSITORY_ROOT / "configs" / "models").glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = load_model_runtime_config(path)
        assert config.api_key_env.startswith("SHORTCUTMINER_")
        assert "api_key" not in raw


def test_run_config_accepts_model_runtime_and_rejects_retired_fields():
    request = {
        "runs_root": "runs",
        "dataset_info": {"name": "dataset", "root": "/data"},
        "task_input": {"type": "classification", "label_column": "label"},
        "seed": 1,
        "model_runtime": _model_config().model_dump(mode="json"),
    }
    parsed = RunRequest.model_validate(request)
    assert parsed.model_runtime.model == "test-model"

    retired_field = "gate" + "way_url"
    with pytest.raises(ValidationError):
        RunRequest.model_validate({**request, retired_field: "http://localhost"})


def test_agents_do_not_import_provider_sdks():
    provider_modules = {"openai", "deepseek", "anthropic"}
    code_roots = (
        REPOSITORY_ROOT / "shortcutminer" / "experiment",
        REPOSITORY_ROOT / "shortcutminer" / "workflow",
    )
    imported: set[str] = set()
    for root in code_roots:
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(provider_modules)


def test_formal_package_has_no_retired_runtime_imports():
    retired_runtime = "open" + "claw"
    violations = []
    for path in (REPOSITORY_ROOT / "shortcutminer").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.lower() for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [(node.module or "").lower()]
            else:
                continue
            if any(retired_runtime in module for module in modules):
                violations.append(str(path.relative_to(REPOSITORY_ROOT)))
    assert violations == []
