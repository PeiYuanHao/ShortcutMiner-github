"""Offline regression checks for the five preserved Agent workspaces."""

import json
import os
from pathlib import Path
import re
import socket

import pytest

from shortcutminer.experiment.agent_workspace import (
    AGENT_NAMES,
    OPTIONAL_WORKSPACE_FILES,
    REQUIRED_WORKSPACE_FILES,
    WORKSPACE_FILES,
    AgentWorkspaceError,
    default_agent_workspace_dir,
    load_agent_workspace_prompt,
)
from shortcutminer.protocol.quantitative import QuantitativeAgentDraft
from shortcutminer.protocol.summarize import SummaryReport


AGENTS = ("identify", "executive", "quantitative", "qualitative", "summarize")
EXPECTED_ORDER = (
    "AGENTS.md", "IDENTITY.md", "SOUL.md", "USER.md", "TOOLS.md",
    "MEMORY.md", "HEARTBEAT.md",
)


def _forbidden_dependency(*args, **kwargs):
    raise AssertionError("Workspace loading must use explicit local file reads only")


class _NoEnvironment(dict):
    __getitem__ = _forbidden_dependency
    __contains__ = _forbidden_dependency
    get = _forbidden_dependency


def test_workspace_roles_and_established_load_order():
    assert AGENT_NAMES == set(AGENTS)
    assert WORKSPACE_FILES == EXPECTED_ORDER
    assert REQUIRED_WORKSPACE_FILES == EXPECTED_ORDER[:3]
    assert OPTIONAL_WORKSPACE_FILES == EXPECTED_ORDER[3:]


@pytest.mark.parametrize("agent", AGENTS)
def test_all_seven_real_documents_are_loaded_exactly_and_repeatably(agent):
    root = default_agent_workspace_dir(agent)
    assert root.is_dir()
    assert {p.name for p in root.glob("*.md")} == set(EXPECTED_ORDER)
    contents = []
    for name in EXPECTED_ORDER:
        path = root / name
        assert path.is_file()
        content = path.read_text(encoding="utf-8")
        assert content.strip(), path
        contents.append(content)
    expected = "\n\n".join(contents)
    assert load_agent_workspace_prompt(agent) == expected
    assert load_agent_workspace_prompt(agent) == expected


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("creation_order", [EXPECTED_ORDER, EXPECTED_ORDER[::-1]])
def test_loading_ignores_creation_order_and_directory_enumeration(
    agent, creation_order, tmp_path, monkeypatch,
):
    for name in creation_order:
        (tmp_path / name).write_text(f"# {agent}: {name}\n", encoding="utf-8")
    (tmp_path / "UNRELATED.md").write_text("Do not load me.", encoding="utf-8")
    expected = "\n\n".join(f"# {agent}: {name}\n" for name in EXPECTED_ORDER)
    with monkeypatch.context() as patch:
        for name in ("iterdir", "glob", "rglob"):
            patch.setattr(Path, name, _forbidden_dependency)
        patch.setattr(os, "listdir", _forbidden_dependency)
        patch.setattr(os, "scandir", _forbidden_dependency)
        first = load_agent_workspace_prompt(agent, tmp_path)
        second = load_agent_workspace_prompt(agent, tmp_path)
    assert first == second == expected


@pytest.mark.parametrize("agent", AGENTS)
def test_real_workspace_loading_needs_no_network_or_environment(agent, monkeypatch):
    expected = load_agent_workspace_prompt(agent)
    with monkeypatch.context() as patch:
        # Block all environment reads, not just one provider's API-key name.
        patch.setattr(os, "environ", _NoEnvironment())
        patch.setattr(os, "getenv", _forbidden_dependency)
        patch.setattr(socket, "socket", _forbidden_dependency)
        patch.setattr(socket, "create_connection", _forbidden_dependency)
        patch.setattr(socket, "getaddrinfo", _forbidden_dependency)
        actual = load_agent_workspace_prompt(agent)
        assert actual == expected


@pytest.mark.parametrize(
    ("agent", "protocol"),
    [
        ("identify", "CandidateReport"),
        ("executive", "RoutingPlan"),
        ("quantitative", "QuantitativeAgentDraft"),
        ("qualitative", "QualitativeResult"),
        ("summarize", "SummaryReport"),
    ],
)
def test_workspace_prompt_ends_with_strict_json_output_rule(agent, protocol):
    prompt = load_agent_workspace_prompt(agent).rstrip()
    assert f"exactly one valid JSON object matching `{protocol}`" in prompt
    assert prompt.endswith("must be `}`.")


@pytest.mark.parametrize("agent", AGENTS)
def test_workspace_has_no_retired_platform_or_obsolete_identifier_contract(agent):
    prompt = load_agent_workspace_prompt(agent).lower()
    retired = (
        "open" + "claw", "gate" + "way", "sessions" + "_send",
        "session_id", "agent_id", "hidden session", "persistent chat",
        "hypothesis_id", "shortcut_type",
    )
    assert not any(term in prompt for term in retired)


@pytest.mark.parametrize("agent,request_name,result_name", [
    ("identify", "IdentifyRequest", "CandidateReport"),
    ("executive", "ExecutiveRequest", "RoutingPlan"),
    ("quantitative", "QuantitativeRequest", "QuantitativeResult"),
    ("qualitative", "QualitativeRequest", "QualitativeResult"),
    ("summarize", "SummarizeRequest", "SummaryReport"),
])
def test_operational_docs_name_actual_contracts_and_parallel_workflow(
    agent, request_name, result_name,
):
    text = (default_agent_workspace_dir(agent) / "AGENTS.md").read_text("utf-8")
    assert request_name in text
    assert result_name in text
    assert "START → Identify → Executive → (Quantitative ∥ Qualitative) → Summarize → END" in text
    assert "parallel branches" in text


@pytest.mark.parametrize("filename", REQUIRED_WORKSPACE_FILES)
@pytest.mark.parametrize("invalid_content", [None, "   \n", b"\xff"])
def test_invalid_required_document_fails_clearly(filename, invalid_content, tmp_path):
    for name in EXPECTED_ORDER:
        if name == filename and invalid_content is None:
            continue
        content = invalid_content if name == filename else b"# Policy\n"
        if isinstance(content, str):
            content = content.encode("utf-8")
        (tmp_path / name).write_bytes(content)
    with pytest.raises(AgentWorkspaceError, match=re.escape(filename)):
        load_agent_workspace_prompt("summarize", tmp_path)


@pytest.mark.parametrize("agent,schema,message_type", [
    ("quantitative", QuantitativeAgentDraft, None),
    ("summarize", SummaryReport, "summary_report"),
])
def test_documented_output_templates_match_real_pydantic_schema(
    agent, schema, message_type,
):
    text = (default_agent_workspace_dir(agent) / "AGENTS.md").read_text("utf-8")
    examples = [json.loads(block) for block in re.findall(r"```json\n(.*?)\n```", text, re.S)]
    if message_type is None:
        example, = [item for item in examples if "evidence_tool_call_ids" in item]
    else:
        example, = [item for item in examples if item.get("message_type") == message_type]

    def substitute(value):
        if isinstance(value, dict):
            return {key: substitute(item) for key, item in value.items()}
        if isinstance(value, list):
            return [substitute(item) for item in value]
        if isinstance(value, str) and value.startswith("<"):
            if "candidate_id" in value:
                return "C001"
            if "task_id" in value:
                return "TASK-C001-QT"
            return "workspace_example"
        return value

    result = schema.model_validate(substitute(example))
    assert result.model_dump(mode="json") == substitute(example)
