"""Identify output protocol instructions; synthetic responses and no provider access."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from shortcutminer.experiment.identify_workspace import load_identify_workspace_prompt
from shortcutminer.protocol.identify import IdentifyAgentDraft
from shortcutminer.runtime.identify_output_contract import compact_identify_output_schema, identify_output_contract
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient, ModelRuntimeError, _agent_schema_correction, _schema_correction
from test_identify_executor import _valid_draft
from test_model_runtime_hardening import SequenceTransport, config, response


@pytest.mark.parametrize("problem,path,error", [
    ("family", ("candidates", 0, "cue", "family"), "literal_error"),
    ("description", ("candidates", 0, "description"), "missing"),
])
def test_invalid_family_and_missing_description_still_rejected(problem, path, error):
    draft = _valid_draft()
    if problem == "family":
        draft["candidates"][0]["cue"]["family"] = "unrecognized-private-family"
    else:
        draft["candidates"][0].pop("description")
    before = deepcopy(draft)
    with pytest.raises(ValidationError) as exc:
        IdentifyAgentDraft.model_validate(draft)
    assert any(item["loc"] == path and item["type"] == error for item in exc.value.errors())
    assert draft == before  # No coercion, filling, or dropping candidates.


def test_complete_candidate_and_documented_generic_example_pass():
    assert IdentifyAgentDraft.model_validate(_valid_draft()).candidates
    document = (Path(__file__).resolve().parents[1] / "agents/identify/workspace/AGENTS.md").read_text(encoding="utf-8")
    example = document.split("### Generic syntax example", 1)[1].split("```json\n", 1)[1].split("```", 1)[0]
    IdentifyAgentDraft.model_validate(json.loads(example))


def test_final_prompt_exposes_schema_enums_and_required_candidate_fields():
    compact = compact_identify_output_schema()
    candidate = compact["properties"]["candidates"]["items"]
    executable = IdentifyAgentDraft.model_json_schema()
    prompt = load_identify_workspace_prompt()
    contract = identify_output_contract()
    assert prompt.endswith(contract) and prompt == load_identify_workspace_prompt()
    families = executable["$defs"]["Cue"]["properties"]["family"]["enum"]
    assert families == ["appearance", "spatial", "context", "geometry", "acquisition", "metadata", "other"]
    assert candidate["properties"]["cue"]["properties"]["family"]["enum"] == families
    for value in families:
        assert f'"{value}"' in contract
    assert candidate["required"] == executable["$defs"]["IdentifyCandidateDraft"]["required"]
    assert "description" in candidate["required"] and "Do not omit description" in contract
    assert candidate["properties"]["cue"]["required"] == ["family", "name"]
    for field in ("evidence_sources", "priority"):
        spec = candidate["properties"][field]
        source = executable["$defs"]["IdentifyCandidateDraft"]["properties"][field]
        assert spec.get("enum", spec.get("items")) == source.get("enum", source.get("items"))


def test_taxonomy_and_generated_contract_contain_no_dataset_answer_hints():
    document = (Path(__file__).resolve().parents[1] / "agents/identify/workspace/AGENTS.md").read_text(encoding="utf-8")
    taxonomy = document.split("### Taxonomy", 1)[1].split("### Fields", 1)[0]
    for forbidden in ("waterbirds", "place", "background shortcut", "hidden truth", "dataset-specific answer"):
        assert forbidden not in taxonomy.lower()
        assert forbidden not in identify_output_contract().lower()


def test_compact_schema_is_derived_without_a_separate_enum_copy(monkeypatch):
    executable = IdentifyAgentDraft.model_json_schema()
    executable["$defs"]["Cue"]["properties"]["family"]["enum"] = ["synthetic-test-literal"]
    monkeypatch.setattr(IdentifyAgentDraft, "model_json_schema", lambda: executable)
    candidate = compact_identify_output_schema()["properties"]["candidates"]["items"]
    assert candidate["properties"]["cue"]["properties"]["family"]["enum"] == ["synthetic-test-literal"]


def test_retry_repeats_authoritative_contract_without_changing_scientific_request():
    valid = _valid_draft()
    invalid = deepcopy(valid)
    invalid["candidates"][0]["cue"]["family"] = "unrecognized-private-family"
    invalid["candidates"][0].pop("description")
    payload = {"request": {"run_id": "identify-contract-retry", "seed": 42},
               "discovery_context": {"metadata_available": True, "observations": ["generic-observation"]}}
    before = deepcopy(payload)
    transport = SequenceTransport(response(invalid), response(valid))
    client = OpenAICompatibleClient(config(), api_key="dummy", transport=transport)
    prompt = load_identify_workspace_prompt()
    result = client.invoke_agent_validated("identify", payload, system_prompt=prompt,
                                          validator=IdentifyAgentDraft.model_validate)
    assert result == valid and payload == before
    assert len(transport.calls) == 2
    original, retried = [call["body"] for call in transport.calls]
    assert retried["messages"][:-1] == original["messages"]
    assert original["messages"][0]["content"] == prompt
    assert json.loads(original["messages"][1]["content"]) == payload
    assert {k:v for k,v in original.items() if k != "messages"} == {k:v for k,v in retried.items() if k != "messages"}
    correction = retried["messages"][-1]["content"]
    assert identify_output_contract() in correction
    assert "Correct only protocol conformance." in correction
    assert "Do not change the scientific interpretation merely to satisfy validation." in correction
    assert "unrecognized-private-family" not in correction
    assert "generic-observation" not in correction
    assert client.invocations[0].status == "protocol_schema_failed"
    assert client.invocations[1].status == "completed"


def test_retry_still_fails_after_configured_budget_without_repair():
    invalid = _valid_draft()
    invalid["candidates"][0].pop("description")
    transport = SequenceTransport(response(invalid), response(invalid))
    client = OpenAICompatibleClient(config(), api_key="dummy", transport=transport)
    with pytest.raises(ModelRuntimeError):
        client.invoke_agent_validated("identify", {"run_id": "invalid-contract-twice"},
            system_prompt=load_identify_workspace_prompt(), validator=IdentifyAgentDraft.model_validate)
    assert len(transport.calls) == 2
    assert all(record.status == "protocol_schema_failed" for record in client.invocations)


@pytest.mark.parametrize("agent", ["executive", "quantitative", "summarize"])
def test_other_agents_keep_existing_retry_instructions(agent):
    assert _agent_schema_correction(agent, None) == _schema_correction(None)


def test_qualitative_retry_preserves_evidence_and_requires_authorized_references():
    correction = _agent_schema_correction("qualitative", None)
    assert correction.startswith(_schema_correction(None))
    assert "actual selected sample ID" in correction
    assert "Preserve genuine counterevidence" in correction
    assert "Do not invent sample references or change scientific interpretation" in correction
