"""Each Agent's schema-valid context failure uses the shared retry boundary."""

import json

import pytest

from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient
from shortcutminer.tooling.agent_loop import (
    ProviderContractValidationError, invoke_agent_with_tools,
)


class TwoResponseTransport:
    def __init__(self, invalid, valid):
        self.responses = [invalid, valid]
        self.calls = []

    def post_json(self, _url, *, headers, body, timeout):
        self.calls.append(body)
        return {
            "choices": [{"message": {"content": json.dumps(self.responses.pop(0))}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        }


_VALID = {
    "identify": {"candidates": [], "additional_limitations": []},
    "executive": {"decisions": [], "limitations": []},
    "quantitative": {
        "status": "blocked", "analysis_type": "unavailable",
        "evidence_direction": "not_tested", "evidence_summary": "No experiment is available.",
        "additional_limitations": ["No authorized experiment."],
        "recommended_followup": [],
    },
    "qualitative": {
        "sample_ids": [], "evidence_status": "insufficient_evidence",
        "evidence_summary": "No inspected samples are available.",
        "observations": [], "comparisons": [], "interpretations": [],
        "counterevidence": [], "metadata_evidence": [], "failures": [],
        "additional_limitations": [], "artifact_references": [],
        "recommended_followup": [],
    },
    "summarize": {"overview": "No candidates were supplied.",
                  "findings": [], "additional_limitations": []},
}


@pytest.mark.parametrize("agent", sorted(_VALID))
def test_context_invalid_then_valid_retries_with_same_scientific_request(agent):
    valid = _VALID[agent]
    invalid = json.loads(json.dumps(valid))
    field = "limitations" if agent == "executive" else "additional_limitations"
    invalid[field] = ["context-invalid-marker"]
    transport = TwoResponseTransport(invalid, valid)
    client = OpenAICompatibleClient(
        OpenAICompatibleModelConfig(
            model="offline", base_url="https://offline.invalid/v1",
            api_key_env="OFFLINE_KEY", temperature=0,
            model_call_budget={"max_total_calls": 5, "max_calls_per_agent": 5,
                               "max_schema_retries": 1},
        ),
        api_key="offline", transport=transport, sleeper=lambda _: None,
    )

    def context_validator(output):
        if "context-invalid-marker" in output[field]:
            raise ProviderContractValidationError("context_invalid", field)

    result = invoke_agent_with_tools(
        client, agent, {"run_id": "five-agent-retry", "scientific_context": {"fixed": True}},
        system_prompt="offline", tool_runtime=None,
        provider_contract_validator=context_validator,
    )
    assert result.output == valid
    assert len(transport.calls) == 2
    assert client.invocations[0].schema_valid is False
    assert client.invocations[1].schema_valid is True
    assert client.invocations[1].schema_retry is True
    assert transport.calls[0]["messages"][1] == transport.calls[1]["messages"][1]
