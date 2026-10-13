import json
import socket
from copy import deepcopy
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from shortcutminer.protocol.model import (
    ModelRuntimeConfigSet,
    OpenAICompatibleModelConfig,
)
from shortcutminer.protocol.model_runtime import ModelCallBudget, ModelCapabilities
from shortcutminer.protocol.qualitative import QualitativeResult
from shortcutminer.runtime.openai_compatible import (
    ModelRuntimeError,
    OpenAICompatibleClient,
    ProviderResponseJSONError,
    build_runtime_snapshot,
)
from shortcutminer.storage.model_calls import load_run_model_telemetry
from shortcutminer.tooling.agent_loop import invoke_agent_with_tools
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolRuntime


class SequenceTransport:
    def __init__(self, *values):
        self.values = list(values)
        self.calls = []

    def post_json(self, url, *, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.mark.parametrize("rule,field", [
    ("candidate_budget_exceeded", "candidates"),
    ("duplicate_metadata_field_selector", "candidates"),
    ("metadata_unavailable", "candidates.0.evidence_sources"),
    ("sample_inspection_unavailable", "candidates.1.evidence_sources"),
    ("image_statistics_unavailable", "candidates.2.evidence_sources"),
    ("invalid_metadata_field_selector", "candidates.0.metadata_field_ref"),
    ("metadata_selector_without_source", "candidates.1.metadata_field_ref"),
])
def test_contract_rule_and_field_are_persisted_without_values(tmp_path, rule, field):
    from shortcutminer.tooling.agent_loop import ProviderContractValidationError
    rejected = {"private_value": "rejected-value-marker"}
    client = OpenAICompatibleClient(config(model_call_budget={"max_schema_retries": 0}),
        api_key="credential-marker", transport=SequenceTransport(response(rejected)), runs_root=tmp_path)
    def validator(_output):
        raise ProviderContractValidationError(rule, field)
    with pytest.raises(ModelRuntimeError) as failure:
        client.invoke_agent_validated("identify", {"run_id": "safe-contract-diagnostic"},
                                      system_prompt="policy", validator=validator)
    record = client.invocations[0]
    issue = record.schema_diagnostics.issues[0]
    assert (issue.error_type, issue.field_path) == (rule, field)
    assert record.status == "protocol_schema_failed" and record.retryable is False
    assert rule in str(failure.value) and field in str(failure.value)
    saved = "\n".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json"))
    assert "credential-marker" not in saved and "rejected-value-marker" not in saved
    events = [json.loads(line) for line in (tmp_path / "safe-contract-diagnostic/events.jsonl").read_text().splitlines()]
    event = next(item for item in events if item["event"] == "MODEL_INVOCATION_FAILED")
    assert event["details"]["schema_diagnostics"]["issues"][0] == issue.model_dump()


def test_contract_diagnostic_preserves_retry_budget_and_scientific_request():
    from shortcutminer.tooling.agent_loop import ProviderContractValidationError
    transport = SequenceTransport(response({"rejected": True}), response({"accepted": True}))
    client = OpenAICompatibleClient(config(), api_key="dummy", transport=transport)
    def validator(value):
        if value.get("rejected"):
            raise ProviderContractValidationError("invalid_metadata_field_selector", "candidates.0.metadata_field_ref")
    assert client.invoke_agent_validated("identify", {"run_id": "same-retry-policy"},
                                        system_prompt="policy", validator=validator) == {"accepted": True}
    assert len(transport.calls) == 2
    correction = transport.calls[1]["body"]["messages"][-1]["content"]
    assert "candidates.0.metadata_field_ref: invalid_metadata_field_selector" in correction
    assert "Correct only protocol conformance." in correction
    assert transport.calls[1]["body"]["messages"][:-1] == transport.calls[0]["body"]["messages"]
    assert client.invocations[0].schema_diagnostics.issues[0].error_type == "invalid_metadata_field_selector"
    assert client.invocations[1].schema_diagnostics is None


@pytest.mark.parametrize("error", [ValueError("private-marker"), TypeError("private-marker")])
def test_generic_validator_errors_record_only_their_type(error):
    client = OpenAICompatibleClient(config(model_call_budget={"max_schema_retries": 0}),
        api_key="dummy", transport=SequenceTransport(response({"ok": True})))
    def validator(_value):
        raise error
    with pytest.raises(ModelRuntimeError) as failure:
        client.invoke_agent_validated("identify", {"run_id": "generic-validator"},
                                      system_prompt="policy", validator=validator)
    issue = client.invocations[0].schema_diagnostics.issues[0]
    assert issue.error_type in {"validator_value_error", "validator_type_error"}
    assert issue.field_path == "<root>"
    assert "private-marker" not in str(failure.value)


def test_unrecognized_contract_metadata_is_redacted():
    from shortcutminer.tooling.agent_loop import ProviderContractValidationError
    from shortcutminer.runtime.openai_compatible import _safe_contract_diagnostics
    error = ProviderContractValidationError("private_category_marker", "candidates.private_field_marker")
    issue = _safe_contract_diagnostics(error).issues[0]
    assert issue.error_type == "provider_contract_rejected"
    assert issue.field_path == "<field>"


def test_actual_identify_validator_still_rejects_label_as_metadata_selector(tmp_path):
    from test_identify_executor import _create_test_dataset, _create_request, _valid_draft
    from shortcutminer.experiment.identify_executor import execute_identify, IdentifyExecutionError
    _create_test_dataset(tmp_path)
    draft = _valid_draft()
    draft["candidates"][0]["metadata_field_ref"] = "label"
    transport = SequenceTransport(response(draft))
    client = OpenAICompatibleClient(config(model_call_budget={"max_schema_retries": 0}),
        api_key="dummy", transport=transport)
    with pytest.raises(IdentifyExecutionError, match="invalid_metadata_field_selector"):
        execute_identify(_create_request(tmp_path), seed=42, client=client)
    issue = client.invocations[0].schema_diagnostics.issues[0]
    assert (issue.error_type, issue.field_path) == ("invalid_metadata_field_selector", "candidates.0.metadata_field_ref")
    from shortcutminer.experiment.identify_executor import _public_metadata_fields
    from shortcutminer.experiment.identify_context import build_identify_context
    request = _create_request(tmp_path)
    expected = sorted(_public_metadata_fields(request, build_identify_context(request, seed=42)))
    prompt = transport.calls[0]["body"]["messages"][0]["content"]
    assert json.loads(prompt.rsplit("\n", 1)[1]) == expected
    assert "label" not in expected
    assert "Non-metadata cue families must use null" in prompt


def response(value, *, usage=None):
    envelope = {"choices": [{"message": {"content": json.dumps(value)}}]}
    if usage is not None:
        envelope["usage"] = usage
    return envelope


def config(**updates):
    values = {
        "model": "fake-model",
        "base_url": "https://provider.example/v1",
        "api_key_env": "SHORTCUTMINER_TEST_KEY",
        "retry_policy": {
            "max_attempts": 3,
            "initial_backoff_seconds": 0.0,
            "backoff_multiplier": 1.0,
            "max_backoff_seconds": 0.0,
        },
    }
    values.update(updates)
    return OpenAICompatibleModelConfig.model_validate(values)


def qualitative_result() -> dict:
    return {
        "protocol_version": "1.0",
        "run_id": "safe-schema-diagnostic",
        "message_id": "qualitative:result:C001",
        "source": "qualitative",
        "target": "summarize",
        "message_type": "qualitative_analysis_result",
        "in_reply_to": "qualitative:request:C001",
        "payload": {
            "task_id": "TASK-C001-QL",
            "candidate_id": "C001",
            "candidate_name": "dark_border",
            "hypothesis": "A dark border may be associated with the label.",
            "evidence_status": "partially_supported",
            "evidence_summary": "A border was visible, with exceptions.",
            "sampling": {
                "strategy": "group_balanced",
                "seed": 42,
                "groups": {"label=A": 1},
                "sample_count": 1,
                "sample_ids": ["S1"],
            },
            "observations": [{
                "observation_id": "QO-C001-001",
                "group": "label=A",
                "sample_refs": ["S1"],
                "observation": "A dark border is visible.",
                "inspection_quality": "sufficient",
            }],
            "comparisons": [],
            "interpretations": [{
                "interpretation": "The border may reflect preprocessing.",
                "supporting_observation_ids": ["QO-C001-001"],
                "supporting_comparison_ids": [],
                "confidence": "low",
            }],
            "counterevidence": [],
            "metadata_evidence": [],
            "failures": [],
            "limitations": [],
            "artifact_references": [],
            "recommended_followup": [],
        },
    }


def validate_qualitative(value: dict) -> None:
    QualitativeResult.model_validate(value)


@pytest.mark.parametrize(
    ("status", "category"),
    [(401, "authentication_failed"), (403, "authorization_failed"),
     (429, "rate_limited"), (500, "provider_5xx"), (400, "provider_rejected")],
)
def test_http_failure_taxonomy(status, category):
    error = HTTPError("https://provider.example", status, "failure", {}, None)
    transport = SequenceTransport(error, error, error)
    client = OpenAICompatibleClient(config(), api_key="secret", transport=transport)
    with pytest.raises(ModelRuntimeError) as exc:
        client.invoke_agent("identify", {"run_id": "failure-taxonomy"})
    assert exc.value.failure.category == category
    expected = 3 if category in {"rate_limited", "provider_5xx"} else 1
    assert len(transport.calls) == expected


def test_timeout_retries_with_bounded_backoff():
    delays = []
    transport = SequenceTransport(URLError(socket.timeout()), response({"ok": True}))
    client = OpenAICompatibleClient(
        config(), api_key="secret", transport=transport, sleeper=delays.append
    )
    assert client.invoke_agent("identify", {"run_id": "timeout-retry"}) == {"ok": True}
    assert delays == [0.0]
    assert [item.status for item in client.invocations] == ["timed_out", "completed"]
    assert client.invocations[-1].transport_retry is True


def test_invalid_json_and_schema_are_retried_separately():
    invalid_json = {"choices": [{"message": {"content": "not-json"}}]}
    transport = SequenceTransport(
        invalid_json,
        response({"wrong": True}),
        response({"ok": True}),
    )
    client = OpenAICompatibleClient(
        config(model_call_budget={"max_schema_retries": 2}),
        api_key="secret",
        transport=transport,
    )

    def validate(value):
        if value != {"ok": True}:
            raise ValueError("wrong schema")

    assert client.invoke_agent_validated(
        "identify", {"run_id": "schema-retry"}, system_prompt="policy", validator=validate
    ) == {"ok": True}
    assert [item.status for item in client.invocations] == [
        "invalid_json", "protocol_schema_failed", "completed"
    ]
    assert client.invocations[-1].schema_retry is True


def test_schema_retry_adds_correction_and_rehashes_without_changing_payload(tmp_path: Path):
    malformed = {
        "choices": [{
            "message": {
                "content": "not-json-sensitive-marker",
                "reasoning_content": "reasoning-sensitive-marker",
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }
    valid = response(
        {"ok": True},
        usage={"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
    )
    transport = SequenceTransport(malformed, valid)
    client = OpenAICompatibleClient(
        config(model_call_budget={"max_schema_retries": 1}),
        api_key="secret", transport=transport, runs_root=tmp_path,
    )

    assert client.invoke_agent_validated(
        "identify", {"run_id": "corrective-schema-retry", "evidence": [1, 2]},
        system_prompt="policy", validator=lambda value: value == {"ok": True},
    ) == {"ok": True}

    assert len(transport.calls) == 2
    first_body, retry_body = (item["body"] for item in transport.calls)
    assert first_body != retry_body
    assert retry_body["messages"][:-1] == first_body["messages"]
    assert "previous model response" in retry_body["messages"][-1]["content"]
    assert "not-json-sensitive-marker" not in retry_body["messages"][-1]["content"]
    assert all(
        body["response_format"] == {"type": "json_object"}
        for body in (first_body, retry_body)
    )
    first, retry = client.invocations
    assert first.payload_hash == retry.payload_hash
    assert first.request_hash != retry.request_hash
    assert first.schema_retry is False
    assert retry.schema_retry is True
    assert first.structured_output_mode == retry.structured_output_mode == "json_object"
    assert first.response_diagnostics.json_parse_status == "invalid_json"
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*") if path.is_file()
    )
    assert "not-json-sensitive-marker" not in persisted
    assert "reasoning-sensitive-marker" not in persisted


def test_valid_json_with_invalid_schema_is_corrected_once():
    transport = SequenceTransport(response({"wrong": True}), response({"ok": True}))
    client = OpenAICompatibleClient(config(), api_key="secret", transport=transport)

    def validate(value):
        if value != {"ok": True}:
            raise ValueError("wrong schema")

    assert client.invoke_agent_validated(
        "identify", {"run_id": "schema-only-retry"},
        system_prompt="policy", validator=validate,
    ) == {"ok": True}
    assert [item.status for item in client.invocations] == [
        "protocol_schema_failed", "completed"
    ]


@pytest.mark.parametrize(
    ("mutate", "field_path", "error_type", "unsafe_value"),
    [
        (
            lambda value: value["payload"]["observations"][0].__setitem__(
                "inspection_quality", "unsafe-invalid-quality"
            ),
            "payload.observations.0.inspection_quality",
            "literal_error",
            "unsafe-invalid-quality",
        ),
        (
            lambda value: value["payload"].pop("artifact_references"),
            "payload.artifact_references",
            "missing",
            None,
        ),
        (
            lambda value: value["payload"].__setitem__(
                "untrusted-dataset-field-name", "unsafe-extra-value"
            ),
            "payload.<extra_field>",
            "extra_forbidden",
            "unsafe-extra-value",
        ),
        (
            lambda value: value["payload"]["sampling"].__setitem__(
                "sample_count", "unsafe-string-count"
            ),
            "payload.sampling.sample_count",
            "int_type",
            "unsafe-string-count",
        ),
    ],
)
def test_pydantic_schema_diagnostics_are_value_free(
    tmp_path: Path, mutate, field_path, error_type, unsafe_value,
):
    invalid = qualitative_result()
    mutate(invalid)
    client = OpenAICompatibleClient(
        config(model_call_budget={"max_schema_retries": 0}),
        api_key="secret",
        transport=SequenceTransport(response(invalid)),
        runs_root=tmp_path,
    )
    with pytest.raises(ModelRuntimeError):
        client.invoke_agent_validated(
            "qualitative", {"run_id": "safe-schema-diagnostic"},
            system_prompt="policy", validator=validate_qualitative,
        )
    record = client.invocations[0]
    assert record.status == "protocol_schema_failed"
    assert record.response_kind == "final_result"
    assert record.schema_diagnostics.total_schema_issue_count == 1
    assert record.schema_diagnostics.issues[0].field_path == field_path
    assert record.schema_diagnostics.issues[0].error_type == error_type
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*.json")
    )
    if unsafe_value is not None:
        assert unsafe_value not in persisted
    assert "untrusted-dataset-field-name" not in persisted


def test_schema_diagnostics_are_capped_at_eight_without_losing_total():
    invalid = qualitative_result()
    for field in (
        "candidate_name", "hypothesis", "evidence_status", "evidence_summary",
        "sampling", "observations", "interpretations", "counterevidence",
        "metadata_evidence", "failures", "limitations", "artifact_references",
    ):
        invalid["payload"].pop(field)
    client = OpenAICompatibleClient(
        config(model_call_budget={"max_schema_retries": 0}),
        api_key="secret", transport=SequenceTransport(response(invalid)),
    )
    with pytest.raises(ModelRuntimeError):
        client.invoke_agent_validated(
            "qualitative", {"run_id": "safe-schema-diagnostic"},
            system_prompt="policy", validator=validate_qualitative,
        )
    diagnostics = client.invocations[0].schema_diagnostics
    assert diagnostics.total_schema_issue_count == 12
    assert len(diagnostics.issues) == 8


def test_targeted_schema_retry_uses_only_safe_issue_metadata(tmp_path: Path):
    invalid = qualitative_result()
    invalid["payload"]["observations"][0]["inspection_quality"] = (
        "unsafe-invalid-quality"
    )
    valid = qualitative_result()
    transport = SequenceTransport(response(invalid), response(valid))
    client = OpenAICompatibleClient(
        config(), api_key="secret", transport=transport, runs_root=tmp_path,
    )
    result = client.invoke_agent_validated(
        "qualitative", {"run_id": "safe-schema-diagnostic"},
        system_prompt="policy", validator=validate_qualitative,
    )
    QualitativeResult.model_validate(result)
    correction = transport.calls[1]["body"]["messages"][-1]["content"]
    assert "payload.observations.0.inspection_quality: literal_error" in correction
    assert "use one of the exact allowed schema literals" in correction
    assert "unsafe-invalid-quality" not in correction
    assert json.dumps(invalid, sort_keys=True) not in correction
    assert client.invocations[0].schema_diagnostics is not None
    assert client.invocations[1].schema_diagnostics is None
    assert client.invocations[1].schema_valid is True


def test_both_protocol_schema_attempts_invalid_fail_with_safe_diagnostics():
    first = qualitative_result()
    second = deepcopy(first)
    first["payload"].pop("artifact_references")
    second["payload"].pop("recommended_followup")
    client = OpenAICompatibleClient(
        config(), api_key="secret",
        transport=SequenceTransport(response(first), response(second)),
    )
    with pytest.raises(ModelRuntimeError) as exc:
        client.invoke_agent_validated(
            "qualitative", {"run_id": "safe-schema-diagnostic"},
            system_prompt="policy", validator=validate_qualitative,
        )
    assert exc.value.failure.category == "protocol_schema_failed"
    assert [record.schema_valid for record in client.invocations] == [False, False]
    assert client.invocations[1].schema_diagnostics.issues[0].field_path == (
        "payload.recommended_followup"
    )


def test_valid_qualitative_result_has_no_schema_issues():
    client = OpenAICompatibleClient(
        config(), api_key="secret",
        transport=SequenceTransport(response(qualitative_result())),
    )
    output = client.invoke_agent_validated(
        "qualitative", {"run_id": "safe-schema-diagnostic"},
        system_prompt="policy", validator=validate_qualitative,
    )
    QualitativeResult.model_validate(output)
    record = client.invocations[0]
    assert record.status == "completed"
    assert record.response_kind == "final_result"
    assert record.schema_diagnostics is None


def test_tool_call_schema_failure_is_classified_separately(tmp_path: Path):
    invalid_tool_call = {
        "tool_calls": [{
            "tool_call_id": "bad-tool-call",
            "arguments": {},
        }]
    }
    client = OpenAICompatibleClient(
        config(model_call_budget={"max_schema_retries": 0}),
        api_key="secret", transport=SequenceTransport(response(invalid_tool_call)),
    )
    runtime = ToolRuntime(
        build_default_tool_registry(),
        run_id="tool-call-schema-diagnostic", agent_name="identify",
        dataset_root=tmp_path, runs_root=tmp_path / "runs",
    )
    with pytest.raises(ModelRuntimeError):
        invoke_agent_with_tools(
            client, "identify", {"request": {"run_id": "tool-call-schema-diagnostic"}},
            system_prompt="policy", tool_runtime=runtime,
        )
    record = client.invocations[0]
    assert record.response_kind == "tool_call"
    assert record.schema_diagnostics.issues[0].field_path == "name"
    assert record.schema_diagnostics.issues[0].error_type == "missing"


def test_both_schema_attempts_invalid_fail_without_silent_repair():
    fenced = "```json\n{\"ok\": true}\n```"
    transport = SequenceTransport(
        {"choices": [{"message": {"content": fenced}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": "still not json"}, "finish_reason": "stop"}]},
    )
    client = OpenAICompatibleClient(config(), api_key="secret", transport=transport)
    with pytest.raises(ModelRuntimeError) as exc:
        client.invoke_agent("identify", {"run_id": "invalid-twice"})
    assert exc.value.failure.category == "invalid_json"
    assert len(transport.calls) == 2
    assert client.invocations[0].response_diagnostics.content_first_token_class == "markdown_fence"
    assert client.invocations[0].response_diagnostics.json_parse_status == "invalid_json"
    assert [item.status for item in client.invocations] == ["invalid_json", "invalid_json"]


def test_transport_500_retry_does_not_add_schema_correction():
    failure = HTTPError("https://provider.example", 500, "failure", {}, None)
    transport = SequenceTransport(failure, response({"ok": True}))
    client = OpenAICompatibleClient(
        config(), api_key="secret", transport=transport, sleeper=lambda _: None,
    )
    assert client.invoke_agent("identify", {"run_id": "transport-only-retry"}) == {"ok": True}
    assert transport.calls[0]["body"] == transport.calls[1]["body"]
    assert client.invocations[0].request_hash == client.invocations[1].request_hash
    assert client.invocations[1].transport_retry is True
    assert all(not item.schema_retry for item in client.invocations)


def test_empty_and_length_responses_have_safe_diagnostics():
    empty = {
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 8, "total_tokens": 11},
    }
    client = OpenAICompatibleClient(
        config(model_call_budget={"max_schema_retries": 0}),
        api_key="secret", transport=SequenceTransport(empty),
    )
    with pytest.raises(ModelRuntimeError):
        client.invoke_agent("identify", {"run_id": "empty-length"})
    diagnostics = client.invocations[0].response_diagnostics
    assert diagnostics.finish_reason == "length"
    assert diagnostics.content_present is False
    assert diagnostics.content_length == 0
    assert diagnostics.content_first_token_class == "empty"
    assert diagnostics.content_last_token_class == "empty"
    assert diagnostics.json_parse_status == "invalid_json"


def test_explicit_reasoning_configuration_is_capability_checked_and_sent():
    with pytest.raises(ValueError, match="supports_reasoning_config"):
        config(reasoning={"mode": "enabled", "effort": "high"})
    transport = SequenceTransport(response({"ok": True}))
    configured = config(
        reasoning={"mode": "enabled", "effort": "high"},
        capabilities={"supports_reasoning_config": True},
    )
    client = OpenAICompatibleClient(configured, api_key="secret", transport=transport)
    client.invoke_agent("identify", {"run_id": "explicit-reasoning"})
    assert transport.calls[0]["body"]["thinking"] == {"type": "enabled"}
    assert transport.calls[0]["body"]["reasoning_effort"] == "high"
    assert client.runtime_snapshot.reasoning == configured.reasoning
    assert client.runtime_snapshot.structured_output_mode == "json_object"


def test_explicit_non_thinking_request_omits_reasoning_effort_and_changes_snapshot():
    enabled = config(
        reasoning={"mode": "enabled", "effort": "high"},
        capabilities={"supports_reasoning_config": True},
    )
    disabled = config(
        reasoning={"mode": "disabled"},
        capabilities={"supports_reasoning_config": True},
    )
    transport = SequenceTransport(response({"ok": True}))
    client = OpenAICompatibleClient(disabled, api_key="secret", transport=transport)
    client.invoke_agent("identify", {"run_id": "non-thinking-request"})
    body = transport.calls[0]["body"]
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body
    assert body["response_format"] == {"type": "json_object"}
    assert (
        build_runtime_snapshot(enabled).snapshot_id
        != build_runtime_snapshot(disabled).snapshot_id
    )


def test_explicit_dashscope_thinking_wire_format():
    configured = config(reasoning={"mode": "disabled", "wire_format": "enable_thinking"},
                        capabilities={"supports_reasoning_config": True})
    transport = SequenceTransport(response({"ok": True}))
    client = OpenAICompatibleClient(configured, api_key="secret", transport=transport)
    client.invoke_agent("identify", {"run_id": "dashscope-wire"})
    assert transport.calls[0]["body"]["enable_thinking"] is False
    assert "thinking" not in transport.calls[0]["body"]


def test_invalid_provider_json_envelope_has_explicit_category():
    client = OpenAICompatibleClient(
        config(), api_key="secret", transport=SequenceTransport(ProviderResponseJSONError("x"))
    )
    with pytest.raises(ModelRuntimeError) as exc:
        client.invoke_agent("identify", {"run_id": "bad-envelope"})
    assert exc.value.failure.category == "invalid_response_envelope"


def test_call_budget_stops_before_second_transport_call():
    transport = SequenceTransport(response({"ok": 1}), response({"ok": 2}))
    client = OpenAICompatibleClient(
        config(model_call_budget=ModelCallBudget(max_total_calls=1, max_calls_per_agent=1)),
        api_key="secret",
        transport=transport,
    )
    client.invoke_agent("identify", {"run_id": "budget-test"})
    with pytest.raises(ModelRuntimeError) as exc:
        client.invoke_agent("identify", {"run_id": "budget-test"})
    assert exc.value.failure.category == "budget_exceeded"
    assert len(transport.calls) == 1


def test_usage_latency_hashes_and_secret_free_persistence(tmp_path: Path):
    ticks = iter([10.0, 10.125])
    transport = SequenceTransport(response(
        {"ok": True},
        usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    ))
    client = OpenAICompatibleClient(
        config(model_seed=7, capabilities={"supports_model_seed": True}),
        api_key="super-secret-value",
        transport=transport,
        runs_root=tmp_path,
        monotonic=lambda: next(ticks),
    )
    client.invoke_agent("identify", {"run_id": "persisted-runtime", "private": "public-input"})
    record = client.invocations[0]
    assert record.latency_ms == 125.0
    assert record.usage.total_tokens == 5
    assert len(record.request_hash) == len(record.payload_hash) == 64
    assert transport.calls[0]["body"]["seed"] == 7
    all_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "persisted-runtime").rglob("*") if path.is_file()
    )
    assert "super-secret-value" not in all_text
    assert '"private": "public-input"' not in all_text
    telemetry = load_run_model_telemetry(tmp_path, "persisted-runtime")
    assert telemetry.total_tokens == 5
    assert telemetry.total_model_latency_ms == 125.0


def test_missing_usage_is_explicit_and_never_estimated(tmp_path: Path):
    client = OpenAICompatibleClient(
        config(), api_key="secret", transport=SequenceTransport(response({"ok": True})),
        runs_root=tmp_path,
    )
    client.invoke_agent("executive", {"run_id": "unknown-usage"})
    telemetry = load_run_model_telemetry(tmp_path, "unknown-usage")
    assert telemetry.usage_complete is False
    assert telemetry.total_tokens is None
    assert telemetry.limitations


def test_secret_marker_never_persists_across_retry_and_failure(tmp_path: Path):
    marker = "FAKE_SECRET_ABC123"
    rate_limit = HTTPError("https://provider.example", 429, "slow down", {}, None)
    retrying = OpenAICompatibleClient(
        config(), api_key=marker,
        transport=SequenceTransport(rate_limit, response({"ok": True})),
        runs_root=tmp_path, sleeper=lambda _: None,
    )
    retrying.invoke_agent("identify", {"run_id": "secret-retry"})
    rejected = OpenAICompatibleClient(
        config(), api_key=marker,
        transport=SequenceTransport(
            HTTPError("https://provider.example", 401, marker, {}, None)
        ),
        runs_root=tmp_path,
    )
    with pytest.raises(ModelRuntimeError):
        rejected.invoke_agent("identify", {"run_id": "secret-failure"})
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    assert marker not in persisted


def test_runtime_snapshot_is_stable_sanitized_and_configuration_sensitive():
    first = build_runtime_snapshot(config())
    same = build_runtime_snapshot(config())
    changed = build_runtime_snapshot(config(model="other-model"))
    assert first == same
    assert first.snapshot_id != changed.snapshot_id
    assert first.endpoint.host == "provider.example"
    assert first.api_key_env == "SHORTCUTMINER_TEST_KEY"
    assert first.raw_response_persistence is False


def test_config_set_resolves_shared_and_per_agent_models():
    shared = config(model="shared")
    special = config(model="identify-model")
    settings = ModelRuntimeConfigSet(shared=shared, per_agent={"identify": special})
    assert settings.resolve("identify").model == "identify-model"
    assert settings.resolve("summarize").model == "shared"


def test_seed_is_omitted_without_declared_provider_capability():
    transport = SequenceTransport(response({"ok": True}))
    client = OpenAICompatibleClient(
        config(model_seed=99, capabilities=ModelCapabilities()),
        api_key="secret", transport=transport,
    )
    client.invoke_agent("identify", {"run_id": "seed-capability"})
    assert "seed" not in transport.calls[0]["body"]
    assert client.runtime_snapshot.model_seed is None


def test_endpoint_credentials_are_rejected():
    with pytest.raises(ValueError, match="credentials"):
        config(base_url="https://user:password@provider.example/v1")


def test_shared_budget_state_applies_across_per_agent_clients():
    from shortcutminer.runtime.openai_compatible import ModelBudgetState

    state = ModelBudgetState()
    budget = ModelCallBudget(max_total_calls=1, max_calls_per_agent=1)
    first_transport = SequenceTransport(response({"ok": True}))
    second_transport = SequenceTransport(response({"ok": True}))
    first = OpenAICompatibleClient(
        config(model_call_budget=budget), api_key="x", transport=first_transport,
        budget_state=state,
    )
    second = OpenAICompatibleClient(
        config(model="other", model_call_budget=budget), api_key="x",
        transport=second_transport, budget_state=state,
    )
    first.invoke_agent("identify", {"run_id": "shared-budget"})
    with pytest.raises(ModelRuntimeError) as exc:
        second.invoke_agent("summarize", {"run_id": "shared-budget"})
    assert exc.value.failure.category == "budget_exceeded"
    assert second_transport.calls == []


def test_per_agent_clients_merge_snapshots_and_run_telemetry(tmp_path: Path):
    from shortcutminer.runtime.openai_compatible import ModelBudgetState

    state = ModelBudgetState()
    first = OpenAICompatibleClient(
        config(model="identify-model"), api_key="x",
        transport=SequenceTransport(response({"ok": True})), runs_root=tmp_path,
        budget_state=state,
    )
    second = OpenAICompatibleClient(
        config(model="summarize-model"), api_key="x",
        transport=SequenceTransport(response({"ok": True})), runs_root=tmp_path,
        budget_state=state,
    )
    first.invoke_agent("identify", {"run_id": "per-agent-telemetry"})
    second.invoke_agent("summarize", {"run_id": "per-agent-telemetry"})
    telemetry = load_run_model_telemetry(tmp_path, "per-agent-telemetry")
    assert telemetry.total_calls == 2
    assert len(telemetry.runtime_snapshot_ids) == 2
    assert telemetry.per_agent[0].total_calls == 1
    assert telemetry.per_agent[4].total_calls == 1


def test_model_budget_and_tool_loop_budget_both_bound_iterations(tmp_path: Path):
    tool_call = {
        "tool_calls": [{
            "name": "inspect_tool_capabilities",
            "arguments": {},
        }]
    }
    transport = SequenceTransport(response(tool_call), response({"unused": True}))
    client = OpenAICompatibleClient(
        config(model_call_budget=ModelCallBudget(max_total_calls=1, max_calls_per_agent=1)),
        api_key="x", transport=transport,
    )
    runtime = ToolRuntime(
        build_default_tool_registry(),
        run_id="bounded-tool-loop",
        agent_name="identify",
        dataset_root=tmp_path,
        runs_root=tmp_path / "runs",
    )
    with pytest.raises(ModelRuntimeError) as exc:
        invoke_agent_with_tools(
            client,
            "identify",
            {"request": {"run_id": "bounded-tool-loop"}},
            system_prompt="policy",
            tool_runtime=runtime,
        )
    assert exc.value.failure.category == "budget_exceeded"
    assert len(transport.calls) == 1
    assert len(runtime.history) == 1
