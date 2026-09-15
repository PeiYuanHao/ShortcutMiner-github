"""Bounded, value-free correction of provider-owned draft invariants."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from shortcutminer.experiment.qualitative_executor import (
    MAX_DRAFT_INVARIANT_RETRIES,
    QualitativeDraftValidationError,
    execute_qualitative,
)
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient
from shortcutminer.storage.model_calls import load_run_model_telemetry
from test_qualitative_context import _dataset, _request
from test_qualitative_finalizer import _draft_dict


class _SequenceTransport:
    def __init__(self, *drafts: dict):
        self.responses = [
            {"choices": [{"message": {"content": json.dumps(draft)}}]}
            for draft in drafts
        ]
        self.calls: list[dict] = []

    def post_json(self, url, *, headers, body, timeout):
        self.calls.append({"url": url, "body": body, "timeout": timeout})
        return self.responses.pop(0)


def _client(runs_root: Path, *drafts: dict):
    transport = _SequenceTransport(*drafts)
    client = OpenAICompatibleClient(
        OpenAICompatibleModelConfig(
            model="offline-contract-model",
            base_url="https://provider.invalid/v1",
            api_key_env="SHORTCUTMINER_TEST_KEY",
            temperature=0.0,
            model_call_budget={
                "max_total_calls": 4,
                "max_calls_per_agent": 4,
                "max_schema_retries": 1,
            },
        ),
        api_key="offline",
        transport=transport,
        runs_root=runs_root,
    )
    return client, transport


def _cross_group_draft() -> dict:
    value = _draft_dict()
    value["observations"][0] = {
        "sample_refs": ["S1", "S3"],
        "observation": "OLD_DRAFT_SENTINEL cross-group comparison text.",
        "inspection_quality": "sufficient",
    }
    value["observations"] = value["observations"][:1]
    value["interpretations"] = []
    return value


def test_same_group_observation_passes_without_invariant_retry(tmp_path: Path):
    _dataset(tmp_path / "dataset")
    request = _request(tmp_path / "dataset")
    draft = _draft_dict()
    client, _ = _client(tmp_path / "runs", draft)

    result = execute_qualitative(request, client, runs_root=tmp_path / "runs")

    assert result.payload.observations[0].sample_refs == ["S1", "S2"]
    telemetry = json.loads(
        (
            tmp_path / "runs" / request.run_id / "qualitative"
            / "draft_invariant_telemetry.json"
        ).read_text(encoding="utf-8")
    )
    assert telemetry["status"] == "passed"
    assert telemetry["draft_invariant_retry_count"] == 0


def test_cross_group_error_has_only_safe_structural_details(tmp_path: Path):
    _dataset(tmp_path)
    request = _request(tmp_path)
    client, _ = _client(tmp_path / "runs", _cross_group_draft(), _cross_group_draft())

    with pytest.raises(QualitativeDraftValidationError) as raised:
        execute_qualitative(request, client, runs_root=tmp_path / "runs")

    assert raised.value.category == "observation_crosses_groups"
    assert raised.value.observation_index == 0
    assert raised.value.referenced_group_count == 2
    assert raised.value.safe_details() == {
        "category": "observation_crosses_groups",
        "observation_index": 0,
        "referenced_group_count": 2,
    }


def test_one_invariant_retry_recovers_without_schema_or_transport_retry(
    tmp_path: Path,
):
    dataset_root = tmp_path / "dataset"
    runs_root = tmp_path / "runs"
    _dataset(dataset_root)
    request = _request(dataset_root)
    first = _cross_group_draft()
    corrected = _draft_dict()
    client, transport = _client(runs_root, first, corrected)
    hidden_truth = tmp_path / "private" / "hidden_truth.json"
    hidden_truth.parent.mkdir()
    hidden_truth.write_text("PRIVATE_TRUTH_SENTINEL", encoding="utf-8")

    result = execute_qualitative(request, client, runs_root=runs_root)

    assert len(client.invocations) == len(transport.calls) == 2
    assert all(record.status == "completed" for record in client.invocations)
    assert all(record.schema_retry is False for record in client.invocations)
    assert all(record.transport_retry is False for record in client.invocations)
    assert len({record.payload_hash for record in client.invocations}) == 1
    first_user = transport.calls[0]["body"]["messages"][-1]
    second_user = transport.calls[1]["body"]["messages"][-1]
    assert first_user == second_user
    second_body = json.dumps(transport.calls[1]["body"], sort_keys=True)
    assert "OLD_DRAFT_SENTINEL" not in second_body
    assert "PRIVATE_TRUTH_SENTINEL" not in second_body
    assert "hidden_truth" not in second_body
    assert "observation_crosses_groups" in second_body
    assert "Observation index: 0" in second_body
    assert "Referenced group count: 2" in second_body
    model_telemetry = load_run_model_telemetry(runs_root, request.run_id)
    assert model_telemetry is not None
    assert model_telemetry.total_calls == 2
    assert model_telemetry.schema_retry_calls == 0
    assert model_telemetry.retry_calls == 0
    assert model_telemetry.protocol_schema_failure_count == 0

    assert result.payload.evidence_status == corrected["evidence_status"]
    assert [item.observation for item in result.payload.observations] == [
        item["observation"] for item in corrected["observations"]
    ]
    assert [item.sample_refs for item in result.payload.observations] == [
        item["sample_refs"] for item in corrected["observations"]
    ]
    assert [item.interpretation for item in result.payload.interpretations] == [
        item["interpretation"] for item in corrected["interpretations"]
    ]

    telemetry_path = (
        runs_root / request.run_id / "qualitative"
        / "draft_invariant_telemetry.json"
    )
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert telemetry["status"] == "passed"
    assert telemetry["max_draft_invariant_retries"] == 1
    assert telemetry["draft_invariant_retry_count"] == 1
    assert [item["draft_invariant_result"] for item in telemetry["attempts"]] == [
        "failed",
        "passed",
    ]
    assert len({item["scientific_input_hash"] for item in telemetry["attempts"]}) == 1
    assert telemetry["scientific_input_hash"] == telemetry["attempts"][0][
        "scientific_input_hash"
    ]
    persisted_draft = (
        runs_root / request.run_id / "qualitative" / "raw_output.json"
    ).read_text(encoding="utf-8")
    assert "OLD_DRAFT_SENTINEL" not in persisted_draft
    persisted_run = b"\n".join(
        path.read_bytes()
        for path in (runs_root / request.run_id).rglob("*")
        if path.is_file()
    )
    assert b"OLD_DRAFT_SENTINEL" not in persisted_run
    assert b"PRIVATE_TRUTH_SENTINEL" not in persisted_run


def test_second_cross_group_draft_fails_after_exactly_one_retry(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    runs_root = tmp_path / "runs"
    _dataset(dataset_root)
    request = _request(dataset_root)
    first = _cross_group_draft()
    second = deepcopy(first)
    second["observations"][0]["observation"] = "Still structurally invalid."
    client, transport = _client(runs_root, first, second)

    with pytest.raises(
        QualitativeDraftValidationError,
        match="sample references cross groups",
    ):
        execute_qualitative(request, client, runs_root=runs_root)

    assert MAX_DRAFT_INVARIANT_RETRIES == 1
    assert len(client.invocations) == len(transport.calls) == 2
    telemetry = json.loads(
        (
            runs_root / request.run_id / "qualitative"
            / "draft_invariant_telemetry.json"
        ).read_text(encoding="utf-8")
    )
    assert telemetry["status"] == "failed"
    assert telemetry["draft_invariant_retry_count"] == 1
    assert [item["draft_invariant_result"] for item in telemetry["attempts"]] == [
        "failed",
        "failed",
    ]
    assert not (
        runs_root / request.run_id / "qualitative" / "raw_output.json"
    ).exists()
