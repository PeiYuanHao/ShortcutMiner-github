"""Regression tests for the disabled observation-specific invariant retry."""

import json
from pathlib import Path

import pytest

from shortcutminer.experiment.qualitative_executor import MAX_DRAFT_INVARIANT_RETRIES, execute_qualitative
from shortcutminer.experiment.qualitative_executor import QualitativeExecutionError
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient
from shortcutminer.storage.model_calls import load_run_model_telemetry
from test_qualitative_context import _dataset, _request
from test_qualitative_finalizer import _draft_dict


class _RecordingTransport:
    def __init__(self, draft: dict):
        self.response = {
            "choices": [{"message": {"content": json.dumps(draft)}}]
        }
        self.calls: list[dict] = []

    def post_json(self, url, *, headers, body, timeout):
        self.calls.append({"url": url, "body": body, "timeout": timeout})
        return self.response


def _client(runs_root: Path, draft: dict):
    transport = _RecordingTransport(draft)
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


def test_cross_group_observation_fails_without_special_retry(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    runs_root = tmp_path / "runs"
    _dataset(dataset_root)
    request = _request(dataset_root)
    draft = _draft_dict()
    draft["observations"][0]["sample_refs"] = ["S1", "S3"]
    client, transport = _client(runs_root, draft)

    with pytest.raises(QualitativeExecutionError, match="protocol schema"):
        execute_qualitative(request, client, runs_root=runs_root)
    assert MAX_DRAFT_INVARIANT_RETRIES == 0
    assert len(client.invocations) == len(transport.calls) == 2
    model_telemetry = load_run_model_telemetry(runs_root, request.run_id)
    assert model_telemetry is not None
    assert model_telemetry.total_calls == 2
    assert model_telemetry.schema_retry_calls == 1
    assert model_telemetry.protocol_schema_failure_count == 2
    assert not (runs_root / request.run_id / "qualitative" / "draft_invariant_telemetry.json").exists()
    assert not (
        runs_root / request.run_id / "qualitative" / "raw_output.json"
    ).exists()


def test_comparison_contract_avoids_correction_and_private_truth(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    runs_root = tmp_path / "runs"
    _dataset(dataset_root)
    request = _request(dataset_root)
    client, transport = _client(runs_root, _draft_dict())
    private = tmp_path / "private" / "hidden_truth.json"
    private.parent.mkdir()
    private.write_text("PRIVATE_TRUTH_SENTINEL", encoding="utf-8")

    result = execute_qualitative(request, client, runs_root=runs_root)

    assert len(transport.calls) == 1
    body = json.dumps(transport.calls[0]["body"], sort_keys=True)
    assert "PRIVATE_TRUTH_SENTINEL" not in body
    assert "hidden_truth" not in body
    assert len(result.payload.comparisons) == 1
    assert result.payload.comparisons[0].groups == ["label=A", "label=B"]
    invariant_telemetry = json.loads(
        (
            runs_root / request.run_id / "qualitative"
            / "draft_invariant_telemetry.json"
        ).read_text(encoding="utf-8")
    )
    assert invariant_telemetry["status"] == "passed"
    assert invariant_telemetry["draft_invariant_retry_count"] == 0
