from pathlib import Path

import json
from PIL import Image

import pytest

from shortcutminer.evaluation.blind.repeated import (
    RepeatedBlindAuditConfig,
    RepeatedBlindAuditResult,
    RepeatedBlindAuditRunner,
    RepeatedRunOutcome,
    RepeatedStageFailure,
    build_repeated_metrics,
)
from shortcutminer.protocol.model_runtime import ModelFailure
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.protocol.visual_discovery import (
    SemanticInspectionRequest, SemanticInspectionSample,
)
from shortcutminer.inspection.semantic_provider import OpenAICompatibleSemanticVisualInspector
from shortcutminer.runtime.openai_compatible import ModelRuntimeError, OpenAICompatibleClient
from shortcutminer.storage.model_calls import load_run_model_telemetry


def _config(kind="positive", **updates):
    values = {
        "snapshot_id": "snapshot-test",
        "challenge_id": f"{kind}-challenge",
        "challenge_kind": kind,
        "repeats": 3,
        "runtime_sampling_seeds": [11, 22, 33],
        "model_seeds": [101, 102, 103],
        "continue_on_run_failure": True,
        "inter_run_delay": 0.0,
    }
    values.update(updates)
    return RepeatedBlindAuditConfig.model_validate(values)


def test_three_positive_repeats_report_raw_stability_distributions(tmp_path: Path):
    calls = []

    def execute(identity):
        calls.append(identity)
        return {
            "status": "completed",
            "truth_discovered": identity.repeat_index != 2,
            "localization": ("exact", "partial", "unknown")[identity.repeat_index],
            "final_assessment": (
                "supported", "partially_supported", "inconclusive"
            )[identity.repeat_index],
            "model_telemetry_reference": "model_calls/telemetry.json",
            "result_reference": "evaluation/result.json",
        }

    result = RepeatedBlindAuditRunner(execute).run_matrix(
        _config(), artifact_root=tmp_path
    )
    assert len(calls) == 3
    assert len({item.run_id for item in calls}) == 3
    assert result.metrics.discovery.discovered_count == 2
    assert result.metrics.discovery.discovery_frequency == pytest.approx(2 / 3)
    assert result.metrics.localization.counts == {
        "exact": 1, "partial": 1, "incorrect": 0, "unknown": 1
    }
    assert result.metrics.final_assessment.counts["supported"] == 1
    assert result.metrics.runtime_reliability.run_success_frequency == 1.0


def test_clean_control_reports_false_positive_stability(tmp_path: Path):
    def execute(identity):
        index = identity.repeat_index
        return {
            "status": "completed",
            "candidate_false_positives": [1, 0, 2][index],
            "partial_final_false_positives": [0, 0, 1][index],
            "strong_final_false_positives": [0, 0, 1][index],
        }

    result = RepeatedBlindAuditRunner(execute).run_matrix(
        _config("clean_control"), artifact_root=tmp_path
    )
    metrics = result.metrics.false_positives
    assert metrics.candidate_false_positives_per_run == [1, 0, 2]
    assert metrics.partial_final_false_positives_per_run == [0, 0, 1]
    assert metrics.strong_final_false_positives_per_run == [0, 0, 1]
    assert metrics.fraction_runs_with_any_strong_final_false_positive == pytest.approx(1 / 3)


def test_failed_repeat_is_retained_and_matrix_continues(tmp_path: Path):
    def execute(identity):
        if identity.repeat_index == 1:
            raise ModelRuntimeError(
                "schema failed",
                failure=ModelFailure(
                    category="protocol_schema_failed",
                    message="schema failed",
                    retryable=False,
                ),
            )
        return {"status": "completed", "truth_discovered": True}

    result = RepeatedBlindAuditRunner(execute).run_matrix(
        _config(), artifact_root=tmp_path
    )
    assert [item.status for item in result.outcomes] == ["completed", "failed", "completed"]
    assert result.outcomes[1].failure_category == "schema_failure"
    assert result.metrics.runtime_reliability.completed_runs == 2
    assert result.metrics.runtime_reliability.failure_counts["schema_failure"] == 1


def test_resume_executes_only_missing_repeats(tmp_path: Path):
    config = _config()
    first_calls = []

    def first(identity):
        first_calls.append(identity.repeat_index)
        return {"status": "completed", "truth_discovered": True}

    runner = RepeatedBlindAuditRunner(first)
    complete = runner.run_matrix(config, artifact_root=tmp_path)
    path = tmp_path / runner.result_reference(config)
    retained = complete.outcomes[:1]
    partial = RepeatedBlindAuditResult(
        matrix_id=complete.matrix_id,
        config=config,
        completed_repeat_count=1,
        outcomes=retained,
        metrics=build_repeated_metrics(config, retained),
    )
    path.write_text(partial.model_dump_json(indent=2), encoding="utf-8")

    resumed_calls = []

    def resumed(identity):
        resumed_calls.append(identity.repeat_index)
        return {"status": "completed", "truth_discovered": True}

    result = RepeatedBlindAuditRunner(resumed).run_matrix(
        config, artifact_root=tmp_path, resume=True
    )
    assert first_calls == [0, 1, 2]
    assert resumed_calls == [1, 2]
    assert result.completed_repeat_count == 3


def test_provider_and_budget_failures_are_separate_from_scientific_results(tmp_path: Path):
    categories = ["rate_limited", "budget_exceeded"]

    def execute(identity):
        if identity.repeat_index < 2:
            category = categories[identity.repeat_index]
            raise ModelRuntimeError(
                category,
                failure=ModelFailure(
                    category=category,
                    message=category,
                    retryable=False,
                ),
            )
        return {"status": "completed", "truth_discovered": False}

    result = RepeatedBlindAuditRunner(execute).run_matrix(
        _config(), artifact_root=tmp_path
    )
    failures = result.metrics.runtime_reliability.failure_counts
    assert failures["provider_failure"] == 1
    assert failures["budget_failure"] == 1
    assert result.metrics.discovery.total_completed_runs == 1


def test_stage_failure_is_aggregated_separately(tmp_path: Path):
    def execute(identity):
        if identity.repeat_index == 0:
            raise RepeatedStageFailure("identify", "identify invocation failed")
        return {"status": "completed", "truth_discovered": False}

    result = RepeatedBlindAuditRunner(execute).run_matrix(
        _config(), artifact_root=tmp_path
    )
    assert result.metrics.runtime_reliability.failure_counts["identify_failure"] == 1


class RepeatedProductionTransport:
    def post_json(self, url, *, headers, body, timeout):
        content = body["messages"][-1]["content"]
        if isinstance(content, list):
            public = json.loads(content[0]["text"])
            sample = public["samples_in_image_order"][0]
            output = {
                "schema_version": "1.0",
                "observations": [{
                    "sample_id": sample["sample_id"],
                    "sample_reference": sample["sample_reference"],
                    "observation_type": "environment",
                    "region": None,
                    "description": "small green test image",
                    "confidence": "low",
                    "limitations": ["Controlled smoke fixture."],
                }],
            }
        else:
            output = {"ok": True}
        return {
            "choices": [{"message": {"content": json.dumps(output)}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        }


@pytest.mark.parametrize("kind", ["positive", "clean_control"])
def test_repeated_smoke_links_text_and_semantic_production_telemetry(
    tmp_path: Path, kind: str,
):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (8, 8), (20, 130, 40)).save(dataset / "tiny.png")
    runs_root = tmp_path / "runs"
    config = _config(kind)
    calls = []

    def execute(identity):
        calls.append(identity.repeat_index)
        client = OpenAICompatibleClient(
            OpenAICompatibleModelConfig(
                model="repeated-fake-model",
                base_url="https://fake.example/v1",
                api_key_env="SHORTCUTMINER_FAKE_KEY",
                capabilities={"supports_image_input": True},
                model_call_budget=config.model_call_budget,
            ),
            api_key="fake-secret",
            transport=RepeatedProductionTransport(),
            runs_root=runs_root,
        )
        semantic = OpenAICompatibleSemanticVisualInspector(client).inspect(
            dataset_root=dataset,
            request=SemanticInspectionRequest(
                run_id=identity.run_id,
                dataset_name="repeated-public-smoke",
                samples=[
                    SemanticInspectionSample(
                        sample_id="tiny", sample_reference="tiny.png"
                    )
                ],
            ),
        )
        assert semantic
        assert client.invoke_agent("identify", {"run_id": identity.run_id}) == {"ok": True}
        if kind == "positive":
            return {
                "status": "completed",
                "truth_discovered": True,
                "localization": "exact",
                "final_assessment": "supported",
                "model_telemetry_reference": "model_calls/telemetry.json",
            }
        return {
            "status": "completed",
            "candidate_false_positives": 0,
            "partial_final_false_positives": 0,
            "strong_final_false_positives": 0,
            "model_telemetry_reference": "model_calls/telemetry.json",
        }

    runner = RepeatedBlindAuditRunner(execute)
    result = runner.run_matrix(config, artifact_root=tmp_path / "matrix")
    assert calls == [0, 1, 2]
    for outcome in result.outcomes:
        telemetry = load_run_model_telemetry(runs_root, outcome.identity.run_id)
        assert telemetry.total_calls == 2
        assert telemetry.total_tokens == 14
        assert telemetry.per_agent[-1].total_calls == 1
    resumed = runner.run_matrix(config, artifact_root=tmp_path / "matrix", resume=True)
    assert resumed == result
    assert calls == [0, 1, 2]
