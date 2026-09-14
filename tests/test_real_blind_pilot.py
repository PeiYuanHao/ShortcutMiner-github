"""Offline contract tests for the Phase 16 real-provider pilot layer."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from shortcutminer.evaluation.pilot.budget import PilotBudgetAccount, PilotBudgetExceeded
from shortcutminer.evaluation.pilot.preflight import RealProviderPreflight
from shortcutminer.evaluation.pilot.protocol import (
    PilotConfig, PilotPreflightResult, PilotRunRecord,
)
from shortcutminer.evaluation.pilot.runner import PilotRunner, PilotRunnerError
from shortcutminer.evaluation.pilot.realness import PilotRealnessError
from shortcutminer.evaluation.pilot.report import render_pilot_report
from shortcutminer.evaluation.pilot.snapshot import build_pilot_snapshot
from shortcutminer.evaluation.pilot.suite import build_suite
from shortcutminer.inspection.semantic_provider import OpenAICompatibleSemanticVisualInspector
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient


class PilotTransport:
    def post_json(self, _url, *, headers, body, timeout):
        content = body["messages"][-1]["content"]
        if isinstance(content, list):
            public = json.loads(content[0]["text"])
            observations = []
            for sample in public["samples_in_image_order"]:
                is_a = sample["sample_id"] == "sample-a"
                observations.append({
                    "sample_id": sample["sample_id"],
                    "sample_reference": sample["sample_reference"],
                    "observation_type": "foreground_object",
                    "region": "center",
                    "description": "yellow circular object" if is_a else "pink square object",
                    "confidence": "high",
                    "limitations": [],
                })
            value = {"schema_version": "1.0", "observations": observations}
        else:
            value = {"status": "ok"}
        return {
            "choices": [{"message": {"content": json.dumps(value)}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }


class FailingPilotTransport:
    def post_json(self, _url, *, headers, body, timeout):
        raise TimeoutError("offline timeout fixture")


def _model(*, image: bool, key_env: str, base: str) -> OpenAICompatibleModelConfig:
    return OpenAICompatibleModelConfig(
        model="offline-contract-model", base_url=base, api_key_env=key_env,
        temperature=0.0,
        capabilities={"supports_image_input": image},
        model_call_budget={"max_total_calls": 200, "max_calls_per_agent": 50},
    )


def _config(tmp_path: Path) -> PilotConfig:
    return PilotConfig(
        pilot_id="phase16-test", artifact_root=str(tmp_path / "pilots"),
        repository_root=str(Path.cwd()),
        frozen_git_commit="a" * 40,
        text_runtime=_model(image=False, key_env="TEXT_TEST_KEY", base="https://text.invalid/v1"),
        semantic_runtime=_model(image=True, key_env="VLM_TEST_KEY", base="https://vlm.invalid/v1"),
        runtime_seeds=[101, 102], bootstrap_seeds=[201, 202],
    )


def test_config_is_strict_and_requires_real_semantic_capability(tmp_path: Path):
    value = _config(tmp_path).model_dump(mode="json")
    value["unexpected"] = True
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(value)
    value.pop("unexpected")
    value["semantic_runtime"]["capabilities"]["supports_image_input"] = False
    with pytest.raises(ValidationError, match="supports_image_input"):
        PilotConfig.model_validate(value)


def test_snapshot_and_provider_combination_are_stable_and_secret_free(tmp_path: Path):
    config = _config(tmp_path)
    first = build_pilot_snapshot(config, created_at="2026-01-01T00:00:00+00:00", git_commit="a" * 40)
    second = build_pilot_snapshot(config, created_at=first.created_at, git_commit=first.git_commit)
    assert first == second
    assert first.provider_combination_id == second.provider_combination_id
    serialized = first.model_dump_json()
    assert "Bearer" not in serialized
    assert "TEXT_TEST_KEY" in serialized


def test_text_and_vlm_may_use_independent_endpoint_and_key_names(tmp_path: Path):
    config = _config(tmp_path)
    assert config.text_runtime.base_url != config.semantic_runtime.base_url
    assert config.text_runtime.api_key_env != config.semantic_runtime.api_key_env


def test_global_budget_accounts_conservatively():
    config = _config(Path("."))
    account = PilotBudgetAccount(config.budget, text_calls=165, vlm_calls=37)
    assert account.can_start_run()
    account.add(text_calls=15, vlm_calls=3)
    assert not account.can_start_run()
    with pytest.raises(PilotBudgetExceeded):
        account.add(text_calls=1)


def test_dry_run_builds_exact_suite_without_provider_calls(tmp_path: Path):
    config = _config(tmp_path)
    runner = PilotRunner(config)
    summary = runner.dry_run()
    assert summary["planned_full_runs"] == 12
    assert summary["real_provider_calls_made"] == 0
    suite = build_suite(config, runner.pilot_root)
    assert [item.challenge.challenge_id for item in suite] == [f"pilot_{i:03d}" for i in range(1, 7)]
    semantic = suite[3]
    truth = json.loads(semantic.hidden_truth_path.read_text(encoding="utf-8"))
    assert truth["shortcuts"][0]["family"] == "semantic"
    assert "semantic" not in semantic.challenge.public.dataset_reference


def test_real_preflight_checks_two_neutrally_named_images(tmp_path: Path):
    text_client = OpenAICompatibleClient(
        _model(image=False, key_env="TEXT_TEST_KEY", base="https://text.invalid/v1"),
        api_key="offline", transport=PilotTransport(),
    )
    semantic_client = OpenAICompatibleClient(
        _model(image=True, key_env="VLM_TEST_KEY", base="https://vlm.invalid/v1"),
        api_key="offline", transport=PilotTransport(),
    )
    result = RealProviderPreflight(
        text_client, OpenAICompatibleSemanticVisualInspector(semantic_client),
    ).run(pilot_id="phase16-test", pilot_root=tmp_path)
    assert result.status == "passed"
    assert result.image_content_distinguishable
    assert result.image_names == ["sample_a.png", "sample_b.png"]
    assert result.text.runtime_kind == result.semantic_vlm.runtime_kind == "real_provider"


def test_preflight_failure_blocks_instead_of_falling_back(tmp_path: Path):
    text_client = OpenAICompatibleClient(
        _model(image=False, key_env="TEXT_TEST_KEY", base="https://text.invalid/v1"),
        api_key="offline", transport=FailingPilotTransport(), sleeper=lambda _: None,
    )
    semantic_client = OpenAICompatibleClient(
        _model(image=True, key_env="VLM_TEST_KEY", base="https://vlm.invalid/v1"),
        api_key="offline", transport=FailingPilotTransport(), sleeper=lambda _: None,
    )
    result = RealProviderPreflight(
        text_client, OpenAICompatibleSemanticVisualInspector(semantic_client),
    ).run(pilot_id="phase16-test", pilot_root=tmp_path)
    assert result.status == "blocked"
    assert not result.text.success
    assert not result.semantic_vlm.success


def test_scientific_runner_rejects_non_network_transport(tmp_path: Path):
    config = _config(tmp_path)
    text_client = OpenAICompatibleClient(config.text_runtime, api_key="offline", transport=PilotTransport())
    semantic_client = OpenAICompatibleClient(config.semantic_runtime, api_key="offline", transport=PilotTransport())
    runner = PilotRunner(
        config, text_client=text_client,
        semantic_inspector=OpenAICompatibleSemanticVisualInspector(semantic_client),
    )
    with pytest.raises(PilotRealnessError, match="real HTTP transport"):
        runner._assert_real()


def test_canary_gate_is_required_and_saved_canary_is_reused(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    runner = PilotRunner(config)
    runner.dry_run()
    with pytest.raises(PilotRunnerError, match="preflight"):
        runner.canary()
    preflight = PilotPreflightResult.model_validate({
        "pilot_id": config.pilot_id, "status": "passed",
        "text": {"name": "text", "success": True, "provider": "openai_compatible", "model": "m", "runtime_kind": "real_provider", "latency_ms": 1.0, "usage_available": True, "message": "ok"},
        "semantic_vlm": {"name": "vlm", "success": True, "provider": "openai_compatible", "model": "m", "runtime_kind": "real_provider", "latency_ms": 1.0, "usage_available": True, "message": "ok"},
        "image_content_distinguishable": True,
        "image_names": ["sample_a.png", "sample_b.png"],
        "completed_at": "2026-01-01T00:00:00+00:00",
    })
    (runner.pilot_root / "preflight.json").write_text(preflight.model_dump_json(), encoding="utf-8")
    saved = PilotRunRecord(
        challenge_id="pilot_001", challenge_kind="positive", repeat_index=0,
        run_id="phase16-test-canary", runtime_seed=101, bootstrap_seed=201,
        status="completed", snapshot_id=runner._snapshot().snapshot_id,
        public_challenge_hash="a" * 64,
    )
    (runner.pilot_root / "canary.json").write_text(saved.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(runner, "_execute", lambda *args, **kwargs: pytest.fail("completed canary was repeated"))
    assert runner.canary() == saved


def test_resume_skips_every_attempted_matrix_cell(tmp_path: Path, monkeypatch):
    config = _config(tmp_path)
    runner = PilotRunner(config)
    runner.dry_run()
    preflight = {
        "schema_version": "1.0", "pilot_id": config.pilot_id, "status": "passed",
        "text": {"name": "text", "success": True, "provider": "openai_compatible", "model": "m", "runtime_kind": "real_provider", "latency_ms": 1.0, "usage_available": True, "failure_category": None, "message": "ok"},
        "semantic_vlm": {"name": "vlm", "success": True, "provider": "openai_compatible", "model": "m", "runtime_kind": "real_provider", "latency_ms": 1.0, "usage_available": True, "failure_category": None, "message": "ok"},
        "image_content_distinguishable": True, "image_names": ["sample_a.png", "sample_b.png"],
        "completed_at": "2026-01-01T00:00:00+00:00", "limitations": [],
    }
    (runner.pilot_root / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
    snapshot = runner._snapshot()
    canary = PilotRunRecord(
        challenge_id="pilot_001", challenge_kind="positive", repeat_index=0,
        run_id="phase16-test-canary", runtime_seed=101, bootstrap_seed=201,
        status="completed", snapshot_id=snapshot.snapshot_id,
        public_challenge_hash="a" * 64,
    )
    (runner.pilot_root / "canary.json").write_text(canary.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(runner, "_assert_real", lambda: None)
    monkeypatch.setattr(runner, "_seed_global_call_caps", lambda aggregate: None)
    calls = []

    def attempted(generated, *, run_id, seed, bootstrap_seed, snapshot):
        calls.append(run_id)
        kind = next(item.kind for item in config.challenges if item.challenge_id == generated.challenge.challenge_id)
        return PilotRunRecord(
            challenge_id=generated.challenge.challenge_id, challenge_kind=kind,
            repeat_index=0, run_id=run_id, runtime_seed=seed,
            bootstrap_seed=bootstrap_seed, status="failed",
            snapshot_id=snapshot.snapshot_id,
            public_challenge_hash=generated.challenge.public.public_dataset_hash,
            failure_category="provider_failure", failure_message="offline fixture",
        )

    monkeypatch.setattr(runner, "_execute", attempted)
    first = runner.run(resume=True)
    assert len(calls) == 12
    assert first.status == "incomplete"
    with pytest.raises(PilotRunnerError, match="use resume"):
        runner.run(resume=False)
    second = runner.run(resume=True)
    assert len(calls) == 12
    assert second.failed_runs == 12
    assert second.aggregate.failure_taxonomy["provider_failure"] == 12
    assert "DEVELOPMENT PILOT" in render_pilot_report(second, snapshot)


def test_production_request_has_no_hidden_truth_fields(tmp_path: Path):
    config = _config(tmp_path)
    runner = PilotRunner(config)
    generated = build_suite(config, runner.pilot_root)[3]
    request = runner._request(generated, run_id="phase16-test-public", seed=101)
    serialized = request.model_dump_json().casefold()
    assert "hidden_truth" not in serialized
    assert "t001" not in serialized
    assert str(generated.hidden_truth_path).casefold() not in serialized
