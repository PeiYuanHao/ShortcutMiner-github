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
    assert (
        "shortcutminer/experiment/qualitative_executor.py"
        in first.backend_hashes
    )
    assert "shortcutminer/experiment/quantitative_executor.py" in first.backend_hashes
    assert "shortcutminer/experiment/check_resolution.py" in first.backend_hashes
    assert "shortcutminer/experiment/execution_contract.py" in first.backend_hashes
    assert "shortcutminer/tooling/protocol.py" in first.backend_hashes
    assert "shortcutminer/workflow/nodes.py" in first.backend_hashes
    assert "shortcutminer/evaluation/evidence_ledger.py" in first.backend_hashes
    assert first.quantitative_check_resolution_version == (
        "quantitative_check_resolution_v1"
    )
    assert len(first.quantitative_check_resolution_hash) == 64
    assert len(first.quantitative_check_resolver_implementation_hash) == 64
    assert len(first.tool_call_draft_protocol_hash) == 64
    assert len(first.tool_call_finalizer_hash) == 64
    assert len(first.experiment_execution_contract_hash) == 64
    assert len(first.experiment_parameter_ownership_hash) == 64
    assert first.quantitative_experiment_plan_version == (
        "quantitative_experiment_plan_v1"
    )
    assert len(first.quantitative_experiment_plan_protocol_hash) == 64
    assert len(first.quantitative_experiment_plan_compiler_hash) == 64
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


def test_real_execution_stops_before_snapshot_or_provider_call_on_placeholder(
    tmp_path: Path,
):
    config = _config(tmp_path).model_copy(
        update={
            "semantic_runtime": _model(
                image=True,
                key_env="VLM_TEST_KEY",
                base="https://<VLM_API_BASE_URL>/v1",
            ).model_copy(update={"model": "<VLM_MODEL_ID>"}),
        }
    )
    transport = PilotTransport()
    text_client = OpenAICompatibleClient(
        config.text_runtime, api_key="offline", transport=transport,
    )
    semantic_client = OpenAICompatibleClient(
        config.semantic_runtime, api_key="offline", transport=transport,
    )
    runner = PilotRunner(
        config,
        text_client=text_client,
        semantic_inspector=OpenAICompatibleSemanticVisualInspector(semantic_client),
    )

    with pytest.raises(
        PilotRunnerError,
        match="VLM_MODEL_ID, VLM_API_BASE_URL",
    ):
        runner.preflight()

    assert text_client.invocations == []
    assert semantic_client.invocations == []
    assert not runner.pilot_root.exists()


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
    assert result.image_input_verified
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
    assert result.semantic_vlm.failure_category == "blocked_by_text_preflight"
    assert semantic_client.invocations == []
    assert not (tmp_path / "preflight_assets").exists()


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
    with pytest.raises(PilotRunnerError, match="legacy_task_negative_replay"):
        runner.canary()
    snapshot = runner._snapshot()
    source_id = "phase16-source-canary-gates"
    (runner.pilot_root / "legacy_task_negative_replay.json").write_text(
        json.dumps({
            "status": "passed", "semantic_gate": "rejected",
            "source_pilot_id": source_id,
        }),
        encoding="utf-8",
    )
    (runner.pilot_root / "regenerated_task_offline.json").write_text(
        json.dumps({
            "status": "passed", "provider_call_count": 0,
            "source_pilot_id": source_id, "ready_step_count": 1,
            "mandatory_backend_attempt_count": 1,
            "successful_tool_result_count": 1,
            "task": {
                "requested_checks": ["association_test"],
                "metadata_fields": ["field"],
            },
        }),
        encoding="utf-8",
    )
    saved = PilotRunRecord(
        challenge_id="pilot_001", challenge_kind="positive", repeat_index=0,
        run_id="phase16-test-canary", runtime_seed=101, bootstrap_seed=201,
        status="completed", snapshot_id=snapshot.snapshot_id,
        public_challenge_hash="a" * 64,
    )
    (runner.pilot_root / "canary.json").write_text(saved.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(runner, "_execute", lambda *args, **kwargs: pytest.fail("completed canary was repeated"))
    assert runner.canary() == saved


def test_canary_quantitative_audit_records_only_safe_execution_metadata(tmp_path: Path):
    runner = PilotRunner(_config(tmp_path))
    run_id = "phase16-test-canary"
    scope = runner.runs_root / run_id / "quantitative" / "TASK-C002-QT"
    scope.mkdir(parents=True)
    executive = runner.runs_root / run_id / "executive"
    executive.mkdir(parents=True)
    (executive / "input.json").write_text(json.dumps({
        "request": {"candidate_report": {"candidates": [{
            "candidate_id": "C002", "cue": {"family": "metadata"},
        }]}},
        "executable_candidate_contexts": [{
            "binding_status": "resolved", "binding_kind": "metadata_field",
        }],
    }), encoding="utf-8")
    (executive / "routing.json").write_text(json.dumps({"tasks": [{
        "task_id": "TASK-C002-QT", "target": "quantitative",
        "requested_checks": ["association_test"],
        "inputs": {
            "metadata_fields": ["device"],
            "required_model_capabilities": [],
            "requires_sample_access": False,
        },
    }]}), encoding="utf-8")
    (scope / "input.json").write_text(json.dumps({
        "request": {
            "task": {
                "task_id": "TASK-C002-QT",
                "requested_checks": ["association_test"],
                "inputs": {"metadata_fields": ["device"]},
            },
        },
        "execution_requirement": {
            "must_attempt_run_experiment": True,
            "reason": "task_is_executable",
            "resolved_experiment_types": ["statistical_check"],
        },
    }), encoding="utf-8")
    (scope / "result.json").write_text(json.dumps({
        "payload": {"status": "completed"},
    }), encoding="utf-8")
    (scope / "experiment_plan.json").write_text(json.dumps({
        "protocol_version": "quantitative_experiment_plan_v1",
        "steps": [
            {"step_id": "QSTEP-safe", "execution_state": "completed"},
            {"step_id": "QSTEP-pending", "execution_state": "pending_dependency"},
        ],
    }), encoding="utf-8")
    (scope / "plan_execution.json").write_text(json.dumps({
        "mandatory_executed_step_ids": ["QSTEP-safe"],
        "backend_attempt_count": 1,
        "successful_tool_result_count": 1,
        "quantitative_agent_call_count": 1,
    }), encoding="utf-8")
    tools = runner.runs_root / run_id / "tools" / "quantitative"
    tools.mkdir(parents=True)
    (tools / "experiment-001.json").write_text(json.dumps({
        "tool_name": "run_experiment",
        "task_id": "TASK-C002-QT",
        "status": "success",
    }), encoding="utf-8")

    path = runner._save_canary_quantitative_audit(run_id)
    audit = json.loads(path.read_text(encoding="utf-8"))

    assert audit["quantitative_check_resolution_version"] == (
        "quantitative_check_resolution_v1"
    )
    assert audit["quantitative_experiment_plan_version"] == (
        "quantitative_experiment_plan_v1"
    )
    assert audit["agents"] == [
        {
            "agent_name": agent_name,
            "tool_execution_count": 1 if agent_name == "quantitative" else 0,
            "successful_tool_result_count": 1 if agent_name == "quantitative" else 0,
            "failed_tool_result_count": 0,
        }
        for agent_name in (
            "identify", "executive", "quantitative", "qualitative", "summarize"
        )
    ]
    assert audit["tasks"] == [{
        "task_id": "TASK-C002-QT",
        "requested_checks": ["association_test"],
        "task_metadata_fields": ["device"],
        "resolved_experiment_types": ["statistical_check"],
        "must_attempt_run_experiment": True,
        "compiled_step_count": 2,
        "ready_step_count": 0,
        "pending_step_count": 1,
        "mandatory_executed_step_ids": ["QSTEP-safe"],
        "backend_attempt_count": 1,
        "successful_tool_result_count": 1,
        "quantitative_agent_calls": 1,
        "canonical_status": "completed",
    }]


def test_qualitative_diagnostic_recovers_completed_result_and_writes_gate(
    tmp_path: Path,
):
    from shortcutminer.experiment.qualitative_context import (
        build_qualitative_context,
    )
    from shortcutminer.experiment.qualitative_executor import (
        finalize_qualitative_draft,
    )
    from shortcutminer.protocol.model_runtime import ModelInvocationRecord
    from shortcutminer.runtime.openai_compatible import canonical_json_hash
    from shortcutminer.storage.model_calls import save_model_invocation
    from test_qualitative_context import _dataset, _request
    from test_qualitative_finalizer import _draft

    config = _config(tmp_path)
    runner = PilotRunner(config)
    runner.dry_run()
    current_snapshot = runner._snapshot()
    pilots_root = runner.pilot_root.parent
    source_pilot_id = "phase16-source"
    recovery_pilot_id = "phase16-recovery"
    source_root = pilots_root / source_pilot_id
    recovery_root = pilots_root / recovery_pilot_id
    source_root.mkdir(parents=True)
    recovery_root.mkdir(parents=True)
    (source_root / "preflight.json").write_text(
        json.dumps({"status": "passed"}), encoding="utf-8"
    )
    (source_root / "identify_diagnostic.json").write_text(
        json.dumps({"status": "passed"}), encoding="utf-8"
    )
    snapshot_json = current_snapshot.model_dump_json()
    (source_root / "snapshot.json").write_text(snapshot_json, encoding="utf-8")
    (recovery_root / "snapshot.json").write_text(
        snapshot_json, encoding="utf-8"
    )

    dataset_root = source_root / "datasets" / "test_dataset"
    _dataset(dataset_root)
    source_request = _request(dataset_root)
    source_context = build_qualitative_context(source_request)
    source_scope = (
        source_root / "runs" / f"{source_pilot_id}-canary"
        / "qualitative" / source_request.task.task_id
    )
    source_scope.mkdir(parents=True)
    source_value = {
        "request": source_request.model_dump(mode="json"),
        "evidence_context": source_context.model_dump(mode="json"),
    }
    (source_scope / "input.json").write_text(
        json.dumps(source_value), encoding="utf-8"
    )
    relative_dataset = dataset_root.relative_to(source_root).as_posix()
    (source_root / "suite_public.json").write_text(
        json.dumps({
            "challenges": [{
                "challenge_id": "test_dataset",
                "dataset_reference": relative_dataset,
                "public_dataset_hash": "c" * 64,
            }]
        }),
        encoding="utf-8",
    )
    challenge_root = source_root / "challenges" / "test_dataset"
    challenge_root.mkdir(parents=True)
    (challenge_root / "challenge.json").write_text(
        json.dumps({"public": {"generator_version": "blind_synthetic_v1"}}),
        encoding="utf-8",
    )

    recovery_run_id = f"{recovery_pilot_id}-qualitative-diagnostic"
    request = source_request.model_copy(update={
        "run_id": recovery_run_id,
        "task": source_request.task.model_copy(update={"run_id": recovery_run_id}),
    })
    context = source_context.model_copy(update={"run_id": recovery_run_id})
    canonical = finalize_qualitative_draft(_draft(), request, context)
    recovery_scope = (
        recovery_root / "runs" / recovery_run_id
        / "qualitative" / request.task.task_id
    )
    recovery_scope.mkdir(parents=True)
    current_value = {
        "request": request.model_dump(mode="json"),
        "evidence_context": context.model_dump(mode="json"),
    }
    (recovery_scope / "input.json").write_text(
        json.dumps(current_value), encoding="utf-8"
    )
    (recovery_scope / "raw_output.json").write_text(
        _draft().model_dump_json(), encoding="utf-8"
    )
    (recovery_scope / "result.json").write_text(
        canonical.model_dump_json(), encoding="utf-8"
    )
    scientific_input_hash = canonical_json_hash(current_value)
    (recovery_scope / "draft_invariant_telemetry.json").write_text(
        json.dumps({
            "scientific_input_hash": scientific_input_hash,
            "draft_invariant_retry_count": 0,
            "status": "passed",
            "attempts": [{
                "attempt": 1,
                "scientific_input_hash": scientific_input_hash,
                "draft_schema_result": "passed",
                "draft_invariant_result": "passed",
                "failure": None,
            }],
        }),
        encoding="utf-8",
    )
    record = ModelInvocationRecord.model_validate({
        "invocation_id": "qualitative-recovery-test",
        "run_id": recovery_run_id,
        "agent_name": "qualitative",
        "runtime_kind": "text_agent",
        "provider": "openai_compatible",
        "model": "offline-contract-model",
        "attempt": 1,
        "request_hash": "a" * 64,
        "system_prompt_hash": "b" * 64,
        "payload_hash": "c" * 64,
        "response_hash": "d" * 64,
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:01+00:00",
        "latency_ms": 1000.0,
        "status": "completed",
        "retryable": False,
        "usage": {
            "available": True,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "schema_valid": True,
        "response_kind": "final_result",
        "response_diagnostics": {
            "finish_reason": "stop",
            "content_present": True,
            "content_length": 100,
            "content_whitespace_only": False,
            "content_first_token_class": "object_open",
            "content_last_token_class": "object_close",
            "json_parse_status": "valid_object",
        },
    })
    save_model_invocation(
        recovery_root / "runs",
        record,
        runtime_snapshot_ids=["model-runtime-test"],
    )

    gate = runner.qualitative_diagnostic(
        source_pilot_id=source_pilot_id,
        recovery_pilot_id=recovery_pilot_id,
    )

    assert gate["status"] == "passed"
    assert gate["gate_materialization_mode"] == (
        "recovered_existing_validated_run"
    )
    assert gate["provider_calls_during_recovery"] == 0
    assert gate["draft_schema_result"] == "passed"
    assert gate["deterministic_finalization_result"] == "passed"
    assert gate["qualitative_result_model_validate"] == "passed"
    assert gate["validate_qualitative_result"] == "passed"
    assert gate["scientific_payload_hash_equal"] is True
    assert gate["scientific_input_hash_equal_across_draft_attempts"] is True
    assert json.loads(
        (runner.pilot_root / "qualitative_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )["status"] == "passed"


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
