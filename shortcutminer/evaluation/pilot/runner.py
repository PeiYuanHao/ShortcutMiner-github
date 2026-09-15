"""Sequential, resumable orchestration for the Phase 16 development pilot."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from shortcutminer.evaluation.blind.evaluator import BlindEvaluator
from shortcutminer.evaluation.blind.protocol import EvaluationSnapshot
from shortcutminer.evaluation.blind.snapshot import build_evaluation_snapshot
from shortcutminer.evaluation.pilot.budget import PilotBudgetAccount, PilotBudgetExceeded
from shortcutminer.evaluation.pilot.preflight import RealProviderPreflight
from shortcutminer.evaluation.pilot.readiness import real_provider_configuration_blockers
from shortcutminer.evaluation.pilot.protocol import (
    PilotAggregate, PilotChallengeResult, PilotConfig, PilotPreflightResult,
    PilotResult, PilotRunRecord, PilotSnapshot,
)
from shortcutminer.evaluation.pilot.realness import (
    PilotRealnessError, assert_real_provider, audit_scientific_artifacts,
)
from shortcutminer.evaluation.pilot.report import save_pilot_report
from shortcutminer.evaluation.pilot.snapshot import (
    build_pilot_snapshot, save_pilot_snapshot, verify_pilot_snapshot,
)
from shortcutminer.evaluation.pilot.suite import build_suite
from shortcutminer.inspection.semantic import SemanticVisualInspector
from shortcutminer.orchestrator import build_initial_workflow_state
from shortcutminer.protocol.batches import QualitativeResultBatch, QuantitativeResultBatch
from shortcutminer.protocol.evidence import EvidenceLedger
from shortcutminer.protocol.executive import RoutingPlan
from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.protocol.summarize import SummaryReport
from shortcutminer.protocol.run import RunRequest
from shortcutminer.runner import build_local_tool_runtimes, run
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient
from shortcutminer.runtime.openai_compatible import UrllibJsonTransport
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.storage.model_calls import load_run_model_telemetry
from shortcutminer.workflow.nodes import identify_node


class PilotRunnerError(RuntimeError):
    """Raised for an immutable-pilot or orchestration violation."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 10) if values else None


def _empty_aggregate(config: PilotConfig) -> PilotAggregate:
    return PilotAggregate(
        planned_runs=len(config.challenges) * config.repeats,
        completed_runs=0, failed_runs=0, positive_truth_count=0,
        positive_truth_discovered=0, recall_at_1=None, recall_at_3=None,
        exact_localization_count=0, partial_localization_count=0,
        clean_candidate_false_positives=0,
        clean_partial_final_false_positives=0,
        clean_strong_final_false_positives=0,
        final_assessment_distribution={}, failure_taxonomy={},
        total_text_calls=0, total_vlm_calls=0, retry_calls=0,
        schema_retries=0, prompt_tokens=None, completion_tokens=None,
        total_tokens=None, usage_complete=False, total_latency_ms=0.0,
    )


class PilotRunner:
    """Own the fixed suite without choosing hypotheses, routes, or experiments."""

    def __init__(
        self,
        config: PilotConfig,
        *,
        text_client: OpenAICompatibleClient | None = None,
        semantic_inspector: SemanticVisualInspector | None = None,
        sleeper=time.sleep,
    ) -> None:
        self.config = config
        repository_root = Path(config.repository_root).resolve(strict=True)
        artifact_root = Path(config.artifact_root)
        if not artifact_root.is_absolute():
            artifact_root = repository_root / artifact_root
        self.pilot_root = artifact_root.resolve(strict=False) / config.pilot_id
        self.runs_root = self.pilot_root / "runs"
        self.text_client = text_client
        self.semantic_inspector = semantic_inspector
        self.sleeper = sleeper

    @property
    def snapshot_path(self) -> Path:
        return self.pilot_root / "snapshot.json"

    def _snapshot(self) -> PilotSnapshot:
        if self.snapshot_path.is_file():
            snapshot = PilotSnapshot.model_validate_json(self.snapshot_path.read_text(encoding="utf-8"))
            verify_pilot_snapshot(self.config, snapshot)
            return snapshot
        snapshot = build_pilot_snapshot(self.config)
        save_pilot_snapshot(snapshot, self.pilot_root)
        return snapshot

    def _require_complete_real_configuration(self) -> None:
        blockers = real_provider_configuration_blockers(self.config)
        if blockers:
            raise PilotRunnerError(
                "real provider configuration is incomplete: " + ", ".join(blockers)
            )

    def dry_run(self) -> dict[str, Any]:
        self.pilot_root.mkdir(parents=True, exist_ok=True)
        suite = build_suite(self.config, self.pilot_root)
        snapshot = self._snapshot()
        config_path = self.pilot_root / "config.json"
        serialized = self.config.model_dump(mode="json")
        if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != serialized:
            raise PilotRunnerError("stored pilot config differs from requested config")
        atomic_write_json(config_path, serialized, error_type=PilotRunnerError)
        summary = {
            "pilot_id": self.config.pilot_id,
            "snapshot_id": snapshot.snapshot_id,
            "provider_combination_id": snapshot.provider_combination_id,
            "challenge_count": len(suite),
            "positive_challenge_count": sum(item.kind == "positive" for item in self.config.challenges),
            "clean_control_count": sum(item.kind == "clean_control" for item in self.config.challenges),
            "repeat_count": self.config.repeats,
            "planned_full_runs": len(suite) * self.config.repeats,
            "max_text_calls": self.config.budget.max_total_text_model_calls,
            "max_vlm_calls": self.config.budget.max_total_vlm_calls,
            "max_vlm_images": self.config.budget.max_total_vlm_images,
            "max_tokens": self.config.budget.max_total_tokens,
            "real_provider_calls_made": 0,
            "development_status": "DEVELOPMENT PILOT - NOT FINAL PAPER RESULTS",
        }
        atomic_write_json(self.pilot_root / "dry_run.json", summary, error_type=PilotRunnerError)
        return summary

    def preflight(self) -> PilotPreflightResult:
        self._require_complete_real_configuration()
        self.dry_run()
        if self.text_client is None or self.semantic_inspector is None:
            raise PilotRunnerError("real text and semantic runtimes are required for preflight")
        return RealProviderPreflight(self.text_client, self.semantic_inspector).run(
            pilot_id=self.config.pilot_id, pilot_root=self.pilot_root,
        )

    def _require_gate(self, name: str) -> dict[str, Any]:
        path = self.pilot_root / name
        if not path.is_file():
            raise PilotRunnerError(f"required pilot gate is missing: {name}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _assert_real(self) -> None:
        if self.text_client is None or self.semantic_inspector is None:
            raise PilotRealnessError("scientific execution requires both real providers")
        assert_real_provider(self.text_client, role="text runtime")
        assert_real_provider(self.semantic_inspector, role="semantic runtime")
        clients = (self.text_client, getattr(self.semantic_inspector, "client", None))
        if clients[0] is clients[1]:
            raise PilotRealnessError(
                "text and semantic runtimes require independent client instances"
            )
        if any(
            isinstance(client, OpenAICompatibleClient)
            and not isinstance(getattr(client, "_transport", None), UrllibJsonTransport)
            for client in clients if client is not None
        ):
            raise PilotRealnessError(
                "scientific OpenAI-compatible execution requires the real HTTP transport"
            )

    def _seed_global_call_caps(self, aggregate: PilotAggregate) -> None:
        """Make frozen runtime attempt counters enforce the stricter pilot caps."""
        assert self.text_client is not None and self.semantic_inspector is not None
        semantic_client = getattr(self.semantic_inspector, "client", None)
        pairs = (
            (self.text_client, self.config.budget.max_total_text_model_calls, aggregate.total_text_calls),
            (semantic_client, self.config.budget.max_total_vlm_calls, aggregate.total_vlm_calls),
        )
        for client, pilot_limit, already_used in pairs:
            if not isinstance(client, OpenAICompatibleClient):
                continue
            runtime_limit = client.config.model_call_budget.max_total_calls
            offset = max(0, runtime_limit - pilot_limit)
            with client._budget_state.lock:
                client._budget_state.calls_total = offset + already_used

    def _request(self, generated, *, run_id: str, seed: int) -> RunRequest:
        challenge = generated.challenge
        max_rows = max(100, next(
            item.sample_count for item in self.config.challenges
            if item.challenge_id == challenge.challenge_id
        ))
        return RunRequest(
            run_id=run_id,
            runs_root=str(self.runs_root),
            dataset_info={
                "name": challenge.challenge_id,
                "root": str(generated.public_dataset_root),
                "metadata_file": challenge.public.metadata_file,
            },
            task_input=challenge.public.task,
            seed=seed,
            model_runtime=self.config.text_runtime,
            identify_config={
                "budget": {
                    "max_samples": self.config.visual_sample_budget,
                    "max_candidates": self.config.candidate_limit,
                    "max_dataset_entries": max_rows,
                    "max_metadata_rows": max_rows,
                },
                "visual_discovery": {
                    "max_samples": self.config.visual_sample_budget,
                    "sample_id_column": "id", "sample_path_column": "path",
                    "semantic_enabled": True,
                },
            },
            executive_config={
                "audit_options": {
                    "validation_depth": self.config.validation_depth,
                    "max_tasks": self.config.budget.max_tool_calls_per_run,
                    "max_samples_per_task": max_rows,
                    "allowed_targets": ["quantitative", "qualitative"],
                    "runtime_constraints": ["no_external_services"],
                },
                "capabilities": {
                    "available_targets": ["quantitative", "qualitative"],
                    "available_metadata_fields": ["id", "label", "site", "device", "batch", "measure", "path"],
                    "quantitative_checks": [
                        "association_test", "task_model_training", "task_model_evaluation",
                        "ood_evaluation", "counterfactual_intervention",
                    ],
                    "qualitative_checks": ["metadata_inspection", "sample_inspection"],
                    "model_capabilities": ["retraining", "checkpoint", "intervention_inference", "predictions", "logits"],
                    "sample_access": True, "external_services": False,
                    "available_output_types": ["metrics", "observations"],
                },
            },
            qualitative_config={
                "preparation": {
                    "seed": seed, "sampling_strategy": "group_balanced",
                    "sample_id_column": "id", "sample_path_column": "path",
                    "group_columns": ["label"],
                    "max_evidence_records": self.config.semantic_sample_budget,
                    "max_metadata_rows": max_rows,
                },
            },
            metadata={
                "evaluation_mode": "blind_public_only", "pilot_id": self.config.pilot_id,
                "human_runtime_intervention": False,
            },
        )

    def _execute(self, generated, *, run_id: str, seed: int,
                 bootstrap_seed: int, snapshot: PilotSnapshot) -> PilotRunRecord:
        self._assert_real()
        verify_pilot_snapshot(self.config, snapshot)
        request = self._request(generated, run_id=run_id, seed=seed)
        runtimes = build_local_tool_runtimes(
            run_id=run_id, dataset_root=generated.public_dataset_root,
            runs_root=self.runs_root,
        )
        execution = run(
            request, agent_client=self.text_client, tool_runtimes=runtimes,
            semantic_inspector=self.semantic_inspector,
        )
        state = execution.workflow_state
        evaluation_snapshot = build_evaluation_snapshot(
            repository_root=self.config.repository_root,
            git_commit=snapshot.git_commit,
            runtime_config={
                "pilot_snapshot_id": snapshot.snapshot_id, "seed": seed,
                "semantic_runtime_snapshot_id": snapshot.semantic_runtime.snapshot_id,
            },
            model_config=self.config.text_runtime.model_dump(mode="json"),
            model_runtime_snapshot=snapshot.text_runtime.model_dump(mode="json"),
            challenge_public_hash=generated.challenge.public.public_dataset_hash,
            runtime_sampling_seed=seed, evaluation_bootstrap_seed=bootstrap_seed,
            candidate_limit=self.config.candidate_limit,
            validation_depth=self.config.validation_depth,
            budget_settings=self.config.budget.model_dump(mode="json"),
        ).model_copy(update={"snapshot_id": snapshot.snapshot_id})
        evaluated = BlindEvaluator().evaluate(
            challenge=generated.challenge,
            challenge_root=generated.challenge_root,
            hidden_truth_path=generated.hidden_truth_path,
            snapshot=EvaluationSnapshot.model_validate(evaluation_snapshot),
            run_id=run_id,
            candidate_report=CandidateReport.model_validate(state["identify_result"]),
            routing_plan=RoutingPlan.model_validate(state["executive_result"]),
            quantitative_results=QuantitativeResultBatch.model_validate(state["quantitative_result"]),
            qualitative_results=QualitativeResultBatch.model_validate(state["qualitative_result"]),
            evidence_ledger=EvidenceLedger.model_validate(state["evidence_ledger"]),
            summary_report=SummaryReport.model_validate(state["summarize_result"]),
            runs_root=self.runs_root,
        )
        challenge_kind = next(
            item.kind for item in self.config.challenges
            if item.challenge_id == generated.challenge.challenge_id
        )
        return PilotRunRecord(
            challenge_id=generated.challenge.challenge_id,
            challenge_kind=challenge_kind,
            repeat_index=0, run_id=run_id, runtime_seed=seed,
            bootstrap_seed=bootstrap_seed, status="completed",
            snapshot_id=snapshot.snapshot_id,
            public_challenge_hash=generated.challenge.public.public_dataset_hash,
            result_reference=f"runs/{run_id}/evaluation/blind_result.json",
            telemetry_reference=f"runs/{run_id}/model_calls/telemetry.json",
            evaluation=evaluated,
        )

    def canary(self) -> PilotRunRecord:
        self._require_complete_real_configuration()
        self.dry_run()
        preflight = PilotPreflightResult.model_validate(self._require_gate("preflight.json"))
        if preflight.status != "passed":
            raise PilotRunnerError("real-provider preflight did not pass")
        diagnostic = self._require_gate("identify_diagnostic.json")
        if diagnostic.get("status") != "passed":
            raise PilotRunnerError("real Identify diagnostic did not pass")
        snapshot = self._snapshot()
        generated = build_suite(self.config, self.pilot_root)[0]
        path = self.pilot_root / "canary.json"
        if path.is_file():
            return PilotRunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        record = self._execute(
            generated, run_id=f"{self.config.pilot_id}-canary",
            seed=self.config.runtime_seeds[0],
            bootstrap_seed=self.config.bootstrap_seeds[0], snapshot=snapshot,
        )
        atomic_write_json(path, record.model_dump(mode="json"), error_type=PilotRunnerError)
        return record

    def identify_diagnostic(self) -> dict[str, Any]:
        """Run only the real Identify path after preflight, with safe diagnostics."""
        self._require_complete_real_configuration()
        self.dry_run()
        preflight = PilotPreflightResult.model_validate(self._require_gate("preflight.json"))
        if preflight.status != "passed":
            raise PilotRunnerError("real-provider preflight did not pass")
        self._assert_real()
        self._snapshot()
        path = self.pilot_root / "identify_diagnostic.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))

        generated = build_suite(self.config, self.pilot_root)[0]
        run_id = f"{self.config.pilot_id}-identify-diagnostic"
        request = self._request(
            generated,
            run_id=run_id,
            seed=self.config.runtime_seeds[0],
        )
        state = build_initial_workflow_state(request, run_id)
        runtimes = build_local_tool_runtimes(
            run_id=run_id,
            dataset_root=generated.public_dataset_root,
            runs_root=self.runs_root,
        )
        text_start = len(self.text_client.invocations)
        semantic_client = getattr(self.semantic_inspector, "client", None)
        semantic_start = (
            len(semantic_client.invocations) if semantic_client is not None else 0
        )
        error: Exception | None = None
        try:
            output = identify_node(
                state,
                client=self.text_client,
                tool_runtime=runtimes["identify"],
                semantic_inspector=self.semantic_inspector,
            )
            CandidateReport.model_validate(output["identify_result"])
        except Exception as exc:
            error = exc

        text_records = self.text_client.invocations[text_start:]
        semantic_records = (
            semantic_client.invocations[semantic_start:]
            if semantic_client is not None else []
        )
        final_record = text_records[-1] if text_records else None
        attempts = [
            {
                "attempt": item.attempt,
                "status": item.status,
                "request_hash": item.request_hash,
                "payload_hash": item.payload_hash,
                "finish_reason": (
                    item.response_diagnostics.finish_reason
                    if item.response_diagnostics is not None else "missing"
                ),
                "content_present": (
                    item.response_diagnostics.content_present
                    if item.response_diagnostics is not None else False
                ),
                "content_length": (
                    item.response_diagnostics.content_length
                    if item.response_diagnostics is not None else 0
                ),
                "content_whitespace_only": (
                    item.response_diagnostics.content_whitespace_only
                    if item.response_diagnostics is not None else False
                ),
                "content_first_token_class": (
                    item.response_diagnostics.content_first_token_class
                    if item.response_diagnostics is not None else "empty"
                ),
                "content_last_token_class": (
                    item.response_diagnostics.content_last_token_class
                    if item.response_diagnostics is not None else "empty"
                ),
                "json_parse_status": (
                    item.response_diagnostics.json_parse_status
                    if item.response_diagnostics is not None else "content_missing"
                ),
                "completion_tokens": item.usage.completion_tokens,
                "reasoning_tokens": item.usage.reasoning_tokens,
                "latency_ms": item.latency_ms,
                "schema_retry": item.schema_retry,
                "transport_retry": item.transport_retry,
                "structured_output_mode": item.structured_output_mode,
            }
            for item in text_records
        ]
        payload_hashes = [item.payload_hash for item in text_records]
        result = {
            "schema_version": "1.0",
            "pilot_id": self.config.pilot_id,
            "run_id": run_id,
            "status": "passed" if error is None else "failed",
            "attempt_count": len(text_records),
            "valid_json": bool(
                final_record is not None
                and final_record.response_diagnostics is not None
                and final_record.response_diagnostics.json_parse_status == "valid_object"
            ),
            "candidate_report_schema_valid": error is None,
            "schema_retry_count": sum(item.schema_retry for item in text_records),
            "transport_retry_count": sum(item.transport_retry for item in text_records),
            "latency_ms": round(sum(item.latency_ms for item in text_records), 6),
            "vlm_latency_ms": round(sum(item.latency_ms for item in semantic_records), 6),
            "usage_available": bool(text_records) and all(
                item.usage.available for item in text_records
            ),
            "response_format_sent": bool(text_records) and all(
                item.structured_output_mode == "json_object" for item in text_records
            ),
            "scientific_payload_hash_equal": (
                bool(payload_hashes) and len(set(payload_hashes)) == 1
            ),
            "attempts": attempts,
            "error_type": type(error).__name__ if error is not None else None,
            "error_message": (
                "Identify diagnostic failed; inspect only the saved safe metadata."
                if error is not None else None
            ),
        }
        atomic_write_json(path, result, error_type=PilotRunnerError)
        return result

    def _load_records(self) -> list[PilotRunRecord]:
        path = self.pilot_root / "run_records.json"
        if not path.is_file():
            return []
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            return [PilotRunRecord.model_validate(item) for item in values]
        except Exception as exc:
            raise PilotRunnerError("stored pilot run records are invalid") from exc

    def _save_records(self, records: list[PilotRunRecord]) -> None:
        atomic_write_json(
            self.pilot_root / "run_records.json",
            [item.model_dump(mode="json") for item in records],
            error_type=PilotRunnerError,
        )

    def _aggregate(self, records: list[PilotRunRecord]) -> PilotAggregate:
        completed = [item for item in records if item.status == "completed" and item.evaluation]
        failed = [item for item in records if item.status in {"failed", "skipped_budget"}]
        positive = [item for item in completed if item.challenge_kind == "positive"]
        clean = [item for item in completed if item.challenge_kind == "clean_control"]
        assessments = Counter(
            judgement.final_assessment
            for item in completed for judgement in item.evaluation.discovery_judgements
            if judgement.final_assessment is not None
        )
        failures = Counter(
            failure.reason for item in completed for failure in item.evaluation.failures
        )
        failures.update(item.failure_category or "unknown_failure" for item in failed)
        telemetry = [
            value for item in records
            if (value := load_run_model_telemetry(self.runs_root, item.run_id)) is not None
        ]
        invocation_records = []
        for item in records:
            directory = self.runs_root / item.run_id / "model_calls"
            for path in directory.glob("*.json") if directory.is_dir() else []:
                if path.name == "telemetry.json" or path.name.startswith("runtime_snapshot-"):
                    continue
                try:
                    invocation_records.append(json.loads(path.read_text(encoding="utf-8")))
                except Exception:
                    pass
        usage_complete = bool(telemetry) and all(item.usage_complete for item in telemetry)
        recall_1 = [
            value for item in positive
            if (value := item.evaluation.discovery_metrics.recall_at_k.get("1")) is not None
        ]
        recall_3 = [
            value for item in positive
            if (value := item.evaluation.discovery_metrics.recall_at_k.get("3")) is not None
        ]
        return PilotAggregate(
            planned_runs=len(self.config.challenges) * self.config.repeats,
            completed_runs=len(completed), failed_runs=len(failed),
            positive_truth_count=sum(item.evaluation.discovery_metrics.truth_count for item in positive),
            positive_truth_discovered=sum(item.evaluation.discovery_metrics.matched_truth_count for item in positive),
            recall_at_1=_mean(recall_1), recall_at_3=_mean(recall_3),
            exact_localization_count=sum(item.evaluation.localization_metrics.exact_count for item in positive),
            partial_localization_count=sum(item.evaluation.localization_metrics.partial_count for item in positive),
            clean_candidate_false_positives=sum(item.evaluation.false_positive_metrics.false_positive_candidate_count for item in clean),
            clean_partial_final_false_positives=sum(item.evaluation.false_positive_metrics.partial_false_positive_findings_count for item in clean),
            clean_strong_final_false_positives=sum(item.evaluation.false_positive_metrics.false_positive_findings_count for item in clean),
            final_assessment_distribution=dict(assessments), failure_taxonomy=dict(failures),
            total_text_calls=sum(record.get("runtime_kind") == "text_agent" for record in invocation_records),
            total_vlm_calls=sum(record.get("runtime_kind") == "semantic_vlm" for record in invocation_records),
            retry_calls=sum(item.retry_calls for item in telemetry),
            schema_retries=sum(item.schema_retry_calls for item in telemetry),
            prompt_tokens=sum(item.prompt_tokens or 0 for item in telemetry) if usage_complete else None,
            completion_tokens=sum(item.completion_tokens or 0 for item in telemetry) if usage_complete else None,
            total_tokens=sum(item.total_tokens or 0 for item in telemetry) if usage_complete else None,
            usage_complete=usage_complete,
            total_latency_ms=round(sum(item.total_model_latency_ms for item in telemetry), 6),
        )

    def _challenge_results(self, records: list[PilotRunRecord]) -> list[PilotChallengeResult]:
        results = []
        for challenge in self.config.challenges:
            selected = [item for item in records if item.challenge_id == challenge.challenge_id]
            completed = [item for item in selected if item.status == "completed" and item.evaluation]
            results.append(PilotChallengeResult(
                challenge_id=challenge.challenge_id, challenge_kind=challenge.kind,
                runs=selected, successful_runs=len(completed),
                failed_runs=sum(item.status in {"failed", "skipped_budget"} for item in selected),
                truth_discovery_count=sum(item.evaluation.discovery_metrics.matched_truth_count for item in completed),
                candidate_false_positive_count=sum(item.evaluation.false_positive_metrics.false_positive_candidate_count for item in completed),
                strong_final_false_positive_count=sum(item.evaluation.false_positive_metrics.false_positive_findings_count for item in completed),
            ))
        return results

    def _result(self, records: list[PilotRunRecord], *, started_at: str) -> PilotResult:
        aggregate = self._aggregate(records)
        all_attempted = len(records) == aggregate.planned_runs
        fully_successful = aggregate.completed_runs == aggregate.planned_runs
        status = "completed" if fully_successful else "incomplete" if records else "planned"
        conclusion = "PASS" if fully_successful else "BLOCKED"
        snapshot = self._snapshot()
        return PilotResult(
            pilot_id=self.config.pilot_id, status=status, phase_conclusion=conclusion,
            development_status="DEVELOPMENT PILOT - NOT FINAL PAPER RESULTS",
            snapshot_id=snapshot.snapshot_id,
            provider_combination_id=snapshot.provider_combination_id,
            started_at=started_at,
            completed_at=_now() if all_attempted else None,
            challenge_count=len(self.config.challenges), repeat_count=self.config.repeats,
            completed_runs=aggregate.completed_runs, failed_runs=aggregate.failed_runs,
            challenge_results=self._challenge_results(records), aggregate=aggregate,
            preflight_reference="preflight.json", canary_reference="canary.json",
            limitations=[
                "DEVELOPMENT PILOT - NOT FINAL PAPER RESULTS",
                "No scientific performance threshold is used for the phase conclusion.",
            ],
        )

    def run(self, *, resume: bool = True) -> PilotResult:
        self._require_complete_real_configuration()
        self.dry_run()
        preflight = PilotPreflightResult.model_validate(self._require_gate("preflight.json"))
        if preflight.status != "passed":
            raise PilotRunnerError("scientific pilot is blocked by failed preflight")
        canary = PilotRunRecord.model_validate(self._require_gate("canary.json"))
        if canary.status != "completed":
            raise PilotRunnerError("scientific pilot is blocked by failed canary")
        self._assert_real()
        snapshot = self._snapshot()
        generated = {item.challenge.challenge_id: item for item in build_suite(self.config, self.pilot_root)}
        stored_records = self._load_records()
        if stored_records and not resume:
            raise PilotRunnerError("pilot records already exist; use resume")
        records = stored_records if resume else []
        expected = {
            (challenge.challenge_id, repeat_index)
            for challenge in self.config.challenges for repeat_index in range(self.config.repeats)
        }
        actual = {(item.challenge_id, item.repeat_index) for item in records}
        if not actual <= expected or len(actual) != len(records):
            raise PilotRunnerError("stored run identity does not match frozen pilot matrix")
        aggregate = self._aggregate(records)
        account = PilotBudgetAccount(
            self.config.budget, text_calls=aggregate.total_text_calls,
            vlm_calls=aggregate.total_vlm_calls,
        )
        self._seed_global_call_caps(aggregate)
        result_path = self.pilot_root / "result.json"
        started_at = _now()
        if result_path.is_file():
            try:
                started_at = PilotResult.model_validate_json(
                    result_path.read_text(encoding="utf-8")
                ).started_at
            except Exception as exc:
                raise PilotRunnerError("stored pilot result is invalid") from exc
        for challenge in self.config.challenges:
            for repeat_index in range(self.config.repeats):
                if (challenge.challenge_id, repeat_index) in actual:
                    continue
                verify_pilot_snapshot(self.config, snapshot)
                run_id = f"{self.config.pilot_id}-{challenge.challenge_id}-r{repeat_index + 1:02d}"
                seed = self.config.runtime_seeds[repeat_index]
                bootstrap = self.config.bootstrap_seeds[repeat_index]
                try:
                    account.require_run_capacity()
                    record = self._execute(
                        generated[challenge.challenge_id], run_id=run_id, seed=seed,
                        bootstrap_seed=bootstrap, snapshot=snapshot,
                    ).model_copy(update={"repeat_index": repeat_index})
                except PilotBudgetExceeded as exc:
                    record = PilotRunRecord(
                        challenge_id=challenge.challenge_id, challenge_kind=challenge.kind,
                        repeat_index=repeat_index, run_id=run_id, runtime_seed=seed,
                        bootstrap_seed=bootstrap, status="skipped_budget",
                        snapshot_id=snapshot.snapshot_id,
                        public_challenge_hash=generated[challenge.challenge_id].challenge.public.public_dataset_hash,
                        failure_category="global_budget_exceeded", failure_message=str(exc),
                    )
                except (PilotRealnessError, PilotRunnerError) as exc:
                    record = PilotRunRecord(
                        challenge_id=challenge.challenge_id, challenge_kind=challenge.kind,
                        repeat_index=repeat_index, run_id=run_id, runtime_seed=seed,
                        bootstrap_seed=bootstrap, status="failed",
                        snapshot_id=snapshot.snapshot_id,
                        public_challenge_hash=generated[challenge.challenge_id].challenge.public.public_dataset_hash,
                        failure_category="protocol_failure", failure_message=f"{type(exc).__name__}: {exc}",
                    )
                    records.append(record)
                    self._save_records(records)
                    result = self._result(records, started_at=started_at).model_copy(
                        update={"status": "failed", "phase_conclusion": "FAIL"}
                    )
                    atomic_write_json(result_path, result.model_dump(mode="json"), error_type=PilotRunnerError)
                    save_pilot_report(result, snapshot, self.pilot_root)
                    raise
                except Exception as exc:
                    record = PilotRunRecord(
                        challenge_id=challenge.challenge_id, challenge_kind=challenge.kind,
                        repeat_index=repeat_index, run_id=run_id, runtime_seed=seed,
                        bootstrap_seed=bootstrap, status="failed",
                        snapshot_id=snapshot.snapshot_id,
                        public_challenge_hash=generated[challenge.challenge_id].challenge.public.public_dataset_hash,
                        failure_category="technical_failure",
                        failure_message=f"{type(exc).__name__}: {exc}",
                    )
                records.append(record)
                actual.add((challenge.challenge_id, repeat_index))
                self._save_records(records)
                result = self._result(records, started_at=started_at)
                atomic_write_json(result_path, result.model_dump(mode="json"), error_type=PilotRunnerError)
                if record.status == "skipped_budget":
                    save_pilot_report(result, snapshot, self.pilot_root)
                    return result
                if record.status == "failed" and not self.config.continue_on_failure:
                    save_pilot_report(result, snapshot, self.pilot_root)
                    return result
                if self.config.inter_run_delay_seconds:
                    self.sleeper(float(self.config.inter_run_delay_seconds))
                aggregate = self._aggregate(records)
                account = PilotBudgetAccount(
                    self.config.budget, text_calls=aggregate.total_text_calls,
                    vlm_calls=aggregate.total_vlm_calls,
                )
        result = self._result(records, started_at=started_at)
        findings = audit_scientific_artifacts(self.runs_root)
        if findings:
            result = result.model_copy(update={
                "status": "failed", "phase_conclusion": "FAIL",
                "limitations": [*result.limitations, *findings],
            })
        atomic_write_json(result_path, result.model_dump(mode="json"), error_type=PilotRunnerError)
        save_pilot_report(result, snapshot, self.pilot_root)
        return result

    resume = run

    def report(self) -> Path:
        result = PilotResult.model_validate(self._require_gate("result.json"))
        return save_pilot_report(result, self._snapshot(), self.pilot_root)


__all__ = ["PilotRunner", "PilotRunnerError"]
