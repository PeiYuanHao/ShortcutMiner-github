"""Sequential, resumable orchestration for the Phase 16 development pilot."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from pydantic import ValidationError

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
    build_pilot_snapshot, file_hash, save_pilot_snapshot, verify_pilot_snapshot,
)
from shortcutminer.evaluation.pilot.suite import build_suite
from shortcutminer.experiment.qualitative_executor import (
    MAX_DRAFT_INVARIANT_RETRIES,
    QualitativeDraftValidationError,
    execute_qualitative,
    finalize_qualitative_draft,
    validate_qualitative_result,
)
from shortcutminer.experiment.qualitative_context import QualitativeEvidenceContext
from shortcutminer.experiment.check_resolution import (
    EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES,
    QUANTITATIVE_CHECK_RESOLUTION_HASH,
    QUANTITATIVE_CHECK_RESOLUTION_VERSION,
)
from shortcutminer.experiment.check_requirements import (
    AnalysisTaskSemanticError,
    build_candidate_contexts,
    validate_analysis_task_semantics,
)
from shortcutminer.experiment.executive_executor import (
    finalize_executive_draft,
    validate_routing_plan,
)
from shortcutminer.experiment.quantitative_executor import (
    determine_quantitative_execution_requirement,
    execute_quantitative,
    finalize_quantitative_result,
    validate_quantitative_result,
)
from shortcutminer.experiment.quantitative_context import build_quantitative_request
from shortcutminer.experiment.quantitative_plan import (
    compile_quantitative_experiment_plan,
    primary_mandatory_steps,
)
from shortcutminer.protocol.quantitative_plan import (
    QUANTITATIVE_EXPERIMENT_PLAN_VERSION,
)
from shortcutminer.inspection.semantic import SemanticVisualInspector
from shortcutminer.orchestrator import build_initial_workflow_state
from shortcutminer.protocol.batches import QualitativeResultBatch, QuantitativeResultBatch
from shortcutminer.protocol.evidence import EvidenceLedger
from shortcutminer.protocol.executive import (
    ExecutiveAgentDraft,
    ExecutiveAuditOptions,
    ExecutiveCapabilities,
    ExecutiveRequest,
    ExecutiveRoutingContext,
    RoutingPlan,
)
from shortcutminer.protocol.identify import (
    Candidate,
    CandidateReport,
    Cue,
    DatasetInspectionSummary,
    DatasetSpec,
    TaskSpec,
)
from shortcutminer.protocol.model_runtime import ModelInvocationRecord
from shortcutminer.protocol.qualitative import (
    QualitativeAgentDraft,
    QualitativeRequest,
    QualitativeResult,
)
from shortcutminer.protocol.quantitative import (
    QuantitativeAgentDraft,
    QuantitativeRequest,
    QuantitativeResult,
)
from shortcutminer.protocol.summarize import SummaryReport
from shortcutminer.protocol.run import RunRequest
from shortcutminer.runner import build_local_tool_runtimes, run
from shortcutminer.runtime.openai_compatible import (
    OpenAICompatibleClient, UrllibJsonTransport, canonical_json_hash,
)
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.storage.model_calls import load_run_model_telemetry
from shortcutminer.tooling.agent_loop import finalize_tool_call_draft
from shortcutminer.tooling.builtins import ExperimentToolInput, ExperimentToolOutput
from shortcutminer.workflow.nodes import identify_node


class PilotRunnerError(RuntimeError):
    """Raised for an immutable-pilot or orchestration violation."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 10) if values else None


def _qualitative_scientific_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Remove only run/message identity before comparing diagnostic evidence."""
    request = value["request"]
    context = value["evidence_context"]
    task = request["task"]
    return {
        "candidate": request["candidate"],
        "task": {
            key: task[key]
            for key in (
                "task_id", "candidate_id", "hypothesis", "objective", "inputs",
                "requested_checks", "expected_outputs", "priority", "dependencies",
                "status",
            )
        },
        "metadata_file": request["metadata_file"],
        "preparation": request["preparation"],
        "limitations": request["limitations"],
        "evidence_context": {
            key: item for key, item in context.items() if key != "run_id"
        },
    }


def _qualitative_invariant_category(exc: Exception) -> str:
    """Map value-free executor messages to deterministic diagnostic categories."""
    if isinstance(exc, QualitativeDraftValidationError):
        return exc.category
    message = str(exc)
    mappings = (
        ("sample_count does not match sample_ids", "sampling_count_mismatch"),
        ("group counts do not match", "sampling_group_mismatch"),
        ("unauthorized metadata field", "unauthorized_metadata_field"),
        ("unselected sample", "unknown_sample_reference"),
        ("unknown sample", "unknown_sample_reference"),
        ("uninspected sample", "unknown_sample_reference"),
        ("omitted evidence-context limitations", "missing_context_limitation"),
        ("artifact", "artifact_consistency_failure"),
        ("observation IDs", "observation_id_invariant_failure"),
        ("sampling strategy", "sampling_strategy_mismatch"),
        ("sampling seed", "sampling_seed_mismatch"),
    )
    for fragment, category in mappings:
        if fragment in message:
            return category
    return "qualitative_invariant_failed"


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

    def _source_pilot_root(self, source_pilot_id: str) -> Path:
        if (
            not source_pilot_id
            or len(source_pilot_id) > 128
            or any(
                character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
                for character in source_pilot_id
            )
        ):
            raise PilotRunnerError("source pilot ID is unsafe")
        if source_pilot_id == self.config.pilot_id:
            raise PilotRunnerError("source pilot must differ from the remediation pilot")
        parent = self.pilot_root.parent.resolve(strict=False)
        source = (parent / source_pilot_id).resolve(strict=False)
        try:
            source.relative_to(parent)
        except ValueError as exc:
            raise PilotRunnerError("source pilot escaped the artifact root") from exc
        if not source.is_dir():
            raise PilotRunnerError("source pilot artifacts are missing")
        return source

    def _require_source_provider_gates(self, source_pilot_id: str) -> Path:
        source = self._source_pilot_root(source_pilot_id)
        preflight = json.loads((source / "preflight.json").read_text(encoding="utf-8"))
        identify = json.loads(
            (source / "identify_diagnostic.json").read_text(encoding="utf-8")
        )
        source_snapshot = PilotSnapshot.model_validate_json(
            (source / "snapshot.json").read_text(encoding="utf-8")
        )
        if preflight.get("status") != "passed" or identify.get("status") != "passed":
            raise PilotRunnerError("source pilot provider gates did not pass")
        if (
            source_snapshot.provider_combination_id
            != self._snapshot().provider_combination_id
        ):
            raise PilotRunnerError(
                "source pilot provider combination differs from remediation runtime"
            )
        return source

    def _require_hardened_source_gates(self, source_pilot_id: str) -> Path:
        """Accept direct provider gates or a validated Qualitative recovery chain."""

        source = self._source_pilot_root(source_pilot_id)
        snapshot = PilotSnapshot.model_validate_json(
            (source / "snapshot.json").read_text(encoding="utf-8")
        )
        if (
            snapshot.provider_combination_id
            != self._snapshot().provider_combination_id
        ):
            raise PilotRunnerError(
                "source pilot provider combination differs from current runtime"
            )
        direct = all(
            (source / name).is_file()
            for name in ("preflight.json", "identify_diagnostic.json")
        )
        if direct:
            self._require_source_provider_gates(source_pilot_id)
        else:
            qualitative_path = source / "qualitative_diagnostic.json"
            if not qualitative_path.is_file():
                raise PilotRunnerError(
                    "source pilot has neither direct provider gates nor a "
                    "Qualitative recovery gate"
                )
            qualitative = json.loads(
                qualitative_path.read_text(encoding="utf-8")
            )
            if (
                qualitative.get("status") != "passed"
                or qualitative.get("provider_calls_during_recovery") != 0
                or not qualitative.get("source_pilot_id")
            ):
                raise PilotRunnerError(
                    "source Qualitative recovery gate is not readiness-qualified"
                )
            self._require_source_provider_gates(
                qualitative["source_pilot_id"]
            )
        return source

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
                    "quantitative_checks": list(
                        EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES
                    ),
                    "qualitative_checks": ["metadata_inspection", "sample_inspection"],
                    "model_capabilities": ["retraining", "checkpoint", "intervention_inference", "predictions", "logits"],
                    "sample_access": True, "external_services": False,
                    "available_output_types": ["metrics", "observations"],
                },
            },
            quantitative_config={
                "execution_resources": {
                    "seed": seed,
                    "sample_id_column": "id",
                    "sample_path_column": "path",
                    "max_rows": max_rows,
                    "max_samples": max_rows,
                    "max_permutations": 100,
                    "bootstrap_iterations": 200,
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

    def _recover_qualitative_diagnostic(
        self,
        *,
        source_pilot_id: str,
        recovery_pilot_id: str,
    ) -> dict[str, Any]:
        """Materialize a gate from one already completed, validated run."""
        source = self._require_source_provider_gates(source_pilot_id)
        recovery = self._source_pilot_root(recovery_pilot_id)
        recovery_snapshot = PilotSnapshot.model_validate_json(
            (recovery / "snapshot.json").read_text(encoding="utf-8")
        )
        if (
            recovery_snapshot.provider_combination_id
            != self._snapshot().provider_combination_id
        ):
            raise PilotRunnerError(
                "recovery pilot provider combination differs from current runtime"
            )

        source_inputs = sorted(
            (
                source / "runs" / f"{source_pilot_id}-canary"
                / "qualitative"
            ).glob("*/input.json")
        )
        recovery_run_id = f"{recovery_pilot_id}-qualitative-diagnostic"
        recovery_runs_root = recovery / "runs"
        recovery_scope = recovery_runs_root / recovery_run_id / "qualitative"
        input_paths = sorted(recovery_scope.glob("*/input.json"))
        draft_paths = sorted(recovery_scope.glob("*/raw_output.json"))
        invariant_paths = sorted(
            recovery_scope.glob("*/draft_invariant_telemetry.json")
        )
        result_paths = sorted(recovery_scope.glob("*/result.json"))
        if not all(
            len(paths) == 1
            for paths in (
                source_inputs,
                input_paths,
                draft_paths,
                invariant_paths,
                result_paths,
            )
        ):
            raise PilotRunnerError(
                "diagnostic recovery requires exactly one complete artifact set"
            )

        source_value = json.loads(source_inputs[0].read_text(encoding="utf-8"))
        current_value = json.loads(input_paths[0].read_text(encoding="utf-8"))
        request = QualitativeRequest.model_validate(current_value["request"])
        context = QualitativeEvidenceContext.model_validate(
            current_value["evidence_context"]
        )
        draft = QualitativeAgentDraft.model_validate_json(
            draft_paths[0].read_text(encoding="utf-8")
        )
        canonical_result = QualitativeResult.model_validate_json(
            result_paths[0].read_text(encoding="utf-8")
        )
        if finalize_qualitative_draft(draft, request, context) != canonical_result:
            raise PilotRunnerError(
                "persisted Qualitative result differs from deterministic finalization"
            )
        validate_qualitative_result(
            canonical_result,
            request,
            context,
            runs_root=recovery_runs_root,
        )

        telemetry = load_run_model_telemetry(
            recovery_runs_root, recovery_run_id
        )
        if telemetry is None or not telemetry.invocation_references:
            raise PilotRunnerError(
                "diagnostic recovery requires persisted model telemetry"
            )
        records: list[ModelInvocationRecord] = []
        run_root = recovery_runs_root / recovery_run_id
        for reference in telemetry.invocation_references:
            record_path = (run_root / reference).resolve(strict=True)
            try:
                record_path.relative_to(run_root.resolve(strict=True))
            except ValueError as exc:
                raise PilotRunnerError(
                    "diagnostic recovery telemetry escaped the run directory"
                ) from exc
            records.append(
                ModelInvocationRecord.model_validate_json(
                    record_path.read_text(encoding="utf-8")
                )
            )
        if (
            len(records) != telemetry.total_calls
            or not records
            or any(
                item.runtime_kind != "text_agent"
                or item.provider != self.config.text_runtime.provider
                or item.model != self.config.text_runtime.model
                for item in records
            )
        ):
            raise PilotRunnerError(
                "diagnostic recovery model telemetry is incomplete or mismatched"
            )

        draft_invariant_telemetry = json.loads(
            invariant_paths[0].read_text(encoding="utf-8")
        )
        draft_invariant_attempts = draft_invariant_telemetry.get("attempts", [])
        scientific_input_hashes = [
            item.get("scientific_input_hash")
            for item in draft_invariant_attempts
        ]
        scientific_input_hash_equal = bool(scientific_input_hashes) and (
            len(set(scientific_input_hashes)) == 1
            and scientific_input_hashes[0]
            == draft_invariant_telemetry.get("scientific_input_hash")
        )
        payload_hash_equal = (
            canonical_json_hash(_qualitative_scientific_payload(source_value))
            == canonical_json_hash(_qualitative_scientific_payload(current_value))
        )
        final_record = records[-1]
        draft_schema_passed = (
            final_record.status == "completed" and final_record.schema_valid is True
        )
        gate_passed = (
            draft_schema_passed
            and payload_hash_equal
            and scientific_input_hash_equal
            and draft_invariant_telemetry.get("status") == "passed"
            and telemetry.failed_calls == 0
        )
        if not gate_passed:
            raise PilotRunnerError(
                "persisted diagnostic artifacts did not satisfy every recovery gate"
            )

        dataset_root = Path(request.task.inputs.dataset_reference).resolve(strict=True)
        challenge_id = request.task.inputs.dataset
        public_manifest = json.loads(
            (source / "suite_public.json").read_text(encoding="utf-8")
        )
        public_entries = [
            item for item in public_manifest["challenges"]
            if item["challenge_id"] == challenge_id
        ]
        if len(public_entries) != 1:
            raise PilotRunnerError(
                "source public suite does not identify the recovery challenge"
            )
        public_entry = public_entries[0]
        if dataset_root != (
            source / public_entry["dataset_reference"]
        ).resolve(strict=True):
            raise PilotRunnerError(
                "recovered diagnostic dataset differs from the public suite"
            )
        if request.metadata_file is None:
            raise PilotRunnerError("recovered diagnostic requires public metadata")
        metadata_path = (dataset_root / request.metadata_file).resolve(strict=True)
        metadata_path.relative_to(dataset_root)
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.reader(handle)
            metadata_columns = next(rows, None)
            sample_count = sum(1 for _ in rows)
        challenge_value = json.loads(
            (
                source / "challenges" / challenge_id / "challenge.json"
            ).read_text(encoding="utf-8")
        )

        attempts = []
        for item in records:
            diagnostics = item.schema_diagnostics
            attempts.append({
                "attempt": item.attempt,
                "status": item.status,
                "response_kind": item.response_kind,
                "json_parse_status": (
                    item.response_diagnostics.json_parse_status
                    if item.response_diagnostics is not None
                    else "content_missing"
                ),
                "finish_reason": (
                    item.response_diagnostics.finish_reason
                    if item.response_diagnostics is not None else "missing"
                ),
                "schema_valid": item.schema_valid,
                "total_schema_issue_count": (
                    diagnostics.total_schema_issue_count if diagnostics else 0
                ),
                "schema_issues": (
                    [
                        issue.model_dump(mode="json")
                        for issue in diagnostics.issues
                    ]
                    if diagnostics else []
                ),
                "latency_ms": item.latency_ms,
                "usage": item.usage.model_dump(mode="json"),
                "schema_retry": item.schema_retry,
                "transport_retry": item.transport_retry,
                "request_hash": item.request_hash,
                "payload_hash": item.payload_hash,
            })
        result = {
            "schema_version": "1.0",
            "pilot_id": self.config.pilot_id,
            "source_pilot_id": source_pilot_id,
            "recovery_pilot_id": recovery_pilot_id,
            "run_id": recovery_run_id,
            "gate_materialization_mode": "recovered_existing_validated_run",
            "provider_calls_during_recovery": 0,
            "dataset_identity": {
                "challenge_id": challenge_id,
                "generator_version": challenge_value.get("public", {}).get(
                    "generator_version"
                ),
                "public_dataset_path": str(dataset_root),
                "public_dataset_hash": public_entry["public_dataset_hash"],
                "sample_count": sample_count,
                "metadata_columns": metadata_columns,
            },
            "status": "passed",
            "attempt_count": len(records),
            "draft_schema_result": "passed",
            "deterministic_finalization_result": "passed",
            "qualitative_output_counts": {
                "observations": len(canonical_result.payload.observations),
                "comparisons": len(canonical_result.payload.comparisons),
                "interpretations": len(canonical_result.payload.interpretations),
            },
            "qualitative_result_model_validate": "passed",
            "validate_qualitative_result": "passed",
            "invariant_failure_category": None,
            "schema_retry_count": telemetry.schema_retry_calls,
            "draft_invariant_retry_count": draft_invariant_telemetry.get(
                "draft_invariant_retry_count", 0
            ),
            "max_draft_invariant_retries": MAX_DRAFT_INVARIANT_RETRIES,
            "transport_retry_count": telemetry.retry_calls,
            "latency_ms": telemetry.total_model_latency_ms,
            "usage_available": telemetry.usage_complete,
            "scientific_payload_hash_equal": payload_hash_equal,
            "scientific_input_hash_equal_across_draft_attempts": (
                scientific_input_hash_equal
            ),
            "scientific_input_hash": scientific_input_hashes[0],
            "draft_invariant_attempts": draft_invariant_attempts,
            "attempts": attempts,
            "artifact_hashes": {
                "input": file_hash(input_paths[0]),
                "draft": file_hash(draft_paths[0]),
                "draft_invariant_telemetry": file_hash(invariant_paths[0]),
                "canonical_result": file_hash(result_paths[0]),
                "model_telemetry": file_hash(
                    run_root / "model_calls" / "telemetry.json"
                ),
            },
            "error_type": None,
            "error_message": None,
        }
        path = self.pilot_root / "qualitative_diagnostic.json"
        atomic_write_json(path, result, error_type=PilotRunnerError)
        return result

    def qualitative_diagnostic(
        self,
        *,
        source_pilot_id: str,
        recovery_pilot_id: str | None = None,
    ) -> dict[str, Any]:
        """Re-run only Qualitative with equivalent persisted canary evidence."""
        self._require_complete_real_configuration()
        self.dry_run()
        if recovery_pilot_id is not None:
            return self._recover_qualitative_diagnostic(
                source_pilot_id=source_pilot_id,
                recovery_pilot_id=recovery_pilot_id,
            )
        self._assert_real()
        source = self._require_source_provider_gates(source_pilot_id)
        path = self.pilot_root / "qualitative_diagnostic.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))

        source_run_id = f"{source_pilot_id}-canary"
        source_inputs = sorted(
            (source / "runs" / source_run_id / "qualitative").glob("*/input.json")
        )
        if len(source_inputs) != 1:
            raise PilotRunnerError(
                "source canary must contain exactly one Qualitative input artifact"
            )
        source_value = json.loads(source_inputs[0].read_text(encoding="utf-8"))
        source_request = QualitativeRequest.model_validate(source_value["request"])
        run_id = f"{self.config.pilot_id}-qualitative-diagnostic"
        request = source_request.model_copy(update={
            "run_id": run_id,
            "task": source_request.task.model_copy(update={"run_id": run_id}),
        })
        dataset_root = Path(request.task.inputs.dataset_reference).resolve(strict=True)
        challenge_id = request.task.inputs.dataset
        if challenge_id != "pilot_001":
            raise PilotRunnerError(
                "Qualitative diagnostic source must be the pilot_001 canary dataset"
            )
        public_manifest = json.loads(
            (source / "suite_public.json").read_text(encoding="utf-8")
        )
        public_entries = [
            item for item in public_manifest["challenges"]
            if item["challenge_id"] == challenge_id
        ]
        if len(public_entries) != 1:
            raise PilotRunnerError(
                "source public suite does not identify pilot_001 exactly once"
            )
        public_entry = public_entries[0]
        expected_dataset_root = (
            source / public_entry["dataset_reference"]
        ).resolve(strict=True)
        if dataset_root != expected_dataset_root:
            raise PilotRunnerError(
                "Qualitative diagnostic dataset path differs from the public suite"
            )
        if request.metadata_file is None:
            raise PilotRunnerError(
                "Qualitative diagnostic requires public metadata"
            )
        metadata_path = (dataset_root / request.metadata_file).resolve(strict=True)
        try:
            metadata_path.relative_to(dataset_root)
        except ValueError as exc:
            raise PilotRunnerError(
                "Qualitative diagnostic metadata escaped the public dataset"
            ) from exc
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.reader(handle)
            metadata_columns = next(rows, None)
            sample_count = sum(1 for _ in rows)
        if not metadata_columns:
            raise PilotRunnerError("Qualitative diagnostic metadata has no header")
        challenge_value = json.loads(
            (source / "challenges" / challenge_id / "challenge.json").read_text(
                encoding="utf-8"
            )
        )
        runtimes = build_local_tool_runtimes(
            run_id=run_id,
            dataset_root=dataset_root,
            runs_root=self.runs_root,
        )
        assert self.text_client is not None
        text_start = len(self.text_client.invocations)
        error: Exception | None = None
        canonical_result = None
        try:
            canonical_result = execute_qualitative(
                request,
                self.text_client,
                runs_root=self.runs_root,
                artifact_scope=request.task.task_id,
                tool_runtime=runtimes["qualitative"],
            )
        except Exception as exc:
            error = exc

        records = self.text_client.invocations[text_start:]
        final_record = records[-1] if records else None
        diagnostic_input = (
            self.runs_root / run_id / "qualitative"
            / request.task.task_id / "input.json"
        )
        current_value = (
            json.loads(diagnostic_input.read_text(encoding="utf-8"))
            if diagnostic_input.is_file() else None
        )
        draft_invariant_telemetry_path = (
            self.runs_root / run_id / "qualitative" / request.task.task_id
            / "draft_invariant_telemetry.json"
        )
        draft_invariant_telemetry = (
            json.loads(
                draft_invariant_telemetry_path.read_text(encoding="utf-8")
            )
            if draft_invariant_telemetry_path.is_file()
            else None
        )
        draft_invariant_attempts = (
            draft_invariant_telemetry.get("attempts", [])
            if draft_invariant_telemetry is not None else []
        )
        scientific_input_hashes = [
            item.get("scientific_input_hash")
            for item in draft_invariant_attempts
        ]
        scientific_input_hash_equal = bool(scientific_input_hashes) and (
            len(set(scientific_input_hashes)) == 1
        )
        payload_hash_equal = bool(current_value) and (
            canonical_json_hash(_qualitative_scientific_payload(source_value))
            == canonical_json_hash(_qualitative_scientific_payload(current_value))
        )
        draft_schema_passed = bool(
            final_record is not None and final_record.status == "completed"
        )
        finalization_passed = bool(
            canonical_result is not None
            or (
                draft_schema_passed
                and error is not None
                and not isinstance(error, QualitativeDraftValidationError)
            )
        )
        finalization_result = (
            "passed" if finalization_passed
            else "failed" if draft_schema_passed
            else "not_run"
        )
        result_model_validate_passed = bool(
            finalization_passed
            and (
                canonical_result is None
                or QualitativeResult.model_validate(
                    canonical_result.model_dump(mode="json")
                ) == canonical_result
            )
        )
        invariant_result = (
            "passed" if canonical_result is not None
            else "failed" if finalization_passed
            else "not_run"
        )
        attempts = []
        for item in records:
            diagnostics = item.schema_diagnostics
            attempts.append({
                "attempt": item.attempt,
                "status": item.status,
                "response_kind": item.response_kind,
                "json_parse_status": (
                    item.response_diagnostics.json_parse_status
                    if item.response_diagnostics is not None else "content_missing"
                ),
                "finish_reason": (
                    item.response_diagnostics.finish_reason
                    if item.response_diagnostics is not None else "missing"
                ),
                "schema_valid": item.schema_valid,
                "total_schema_issue_count": (
                    diagnostics.total_schema_issue_count if diagnostics else 0
                ),
                "schema_issues": (
                    [issue.model_dump(mode="json") for issue in diagnostics.issues]
                    if diagnostics else []
                ),
                "latency_ms": item.latency_ms,
                "usage": item.usage.model_dump(mode="json"),
                "schema_retry": item.schema_retry,
                "transport_retry": item.transport_retry,
                "request_hash": item.request_hash,
                "payload_hash": item.payload_hash,
            })
        result = {
            "schema_version": "1.0",
            "pilot_id": self.config.pilot_id,
            "source_pilot_id": source_pilot_id,
            "run_id": run_id,
            "dataset_identity": {
                "challenge_id": challenge_id,
                "generator_version": challenge_value.get("public", {}).get(
                    "generator_version"
                ),
                "public_dataset_path": str(dataset_root),
                "public_dataset_hash": public_entry["public_dataset_hash"],
                "sample_count": sample_count,
                "metadata_columns": metadata_columns,
            },
            "status": (
                "passed"
                if (
                    error is None
                    and payload_hash_equal
                    and scientific_input_hash_equal
                    and draft_invariant_telemetry is not None
                    and draft_invariant_telemetry.get("status") == "passed"
                )
                else "failed"
            ),
            "attempt_count": len(records),
            "draft_schema_result": (
                "passed" if draft_schema_passed else "failed"
            ),
            "deterministic_finalization_result": finalization_result,
            "qualitative_output_counts": {
                "observations": (
                    len(canonical_result.payload.observations)
                    if canonical_result is not None else None
                ),
                "comparisons": (
                    len(canonical_result.payload.comparisons)
                    if canonical_result is not None else None
                ),
                "interpretations": (
                    len(canonical_result.payload.interpretations)
                    if canonical_result is not None else None
                ),
            },
            "qualitative_result_model_validate": (
                "passed" if result_model_validate_passed else "not_run"
            ),
            "validate_qualitative_result": invariant_result,
            "invariant_failure_category": (
                _qualitative_invariant_category(error)
                if error is not None and draft_schema_passed else None
            ),
            "schema_retry_count": sum(item.schema_retry for item in records),
            "draft_invariant_retry_count": (
                draft_invariant_telemetry.get(
                    "draft_invariant_retry_count", 0
                )
                if draft_invariant_telemetry is not None else 0
            ),
            "max_draft_invariant_retries": MAX_DRAFT_INVARIANT_RETRIES,
            "transport_retry_count": sum(item.transport_retry for item in records),
            "latency_ms": round(sum(item.latency_ms for item in records), 6),
            "usage_available": bool(records) and all(
                item.usage.available for item in records
            ),
            "scientific_payload_hash_equal": payload_hash_equal,
            "scientific_input_hash_equal_across_draft_attempts": (
                scientific_input_hash_equal
            ),
            "scientific_input_hash": (
                scientific_input_hashes[0]
                if scientific_input_hashes else None
            ),
            "draft_invariant_attempts": draft_invariant_attempts,
            "attempts": attempts,
            "error_type": type(error).__name__ if error is not None else None,
            "error_message": (
                "Qualitative protocol validation failed; inspect only saved safe metadata."
                if error is not None else None
            ),
        }
        atomic_write_json(path, result, error_type=PilotRunnerError)
        return result

    def quantitative_diagnostic(
        self,
        *,
        source_pilot_id: str,
    ) -> dict[str, Any]:
        """Run exactly one real Quantitative task from a persisted pilot_001 canary."""

        self._require_complete_real_configuration()
        self.dry_run()
        self._assert_real()
        source = self._require_hardened_source_gates(source_pilot_id)
        path = self.pilot_root / "quantitative_diagnostic.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))

        source_run_id = f"{source_pilot_id}-canary"
        inputs = sorted(
            (source / "runs" / source_run_id / "quantitative").glob(
                "*/input.json"
            )
        )
        matches = [
            item for item in inputs if item.parent.name == "TASK-C002-QT"
        ]
        if len(matches) != 1:
            raise PilotRunnerError(
                "source canary must contain exactly one TASK-C002-QT input"
            )
        source_value = json.loads(matches[0].read_text(encoding="utf-8"))
        legacy_source_request = QuantitativeRequest.model_validate(
            source_value["request"]
        )
        source_run_root = source / "runs" / source_run_id
        executive_value = json.loads(
            (source_run_root / "executive" / "input.json").read_text(
                encoding="utf-8"
            )
        )
        executive_request = ExecutiveRequest.model_validate(
            executive_value["request"]
        )
        routing_path = source_run_root / "executive" / "result.json"
        if not routing_path.is_file():
            routing_path = routing_path.with_name("routing.json")
        routing_plan = RoutingPlan.model_validate_json(
            routing_path.read_text(encoding="utf-8")
        )
        initial_state = json.loads(
            (source_run_root / "input.json").read_text(encoding="utf-8")
        )["initial_state"]
        visual_options = initial_state.get("identify_config", {}).get(
            "visual_discovery", {}
        )
        identify_budget = initial_state.get("identify_config", {}).get(
            "budget", {}
        )
        source_request = build_quantitative_request(
            executive_request=executive_request,
            routing_plan=routing_plan,
            task_id="TASK-C002-QT",
            message_id=legacy_source_request.message_id,
            limitations=legacy_source_request.limitations,
            execution_resource_options={
                "seed": initial_state["identify_config"]["seed"],
                "sample_id_column": visual_options.get("sample_id_column"),
                "sample_path_column": visual_options.get("sample_path_column"),
                "split_column": visual_options.get("split_column"),
                "max_rows": identify_budget.get("max_metadata_rows", 100_000),
                "max_samples": visual_options.get("max_samples", 10_000),
            },
        )
        if (
            source_request.task.inputs.dataset != "pilot_001"
            or source_request.task.task_id != "TASK-C002-QT"
        ):
            raise PilotRunnerError(
                "Quantitative diagnostic source is not pilot_001/TASK-C002-QT"
            )
        public_manifest = json.loads(
            (source / "suite_public.json").read_text(encoding="utf-8")
        )
        public_entries = [
            item for item in public_manifest["challenges"]
            if item["challenge_id"] == "pilot_001"
        ]
        if len(public_entries) != 1:
            raise PilotRunnerError(
                "source public suite does not identify pilot_001 exactly once"
            )
        public_entry = public_entries[0]
        dataset_root = Path(
            source_request.task.inputs.dataset_reference
        ).resolve(strict=True)
        expected_dataset_root = (
            source / public_entry["dataset_reference"]
        ).resolve(strict=True)
        if dataset_root != expected_dataset_root:
            raise PilotRunnerError(
                "Quantitative diagnostic dataset differs from the public suite"
            )
        run_id = f"{self.config.pilot_id}-quantitative-diagnostic"
        task = source_request.task.model_copy(update={
            "run_id": run_id,
            "message_id": f"{run_id}:executive:{source_request.task.task_id}:task",
        })
        request = source_request.model_copy(update={
            "run_id": run_id,
            "message_id": (
                f"{run_id}:quantitative:{source_request.task.task_id}:request"
            ),
            "in_reply_to": task.message_id,
            "task": task,
        })
        runtimes = build_local_tool_runtimes(
            run_id=run_id,
            dataset_root=dataset_root,
            runs_root=self.runs_root,
        )
        execution_requirement = determine_quantitative_execution_requirement(
            request,
            runtimes["quantitative"],
            feasibility_confirmed=True,
        )
        assert self.text_client is not None
        start = len(self.text_client.invocations)
        error: Exception | None = None
        canonical_result: QuantitativeResult | None = None
        try:
            canonical_result = execute_quantitative(
                request,
                self.text_client,
                runs_root=self.runs_root,
                tool_runtime=runtimes["quantitative"],
                execution_requirement=execution_requirement,
            )
            QuantitativeResult.model_validate(
                canonical_result.model_dump(mode="json")
            )
            validate_quantitative_result(
                canonical_result,
                request,
                tool_results=tuple(runtimes["quantitative"].history),
            )
        except Exception as exc:
            error = exc

        records = self.text_client.invocations[start:]
        final_record = records[-1] if records else None
        draft_path = (
            self.runs_root / run_id / "quantitative"
            / request.task.task_id / "draft.json"
        )
        result_path = draft_path.with_name("result.json")
        draft_valid = False
        if draft_path.is_file():
            try:
                QuantitativeAgentDraft.model_validate_json(
                    draft_path.read_text(encoding="utf-8")
                )
                draft_valid = True
            except ValidationError:
                draft_valid = False
        tool_results = runtimes["quantitative"].history
        attempts_for_task = [
            item for item in tool_results
            if item.tool_name == "run_experiment"
            and item.run_id == run_id
            and item.task_id == request.task.task_id
            and item.candidate_id == request.candidate.candidate_id
        ]
        successful = [
            item for item in attempts_for_task if item.status == "success"
        ]
        failed = [item for item in attempts_for_task if item.status == "failed"]
        obligation_path = draft_path.with_name("execution_obligation.json")
        obligation = (
            json.loads(obligation_path.read_text(encoding="utf-8"))
            if obligation_path.is_file() else {}
        )
        contract_failure_path = draft_path.with_name(
            "execution_contract_failure.json"
        )
        contract_failure = (
            json.loads(contract_failure_path.read_text(encoding="utf-8"))
            if contract_failure_path.is_file() else {}
        )
        plan_path = draft_path.with_name("experiment_plan.json")
        plan_execution_path = draft_path.with_name("plan_execution.json")
        plan = (
            json.loads(plan_path.read_text(encoding="utf-8"))
            if plan_path.is_file() else {}
        )
        plan_execution = (
            json.loads(plan_execution_path.read_text(encoding="utf-8"))
            if plan_execution_path.is_file() else {}
        )
        mandatory_step_ids = plan_execution.get(
            "mandatory_executed_step_ids", []
        )
        passed = bool(
            error is None
            and canonical_result is not None
            and draft_valid
            and result_path.is_file()
            and plan.get("plan_version")
            == QUANTITATIVE_EXPERIMENT_PLAN_VERSION
            and mandatory_step_ids
            and plan_execution.get("backend_attempt_count", 0) >= 1
            and plan_execution.get("successful_tool_result_count", 0) >= 1
            and execution_requirement.must_attempt_run_experiment
            and "statistical_check"
            in execution_requirement.resolved_experiment_types
            and len(records) == 1
            and final_record is not None
            and final_record.status == "completed"
            and final_record.schema_valid is True
        )
        attempts = [
            {
                "attempt": item.attempt,
                "status": item.status,
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
                "json_parse_status": (
                    item.response_diagnostics.json_parse_status
                    if item.response_diagnostics is not None
                    else "content_missing"
                ),
                "schema_valid": item.schema_valid,
                "completion_tokens": item.usage.completion_tokens,
                "reasoning_tokens": item.usage.reasoning_tokens,
                "total_tokens": item.usage.total_tokens,
                "latency_ms": item.latency_ms,
                "schema_retry": item.schema_retry,
                "transport_retry": item.transport_retry,
                "request_hash": item.request_hash,
                "payload_hash": item.payload_hash,
            }
            for item in records
        ]
        result = {
            "schema_version": "1.0",
            "pilot_id": self.config.pilot_id,
            "source_pilot_id": source_pilot_id,
            "run_id": run_id,
            "dataset_identity": {
                "challenge_id": "pilot_001",
                "public_dataset_hash": public_entry["public_dataset_hash"],
                "task_id": request.task.task_id,
                "candidate_id": request.candidate.candidate_id,
            },
            "requested_checks": request.task.requested_checks,
            "execution_requirement": execution_requirement.model_dump(mode="json"),
            "quantitative_check_resolution_version": (
                QUANTITATIVE_CHECK_RESOLUTION_VERSION
            ),
            "quantitative_check_resolution_hash": (
                QUANTITATIVE_CHECK_RESOLUTION_HASH
            ),
            "status": "passed" if passed else "failed",
            "quantitative_experiment_plan_version": (
                QUANTITATIVE_EXPERIMENT_PLAN_VERSION
            ),
            "plan_compile_passed": bool(plan),
            "compiled_step_count": len(plan.get("steps", [])),
            "mandatory_step_count": len(mandatory_step_ids),
            "mandatory_executed_step_ids": mandatory_step_ids,
            "backend_attempt_count": plan_execution.get(
                "backend_attempt_count", len(attempts_for_task)
            ),
            "successful_tool_result_count": plan_execution.get(
                "successful_tool_result_count", len(successful)
            ),
            "provider_call_count": len(records),
            "tool_loop_iteration_count": sum(
                item.status == "completed"
                and item.response_kind in {"tool_calls", "final_result"}
                for item in records
            ),
            "tool_call_count": len(tool_results),
            "provider_tool_draft_count": obligation.get(
                "provider_tool_draft_count", 0
            ),
            "canonical_tool_call_count": obligation.get(
                "canonical_tool_call_count", 0
            ),
            "run_experiment_attempt_count": len(attempts_for_task),
            "failed_tool_result_count": len(failed),
            "tool_names": [item.tool_name for item in tool_results],
            "valid_json": bool(
                final_record is not None
                and final_record.response_diagnostics is not None
                and final_record.response_diagnostics.json_parse_status
                == "valid_object"
            ),
            "draft_schema_valid": draft_valid,
            "draft_invariant_valid": canonical_result is not None,
            "deterministic_finalization_valid": canonical_result is not None,
            "canonical_result_valid": canonical_result is not None,
            "schema_retry_count": sum(item.schema_retry for item in records),
            "execution_obligation_retry_count": obligation.get(
                "execution_obligation_retry_count", 0
            ),
            "transport_retry_count": sum(item.transport_retry for item in records),
            "latency_ms": round(sum(item.latency_ms for item in records), 6),
            "usage_available": bool(records) and all(
                item.usage.available for item in records
            ),
            "prompt_tokens": (
                sum(item.usage.prompt_tokens or 0 for item in records)
                if records and all(item.usage.available for item in records)
                else None
            ),
            "completion_tokens": (
                sum(item.usage.completion_tokens or 0 for item in records)
                if records and all(item.usage.available for item in records)
                else None
            ),
            "total_tokens": (
                sum(item.usage.total_tokens or 0 for item in records)
                if records and all(item.usage.available for item in records)
                else None
            ),
            "attempts": attempts,
            "error_type": type(error).__name__ if error is not None else None,
            "failure_layer": contract_failure.get("category"),
            "error_message": (
                "Quantitative diagnostic failed; inspect saved safe metadata only."
                if error is not None else None
            ),
        }
        atomic_write_json(path, result, error_type=PilotRunnerError)
        return result

    def legacy_task_negative_replay(
        self,
        *,
        source_pilot_id: str,
    ) -> dict[str, Any]:
        """Prove that the persisted malformed legacy task is rejected upstream."""

        self.dry_run()
        source = self._source_pilot_root(source_pilot_id)
        output_path = self.pilot_root / "legacy_task_negative_replay.json"
        if output_path.is_file():
            return json.loads(output_path.read_text(encoding="utf-8"))
        source_run_id = f"{source_pilot_id}-canary"
        source_root = source / "runs" / source_run_id
        legacy_paths = sorted(
            (source_root / "quantitative").glob("*/input.json")
        )
        requests: list[QuantitativeRequest] = []
        for path in legacy_paths:
            value = json.loads(path.read_text(encoding="utf-8"))
            request = QuantitativeRequest.model_validate(value["request"])
            requests.append(request)
        selected = [
            request for request in requests
            if request.task.task_id == "TASK-C002-QT"
        ]
        if len(selected) != 1:
            raise PilotRunnerError(
                "legacy negative replay requires exactly one persisted TASK-C002-QT"
            )
        legacy = selected[0]
        if (
            "association_test" not in legacy.task.requested_checks
            or legacy.task.inputs.metadata_fields
        ):
            raise PilotRunnerError(
                "TASK-C002-QT no longer has the expected malformed association shape"
            )
        malformed = [
            request for request in requests
            if "association_test" in request.task.requested_checks
            and not request.task.inputs.metadata_fields
        ]
        executive_value = json.loads(
            (source_root / "executive" / "input.json").read_text(encoding="utf-8")
        )
        executive_request = ExecutiveRequest.model_validate(
            executive_value["request"]
        )
        contexts = {
            item.candidate_id: item
            for item in build_candidate_contexts(executive_request)
        }
        rejected: dict[str, str] = {}
        for request in malformed:
            candidate_context = contexts.get(request.task.candidate_id)
            if candidate_context is None:
                raise PilotRunnerError(
                    "legacy malformed task has no matching public candidate context"
                )
            try:
                validate_analysis_task_semantics(
                    request.task,
                    candidate_context,
                    executive_request.routing_context.capabilities,
                )
            except AnalysisTaskSemanticError as exc:
                rejected[request.task.task_id] = str(exc)
            else:
                raise PilotRunnerError(
                    f"legacy malformed task unexpectedly passed semantic validation: "
                    f"{request.task.task_id}"
                )
        if legacy.task.task_id not in rejected:
            raise PilotRunnerError("TASK-C002-QT was not rejected by the semantic gate")
        result = {
            "schema_version": "1.0",
            "status": "passed",
            "provider_call_count": 0,
            "source_pilot_id": source_pilot_id,
            "selected_task_id": legacy.task.task_id,
            "requested_checks": legacy.task.requested_checks,
            "metadata_fields": legacy.task.inputs.metadata_fields,
            "semantic_gate": "rejected",
            "rejection_category": rejected[legacy.task.task_id],
            "historical_malformed_task_count": len(malformed),
            "rejected_malformed_task_ids": sorted(rejected),
        }
        atomic_write_json(output_path, result, error_type=PilotRunnerError)
        return result

    def historical_candidate_binding_replay(
        self,
        *,
        source_pilot_id: str,
    ) -> dict[str, Any]:
        """Prove that ambiguous legacy public candidates remain unresolved."""

        self.dry_run()
        source = self._source_pilot_root(source_pilot_id)
        output_path = self.pilot_root / "historical_candidate_binding_replay.json"
        if output_path.is_file():
            return json.loads(output_path.read_text(encoding="utf-8"))
        source_run_id = f"{source_pilot_id}-canary"
        source_root = source / "runs" / source_run_id
        executive_value = json.loads(
            (source_root / "executive" / "input.json").read_text(encoding="utf-8")
        )
        request = ExecutiveRequest.model_validate(executive_value["request"])
        initial_state = json.loads(
            (source_root / "input.json").read_text(encoding="utf-8")
        )["initial_state"]
        visual = initial_state.get("identify_config", {}).get("visual_discovery", {})
        addressing = list(request.routing_context.capabilities.addressing_fields)
        for field in (
            request.routing_context.task.label_column,
            visual.get("sample_id_column"),
            visual.get("sample_path_column"),
            visual.get("split_column"),
        ):
            if field is not None and field not in addressing:
                addressing.append(field)
        capabilities = request.routing_context.capabilities.model_copy(
            update={"addressing_fields": addressing}
        )
        context = request.routing_context.model_copy(
            update={"capabilities": capabilities}
        )
        request = request.model_copy(update={"routing_context": context})
        contexts = build_candidate_contexts(request)
        bindings = [
            {
                "candidate_id": candidate.candidate_id,
                "cue_family": candidate.cue.family,
                "cue_name": candidate.cue.name,
                "binding_status": executable.binding_status,
                "binding_kind": executable.binding_kind,
            }
            for candidate, executable in zip(
                request.candidate_report.candidates,
                contexts,
                strict=True,
            )
        ]
        expected_legacy = [
            (candidate, executable)
            for candidate, executable in zip(
                request.candidate_report.candidates, contexts, strict=True
            )
            if candidate.cue.family == "metadata"
            and candidate.cue.name == "site_device_batch"
        ]
        if not expected_legacy:
            raise PilotRunnerError(
                "historical public CandidateReport has no site_device_batch metadata candidate"
            )
        fuzzy_fallback_count = sum(
            executable.binding_status != "unresolved"
            or executable.binding_kind != "unresolved"
            or bool(executable.bound_metadata_fields)
            for _, executable in expected_legacy
        )
        if fuzzy_fallback_count:
            raise PilotRunnerError(
                "historical site_device_batch candidate was fuzzily bound"
            )
        result = {
            "schema_version": "1.0",
            "status": "passed",
            "provider_call_count": 0,
            "source_pilot_id": source_pilot_id,
            "candidate_report_loaded": True,
            "hidden_truth_accessed": False,
            "candidate_bindings": bindings,
            "site_device_batch_result": "unresolved",
            "fuzzy_fallback_count": 0,
            "generated_task_count": 0,
        }
        atomic_write_json(output_path, result, error_type=PilotRunnerError)
        return result

    def current_contract_task_offline(self) -> dict[str, Any]:
        """Run a current exact-binding engineering fixture with zero providers."""

        self.dry_run()
        output_path = self.pilot_root / "current_contract_task_offline.json"
        if output_path.is_file():
            return json.loads(output_path.read_text(encoding="utf-8"))
        run_id = f"{self.config.pilot_id}-current-contract-offline"
        fixture_root = self.pilot_root / "offline" / "current_contract_fixture" / "public" / "dataset"
        fixture_root.mkdir(parents=True, exist_ok=True)
        scientific_field = "scientific_cue"
        fieldnames = ["id", "label", scientific_field, "path"]
        metadata_path = fixture_root / "metadata.csv"
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(8):
                writer.writerow({
                    "id": f"sample-{index:03d}",
                    "label": index % 2,
                    scientific_field: "group-a" if index % 2 == 0 else "group-b",
                    "path": f"images/sample-{index:03d}.png",
                })
        limitation = (
            "Synthetic public engineering contract fixture; not a scientific conclusion."
        )
        candidate = Candidate(
            candidate_id="C001",
            cue=Cue(family="metadata", name=scientific_field),
            description="Exact metadata binding engineering fixture.",
            observation="The public fixture exposes one exact scientific metadata field.",
            hypothesis="The fixture field may be associated with the fixture label.",
            evidence_sources=["metadata"],
            priority="high",
            status="unverified",
        )
        report = CandidateReport(
            protocol_version="1.0",
            run_id=run_id,
            message_id=f"{run_id}:identify:report",
            source="identify",
            target="executive",
            message_type="candidate_report",
            in_reply_to=f"{run_id}:identify:request",
            artifact_references=[],
            dataset_summary=DatasetInspectionSummary(
                dataset="current_contract_fixture",
                samples_inspected=0,
                metadata_used=True,
            ),
            candidates=[candidate],
            limitations=[limitation],
        )
        request = ExecutiveRequest(
            protocol_version="1.0",
            run_id=run_id,
            message_id=f"{run_id}:executive:request",
            source="orchestrator",
            target="executive",
            message_type="executive_request",
            in_reply_to=report.message_id,
            candidate_report=report,
            routing_context=ExecutiveRoutingContext(
                protocol_version="1.0",
                run_id=run_id,
                dataset=DatasetSpec(
                    name="current_contract_fixture",
                    root=str(fixture_root),
                    metadata_file="metadata.csv",
                ),
                task=TaskSpec(type="classification", label_column="label"),
                audit_options=ExecutiveAuditOptions(
                    validation_depth="standard",
                    max_tasks=1,
                    max_samples_per_task=8,
                    allowed_targets=["quantitative"],
                    requested_outputs=["metrics"],
                    runtime_constraints=["no_external_services"],
                ),
                capabilities=ExecutiveCapabilities(
                    available_targets=["quantitative"],
                    available_metadata_fields=fieldnames,
                    addressing_fields=["id", "label", "path"],
                    quantitative_checks=["association_test"],
                    qualitative_checks=[],
                    model_capabilities=[],
                    sample_access=True,
                    external_services=False,
                    available_output_types=["metrics"],
                ),
                artifact_references=[],
                limitations=[limitation],
            ),
        )
        contexts = build_candidate_contexts(request)
        draft = ExecutiveAgentDraft.model_validate({
            "decisions": [{
                "candidate_id": candidate.candidate_id,
                "route": "quantitative",
                "routing_reason": "The exact fixture binding permits association testing.",
                "quantitative": {
                    "objective": "Verify the current exact-binding execution contract.",
                    "requested_checks": ["association_test"],
                },
                "qualitative": None,
                "missing_requirements": [],
                "limitations": [limitation],
            }],
            "limitations": [limitation],
        })
        plan = finalize_executive_draft(request, contexts, draft)
        validate_routing_plan(plan, request, candidate_contexts=contexts)
        task = plan.tasks[0]
        quantitative_request = build_quantitative_request(
            executive_request=request,
            routing_plan=plan,
            task_id=task.task_id,
            message_id=f"{request.run_id}:quantitative:{task.task_id}:offline-request",
            execution_resource_options={
                "seed": self.config.runtime_seeds[0],
                "sample_id_column": "id",
                "sample_path_column": "path",
                "max_rows": 8,
                "max_samples": 8,
            },
        )
        runtime = build_local_tool_runtimes(
            run_id=request.run_id,
            dataset_root=fixture_root,
            runs_root=self.runs_root,
        )["quantitative"]
        requirement = determine_quantitative_execution_requirement(
            quantitative_request, runtime, feasibility_confirmed=True
        )
        experiment_plan = compile_quantitative_experiment_plan(
            quantitative_request, requirement, runtime
        )
        mandatory = primary_mandatory_steps(
            experiment_plan, quantitative_request.task.requested_checks
        )
        results = []
        payload = {
            "request": quantitative_request.model_dump(
                mode="json", exclude={"execution_resources"}
            ),
            "execution_requirement": requirement.model_dump(mode="json"),
        }
        for index, step in enumerate(mandatory):
            call = finalize_tool_call_draft(
                {"name": "run_experiment", "arguments": {"plan_step_id": step.step_id}},
                agent_name="quantitative",
                base_payload=payload,
                tool_runtime=runtime,
                iteration=0,
                call_index=index,
                experiment_plan=experiment_plan,
            )
            ExperimentToolInput.model_validate(call.arguments)
            result = runtime.execute(call)
            if result.status != "success" or result.output is None:
                raise PilotRunnerError("current contract mandatory real backend step failed")
            ExperimentToolOutput.model_validate(result.output)
            results.append(result)
        if not mandatory or not results:
            raise PilotRunnerError("current contract compiled no executable mandatory step")
        if any(
            result.output["metadata"].get("backend") != "real_cpu_v1"
            for result in results
        ):
            raise PilotRunnerError("current contract did not use RealExperimentExecutor")
        quantitative_draft = QuantitativeAgentDraft(
            status="completed",
            analysis_type="offline_engineering_contract_verification",
            evidence_level="Q1",
            evidence_direction="inconclusive",
            evidence_summary="The real local backend completed the engineering fixture.",
            evidence_tool_call_ids=[result.tool_call_id for result in results],
            additional_limitations=[limitation],
            recommended_followup=[],
        )
        quantitative_result = finalize_quantitative_result(
            quantitative_draft,
            quantitative_request,
            tool_results=tuple(results),
        )
        validate_quantitative_result(
            quantitative_result,
            quantitative_request,
            tool_results=tuple(results),
        )
        result = {
            "schema_version": "1.0",
            "status": "passed",
            "provider_call_count": 0,
            "fixture_kind": "current_contract_public_engineering_fixture",
            "hidden_truth_accessed": False,
            "binding_status": contexts[0].binding_status,
            "binding_kind": contexts[0].binding_kind,
            "bound_metadata_fields": contexts[0].bound_metadata_fields,
            "task": {
                "task_id": task.task_id,
                "candidate_id": task.candidate_id,
                "requested_checks": task.requested_checks,
                "metadata_fields": task.inputs.metadata_fields,
                "required_model_capabilities": task.inputs.required_model_capabilities,
                "requires_sample_access": task.inputs.requires_sample_access,
                "semantic_completeness": "passed",
            },
            "ready_step_count": sum(
                step.execution_state == "ready" for step in experiment_plan.steps
            ),
            "mandatory_backend_attempt_count": len(results),
            "successful_tool_result_count": sum(
                result.status == "success" for result in results
            ),
            "real_backend": "real_cpu_v1",
            "quantitative_finalization": "passed",
            "quantitative_result_status": quantitative_result.payload.status,
        }
        atomic_write_json(output_path, result, error_type=PilotRunnerError)
        return result

    def _save_canary_quantitative_audit(self, run_id: str) -> Path:
        """Persist safe plan, backend, and model-call facts for every QT task."""

        run_root = self.runs_root / run_id
        executive_input = json.loads(
            (run_root / "executive" / "input.json").read_text(encoding="utf-8")
        )
        routing_path = run_root / "executive" / "routing.json"
        if not routing_path.is_file():
            routing_path = run_root / "executive" / "result.json"
        routing = json.loads(routing_path.read_text(encoding="utf-8"))
        candidates = executive_input["request"]["candidate_report"]["candidates"]
        executable_contexts = executive_input["executable_candidate_contexts"]
        agent_audits = []
        for agent_name in (
            "identify", "executive", "quantitative", "qualitative", "summarize"
        ):
            paths = sorted((run_root / "tools" / agent_name).glob("*.json"))
            results = [json.loads(item.read_text(encoding="utf-8")) for item in paths]
            agent_audits.append({
                "agent_name": agent_name,
                "tool_execution_count": len(results),
                "successful_tool_result_count": sum(
                    item.get("status") == "success" for item in results
                ),
                "failed_tool_result_count": sum(
                    item.get("status") == "failed" for item in results
                ),
            })
        tool_paths = sorted((run_root / "tools" / "quantitative").glob("*.json"))
        tool_results = [
            json.loads(item.read_text(encoding="utf-8")) for item in tool_paths
        ]
        tasks = []
        for input_path in sorted((run_root / "quantitative").glob("*/input.json")):
            value = json.loads(input_path.read_text(encoding="utf-8"))
            request = value["request"]
            requirement = value["execution_requirement"]
            task_id = request["task"]["task_id"]
            attempts = [
                item for item in tool_results
                if item.get("tool_name") == "run_experiment"
                and item.get("task_id") == task_id
            ]
            result_path = input_path.with_name("result.json")
            result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.is_file() else {}
            )
            plan_path = input_path.with_name("experiment_plan.json")
            plan = (
                json.loads(plan_path.read_text(encoding="utf-8"))
                if plan_path.is_file() else {}
            )
            plan_execution_path = input_path.with_name("plan_execution.json")
            plan_execution = (
                json.loads(plan_execution_path.read_text(encoding="utf-8"))
                if plan_execution_path.is_file() else {}
            )
            steps = plan.get("steps", [])
            tasks.append({
                "task_id": task_id,
                "requested_checks": request["task"]["requested_checks"],
                "task_metadata_fields": request["task"].get("inputs", {}).get(
                    "metadata_fields", []
                ),
                "resolved_experiment_types": requirement[
                    "resolved_experiment_types"
                ],
                "must_attempt_run_experiment": requirement[
                    "must_attempt_run_experiment"
                ],
                "compiled_step_count": len(steps),
                "ready_step_count": sum(
                    item.get("execution_state") == "ready" for item in steps
                ),
                "pending_step_count": sum(
                    item.get("execution_state") == "pending_dependency"
                    for item in steps
                ),
                "mandatory_executed_step_ids": plan_execution.get(
                    "mandatory_executed_step_ids", []
                ),
                "backend_attempt_count": plan_execution.get(
                    "backend_attempt_count", len(attempts)
                ),
                "successful_tool_result_count": plan_execution.get(
                    "successful_tool_result_count",
                    sum(item.get("status") == "success" for item in attempts),
                ),
                "quantitative_agent_calls": plan_execution.get(
                    "quantitative_agent_call_count", 0
                ),
                "canonical_status": result.get("payload", {}).get("status"),
            })
        payload = {
            "schema_version": "1.0",
            "run_id": run_id,
            "candidate_bindings": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "cue_family": candidate["cue"]["family"],
                    "binding_status": executable["binding_status"],
                    "binding_kind": executable["binding_kind"],
                }
                for candidate, executable in zip(
                    candidates, executable_contexts, strict=True
                )
            ],
            "new_task_summaries": [
                {
                    "task_id": task["task_id"],
                    "target": task["target"],
                    "requested_checks": task["requested_checks"],
                    "metadata_fields": task["inputs"]["metadata_fields"],
                    "required_model_capabilities": task["inputs"][
                        "required_model_capabilities"
                    ],
                    "requires_sample_access": task["inputs"][
                        "requires_sample_access"
                    ],
                    "semantic_completeness": "passed",
                }
                for task in routing["tasks"]
            ],
            "quantitative_check_resolution_version": (
                QUANTITATIVE_CHECK_RESOLUTION_VERSION
            ),
            "quantitative_check_resolution_hash": (
                QUANTITATIVE_CHECK_RESOLUTION_HASH
            ),
            "quantitative_experiment_plan_version": (
                QUANTITATIVE_EXPERIMENT_PLAN_VERSION
            ),
            "agents": agent_audits,
            "tasks": tasks,
        }
        path = self.pilot_root / "canary_quantitative_audit.json"
        atomic_write_json(path, payload, error_type=PilotRunnerError)
        return path

    def canary(self) -> PilotRunRecord:
        self._require_complete_real_configuration()
        self.dry_run()
        legacy = self._require_gate("legacy_task_negative_replay.json")
        historical = self._require_gate("historical_candidate_binding_replay.json")
        current = self._require_gate("current_contract_task_offline.json")
        if legacy.get("status") != "passed" or legacy.get("semantic_gate") != "rejected":
            raise PilotRunnerError("legacy malformed-task negative replay did not pass")
        if (
            historical.get("status") != "passed"
            or historical.get("provider_call_count") != 0
            or historical.get("site_device_batch_result") != "unresolved"
            or historical.get("fuzzy_fallback_count") != 0
            or historical.get("generated_task_count") != 0
        ):
            raise PilotRunnerError("historical candidate binding replay did not pass")
        if legacy.get("source_pilot_id") != historical.get("source_pilot_id"):
            raise PilotRunnerError("historical replay gates use different source pilots")
        if (
            current.get("status") != "passed"
            or current.get("provider_call_count") != 0
            or current.get("binding_status") != "resolved"
            or current.get("ready_step_count", 0) < 1
            or current.get("mandatory_backend_attempt_count", 0) < 1
            or current.get("successful_tool_result_count", 0) < 1
            or current.get("real_backend") != "real_cpu_v1"
            or current.get("quantitative_finalization") != "passed"
            or not current.get("task", {}).get("metadata_fields")
            or "association_test" not in current.get("task", {}).get(
                "requested_checks", []
            )
        ):
            raise PilotRunnerError("current-contract zero-provider task gate did not pass")
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
        self._save_canary_quantitative_audit(record.run_id)
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
