"""Sequential, resumable repeated blind-audit matrix execution."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Literal

from pydantic import Field, FiniteFloat, ValidationError, field_validator, model_validator

from shortcutminer.protocol.identify import (
    NonEmptyText,
    SafeRelativePath,
    SafeRunId,
    StrictProtocolModel,
)
from shortcutminer.protocol.model_runtime import ModelCallBudget
from shortcutminer.runtime.openai_compatible import ModelRuntimeError
from shortcutminer.storage.json_artifacts import atomic_write_json


ChallengeKind = Literal["positive", "clean_control"]
LocalizationOutcome = Literal["exact", "partial", "incorrect", "unknown"]
AssessmentOutcome = Literal[
    "supported", "partially_supported", "inconclusive", "contradicted",
    "insufficient_evidence",
]
RuntimeFailureCategory = Literal[
    "provider_failure", "schema_failure", "budget_failure", "identify_failure",
    "executive_failure", "quantitative_failure", "qualitative_failure",
    "summarize_failure", "unknown_failure",
]


class RepeatedBlindAuditConfig(StrictProtocolModel):
    snapshot_id: NonEmptyText
    challenge_id: SafeRunId
    challenge_kind: ChallengeKind
    repeats: int = Field(ge=1, le=100)
    runtime_sampling_seeds: list[int]
    model_seeds: list[int | None] | None = None
    continue_on_run_failure: bool = True
    inter_run_delay: FiniteFloat = Field(default=0.0, ge=0, le=3600)
    model_call_budget: ModelCallBudget = Field(default_factory=ModelCallBudget)

    @model_validator(mode="after")
    def validate_seed_counts(self):
        if len(self.runtime_sampling_seeds) != self.repeats:
            raise ValueError("runtime_sampling_seeds length must equal repeats")
        if self.model_seeds is not None and len(self.model_seeds) != self.repeats:
            raise ValueError("model_seeds length must equal repeats")
        return self


class RepeatIdentity(StrictProtocolModel):
    run_id: SafeRunId
    repeat_index: int = Field(ge=0)
    runtime_sampling_seed: int
    model_seed: int | None = None


class RepeatedRunOutcome(StrictProtocolModel):
    identity: RepeatIdentity
    status: Literal["completed", "failed"]
    truth_discovered: bool | None = None
    localization: LocalizationOutcome | None = None
    final_assessment: AssessmentOutcome | None = None
    candidate_false_positives: int | None = Field(default=None, ge=0)
    partial_final_false_positives: int | None = Field(default=None, ge=0)
    strong_final_false_positives: int | None = Field(default=None, ge=0)
    failure_category: RuntimeFailureCategory | None = None
    failure_message: NonEmptyText | None = None
    model_telemetry_reference: SafeRelativePath | None = None
    result_reference: SafeRelativePath | None = None

    @model_validator(mode="after")
    def validate_status(self):
        if self.status == "completed" and (
            self.failure_category is not None or self.failure_message is not None
        ):
            raise ValueError("completed repeat cannot contain a runtime failure")
        if self.status == "failed" and (
            self.failure_category is None or self.failure_message is None
        ):
            raise ValueError("failed repeat requires a failure category and message")
        return self


class FrequencyDistribution(StrictProtocolModel):
    counts: dict[NonEmptyText, int]
    frequencies: dict[NonEmptyText, FiniteFloat | None]

    @model_validator(mode="after")
    def validate_keys(self):
        if set(self.counts) != set(self.frequencies):
            raise ValueError("frequency keys must match count keys")
        if any(value < 0 for value in self.counts.values()):
            raise ValueError("frequency counts must be non-negative")
        return self


class DiscoveryStability(StrictProtocolModel):
    discovered_count: int = Field(ge=0)
    total_completed_runs: int = Field(ge=0)
    discovery_frequency: FiniteFloat | None = Field(default=None, ge=0, le=1)


class FalsePositiveStability(StrictProtocolModel):
    candidate_false_positives_per_run: list[int]
    partial_final_false_positives_per_run: list[int]
    strong_final_false_positives_per_run: list[int]
    fraction_runs_with_any_strong_final_false_positive: FiniteFloat | None = Field(
        default=None, ge=0, le=1
    )


class RuntimeReliability(StrictProtocolModel):
    total_planned_runs: int = Field(ge=1)
    completed_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    run_success_frequency: FiniteFloat = Field(ge=0, le=1)
    failure_counts: dict[RuntimeFailureCategory, int]


class RepeatedBlindAuditMetrics(StrictProtocolModel):
    discovery: DiscoveryStability
    localization: FrequencyDistribution
    final_assessment: FrequencyDistribution
    false_positives: FalsePositiveStability
    runtime_reliability: RuntimeReliability


class RepeatedBlindAuditResult(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    matrix_id: SafeRunId
    config: RepeatedBlindAuditConfig
    completed_repeat_count: int = Field(ge=0)
    outcomes: list[RepeatedRunOutcome]
    metrics: RepeatedBlindAuditMetrics

    @field_validator("outcomes")
    @classmethod
    def unique_repeat_identity(cls, values: list[RepeatedRunOutcome]):
        indexes = [item.identity.repeat_index for item in values]
        run_ids = [item.identity.run_id for item in values]
        if len(indexes) != len(set(indexes)) or len(run_ids) != len(set(run_ids)):
            raise ValueError("repeat indexes and run IDs must be unique")
        return values

    @model_validator(mode="after")
    def validate_progress(self):
        if any(item.identity.repeat_index >= self.config.repeats for item in self.outcomes):
            raise ValueError("repeat outcome index exceeds the configured matrix")
        completed = sum(item.status == "completed" for item in self.outcomes)
        if self.completed_repeat_count != completed:
            raise ValueError("completed_repeat_count does not match outcomes")
        return self


class RepeatedBlindAuditError(RuntimeError):
    """Raised for an invalid matrix, unsafe resume, or fail-fast run failure."""


class RepeatedStageFailure(RuntimeError):
    """Explicitly classify a production-stage failure without recoding science."""

    def __init__(
        self,
        stage: Literal[
            "identify", "executive", "quantitative", "qualitative", "summarize"
        ],
        message: str,
    ) -> None:
        super().__init__(message)
        self.stage = stage


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _frequency(count: int, total: int) -> float | None:
    return round(count / total, 10) if total else None


def _distribution(
    outcomes: list[RepeatedRunOutcome], field: str, names: tuple[str, ...]
) -> FrequencyDistribution:
    values = [
        getattr(item, field)
        for item in outcomes
        if item.status == "completed" and getattr(item, field) is not None
    ]
    counts = {name: values.count(name) for name in names}
    return FrequencyDistribution(
        counts=counts,
        frequencies={name: _frequency(counts[name], len(values)) for name in names},
    )


def build_repeated_metrics(
    config: RepeatedBlindAuditConfig, outcomes: list[RepeatedRunOutcome]
) -> RepeatedBlindAuditMetrics:
    completed = [item for item in outcomes if item.status == "completed"]
    discoveries = [
        item.truth_discovered for item in completed if item.truth_discovered is not None
    ]
    fp_completed = [
        item for item in completed
        if item.candidate_false_positives is not None
        and item.partial_final_false_positives is not None
        and item.strong_final_false_positives is not None
    ]
    failure_names = (
        "provider_failure", "schema_failure", "budget_failure", "identify_failure",
        "executive_failure", "quantitative_failure", "qualitative_failure",
        "summarize_failure", "unknown_failure",
    )
    failures = [item.failure_category for item in outcomes if item.status == "failed"]
    return RepeatedBlindAuditMetrics(
        discovery=DiscoveryStability(
            discovered_count=sum(value is True for value in discoveries),
            total_completed_runs=len(discoveries),
            discovery_frequency=_frequency(
                sum(value is True for value in discoveries), len(discoveries)
            ),
        ),
        localization=_distribution(
            completed, "localization", ("exact", "partial", "incorrect", "unknown")
        ),
        final_assessment=_distribution(
            completed, "final_assessment",
            ("supported", "partially_supported", "inconclusive", "contradicted",
             "insufficient_evidence"),
        ),
        false_positives=FalsePositiveStability(
            candidate_false_positives_per_run=[
                item.candidate_false_positives for item in fp_completed
            ],
            partial_final_false_positives_per_run=[
                item.partial_final_false_positives for item in fp_completed
            ],
            strong_final_false_positives_per_run=[
                item.strong_final_false_positives for item in fp_completed
            ],
            fraction_runs_with_any_strong_final_false_positive=_frequency(
                sum((item.strong_final_false_positives or 0) > 0 for item in fp_completed),
                len(fp_completed),
            ),
        ),
        runtime_reliability=RuntimeReliability(
            total_planned_runs=config.repeats,
            completed_runs=len(completed),
            failed_runs=len(outcomes) - len(completed),
            run_success_frequency=round(len(completed) / config.repeats, 10),
            failure_counts={name: failures.count(name) for name in failure_names},
        ),
    )


class RepeatedBlindAuditRunner:
    """Run each repeat once, sequentially, checkpointing after every outcome."""

    def __init__(
        self,
        executor: Callable[[RepeatIdentity], RepeatedRunOutcome | dict[str, Any]],
        *,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._executor = executor
        self._sleeper = sleeper

    @staticmethod
    def matrix_id(config: RepeatedBlindAuditConfig) -> str:
        return f"repeat-{config.challenge_id}-{_hash(config.model_dump(mode='json'))[:16]}"

    @classmethod
    def result_reference(cls, config: RepeatedBlindAuditConfig) -> str:
        return f"evaluation/repeated/{cls.matrix_id(config)}.json"

    @staticmethod
    def _identity(config: RepeatedBlindAuditConfig, index: int) -> RepeatIdentity:
        model_seed = config.model_seeds[index] if config.model_seeds is not None else None
        digest = _hash({
            "snapshot_id": config.snapshot_id,
            "challenge_id": config.challenge_id,
            "repeat_index": index,
            "runtime_sampling_seed": config.runtime_sampling_seeds[index],
            "model_seed": model_seed,
        })[:12]
        return RepeatIdentity(
            run_id=f"{config.challenge_id}-repeat-{index + 1:03d}-{digest}",
            repeat_index=index,
            runtime_sampling_seed=config.runtime_sampling_seeds[index],
            model_seed=model_seed,
        )

    @staticmethod
    def _failure(identity: RepeatIdentity, exc: Exception) -> RepeatedRunOutcome:
        category: RuntimeFailureCategory = "unknown_failure"
        if isinstance(exc, ModelRuntimeError) and exc.failure is not None:
            if exc.failure.category == "protocol_schema_failed":
                category = "schema_failure"
            elif exc.failure.category == "budget_exceeded":
                category = "budget_failure"
            else:
                category = "provider_failure"
        elif isinstance(exc, ValidationError):
            category = "schema_failure"
        stage = getattr(exc, "stage", None)
        if stage in {"identify", "executive", "quantitative", "qualitative", "summarize"}:
            category = f"{stage}_failure"
        return RepeatedRunOutcome(
            identity=identity,
            status="failed",
            failure_category=category,
            failure_message=f"{type(exc).__name__}: {exc}",
        )

    def run_matrix(
        self,
        config: RepeatedBlindAuditConfig,
        *,
        artifact_root: str | Path,
        resume: bool = True,
    ) -> RepeatedBlindAuditResult:
        root = Path(artifact_root).resolve(strict=False)
        path = root / self.result_reference(config)
        outcomes: list[RepeatedRunOutcome] = []
        if path.is_file() and resume:
            try:
                existing = RepeatedBlindAuditResult.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise RepeatedBlindAuditError("stored repeated matrix is invalid") from exc
            if existing.config != config or existing.matrix_id != self.matrix_id(config):
                raise RepeatedBlindAuditError("stored repeated matrix config does not match")
            outcomes = list(existing.outcomes)
            if any(
                item.identity != self._identity(config, item.identity.repeat_index)
                for item in outcomes
            ):
                raise RepeatedBlindAuditError("stored repeat identity does not match config")
        elif path.is_file():
            raise RepeatedBlindAuditError(
                "repeated matrix already exists; enable resume to reuse it"
            )

        completed_indexes = {item.identity.repeat_index for item in outcomes}
        for index in range(config.repeats):
            if index in completed_indexes:
                continue
            identity = self._identity(config, index)
            try:
                value = self._executor(identity)
                if isinstance(value, dict):
                    value = RepeatedRunOutcome.model_validate(
                        {"identity": identity.model_dump(mode="json"), **value}
                    )
                else:
                    value = RepeatedRunOutcome.model_validate(value)
                if value.identity != identity:
                    raise RepeatedBlindAuditError(
                        "executor returned the wrong repeat identity"
                    )
            except Exception as exc:
                value = self._failure(identity, exc)
            outcomes.append(value)
            outcomes.sort(key=lambda item: item.identity.repeat_index)
            result = RepeatedBlindAuditResult(
                matrix_id=self.matrix_id(config),
                config=config,
                completed_repeat_count=sum(
                    item.status == "completed" for item in outcomes
                ),
                outcomes=outcomes,
                metrics=build_repeated_metrics(config, outcomes),
            )
            atomic_write_json(
                path, result.model_dump(mode="json"), error_type=RepeatedBlindAuditError
            )
            if value.status == "failed" and not config.continue_on_run_failure:
                raise RepeatedBlindAuditError(
                    f"repeat {index} failed and continue_on_run_failure is false"
                )
            if index < config.repeats - 1 and config.inter_run_delay:
                self._sleeper(float(config.inter_run_delay))
        return RepeatedBlindAuditResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    run = run_matrix


__all__ = [
    "AssessmentOutcome", "ChallengeKind", "DiscoveryStability",
    "FalsePositiveStability", "FrequencyDistribution", "LocalizationOutcome",
    "RepeatIdentity", "RepeatedBlindAuditConfig", "RepeatedBlindAuditError",
    "RepeatedBlindAuditMetrics", "RepeatedBlindAuditResult",
    "RepeatedBlindAuditRunner", "RepeatedRunOutcome", "RuntimeFailureCategory",
    "RepeatedStageFailure", "RuntimeReliability", "build_repeated_metrics",
]
