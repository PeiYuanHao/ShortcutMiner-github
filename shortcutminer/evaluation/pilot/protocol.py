"""Strict contracts for the Phase 16 real-provider development pilot."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, FiniteFloat, field_validator, model_validator

from shortcutminer.evaluation.blind.protocol import BlindEvaluationResult
from shortcutminer.protocol.identify import NonEmptyText, SafeRelativePath, SafeRunId, StrictProtocolModel
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.protocol.model_runtime import ModelRuntimeSnapshot


PilotStatus = Literal[
    "planned", "preflight_failed", "blocked", "running", "incomplete",
    "completed", "failed",
]
PhaseConclusion = Literal["PASS", "FAIL", "BLOCKED"]
ChallengeKind = Literal["positive", "clean_control"]


class PilotBudget(StrictProtocolModel):
    max_total_text_model_calls: int = Field(default=180, ge=1, le=100_000)
    max_total_vlm_calls: int = Field(default=40, ge=1, le=100_000)
    max_total_vlm_images: int = Field(default=160, ge=2, le=1_000_000)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_tool_calls_per_run: int = Field(default=20, ge=1, le=1_000)
    max_text_calls_per_run: int = Field(default=15, ge=5, le=1_000)
    max_vlm_calls_per_run: int = Field(default=3, ge=1, le=100)


class PilotChallengeConfig(StrictProtocolModel):
    challenge_id: SafeRunId
    kind: ChallengeKind
    family: Literal[
        "metadata_association", "spatial_visual", "acquisition_quality",
        "semantic_background", "clean_control",
    ]
    generation_seed: int
    sample_count: int = Field(default=48, ge=16, le=1_000)
    injection_strength: Literal["weak", "moderate", "strong"] = "strong"

    @field_validator("challenge_id")
    @classmethod
    def neutral_identifier(cls, value: str) -> str:
        if not value.startswith("pilot_") or not value[6:].isdigit():
            raise ValueError("pilot challenge IDs must use the neutral pilot_NNN form")
        return value

    @model_validator(mode="after")
    def kind_matches_family(self):
        if (self.family == "clean_control") != (self.kind == "clean_control"):
            raise ValueError("clean_control kind and family must agree")
        return self


def default_challenge_suite() -> list[PilotChallengeConfig]:
    families = (
        "metadata_association", "spatial_visual", "acquisition_quality",
        "semantic_background", "clean_control", "clean_control",
    )
    return [
        PilotChallengeConfig(
            challenge_id=f"pilot_{index:03d}",
            kind="clean_control" if family == "clean_control" else "positive",
            family=family,
            generation_seed=16_000 + index * 137,
        )
        for index, family in enumerate(families, 1)
    ]


class PilotConfig(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    pilot_id: SafeRunId
    artifact_root: NonEmptyText = "evaluation/pilots"
    repository_root: NonEmptyText = "."
    frozen_git_commit: NonEmptyText | None = None
    text_runtime: OpenAICompatibleModelConfig
    semantic_runtime: OpenAICompatibleModelConfig
    repeats: int = Field(default=2, ge=2, le=10)
    candidate_limit: int = Field(default=3, ge=1, le=20)
    validation_depth: Literal["standard", "deep"] = "deep"
    visual_sample_budget: int = Field(default=8, ge=2, le=100)
    semantic_sample_budget: int = Field(default=6, ge=2, le=100)
    runtime_seeds: list[int]
    bootstrap_seeds: list[int]
    challenges: list[PilotChallengeConfig] = Field(default_factory=default_challenge_suite)
    budget: PilotBudget = Field(default_factory=PilotBudget)
    inter_run_delay_seconds: FiniteFloat = Field(default=0.0, ge=0, le=3600)
    continue_on_failure: bool = True
    development_pilot: Literal[True] = True

    @model_validator(mode="after")
    def validate_scientific_shape(self):
        if self.frozen_git_commit is not None and (
            len(self.frozen_git_commit) != 40
            or any(character not in "0123456789abcdef" for character in self.frozen_git_commit)
        ):
            raise ValueError("frozen_git_commit must be a lowercase 40-character git hash")
        if len(self.challenges) != 6:
            raise ValueError("Phase 16 requires exactly six challenges")
        if len({item.challenge_id for item in self.challenges}) != 6:
            raise ValueError("challenge IDs must be unique")
        if sum(item.kind == "positive" for item in self.challenges) != 4:
            raise ValueError("Phase 16 requires four positive challenges")
        if sum(item.kind == "clean_control" for item in self.challenges) != 2:
            raise ValueError("Phase 16 requires two clean controls")
        if len(self.runtime_seeds) != self.repeats:
            raise ValueError("runtime_seeds length must equal repeats")
        if len(self.bootstrap_seeds) != self.repeats:
            raise ValueError("bootstrap_seeds length must equal repeats")
        if self.text_runtime.temperature != 0.0:
            raise ValueError("Phase 16 text runtime temperature must be 0.0")
        if not self.semantic_runtime.capabilities.supports_image_input:
            raise ValueError("semantic_runtime must declare supports_image_input=true")
        planned = len(self.challenges) * self.repeats
        if self.budget.max_total_text_model_calls < planned * 5:
            raise ValueError("global text budget cannot cover one call per Agent per planned run")
        return self


class FrozenWorkspace(StrictProtocolModel):
    agent_name: Literal["identify", "executive", "quantitative", "qualitative", "summarize"]
    sha256: NonEmptyText


class PilotSnapshot(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    snapshot_id: NonEmptyText
    pilot_id: SafeRunId
    created_at: NonEmptyText
    git_commit: NonEmptyText
    repository_dirty: bool
    workspace_hashes: list[FrozenWorkspace]
    protocol_hashes: dict[NonEmptyText, NonEmptyText]
    backend_hashes: dict[NonEmptyText, NonEmptyText]
    text_runtime: ModelRuntimeSnapshot
    semantic_runtime: ModelRuntimeSnapshot
    provider_combination_id: NonEmptyText
    config_hash: NonEmptyText
    challenge_suite_hash: NonEmptyText
    candidate_limit: int = Field(ge=1)
    validation_depth: NonEmptyText
    visual_sample_budget: int = Field(ge=1)
    semantic_sample_budget: int = Field(ge=1)
    runtime_seeds: list[int]
    bootstrap_seeds: list[int]
    budget: PilotBudget


class PreflightCheck(StrictProtocolModel):
    name: NonEmptyText
    success: bool
    provider: NonEmptyText
    model: NonEmptyText
    runtime_kind: Literal["real_provider"]
    latency_ms: FiniteFloat = Field(ge=0)
    usage_available: bool
    schema_retry_count: int = Field(default=0, ge=0)
    transport_retry_count: int = Field(default=0, ge=0)
    failure_category: NonEmptyText | None = None
    message: NonEmptyText


class PilotPreflightResult(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    pilot_id: SafeRunId
    status: Literal["passed", "failed", "blocked"]
    text: PreflightCheck
    semantic_vlm: PreflightCheck
    image_input_verified: bool = False
    image_content_distinguishable: bool
    image_names: list[Literal["sample_a.png", "sample_b.png"]]
    completed_at: NonEmptyText
    limitations: list[NonEmptyText] = Field(default_factory=list)


class PilotRunRecord(StrictProtocolModel):
    challenge_id: SafeRunId
    challenge_kind: ChallengeKind
    repeat_index: int = Field(ge=0)
    run_id: SafeRunId
    runtime_seed: int
    bootstrap_seed: int
    status: Literal["planned", "running", "completed", "failed", "skipped_budget"]
    snapshot_id: NonEmptyText
    public_challenge_hash: NonEmptyText
    result_reference: SafeRelativePath | None = None
    telemetry_reference: SafeRelativePath | None = None
    failure_category: NonEmptyText | None = None
    failure_message: NonEmptyText | None = None
    evaluation: BlindEvaluationResult | None = None


class PilotChallengeResult(StrictProtocolModel):
    challenge_id: SafeRunId
    challenge_kind: ChallengeKind
    runs: list[PilotRunRecord]
    successful_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    truth_discovery_count: int = Field(ge=0)
    candidate_false_positive_count: int = Field(ge=0)
    strong_final_false_positive_count: int = Field(ge=0)


class PilotAggregate(StrictProtocolModel):
    planned_runs: int = Field(ge=1)
    completed_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    positive_truth_count: int = Field(ge=0)
    positive_truth_discovered: int = Field(ge=0)
    recall_at_1: FiniteFloat | None = Field(default=None, ge=0, le=1)
    recall_at_3: FiniteFloat | None = Field(default=None, ge=0, le=1)
    exact_localization_count: int = Field(ge=0)
    partial_localization_count: int = Field(ge=0)
    clean_candidate_false_positives: int = Field(ge=0)
    clean_partial_final_false_positives: int = Field(ge=0)
    clean_strong_final_false_positives: int = Field(ge=0)
    final_assessment_distribution: dict[NonEmptyText, int]
    failure_taxonomy: dict[NonEmptyText, int]
    total_text_calls: int = Field(ge=0)
    total_vlm_calls: int = Field(ge=0)
    retry_calls: int = Field(ge=0)
    schema_retries: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    usage_complete: bool
    total_latency_ms: FiniteFloat = Field(ge=0)


class PilotResult(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    pilot_id: SafeRunId
    status: PilotStatus
    phase_conclusion: PhaseConclusion
    development_status: Literal["DEVELOPMENT PILOT - NOT FINAL PAPER RESULTS"]
    snapshot_id: NonEmptyText
    provider_combination_id: NonEmptyText
    started_at: NonEmptyText
    completed_at: NonEmptyText | None = None
    challenge_count: int = Field(ge=0)
    repeat_count: int = Field(ge=2)
    completed_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    challenge_results: list[PilotChallengeResult]
    aggregate: PilotAggregate
    preflight_reference: SafeRelativePath | None = None
    canary_reference: SafeRelativePath | None = None
    limitations: list[NonEmptyText] = Field(default_factory=list)


__all__ = [
    "ChallengeKind", "FrozenWorkspace", "PhaseConclusion", "PilotAggregate",
    "PilotBudget", "PilotChallengeConfig", "PilotChallengeResult", "PilotConfig",
    "PilotPreflightResult", "PilotResult", "PilotRunRecord", "PilotSnapshot",
    "PilotStatus", "PreflightCheck", "default_challenge_suite",
]
