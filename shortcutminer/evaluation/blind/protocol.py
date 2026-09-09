"""Strict JSON contracts for evaluation-only blind shortcut challenges."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, FiniteFloat, field_validator, model_validator

from shortcutminer.protocol.evidence import UncertaintyEstimate
from shortcutminer.protocol.identify import (
    CandidateId,
    NonEmptyText,
    SafeRelativePath,
    SafeRunId,
    StrictProtocolModel,
    TaskSpec,
)


ChallengeFamily = Literal[
    "metadata_association",
    "spatial_visual",
    "acquisition_quality",
    "mixed",
    "clean_control",
]
ShortcutFamily = Literal["metadata", "spatial", "acquisition"]
InjectionStrength = Literal["weak", "moderate", "strong"]
MatchStatus = Literal[
    "matched", "partial", "not_matched", "ambiguous", "potential_novel", "unassessed"
]
LocalizationStatus = Literal["exact", "partial", "incorrect", "unknown", "not_applicable"]
AdjudicationSource = Literal["deterministic", "manual", "multi_rater", "none"]
FinalAssessment = Literal[
    "supported", "partially_supported", "inconclusive", "contradicted", "insufficient_evidence"
]
MissReason = Literal[
    "not_discovered",
    "discovered_wrong_localization",
    "routing_failure",
    "quantitative_blocked",
    "qualitative_unusable",
    "evidence_inconclusive",
    "summary_overclaim",
    "summary_underclaim",
]


def _unique(values: list[str], field_name: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


def _require_neutral_challenge_id(value: str) -> str:
    forbidden = (
        "shortcut", "bias", "spurious", "metadata", "spatial",
        "acquisition", "mixed", "clean", "control",
    )
    if any(token in value.casefold() for token in forbidden):
        raise ValueError("challenge_id must not reveal the challenge family or truth")
    return value


class BlindChallengeConfig(StrictProtocolModel):
    challenge_id: SafeRunId
    family: ChallengeFamily
    challenge_generation_seed: int
    sample_count: int = Field(default=40, ge=16, le=1_000)
    injection_strength: InjectionStrength = "strong"

    @field_validator("challenge_id")
    @classmethod
    def require_neutral_public_id(cls, value: str) -> str:
        return _require_neutral_challenge_id(value)


class PublicChallenge(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    challenge_id: SafeRunId
    generator_version: NonEmptyText
    dataset_reference: SafeRelativePath
    metadata_file: SafeRelativePath
    task: TaskSpec
    runtime_sampling_seed: int
    public_dataset_hash: NonEmptyText

    @field_validator("challenge_id")
    @classmethod
    def require_neutral_public_id(cls, value: str) -> str:
        return _require_neutral_challenge_id(value)


class HiddenShortcutSpec(StrictProtocolModel):
    truth_id: NonEmptyText
    family: ShortcutFamily
    target_field: NonEmptyText | None = None
    spatial_region: Literal["corner", "border", "center_offset"] | None = None
    acquisition_family: Literal[
        "compression", "resolution", "blur", "contrast", "brightness", "noise"
    ] | None = None
    injection_mechanism: NonEmptyText
    affected_sample_ids: list[NonEmptyText]
    injection_strength: InjectionStrength
    strength_parameters: dict[NonEmptyText, FiniteFloat | int | str | bool]

    @field_validator("affected_sample_ids")
    @classmethod
    def unique_samples(cls, values: list[str]) -> list[str]:
        return _unique(values, "affected_sample_ids")

    @model_validator(mode="after")
    def validate_localization(self):
        populated = {
            "metadata": self.target_field is not None,
            "spatial": self.spatial_region is not None,
            "acquisition": self.acquisition_family is not None,
        }
        if not populated[self.family]:
            raise ValueError("hidden shortcut must define family-specific localization")
        if any(is_populated for family, is_populated in populated.items() if family != self.family):
            raise ValueError("hidden shortcut cannot define localization for another family")
        return self


class CleanControlSpec(StrictProtocolModel):
    shortcut_present: Literal[False] = False
    independent_variations: list[NonEmptyText] = Field(min_length=3)
    construction_notes: list[NonEmptyText] = Field(min_length=1)


class HiddenShortcutTruth(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    challenge_id: SafeRunId
    generator_version: NonEmptyText
    challenge_family: ChallengeFamily
    challenge_generation_seed: int
    shortcut_present: bool
    shortcuts: list[HiddenShortcutSpec]
    clean_control: CleanControlSpec | None = None
    public_dataset_hash: NonEmptyText

    @model_validator(mode="after")
    def validate_presence(self):
        if self.shortcut_present != bool(self.shortcuts):
            raise ValueError("shortcut_present must match shortcuts")
        if self.shortcut_present and self.clean_control is not None:
            raise ValueError("positive truth cannot contain a clean control")
        if not self.shortcut_present and self.clean_control is None:
            raise ValueError("clean truth requires clean_control")
        return self


class BlindChallenge(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    challenge_id: SafeRunId
    public: PublicChallenge
    hidden_truth_reference: SafeRelativePath
    private_truth_hash: NonEmptyText

    @field_validator("challenge_id")
    @classmethod
    def require_neutral_public_id(cls, value: str) -> str:
        return _require_neutral_challenge_id(value)

    @model_validator(mode="after")
    def validate_ids(self):
        if self.public.challenge_id != self.challenge_id:
            raise ValueError("public challenge_id mismatch")
        if self.public.dataset_reference.startswith("private/"):
            raise ValueError("public dataset cannot be beneath private storage")
        if not self.hidden_truth_reference.startswith("private/"):
            raise ValueError("hidden truth must be beneath private storage")
        return self


class CandidateTruthMatch(StrictProtocolModel):
    candidate_id: CandidateId
    truth_id: NonEmptyText | None = None
    match_status: MatchStatus
    localization_status: LocalizationStatus
    rationale: NonEmptyText
    adjudication_source: AdjudicationSource


class ManualAdjudication(StrictProtocolModel):
    reviewer_id: NonEmptyText
    candidate_id: CandidateId
    truth_id: NonEmptyText | None = None
    decision: MatchStatus
    localization: LocalizationStatus
    rationale: NonEmptyText


class AdjudicationAggregate(StrictProtocolModel):
    candidate_id: CandidateId
    truth_id: NonEmptyText | None = None
    rater_count: int = Field(ge=1)
    decision_counts: dict[MatchStatus, int]
    localization_counts: dict[LocalizationStatus, int]
    raw_agreement: FiniteFloat = Field(ge=0, le=1)
    cohens_kappa: FiniteFloat | None = Field(default=None, ge=-1, le=1)
    consensus_decision: MatchStatus | None = None
    consensus_localization: LocalizationStatus | None = None


class AdjudicationAgreement(StrictProtocolModel):
    reviewer_ids: list[NonEmptyText]
    item_count: int = Field(ge=1)
    raw_agreement: FiniteFloat = Field(ge=0, le=1)
    cohens_kappa: FiniteFloat | None = Field(default=None, ge=-1, le=1)
    item_aggregates: list[AdjudicationAggregate]


class DiscoveryJudgement(StrictProtocolModel):
    candidate_id: CandidateId
    matched_truth_ids: list[NonEmptyText]
    routed: bool
    quantitatively_tested: bool
    qualitatively_tested: bool
    evidence_usable: bool
    final_assessment: FinalAssessment | None = None

    @field_validator("matched_truth_ids")
    @classmethod
    def unique_truths(cls, values: list[str]) -> list[str]:
        return _unique(values, "matched_truth_ids")


class DiscoveryMetrics(StrictProtocolModel):
    candidate_count: int = Field(ge=0)
    truth_count: int = Field(ge=0)
    matched_truth_count: int = Field(ge=0)
    missed_truth_count: int = Field(ge=0)
    unmatched_candidate_count: int = Field(ge=0)
    ambiguous_candidate_count: int = Field(ge=0)
    discovery_recall: FiniteFloat | None = Field(default=None, ge=0, le=1)
    candidate_precision: FiniteFloat | None = Field(default=None, ge=0, le=1)
    assessed_candidate_count: int = Field(ge=0)
    recall_at_k: dict[NonEmptyText, FiniteFloat | None]


class FalsePositiveMetrics(StrictProtocolModel):
    false_positive_candidate_count: int = Field(ge=0)
    false_positive_findings_count: int = Field(ge=0)
    partial_false_positive_findings_count: int = Field(ge=0)
    false_positive_per_audit: FiniteFloat = Field(ge=0)


class LocalizationMetrics(StrictProtocolModel):
    exact_count: int = Field(ge=0)
    partial_count: int = Field(ge=0)
    incorrect_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    localization_rate: FiniteFloat | None = Field(default=None, ge=0, le=1)


class FunnelMetrics(StrictProtocolModel):
    truth_count: int = Field(ge=0)
    discovered_count: int = Field(ge=0)
    routed_count: int = Field(ge=0)
    quantitatively_tested_count: int = Field(ge=0)
    qualitatively_tested_count: int = Field(ge=0)
    evidence_usable_count: int = Field(ge=0)
    final_supported_count: int = Field(ge=0)
    validation_completion_rate: FiniteFloat | None = Field(default=None, ge=0, le=1)


class FailureRecord(StrictProtocolModel):
    truth_id: NonEmptyText | None = None
    candidate_id: CandidateId | None = None
    reason: MissReason
    details: NonEmptyText


class DownstreamImpactRecord(StrictProtocolModel):
    available: bool
    model_reference: SafeRelativePath | None = None
    metric_before: FiniteFloat | None = None
    metric_after: FiniteFloat | None = None
    ood_metric: FiniteFloat | None = None
    group_gap: FiniteFloat | None = None
    intervention_delta: FiniteFloat | None = None
    reason_unavailable: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_availability(self):
        metrics = (
            self.metric_before,
            self.metric_after,
            self.ood_metric,
            self.group_gap,
            self.intervention_delta,
        )
        if self.available and not any(value is not None for value in metrics):
            raise ValueError("available downstream impact requires a measured metric")
        if not self.available and not self.reason_unavailable:
            raise ValueError("unavailable downstream impact requires a reason")
        return self


class EvaluationSnapshot(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    snapshot_id: NonEmptyText
    git_commit: NonEmptyText
    workspace_hashes: dict[NonEmptyText, NonEmptyText]
    protocol_hashes: dict[NonEmptyText, NonEmptyText]
    experiment_backend_version: NonEmptyText
    qualitative_backend_version: NonEmptyText
    evidence_calibration_version: NonEmptyText
    runtime_config_hash: NonEmptyText
    sanitized_model_config: dict[str, Any] | None = None
    model_runtime_snapshot_id: NonEmptyText | None = None
    model_runtime_snapshot_hash: NonEmptyText | None = None
    challenge_public_hash: NonEmptyText
    runtime_sampling_seed: int
    evaluation_bootstrap_seed: int
    candidate_limit: int = Field(ge=1)
    validation_depth: NonEmptyText
    budget_settings: dict[str, Any]


class BlindEvaluationResult(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    challenge_id: SafeRunId
    snapshot_id: NonEmptyText
    run_id: SafeRunId
    truth_reference_hash: NonEmptyText
    candidate_matches: list[CandidateTruthMatch]
    discovery_judgements: list[DiscoveryJudgement]
    discovery_metrics: DiscoveryMetrics
    false_positive_metrics: FalsePositiveMetrics
    localization_metrics: LocalizationMetrics
    funnel_metrics: FunnelMetrics
    failures: list[FailureRecord]
    downstream_impact: DownstreamImpactRecord
    limitations: list[NonEmptyText]


class StratifiedAggregateMetrics(StrictProtocolModel):
    challenge_count: int = Field(ge=0)
    truth_count: int = Field(ge=0)
    discovery_recall: FiniteFloat | None = Field(default=None, ge=0, le=1)
    candidate_false_positive_per_audit: FiniteFloat | None = Field(default=None, ge=0)
    final_false_positive_per_audit: FiniteFloat | None = Field(default=None, ge=0)
    localization_rate: FiniteFloat | None = Field(default=None, ge=0, le=1)
    validation_completion_rate: FiniteFloat | None = Field(default=None, ge=0, le=1)


class BlindEvaluationBatchResult(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    snapshot_id: NonEmptyText
    number_of_challenges: int = Field(ge=0)
    number_positive: int = Field(ge=0)
    number_clean: int = Field(ge=0)
    positive: StratifiedAggregateMetrics
    clean: StratifiedAggregateMetrics
    uncertainty: list[UncertaintyEstimate]
    challenge_result_references: list[SafeRelativePath]
    limitations: list[NonEmptyText]


__all__ = [
    "AdjudicationAggregate", "AdjudicationAgreement", "BlindChallenge", "BlindChallengeConfig",
    "BlindEvaluationBatchResult", "BlindEvaluationResult", "CandidateTruthMatch",
    "ChallengeFamily", "CleanControlSpec", "DiscoveryJudgement", "DiscoveryMetrics",
    "DownstreamImpactRecord", "EvaluationSnapshot", "FailureRecord", "FalsePositiveMetrics",
    "FunnelMetrics", "HiddenShortcutSpec", "HiddenShortcutTruth", "LocalizationMetrics",
    "ManualAdjudication", "PublicChallenge", "StratifiedAggregateMetrics",
]
