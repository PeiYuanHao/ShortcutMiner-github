"""Versioned, threshold-free evidence calibration contracts.

The ledger records what upstream agents and deterministic backends produced.  It
never turns a raw metric, p-value, or confidence interval into a shortcut verdict.
"""

from typing import Any, Literal

from pydantic import Field, FiniteFloat, field_validator, model_validator

from shortcutminer.protocol.executive import TaskId
from shortcutminer.protocol.identify import (
    CandidateId,
    NonEmptyText,
    SafeRelativePath,
    SafeRunId,
    StrictProtocolModel,
)


EvidenceKind = Literal[
    "quantitative_association",
    "quantitative_subgroup",
    "quantitative_predictive_probe",
    "quantitative_intervention",
    "quantitative_ablation",
    "quantitative_metric",
    "qualitative_metadata",
    "qualitative_visual",
    "qualitative_spatial",
    "qualitative_group_comparison",
    "qualitative_counterevidence",
    "qualitative_failure",
]
DesignStrength = Literal["descriptive", "observational", "predictive", "interventional"]
ExecutionState = Literal["usable", "scientifically_inconclusive", "failed", "blocked"]
EvidenceDirection = Literal["supports", "contradicts", "mixed", "inconclusive", "not_tested"]
EvidenceConsistency = Literal[
    "consistent_support",
    "consistent_contradiction",
    "mixed",
    "inconclusive",
    "no_usable_evidence",
]


class UncertaintyEstimate(StrictProtocolModel):
    metric: NonEmptyText
    method: NonEmptyText | None = None
    confidence_level: FiniteFloat | None = Field(default=None, ge=0.8, lt=1.0)
    lower: FiniteFloat | None = None
    upper: FiniteFloat | None = None
    standard_error: FiniteFloat | None = Field(default=None, ge=0)
    bootstrap_iterations: int | None = Field(default=None, ge=1, le=1_000)
    seed: int | None = None
    resampling_unit: NonEmptyText | None = None
    available: bool
    reason_unavailable: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_availability(self):
        if self.available:
            if self.method is None or self.lower is None or self.upper is None:
                raise ValueError("available uncertainty requires method and bounds")
            if self.lower > self.upper:
                raise ValueError("uncertainty lower bound exceeds upper bound")
            if self.reason_unavailable is not None:
                raise ValueError("available uncertainty cannot have reason_unavailable")
        elif not self.reason_unavailable:
            raise ValueError("unavailable uncertainty requires a reason")
        return self


class StatisticalTestRecord(StrictProtocolModel):
    metric: NonEmptyText
    raw_p_value: FiniteFloat = Field(ge=0, le=1)
    adjusted_p_value: FiniteFloat | None = Field(default=None, ge=0, le=1)
    correction_method: Literal["benjamini_hochberg"] | None = None
    family_id: NonEmptyText | None = None
    family_size: int | None = Field(default=None, ge=1)


class EvidenceCoverage(StrictProtocolModel):
    requested: int = Field(default=0, ge=0)
    attempted: int = Field(default=0, ge=0)
    usable: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    inconclusive: int = Field(default=0, ge=0)
    sample_count: int | None = Field(default=None, ge=0)
    valid_sample_count: int | None = Field(default=None, ge=0)
    missing_count: int | None = Field(default=None, ge=0)
    selected: int | None = Field(default=None, ge=0)
    inspected: int | None = Field(default=None, ge=0)


class EvidenceProvenance(StrictProtocolModel):
    run_id: SafeRunId
    candidate_id: CandidateId
    task_id: TaskId | None = None
    result_id: NonEmptyText | None = None
    experiment_id: NonEmptyText | None = None
    inspection_id: NonEmptyText | None = None
    dataset_reference: NonEmptyText | None = None
    input_family: NonEmptyText | None = None
    shared_source_identifiers: list[NonEmptyText] = Field(default_factory=list)

    @field_validator("shared_source_identifiers")
    @classmethod
    def unique_sources(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("shared_source_identifiers must not contain duplicates")
        return values


class EvidenceRecord(StrictProtocolModel):
    evidence_id: NonEmptyText
    candidate_id: CandidateId
    task_id: TaskId | None = None
    source_branch: Literal["quantitative", "qualitative"]
    source_result_id: NonEmptyText | None = None
    evidence_kind: EvidenceKind
    design_strength: DesignStrength
    execution_state: ExecutionState
    direction: EvidenceDirection
    metrics: dict[str, Any] = Field(default_factory=dict)
    uncertainty: list[UncertaintyEstimate] = Field(default_factory=list)
    statistical_tests: list[StatisticalTestRecord] = Field(default_factory=list)
    coverage: EvidenceCoverage
    artifact_references: list[SafeRelativePath] = Field(default_factory=list)
    limitations: list[NonEmptyText] = Field(default_factory=list)
    warnings: list[NonEmptyText] = Field(default_factory=list)
    provenance: EvidenceProvenance

    @field_validator("artifact_references", "limitations", "warnings")
    @classmethod
    def unique_lists(cls, values: list[str], info) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError(f"{info.field_name} must not contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_failure_semantics(self):
        if self.execution_state in {"failed", "blocked"} and self.direction != "not_tested":
            raise ValueError("failed or blocked evidence must be not_tested")
        if self.execution_state == "scientifically_inconclusive" and self.direction not in {
            "inconclusive", "not_tested"
        }:
            raise ValueError("scientifically inconclusive evidence cannot support or contradict")
        return self


class CandidateEvidenceProfile(StrictProtocolModel):
    candidate_id: CandidateId
    evidence_kinds: list[EvidenceKind]
    design_strengths: list[DesignStrength]
    max_design_strength: DesignStrength | None = None
    consistency: EvidenceConsistency
    coverage: EvidenceCoverage
    uncertainty_available: bool
    supporting_record_ids: list[NonEmptyText]
    contradicting_record_ids: list[NonEmptyText]
    mixed_record_ids: list[NonEmptyText]
    inconclusive_record_ids: list[NonEmptyText]
    failed_record_ids: list[NonEmptyText]
    blocked_record_ids: list[NonEmptyText]
    shared_source_groups: dict[NonEmptyText, list[NonEmptyText]]
    limitations: list[NonEmptyText]


class CandidateEvidenceLedger(StrictProtocolModel):
    candidate_id: CandidateId
    record_ids: list[NonEmptyText]
    records: list[EvidenceRecord]
    profile: CandidateEvidenceProfile


class EvidenceLedger(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: SafeRunId
    fingerprint: NonEmptyText
    artifact_reference: SafeRelativePath = "evidence/ledger.json"
    multiple_testing_method: Literal["benjamini_hochberg", "none"]
    test_family_rule: NonEmptyText
    candidates: list[CandidateEvidenceLedger]
    limitations: list[NonEmptyText] = Field(default_factory=list)


__all__ = [
    "CandidateEvidenceLedger", "CandidateEvidenceProfile", "DesignStrength",
    "EvidenceConsistency", "EvidenceCoverage", "EvidenceDirection", "EvidenceKind",
    "EvidenceLedger", "EvidenceProvenance", "EvidenceRecord", "ExecutionState",
    "StatisticalTestRecord", "UncertaintyEstimate",
]
