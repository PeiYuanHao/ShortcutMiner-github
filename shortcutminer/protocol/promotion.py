"""Controller-only contracts; promotion is not a scientific verdict."""
from typing import Literal
from pydantic import Field, FiniteFloat, model_validator
from shortcutminer.protocol.identify import StrictProtocolModel, SafeRunId, CandidateId


class MetricPromotionRule(StrictProtocolModel):
    experiment_type: Literal["statistical_check", "subgroup_evaluation", "metric_computation"]
    metric: Literal["effect_size", "p_value", "statistic", "accuracy", "balanced_accuracy", "overall_accuracy", "worst_group_accuracy", "group_accuracy_gap"]
    operator: Literal["ge", "le", "gt", "lt"]
    threshold: FiniteFloat


class PromotionPolicy(StrictProtocolModel):
    policy_version: str = "unconfigured-v1"
    calibration_status: Literal["unconfigured", "test_only", "legacy_uncalibrated", "externally_frozen"] = "unconfigured"
    applicable_evidence_types: list[Literal["statistical_check", "subgroup_evaluation", "metric_computation"]] = Field(default_factory=lambda: ["statistical_check", "subgroup_evaluation", "metric_computation"])
    metric_rules: list[MetricPromotionRule] = Field(default_factory=list)
    minimum_samples: int | None = Field(default=None, ge=1)
    quantitative_rule: Literal["any_rule_satisfied"] = "any_rule_satisfied"
    qualitative_promotion_rule: Literal["unavailable"] = "unavailable"
    missing_evidence_behavior: Literal["deferred_insufficient_evidence"] = "deferred_insufficient_evidence"
    strategy: Literal['metric_gate', 'budgeted_paired_probe'] = 'metric_gate'
    max_probe_candidates: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def coherent(self):
        if any(rule.experiment_type not in self.applicable_evidence_types for rule in self.metric_rules):
            raise ValueError("metric rule evidence type must be applicable")
        if self.metric_rules and (self.minimum_samples is None or self.calibration_status == "unconfigured"):
            raise ValueError("configured rules require explicit sample requirement and calibration status")
        if self.strategy == 'budgeted_paired_probe' and (self.max_probe_candidates is None or self.calibration_status == 'unconfigured' or self.metric_rules):
            raise ValueError('Budgeted probes require a frozen candidate budget and no hard metric gate')
        return self


class ValidationStagePlan(StrictProtocolModel):
    run_id: SafeRunId
    classification_version: Literal["backend-cost-v1"] = "backend-cost-v1"
    policy: PromotionPolicy
    policy_hash: str
    input_hash: str
    result_hash: str
    steps: dict[str, Literal["cheap_validation", "expensive_validation"]]
    task_checks: dict[str, dict[str, list[str]]]
    probe_eligible_candidate_ids: list[CandidateId] = Field(default_factory=list)


class PromotionResult(StrictProtocolModel):
    candidate_id: CandidateId
    decision: Literal["promoted", "terminated_low_signal", "deferred_insufficient_evidence"]
    reasons: list[str]
    measurement_ids: list[str]
    limitations: list[str] = Field(default_factory=list)


class PromotionReport(StrictProtocolModel):
    run_id: SafeRunId
    promotion_policy_version: str
    promotion_policy_hash: str
    input_hash: str
    result_hash: str
    results: list[PromotionResult]
