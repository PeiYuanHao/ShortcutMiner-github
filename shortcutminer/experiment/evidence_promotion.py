"""Frozen evidence-only promotion policy and deterministic recovery cache.

No models, datasets, evaluator inputs, provider reasoning or Stage-2 results.
No default scientific thresholds: an unconfigured policy always defers.
"""
import hashlib
import json
import math
import operator
from pathlib import Path

from shortcutminer.protocol.promotion import PromotionPolicy, PromotionResult, PromotionReport, ValidationStagePlan
from shortcutminer.protocol.batches import QuantitativeResultBatch, QualitativeResultBatch
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.storage.paths import get_run_dir

# Audited real_cpu_v1 implementations: bounded metadata statistics/counts and
# metrics over already authorized rows. Model probes, training, intervention,
# ablation and structured model-reliance backends all belong to Stage-2.
CHEAP_BACKENDS = frozenset({"statistical_check", "subgroup_evaluation", "metric_computation"})
# Existing descriptor backend and sample selection are bounded by their original
# preparation/image/sample budgets. No new budgets or inspection implementation.
CHEAP_QUALITATIVE_CHECKS = frozenset({"metadata_inspection", "sample_inspection", "visual_inspection", "group_comparison", "spatial_inspection", "failure_analysis", "evidence_synthesis"})
PUBLIC_NUMERIC_METRICS = frozenset({"effect_size", "p_value", "statistic", "accuracy", "balanced_accuracy", "overall_accuracy", "worst_group_accuracy", "group_accuracy_gap", "sample_count", "valid_count", "valid_sample_count"})


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _payload_hash(model):
    return stable_hash(model.model_dump(mode="json", exclude={"result_hash"}))


def make_stage_plan(run_id, routing, plans, policy, *, probe_candidate_ids=()):
    from shortcutminer.experiment.check_resolution import resolve_quantitative_checks
    steps = {step["step_id"]: ("cheap_validation" if step["experiment_type"] in CHEAP_BACKENDS else "expensive_validation")
             for plan in plans.values() for step in plan["steps"]}
    all_steps = [step for plan in plans.values() for step in plan["steps"]]
    for _ in range(len(all_steps)):
        for step in all_steps:
            if any(steps[dependency] == "expensive_validation" for dependency in step["dependency_step_ids"]):
                steps[step["step_id"]] = "expensive_validation"
    tasks = {}
    for task in routing["tasks"]:
        partition = {"cheap_validation": [], "expensive_validation": []}
        for check in task["requested_checks"]:
            cheap = (all(backend in CHEAP_BACKENDS for backend in resolve_quantitative_checks([check]))
                     if task["target"] == "quantitative" else check in CHEAP_QUALITATIVE_CHECKS)
            if any(step["planning_check"] == check and steps[step["step_id"]] == "expensive_validation"
                   for step in plans.get(task["task_id"], {}).get("steps", [])):
                cheap = False
            partition["cheap_validation" if cheap else "expensive_validation"].append(check)
        tasks[task["task_id"]] = partition
    result = ValidationStagePlan(run_id=run_id, policy=policy, policy_hash=stable_hash(policy.model_dump(mode="json")),
        input_hash=stable_hash({"routing": routing, "plans": plans}), result_hash="", steps=steps, task_checks=tasks,
        probe_eligible_candidate_ids=list(probe_candidate_ids))
    return result.model_copy(update={"result_hash": _payload_hash(result)})


def _cache_path(runs_root, run_id, relative):
    root = get_run_dir(runs_root, run_id)
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError("Unsafe promotion artifact path")
    return path


def freeze_stage_plan(runs_root, plan):
    path = _cache_path(runs_root, plan.run_id, "validation/stage_plan.json")
    if path.is_file():
        cached = ValidationStagePlan.model_validate_json(path.read_text(encoding="utf-8"))
        if cached != plan:
            raise ValueError("Frozen validation plan or promotion policy changed")
        policy_path = _cache_path(runs_root, plan.run_id, "promotion/policy.json")
        if policy_path.is_file():
            if PromotionPolicy.model_validate_json(policy_path.read_text(encoding="utf-8")) != plan.policy:
                raise ValueError("Frozen promotion policy artifact changed")
        else:
            atomic_write_json(policy_path, plan.policy.model_dump(mode="json"), error_type=ValueError)
        return cached
    atomic_write_json(path, plan.model_dump(mode="json"), error_type=ValueError)
    atomic_write_json(_cache_path(runs_root, plan.run_id, "promotion/policy.json"), plan.policy.model_dump(mode="json"), error_type=ValueError)
    return plan


def stage1_projection(candidates, quantitative, qualitative, *, run_id=None):
    qt = QuantitativeResultBatch.model_validate(quantitative)
    ql = QualitativeResultBatch.model_validate(qualitative)
    if qt.run_id != ql.run_id or (run_id is not None and qt.run_id != run_id):
        raise ValueError("Stage-1 evidence belongs to another run")
    measurements = []
    for result in qt.results:
        for item in result.payload.measurements:
            if item.experiment_type not in CHEAP_BACKENDS:
                raise ValueError("Stage-2 measurement cannot enter promotion inputs")
            if item.candidate_id != result.payload.candidate_id or item.task_id != result.payload.task_id:
                raise ValueError("Stage-1 measurement correlation changed")
            metrics = {key: value for key, value in item.metrics.items() if key in PUBLIC_NUMERIC_METRICS
                       and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)}
            measurements.append({"candidate_id": item.candidate_id, "measurement_id": item.measurement_id,
                                 "experiment_type": item.experiment_type, "execution_outcome": item.execution_outcome if result.payload.status == "completed" else "blocked",
                                 "metrics": metrics})
    # The current qualitative schema does not encode a validated strength rule.
    # Only finalized presence/coverage are retained; LLM assessment is never a gate.
    observations = [{"candidate_id": result.payload.candidate_id, "sample_count": result.payload.sampling.sample_count,
                     "observation_count": len(result.payload.observations), "comparison_count": len(result.payload.comparisons)} for result in ql.results]
    return {"candidate_ids": candidates, "measurements": measurements, "qualitative_coverage": observations}


def evaluate_promotion(run_id, policy, inputs, *, probe_candidate_ids=()):
    _validate_inputs(inputs)
    results = []
    probe_candidates = set(probe_candidate_ids)
    if policy.strategy == 'budgeted_paired_probe':
        # Rank only by finalized cheap measurements; provider confidence is not a gate.
        def score(candidate):
            values = [m['metrics'].get('effect_size', 0) for m in inputs['measurements']
                      if m['candidate_id'] == candidate and m['execution_outcome'] == 'completed']
            return max(values, default=0)
        ordered = sorted((c for c in inputs['candidate_ids'] if c in probe_candidates), key=lambda c: (-score(c), c))
        chosen = set(ordered[:policy.max_probe_candidates])
        for candidate in inputs['candidate_ids']:
            reason = 'bounded_paired_probe_allocated' if candidate in chosen else ('probe_candidate_budget_exhausted' if candidate in probe_candidates else 'paired_probe_not_materializable')
            results.append(PromotionResult(candidate_id=candidate, decision='promoted' if candidate in chosen else 'deferred_insufficient_evidence',
                reasons=[reason], measurement_ids=[], limitations=['Promotion allocates probe budget; it is not a shortcut verdict.']))
        result = PromotionReport(run_id=run_id, promotion_policy_version=policy.policy_version,
            promotion_policy_hash=stable_hash(policy.model_dump(mode='json')),
            input_hash=stable_hash({'stage1': inputs, 'probe_candidate_ids': list(probe_candidate_ids)}), result_hash='', results=results)
        return result.model_copy(update={'result_hash': _payload_hash(result)})
    for candidate in inputs["candidate_ids"]:
        selected = [m for m in inputs["measurements"] if m["candidate_id"] == candidate and m["execution_outcome"] == "completed"]
        eligible = []
        for rule in policy.metric_rules:
            values = [m for m in selected if m["experiment_type"] == rule.experiment_type and rule.metric in m["metrics"]
                      and m["metrics"].get("valid_count", m["metrics"].get("valid_sample_count", m["metrics"].get("sample_count", 0))) >= policy.minimum_samples]
            compare = {"ge": operator.ge, "le": operator.le, "gt": operator.gt, "lt": operator.lt}[rule.operator]
            eligible.extend((m, rule, compare(m["metrics"][rule.metric], rule.threshold)) for m in values)
        if any(match for _, _, match in eligible):
            decision, reasons = "promoted", ["configured_metric_rule_satisfied"]
        elif eligible and all(any(rule == checked_rule for _, checked_rule, _ in eligible) for rule in policy.metric_rules):
            decision, reasons = "terminated_low_signal", ["configured_metric_rules_not_satisfied"]
        else:
            decision, reasons = "deferred_insufficient_evidence", ["promotion_policy_unconfigured" if not policy.metric_rules else "insufficient_stage1_evidence"]
        results.append(PromotionResult(candidate_id=candidate, decision=decision, reasons=reasons,
            measurement_ids=[m["measurement_id"] for m, _, _ in eligible], limitations=["qualitative_promotion_unavailable"]))
    result = PromotionReport(run_id=run_id, promotion_policy_version=policy.policy_version,
        promotion_policy_hash=stable_hash(policy.model_dump(mode="json")), input_hash=stable_hash(inputs), result_hash="", results=results)
    return result.model_copy(update={"result_hash": _payload_hash(result)})


def load_or_evaluate_promotion(runs_root, run_id, policy, inputs, *, probe_candidate_ids=()):
    _validate_inputs(inputs)
    path = _cache_path(runs_root, run_id, "promotion/result.json")
    if path.is_file():
        result = PromotionReport.model_validate_json(path.read_text(encoding="utf-8"))
        expected_inputs = {'stage1': inputs, 'probe_candidate_ids': list(probe_candidate_ids)} if policy.strategy == 'budgeted_paired_probe' else inputs
        if result.run_id != run_id or result.input_hash != stable_hash(expected_inputs) or result.promotion_policy_hash != stable_hash(policy.model_dump(mode="json")) or result.result_hash != _payload_hash(result):
            raise ValueError("Frozen promotion result does not match Stage-1 inputs/policy")
        return result
    result = evaluate_promotion(run_id, policy, inputs, probe_candidate_ids=probe_candidate_ids)
    atomic_write_json(path, result.model_dump(mode="json"), error_type=ValueError)
    return result


def _validate_inputs(inputs):
    if set(inputs) != {"candidate_ids", "measurements", "qualitative_coverage"}:
        raise ValueError("Only projected Stage-1 evidence can enter promotion")
    if len(set(inputs["candidate_ids"])) != len(inputs["candidate_ids"]):
        raise ValueError("Duplicate candidate identity")
    for measurement in inputs["measurements"]:
        if set(measurement) != {"candidate_id", "measurement_id", "experiment_type", "execution_outcome", "metrics"}:
            raise ValueError("Only finalized measurement projection can enter promotion")
        if measurement["experiment_type"] not in CHEAP_BACKENDS or measurement["candidate_id"] not in inputs["candidate_ids"]:
            raise ValueError("Stage-2 or foreign-candidate evidence is forbidden")
        if set(measurement["metrics"]) - PUBLIC_NUMERIC_METRICS:
            raise ValueError("Private or unsupported metrics cannot enter promotion")
    for coverage in inputs["qualitative_coverage"]:
        if set(coverage) != {"candidate_id", "sample_count", "observation_count", "comparison_count"} or coverage["candidate_id"] not in inputs["candidate_ids"]:
            raise ValueError("Only finalized qualitative coverage can enter promotion")
