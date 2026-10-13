"""Fixture-only numerical cutoffs verify mechanics, not scientific calibration."""
import copy
import pytest
from pydantic import ValidationError
from shortcutminer.experiment.evidence_promotion import evaluate_promotion, load_or_evaluate_promotion, stable_hash
from shortcutminer.protocol.promotion import PromotionPolicy


def fixture_policy():
    return PromotionPolicy(policy_version="mechanics-fixture-v1", calibration_status="test_only", minimum_samples=8,
        metric_rules=[{"experiment_type": "statistical_check", "metric": "effect_size", "operator": "ge", "threshold": 0.2}])


def inputs(value=0.9, count=32, outcome="completed"):
    return {"candidate_ids": ["C001"], "measurements": [{"candidate_id": "C001", "measurement_id": "M-1",
        "experiment_type": "statistical_check", "execution_outcome": outcome,
        "metrics": {"effect_size": value, "valid_count": count}}], "qualitative_coverage": []}


@pytest.mark.parametrize("signal,count,status,expected", [
    (0.0, 32, "completed", "terminated_low_signal"), (0.9, 32, "completed", "promoted"),
    (0.9, 2, "completed", "deferred_insufficient_evidence"), (0.9, 32, "failed", "deferred_insufficient_evidence")])
def test_finalized_signal_and_coverage(signal, count, status, expected):
    report = evaluate_promotion("gate", fixture_policy(), inputs(signal, count, status))
    assert report.results[0].decision == expected
    assert report.results[0].decision not in {"confirmed", "contradicted", "false shortcut"}


def test_default_has_no_scientific_thresholds():
    policy = PromotionPolicy()
    assert policy.metric_rules == [] and policy.minimum_samples is None
    result = evaluate_promotion("gate", policy, inputs())
    assert result.results[0].decision == "deferred_insufficient_evidence"
    assert "promotion_policy_unconfigured" in result.results[0].reasons


def test_qualitative_llm_strength_does_not_promote():
    value = inputs()
    value["measurements"] = []
    value["qualitative_coverage"] = [{"candidate_id": "C001", "sample_count": 32, "observation_count": 25, "comparison_count": 5}]
    result = evaluate_promotion("gate", fixture_policy(), value)
    assert result.results[0].decision == "deferred_insufficient_evidence"
    assert "qualitative_promotion_unavailable" in result.results[0].limitations


def test_hash_stability_and_frozen_cache(tmp_path, monkeypatch):
    policy, value = fixture_policy(), inputs()
    original = load_or_evaluate_promotion(tmp_path, "gate", policy, value)
    assert original == evaluate_promotion("gate", policy, copy.deepcopy(value))
    assert original.promotion_policy_hash == stable_hash(policy.model_dump(mode="json"))
    monkeypatch.setattr("shortcutminer.experiment.evidence_promotion.evaluate_promotion", lambda *args: pytest.fail("Cached gate must be reused"))
    assert load_or_evaluate_promotion(tmp_path, "gate", policy, value) == original
    changed = policy.model_copy(update={"minimum_samples": 99})
    with pytest.raises(ValueError, match="inputs/policy"):
        load_or_evaluate_promotion(tmp_path, "gate", changed, value)


@pytest.mark.parametrize("where", ["top", "metrics", "future"])
def test_private_truth_and_future_measurements_rejected(where):
    value = inputs()
    if where == "top":
        value["hidden_truth"] = "never admitted"
    elif where == "metrics":
        value["measurements"][0]["metrics"]["hidden_truth"] = 1
    else:
        value["measurements"][0]["experiment_type"] = "model_evaluation"
    with pytest.raises(ValueError):
        evaluate_promotion("gate", fixture_policy(), value)


def test_policy_cannot_use_hidden_metric_or_unlabeled_threshold():
    with pytest.raises(ValidationError):
        PromotionPolicy(metric_rules=[{"experiment_type": "statistical_check", "metric": "hidden_truth", "operator": "ge", "threshold": 1}])
    with pytest.raises(ValidationError):
        PromotionPolicy(metric_rules=[{"experiment_type": "statistical_check", "metric": "effect_size", "operator": "ge", "threshold": 0.2}])
