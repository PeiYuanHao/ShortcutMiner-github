"""Post-workflow adapter from measured experiment output to blind impact records."""

from shortcutminer.evaluation.blind.protocol import DownstreamImpactRecord


def downstream_impact_from_output(output) -> DownstreamImpactRecord:
    metrics = output["metrics"] if isinstance(output, dict) else output.metrics
    references = output["artifact_references"] if isinstance(output, dict) else output.artifact_references
    metadata = output.get("metadata", {}) if isinstance(output, dict) else output.metadata
    details = metadata.get("details", {}) if isinstance(metadata, dict) else {}
    model_reference = next((ref for ref in references if ref.endswith("model.json")), None)
    before = metrics.get("id_accuracy", metrics.get("original_accuracy", metrics.get("accuracy")))
    after = metrics.get("counterfactual_accuracy", metrics.get("ood_accuracy"))
    ood = metrics.get("ood_accuracy")
    delta = metrics.get("accuracy_delta")
    available = any(isinstance(value, (int, float)) for value in (before, after, ood, delta))
    return DownstreamImpactRecord(
        available=available,
        model_reference=model_reference,
        model_id=details.get("model_id") or metrics.get("model_id"),
        model_hash=details.get("model_hash") or metrics.get("model_hash"),
        intervention_id=details.get("intervention_id") or metrics.get("intervention_id"),
        ood_id=details.get("ood_id") or metrics.get("ood_id"),
        comparison_mode=details.get("comparison_mode"),
        pair_coverage=details.get("pair_coverage"),
        uncertainty=list(metadata.get("uncertainty", [])) if isinstance(metadata, dict) else [],
        metric_before=float(before) if isinstance(before, (int, float)) else None,
        metric_after=float(after) if isinstance(after, (int, float)) else None,
        ood_metric=float(ood) if isinstance(ood, (int, float)) else None,
        intervention_delta=float(delta) if isinstance(delta, (int, float)) else None,
        reason_unavailable=None if available else "No model-level impact metric was present.",
    )


__all__ = ["downstream_impact_from_output"]
