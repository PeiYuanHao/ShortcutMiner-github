"""Snapshot-frozen, stratified aggregation of blind evaluation results."""

from __future__ import annotations

import numpy as np

from shortcutminer.evaluation.calibration import bootstrap_interval
from shortcutminer.evaluation.blind.protocol import (
    BlindEvaluationBatchResult,
    BlindEvaluationResult,
    StratifiedAggregateMetrics,
)
from shortcutminer.protocol.evidence import UncertaintyEstimate


class BlindEvaluationBatch:
    """Aggregate one fixed-snapshot batch without mixing positives and controls."""

    def __init__(
        self,
        *,
        snapshot_id: str,
        evaluation_bootstrap_seed: int,
        bootstrap_iterations: int = 500,
        confidence_level: float = 0.95,
    ) -> None:
        if not 0 <= bootstrap_iterations <= 1_000:
            raise ValueError("bootstrap_iterations must be between zero and 1,000")
        if not 0.8 <= confidence_level < 1.0:
            raise ValueError("confidence_level must be in [0.8, 1.0)")
        self.snapshot_id = snapshot_id
        self.evaluation_bootstrap_seed = evaluation_bootstrap_seed
        self.bootstrap_iterations = bootstrap_iterations
        self.confidence_level = confidence_level

    @staticmethod
    def _aggregate(items: list[BlindEvaluationResult], *, positive: bool) -> StratifiedAggregateMetrics:
        truth_count = sum(item.discovery_metrics.truth_count for item in items)
        matched = sum(item.discovery_metrics.matched_truth_count for item in items)
        exact = sum(item.localization_metrics.exact_count for item in items)
        localized = sum(
            item.localization_metrics.exact_count
            + item.localization_metrics.partial_count
            + item.localization_metrics.incorrect_count
            + item.localization_metrics.unknown_count
            for item in items
        )
        usable = sum(item.funnel_metrics.evidence_usable_count for item in items)
        return StratifiedAggregateMetrics(
            challenge_count=len(items),
            truth_count=truth_count,
            discovery_recall=(matched / truth_count if truth_count else None),
            candidate_false_positive_per_audit=(
                sum(item.false_positive_metrics.false_positive_candidate_count for item in items) / len(items)
                if items else None
            ),
            final_false_positive_per_audit=(
                sum(item.false_positive_metrics.false_positive_findings_count for item in items) / len(items)
                if items else None
            ),
            localization_rate=(exact / localized if localized else None),
            validation_completion_rate=(usable / truth_count if positive and truth_count else None),
        )

    def aggregate(
        self,
        results: list[BlindEvaluationResult],
        *,
        result_references: list[str] | None = None,
    ) -> BlindEvaluationBatchResult:
        if any(item.snapshot_id != self.snapshot_id for item in results):
            raise ValueError("all batch results must use the frozen snapshot_id")
        references = result_references or [
            f"{item.run_id}/evaluation/blind_result.json" for item in results
        ]
        if len(references) != len(results):
            raise ValueError("result reference count must match results")
        positive = [item for item in results if item.discovery_metrics.truth_count > 0]
        clean = [item for item in results if item.discovery_metrics.truth_count == 0]
        uncertainty: list[UncertaintyEstimate] = []
        if len(positive) >= 2:
            uncertainty.append(bootstrap_interval(
                metric="positive_discovery_recall",
                sample_count=len(positive),
                statistic=lambda indices: self._sampled_recall(positive, indices),
                confidence_level=self.confidence_level,
                iterations=self.bootstrap_iterations,
                seed=self.evaluation_bootstrap_seed,
                resampling_unit="independent_positive_challenge",
            ))
        else:
            uncertainty.append(UncertaintyEstimate(
                metric="positive_discovery_recall",
                available=False,
                reason_unavailable="At least two independent positive challenges are required for a batch interval.",
            ))
        if len(clean) >= 2:
            values = np.asarray([
                item.false_positive_metrics.false_positive_findings_count for item in clean
            ], dtype=float)
            uncertainty.append(bootstrap_interval(
                metric="clean_final_false_positive_per_audit",
                sample_count=len(clean),
                statistic=lambda indices: float(values[indices].mean()),
                confidence_level=self.confidence_level,
                iterations=self.bootstrap_iterations,
                seed=self.evaluation_bootstrap_seed + 1,
                resampling_unit="independent_clean_challenge",
            ))
        else:
            uncertainty.append(UncertaintyEstimate(
                metric="clean_final_false_positive_per_audit",
                available=False,
                reason_unavailable="At least two independent clean challenges are required for a batch interval.",
            ))
        return BlindEvaluationBatchResult(
            snapshot_id=self.snapshot_id,
            number_of_challenges=len(results),
            number_positive=len(positive),
            number_clean=len(clean),
            positive=self._aggregate(positive, positive=True),
            clean=self._aggregate(clean, positive=False),
            uncertainty=uncertainty,
            challenge_result_references=references,
            limitations=[
                "Batch intervals resample independent challenges, not samples within one challenge."
            ],
        )

    @staticmethod
    def _sampled_recall(
        items: list[BlindEvaluationResult], indices: np.ndarray
    ) -> float | None:
        selected = [items[int(index)] for index in indices]
        truth_count = sum(item.discovery_metrics.truth_count for item in selected)
        if not truth_count:
            return None
        matched = sum(item.discovery_metrics.matched_truth_count for item in selected)
        return matched / truth_count


__all__ = ["BlindEvaluationBatch"]
