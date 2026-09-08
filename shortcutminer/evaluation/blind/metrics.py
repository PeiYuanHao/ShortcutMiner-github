"""Transparent discovery, localization, funnel, and false-positive metrics."""

from __future__ import annotations

from shortcutminer.evaluation.blind.protocol import (
    CandidateTruthMatch,
    DiscoveryMetrics,
    FalsePositiveMetrics,
    FunnelMetrics,
    HiddenShortcutTruth,
    LocalizationMetrics,
)


def discovery_metrics(
    matches: list[CandidateTruthMatch],
    truth: HiddenShortcutTruth,
    *,
    candidate_limit: int,
) -> DiscoveryMetrics:
    truth_ids = [item.truth_id for item in truth.shortcuts]
    matched_truths = {
        item.truth_id for item in matches
        if item.match_status == "matched" and item.truth_id is not None
    }
    ambiguous = sum(item.match_status in {"ambiguous", "potential_novel", "unassessed"} for item in matches)
    assessed = len(matches) - ambiguous
    matched_candidates = sum(item.match_status == "matched" for item in matches)
    recall_at_k = {}
    for k in sorted({1, 3, candidate_limit}):
        if not truth_ids:
            recall_at_k[f"recall@{k}"] = None
        else:
            top_truths = {
                item.truth_id for item in matches[:k]
                if item.match_status == "matched" and item.truth_id is not None
            }
            recall_at_k[f"recall@{k}"] = len(top_truths) / len(truth_ids)
    return DiscoveryMetrics(
        candidate_count=len(matches),
        truth_count=len(truth_ids),
        matched_truth_count=len(matched_truths),
        missed_truth_count=len(set(truth_ids) - matched_truths),
        unmatched_candidate_count=sum(item.match_status == "not_matched" for item in matches),
        ambiguous_candidate_count=ambiguous,
        discovery_recall=(len(matched_truths) / len(truth_ids) if truth_ids else None),
        candidate_precision=(matched_candidates / assessed if assessed else None),
        assessed_candidate_count=assessed,
        recall_at_k=recall_at_k,
    )


def false_positive_metrics(
    matches: list[CandidateTruthMatch],
    truth: HiddenShortcutTruth,
    final_assessments: list[str],
) -> FalsePositiveMetrics:
    if truth.shortcut_present:
        return FalsePositiveMetrics(
            false_positive_candidate_count=sum(item.match_status == "not_matched" for item in matches),
            false_positive_findings_count=0,
            partial_false_positive_findings_count=0,
            false_positive_per_audit=float(sum(item.match_status == "not_matched" for item in matches)),
        )
    candidate_fp = sum(item.match_status == "not_matched" for item in matches)
    return FalsePositiveMetrics(
        false_positive_candidate_count=candidate_fp,
        false_positive_findings_count=sum(value == "supported" for value in final_assessments),
        partial_false_positive_findings_count=sum(value == "partially_supported" for value in final_assessments),
        false_positive_per_audit=float(candidate_fp),
    )


def localization_metrics(matches: list[CandidateTruthMatch]) -> LocalizationMetrics:
    relevant = [item for item in matches if item.match_status in {"matched", "partial"}]
    exact = sum(item.localization_status == "exact" for item in relevant)
    partial = sum(item.localization_status == "partial" for item in relevant)
    incorrect = sum(item.localization_status == "incorrect" for item in relevant)
    unknown = sum(item.localization_status == "unknown" for item in relevant)
    denominator = exact + partial + incorrect + unknown
    return LocalizationMetrics(
        exact_count=exact,
        partial_count=partial,
        incorrect_count=incorrect,
        unknown_count=unknown,
        localization_rate=(exact / denominator if denominator else None),
    )


def funnel_metrics(
    *,
    truth_count: int,
    discovered_truth_ids: set[str],
    routed_truth_ids: set[str],
    quantitatively_tested_truth_ids: set[str],
    qualitatively_tested_truth_ids: set[str],
    usable_truth_ids: set[str],
    final_supported_truth_ids: set[str],
) -> FunnelMetrics:
    return FunnelMetrics(
        truth_count=truth_count,
        discovered_count=len(discovered_truth_ids),
        routed_count=len(routed_truth_ids),
        quantitatively_tested_count=len(quantitatively_tested_truth_ids),
        qualitatively_tested_count=len(qualitatively_tested_truth_ids),
        evidence_usable_count=len(usable_truth_ids),
        final_supported_count=len(final_supported_truth_ids),
        validation_completion_rate=(len(usable_truth_ids) / truth_count if truth_count else None),
    )


__all__ = [
    "discovery_metrics", "false_positive_metrics", "funnel_metrics", "localization_metrics"
]
