"""Explicit, evaluation-only candidate-to-truth matching and adjudication."""

from __future__ import annotations

from collections import Counter, defaultdict
import re

from shortcutminer.evaluation.blind.protocol import (
    AdjudicationAggregate,
    AdjudicationAgreement,
    CandidateTruthMatch,
    HiddenShortcutSpec,
    HiddenShortcutTruth,
    ManualAdjudication,
)
from shortcutminer.protocol.evidence import EvidenceLedger
from shortcutminer.protocol.identify import Candidate, CandidateReport


def _tokens(*values: str) -> set[str]:
    return {
        token
        for value in values
        for token in re.findall(r"[a-z0-9_]+", value.casefold())
    }


def _record_tokens(candidate_id: str, ledger: EvidenceLedger | None) -> set[str]:
    if ledger is None:
        return set()
    tokens: set[str] = set()
    for entry in ledger.candidates:
        if entry.candidate_id != candidate_id:
            continue
        for record in entry.records:
            for key in ("field", "cue_field", "spatial_region", "acquisition_family"):
                value = record.metrics.get(key)
                if isinstance(value, str):
                    tokens.update(_tokens(value))
    return tokens


def _deterministic_match(
    candidate: Candidate,
    truth: HiddenShortcutSpec,
    ledger: EvidenceLedger | None,
) -> CandidateTruthMatch:
    candidate_tokens = _tokens(
        candidate.cue.name,
        candidate.description,
        candidate.observation,
        candidate.hypothesis,
    )
    evidence_tokens = _record_tokens(candidate.candidate_id, ledger)
    explicit_tokens = candidate_tokens | evidence_tokens
    expected_family = {
        "metadata": "metadata",
        "spatial": "spatial",
        "acquisition": "acquisition",
        "semantic": "context",
    }[truth.family]
    family_compatible = candidate.cue.family == expected_family

    target = None
    if truth.family == "metadata":
        target = truth.target_field
    elif truth.family == "spatial":
        target = truth.spatial_region
    elif truth.family == "acquisition":
        target = truth.acquisition_family
    else:
        target = truth.target_field
    assert target is not None
    target_tokens = _tokens(target)
    exact = target.casefold() == candidate.cue.name.casefold() or target_tokens <= explicit_tokens
    if exact and family_compatible:
        return CandidateTruthMatch(
            candidate_id=candidate.candidate_id,
            truth_id=truth.truth_id,
            match_status="matched",
            localization_status="exact",
            rationale="Candidate/evidence explicitly names the hidden target and compatible cue family.",
            adjudication_source="deterministic",
        )
    if exact:
        return CandidateTruthMatch(
            candidate_id=candidate.candidate_id,
            truth_id=truth.truth_id,
            match_status="partial",
            localization_status="exact",
            rationale="Target localization is explicit, but the candidate cue family differs.",
            adjudication_source="deterministic",
        )
    if family_compatible:
        return CandidateTruthMatch(
            candidate_id=candidate.candidate_id,
            truth_id=truth.truth_id,
            match_status="ambiguous",
            localization_status="unknown",
            rationale="Cue family is compatible but no exact target field, region, or transformation is present.",
            adjudication_source="none",
        )
    return CandidateTruthMatch(
        candidate_id=candidate.candidate_id,
        truth_id=truth.truth_id,
        match_status="not_matched",
        localization_status="incorrect",
        rationale="No explicit family-compatible target localization was found; fuzzy matching was not used.",
        adjudication_source="deterministic",
    )


def match_candidates(
    candidate_report: CandidateReport,
    hidden_truth: HiddenShortcutTruth,
    *,
    evidence_ledger: EvidenceLedger | None = None,
    adjudications: list[ManualAdjudication] | None = None,
    clean_truth_is_exhaustive: bool = True,
) -> list[CandidateTruthMatch]:
    """Match only explicit controlled targets; ambiguous cases remain reviewable."""
    manual = {
        (item.candidate_id, item.truth_id): item for item in adjudications or []
    }
    matches: list[CandidateTruthMatch] = []
    for candidate in candidate_report.candidates:
        if not hidden_truth.shortcuts:
            matches.append(CandidateTruthMatch(
                candidate_id=candidate.candidate_id,
                truth_id=None,
                match_status="not_matched" if clean_truth_is_exhaustive else "potential_novel",
                localization_status="not_applicable",
                rationale=(
                    "The controlled clean generator declares an exhaustive no-shortcut construction."
                    if clean_truth_is_exhaustive
                    else "No injected truth matched; the candidate requires novel-pattern adjudication."
                ),
                adjudication_source="deterministic" if clean_truth_is_exhaustive else "none",
            ))
            continue
        candidate_matches = [
            _deterministic_match(candidate, truth, evidence_ledger)
            for truth in hidden_truth.shortcuts
        ]
        rank = {"matched": 0, "partial": 1, "ambiguous": 2, "not_matched": 3}
        selected = min(candidate_matches, key=lambda item: rank[item.match_status])
        override = manual.get((selected.candidate_id, selected.truth_id))
        if override is not None:
            selected = CandidateTruthMatch(
                candidate_id=override.candidate_id,
                truth_id=override.truth_id,
                match_status=override.decision,
                localization_status=override.localization,
                rationale=override.rationale,
                adjudication_source="manual",
            )
        matches.append(selected)
    return matches


def aggregate_adjudications(
    adjudications: list[ManualAdjudication],
) -> AdjudicationAgreement:
    """Aggregate future multi-rater labels without inventing reviewer evidence."""
    if not adjudications:
        raise ValueError("at least one adjudication is required")
    grouped: dict[tuple[str, str | None], list[ManualAdjudication]] = defaultdict(list)
    reviewers = sorted({item.reviewer_id for item in adjudications})
    for item in adjudications:
        grouped[(item.candidate_id, item.truth_id)].append(item)
    aggregates = []
    agreements = 0
    for (candidate_id, truth_id), items in sorted(grouped.items()):
        decisions = Counter(item.decision for item in items)
        localizations = Counter(item.localization for item in items)
        decision_consensus = decisions.most_common()
        localization_consensus = localizations.most_common()
        unanimous = len(decisions) == 1 and len(localizations) == 1
        agreements += int(unanimous)
        aggregates.append(AdjudicationAggregate(
            candidate_id=candidate_id,
            truth_id=truth_id,
            rater_count=len(items),
            decision_counts=dict(decisions),
            localization_counts=dict(localizations),
            raw_agreement=1.0 if unanimous else 0.0,
            consensus_decision=(decision_consensus[0][0] if len(decision_consensus) == 1 or decision_consensus[0][1] > decision_consensus[1][1] else None),
            consensus_localization=(localization_consensus[0][0] if len(localization_consensus) == 1 or localization_consensus[0][1] > localization_consensus[1][1] else None),
        ))

    kappa = None
    if len(reviewers) == 2 and all(
        {item.reviewer_id for item in items} == set(reviewers) and len(items) == 2
        for items in grouped.values()
    ):
        first = []
        second = []
        for key in sorted(grouped):
            by_reviewer = {item.reviewer_id: item.decision for item in grouped[key]}
            first.append(by_reviewer[reviewers[0]])
            second.append(by_reviewer[reviewers[1]])
        observed = sum(left == right for left, right in zip(first, second, strict=True)) / len(first)
        first_counts = Counter(first)
        second_counts = Counter(second)
        categories = set(first_counts) | set(second_counts)
        expected = sum(
            (first_counts[value] / len(first)) * (second_counts[value] / len(second))
            for value in categories
        )
        if expected < 1.0:
            kappa = (observed - expected) / (1.0 - expected)
    return AdjudicationAgreement(
        reviewer_ids=reviewers,
        item_count=len(grouped),
        raw_agreement=agreements / len(grouped),
        cohens_kappa=kappa,
        item_aggregates=aggregates,
    )


__all__ = ["aggregate_adjudications", "match_candidates"]
