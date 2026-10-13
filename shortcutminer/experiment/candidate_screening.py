"""Deterministic, public-information-only early candidate filtering.

No providers, tools, dataset files, private truth, or downstream evidence are read.
The only filesystem access is the current run's fixed screening artifact.
"""

import hashlib
import json
from pathlib import Path

from shortcutminer.protocol.dataset_profile import DatasetProfile
from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.protocol.screening import (
    CandidateScreeningPolicy, CandidateScreeningReport, CandidateScreeningResult,
    CandidateScreeningSignals,
)
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.storage.paths import get_run_dir


def _hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def screening_input_hash(report, profile, policy, signals) -> str:
    """Fingerprint only explicit upstream inputs; never whole workflow state."""
    return _hash({"report": report.model_dump(mode="json"),
                  "profile": profile.model_dump(mode="json"),
                  "policy": policy.model_dump(mode="json"),
                  "signals": signals.model_dump(mode="json")})


def _validate_inputs(report, profile, signals):
    if profile.dataset.name != report.dataset_summary.dataset:
        raise ValueError("Screening profile and report refer to different datasets")
    ids = [candidate.candidate_id for candidate in report.candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Screening input contains duplicate candidate IDs")
    if not (set(signals.confidence_by_candidate) | set(signals.cost_by_candidate)).issubset(ids):
        raise ValueError("Screening signals reference an unknown candidate")


def screen_candidates(
    report: CandidateReport, profile: DatasetProfile, *,
    policy: CandidateScreeningPolicy | None = None,
    signals: CandidateScreeningSignals | None = None,
) -> list[CandidateScreeningResult]:
    """Filter hypotheses, preserving IDs and report order without inferring truth."""
    policy = policy or CandidateScreeningPolicy()
    signals = signals or CandidateScreeningSignals()
    _validate_inputs(report, profile, signals)
    proposed = []
    for candidate in report.candidates:
        sources = set(candidate.evidence_sources)
        confidence = signals.confidence_by_candidate.get(candidate.candidate_id)
        # Relative initial inspection work only; this does not select experiments.
        cost = signals.cost_by_candidate.get(candidate.candidate_id,
                    1.0 if candidate.cue.family == "metadata" else 3.0 if sources & {
                        "sample_inspection", "image_statistics"} else 2.0)
        availability = []
        if "metadata" in sources and profile.metadata.available:
            availability.append(profile.metadata.rows_profiled)
        if sources & {"sample_inspection", "image_statistics"}:
            availability.append(report.dataset_summary.samples_inspected)
        if "dataset_structure" in sources:
            availability.append(profile.structure.entries_scanned)
        sample_count = max(availability, default=0)
        reasons = []
        if confidence is not None and confidence < policy.min_confidence:
            reasons.append("weak_signal")
        if candidate.cue.family == "metadata" and not profile.metadata.available:
            reasons.append("insufficient_metadata")
        if sample_count < policy.min_samples:
            reasons.append("insufficient_samples")
        if not availability or not any(availability):
            reasons.append("unsupported_validation")
        if policy.max_candidate_cost is not None and cost > policy.max_candidate_cost:
            reasons.append("resource_budget")
        proposed.append(CandidateScreeningResult(
            candidate_id=candidate.candidate_id, decision="screened_out" if reasons else "promote",
            confidence=confidence, sample_availability=sample_count,
            estimated_validation_cost=cost,
            reason=list(dict.fromkeys(reasons)) or ["public_evidence_available", "sufficient_samples"],
        ))
    # Budget preference is deterministic. Priority only ranks work; it is not confidence.
    priorities = {item.candidate_id: {"high": 0, "medium": 1, "low": 2}[item.priority]
                  for item in report.candidates}
    eligible = sorted((item for item in proposed if item.decision == "promote"), key=lambda item: (
        -(item.confidence if item.confidence is not None else -1),
        priorities[item.candidate_id], item.estimated_validation_cost, item.candidate_id))
    used_cost, accepted = 0.0, 0
    for item in eligible:
        if ((policy.max_promoted is not None and accepted >= policy.max_promoted)
                or (policy.max_total_cost is not None
                    and used_cost + item.estimated_validation_cost > policy.max_total_cost)):
            item.decision = "screened_out"
            item.reason = ["resource_budget"]
        else:
            accepted += 1
            used_cost += item.estimated_validation_cost
    return proposed


def build_screening_report(report, profile, *, policy=None, signals=None) -> CandidateScreeningReport:
    policy = policy or CandidateScreeningPolicy()
    signals = signals or CandidateScreeningSignals()
    results = screen_candidates(report, profile, policy=policy, signals=signals)
    promoted = sum(item.decision == "promote" for item in results)
    return CandidateScreeningReport(
        run_id=report.run_id, input_hash=screening_input_hash(report, profile, policy, signals),
        result_hash=_hash([item.model_dump(mode="json") for item in results]),
        candidates_generated=len(results), promoted=promoted, screened_out=len(results) - promoted,
        results=results,
    )


def validate_screening_report(screening, report, profile, policy, signals):
    _validate_inputs(report, profile, signals)
    if screening.run_id != report.run_id:
        raise ValueError("Screening run ID does not match Identify")
    if [item.candidate_id for item in screening.results] != [item.candidate_id for item in report.candidates]:
        raise ValueError("Screening results must cover every original candidate in order")
    if screening.input_hash != screening_input_hash(report, profile, policy, signals):
        raise ValueError("Screening inputs changed; persisted decisions cannot be reused")
    if screening.result_hash != _hash([item.model_dump(mode="json") for item in screening.results]):
        raise ValueError("Screening result hash does not match persisted decisions")


def load_or_screen_candidates(report, profile, *, policy=None, signals=None, runs_root=None):
    """Reuse matching persisted decisions after an interruption before checkpointing."""
    policy = policy or CandidateScreeningPolicy()
    signals = signals or CandidateScreeningSignals()
    _validate_inputs(report, profile, signals)
    path = None
    if runs_root is not None:
        run_dir = get_run_dir(runs_root, report.run_id)
        path = run_dir / "screening" / "result.json"
        if not path.resolve().is_relative_to(run_dir):
            raise ValueError("Screening artifact must remain within the current run")
        if path.is_file():
            prior = CandidateScreeningReport.model_validate_json(path.read_text(encoding="utf-8"))
            validate_screening_report(prior, report, profile, policy, signals)
            return prior
    screening = build_screening_report(report, profile, policy=policy, signals=signals)
    if path is not None:
        atomic_write_json(path, screening.model_dump(mode="json"), error_type=ValueError)
    return screening
