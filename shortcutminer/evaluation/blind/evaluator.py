"""Post-workflow blind evaluator; hidden truth never enters production runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from shortcutminer.evaluation.blind.leakage import assert_path_isolation
from shortcutminer.evaluation.blind.matching import match_candidates
from shortcutminer.evaluation.blind.metrics import (
    discovery_metrics,
    false_positive_metrics,
    funnel_metrics,
    localization_metrics,
)
from shortcutminer.evaluation.blind.protocol import (
    BlindChallenge,
    BlindEvaluationResult,
    DiscoveryJudgement,
    DownstreamImpactRecord,
    EvaluationSnapshot,
    FailureRecord,
    HiddenShortcutTruth,
    ManualAdjudication,
)
from shortcutminer.evaluation.blind.snapshot import save_evaluation_snapshot
from shortcutminer.protocol.batches import QualitativeResultBatch, QuantitativeResultBatch
from shortcutminer.protocol.evidence import EvidenceLedger
from shortcutminer.protocol.executive import RoutingPlan
from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.protocol.summarize import SummaryReport
from shortcutminer.storage.json_artifacts import atomic_write_json


class BlindEvaluationError(RuntimeError):
    """Raised when post-workflow evaluation violates its firewall or contract."""


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_hidden_truth(
    path: str | Path,
    *,
    expected_hash: str,
) -> HiddenShortcutTruth:
    """Read only the explicitly approved private truth path after workflow completion."""
    resolved = Path(path).resolve(strict=True)
    try:
        truth = HiddenShortcutTruth.model_validate_json(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BlindEvaluationError("private truth is unavailable or invalid") from exc
    actual_hash = _canonical_hash(truth.model_dump(mode="json"))
    if actual_hash != expected_hash:
        raise BlindEvaluationError("private truth hash does not match challenge")
    return truth


def _model_hash(*models) -> str:
    return _canonical_hash([item.model_dump(mode="json") for item in models])


class BlindEvaluator:
    """Match, label, and measure immutable completed workflow outputs."""

    def evaluate(
        self,
        *,
        challenge: BlindChallenge,
        challenge_root: str | Path,
        hidden_truth_path: str | Path,
        snapshot: EvaluationSnapshot,
        run_id: str,
        candidate_report: CandidateReport,
        routing_plan: RoutingPlan,
        quantitative_results: QuantitativeResultBatch,
        qualitative_results: QualitativeResultBatch,
        evidence_ledger: EvidenceLedger,
        summary_report: SummaryReport,
        adjudications: list[ManualAdjudication] | None = None,
        downstream_impact: DownstreamImpactRecord | None = None,
        runs_root: str | Path | None = None,
    ) -> BlindEvaluationResult:
        root = Path(challenge_root).resolve(strict=True)
        dataset_root = (root / challenge.public.dataset_reference).resolve(strict=True)
        assert_path_isolation(
            challenge_root=root,
            dataset_root=dataset_root,
            hidden_truth_path=hidden_truth_path,
        )
        if snapshot.challenge_public_hash != challenge.public.public_dataset_hash:
            raise BlindEvaluationError("snapshot public hash differs from challenge")
        if any(item.run_id != run_id for item in (
            candidate_report, routing_plan, quantitative_results,
            qualitative_results, evidence_ledger, summary_report,
        )):
            raise BlindEvaluationError("workflow outputs do not share evaluation run_id")
        before = _model_hash(
            candidate_report, routing_plan, quantitative_results,
            qualitative_results, evidence_ledger, summary_report,
        )
        truth = load_hidden_truth(
            hidden_truth_path, expected_hash=challenge.private_truth_hash
        )
        if truth.challenge_id != challenge.challenge_id:
            raise BlindEvaluationError("private truth challenge_id mismatch")
        if truth.public_dataset_hash != challenge.public.public_dataset_hash:
            raise BlindEvaluationError("private truth refers to another public dataset")

        matches = match_candidates(
            candidate_report,
            truth,
            evidence_ledger=evidence_ledger,
            adjudications=adjudications,
        )
        final_by_candidate = {
            item.candidate_id: item.assessment for item in summary_report.payload.findings
        }
        decisions_by_candidate = {
            item.candidate.candidate_id: item for item in routing_plan.decisions
        }
        quant_by_candidate = {
            item.payload.candidate_id: item for item in quantitative_results.results
        }
        qual_by_candidate = {
            item.payload.candidate_id: item for item in qualitative_results.results
        }
        ledger_by_candidate = {
            item.candidate_id: item for item in evidence_ledger.candidates
        }
        judgements = []
        for candidate in candidate_report.candidates:
            candidate_matches = [
                item for item in matches
                if item.candidate_id == candidate.candidate_id and item.match_status == "matched"
            ]
            decision = decisions_by_candidate.get(candidate.candidate_id)
            quant = quant_by_candidate.get(candidate.candidate_id)
            qual = qual_by_candidate.get(candidate.candidate_id)
            ledger_entry = ledger_by_candidate.get(candidate.candidate_id)
            judgements.append(DiscoveryJudgement(
                candidate_id=candidate.candidate_id,
                matched_truth_ids=[item.truth_id for item in candidate_matches if item.truth_id],
                routed=bool(decision and decision.task_ids),
                quantitatively_tested=bool(quant and quant.payload.status == "completed"),
                qualitatively_tested=bool(
                    qual and (
                        qual.payload.observations
                        or qual.payload.metadata_evidence
                        or qual.payload.counterevidence
                    )
                ),
                evidence_usable=bool(
                    ledger_entry
                    and ledger_entry.profile.consistency not in {
                        "inconclusive", "no_usable_evidence"
                    }
                ),
                final_assessment=final_by_candidate.get(candidate.candidate_id),
            ))

        discovered = {
            truth_id for item in judgements for truth_id in item.matched_truth_ids
        }
        by_candidate_judgement = {item.candidate_id: item for item in judgements}
        routed = {
            item.truth_id for item in matches
            if item.match_status == "matched" and item.truth_id
            and by_candidate_judgement[item.candidate_id].routed
        }
        quant_tested = {
            item.truth_id for item in matches
            if item.match_status == "matched" and item.truth_id
            and by_candidate_judgement[item.candidate_id].quantitatively_tested
        }
        qual_tested = {
            item.truth_id for item in matches
            if item.match_status == "matched" and item.truth_id
            and by_candidate_judgement[item.candidate_id].qualitatively_tested
        }
        usable = {
            item.truth_id for item in matches
            if item.match_status == "matched" and item.truth_id
            and by_candidate_judgement[item.candidate_id].evidence_usable
        }
        supported = {
            item.truth_id for item in matches
            if item.match_status == "matched" and item.truth_id
            and by_candidate_judgement[item.candidate_id].final_assessment
            in {"supported", "partially_supported"}
        }
        failures = []
        for hidden in truth.shortcuts:
            related_matches = [
                item for item in matches if item.truth_id == hidden.truth_id
            ]
            if hidden.truth_id not in discovered:
                reason = (
                    "discovered_wrong_localization"
                    if any(
                        item.match_status in {"partial", "ambiguous"}
                        for item in related_matches
                    )
                    else "not_discovered"
                )
                failures.append(FailureRecord(
                    truth_id=hidden.truth_id,
                    reason=reason,
                    details=(
                        "A related candidate was present, but its cue family or exact localization was incomplete."
                        if reason == "discovered_wrong_localization"
                        else "No candidate was deterministically or manually matched to this truth."
                    ),
                ))
                continue
            if hidden.truth_id not in routed:
                failures.append(FailureRecord(
                    truth_id=hidden.truth_id,
                    reason="routing_failure",
                    details="The matched candidate did not receive a routed validation task.",
                ))
                continue
            matched_candidate_ids = {
                item.candidate_id
                for item in related_matches
                if item.match_status == "matched"
            }
            matched_decisions = [
                decisions_by_candidate[candidate_id]
                for candidate_id in matched_candidate_ids
                if candidate_id in decisions_by_candidate
            ]
            missing_quantitative = (
                any(item.route in {"quantitative", "both"} for item in matched_decisions)
                and hidden.truth_id not in quant_tested
            )
            missing_qualitative = (
                any(item.route in {"qualitative", "both"} for item in matched_decisions)
                and hidden.truth_id not in qual_tested
            )
            if missing_quantitative:
                failures.append(FailureRecord(
                    truth_id=hidden.truth_id,
                    reason="quantitative_blocked",
                    details="The routed quantitative validation did not complete successfully.",
                ))
            if missing_qualitative:
                failures.append(FailureRecord(
                    truth_id=hidden.truth_id,
                    reason="qualitative_unusable",
                    details="The routed qualitative validation produced no usable inspection evidence.",
                ))
            if missing_quantitative or missing_qualitative:
                continue
            if hidden.truth_id not in usable:
                failures.append(FailureRecord(
                    truth_id=hidden.truth_id,
                    reason="evidence_inconclusive",
                    details="The candidate was discovered but its calibrated evidence was not usable.",
                ))
            elif hidden.truth_id not in supported:
                failures.append(FailureRecord(
                    truth_id=hidden.truth_id,
                    reason="summary_underclaim",
                    details="Usable matched evidence did not produce a supported or partial final assessment.",
                ))
        if not truth.shortcut_present:
            for finding in summary_report.payload.findings:
                if finding.assessment in {"supported", "partially_supported"}:
                    failures.append(FailureRecord(
                        candidate_id=finding.candidate_id,
                        reason="summary_overclaim",
                        details="A clean control received a supported or partially supported final finding.",
                    ))

        result = BlindEvaluationResult(
            challenge_id=challenge.challenge_id,
            snapshot_id=snapshot.snapshot_id,
            run_id=run_id,
            truth_reference_hash=challenge.private_truth_hash,
            candidate_matches=matches,
            discovery_judgements=judgements,
            discovery_metrics=discovery_metrics(
                matches, truth, candidate_limit=snapshot.candidate_limit
            ),
            false_positive_metrics=false_positive_metrics(
                matches, truth, list(final_by_candidate.values())
            ),
            localization_metrics=localization_metrics(matches),
            funnel_metrics=funnel_metrics(
                truth_count=len(truth.shortcuts),
                discovered_truth_ids=discovered,
                routed_truth_ids=routed,
                quantitatively_tested_truth_ids=quant_tested,
                qualitatively_tested_truth_ids=qual_tested,
                usable_truth_ids=usable,
                final_supported_truth_ids=supported,
            ),
            failures=failures,
            downstream_impact=downstream_impact or DownstreamImpactRecord(
                available=False,
                reason_unavailable="No downstream model-impact measurement was supplied for this infrastructure evaluation.",
            ),
            limitations=[
                "Synthetic controlled truth evaluates harness behavior, not real-world shortcut-discovery performance."
            ],
        )
        after = _model_hash(
            candidate_report, routing_plan, quantitative_results,
            qualitative_results, evidence_ledger, summary_report,
        )
        if before != after:
            raise BlindEvaluationError("evaluator mutated production workflow outputs")
        if runs_root is not None:
            run_root = (Path(runs_root) / run_id).resolve(strict=False)
            save_evaluation_snapshot(snapshot, run_root=run_root)
            path = (run_root / "evaluation" / "blind_result.json").resolve(strict=False)
            try:
                path.relative_to(run_root)
            except ValueError as exc:
                raise BlindEvaluationError("evaluation artifact escaped run root") from exc
            atomic_write_json(
                path, result.model_dump(mode="json"), error_type=BlindEvaluationError
            )
        return result


__all__ = ["BlindEvaluationError", "BlindEvaluator", "load_hidden_truth"]
