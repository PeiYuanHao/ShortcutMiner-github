"""Deterministic construction and persistence of the evidence ledger."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from shortcutminer.evaluation.calibration import benjamini_hochberg
from shortcutminer.protocol.batches import QualitativeResultBatch, QuantitativeResultBatch
from shortcutminer.protocol.evidence import (
    CandidateEvidenceLedger,
    CandidateEvidenceProfile,
    EvidenceCoverage,
    EvidenceLedger,
    EvidenceProvenance,
    EvidenceRecord,
    StatisticalTestRecord,
    UncertaintyEstimate,
)
from shortcutminer.protocol.executive import AnalysisTask, RoutingPlan
from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.storage.json_artifacts import atomic_write_json


_DESIGN_ORDER = {"descriptive": 0, "observational": 1, "predictive": 2, "interventional": 3}
_Q_DESIGN = {"Q0": "descriptive", "Q1": "descriptive", "Q2": "observational", "Q3": "predictive", "Q4": "interventional"}


def _fingerprint(*objects: Any) -> str:
    payload = [obj.model_dump(mode="json") for obj in objects]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quantitative_kind(result, task: AnalysisTask | None) -> str:
    text = " ".join(
        [result.payload.analysis_type, *(task.requested_checks if task else [])]
    ).casefold()
    if "counterfactual" in text:
        return "quantitative_counterfactual"
    if "ood" in text or "out-of-distribution" in text:
        return "quantitative_ood"
    if "task_model" in text or "model reliance" in text:
        return "quantitative_model_reliance"
    if "ablation" in text:
        return "quantitative_ablation"
    if "intervention" in text or "permutation" in text:
        return "quantitative_intervention"
    if "probe" in text or "predict" in text or result.payload.evidence_level == "Q3":
        return "quantitative_predictive_probe"
    if "subgroup" in text or "group" in text:
        return "quantitative_subgroup"
    if "metric" in text or "accuracy" in text:
        return "quantitative_metric"
    return "quantitative_association"


def _artifact_metadata(
    run_root: Path | None,
    references: list[str],
    *,
    run_id: str,
    candidate_id: str,
    task_id: str,
) -> tuple[list[str], list[UncertaintyEstimate], list[str], list[str], list[str], dict[str, Any]]:
    approved: list[str] = []
    uncertainty: list[UncertaintyEstimate] = []
    inputs: list[str] = []
    warnings: list[str] = []
    limitations: list[str] = []
    model_details: dict[str, Any] = {}
    for reference in references:
        if run_root is None:
            approved.append(reference)
            continue
        path = (run_root / reference).resolve(strict=False)
        try:
            path.relative_to(run_root.resolve(strict=False))
        except ValueError:
            limitations.append(f"Artifact reference escaped the approved run root: {reference}")
            continue
        if not path.is_file():
            limitations.append(f"Referenced artifact was unavailable during calibration: {reference}")
            continue
        if path.suffix.casefold() != ".json":
            approved.append(reference)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            limitations.append(f"Referenced artifact could not be validated during calibration: {reference}")
            continue
        expected = {"run_id": run_id, "candidate_id": candidate_id, "task_id": task_id}
        if any(key in data and data[key] != value for key, value in expected.items()):
            limitations.append(f"Referenced artifact provenance did not match the result: {reference}")
            continue
        approved.append(reference)
        inputs.extend(str(value) for value in data.get("input_references", []) if value)
        warnings.extend(str(value) for value in data.get("warnings", []) if value)
        limitations.extend(str(value) for value in data.get("limitations", []) if value)
        details = data.get("details", {})
        if isinstance(details, dict):
            for key in ("model_id", "model_hash", "intervention_id", "ood_id", "comparison_mode", "pair_coverage"):
                if key in details:
                    model_details[key] = details[key]
        for key in ("model_id", "model_hash", "intervention_id", "ood_id", "comparison_mode", "pair_coverage"):
            if key in data:
                model_details[key] = data[key]
        for item in data.get("uncertainty", []):
            try:
                uncertainty.append(UncertaintyEstimate.model_validate(item))
            except Exception:
                limitations.append(f"Artifact uncertainty was incompatible and omitted: {reference}")
    return approved, uncertainty, inputs, warnings, limitations, model_details


def _validated_artifact_references(
    run_root: Path | None,
    references: list[str],
    *,
    run_id: str,
    candidate_id: str,
    task_id: str,
) -> tuple[list[str], list[str]]:
    if run_root is None:
        return list(references), []
    approved: list[str] = []
    limitations: list[str] = []
    for reference in references:
        path = (run_root / reference).resolve(strict=False)
        try:
            path.relative_to(run_root.resolve(strict=False))
        except ValueError:
            limitations.append(f"Artifact reference escaped the approved run root: {reference}")
            continue
        if not path.is_file():
            limitations.append(f"Referenced artifact was unavailable during calibration: {reference}")
            continue
        if path.suffix.casefold() == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                limitations.append(f"Referenced artifact could not be validated during calibration: {reference}")
                continue
            expected = {"run_id": run_id, "candidate_id": candidate_id, "task_id": task_id}
            if any(key in data and data[key] != value for key, value in expected.items()):
                limitations.append(f"Referenced artifact provenance did not match the result: {reference}")
                continue
        approved.append(reference)
    return approved, limitations


def _coverage_from_metrics(metrics: dict[str, Any], *, state: str) -> EvidenceCoverage:
    sample_count = metrics.get("sample_count")
    valid = metrics.get("valid_count", metrics.get("test_n"))
    missing = metrics.get("missing_count", metrics.get("missing_prediction_count"))
    return EvidenceCoverage(
        requested=1,
        attempted=0 if state == "blocked" else 1,
        usable=1 if state == "usable" else 0,
        failed=1 if state == "failed" else 0,
        inconclusive=1 if state == "scientifically_inconclusive" else 0,
        sample_count=int(sample_count) if isinstance(sample_count, int) else None,
        valid_sample_count=int(valid) if isinstance(valid, int) else None,
        missing_count=int(missing) if isinstance(missing, int) else None,
    )


def _quantitative_records(result, task: AnalysisTask | None, run_root: Path | None) -> list[EvidenceRecord]:
    payload = result.payload
    if payload.status == "completed" and not payload.measurements:
        raise ValueError("completed QuantitativeResult lacks canonical measurements")
    measurements = payload.measurements or [None]
    records: list[EvidenceRecord] = []
    for measurement in measurements:
        records.append(_quantitative_measurement_record(result, task, run_root, measurement))
    return records


_MEASUREMENT_KINDS = {
    "statistical_check": "quantitative_association",
    "subgroup_evaluation": "quantitative_subgroup",
    "model_evaluation": "quantitative_predictive_probe",
    "intervention": "quantitative_intervention",
    "ablation": "quantitative_ablation",
    "metric_computation": "quantitative_metric",
    "task_model_training": "quantitative_model_reliance",
    "task_model_evaluation": "quantitative_model_reliance",
    "ood_evaluation": "quantitative_ood",
    "counterfactual_intervention": "quantitative_counterfactual",
}


def _quantitative_measurement_record(result, task, run_root, measurement) -> EvidenceRecord:
    payload = result.payload
    metrics = measurement.metrics if measurement is not None else payload.metrics
    artifact_references = measurement.artifact_references if measurement is not None else payload.artifact_references
    direction = (
        "inconclusive" if measurement is not None and (
            measurement.execution_outcome == "inconclusive"
            or measurement.experiment_type == "task_model_training"
        ) else payload.evidence_direction
    )
    if payload.status == "completed":
        state = "scientifically_inconclusive" if direction == "inconclusive" else "usable"
    else:
        state = payload.status
    references, uncertainty, inputs, warnings, artifact_limits, model_details = _artifact_metadata(
        run_root,
        artifact_references,
        run_id=result.run_id,
        candidate_id=payload.candidate_id,
        task_id=payload.task_id,
    )
    if not uncertainty:
        uncertainty = [UncertaintyEstimate(
            metric="reported_metrics",
            available=False,
            reason_unavailable="No compatible uncertainty estimate was supplied by the upstream result artifact.",
        )]
    tests = []
    for name, value in sorted(metrics.items()):
        if name.endswith("p_value") and isinstance(value, (int, float)) and 0 <= float(value) <= 1:
            tests.append(StatisticalTestRecord(metric=name, raw_p_value=float(value)))
    task_sources = list(task.inputs.metadata_fields) if task else []
    dataset_reference = task.inputs.dataset_reference if task else None
    source_ids = sorted(set([
        *inputs,
        *task_sources,
        *(str(model_details[key]) for key in ("model_hash", "intervention_id", "ood_id") if key in model_details),
    ]))
    return EvidenceRecord(
        evidence_id=(
            f"EV-{payload.candidate_id}-{payload.task_id}-{measurement.measurement_id}"
            if measurement is not None else f"EV-{payload.candidate_id}-{payload.task_id}-QT"
        ),
        candidate_id=payload.candidate_id,
        task_id=payload.task_id,
        source_branch="quantitative",
        source_result_id=result.message_id,
        evidence_kind=(
            _MEASUREMENT_KINDS[measurement.experiment_type]
            if measurement is not None else _quantitative_kind(result, task)
        ),
        design_strength=(
            measurement.design_strength if measurement is not None
            else _Q_DESIGN[payload.evidence_level]
        ),
        execution_state=state,
        direction=direction,
        metrics={**dict(metrics), **model_details},
        uncertainty=uncertainty,
        statistical_tests=tests,
        coverage=_coverage_from_metrics(metrics, state=state),
        artifact_references=references,
        limitations=list(dict.fromkeys([*payload.limitations, *artifact_limits])),
        warnings=list(dict.fromkeys(warnings)),
        provenance=EvidenceProvenance(
            run_id=result.run_id,
            candidate_id=payload.candidate_id,
            task_id=payload.task_id,
            result_id=result.message_id,
            measurement_id=measurement.measurement_id if measurement is not None else None,
            plan_step_id=measurement.plan_step_id if measurement is not None else None,
            planning_check=measurement.planning_check if measurement is not None else None,
            experiment_type=measurement.experiment_type if measurement is not None else None,
            tool_call_id=measurement.tool_call_id if measurement is not None else None,
            backend_identity=measurement.backend_identity if measurement is not None else None,
            experiment_id=(
                str(model_details.get("ood_id") or model_details.get("intervention_id") or model_details.get("model_id"))
                if model_details else None
            ),
            dataset_reference=dataset_reference,
            input_family=f"{payload.candidate_id}:{','.join(source_ids) or payload.task_id}",
            shared_source_identifiers=source_ids,
        ),
    )


def _qual_direction(status: str) -> str:
    return {
        "supported": "supports",
        "partially_supported": "mixed",
        "contradicted": "contradicts",
        "inconclusive": "inconclusive",
        "insufficient_evidence": "not_tested",
    }[status]


def _qualitative_records(result, task: AnalysisTask | None, run_root: Path | None) -> list[EvidenceRecord]:
    payload = result.payload
    direction = _qual_direction(payload.evidence_status)
    records: list[EvidenceRecord] = []
    approved_artifacts, artifact_limitations = _validated_artifact_references(
        run_root,
        payload.artifact_references,
        run_id=result.run_id,
        candidate_id=payload.candidate_id,
        task_id=payload.task_id,
    )
    source_ids = sorted(set(task.inputs.metadata_fields if task else []))
    common = dict(
        candidate_id=payload.candidate_id,
        task_id=payload.task_id,
        source_branch="qualitative",
        source_result_id=result.message_id,
        design_strength="descriptive",
        statistical_tests=[],
        provenance=EvidenceProvenance(
            run_id=result.run_id,
            candidate_id=payload.candidate_id,
            task_id=payload.task_id,
            result_id=result.message_id,
            dataset_reference=task.inputs.dataset_reference if task else None,
            input_family=f"{payload.candidate_id}:{','.join(source_ids) or payload.task_id}",
            shared_source_identifiers=source_ids,
        ),
    )
    inspected = payload.sampling.sample_count
    base_coverage = EvidenceCoverage(
        requested=1,
        attempted=1,
        usable=1 if direction in {"supports", "contradicts", "mixed"} else 0,
        inconclusive=1 if direction in {"inconclusive", "not_tested"} else 0,
        sample_count=payload.sampling.sample_count,
        selected=payload.sampling.sample_count,
        inspected=inspected,
        failed=len(payload.failures),
    )
    for index, item in enumerate(payload.metadata_evidence, 1):
        records.append(EvidenceRecord(
            evidence_id=f"EV-{payload.candidate_id}-{payload.task_id}-QM{index:03d}",
            evidence_kind="qualitative_metadata",
            execution_state="usable" if direction not in {"inconclusive", "not_tested"} else "scientifically_inconclusive",
            direction="inconclusive",
            metrics={"field": item.field, "observation": item.observation, "sample_refs": item.sample_refs},
            uncertainty=[UncertaintyEstimate(metric="qualitative_interpretation", available=False, reason_unavailable="Qualitative evidence does not have a statistical confidence interval.")],
            coverage=base_coverage,
            artifact_references=approved_artifacts,
            limitations=list(dict.fromkeys([*payload.limitations, *artifact_limitations])),
            warnings=[],
            **common,
        ))
    for index, observation in enumerate(payload.observations, 1):
        kind = "qualitative_spatial" if "spatial" in observation.observation.casefold() else "qualitative_visual"
        records.append(EvidenceRecord(
            evidence_id=f"EV-{payload.candidate_id}-{payload.task_id}-QO{index:03d}",
            evidence_kind=kind,
            execution_state="usable" if observation.inspection_quality != "unusable" else "scientifically_inconclusive",
            direction="inconclusive",
            metrics={"group": observation.group, "observation": observation.observation, "sample_refs": observation.sample_refs, "inspection_quality": observation.inspection_quality},
            uncertainty=[UncertaintyEstimate(metric="qualitative_interpretation", available=False, reason_unavailable="Qualitative evidence does not have a statistical confidence interval.")],
            coverage=base_coverage,
            artifact_references=approved_artifacts,
            limitations=list(dict.fromkeys([*payload.limitations, *artifact_limitations])),
            warnings=[],
            **common,
        ))
    for index, comparison in enumerate(payload.comparisons, 1):
        usable = comparison.inspection_quality != "unusable"
        records.append(EvidenceRecord(
            evidence_id=f"EV-{payload.candidate_id}-{payload.task_id}-QG{index:03d}",
            evidence_kind="qualitative_group_comparison",
            execution_state=(
                "usable" if usable else "scientifically_inconclusive"
            ),
            direction=(
                "inconclusive"
            ),
            metrics={
                "groups": comparison.groups,
                "comparison": comparison.comparison,
                "sample_refs": comparison.sample_refs,
                "inspection_quality": comparison.inspection_quality,
            },
            uncertainty=[UncertaintyEstimate(
                metric="qualitative_group_comparison",
                available=False,
                reason_unavailable=(
                    "Qualitative evidence does not have a statistical "
                    "confidence interval."
                ),
            )],
            coverage=base_coverage,
            artifact_references=approved_artifacts,
            limitations=list(dict.fromkeys([
                *payload.limitations,
                *artifact_limitations,
            ])),
            warnings=[],
            **common,
        ))
    for index, counter in enumerate(payload.counterevidence, 1):
        records.append(EvidenceRecord(
            evidence_id=f"EV-{payload.candidate_id}-{payload.task_id}-QC{index:03d}",
            evidence_kind="qualitative_counterevidence",
            execution_state="usable",
            direction="contradicts",
            metrics={"description": counter.description, "sample_refs": counter.sample_refs},
            uncertainty=[UncertaintyEstimate(metric="qualitative_interpretation", available=False, reason_unavailable="Qualitative evidence does not have a statistical confidence interval.")],
            coverage=base_coverage,
            artifact_references=approved_artifacts,
            limitations=list(dict.fromkeys([*payload.limitations, *artifact_limitations])),
            warnings=[],
            **common,
        ))
    for index, failure in enumerate(payload.failures, 1):
        failure_artifacts, failure_artifact_limits = _validated_artifact_references(
            run_root,
            failure.artifact_references,
            run_id=result.run_id,
            candidate_id=payload.candidate_id,
            task_id=payload.task_id,
        )
        records.append(EvidenceRecord(
            evidence_id=f"EV-{payload.candidate_id}-{payload.task_id}-QF{index:03d}",
            evidence_kind="qualitative_failure",
            execution_state="failed",
            direction="not_tested",
            metrics={"category": failure.category, "sample_refs": failure.sample_refs},
            uncertainty=[UncertaintyEstimate(metric="qualitative_interpretation", available=False, reason_unavailable="Inspection failed; no uncertainty estimate exists.")],
            coverage=EvidenceCoverage(requested=1, attempted=1, failed=1),
            artifact_references=failure_artifacts,
            limitations=list(dict.fromkeys([*payload.limitations, failure.description, *failure_artifact_limits])),
            warnings=[],
            **common,
        ))
    if direction != "not_tested":
        records.append(EvidenceRecord(
            evidence_id=f"EV-{payload.candidate_id}-{payload.task_id}-QA",
            evidence_kind="qualitative_assessment",
            execution_state="usable" if direction in {"supports", "contradicts", "mixed"} else "scientifically_inconclusive",
            direction=direction,
            metrics={"evidence_summary": payload.evidence_summary},
            uncertainty=[UncertaintyEstimate(metric="qualitative_interpretation", available=False, reason_unavailable="Qualitative assessment has no statistical confidence interval.")],
            coverage=base_coverage,
            artifact_references=approved_artifacts,
            limitations=list(dict.fromkeys([*payload.limitations, *artifact_limitations])),
            warnings=[],
            **common,
        ))
    if not records:
        records.append(EvidenceRecord(
            evidence_id=f"EV-{payload.candidate_id}-{payload.task_id}-QS",
            evidence_kind="qualitative_visual",
            execution_state="blocked" if direction == "not_tested" else "scientifically_inconclusive",
            direction="not_tested" if direction == "not_tested" else "inconclusive",
            metrics={"evidence_summary": payload.evidence_summary},
            uncertainty=[UncertaintyEstimate(metric="qualitative_interpretation", available=False, reason_unavailable="No usable qualitative evidence was available.")],
            coverage=base_coverage,
            artifact_references=[],
            limitations=list(payload.limitations) or [payload.evidence_summary],
            warnings=[],
            **common,
        ))
    return records


def _apply_multiple_testing(records: list[EvidenceRecord], enabled: bool) -> list[EvidenceRecord]:
    located: list[tuple[int, int, float]] = []
    for record_index, record in enumerate(records):
        for test_index, test in enumerate(record.statistical_tests):
            located.append((record_index, test_index, float(test.raw_p_value)))
    if not enabled or not located:
        return records
    adjusted = benjamini_hochberg([item[2] for item in located])
    family_id = f"{records[0].candidate_id}:related_statistical_tests"
    size = len(located)
    mutable = [record.model_copy(deep=True) for record in records]
    for (record_index, test_index, _), value in zip(located, adjusted, strict=True):
        mutable[record_index].statistical_tests[test_index] = mutable[record_index].statistical_tests[test_index].model_copy(update={
            "adjusted_p_value": value,
            "correction_method": "benjamini_hochberg",
            "family_id": family_id,
            "family_size": size,
        })
    return mutable


def _profile(candidate_id: str, records: list[EvidenceRecord], planned: int) -> CandidateEvidenceProfile:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.provenance.input_family:
            groups[record.provenance.input_family].append(record.evidence_id)
    directions = [record.direction for record in records if record.execution_state == "usable"]
    has_support = any(value in {"supports", "mixed"} for value in directions)
    has_contradiction = any(value in {"contradicts", "mixed"} for value in directions)
    if has_support and has_contradiction:
        consistency = "mixed"
    elif has_support:
        consistency = "consistent_support"
    elif has_contradiction:
        consistency = "consistent_contradiction"
    elif any(record.execution_state == "scientifically_inconclusive" for record in records):
        consistency = "inconclusive"
    else:
        consistency = "no_usable_evidence"
    strengths = sorted({record.design_strength for record in records}, key=_DESIGN_ORDER.get)
    usable = sum(record.execution_state == "usable" for record in records)
    failed = sum(record.execution_state == "failed" for record in records)
    inconclusive = sum(record.execution_state == "scientifically_inconclusive" for record in records)
    return CandidateEvidenceProfile(
        candidate_id=candidate_id,
        evidence_kinds=sorted({record.evidence_kind for record in records}),
        design_strengths=strengths,
        max_design_strength=max(strengths, key=_DESIGN_ORDER.get) if strengths else None,
        consistency=consistency,
        coverage=EvidenceCoverage(requested=planned, attempted=sum(record.execution_state != "blocked" for record in records), usable=usable, failed=failed, inconclusive=inconclusive),
        uncertainty_available=any(item.available for record in records for item in record.uncertainty),
        supporting_record_ids=[record.evidence_id for record in records if record.direction == "supports"],
        contradicting_record_ids=[record.evidence_id for record in records if record.direction == "contradicts"],
        mixed_record_ids=[record.evidence_id for record in records if record.direction == "mixed"],
        inconclusive_record_ids=[record.evidence_id for record in records if record.direction == "inconclusive"],
        failed_record_ids=[record.evidence_id for record in records if record.execution_state == "failed"],
        blocked_record_ids=[record.evidence_id for record in records if record.execution_state == "blocked"],
        shared_source_groups={key: value for key, value in sorted(groups.items()) if len(value) > 1},
        limitations=list(dict.fromkeys(limit for record in records for limit in record.limitations)),
    )


class EvidenceLedgerBuilder:
    """Organize upstream evidence; never execute experiments or infer directions."""

    def __init__(self, *, apply_bh_fdr: bool = True) -> None:
        self.apply_bh_fdr = apply_bh_fdr

    def build(
        self,
        candidate_report: CandidateReport,
        routing_plan: RoutingPlan,
        quantitative_results: QuantitativeResultBatch,
        qualitative_results: QualitativeResultBatch,
        *,
        runs_root: str | Path | None = None,
    ) -> EvidenceLedger:
        run_id = candidate_report.run_id
        if any(item.run_id != run_id for item in (routing_plan, quantitative_results, qualitative_results)):
            raise ValueError("Evidence fan-in contains a foreign run")
        candidates = {item.candidate_id for item in candidate_report.candidates}
        if any(task.candidate_id not in candidates for task in routing_plan.tasks):
            raise ValueError("Evidence fan-in contains an unknown task candidate")
        for target, batch in (("quantitative", quantitative_results), ("qualitative", qualitative_results)):
            planned = [task for task in routing_plan.tasks if task.target == target]
            if [item.payload.task_id for item in batch.results] != [task.task_id for task in planned]:
                raise ValueError(f"Evidence fan-in {target} results do not exactly cover planned tasks")
            for task, result in zip(planned, batch.results, strict=True):
                if result.payload.candidate_id != task.candidate_id:
                    raise ValueError(f"Evidence fan-in {target} result has a foreign candidate")
                if result.in_reply_to != f"{run_id}:{target}:{task.task_id}:request":
                    raise ValueError(f"Evidence fan-in {target} result has invalid request correlation")
        fingerprint = _fingerprint(candidate_report, routing_plan, quantitative_results, qualitative_results)
        artifact_path = None
        run_root = None
        if runs_root is not None:
            run_root = (Path(runs_root) / candidate_report.run_id).resolve(strict=False)
            artifact_path = run_root / "evidence" / "ledger.json"
            if artifact_path.is_file():
                try:
                    prior = EvidenceLedger.model_validate_json(artifact_path.read_text(encoding="utf-8"))
                    if prior.fingerprint == fingerprint:
                        return prior
                except Exception:
                    pass
        tasks = {task.task_id: task for task in routing_plan.tasks}
        by_candidate: dict[str, list[EvidenceRecord]] = defaultdict(list)
        for result in quantitative_results.results:
            by_candidate[result.payload.candidate_id].extend(_quantitative_records(result, tasks.get(result.payload.task_id), run_root))
        for result in qualitative_results.results:
            by_candidate[result.payload.candidate_id].extend(_qualitative_records(result, tasks.get(result.payload.task_id), run_root))
        ledgers = []
        limitations = []
        for candidate in candidate_report.candidates:
            records = _apply_multiple_testing(by_candidate[candidate.candidate_id], self.apply_bh_fdr)
            planned = sum(task.candidate_id == candidate.candidate_id for task in routing_plan.tasks)
            if not records:
                limitations.append(f"Candidate {candidate.candidate_id} has no routed or returned evidence records.")
            ledgers.append(CandidateEvidenceLedger(
                candidate_id=candidate.candidate_id,
                record_ids=[record.evidence_id for record in records],
                records=records,
                profile=_profile(candidate.candidate_id, records, planned),
            ))
        ledger = EvidenceLedger(
            run_id=candidate_report.run_id,
            fingerprint=fingerprint,
            multiple_testing_method="benjamini_hochberg" if self.apply_bh_fdr else "none",
            test_family_rule="All formal p-values returned for one candidate form candidate_id:related_statistical_tests; no cross-candidate correction is applied.",
            candidates=ledgers,
            limitations=limitations,
        )
        if artifact_path is not None:
            atomic_write_json(artifact_path, ledger.model_dump(mode="json"), error_type=RuntimeError)
        return ledger


__all__ = ["EvidenceLedgerBuilder"]
