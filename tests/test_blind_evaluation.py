"""Blind evaluation firewall, generator, matching, metrics, and smoke tests."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from shortcutminer.evaluation.blind.batch import BlindEvaluationBatch
from shortcutminer.evaluation.blind.evaluator import BlindEvaluator, load_hidden_truth
from shortcutminer.evaluation.blind.generator import (
    BlindChallengeGenerator,
    hash_public_dataset,
)
from shortcutminer.evaluation.blind.leakage import (
    assert_path_isolation,
    assert_public_names_are_neutral,
    production_private_import_violations,
)
from shortcutminer.evaluation.blind.matching import (
    aggregate_adjudications,
    match_candidates,
)
from shortcutminer.evaluation.blind.protocol import (
    BlindChallengeConfig,
    ManualAdjudication,
)
from shortcutminer.evaluation.blind.snapshot import (
    PROTOCOL_FILES,
    SnapshotError,
    WORKSPACE_FILES,
    build_evaluation_snapshot,
)
from shortcutminer.evaluation.evidence_ledger import EvidenceLedgerBuilder
from shortcutminer.experiment.dataset_profiler import build_dataset_profile
from shortcutminer.protocol.batches import QualitativeResultBatch, QuantitativeResultBatch
from shortcutminer.protocol.executive import RoutingPlan
from shortcutminer.protocol.identify import CandidateReport, IdentifyRequest
from shortcutminer.protocol.qualitative import QualitativeResult
from shortcutminer.protocol.quantitative import QuantitativeResult
from shortcutminer.protocol.run import RunRequest
from shortcutminer.protocol.summarize import SummaryReport
from shortcutminer.runner import run
from shortcutminer.runtime.demo import DemoAgentClient
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.errors import ToolInputValidationError
from shortcutminer.tooling.runtime import ToolRuntime


FAMILIES = (
    "metadata_association",
    "spatial_visual",
    "acquisition_quality",
    "mixed",
    "clean_control",
)
FAMILY_IDS = dict(zip(FAMILIES, ("alpha", "bravo", "charlie", "delta", "echo"), strict=True))


def _generate(tmp_path: Path, family: str, suffix: str = ""):
    return BlindChallengeGenerator().generate(
        BlindChallengeConfig(
            challenge_id=f"challenge-{FAMILY_IDS[family]}{suffix}",
            family=family,
            challenge_generation_seed=73,
            sample_count=40,
            injection_strength="strong",
        ),
        tmp_path,
    )


def test_public_challenge_id_cannot_reveal_private_family():
    with pytest.raises(ValueError, match="challenge_id"):
        BlindChallengeConfig(
            challenge_id="challenge-spatial-shortcut",
            family="spatial_visual",
            challenge_generation_seed=1,
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_generator_supports_all_families_with_physical_isolation_and_neutral_public_names(
    tmp_path: Path, family: str
):
    generated = _generate(tmp_path, family)
    assert_path_isolation(
        challenge_root=generated.challenge_root,
        dataset_root=generated.public_dataset_root,
        hidden_truth_path=generated.hidden_truth_path,
    )
    assert_public_names_are_neutral(generated.public_dataset_root)
    assert generated.hidden_truth_path.parent.name == "private"
    assert not generated.hidden_truth_path.is_relative_to(generated.public_dataset_root)
    assert hash_public_dataset(generated.public_dataset_root) == generated.challenge.public.public_dataset_hash
    truth = load_hidden_truth(
        generated.hidden_truth_path,
        expected_hash=generated.challenge.private_truth_hash,
    )
    assert truth.shortcut_present is (family != "clean_control")
    if family == "clean_control":
        assert truth.clean_control is not None
        assert len(truth.clean_control.independent_variations) >= 3
    else:
        assert truth.shortcuts
    public_text = json.dumps(generated.challenge.public.model_dump(mode="json"))
    assert family not in public_text
    assert family.replace("_", "-") not in generated.challenge_root.name


@pytest.mark.parametrize("family", FAMILIES)
def test_generation_is_deterministic_for_same_seed_and_config(tmp_path: Path, family: str):
    first = _generate(tmp_path / "one", family)
    second = _generate(tmp_path / "two", family)
    assert first.challenge.public.public_dataset_hash == second.challenge.public.public_dataset_hash
    assert first.challenge.private_truth_hash == second.challenge.private_truth_hash
    assert first.hidden_truth_path.read_bytes() == second.hidden_truth_path.read_bytes()


def _controlled_outputs(generated, *, run_id: str):
    truth = load_hidden_truth(
        generated.hidden_truth_path,
        expected_hash=generated.challenge.private_truth_hash,
    )
    if truth.shortcuts:
        hidden = truth.shortcuts[0]
        cue_name = hidden.target_field or hidden.spatial_region or hidden.acquisition_family
        cue_family = hidden.family
        assessment = "supported"
    else:
        cue_name = "site"
        cue_family = "metadata"
        assessment = "inconclusive"
    candidate = {
        "candidate_id": "C001",
        "cue": {"family": cue_family, "name": cue_name},
        "description": f"Observed {cue_name} variation may be associated with the label.",
        "observation": f"The public evidence explicitly references {cue_name}.",
        "hypothesis": f"The neutral cue {cue_name} may be spuriously associated with label.",
        "evidence_sources": ["metadata" if cue_family == "metadata" else "sample_inspection"],
        "priority": "high",
        "status": "unverified",
    }
    report = CandidateReport.model_validate({
        "protocol_version": "1.0", "run_id": run_id,
        "message_id": f"{run_id}:identify:report", "source": "identify",
        "target": "executive", "message_type": "candidate_report",
        "in_reply_to": f"{run_id}:identify:request", "artifact_references": [],
        "dataset_summary": {"dataset": generated.challenge.challenge_id, "samples_inspected": 40, "metadata_used": True},
        "candidates": [candidate], "limitations": [],
    })
    common_inputs = {
        "dataset": generated.challenge.challenge_id,
        "dataset_reference": str(generated.public_dataset_root),
        "label_column": "label", "split": None,
        "metadata_fields": ["site", "device", "batch"],
        "artifact_references": [], "sample_limit": 8,
        "requires_sample_access": True, "requires_external_services": False,
        "runtime_constraints": ["no_external_services"],
        "required_model_capabilities": [],
    }
    tasks = []
    for suffix, target in (("QT", "quantitative"), ("QL", "qualitative")):
        tasks.append({
            "protocol_version": "1.0", "run_id": run_id,
            "message_id": f"{run_id}:{target}:task", "source": "executive",
            "target": target, "message_type": "analysis_task",
            "in_reply_to": f"{run_id}:executive:plan",
            "task_id": f"TASK-C001-{suffix}", "candidate_id": "C001",
            "hypothesis": candidate["hypothesis"], "objective": "Evaluate the public candidate.",
            "inputs": common_inputs, "requested_checks": ["public_evidence_check"],
            "expected_outputs": ["structured evidence"], "priority": "high",
            "dependencies": [], "status": "planned",
        })
    plan = RoutingPlan.model_validate({
        "protocol_version": "1.0", "run_id": run_id,
        "message_id": f"{run_id}:executive:plan", "source": "executive",
        "target": "orchestrator", "message_type": "routing_plan",
        "in_reply_to": report.message_id, "artifact_references": [],
        "decisions": [{"candidate": candidate, "route": "both", "routing_reason": "Both public evidence paths are available.", "task_ids": ["TASK-C001-QT", "TASK-C001-QL"], "priority": "high", "status": "planned", "missing_requirements": [], "limitations": []}],
        "tasks": tasks, "limitations": [],
    })
    quant = QuantitativeResult.model_validate({
        "protocol_version": "1.0", "run_id": run_id,
        "message_id": f"{run_id}:quantitative:result", "source": "quantitative",
        "target": "summarize", "message_type": "quantitative_analysis_result",
        "in_reply_to": f"{run_id}:quantitative:request",
        "payload": {"task_id": "TASK-C001-QT", "candidate_id": "C001", "hypothesis": candidate["hypothesis"], "status": "completed", "analysis_type": "controlled public fixture", "metrics": {"sample_count": 40}, "evidence_level": "Q2", "evidence_direction": "supports", "evidence_summary": "Controlled public fixture evidence.", "limitations": [], "artifact_references": [], "recommended_followup": []},
    })
    qual = QualitativeResult.model_validate({
        "protocol_version": "1.0", "run_id": run_id,
        "message_id": f"{run_id}:qualitative:result", "source": "qualitative",
        "target": "summarize", "message_type": "qualitative_analysis_result",
        "in_reply_to": f"{run_id}:qualitative:request",
        "payload": {"task_id": "TASK-C001-QL", "candidate_id": "C001", "candidate_name": cue_name, "hypothesis": candidate["hypothesis"], "evidence_status": "supported", "evidence_summary": "Controlled public fixture evidence.", "sampling": {"strategy": "random", "seed": 5, "groups": {"all": 1}, "sample_count": 1, "sample_ids": ["sample-0001"]}, "observations": [{"observation_id": "QO-C001-001", "group": "all", "sample_refs": ["sample-0001"], "observation": f"The public sample evidence references {cue_name}.", "inspection_quality": "sufficient"}], "comparisons": [], "interpretations": [{"interpretation": "Controlled harness fixture only.", "supporting_observation_ids": ["QO-C001-001"], "supporting_comparison_ids": [], "confidence": "low"}], "counterevidence": [], "metadata_evidence": [], "failures": [], "limitations": [], "artifact_references": [], "recommended_followup": []},
    })
    quant_batch = QuantitativeResultBatch(protocol_version="1.0", run_id=run_id, source="quantitative", target="summarize", message_type="quantitative_result_batch", results=[quant])
    qual_batch = QualitativeResultBatch(protocol_version="1.0", run_id=run_id, source="qualitative", target="summarize", message_type="qualitative_result_batch", results=[qual])
    ledger = EvidenceLedgerBuilder().build(report, plan, quant_batch, qual_batch)
    summary = SummaryReport.model_validate({
        "protocol_version": "1.0", "run_id": run_id,
        "message_id": f"{run_id}:summary:report", "source": "summarize",
        "target": "orchestrator", "message_type": "summary_report",
        "in_reply_to": f"{run_id}:summary:request",
        "payload": {"overview": "Controlled evaluation harness fixture.", "findings": [{"candidate_id": "C001", "assessment": assessment, "summary": "Controlled fixture result, not a performance claim.", "quantitative_result_ids": [quant.message_id], "qualitative_result_ids": [qual.message_id], "evidence_record_ids": ledger.candidates[0].record_ids, "limitations": [], "recommendations": []}], "limitations": [], "artifact_references": [], "evidence_ledger_reference": None},
    })
    return report, plan, quant_batch, qual_batch, ledger, summary


@pytest.mark.parametrize("family", FAMILIES)
def test_blind_evaluator_scientific_harness_smoke(tmp_path: Path, family: str):
    generated = _generate(tmp_path / "challenges", family)
    run_id = f"eval-{FAMILY_IDS[family]}"
    report, plan, quant, qual, ledger, summary = _controlled_outputs(generated, run_id=run_id)
    snapshot = build_evaluation_snapshot(
        repository_root=Path.cwd(), git_commit="a" * 40,
        runtime_config={"provider": "controlled-mock"},
        model_config=None, challenge_public_hash=generated.challenge.public.public_dataset_hash,
        runtime_sampling_seed=11, evaluation_bootstrap_seed=22,
        candidate_limit=5, validation_depth="standard", budget_settings={"max_samples": 40},
    )
    result = BlindEvaluator().evaluate(
        challenge=generated.challenge, challenge_root=generated.challenge_root,
        hidden_truth_path=generated.hidden_truth_path, snapshot=snapshot,
        run_id=run_id, candidate_report=report, routing_plan=plan,
        quantitative_results=quant, qualitative_results=qual,
        evidence_ledger=ledger, summary_report=summary,
        runs_root=tmp_path / "runs",
    )
    if family == "clean_control":
        assert result.discovery_metrics.truth_count == 0
        assert result.discovery_metrics.discovery_recall is None
        assert result.false_positive_metrics.false_positive_candidate_count == 1
        assert result.false_positive_metrics.false_positive_findings_count == 0
    else:
        assert result.discovery_metrics.truth_count == 1
        assert result.discovery_metrics.matched_truth_count == 1
        assert result.localization_metrics.exact_count == 1
        assert result.funnel_metrics.discovered_count == 1
        assert result.funnel_metrics.evidence_usable_count == 1
    persisted = json.loads((tmp_path / "runs" / run_id / "evaluation" / "blind_result.json").read_text("utf-8"))
    assert (tmp_path / "runs" / run_id / "evaluation" / "snapshots" / f"{snapshot.snapshot_id}.json").is_file()
    assert "injection_mechanism" not in json.dumps(persisted)
    assert "affected_sample_ids" not in json.dumps(persisted)


def test_exact_matching_avoids_fuzzy_guessing_and_supports_manual_adjudication(tmp_path: Path):
    generated = _generate(tmp_path, "metadata_association")
    report, _, _, _, ledger, _ = _controlled_outputs(generated, run_id="match-run")
    truth = load_hidden_truth(generated.hidden_truth_path, expected_hash=generated.challenge.private_truth_hash)
    matched = match_candidates(report, truth, evidence_ledger=ledger)
    assert matched[0].match_status == "matched"
    data = report.model_dump(mode="json")
    data["candidates"][0]["cue"]["name"] = "capture pattern"
    data["candidates"][0]["description"] = "A broad capture pattern is visible."
    data["candidates"][0]["observation"] = "Capture groups differ."
    data["candidates"][0]["hypothesis"] = "A capture pattern may be associated."
    ambiguous_report = CandidateReport.model_validate(data)
    ambiguous = match_candidates(ambiguous_report, truth)
    assert ambiguous[0].match_status == "ambiguous"
    manual = ManualAdjudication(reviewer_id="reviewer-1", candidate_id="C001", truth_id="T001", decision="partial", localization="partial", rationale="Reviewer found a semantic but not exact localization.")
    adjudicated = match_candidates(ambiguous_report, truth, adjudications=[manual])
    assert adjudicated[0].match_status == "partial"
    assert adjudicated[0].adjudication_source == "manual"


def test_evaluator_distinguishes_wrong_localization_from_non_discovery(tmp_path: Path):
    generated = _generate(tmp_path, "metadata_association")
    outputs = _controlled_outputs(generated, run_id="wrong-localization")
    report_data = outputs[0].model_dump(mode="json")
    report_data["candidates"][0].update({
        "cue": {"family": "metadata", "name": "capture_pattern"},
        "description": "A broad capture pattern is visible.",
        "observation": "Capture groups differ.",
        "hypothesis": "A capture pattern may be associated.",
    })
    report = CandidateReport.model_validate(report_data)
    snapshot = build_evaluation_snapshot(
        repository_root=Path.cwd(), git_commit="a" * 40,
        runtime_config={"provider": "controlled-mock"}, model_config=None,
        challenge_public_hash=generated.challenge.public.public_dataset_hash,
        runtime_sampling_seed=11, evaluation_bootstrap_seed=22,
        candidate_limit=5, validation_depth="standard",
        budget_settings={"max_samples": 40},
    )
    result = BlindEvaluator().evaluate(
        challenge=generated.challenge, challenge_root=generated.challenge_root,
        hidden_truth_path=generated.hidden_truth_path, snapshot=snapshot,
        run_id="wrong-localization", candidate_report=report,
        routing_plan=outputs[1], quantitative_results=outputs[2],
        qualitative_results=outputs[3], evidence_ledger=outputs[4],
        summary_report=outputs[5],
    )
    assert result.failures[0].reason == "discovered_wrong_localization"


def test_multi_rater_aggregation_and_kappa_interface_is_deterministic():
    items = [
        ManualAdjudication(reviewer_id="r1", candidate_id="C001", truth_id="T001", decision="matched", localization="exact", rationale="exact"),
        ManualAdjudication(reviewer_id="r2", candidate_id="C001", truth_id="T001", decision="matched", localization="exact", rationale="exact"),
        ManualAdjudication(reviewer_id="r1", candidate_id="C002", truth_id="T002", decision="not_matched", localization="incorrect", rationale="no"),
        ManualAdjudication(reviewer_id="r2", candidate_id="C002", truth_id="T002", decision="matched", localization="exact", rationale="yes"),
    ]
    first = aggregate_adjudications(items)
    assert first == aggregate_adjudications(items)
    assert first.raw_agreement == 0.5
    assert first.cohens_kappa == pytest.approx(0.0)


def _copy_snapshot_inputs(source: Path, destination: Path) -> None:
    (destination / ".git" / "refs" / "heads").mkdir(parents=True)
    (destination / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (destination / ".git" / "refs" / "heads" / "main").write_text("a" * 40 + "\n", encoding="utf-8")
    for agent in ("identify", "executive", "quantitative", "qualitative", "summarize"):
        for filename in WORKSPACE_FILES:
            relative = Path("agents") / agent / "workspace" / filename
            (destination / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, destination / relative)
    for relative_text in PROTOCOL_FILES:
        relative = Path(relative_text)
        (destination / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination / relative)


def test_snapshot_changes_with_workspace_or_runtime_but_not_private_truth(tmp_path: Path):
    repo = tmp_path / "repo"
    _copy_snapshot_inputs(Path.cwd(), repo)
    kwargs = dict(
        repository_root=repo, runtime_config={"temperature": 0, "path": r"C:\private\run"},
        model_config={"model": "mock", "api_key": "must-not-appear"},
        model_runtime_snapshot={
            "snapshot_id": "model-runtime-a",
            "model": "mock",
            "api_key_env": "SHORTCUTMINER_TEST_KEY",
        },
        challenge_public_hash="b" * 64, runtime_sampling_seed=10,
        evaluation_bootstrap_seed=20, candidate_limit=5,
        validation_depth="standard", budget_settings={"max_samples": 40},
    )
    first = build_evaluation_snapshot(**kwargs)
    same_with_different_private_truth = build_evaluation_snapshot(**kwargs)
    assert first.snapshot_id == same_with_different_private_truth.snapshot_id
    assert first.model_runtime_snapshot_id == "model-runtime-a"
    assert first.model_runtime_snapshot_hash is not None
    assert first.sanitized_model_config == {"model": "mock"}
    assert "must-not-appear" not in json.dumps(first.model_dump(mode="json"))
    workspace = repo / "agents" / "identify" / "workspace" / "AGENTS.md"
    workspace.write_text(workspace.read_text("utf-8") + "\nfreeze-change\n", encoding="utf-8")
    assert build_evaluation_snapshot(**kwargs).snapshot_id != first.snapshot_id
    workspace.write_bytes((Path.cwd() / "agents" / "identify" / "workspace" / "AGENTS.md").read_bytes())
    changed = dict(kwargs)
    changed["runtime_config"] = {"temperature": 1}
    assert build_evaluation_snapshot(**changed).snapshot_id != first.snapshot_id
    changed_model_runtime = dict(kwargs)
    changed_model_runtime["model_runtime_snapshot"] = {
        "snapshot_id": "model-runtime-b", "model": "other"
    }
    assert build_evaluation_snapshot(**changed_model_runtime).snapshot_id != first.snapshot_id
    with pytest.raises(SnapshotError, match="git_commit"):
        build_evaluation_snapshot(**kwargs, git_commit="unknown")


def _blind_run_request(generated, runs_root: Path) -> RunRequest:
    return RunRequest(
        run_id="blind-runtime-hash", runs_root=str(runs_root),
        dataset_info={"name": "challenge_random_name_8472", "root": str(generated.public_dataset_root), "metadata_file": "metadata.csv"},
        task_input=generated.challenge.public.task,
        seed=generated.challenge.public.runtime_sampling_seed,
        identify_config={"budget": {"max_samples": 12, "max_candidates": 1, "max_dataset_entries": 100, "max_metadata_rows": 100}},
        executive_config={"audit_options": {"validation_depth": "standard", "max_tasks": 2, "max_samples_per_task": 8, "allowed_targets": ["quantitative", "qualitative"], "runtime_constraints": ["no_external_services"]}, "capabilities": {"available_targets": ["quantitative", "qualitative"], "available_metadata_fields": ["id", "label", "site", "device", "batch", "measure", "path"], "quantitative_checks": ["association_test"], "qualitative_checks": ["metadata_inspection"], "sample_access": False, "external_services": False, "available_output_types": ["metrics", "observations"]}},
        qualitative_config={"preparation": {"seed": generated.challenge.public.runtime_sampling_seed, "sample_id_column": "id", "group_columns": ["label"], "max_evidence_records": 4, "max_metadata_rows": 100}},
        metadata={"evaluation_mode": "blind_public_only"},
    )


def _runtime_output_hash(execution) -> str:
    state = execution.workflow_state
    output = {
        key: state[key] for key in (
            "identify_result", "executive_result", "quantitative_result",
            "qualitative_result", "evidence_ledger", "summarize_result",
        )
    }
    return hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_truth_present_removed_and_replaced_have_identical_complete_runtime_outputs(tmp_path: Path):
    generated = _generate(tmp_path / "challenge", "metadata_association")
    identify_request = IdentifyRequest(
        protocol_version="1.0", run_id="profile-firewall", message_id="profile-request",
        source="orchestrator", target="identify", message_type="identify_request",
        dataset={"name": "challenge_random_name_8472", "root": str(generated.public_dataset_root), "metadata_file": "metadata.csv"},
        task=generated.challenge.public.task,
    )
    profile = build_dataset_profile(identify_request, seed=9)
    assert "hidden_truth" not in json.dumps(profile.model_dump(mode="json"))
    truth_bytes = generated.hidden_truth_path.read_bytes()

    clients = []
    executions = []
    for label in ("present", "removed", "replaced"):
        if label == "removed":
            generated.hidden_truth_path.unlink()
        elif label == "replaced":
            generated.hidden_truth_path.write_text('{"unrelated_private_marker":"different"}\n', encoding="utf-8")
        client = DemoAgentClient()
        executions.append(run(_blind_run_request(generated, tmp_path / f"runs-{label}"), agent_client=client))
        clients.append(client)
    generated.hidden_truth_path.write_bytes(truth_bytes)

    hashes = [_runtime_output_hash(item) for item in executions]
    assert hashes[0] == hashes[1] == hashes[2]
    for label, client, execution in zip(("present", "removed", "replaced"), clients, executions, strict=True):
        payload_text = json.dumps(client.calls, default=str)
        state_text = json.dumps(execution.workflow_state)
        persisted_text = "\n".join(
            path.read_text(encoding="utf-8")
            for pattern in ("*.json", "*.jsonl")
            for path in sorted((tmp_path / f"runs-{label}").rglob(pattern))
        )
        for forbidden in ("private_truth_hash", "shortcut_present", "injection_mechanism", "T001"):
            assert forbidden not in payload_text
            assert forbidden not in state_text
            assert forbidden not in persisted_text
        assert "hidden_truth" not in json.dumps(execution.final_result)
        assert "hidden_truth" not in json.dumps(execution.manifest.model_dump(mode="json"))


def test_visual_profile_and_runtime_inputs_ignore_private_truth(tmp_path: Path):
    generated = _generate(tmp_path / "visual-firewall", "spatial_visual")
    truth_bytes = generated.hidden_truth_path.read_bytes()
    fingerprints = []
    for label in ("present", "removed", "replaced"):
        if label == "removed":
            generated.hidden_truth_path.unlink()
        elif label == "replaced":
            generated.hidden_truth_path.write_text(
                '{"unrelated_private_marker":"different"}\n', encoding="utf-8"
            )
        request = _blind_run_request(generated, tmp_path / f"visual-runs-{label}")
        request.identify_config["visual_discovery"] = {
            "sample_id_column": "id",
            "sample_path_column": "path",
            "max_samples": 12,
        }
        client = DemoAgentClient()
        execution = run(request, agent_client=client)
        state = execution.workflow_state
        visual = state["identify_context"]["visual_discovery"]
        assert visual["coverage"]["decoded_samples"] > 0
        identify_call = next(
            call for call in client.calls if call["agent_name"] == "identify"
        )
        executive_call = next(
            call for call in client.calls if call["agent_name"] == "executive"
        )
        fingerprints.append(
            tuple(
                hashlib.sha256(
                    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                for value in (
                    visual,
                    identify_call["payload"],
                    executive_call["payload"]["request"]["candidate_report"],
                )
            )
        )
    generated.hidden_truth_path.write_bytes(truth_bytes)
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_tools_and_production_imports_cannot_cross_private_boundary(tmp_path: Path):
    generated = _generate(tmp_path, "metadata_association")
    assert production_private_import_violations(Path.cwd()) == []
    registry = build_default_tool_registry()
    sample_runtime = ToolRuntime(
        registry, run_id="firewall-run", agent_name="qualitative",
        dataset_root=generated.public_dataset_root,
        approved_samples={"../private/hidden_truth.json"},
    )
    with pytest.raises(ToolInputValidationError):
        sample_runtime.execute({"protocol_version": "1.0", "tool_call_id": "private-sample", "agent_name": "qualitative", "run_id": "firewall-run", "tool_name": "inspect_sample_metadata", "candidate_id": "C001", "task_id": "TASK-C001-QL", "arguments": {"sample_id": "private", "sample_reference": "../private/hidden_truth.json"}})
    artifact_runtime = ToolRuntime(
        registry, run_id="firewall-run", agent_name="summarize",
        runs_root=tmp_path, approved_artifacts={"../private/hidden_truth.json"},
    )
    with pytest.raises(ToolInputValidationError):
        artifact_runtime.execute({"protocol_version": "1.0", "tool_call_id": "private-artifact", "agent_name": "summarize", "run_id": "firewall-run", "tool_name": "read_artifact", "candidate_id": "C001", "task_id": "TASK-C001-QL", "arguments": {"artifact": "../private/hidden_truth.json"}})


def test_batch_aggregation_is_stratified_and_bootstraps_only_multiple_challenges(tmp_path: Path):
    results = []
    snapshot_id = None
    for index, family in enumerate(("metadata_association", "spatial_visual", "clean_control", "clean_control")):
        generated = _generate(tmp_path / f"c{index}", family, suffix=f"-{index}")
        run_id = f"batch-{index}"
        outputs = _controlled_outputs(generated, run_id=run_id)
        snapshot = build_evaluation_snapshot(
            repository_root=Path.cwd(), git_commit="a" * 40,
            runtime_config={"frozen": True}, model_config=None,
            challenge_public_hash=generated.challenge.public.public_dataset_hash,
            runtime_sampling_seed=9, evaluation_bootstrap_seed=19,
            candidate_limit=5, validation_depth="standard", budget_settings={"max_samples": 40},
        )
        if snapshot_id is None:
            snapshot_id = snapshot.snapshot_id
        else:
            assert snapshot.snapshot_id == snapshot_id
        results.append(BlindEvaluator().evaluate(
            challenge=generated.challenge, challenge_root=generated.challenge_root,
            hidden_truth_path=generated.hidden_truth_path, snapshot=snapshot,
            run_id=run_id, candidate_report=outputs[0], routing_plan=outputs[1],
            quantitative_results=outputs[2], qualitative_results=outputs[3],
            evidence_ledger=outputs[4], summary_report=outputs[5],
        ))
    batch = BlindEvaluationBatch(snapshot_id=snapshot_id, evaluation_bootstrap_seed=19, bootstrap_iterations=50).aggregate(results)
    assert batch.number_positive == 2
    assert batch.number_clean == 2
    assert batch.positive.discovery_recall == 1.0
    assert batch.clean.discovery_recall is None
    assert all(item.available for item in batch.uncertainty)
