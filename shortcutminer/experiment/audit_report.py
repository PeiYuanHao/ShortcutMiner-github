"""Deterministic export of completed audit authority, without executing work.

Only fixed canonical artifacts and recorded telemetry references are read. No
dataset contents, environment variables, provider bodies or evaluator files are
read. Missing historical information stays unavailable, including git state.
"""

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath

from shortcutminer.experiment.audit_capabilities import AuditCapabilityContext, PRIVATE_FIELDS
from shortcutminer.protocol.audit import AuditRequest
from shortcutminer.protocol.audit_report import AuditReport
from shortcutminer.protocol.batches import QuantitativeResultBatch, QualitativeResultBatch
from shortcutminer.protocol.dataset_profile import DatasetProfile
from shortcutminer.protocol.evidence import EvidenceLedger
from shortcutminer.protocol.executive import RoutingPlan
from shortcutminer.protocol.identify import CandidateReport
from shortcutminer.protocol.model_runtime import ModelInvocationRecord, ModelRuntimeSnapshot, RunModelTelemetry
from shortcutminer.protocol.qualitative import QualitativeRequest
from shortcutminer.protocol.quantitative_plan import QuantitativeExperimentPlan
from shortcutminer.protocol.screening import CandidateScreeningReport
from shortcutminer.protocol.promotion import PromotionReport, ValidationStagePlan
from shortcutminer.protocol.summarize import SummaryReport
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.storage.paths import get_run_dir
from shortcutminer.storage.run_store import load_run_record, load_run_manifest, load_workflow_state, load_final_result
from shortcutminer.tooling.protocol import ToolResult


SCREENING_EXPLANATION = (
    "Screened-out candidates were not experimentally disproved; they were terminated "
    "before expensive validation under the configured screening policy."
)
_PRIVATE = re.compile(
    r"hidden[ _-]?truth|private[ _-]?truth|ground[ _-]?truth|expected_answer|truth_family|shortcut_truth|"
    r"known_shortcut|unknown_shortcut", re.I)
_SECRET = re.compile(
    r"authorization|bearer\s|api[ _-]?key|secret[ _-]?(?:env|value|token)|"
    r"\bsk-[\w-]+|raw[ _-]?provider|raw[ _-]?response|raw[ _-]?reasoning|"
    r"reasoning[ _-]?(?:content|text)|data:image|base64|<think>|password|access_token", re.I)
_BLOB = re.compile(r"[A-Za-z0-9+/]{256,}={0,2}")
_SENSITIVE_KEY = re.compile(r"secret|credential|password|raw_payload|provider_body|reasoning_response|chain_of_thought|image_bytes", re.I)


class AuditReportError(RuntimeError):
    """A canonical completed audit cannot be exported safely."""


def _public(value):
    """Defense in depth for untrusted free text and extensible metrics maps.

    Benign scientific text is copied exactly. Sensitive text is withheld rather
    than interpreted; secret variables are never inspected to find their values.
    """
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items()
                if not _PRIVATE.search(key) and not _SECRET.search(key) and not _SENSITIVE_KEY.search(key)}
    if isinstance(value, list):
        return [_public(item) for item in value]
    if isinstance(value, str):
        if _PRIVATE.search(value) or _SECRET.search(value) or _BLOB.search(value):
            return "[redacted]"
        return value
    return value


def _relative(root, value):
    if not isinstance(value, str) or _public(value) != value:
        return None
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in value or not path.parts:
        return None
    if not (root / str(path)).resolve().is_relative_to(root.resolve()):
        return None
    # References must not index private evaluators or raw payload stores.
    if any(word in str(path).lower() for word in ("raw_provider", "raw_response", "raw_reasoning", "ground_truth", "hidden_truth")):
        return None
    return str(path)


def _read(root, relative):
    safe = _relative(root, relative)
    if safe is None:
        raise AuditReportError("Unsafe audit artifact reference")
    return json.loads((root / safe).read_text(encoding="utf-8"))


def _authority(root, state, key, relative, model):
    value = _read(root, relative) if (root / relative).is_file() else state.get(key)
    return model.model_validate(value).model_dump(mode="json")


def _elapsed(start, end):
    if start is None or end is None:
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def _events(root):
    path = root / "events.jsonl"
    if not path.is_file():
        return None
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # An incomplete presentation event does not change authority.
        if isinstance(event, dict):
            result.append(event)
    return result


def _timings(events):
    started, result = {}, []
    for event in events or []:
        kind = event.get("event", "")
        stage = event.get("stage")
        if stage in {"audit_report", "run"}:
            continue  # Exporting itself must not alter repeated report builds.
        details = event.get("details", {})
        identity = (stage, details.get("task_id"))
        if kind in {"STAGE_STARTED", "TASK_STARTED"}:
            started[identity] = event.get("timestamp")
        elif kind in {"STAGE_COMPLETED", "STAGE_SKIPPED", "STAGE_FAILED", "TASK_COMPLETED", "TASK_FAILED"}:
            result.append({"stage": stage, "task_id": identity[1], "status": kind.lower(),
                           "started_at": started.pop(identity, None), "completed_at": event.get("timestamp")})
            result[-1]["elapsed_seconds"] = _elapsed(result[-1]["started_at"], result[-1]["completed_at"])
    return result


def _runtime(root, run_id):
    if not (root / "model_calls/telemetry.json").is_file():
        return {"snapshots": [], "invocations": [], "telemetry": None}
    telemetry = RunModelTelemetry.model_validate(_read(root, "model_calls/telemetry.json"))
    if telemetry.run_id != run_id:
        raise AuditReportError("Telemetry belongs to another run")
    records, snapshots = [], []
    for reference in telemetry.invocation_references:
        if not re.fullmatch(r"model_calls/[A-Za-z0-9._-]+\.json", reference):
            raise AuditReportError("Invocation reference is outside canonical model calls")
        record = ModelInvocationRecord.model_validate(_read(root, reference))
        if record.run_id != run_id or reference != f"model_calls/{record.invocation_id}.json":
            raise AuditReportError("Invocation belongs to another run")
        records.append(record.model_dump(mode="json", include={
            "invocation_id", "agent_name", "runtime_kind", "provider", "model", "attempt", "request_hash",
            "started_at", "completed_at", "latency_ms", "status", "usage", "transport_retry", "schema_retry"}))
        records[-1]["image_count"] = len(record.image_inputs)
    for identity in telemetry.runtime_snapshot_ids:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", identity):
            raise AuditReportError("Unsafe runtime snapshot identity")
        snapshot = ModelRuntimeSnapshot.model_validate(_read(root, f"model_calls/runtime_snapshot-{identity}.json"))
        snapshots.append(snapshot.model_dump(mode="json", exclude={"api_key_env", "raw_response_persistence"}))
    return {"snapshots": snapshots, "invocations": records, "telemetry": telemetry.model_dump(mode="json")}


def build_audit_report(runs_root: str | Path, run_id: str) -> AuditReport:
    """Read a completed audit's persisted authority; no calls or mutations."""
    root = get_run_dir(runs_root, run_id)
    record = load_run_record(runs_root, run_id)
    manifest = load_run_manifest(runs_root, run_id)
    if record.status != "COMPLETED" or manifest.metadata.get("entrypoint") != "audit":
        raise AuditReportError("Report export requires a completed dataset audit")
    state = load_workflow_state(runs_root, run_id)
    audit = AuditRequest.model_validate(manifest.metadata.get("audit_request")).model_dump(mode="json")
    profile = _authority(root, state, "dataset_profile", "dataset/profile.json", DatasetProfile)
    identify = _authority(root, state, "identify_result", "identify/output.json", CandidateReport)
    screening = _authority(root, state, "screening_result", "screening/result.json", CandidateScreeningReport)
    capabilities = _authority(root, state, "audit_capabilities", "capabilities/result.json", AuditCapabilityContext)
    routing = _authority(root, state, "executive_result", "executive/routing.json", RoutingPlan)
    ledger = _authority(root, state, "evidence_ledger", "evidence/ledger.json", EvidenceLedger)
    summary = SummaryReport.model_validate(load_final_result(runs_root, run_id)).model_dump(mode="json")
    qt = QuantitativeResultBatch.model_validate(state["quantitative_result"]).model_dump(mode="json")["results"]
    ql = QualitativeResultBatch.model_validate(state["qualitative_result"]).model_dump(mode="json")["results"]
    for authority in (identify, screening, capabilities, routing, ledger, summary, *qt, *ql):
        if authority["run_id"] != run_id:
            raise AuditReportError("Canonical artifact belongs to another run")
    runtime = _runtime(root, run_id)
    promotion = None
    if state.get("promotion_result") is not None:
        promoted_report = _authority(root, state, "promotion_result", "promotion/result.json", PromotionReport)
        stage_plan = _authority(root, state, "validation_stage_plan", "validation/stage_plan.json", ValidationStagePlan)
        if promoted_report["run_id"] != run_id or stage_plan["run_id"] != run_id or promoted_report["promotion_policy_hash"] != stage_plan["policy_hash"]:
            raise AuditReportError("Promotion authority does not match its frozen plan")
        promotion = {"report": promoted_report, "policy": stage_plan["policy"], "stage_plan": stage_plan,
                     "stage1_quantitative": QuantitativeResultBatch.model_validate(state["stage1_quantitative_result"]).model_dump(mode="json")["results"],
                     "stage1_qualitative": QualitativeResultBatch.model_validate(state["stage1_qualitative_result"]).model_dump(mode="json")["results"],
                     "stage2": state["stage2_validation"],
                     "explanation": "Promotion decisions control validation cost; terminated_low_signal is not experimental contradiction or proof of absence."}
    events = _events(root)
    elapsed = _elapsed(record.started_at, record.completed_at)
    telemetry = runtime["telemetry"]
    invocations = runtime["invocations"]
    tools = []
    for event in events or []:
        if event.get("event") != "TOOL_STARTED":
            continue
        details = event.get("details", {})
        tools.append({key: details.get(key) for key in (
            "tool_call_id", "tool", "candidate_id", "task_id", "plan_step_id", "planning_check", "experiment_type")})
        # Record actual scientific parameters, not arbitrary tool arguments/bodies.
        params = details.get("arguments", {}).get("parameters", {})
        tools[-1]["parameters"] = {key: params[key] for key in (
            "seed", "split_seed", "model_seed", "confidence_level", "bootstrap_iterations", "max_rows",
            "max_categories", "min_group_size", "n_permutations", "variable_mode", "cue_field", "label_column",
            "metadata_file", "epochs", "batch_size", "learning_rate", "test_size") if key in params}
        tools[-1]["timeout_seconds"] = None
        agent = event.get("stage", "").removeprefix("tool:")
        call_id = details.get("tool_call_id", "")
        if agent in {"identify", "executive", "quantitative", "qualitative", "summarize"} and re.fullmatch(r"[A-Za-z0-9._-]{1,128}", call_id):
            path = f"tools/{agent}/{call_id}.json"
            if (root / path).is_file():
                result = ToolResult.model_validate(_read(root, path))
                if result.run_id != run_id or result.tool_call_id != call_id:
                    raise AuditReportError("Tool header belongs to another run")
                tools[-1].update(status=result.status, duration_ms=result.duration_ms,
                                 timeout_seconds=result.metadata.get("timeout_seconds"))
    plans = {task_id: QuantitativeExperimentPlan.model_validate(value).model_dump(mode="json")
             for task_id, value in state.get("quantitative_experiment_plans", {}).items()}
    qualitative_preparation = {}
    for task in routing["tasks"]:
        if task["target"] != "qualitative":
            continue
        path = f"qualitative/{task['task_id']}/input.json"
        if (root / path).is_file():
            preparation_request = QualitativeRequest.model_validate(_read(root, path)["request"])
            if preparation_request.run_id != run_id or preparation_request.task.task_id != task["task_id"]:
                raise AuditReportError("Qualitative preparation belongs to another task")
            qualitative_preparation[task["task_id"]] = preparation_request.preparation.model_dump(mode="json")
    cost = {"text_provider_calls": sum(r["runtime_kind"] == "text_agent" for r in invocations) if telemetry else None,
            "vlm_calls": sum(r["runtime_kind"] == "semantic_vlm" for r in invocations) if telemetry else None,
            "vlm_image_count": sum(r["image_count"] for r in invocations if r["runtime_kind"] == "semantic_vlm") if telemetry else None,
            "tool_calls": len(tools) if events is not None else None,
            "prompt_tokens": telemetry["prompt_tokens"] if telemetry else None,
            "completion_tokens": telemetry["completion_tokens"] if telemetry else None,
            "total_tokens": telemetry["total_tokens"] if telemetry else None,
            "transport_retries": telemetry["retry_calls"] if telemetry else None,
            "schema_retries": telemetry["schema_retry_calls"] if telemetry else None,
            "retries": sum(r["transport_retry"] or r["schema_retry"] for r in invocations) if telemetry else None,
            "elapsed_seconds": elapsed, "timings": _timings(events)}
    # Deduplicate recorded limitations, preserving every source location.
    limitations = {}
    def collect(value, source):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "limitations" and isinstance(item, list):
                    for text in item:
                        if isinstance(text, str):
                            limitations.setdefault(text, set()).add(source)
                else:
                    collect(item, f"{source}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                collect(item, f"{source}[{index}]")
    for name, value in (("audit_intake", audit), ("dataset_profile", profile), ("identify", identify),
                        ("capabilities", capabilities["capabilities"]), ("routing", routing), ("quantitative", qt),
                        ("qualitative", ql), ("evidence", ledger), ("summary", summary), ("telemetry", telemetry), ("evidence_promotion", promotion)):
        collect(value, name)
    references = set()
    def refs(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"artifact_references", "invocation_references"} and isinstance(item, list):
                    references.update(path for ref in item if (path := _relative(root, ref)))
                elif key in {"artifact_reference", "evidence_ledger_reference"}:
                    if path := _relative(root, item):
                        references.add(path)
                else:
                    refs(item)
        elif isinstance(value, list):
            for item in value:
                refs(item)
    for value in (identify, routing, qt, ql, ledger, summary, telemetry):
        refs(value)
    if promotion:
        refs(promotion)
        for task in routing["tasks"]:
            path = f"validation/stage1/{task['task_id']}-{task['target']}.json"
            if (root / path).is_file() and _relative(root, path):
                references.add(path)
    for path in ("audit_request.json", "dataset/profile.json", "identify/output.json", "screening/result.json",
                 "capabilities/result.json", "executive/routing.json", "evidence/ledger.json", "final_result.json",
                 "promotion/policy.json", "promotion/result.json", "validation/stage_plan.json"):
        if (root / path).is_file() and _relative(root, path):
            references.add(path)
    candidates = [{**candidate, "identify_confidence": candidate.get("confidence")} for candidate in identify["candidates"]]
    intake_info = audit.get("dataset_info") or {}
    public_fields = [field for field in intake_info.get("metadata_columns", []) if field not in PRIVATE_FIELDS]
    # Profile examples and raw metadata rows are deliberately not copied.
    dataset = {"name": profile["dataset"]["name"], "input_path": audit["dataset_path"],
               "description": audit.get("dataset_description"), "dataset_type": intake_info.get("dataset_type"),
               "label_information": audit.get("label_information"),
               "modality": profile["structure"]["modality"],
               "metadata_file": intake_info.get("selected_metadata_file"),
               "metadata_files": intake_info.get("metadata_files", []),
               "label_source": (audit.get("dataset_info") or {}).get("label_source"),
               "sample_count": profile["metadata"]["row_count"] if not profile["metadata"]["truncated"] else None,
               "sample_count_basis": "complete_metadata_rows" if profile["metadata"]["available"] and not profile["metadata"]["truncated"] else None,
               "rows_profiled": profile["metadata"]["rows_profiled"],
               "metadata_truncated": profile["metadata"]["truncated"], "structure": profile["structure"],
               "splits": profile["splits"], "public_metadata_fields": public_fields,
               "validation_metadata_fields": capabilities["capabilities"]["available_metadata_fields"],
               "fingerprint": None, "intake_limitations": (audit.get("dataset_info") or {}).get("limitations", [])}
    report = {"run": {"run_id": run_id, "mode": "audit", "status": record.status,
                      "started_at": record.started_at, "completed_at": record.completed_at, "elapsed_seconds": elapsed,
                      "git_commit": None, "repository_dirty": None},
              "dataset": dataset, "runtime": runtime,
              "discovery": {"candidates_generated": len(candidates), "candidates": candidates},
              "screening": {**screening, "explanation": SCREENING_EXPLANATION},
              "validation": {"capabilities": capabilities["capabilities"], "routing": routing,
                             "plans": plans,
                             "feasibility": state.get("quantitative_execution_requirements", {}), "tools": tools},
              "quantitative": qt, "qualitative": ql, "evidence": ledger, "findings": summary["payload"],
              "evidence_promotion": promotion,
              "cost": cost,
              "reproducibility": {"shortcutminer_version": None, "git_commit": None, "run_id": run_id,
                   "seed": manifest.seed, "profile_sampling": profile["profile_metadata"],
                   "intake_budget": audit["intake_budget"],
                   "identify_budget": state.get("identify_request", {}).get("budget"),
                   "qualitative_preparation": qualitative_preparation,
                   "tool_budget": None,
                   "screening_version": screening["screening_version"], "screening_input_hash": screening["input_hash"],
                   "promotion_policy_version": promotion["report"]["promotion_policy_version"] if promotion else None,
                   "promotion_policy_hash": promotion["report"]["promotion_policy_hash"] if promotion else None,
                   "screening_result_hash": screening["result_hash"],
                   "screening_policy": manifest.config.get("identify_config", {}).get("candidate_screening", {}).get("policy"),
                   "capability_hash": capabilities["result_hash"], "evidence_fingerprint": ledger["fingerprint"],
                   "model_runtime_snapshot_ids": [s["snapshot_id"] for s in runtime["snapshots"]],
                   "task_ids": [t["task_id"] for t in routing["tasks"]],
                   "environment": None},
              "limitations": [{"text": text, "sources": sorted(sources)} for text, sources in sorted(limitations.items())],
              "artifacts": {"references": sorted(references), "canonical_report": "audit_report.json", "markdown": "audit_report.md"}}
    # Remove unsafe references wherever upstream extensible content carries them.
    def safe_refs(value):
        if isinstance(value, dict):
            return {key: ([path for ref in item if (path := _relative(root, ref))]
                          if key in {"artifact_references", "invocation_references", "references"} and isinstance(item, list)
                          else _relative(root, item) if key in {"artifact_reference", "evidence_ledger_reference"}
                          else safe_refs(item)) for key, item in value.items()}
        if isinstance(value, list):
            return [safe_refs(item) for item in value]
        return value
    return AuditReport.model_validate(_public(safe_refs(report)))


def render_audit_markdown(report: AuditReport) -> str:
    """Render the bounded reader-facing default from canonical evidence."""
    from shortcutminer.experiment.audit_readable import render_readable_audit
    return render_readable_audit(report)


def render_audit_technical_markdown(report: AuditReport) -> str:
    """Render only the canonical report; never read other artifacts."""
    data = report.model_dump(mode="json")
    sections = ["# ShortcutMiner Audit Report", "", "Scientific authority: EvidenceLedger + SummaryReport.", ""]
    mapping = [("Run Summary", "run"), ("Dataset", "dataset"), ("Shortcut Candidate Discovery", "discovery"),
               ("Candidate Screening", "screening"), ("Validation Plan", "validation"),
               ("Evidence Promotion", "evidence_promotion"),
               ("Quantitative Evidence", "quantitative"), ("Qualitative Evidence", "qualitative"),
               ("Evidence Synthesis", "evidence"), ("Final Findings", "findings"), ("Runtime and Cost", "cost"),
               ("Model Runtime", "runtime"), ("Limitations", "limitations"),
               ("Reproducibility", "reproducibility"), ("Artifact Index", "artifacts")]
    def text(value):
        if value is None:
            return "N/A"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True).replace("`", "\\`")
        return str(value).replace("\n", " ").replace("`", "\\`").replace("<", "&lt;").replace(">", "&gt;")
    def entries(value, depth=0):
        indent = "  " * depth
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (dict, list)) and item:
                    sections.append(f"{indent}- **{text(key)}**:")
                    entries(item, depth + 1)
                else:
                    sections.append(f"{indent}- **{text(key)}**: {text(item)}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                sections.append(f"{indent}- Record {index + 1}:")
                entries(item, depth + 1)
        else:
            sections.append(f"{indent}- {text(value)}")
    for title, key in mapping:
        sections.extend([f"## {title}", ""])
        if key == "screening":
            sections.extend([f"**{SCREENING_EXPLANATION}**", ""])
        entries(data[key])
        sections.append("")
    return "\n".join(sections)


def export_audit_report(runs_root: str | Path, run_id: str) -> AuditReport:
    """Idempotently write JSON and Markdown from one validated canonical model."""
    report = build_audit_report(runs_root, run_id)
    root = get_run_dir(runs_root, run_id)
    for name in ("audit_report.json", "audit_report.md", "audit_report.technical.md"):
        if _relative(root, name) is None:
            raise AuditReportError("Unsafe report output path")
    atomic_write_json(root / "audit_report.json", report.model_dump(mode="json"), error_type=AuditReportError)
    for name, render in (("audit_report.md", render_audit_markdown),
                         ("audit_report.technical.md", render_audit_technical_markdown)):
        temp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, suffix=".tmp", delete=False) as file:
                temp = Path(file.name)
                file.write(render(report))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp, root / name)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
    return report
