"""Build audit routing authority from bounded public facts and static backends.

This controller never invokes tools/models, reads source data, or constructs tasks.
Only its fixed current-run cache is read when recovering an interrupted run.
"""

import hashlib
import json
from typing import Any
from pathlib import Path
from typing import Literal

from pydantic import Field

from shortcutminer.experiment.check_resolution import EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES
from shortcutminer.experiment.candidate_binding import resolve_candidate_binding
from shortcutminer.experiment.check_requirements import build_executable_candidate_context
from shortcutminer.protocol.audit import AuditRequest
from shortcutminer.protocol.dataset_profile import DatasetProfile
from shortcutminer.protocol.executive import ExecutiveCapabilities
from shortcutminer.protocol.identify import CandidateReport, StrictProtocolModel
from shortcutminer.storage.json_artifacts import atomic_write_json
from shortcutminer.storage.paths import get_run_dir
from shortcutminer.tooling.runtime import ToolRuntime


# Exact reserved evaluation fields, never fuzzy matching or inference of field names.
PRIVATE_FIELDS = frozenset({"hidden_truth", "private_truth",
                           "expected_answer", "truth_family", "shortcut_truth"})
CHECKS_WITHOUT_EXTRA_SPEC = ("association_test", "subgroup_evaluation",
                             "model_evaluation", "intervention")
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})
TEXT_EXTENSIONS = frozenset({".txt", ".md"})


class AuditCapabilityContext(StrictProtocolModel):
    capability_version: Literal["v1"] = "v1"
    run_id: str
    input_hash: str
    result_hash: str
    capabilities: ExecutiveCapabilities
    dataset_profile: DatasetProfile
    execution_resources: dict[str, Any] = Field(default_factory=dict)
    qualitative_preparation: dict[str, str] = Field(default_factory=dict)


def _hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def backend_snapshot(runtime: ToolRuntime | None, *, agent: str, run_id: str, root: str) -> dict:
    """Read registration and immutable authority, excluding histories and budgets used."""
    if runtime is None:
        return {"tools": [], "experiment_types": [], "model_capabilities": [], "constraints": []}
    if (runtime.agent_name != agent or runtime.run_id != run_id
            or runtime.context.dataset_root is None
            or runtime.context.dataset_root.resolve() != Path(root).resolve()):
        raise ValueError("Audit backend authority does not match agent/run/dataset")
    experiments = runtime.registry.backend_types_for(agent, "run_experiment")
    if runtime.context.allowed_experiment_types:
        experiments = tuple(item for item in experiments if item in runtime.context.allowed_experiment_types)
    return {"tools": [spec.model_dump(mode="json") for spec in runtime.tool_specs()],
            "experiment_types": sorted(experiments),
            "model_capabilities": sorted(runtime.context.model_capabilities),
            "constraints": sorted(runtime.context.runtime_constraints)}


def _inputs(audit, profile, candidates, quantitative_runtime, qualitative_runtime):
    intake = audit.dataset_info
    if intake is None:
        raise ValueError("Capability building requires completed dataset intake")
    source = Path(audit.dataset_path).expanduser().resolve()
    root = Path(intake.root).resolve()
    source_matches = source == root or any(source == (root / reference).resolve()
                                          for reference in intake.metadata_files)
    if (intake.name != profile.dataset.name
            or profile.dataset.name != candidates.dataset_summary.dataset
            or Path(intake.root).resolve() != Path(profile.dataset.root_reference).resolve()
            or not source_matches):
        raise ValueError("Audit capability inputs refer to different datasets")
    if profile.task.label_column in PRIVATE_FIELDS or profile.labels.column in PRIVATE_FIELDS:
        raise ValueError("Private truth cannot be used as an audit label source")
    if profile.metadata.available and (profile.metadata.file != intake.selected_metadata_file
            or profile.dataset.metadata_file != intake.selected_metadata_file):
        raise ValueError("Profile metadata file is not authorized by intake")
    # Extra workflow metadata, evaluator files, future results and candidate field claims
    # cannot authorize columns. Restrict the profile before hashing or exposing it.
    names = sorted(set(intake.metadata_columns) & set(profile.metadata.column_names)
                   & {column.name for column in profile.metadata.columns} - PRIVATE_FIELDS)
    if profile.labels.validity == "valid" and profile.labels.column not in names:
        raise ValueError("Profile label field is not authorized by intake")
    public = profile.model_dump(mode="json")
    public["metadata"]["column_names"] = names
    public["metadata"]["columns"] = sorted(
        [column for column in public["metadata"]["columns"] if column["name"] in names],
        key=lambda column: column["name"])
    if profile.splits.detected_field not in names:
        public["splits"] = {"detected_field": None, "distribution": {}}
    public_profile = DatasetProfile.model_validate(public)
    snapshots = {agent: backend_snapshot(runtime, agent=agent, run_id=candidates.run_id, root=intake.root)
                 for agent, runtime in (("quantitative", quantitative_runtime), ("qualitative", qualitative_runtime))}
    public_intake = intake.model_dump(mode="json")
    public_intake["metadata_columns"] = names
    fingerprint = _hash({"version": "v1", "intake": public_intake,
                         "execution_options": audit.execution_options.model_dump(mode="json") if audit.execution_options else None,
                         "profile": public_profile.model_dump(mode="json"),
                         "candidates": candidates.model_dump(mode="json"), "backends": snapshots})
    return public_profile, names, snapshots, fingerprint


def build_audit_capabilities(
    audit: AuditRequest, profile: DatasetProfile, candidates: CandidateReport, *,
    quantitative_runtime: ToolRuntime | None = None, qualitative_runtime: ToolRuntime | None = None,
) -> AuditCapabilityContext:
    """Offer only checks whose resource prerequisites can be bound without guesses."""
    public, fields, backends, fingerprint = _inputs(
        audit, profile, candidates, quantitative_runtime, qualitative_runtime)
    intake = audit.dataset_info
    limitations = list(dict.fromkeys([*intake.limitations, *profile.limitations]))
    if set(profile.metadata.column_names) - set(fields):
        limitations.append("metadata_fields_not_authorized_by_intake_or_reserved")
    resources = {}
    addressing = [profile.task.label_column]
    configured = audit.execution_options
    if configured is not None:
        configured_addressing = [configured.sample_id_column, configured.sample_path_column, configured.split_column]
        if len(set(configured_addressing + addressing)) != 4 or not set(configured_addressing).issubset(fields):
            raise ValueError("Explicit audit addressing must identify distinct authorized metadata columns")
        resources.update(configured.model_dump(mode="json", exclude={"excluded_scientific_fields", "visual_measurement_bindings"}))
        addressing.extend(configured_addressing)
        if not set(configured.excluded_scientific_fields).issubset(fields):
            raise ValueError("Excluded scientific fields must be authorized metadata columns")
        addressing.extend(configured.excluded_scientific_fields)
    # Only unambiguous exact conventional addressing names are bound. These fields
    # remain control inputs and cannot become scientific candidate bindings.
    for role, choices in (("sample_id_column", {"id", "sample_id"}),
                          ("sample_path_column", {"path", "sample_path", "image_path"})):
        found = sorted(set(fields) & choices)
        addressing.extend(found)
        if role in resources:
            continue
        if len(found) == 1:
            resources[role] = found[0]
        elif len(found) > 1:
            limitations.append(f"ambiguous_{role}")
    if profile.splits.detected_field in fields:
        resources["split_column"] = profile.splits.detected_field
        addressing.append(profile.splits.detected_field)
    addressing = list(dict.fromkeys(addressing))
    scientific = set(fields) - set(addressing)
    metadata_ready = bool(profile.metadata.available and profile.metadata.rows_profiled and fields)
    labelled = (intake.label_source.status == "resolved" and intake.label_source.kind == "metadata_column"
                and intake.label_source.column == profile.task.label_column
                and profile.labels.column == profile.task.label_column
                and profile.labels.validity == "valid" and profile.task.label_column in fields)
    if not labelled:
        limitations.append("executable_label_source_unavailable")
    qt, ql = backends["quantitative"], backends["qualitative"]
    ql_tools = {spec["name"] for spec in ql["tools"]}
    extensions = {sample.extension.lower() for sample in profile.sample_references}
    file_samples = bool(extensions & (IMAGE_EXTENSIONS | TEXT_EXTENSIONS))
    # With metadata, raw samples must have an exact bound path. Directory discovery
    # is supported by the existing qualitative preparation only without metadata.
    raw_samples = bool(file_samples and (not profile.metadata.available or "sample_path_column" in resources))
    sample_access = bool(raw_samples and "inspect_sample" in ql_tools)
    image_access = bool(sample_access and extensions & IMAGE_EXTENSIONS)
    ql_checks = []
    if metadata_ready and labelled and scientific and "inspect_sample_metadata" in ql_tools:
        ql_checks.append("metadata_inspection")
    if sample_access:
        ql_checks.extend(["sample_inspection", "counterexample_inspection"])
    if not sample_access:
        limitations.append("raw_sample_inspection_unavailable")
    qt_checks = []
    if metadata_ready and labelled and scientific:
        qt_checks = [check for check in CHECKS_WITHOUT_EXTRA_SPEC
                     if set(EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES[check]).issubset(qt["experiment_types"])]
    if not qt["experiment_types"]:
        limitations.append("quantitative_backend_unavailable")
    # Training/OOD/interventions needing a structured model or intervention spec
    # are deliberately not invented from file extensions or candidate prose.
    if configured is not None and image_access and {"retraining", "checkpoint"}.issubset(qt["model_capabilities"]):
        qt_checks.extend(check for check in ("task_model_training", "task_model_evaluation", "ood_evaluation")
                         if set(EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES[check]).issubset(qt["experiment_types"])
                         and (check != "ood_evaluation" or configured.ood_specs or configured.background_ood_specs or configured.color_ood_specs or configured.patch_ood_specs or configured.visual_ood_specs or configured.ood_recipe_selection))
    else:
        limitations.append("model_training_and_structured_checks_require_verified_model_inputs")
    supported = sorted({kind for check in qt_checks for kind in EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES[check]})
    capabilities = ExecutiveCapabilities(
        available_targets=(["quantitative"] if qt_checks else []) + (["qualitative"] if ql_checks else []),
        available_metadata_fields=fields, addressing_fields=addressing,
        available_splits=sorted(profile.splits.distribution) if "split_column" in resources else [],
        quantitative_checks=qt_checks, qualitative_checks=ql_checks,
        sample_access=sample_access, image_access=image_access,
        model_capabilities=qt["model_capabilities"] if configured is not None else [],
        structured_intervention_fields=list(dict.fromkeys([*configured.ood_specs, *configured.background_ood_specs, *configured.color_ood_specs, *configured.patch_ood_specs, *configured.visual_ood_specs,
            *(configured.ood_recipe_selection.fields if configured.ood_recipe_selection else [])])) if configured is not None else [],
        available_output_types=(["metrics"] if qt_checks else []) + (["observations"] if ql_checks else []),
        available_validation_methods=[*qt_checks, *ql_checks], supported_experiment_types=supported,
        limitations=list(dict.fromkeys(limitations)),
        **({'visual_measurement_bindings': configured.visual_measurement_bindings} if configured and configured.visual_measurement_bindings else {}),
    )
    # Reuse the Phase16 exact binding rules to retain only methods usable by at
    # least one promoted hypothesis. This does not create a validation task.
    contexts = []
    for candidate in candidates.candidates:
        binding = resolve_candidate_binding(candidate, available_metadata_fields=fields,
            label_column=profile.task.label_column, addressing_fields=addressing,
            visual_measurement_bindings=capabilities.visual_measurement_bindings)
        if binding.status == "unresolved":
            limitations.append(f"candidate_binding_unresolved:{candidate.candidate_id}")
            continue
        contexts.append(build_executable_candidate_context(binding, capabilities=capabilities,
                        runtime_constraints=qt["constraints"],
                        structured_spec_available=bool(set(binding.metadata_fields) & set(capabilities.structured_intervention_fields))))
    qt_checks = [check for check in qt_checks if any(check in item.allowed_quantitative_checks for item in contexts)]
    ql_checks = [check for check in ql_checks if any(check in item.allowed_qualitative_checks for item in contexts)]
    capabilities.quantitative_checks = qt_checks
    capabilities.qualitative_checks = ql_checks
    capabilities.available_targets = (["quantitative"] if qt_checks else []) + (["qualitative"] if ql_checks else [])
    capabilities.available_validation_methods = [*qt_checks, *ql_checks]
    capabilities.supported_experiment_types = sorted({kind for check in qt_checks
                                                     for kind in EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES[check]})
    capabilities.available_output_types = (["metrics"] if qt_checks else []) + (["observations"] if ql_checks else [])
    capabilities.limitations = list(dict.fromkeys(limitations))
    payload = {"capabilities": capabilities.model_dump(mode="json"),
               "dataset_profile": public.model_dump(mode="json"), "execution_resources": resources,
               "qualitative_preparation": {**{k:v for k,v in resources.items() if k in {"sample_id_column", "sample_path_column", "split_column"}}, "sampling_strategy": "random"}}
    return AuditCapabilityContext(run_id=candidates.run_id, input_hash=fingerprint,
                                  result_hash=_hash(payload), **payload)


def load_or_build_audit_capabilities(audit, profile, candidates, *, runs_root=None,
                                     persisted=None, **runtimes) -> AuditCapabilityContext:
    """Revalidate authority and reuse a matching checkpoint or atomic cache."""
    _, _, _, fingerprint = _inputs(audit, profile, candidates,
                                  runtimes.get("quantitative_runtime"), runtimes.get("qualitative_runtime"))
    path = None
    if runs_root is not None:
        run_dir = get_run_dir(runs_root, candidates.run_id)
        path = run_dir / "capabilities" / "result.json"
        if not path.resolve().is_relative_to(run_dir):
            raise ValueError("Capability artifact escapes current run")
    if persisted is not None or (path is not None and path.is_file()):
        result = (AuditCapabilityContext.model_validate(persisted) if persisted is not None
                  else AuditCapabilityContext.model_validate_json(path.read_text(encoding="utf-8")))
        payload = result.model_dump(mode="json", exclude={"capability_version", "run_id", "input_hash", "result_hash"})
        if result.run_id != candidates.run_id or result.input_hash != fingerprint:
            raise ValueError("Audit capability authority changed; cached context cannot be reused")
        if result.result_hash != _hash(payload):
            raise ValueError("Audit capability result hash mismatch")
        return result
    result = build_audit_capabilities(audit, profile, candidates, **runtimes)
    if path is not None:
        atomic_write_json(path, result.model_dump(mode="json"), error_type=ValueError)
    return result
