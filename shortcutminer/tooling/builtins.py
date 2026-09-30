"""Typed wrappers around existing bounded inspection and storage functions."""

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import Field, field_validator

from shortcutminer.experiment.tool_executor import (
    ExperimentExecutionRequest,
    ExperimentExecutor,
    ExperimentType,
)
from shortcutminer.protocol.experiment import (
    EmptyExperimentParameters,
    ExperimentParameters,
)
from shortcutminer.protocol.executive import TaskId
from shortcutminer.protocol.identify import (
    CandidateId,
    NonEmptyText,
    SafeRelativePath,
    StrictProtocolModel,
)
from shortcutminer.protocol.qualitative import (
    ImageDescriptorName,
    QualitativePreparationOptions,
)
from shortcutminer.protocol.qualitative_inspection import SampleEvidence
from shortcutminer.tooling.base import StructuredTool, ToolExecutionContext
from shortcutminer.tooling.protocol import AgentName
from shortcutminer.tooling.registry import ToolRegistry
from tools.common.dataset_inspector import inspect_dataset
from tools.common.metadata_profiler import profile_metadata
from tools.common.metadata_profiler import _normalize_value, load_metadata
from tools.common.sample_selector import select_samples
from tools.qualitative.evidence_builder import build_evidence_manifest
from shortcutminer.experiment.qualitative_backend import inspect_sample_record


class EmptyInput(StrictProtocolModel):
    pass


class DatasetInspectionInput(StrictProtocolModel):
    max_entries: int = Field(default=10_000, ge=1, le=100_000)


class DatasetInspectionOutput(StrictProtocolModel):
    root: NonEmptyText
    exists: bool
    num_files: int = Field(ge=0)
    num_directories: int = Field(ge=0)
    file_extensions: list[str]
    file_extension_counts: dict[str, int]
    discovered_files: list[SafeRelativePath]
    top_level_entries: list[str]
    max_depth_observed: int = Field(ge=0)
    entries_scanned: int = Field(ge=0)
    max_entries: int = Field(ge=1)
    truncated: bool
    scan_errors: list[str]


class MetadataInspectionInput(StrictProtocolModel):
    metadata_file: SafeRelativePath
    sample_values_per_column: int = Field(default=5, ge=0, le=100)
    max_rows: int = Field(default=100_000, ge=1, le=1_000_000)


class MetadataInspectionOutput(StrictProtocolModel):
    path: NonEmptyText
    exists: bool
    num_rows: int = Field(ge=0)
    num_columns: int = Field(ge=0)
    columns: list[dict[str, Any]]
    rows_profiled: int = Field(ge=0)
    row_limit: int = Field(ge=1)
    truncated: bool


class ColumnDistributionInput(StrictProtocolModel):
    metadata_file: SafeRelativePath
    column: NonEmptyText
    max_values: int = Field(default=100, ge=1, le=10_000)
    max_rows: int = Field(default=100_000, ge=1, le=1_000_000)


class ColumnDistributionOutput(StrictProtocolModel):
    column: NonEmptyText
    values: list[dict[str, Any]]
    missing_count: int = Field(ge=0)
    rows_profiled: int = Field(ge=0)
    truncated_rows: bool
    truncated_values: bool


class SampleSelectionInput(StrictProtocolModel):
    metadata_file: SafeRelativePath
    max_samples: int = Field(ge=1, le=100_000)
    seed: int
    strategy: Literal["random", "class_balanced", "stratified"] = "random"
    label_column: NonEmptyText | None = None
    stratify_column: NonEmptyText | None = None
    max_metadata_rows: int = Field(default=100_000, ge=1, le=1_000_000)


class SampleSelectionOutput(StrictProtocolModel):
    samples: list[dict[str, Any]]


class ArtifactReadInput(StrictProtocolModel):
    artifact: SafeRelativePath
    max_bytes: int = Field(default=1_000_000, ge=1, le=10_000_000)
    max_rows: int = Field(default=1_000, ge=1, le=100_000)


class ArtifactReadOutput(StrictProtocolModel):
    artifact: SafeRelativePath
    media_type: Literal["json", "jsonl", "csv", "text", "image_metadata"]
    byte_size: int = Field(ge=0)
    content: Any
    truncated: bool


class SampleMetadataInput(StrictProtocolModel):
    sample_id: NonEmptyText
    sample_reference: SafeRelativePath


class SampleMetadataOutput(StrictProtocolModel):
    sample_id: NonEmptyText
    sample_reference: SafeRelativePath | None
    sample_available: bool
    media_type: str | None
    byte_size: int | None
    limitations: list[str]


class SampleInspectionInput(StrictProtocolModel):
    sample_id: NonEmptyText
    sample_reference: SafeRelativePath
    requested_descriptors: list[ImageDescriptorName] = Field(
        default_factory=lambda: [
            "dimensions",
            "intensity",
            "rgb_stats",
            "entropy",
            "edge_density",
            "spatial_stats",
        ]
    )
    max_file_bytes: int = Field(default=25_000_000, ge=1, le=100_000_000)
    max_pixels: int = Field(default=20_000_000, ge=1, le=100_000_000)
    max_frames: int = Field(default=1, ge=1, le=100)
    max_text_bytes: int = Field(default=16_384, ge=1, le=1_000_000)

    @field_validator("requested_descriptors")
    @classmethod
    def validate_unique_descriptors(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("requested_descriptors must not contain duplicates")
        return values


class TraceReadInput(StrictProtocolModel):
    max_events: int = Field(default=100, ge=1, le=10_000)


class TraceReadOutput(StrictProtocolModel):
    events: list[dict[str, Any]]
    truncated: bool


class CapabilityOutput(StrictProtocolModel):
    tools: list[dict[str, Any]]


class ExperimentToolInput(StrictProtocolModel):
    candidate_id: CandidateId
    task_id: TaskId
    experiment_type: ExperimentType
    parameters: ExperimentParameters = Field(
        default_factory=EmptyExperimentParameters
    )


class ExperimentToolOutput(StrictProtocolModel):
    outcome: Literal["completed", "inconclusive"]
    metrics: dict[str, str | int | float | bool | None]
    summary: NonEmptyText
    artifact_references: list[SafeRelativePath]
    metadata: dict[str, Any]


def _dataset_root(context: ToolExecutionContext) -> Path:
    if context.dataset_root is None:
        raise ValueError("Tool Runtime has no approved dataset root")
    root = context.dataset_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Approved dataset root is not a directory: {root}")
    return root


def _run_root(context: ToolExecutionContext) -> Path:
    if context.runs_root is None:
        raise ValueError("Tool Runtime has no approved runs root")
    root = (context.runs_root / context.run_id).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Approved run root is not a directory: {root}")
    return root


def _resolve_beneath(root: Path, reference: str, *, require_file: bool) -> Path:
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / reference).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Path resolves outside its approved root") from exc
    if require_file and not candidate.is_file():
        raise FileNotFoundError(f"Approved file does not exist: {reference}")
    return candidate


def _inspect_dataset(arguments, context):
    return DatasetInspectionOutput.model_validate(
        inspect_dataset(str(_dataset_root(context)), arguments.max_entries)
    )


def _inspect_metadata(arguments, context):
    path = _resolve_beneath(
        _dataset_root(context), arguments.metadata_file, require_file=False
    )
    return MetadataInspectionOutput.model_validate(
        profile_metadata(
            str(path),
            sample_values_per_column=arguments.sample_values_per_column,
            max_rows=arguments.max_rows,
        )
    )


def _select_samples(arguments, context):
    if arguments.max_samples > context.max_selected_samples:
        raise ValueError("Sample selection exceeds the Tool Runtime budget")
    path = _resolve_beneath(
        _dataset_root(context), arguments.metadata_file, require_file=True
    )
    samples = select_samples(
        str(path),
        max_samples=arguments.max_samples,
        seed=arguments.seed,
        strategy=arguments.strategy,
        label_column=arguments.label_column,
        stratify_column=arguments.stratify_column,
        max_metadata_rows=arguments.max_metadata_rows,
    )
    return SampleSelectionOutput(samples=samples)


def _column_distribution(arguments, context):
    path = _resolve_beneath(
        _dataset_root(context), arguments.metadata_file, require_file=True
    )
    loaded = load_metadata(str(path), max_rows=arguments.max_rows)
    if arguments.column not in loaded.dataframe.columns:
        raise ValueError(f"Column not found: {arguments.column}")
    series = loaded.dataframe[arguments.column]
    counts = [
        {"value": _normalize_value(value), "count": int(count)}
        for value, count in series.dropna().value_counts(sort=False).items()
    ]
    counts.sort(key=lambda item: (-item["count"], str(item["value"])))
    return ColumnDistributionOutput(
        column=arguments.column,
        values=counts[: arguments.max_values],
        missing_count=int(series.isna().sum()),
        rows_profiled=len(series),
        truncated_rows=loaded.truncated,
        truncated_values=len(counts) > arguments.max_values,
    )


def _read_artifact(arguments, context):
    if arguments.artifact not in context.approved_artifacts:
        raise PermissionError("Artifact is not in the approved reference set")
    path = _resolve_beneath(
        _run_root(context), arguments.artifact, require_file=True
    )
    byte_size = path.stat().st_size
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}:
        return ArtifactReadOutput(
            artifact=arguments.artifact,
            media_type="image_metadata",
            byte_size=byte_size,
            content={"suffix": suffix},
            truncated=False,
        )
    raw = path.read_bytes()[: arguments.max_bytes + 1]
    truncated = len(raw) > arguments.max_bytes
    raw = raw[: arguments.max_bytes]
    text = raw.decode("utf-8")
    if suffix == ".json":
        content = json.loads(text)
        media_type = "json"
    elif suffix == ".jsonl":
        lines = text.splitlines()[: arguments.max_rows]
        content = [json.loads(line) for line in lines]
        truncated = truncated or len(text.splitlines()) > arguments.max_rows
        media_type = "jsonl"
    elif suffix == ".csv":
        dataframe = pd.read_csv(path, nrows=arguments.max_rows + 1)
        truncated = truncated or len(dataframe) > arguments.max_rows
        content = dataframe.iloc[: arguments.max_rows].to_dict(orient="records")
        media_type = "csv"
    elif suffix in {".txt", ".md"}:
        content = text
        media_type = "text"
    else:
        raise ValueError(f"Unsupported artifact format: {suffix or '<none>'}")
    return ArtifactReadOutput(
        artifact=arguments.artifact,
        media_type=media_type,
        byte_size=byte_size,
        content=content,
        truncated=truncated,
    )


def _inspect_sample_metadata(arguments, context):
    if arguments.sample_reference not in context.approved_samples:
        raise PermissionError("Sample is not in the approved reference set")
    records, limitations = build_evidence_manifest(
        str(_dataset_root(context)),
        [
            {
                "sample_id": arguments.sample_id,
                "sample_reference": arguments.sample_reference,
            }
        ],
    )
    record = records[0]
    return SampleMetadataOutput(
        sample_id=record["sample_id"],
        sample_reference=record["sample_reference"],
        sample_available=record["sample_available"],
        media_type=record["media_type"],
        byte_size=record["byte_size"],
        limitations=limitations,
    )


def _inspect_sample(arguments, context):
    if arguments.sample_reference not in context.approved_samples:
        raise PermissionError("Sample is not in the approved reference set")
    records, manifest_limitations = build_evidence_manifest(
        str(_dataset_root(context)),
        [
            {
                "sample_id": arguments.sample_id,
                "metadata_row_index": None,
                "group": "ungrouped",
                "group_values": {},
                "sample_reference": arguments.sample_reference,
                "metadata": {},
                "selection_reason": "random",
            }
        ],
    )
    options = QualitativePreparationOptions(
        sampling_strategy="random",
        seed=0,
        requested_descriptors=arguments.requested_descriptors,
        max_file_bytes=arguments.max_file_bytes,
        max_pixels=arguments.max_pixels,
        max_frames=arguments.max_frames,
        max_text_bytes=arguments.max_text_bytes,
        max_total_inspection_bytes=arguments.max_file_bytes,
        create_contact_sheet=False,
    )
    evidence = inspect_sample_record(
        str(_dataset_root(context)),
        records[0],
        options,
    )
    return evidence.model_copy(
        update={
            "limitations": list(
                dict.fromkeys([*manifest_limitations, *evidence.limitations])
            )
        }
    )


def _read_trace(arguments, context):
    path = _run_root(context) / "events.jsonl"
    if not path.is_file():
        return TraceReadOutput(events=[], truncated=False)
    lines = path.read_text(encoding="utf-8").splitlines()
    selected = lines[-arguments.max_events :]
    return TraceReadOutput(
        events=[json.loads(line) for line in selected],
        truncated=len(lines) > arguments.max_events,
    )


def _inspect_capabilities(_arguments, context):
    return CapabilityOutput(tools=list(context.tool_schemas))


def _experiment_handler(executor: ExperimentExecutor):
    def execute(arguments, context):
        if context.agent_name != "quantitative":
            raise PermissionError("Only Quantitative may execute experiments")
        if (
            context.allowed_experiment_types
            and arguments.experiment_type not in context.allowed_experiment_types
        ):
            raise PermissionError(
                "Experiment type was not authorized by the bound Executive task"
            )
        required = {
            "task_model_training": {"retraining"},
            "task_model_evaluation": {"checkpoint"},
            "ood_evaluation": {"checkpoint", "intervention_inference"},
            "counterfactual_intervention": {"checkpoint", "intervention_inference"},
        }.get(arguments.experiment_type, set())
        if arguments.experiment_type == "task_model_training" and "no_retraining" in context.runtime_constraints:
            raise PermissionError("task_model_training is prohibited by no_retraining")
        missing = required - set(context.model_capabilities)
        if missing:
            raise PermissionError(
                "Experiment requires unavailable model capabilities: "
                + ", ".join(sorted(missing))
            )
        request = ExperimentExecutionRequest(
            run_id=context.run_id,
            candidate_id=arguments.candidate_id,
            task_id=arguments.task_id,
            experiment_type=arguments.experiment_type,
            parameters=arguments.parameters,
        )
        result = executor.execute(request, context=context)
        return ExperimentToolOutput.model_validate(result.model_dump(mode="json"))

    return execute


class _UnavailableExperimentExecutor:
    def execute(self, _request, *, context=None):
        raise RuntimeError("No ExperimentExecutor is configured for this runtime")


def build_default_tool_registry(
    *,
    experiment_executor: ExperimentExecutor | None = None,
) -> ToolRegistry:
    """Build the least-privilege registry derived from the five workspaces."""
    registry = ToolRegistry()
    configured_experiment_executor = (
        experiment_executor or _UnavailableExperimentExecutor()
    )

    def add(name, description, input_model, output_model, handler, agents):
        registry.register(
            StructuredTool(name, description, input_model, output_model, handler),
            agents=set(agents),
        )

    all_agents: set[AgentName] = {
        "identify", "executive", "quantitative", "qualitative", "summarize"
    }
    add(
        "inspect_tool_capabilities",
        "List the typed tools permitted for the current Agent.",
        EmptyInput,
        CapabilityOutput,
        _inspect_capabilities,
        all_agents,
    )
    add(
        "inspect_dataset",
        "Inspect the approved dataset root within a bounded entry budget.",
        DatasetInspectionInput,
        DatasetInspectionOutput,
        _inspect_dataset,
        {"identify"},
    )
    add(
        "inspect_metadata",
        "Profile an approved dataset-relative CSV within a row budget.",
        MetadataInspectionInput,
        MetadataInspectionOutput,
        _inspect_metadata,
        {"identify"},
    )
    add(
        "select_samples",
        "Select deterministic metadata records from an approved CSV.",
        SampleSelectionInput,
        SampleSelectionOutput,
        _select_samples,
        {"identify"},
    )
    add(
        "get_column_distribution",
        "Count bounded values for one approved metadata column.",
        ColumnDistributionInput,
        ColumnDistributionOutput,
        _column_distribution,
        {"identify"},
    )
    add(
        "read_artifact",
        "Read a bounded approved run artifact in a supported format.",
        ArtifactReadInput,
        ArtifactReadOutput,
        _read_artifact,
        all_agents,
    )
    add(
        "inspect_sample_metadata",
        "Resolve approved sample metadata without claiming visual understanding.",
        SampleMetadataInput,
        SampleMetadataOutput,
        _inspect_sample_metadata,
        {"identify", "qualitative"},
    )
    add(
        "inspect_sample",
        "Inspect one approved local image or text sample with allowlisted bounded descriptors.",
        SampleInspectionInput,
        SampleEvidence,
        _inspect_sample,
        {"qualitative"},
    )
    add(
        "read_run_trace",
        "Read a bounded tail of the current run event trace.",
        TraceReadInput,
        TraceReadOutput,
        _read_trace,
        {"executive", "summarize"},
    )
    add(
        "run_experiment",
        "Run one typed experiment through the configured ExperimentExecutor.",
        ExperimentToolInput,
        ExperimentToolOutput,
        _experiment_handler(configured_experiment_executor),
        {"quantitative"},
    )
    return registry
