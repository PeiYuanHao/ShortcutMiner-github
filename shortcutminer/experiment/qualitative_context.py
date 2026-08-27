from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field

from shortcutminer.experiment.executive_executor import (
    ExecutiveExecutionError,
    validate_routing_plan,
)
from shortcutminer.protocol.executive import (
    ExecutiveRequest,
    RoutingPlan,
)
from shortcutminer.protocol.identify import NonEmptyText, StrictProtocolModel
from shortcutminer.protocol.qualitative import (
    QualitativePreparationOptions,
    QualitativeRequest,
    SamplingStrategy,
)
from tools.common.metadata_profiler import load_metadata
from tools.qualitative.evidence_builder import build_evidence_manifest
from tools.qualitative.image_sampler import select_qualitative_samples


class QualitativeContextError(RuntimeError):
    """Raised when authoritative state cannot form qualitative evidence."""


class PreparedQualitativeSample(StrictProtocolModel):
    sample_id: NonEmptyText
    metadata_row_index: int = Field(ge=0)
    group: NonEmptyText
    group_values: dict[str, Any]
    sample_reference: str | None = None
    metadata: dict[str, Any]
    selection_reason: SamplingStrategy
    sample_available: bool
    media_type: str | None = None
    byte_size: int | None = Field(default=None, ge=0)


class QualitativeEvidenceContext(StrictProtocolModel):
    protocol_version: str
    run_id: str
    task_id: str
    candidate_id: str
    seed: int
    sampling_strategy: SamplingStrategy
    requested_checks: list[str]
    metadata_fields: list[str]
    selected_samples: list[PreparedQualitativeSample]
    group_coverage: dict[str, int]
    metadata_available: bool
    metadata_rows_read: int = Field(ge=0)
    samples_selected: int = Field(ge=0)
    sample_files_available: int = Field(ge=0)
    authoritative_artifact_references: list[str]
    allowed_output_artifact_prefix: str
    limitations: list[str]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _evidence_metadata_fields(request: QualitativeRequest) -> list[str]:
    """Return every task-authorized field whose value may support evidence."""

    return _unique(
        [request.task.inputs.label_column, *request.task.inputs.metadata_fields]
    )


def build_qualitative_request(
    *,
    executive_request: ExecutiveRequest,
    routing_plan: RoutingPlan,
    task_id: str,
    message_id: str,
    preparation: QualitativePreparationOptions,
    limitations: list[str] | None = None,
) -> QualitativeRequest:
    """Build one qualitative request from a validated Executive plan."""

    try:
        validate_routing_plan(routing_plan, executive_request)
    except ExecutiveExecutionError as exc:
        raise QualitativeContextError(
            f"Executive state is invalid: {exc}"
        ) from exc

    matches = [task for task in routing_plan.tasks if task.task_id == task_id]
    if len(matches) != 1:
        raise QualitativeContextError(
            "Qualitative task_id must identify exactly one planned task."
        )
    task = matches[0]
    if task.target != "qualitative":
        raise QualitativeContextError("Selected task is not qualitative.")

    candidates = [
        candidate
        for candidate in executive_request.candidate_report.candidates
        if candidate.candidate_id == task.candidate_id
    ]
    if len(candidates) != 1:
        raise QualitativeContextError(
            "Qualitative task does not map to exactly one candidate."
        )
    candidate = candidates[0]

    permitted_fields = {task.inputs.label_column, *task.inputs.metadata_fields}
    requested_preparation_fields = {
        *preparation.group_columns,
        *(
            [preparation.sample_id_column]
            if preparation.sample_id_column is not None
            else []
        ),
        *(
            [preparation.sample_path_column]
            if preparation.sample_path_column is not None
            else []
        ),
        *(
            [preparation.split_column]
            if preparation.split_column is not None
            else []
        ),
    }
    unavailable_fields = requested_preparation_fields - permitted_fields
    if unavailable_fields:
        raise QualitativeContextError(
            "Qualitative preparation uses fields absent from AnalysisTask: "
            + ", ".join(sorted(unavailable_fields))
        )

    if (
        task.inputs.sample_limit is not None
        and preparation.max_evidence_records > task.inputs.sample_limit
    ):
        raise QualitativeContextError(
            "Qualitative preparation exceeds the AnalysisTask sample_limit."
        )

    combined_limitations = _unique(
        [
            *executive_request.routing_context.limitations,
            *routing_plan.limitations,
            *next(
                decision.limitations
                for decision in routing_plan.decisions
                if decision.candidate.candidate_id == task.candidate_id
            ),
            *(limitations or []),
        ]
    )
    return QualitativeRequest(
        protocol_version=executive_request.protocol_version,
        run_id=executive_request.run_id,
        message_id=message_id,
        source="orchestrator",
        target="qualitative",
        message_type="qualitative_analysis_request",
        in_reply_to=task.message_id,
        candidate=candidate,
        task=task,
        metadata_file=executive_request.routing_context.dataset.metadata_file,
        preparation=preparation,
        limitations=combined_limitations,
    )


def _metadata_path(dataset_root: str, metadata_file: str) -> Path:
    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")
    path = (root / metadata_file).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Metadata path resolves outside the dataset root.") from exc
    return path


def _empty_context(
    request: QualitativeRequest,
    *,
    limitations: list[str],
) -> QualitativeEvidenceContext:
    return QualitativeEvidenceContext(
        protocol_version=request.protocol_version,
        run_id=request.run_id,
        task_id=request.task.task_id,
        candidate_id=request.candidate.candidate_id,
        seed=request.preparation.seed,
        sampling_strategy=request.preparation.sampling_strategy,
        requested_checks=request.task.requested_checks,
        metadata_fields=_evidence_metadata_fields(request),
        selected_samples=[],
        group_coverage={},
        metadata_available=False,
        metadata_rows_read=0,
        samples_selected=0,
        sample_files_available=0,
        authoritative_artifact_references=request.task.inputs.artifact_references,
        allowed_output_artifact_prefix=(
            f"qualitative/artifacts/{request.candidate.candidate_id}"
        ),
        limitations=_unique([*request.limitations, *limitations]),
    )


def build_qualitative_context(
    request: QualitativeRequest,
) -> QualitativeEvidenceContext:
    """Prepare bounded metadata-linked evidence without altering samples."""

    root = request.task.inputs.dataset_reference
    if request.metadata_file is None:
        return _empty_context(
            request,
            limitations=["No metadata file was configured for qualitative sampling."],
        )

    try:
        metadata_path = _metadata_path(root, request.metadata_file)
        loaded = load_metadata(
            str(metadata_path),
            max_rows=request.preparation.max_metadata_rows,
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        OSError,
        ValueError,
    ) as exc:
        return _empty_context(
            request,
            limitations=[f"Qualitative metadata is unavailable or invalid: {exc}"],
        )

    limitations: list[str] = []
    if loaded.truncated:
        limitations.append(
            "Qualitative metadata reading reached the configured "
            f"max_metadata_rows={request.preparation.max_metadata_rows}; "
            "sampling uses only the bounded prefix."
        )

    evidence_metadata_fields = _evidence_metadata_fields(request)
    required_fields = _unique(
        [
            *request.preparation.group_columns,
            *evidence_metadata_fields,
            *(
                [request.preparation.sample_id_column]
                if request.preparation.sample_id_column
                else []
            ),
            *(
                [request.preparation.sample_path_column]
                if request.preparation.sample_path_column
                else []
            ),
            *(
                [request.preparation.split_column]
                if request.preparation.split_column
                else []
            ),
        ]
    )
    missing_fields = [
        field for field in required_fields if field not in loaded.dataframe.columns
    ]
    if missing_fields:
        return _empty_context(
            request,
            limitations=[
                "Required qualitative metadata fields are unavailable: "
                + ", ".join(missing_fields)
            ],
        ).model_copy(
            update={
                "metadata_available": True,
                "metadata_rows_read": len(loaded.dataframe),
            }
        )

    try:
        selected, coverage = select_qualitative_samples(
            loaded,
            max_samples=request.preparation.max_evidence_records,
            seed=request.preparation.seed,
            strategy=request.preparation.sampling_strategy,
            group_columns=request.preparation.group_columns,
            metadata_fields=evidence_metadata_fields,
            sample_id_column=request.preparation.sample_id_column,
            sample_path_column=request.preparation.sample_path_column,
            split_column=request.preparation.split_column,
            split=request.task.inputs.split,
        )
    except (TypeError, ValueError) as exc:
        return _empty_context(
            request,
            limitations=[f"Qualitative sample selection failed: {exc}"],
        ).model_copy(
            update={
                "metadata_available": True,
                "metadata_rows_read": len(loaded.dataframe),
            }
        )

    manifest, manifest_limitations = build_evidence_manifest(root, selected)
    limitations.extend(manifest_limitations)
    if request.task.inputs.requires_sample_access and not request.preparation.sample_path_column:
        limitations.append(
            "The qualitative task requires sample access, but no sample path "
            "column was configured."
        )

    available_count = sum(bool(item["sample_available"]) for item in manifest)
    if request.task.inputs.requires_sample_access and available_count == 0:
        limitations.append("No selected sample file is available for inspection.")

    return QualitativeEvidenceContext(
        protocol_version=request.protocol_version,
        run_id=request.run_id,
        task_id=request.task.task_id,
        candidate_id=request.candidate.candidate_id,
        seed=request.preparation.seed,
        sampling_strategy=request.preparation.sampling_strategy,
        requested_checks=request.task.requested_checks,
        metadata_fields=evidence_metadata_fields,
        selected_samples=manifest,
        group_coverage=coverage,
        metadata_available=True,
        metadata_rows_read=len(loaded.dataframe),
        samples_selected=len(manifest),
        sample_files_available=available_count,
        authoritative_artifact_references=request.task.inputs.artifact_references,
        allowed_output_artifact_prefix=(
            f"qualitative/artifacts/{request.candidate.candidate_id}"
        ),
        limitations=_unique([*request.limitations, *limitations]),
    )
