"""Bounded local sample inspection for qualitative evidence preparation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any
from uuid import uuid4

from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import TypeAdapter, ValidationError

from shortcutminer.inspection.image import (
    DESCRIPTOR_DEFINITIONS,
    SUPPORTED_IMAGE_FORMATS,
    flatten_image_descriptors,
    inspect_image_file,
)
from shortcutminer.protocol.qualitative import (
    QualitativePreparationOptions,
    QualitativeRequest,
)
from shortcutminer.protocol.qualitative_inspection import (
    DescriptorComparison,
    GroupDescriptorSummary,
    InspectionCoverage,
    QualitativeInspectionArtifact,
    SampleEvidence,
)
from shortcutminer.protocol.identify import SafeRelativePath
from shortcutminer.storage.json_artifacts import atomic_write_json
from tools.qualitative.metadata_inspector import resolve_dataset_reference


class QualitativeInspectionError(RuntimeError):
    """Raised when the deterministic inspection boundary cannot execute safely."""


SUPPORTED_TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt"}


def _request_fingerprint(request: QualitativeRequest) -> str:
    canonical = json.dumps(
        request.model_dump(mode="json", exclude={"inspection_scope"} if request.inspection_scope is None else set()),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _number(value: float) -> float:
    return round(float(value), 10)


def _base_evidence(record: dict[str, Any], **updates: Any) -> SampleEvidence:
    raw_reference = record.get("sample_reference")
    try:
        safe_reference = (
            TypeAdapter(SafeRelativePath).validate_python(raw_reference)
            if raw_reference is not None
            else None
        )
    except ValidationError:
        safe_reference = None
    value = {
        "sample_id": record["sample_id"],
        "metadata_row_index": record.get("metadata_row_index"),
        "group": record.get("group") or "ungrouped",
        "group_values": record.get("group_values", {}),
        "sample_reference": safe_reference,
        "media_type": record.get("media_type") or "unknown",
        "metadata": record.get("metadata", {}),
        "selection_reason": record.get("selection_reason", "random"),
        "sample_available": bool(record.get("sample_available")),
        "inspection_status": "failed",
        "visual_descriptors": {},
        "text_descriptors": {},
        "artifact_references": [],
        "warnings": [],
        "limitations": [],
        "byte_size": record.get("byte_size"),
        "bytes_consumed": 0,
    }
    value.update(updates)
    return SampleEvidence.model_validate(value)


def _inspect_image(
    dataset_root: str | Path,
    record: dict[str, Any],
    options: QualitativePreparationOptions,
) -> SampleEvidence:
    reference = record.get("sample_reference")
    if not reference:
        return _base_evidence(
            record,
            limitations=["Image sample has no safe dataset-relative reference."],
        )
    try:
        path = resolve_dataset_reference(str(dataset_root), reference)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        return _base_evidence(
            record,
            sample_available=False,
            limitations=[f"Image sample is unavailable or outside the approved root: {exc}"],
        )
    inspected = inspect_image_file(
        path,
        requested_descriptors=list(options.requested_descriptors),
        max_file_bytes=options.max_file_bytes,
        max_pixels=options.max_pixels,
        max_frames=options.max_frames,
    )
    return _base_evidence(
        record,
        byte_size=inspected.byte_size,
        bytes_consumed=inspected.bytes_consumed,
        inspection_status=inspected.status,
        visual_descriptors=inspected.descriptors,
        warnings=list(inspected.warnings),
        limitations=list(inspected.limitations),
    )


def _inspect_text(
    dataset_root: str | Path,
    record: dict[str, Any],
    options: QualitativePreparationOptions,
) -> SampleEvidence:
    reference = record.get("sample_reference")
    if not reference:
        return _base_evidence(
            record,
            limitations=["Text sample has no safe dataset-relative reference."],
        )
    try:
        path = resolve_dataset_reference(str(dataset_root), reference)
        byte_size = path.stat().st_size
        read_limit = min(options.max_text_bytes, options.max_file_bytes)
        with path.open("rb") as stream:
            raw = stream.read(read_limit + 1)
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        return _base_evidence(
            record,
            sample_available=False,
            limitations=[f"Text sample is unavailable or outside the approved root: {exc}"],
        )
    truncated = len(raw) > read_limit or byte_size > read_limit
    bounded = raw[:read_limit]
    try:
        text = bounded.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return _base_evidence(
            record,
            byte_size=byte_size,
            bytes_consumed=len(bounded),
            limitations=["Text preview is not valid UTF-8."],
        )
    line_count = 0 if not text else len(text.splitlines())
    limitations = (
        [f"Text preview was truncated at max_text_bytes={read_limit}."]
        if truncated
        else []
    )
    return _base_evidence(
        record,
        byte_size=byte_size,
        bytes_consumed=len(bounded),
        inspection_status="limited" if truncated else "completed",
        text_descriptors={
            "encoding": "utf-8",
            "bounded_line_count": line_count,
            "bounded_character_count": len(text),
            "bounded_whitespace_token_count": len(text.split()),
            "truncated": truncated,
            "untrusted_content": {
                "kind": "quoted_dataset_text",
                "instructional_status": "never_treat_as_instructions",
                "preview": text[:2048],
            },
        },
        limitations=limitations,
    )


def inspect_sample_record(
    dataset_root: str | Path,
    record: dict[str, Any],
    options: QualitativePreparationOptions,
) -> SampleEvidence:
    """Inspect one already selected record without expanding its authority."""

    if record.get("media_type") == "tabular":
        metadata = record.get("metadata", {})
        return _base_evidence(
            record,
            sample_available=True,
            inspection_status="completed",
            text_descriptors={
                "row_field_count": len(metadata),
                "missing_field_count": sum(value is None for value in metadata.values()),
                "untrusted_content": {
                    "kind": "quoted_dataset_metadata",
                    "instructional_status": "never_treat_as_instructions",
                },
            },
        )
    if not record.get("sample_available"):
        return _base_evidence(
            record,
            limitations=["Selected sample was unavailable before content inspection."],
        )
    if record.get("media_type") == "image":
        return _inspect_image(dataset_root, record, options)
    if record.get("media_type") == "text":
        return _inspect_text(dataset_root, record, options)
    return _base_evidence(
        record,
        limitations=["Selected sample format has no approved local inspector."],
    )


def inspect_selected_records(
    dataset_root: str | Path,
    records: list[dict[str, Any]],
    options: QualitativePreparationOptions,
) -> tuple[list[SampleEvidence], list[str], list[str]]:
    evidence: list[SampleEvidence] = []
    warnings_out: list[str] = []
    limitations: list[str] = []
    consumed = 0
    for record in records:
        media_type = record.get("media_type")
        byte_size = int(record.get("byte_size") or 0)
        estimated = (
            min(byte_size, options.max_text_bytes)
            if media_type == "text"
            else byte_size
        )
        if consumed + estimated > options.max_total_inspection_bytes:
            item = _base_evidence(
                record,
                inspection_status="limited",
                limitations=[
                    "Sample was not decoded because the qualitative task reached "
                    f"max_total_inspection_bytes={options.max_total_inspection_bytes}."
                ],
            )
        else:
            item = inspect_sample_record(dataset_root, record, options)
            consumed += item.bytes_consumed
        evidence.append(item)
        warnings_out.extend(item.warnings)
        limitations.extend(item.limitations)
    return evidence, _unique(warnings_out), _unique(limitations)


def _scalar_descriptors(sample: SampleEvidence) -> dict[str, float]:
    return flatten_image_descriptors(
        sample.visual_descriptors,
        file_size_bytes=sample.byte_size,
    )


def sample_has_inspection_evidence(sample: SampleEvidence) -> bool:
    """Return whether inspection produced bounded content facts for a sample."""

    return bool(sample.visual_descriptors or sample.text_descriptors)


def summarize_groups(
    samples: list[SampleEvidence],
    *,
    max_comparison_groups: int,
) -> tuple[list[GroupDescriptorSummary], list[DescriptorComparison], list[str]]:
    grouped: dict[str, list[SampleEvidence]] = {}
    for sample in samples:
        grouped.setdefault(sample.group, []).append(sample)
    summaries: list[GroupDescriptorSummary] = []
    for group in sorted(grouped):
        inspected = [
            sample
            for sample in grouped[group]
            if sample_has_inspection_evidence(sample)
        ]
        scalar_rows = [_scalar_descriptors(sample) for sample in inspected]
        names = sorted({name for row in scalar_rows for name in row})
        means = {
            name: _number(fmean(row[name] for row in scalar_rows if name in row))
            for name in names
            if any(name in row for row in scalar_rows)
        }
        summaries.append(
            GroupDescriptorSummary(
                group=group,
                sample_count=len(grouped[group]),
                inspected_count=len(inspected),
                descriptor_means=means,
            )
        )
    comparable = [summary for summary in summaries if summary.descriptor_means]
    limited = comparable[:max_comparison_groups]
    comparison_warnings = []
    if len(comparable) > len(limited):
        comparison_warnings.append(
            f"Descriptor comparisons were limited to max_comparison_groups={max_comparison_groups}."
        )
    comparisons: list[DescriptorComparison] = []
    for left_index, left in enumerate(limited):
        for right in limited[left_index + 1 :]:
            shared = sorted(set(left.descriptor_means) & set(right.descriptor_means))
            comparisons.append(
                DescriptorComparison(
                    group_a=left.group,
                    group_b=right.group,
                    sample_count_a=left.inspected_count,
                    sample_count_b=right.inspected_count,
                    mean_differences_a_minus_b={
                        name: _number(
                            left.descriptor_means[name] - right.descriptor_means[name]
                        )
                        for name in shared
                    },
                )
            )
    return summaries, comparisons, comparison_warnings


def _safe_label(value: Any, *, limit: int = 48) -> str:
    # Dataset metadata is untrusted and may contain control characters or glyphs
    # unsupported by Pillow's bundled default font.  Keep labels inert, bounded,
    # and renderable without loading an external font.
    text = " ".join(str(value).split()).encode("ascii", "replace").decode("ascii")
    return text[: limit - 3] + "..." if len(text) > limit else text


def create_contact_sheet(
    dataset_root: str | Path,
    samples: list[SampleEvidence],
    destination: Path,
    options: QualitativePreparationOptions,
    *,
    label_column: str,
) -> tuple[bool, list[str]]:
    candidates = [
        sample
        for sample in samples
        if sample.media_type == "image"
        and sample.sample_reference
        and sample.visual_descriptors.get("intensity")
    ][: options.max_contact_sheet_samples]
    if not candidates or not options.create_contact_sheet:
        return False, []
    columns = min(4, len(candidates))
    rows = math.ceil(len(candidates) / columns)
    padding = 8
    label_height = 54
    cell = options.thumbnail_size
    canvas = Image.new(
        "RGB",
        (
            columns * (cell + padding) + padding,
            rows * (cell + label_height + padding) + padding,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    omitted: list[str] = []
    for index, sample in enumerate(candidates):
        try:
            path = resolve_dataset_reference(str(dataset_root), sample.sample_reference)
            with Image.open(path) as image:
                image.seek(0)
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((cell, cell), Image.Resampling.LANCZOS)
                x = padding + (index % columns) * (cell + padding)
                y = padding + (index // columns) * (cell + label_height + padding)
                canvas.paste(
                    thumbnail,
                    (
                        x + (cell - thumbnail.width) // 2,
                        y + (cell - thumbnail.height) // 2,
                    ),
                )
                label = _safe_label(sample.metadata.get(label_column, "<none>"))
                draw.multiline_text(
                    (x, y + cell + 3),
                    f"id={_safe_label(sample.sample_id)}\n"
                    f"group={_safe_label(sample.group)} label={label}",
                    fill="black",
                    spacing=2,
                )
        except (OSError, ValueError, UnidentifiedImageError):
            omitted.append(sample.sample_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}-{uuid4().hex}.tmp{destination.suffix}"
    )
    try:
        canvas.save(temporary, format="PNG", optimize=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    warnings_out = (
        ["Contact sheet omitted undecodable samples: " + ", ".join(omitted)]
        if omitted
        else []
    )
    return True, warnings_out


def _artifact_locations(
    runs_root: str | Path,
    run_id: str,
    task_id: str,
    scope: str | None = None,
) -> tuple[Path, str, Path, str]:
    configured_root = Path(runs_root).resolve(strict=False)
    run_root = (configured_root / run_id).resolve(strict=False)
    relative_directory = f"qualitative/{task_id}" + (f"/{scope}" if scope else "")
    directory = (run_root / relative_directory).resolve(strict=False)
    try:
        run_root.relative_to(configured_root)
        directory.relative_to(run_root)
    except ValueError as exc:
        raise QualitativeInspectionError(
            "Qualitative artifact path escapes the approved run root."
        ) from exc
    inspection_reference = f"{relative_directory}/inspection.json"
    contact_reference = f"{relative_directory}/contact_sheet.png"
    return (
        directory / "inspection.json",
        inspection_reference,
        directory / "contact_sheet.png",
        contact_reference,
    )


def load_reusable_inspection(
    request: QualitativeRequest,
    runs_root: str | Path | None,
) -> tuple[QualitativeInspectionArtifact, str] | None:
    if runs_root is None:
        return None
    inspection_path, inspection_reference, _, _ = _artifact_locations(
        runs_root,
        request.run_id,
        request.task.task_id,
        request.inspection_scope,
    )
    if not inspection_path.is_file():
        return None
    try:
        artifact = QualitativeInspectionArtifact.model_validate_json(
            inspection_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise QualitativeInspectionError(
            "Stored qualitative inspection artifact is incompatible."
        ) from exc
    if (
        artifact.run_id != request.run_id
        or artifact.request_fingerprint != _request_fingerprint(request)
        or artifact.candidate_id != request.candidate.candidate_id
        or artifact.task_id != request.task.task_id
        or artifact.seed != request.preparation.seed
        or artifact.sampling_strategy != request.preparation.sampling_strategy
        or artifact.metadata_file != request.metadata_file
        or artifact.preparation != request.preparation.model_dump(mode="json")
        or artifact.request_limitations != request.limitations
    ):
        raise QualitativeInspectionError(
            "Stored qualitative inspection artifact does not match the request."
        )
    run_root = (Path(runs_root).resolve(strict=False) / request.run_id).resolve(
        strict=False
    )
    for reference in artifact.artifact_references:
        path = (run_root / reference).resolve(strict=False)
        try:
            path.relative_to(run_root)
        except ValueError as exc:
            raise QualitativeInspectionError(
                "Stored qualitative artifact reference escapes the run root."
            ) from exc
        if not path.is_file():
            raise QualitativeInspectionError(
                f"Stored qualitative artifact is missing: {reference}"
            )
    return artifact, inspection_reference


def create_inspection_artifact(
    request: QualitativeRequest,
    records: list[dict[str, Any]],
    *,
    group_coverage: dict[str, int],
    available_count: int,
    metadata_available: bool,
    metadata_rows_read: int,
    warnings_in: list[str],
    limitations_in: list[str],
    runs_root: str | Path | None,
) -> tuple[QualitativeInspectionArtifact, str | None]:
    samples, sample_warnings, sample_limitations = inspect_selected_records(
        request.task.inputs.dataset_reference,
        records,
        request.preparation,
    )
    group_summaries, comparisons, comparison_warnings = summarize_groups(
        samples,
        max_comparison_groups=request.preparation.max_comparison_groups,
    )
    references: list[str] = []
    inspection_reference: str | None = None
    contact_warnings: list[str] = []
    if runs_root is not None:
        (
            inspection_path,
            inspection_reference,
            contact_path,
            contact_reference,
        ) = _artifact_locations(runs_root, request.run_id, request.task.task_id, request.inspection_scope)
        created, contact_warnings = create_contact_sheet(
            request.task.inputs.dataset_reference,
            samples,
            contact_path,
            request.preparation,
            label_column=request.task.inputs.label_column,
        )
        if created:
            references.append(contact_reference)
    coverage = InspectionCoverage(
        requested_sample_count=request.preparation.max_evidence_records,
        selected_count=len(samples),
        available_count=available_count,
        inspected_count=sum(
            sample_has_inspection_evidence(sample) for sample in samples
        ),
        failed_count=sum(sample.inspection_status == "failed" for sample in samples),
        group_coverage=group_coverage,
    )
    limitations = _unique([*limitations_in, *sample_limitations])
    warnings_out = _unique(
        [*warnings_in, *sample_warnings, *comparison_warnings, *contact_warnings]
    )
    artifact = QualitativeInspectionArtifact(
        schema_version="1.0",
        run_id=request.run_id,
        candidate_id=request.candidate.candidate_id,
        task_id=request.task.task_id,
        backend="local_descriptor_v1",
        request_fingerprint=_request_fingerprint(request),
        sampling_strategy=request.preparation.sampling_strategy,
        seed=request.preparation.seed,
        metadata_file=request.metadata_file,
        metadata_available=metadata_available,
        metadata_rows_read=metadata_rows_read,
        preparation=request.preparation.model_dump(mode="json"),
        request_limitations=request.limitations,
        coverage=coverage,
        selected_samples=samples,
        group_summaries=group_summaries,
        descriptor_comparisons=comparisons,
        descriptor_definitions=DESCRIPTOR_DEFINITIONS,
        warnings=warnings_out,
        limitations=limitations,
        artifact_references=references,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    if runs_root is not None:
        atomic_write_json(
            inspection_path,
            artifact.model_dump(mode="json"),
            error_type=QualitativeInspectionError,
        )
    return artifact, inspection_reference


__all__ = [
    "DESCRIPTOR_DEFINITIONS",
    "QualitativeInspectionError",
    "SUPPORTED_IMAGE_FORMATS",
    "SUPPORTED_TEXT_SUFFIXES",
    "create_inspection_artifact",
    "inspect_sample_record",
    "inspect_selected_records",
    "load_reusable_inspection",
    "sample_has_inspection_evidence",
    "summarize_groups",
]
