"""Bounded deterministic image-content preparation for Identify discovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from random import Random
from statistics import fmean, pstdev
from typing import Any, Callable

from shortcutminer.inspection.image import (
    DESCRIPTOR_DEFINITION_VERSION,
    DESCRIPTOR_DEFINITIONS,
    SUPPORTED_IMAGE_FORMATS,
    ImageInspectionResult,
    flatten_image_descriptors,
    inspect_image_file,
)
from shortcutminer.inspection.semantic import SemanticVisualInspector
from shortcutminer.protocol.dataset_profile import DatasetProfile
from shortcutminer.protocol.identify import IdentifyRequest
from shortcutminer.protocol.visual_discovery import (
    SemanticDiscoveryStatus,
    SemanticInspectionRequest,
    SemanticInspectionSample,
    SemanticObservation,
    VisualCoverage,
    VisualDescriptorDifference,
    VisualDescriptorSet,
    VisualDiscoveryOptions,
    VisualDiscoveryProfile,
    VisualGroupSummary,
    VisualSampleObservation,
)
from shortcutminer.storage.visual_discovery import (
    VISUAL_DISCOVERY_REFERENCE,
    load_visual_discovery_profile,
    save_visual_discovery_profile,
)
from tools.common.dataset_inspector import inspect_dataset
from tools.common.metadata_profiler import load_metadata
from tools.qualitative.evidence_builder import build_evidence_manifest
from tools.qualitative.image_sampler import select_qualitative_samples
from tools.qualitative.metadata_inspector import resolve_dataset_reference


class VisualDiscoveryError(RuntimeError):
    """Raised when the deterministic visual discovery boundary is invalid."""


ImageInspector = Callable[..., ImageInspectionResult]


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _metadata_path(request: IdentifyRequest) -> Path:
    if request.dataset.metadata_file is None:
        raise VisualDiscoveryError("No metadata file is configured.")
    root = Path(request.dataset.root).resolve(strict=True)
    path = (root / request.dataset.metadata_file).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise VisualDiscoveryError("Metadata path escapes the dataset root.") from exc
    return path


def _metadata_records(
    request: IdentifyRequest,
    profile: DatasetProfile,
    options: VisualDiscoveryOptions,
) -> tuple[list[dict[str, Any]], str, list[str], bool]:
    limitations: list[str] = []
    loaded = load_metadata(
        str(_metadata_path(request)),
        max_rows=request.budget.max_metadata_rows,
    )
    label_available = (
        profile.labels.validity == "valid"
        and request.task.label_column in loaded.dataframe.columns
    )
    if options.sampling_strategy in {"class_balanced", "class_representative_v1"} and not label_available:
        raise VisualDiscoveryError(
            "class_balanced visual sampling requires an available label column"
        )
    strategy = (
        "class_balanced"
        if options.sampling_strategy in {"auto", "class_balanced", "class_representative_v1"} and label_available
        else "random"
    )
    if not label_available:
        limitations.append(
            "label_grouping_unavailable: visual samples have no reliable metadata label source."
        )
    group_columns = [request.task.label_column] if label_available else []
    split_column = options.split_column or profile.splits.detected_field
    metadata_fields = list(group_columns)
    if split_column is not None and split_column not in metadata_fields:
        metadata_fields.append(split_column)
    selected, group_coverage = select_qualitative_samples(
        loaded,
        max_samples=options.max_samples,
        seed=profile.profile_metadata.deterministic_seed,
        strategy="group_balanced" if strategy == "class_balanced" else "random",
        group_columns=group_columns,
        metadata_fields=metadata_fields,
        sample_id_column=options.sample_id_column,
        sample_path_column=options.sample_path_column,
        split_column=split_column,
        split=options.selected_split,
        max_groups=options.max_groups,
    )
    manifest, manifest_limitations = build_evidence_manifest(
        request.dataset.root, selected
    )
    limitations.extend(manifest_limitations)
    groups_omitted = int(group_coverage.get("groups_omitted", 0))
    if groups_omitted:
        limitations.append(
            f"Visual grouping omitted {groups_omitted} high-cardinality groups under max_groups={options.max_groups}."
        )
    truncated = loaded.truncated or len(loaded.dataframe) > len(selected)
    return manifest, strategy, limitations, truncated


def _scan_records(
    request: IdentifyRequest,
    options: VisualDiscoveryOptions,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], str, list[str], bool]:
    inspection = inspect_dataset(
        request.dataset.root,
        max_entries=options.max_scan_candidates,
    )
    eligible = sorted(
        reference
        for reference in inspection["discovered_files"]
        if Path(reference).suffix.casefold() in SUPPORTED_IMAGE_FORMATS
    )
    shuffled = list(eligible)
    Random(seed).shuffle(shuffled)
    chosen = sorted(shuffled[: options.max_samples])
    records = [
        {
            "sample_id": f"visual-sample-{index:04d}",
            "metadata_row_index": None,
            "group": "ungrouped",
            "group_values": {},
            "sample_reference": reference,
            "metadata": {},
            "selection_reason": "random",
        }
        for index, reference in enumerate(chosen, 1)
    ]
    manifest, manifest_limitations = build_evidence_manifest(
        request.dataset.root, records
    )
    limitations = [
        "label_grouping_unavailable: bounded directory-scan samples are not mapped to an explicit label source.",
        *manifest_limitations,
    ]
    if not eligible:
        limitations.append("No supported image files were found in the bounded dataset scan.")
    truncated = bool(inspection["truncated"] or len(eligible) > len(chosen))
    return manifest, "random", limitations, truncated


def _selected_content_identity(
    dataset_root: str,
    records: list[dict[str, Any]],
    *,
    max_file_bytes: int,
    max_total_bytes: int,
) -> list[dict[str, Any]]:
    identity = []
    consumed = 0
    for record in records:
        reference = record.get("sample_reference")
        entry: dict[str, Any] = {"reference": reference}
        if not reference:
            entry["status"] = "missing_reference"
            identity.append(entry)
            continue
        try:
            path = resolve_dataset_reference(dataset_root, reference)
            size = path.stat().st_size
            entry["size"] = size
            if size <= max_file_bytes and consumed + size <= max_total_bytes:
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                consumed += size
            elif size <= max_file_bytes:
                entry["status"] = "over_total_identity_budget"
            else:
                entry["status"] = "over_file_budget"
        except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
            entry["status"] = type(exc).__name__
        identity.append(entry)
    return identity


def _dataset_fingerprint(
    request: IdentifyRequest,
    profile: DatasetProfile,
    records: list[dict[str, Any]],
    options: VisualDiscoveryOptions,
) -> str:
    profile_data = profile.model_dump(mode="json")
    profile_data["dataset"].pop("root_reference", None)
    return _canonical_hash({
        "dataset_profile": profile_data,
        "selected_content": _selected_content_identity(
            request.dataset.root,
            records,
            max_file_bytes=options.max_file_bytes,
            max_total_bytes=options.max_total_inspection_bytes,
        ),
    })


def _request_fingerprint(
    request: IdentifyRequest,
    options: VisualDiscoveryOptions,
    dataset_fingerprint: str,
    semantic_inspector: SemanticVisualInspector | None,
    *,
    seed: int,
) -> str:
    return _canonical_hash({
        "run_id": request.run_id,
        "dataset_name": request.dataset.name,
        "task": request.task.model_dump(mode="json"),
        "seed": seed,
        "dataset_fingerprint": dataset_fingerprint,
        "options": options.model_dump(mode="json"),
        "descriptor_definition_version": DESCRIPTOR_DEFINITION_VERSION,
        "semantic_provider": (
            {
                "provider": semantic_inspector.provider_name,
                "version": semantic_inspector.version,
                "runtime_kind": semantic_inspector.runtime_kind,
                "model": semantic_inspector.model_name,
            }
            if semantic_inspector is not None and options.semantic_enabled
            else None
        ),
    })


def _group_summaries(
    observations: list[VisualSampleObservation],
) -> list[VisualGroupSummary]:
    grouped: dict[str, list[VisualSampleObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.group, []).append(observation)
    summaries = []
    for group in sorted(grouped):
        selected = grouped[group]
        decoded = [item for item in selected if item.descriptors is not None]
        names = sorted({name for item in decoded for name in item.descriptors.values})
        means = {}
        stds = {}
        for name in names:
            values = [
                float(item.descriptors.values[name])
                for item in decoded
                if name in item.descriptors.values
            ]
            if values:
                means[name] = round(fmean(values), 10)
                stds[name] = round(pstdev(values), 10)
        summaries.append(VisualGroupSummary(
            group=group,
            selected_count=len(selected),
            decoded_count=len(decoded),
            descriptor_means=means,
            descriptor_stds=stds,
        ))
    return summaries


def _descriptor_differences(
    summaries: list[VisualGroupSummary],
    *,
    limit: int,
) -> list[VisualDescriptorDifference]:
    rows: list[dict[str, Any]] = []
    comparable = [item for item in summaries if item.decoded_count]
    for left_index, left in enumerate(comparable):
        for right in comparable[left_index + 1 :]:
            for name in sorted(set(left.descriptor_means) & set(right.descriptor_means)):
                raw = float(left.descriptor_means[name] - right.descriptor_means[name])
                pooled = (
                    (float(left.descriptor_stds[name]) ** 2 + float(right.descriptor_stds[name]) ** 2)
                    / 2.0
                ) ** 0.5
                standardized = round(raw / pooled, 10) if pooled > 0 else None
                rows.append({
                    "descriptor": name,
                    "group_a": left.group,
                    "group_b": right.group,
                    "sample_count_a": left.decoded_count,
                    "sample_count_b": right.decoded_count,
                    "raw_difference_a_minus_b": round(raw, 10),
                    "absolute_difference": round(abs(raw), 10),
                    "standardized_descriptive_difference": standardized,
                    "_ranking_magnitude": (
                        abs(standardized) if standardized is not None else abs(raw)
                    ),
                })
    rows.sort(key=lambda row: (
        -row["_ranking_magnitude"],
        row["descriptor"],
        row["group_a"],
        row["group_b"],
    ))
    return [
        VisualDescriptorDifference(
            rank=index,
            **{key: value for key, value in row.items() if key != "_ranking_magnitude"},
        )
        for index, row in enumerate(rows[:limit], 1)
    ]


def _semantic_status(
    request: IdentifyRequest,
    observations: list[VisualSampleObservation],
    options: VisualDiscoveryOptions,
    inspector: SemanticVisualInspector | None,
) -> tuple[SemanticDiscoveryStatus, list[str]]:
    if not options.semantic_enabled:
        return SemanticDiscoveryStatus(
            requested=False,
            available=False,
            reason_unavailable="Semantic inspection was not requested; deterministic image descriptors remain available.",
            observations=[],
        ), []
    if inspector is None:
        return SemanticDiscoveryStatus(
            requested=True,
            available=False,
            reason_unavailable="No SemanticVisualInspector capability was injected.",
            observations=[],
        ), []
    samples = [
        SemanticInspectionSample(
            sample_id=item.sample_id,
            sample_reference=item.sample_reference,
        )
        for item in observations
        if item.decoded
    ]
    if not samples:
        return SemanticDiscoveryStatus(
            requested=True,
            available=False,
            runtime_kind=inspector.runtime_kind,
            provider=inspector.provider_name,
            model=inspector.model_name,
            version=inspector.version,
            reason_unavailable="No approved decoded image was available for semantic inspection.",
            observations=[],
        ), ["Optional semantic inspection had no approved decoded image input."]
    semantic_request = SemanticInspectionRequest(
        run_id=request.run_id,
        dataset_name=request.dataset.name,
        samples=samples,
    )
    try:
        if options.sampling_strategy == 'class_representative_v1':
            from shortcutminer.experiment.representative_discovery import inspect_representative
            returned = inspect_representative(inspector, request, observations)
        else:
            returned = inspector.inspect(dataset_root=request.dataset.root, request=semantic_request)
        parsed = [
            item if isinstance(item, SemanticObservation) else SemanticObservation.model_validate(item)
            for item in returned
        ]
        allowed = {(item.sample_id, item.sample_reference) for item in samples}
        if any((item.sample_id, item.sample_reference) not in allowed for item in parsed):
            raise VisualDiscoveryError(
                "Semantic inspector returned an observation outside the approved sample set."
            )
        return SemanticDiscoveryStatus(
            requested=True,
            available=True,
            runtime_kind=inspector.runtime_kind,
            provider=inspector.provider_name,
            model=inspector.model_name,
            version=inspector.version,
            observations=parsed,
        ), []
    except Exception as exc:
        return SemanticDiscoveryStatus(
            requested=True,
            available=False,
            runtime_kind=inspector.runtime_kind,
            provider=inspector.provider_name,
            model=inspector.model_name,
            version=inspector.version,
            reason_unavailable=f"Semantic inspection failed safely: {type(exc).__name__}.",
            observations=[],
        ), ["Optional semantic inspection failed; deterministic descriptors were retained."]


class VisualDiscoveryProfiler:
    """Produce observed visual facts only; never candidates or verdicts."""

    def __init__(self, *, image_inspector: ImageInspector = inspect_image_file) -> None:
        self._image_inspector = image_inspector

    def profile(
        self,
        request: IdentifyRequest,
        dataset_profile: DatasetProfile,
        *,
        seed: int,
        options: VisualDiscoveryOptions,
        runs_root: str | Path | None = None,
        semantic_inspector: SemanticVisualInspector | None = None,
    ) -> VisualDiscoveryProfile | None:
        if not options.enabled:
            return None
        if options.max_samples > request.budget.max_samples:
            raise VisualDiscoveryError(
                "Visual discovery max_samples exceeds IdentifyBudget.max_samples."
            )
        if dataset_profile.dataset.name != request.dataset.name:
            raise VisualDiscoveryError("DatasetProfile does not match IdentifyRequest.")
        if dataset_profile.profile_metadata.deterministic_seed != seed:
            raise VisualDiscoveryError("DatasetProfile seed does not match visual discovery seed.")
        if dataset_profile.metadata.available and options.sample_path_column:
            records, strategy, limitations, selection_truncated = _metadata_records(
                request, dataset_profile, options
            )
        else:
            records, strategy, limitations, selection_truncated = _scan_records(
                request, options, seed=seed
            )
            if dataset_profile.metadata.available and not options.sample_path_column:
                limitations.append(
                    "Metadata was available, but no explicit sample_path_column was configured; filenames were not guessed and label grouping was not performed."
                )

        dataset_fingerprint = _dataset_fingerprint(
            request, dataset_profile, records, options
        )
        request_fingerprint = _request_fingerprint(
            request,
            options,
            dataset_fingerprint,
            semantic_inspector,
            seed=seed,
        )
        if runs_root is not None:
            reusable = load_visual_discovery_profile(
                runs_root,
                request.run_id,
                expected_fingerprint=request_fingerprint,
            )
            if reusable is not None:
                return reusable

        consumed = 0
        observations: list[VisualSampleObservation] = []
        budget_truncated = False
        for record in records:
            reference = record.get("sample_reference")
            if not reference:
                continue
            available = bool(record.get("sample_available"))
            byte_size = record.get("byte_size")
            record_limitations: list[str] = []
            if not available:
                record_limitations.append(
                    "Selected image was unavailable, unsupported, or failed signature validation."
                )
                inspected = ImageInspectionResult(
                    status="failed",
                    decoded=False,
                    descriptors={},
                    byte_size=byte_size,
                    bytes_consumed=0,
                    truncated=False,
                    limitations=tuple(record_limitations),
                )
            elif consumed + int(byte_size or 0) > options.max_total_inspection_bytes:
                budget_truncated = True
                inspected = ImageInspectionResult(
                    status="limited",
                    decoded=False,
                    descriptors={},
                    byte_size=byte_size,
                    bytes_consumed=0,
                    truncated=True,
                    limitations=(
                        f"Image was not decoded because max_total_inspection_bytes={options.max_total_inspection_bytes} was reached.",
                    ),
                )
            else:
                try:
                    path = resolve_dataset_reference(request.dataset.root, reference)
                    inspected = self._image_inspector(
                        path,
                        requested_descriptors=list(options.requested_descriptors),
                        max_file_bytes=options.max_file_bytes,
                        max_pixels=options.max_pixels,
                        max_frames=options.max_frames,
                    )
                except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
                    inspected = ImageInspectionResult(
                        status="failed",
                        decoded=False,
                        descriptors={},
                        byte_size=byte_size,
                        bytes_consumed=0,
                        truncated=False,
                        limitations=(f"Image path resolution failed safely: {type(exc).__name__}.",),
                    )
                consumed += inspected.bytes_consumed
            descriptor_set = None
            if inspected.decoded:
                descriptor_set = VisualDescriptorSet(
                    definition_version=DESCRIPTOR_DEFINITION_VERSION,
                    values=flatten_image_descriptors(
                        inspected.descriptors,
                        file_size_bytes=inspected.byte_size,
                    ),
                    structured=inspected.descriptors,
                )
            observations.append(VisualSampleObservation(
                sample_id=record["sample_id"],
                sample_reference=reference,
                metadata_row_index=record.get("metadata_row_index"),
                group=record.get("group") or "ungrouped",
                group_values=record.get("group_values", {}),
                selection_reason=strategy,
                available=available,
                decoded=inspected.decoded,
                inspection_status=inspected.status,
                descriptors=descriptor_set,
                file_size_bytes=inspected.byte_size,
                bytes_inspected=inspected.bytes_consumed,
                warnings=list(inspected.warnings),
                limitations=list(inspected.limitations),
            ))

        summaries = _group_summaries(observations)
        differences = _descriptor_differences(
            summaries, limit=options.max_descriptor_differences
        )
        semantic, semantic_warnings = _semantic_status(
            request, observations, options, semantic_inspector
        )
        selected_per_group: dict[str, int] = {}
        for observation in observations:
            selected_per_group[observation.group] = (
                selected_per_group.get(observation.group, 0) + 1
            )
        decoded_count = sum(item.decoded for item in observations)
        failed_count = len(observations) - decoded_count
        if decoded_count == 0:
            limitations.append(
                "No selected image produced usable pixel descriptors; image_statistics and sample_inspection evidence are unavailable."
            )
        if semantic.available is False:
            limitations.append(
                "Semantic visual understanding is unavailable; low-level descriptors cannot reliably identify semantic objects or background context."
            )
        artifact_references = (
            [VISUAL_DISCOVERY_REFERENCE] if runs_root is not None else []
        )
        result = VisualDiscoveryProfile(
            run_id=request.run_id,
            dataset_name=request.dataset.name,
            dataset_fingerprint=dataset_fingerprint,
            request_fingerprint=request_fingerprint,
            descriptor_definition_version=DESCRIPTOR_DEFINITION_VERSION,
            seed=seed,
            sampling_strategy=strategy,
            options=options,
            coverage=VisualCoverage(
                requested_samples=options.max_samples,
                selected_samples=len(observations),
                available_samples=sum(item.available for item in observations),
                decoded_samples=decoded_count,
                failed_samples=failed_count,
                truncated=selection_truncated or budget_truncated,
                bytes_inspected=consumed,
                selected_per_group=selected_per_group,
            ),
            sample_observations=observations,
            group_summaries=summaries,
            descriptor_differences=differences,
            descriptor_definitions=DESCRIPTOR_DEFINITIONS,
            semantic=semantic,
            warnings=_unique(semantic_warnings),
            limitations=_unique(limitations),
            artifact_references=artifact_references,
        )
        if runs_root is not None:
            save_visual_discovery_profile(runs_root, result)
        return result


__all__ = ["VisualDiscoveryError", "VisualDiscoveryProfiler"]
