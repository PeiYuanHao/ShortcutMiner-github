"""Deterministic, bounded profiling for user-provided local datasets."""

from pathlib import Path
from random import Random
from typing import Any

import pandas as pd

from shortcutminer.protocol.dataset_profile import (
    DatasetInspectionBudget,
    DatasetMetadataProfile,
    DatasetProfile,
    DatasetProfileDataset,
    DatasetProfileMetadata,
    DatasetProfileTask,
    DatasetStructureProfile,
    LabelProfile,
    MetadataColumnProfile,
    SampleReference,
    SplitProfile,
)
from shortcutminer.protocol.identify import IdentifyRequest
from tools.common.dataset_inspector import inspect_dataset
from tools.common.metadata_profiler import (
    LoadedMetadata,
    load_metadata,
    profile_loaded_metadata,
)


class DatasetProfileError(RuntimeError):
    """Base error for deterministic dataset intake."""


class DatasetRootNotFound(DatasetProfileError):
    pass


class DatasetRootInvalid(DatasetProfileError):
    pass


class MetadataNotFound(DatasetProfileError):
    pass


class MetadataParseError(DatasetProfileError):
    pass


class LabelColumnNotFound(DatasetProfileError):
    pass


class UnsupportedDatasetStructure(DatasetProfileError):
    pass


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TABULAR_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet", ".tsv"}
TEXT_EXTENSIONS = {".md", ".text", ".txt"}


def _primitive_type(dtype: str) -> str:
    lowered = dtype.casefold()
    if "bool" in lowered:
        return "boolean"
    if "int" in lowered:
        return "integer"
    if "float" in lowered or "double" in lowered or "decimal" in lowered:
        return "number"
    if any(token in lowered for token in ("object", "string", "category")):
        return "string"
    return "unknown"


def _display_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "<missing>"
    return str(value)


def _distribution(series: pd.Series) -> dict[str, int]:
    counts = series.dropna().value_counts(sort=False)
    items = [(_display_value(value), int(count)) for value, count in counts.items()]
    return dict(sorted(items, key=lambda item: (item[0].casefold(), item[0])))


def _resolve_metadata(root: Path, metadata_file: str) -> Path:
    try:
        candidate = (root / metadata_file).resolve(strict=False)
        candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DatasetRootInvalid(
            "Configured metadata path resolves outside the dataset root."
        ) from exc
    return candidate


def _modality(extension_counts: dict[str, int], metadata_file: str | None) -> str:
    counts = dict(extension_counts)
    if metadata_file:
        suffix = Path(metadata_file).suffix.lower() or "<none>"
        if counts.get(suffix, 0) > 0:
            counts[suffix] -= 1

    categories: set[str] = set()
    for extension, count in counts.items():
        if count <= 0:
            continue
        if extension in IMAGE_EXTENSIONS:
            categories.add("image")
        elif extension in TABULAR_EXTENSIONS:
            categories.add("tabular")
        elif extension in TEXT_EXTENSIONS:
            categories.add("text")

    if len(categories) > 1:
        return "mixed"
    if len(categories) == 1:
        return next(iter(categories))
    return "unknown"


def _sample_references(
    files: list[str],
    *,
    metadata_file: str | None,
    max_samples: int,
    seed: int,
) -> list[SampleReference]:
    eligible = sorted(
        reference
        for reference in files
        if reference != metadata_file
    )
    random = Random(seed)
    random.shuffle(eligible)
    chosen = sorted(eligible[:max_samples])
    return [
        SampleReference(
            reference=reference,
            kind="file",
            extension=Path(reference).suffix.lower(),
        )
        for reference in chosen
    ]


def _unavailable_metadata(
    request: IdentifyRequest,
) -> DatasetMetadataProfile:
    return DatasetMetadataProfile(
        available=False,
        file=request.dataset.metadata_file,
        row_count=None,
        rows_profiled=0,
        row_limit=request.budget.max_metadata_rows,
        truncated=False,
        column_names=[],
        columns=[],
    )


def _profile_metadata(
    request: IdentifyRequest,
    root: Path,
    warnings: list[str],
    limitations: list[str],
) -> tuple[DatasetMetadataProfile, LabelProfile, SplitProfile]:
    metadata_file = request.dataset.metadata_file
    if metadata_file is None:
        limitations.append(
            "label_source_unavailable: no metadata file was configured and "
            "the current profiler does not infer labels from directory names."
        )
        return (
            _unavailable_metadata(request),
            LabelProfile(
                column=request.task.label_column,
                validity="source_unavailable",
                distribution={},
                distinct_count=0,
                missing_count=0,
            ),
            SplitProfile(detected_field=None, distribution={}),
        )

    metadata_path = _resolve_metadata(root, metadata_file)
    if not metadata_path.is_file():
        warnings.append(
            f"MetadataNotFound: configured metadata file is unavailable: {metadata_file}"
        )
        limitations.append("Configured metadata could not be inspected.")
        return (
            _unavailable_metadata(request),
            LabelProfile(
                column=request.task.label_column,
                validity="source_unavailable",
                distribution={},
                distinct_count=0,
                missing_count=0,
            ),
            SplitProfile(detected_field=None, distribution={}),
        )

    try:
        loaded = load_metadata(
            str(metadata_path),
            max_rows=request.budget.max_metadata_rows,
        )
        raw_profile = profile_loaded_metadata(loaded)
    except (PermissionError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, OSError) as exc:
        warnings.append(f"MetadataParseError: {type(exc).__name__}: {exc}")
        limitations.append("Configured metadata could not be parsed by the bounded CSV reader.")
        return (
            _unavailable_metadata(request),
            LabelProfile(
                column=request.task.label_column,
                validity="source_unavailable",
                distribution={},
                distinct_count=0,
                missing_count=0,
            ),
            SplitProfile(detected_field=None, distribution={}),
        )

    dataframe = loaded.dataframe
    if request.task.label_column not in dataframe.columns:
        raise LabelColumnNotFound(
            f"Label column not found in metadata: {request.task.label_column}"
        )

    rows_profiled = int(raw_profile["rows_profiled"])
    columns = [
        MetadataColumnProfile(
            name=column["name"],
            primitive_type=_primitive_type(column["dtype"]),
            source_type=column["dtype"],
            non_null_count=column["non_null_count"],
            missing_count=column["missing_count"],
            missing_rate=(column["missing_count"] / rows_profiled if rows_profiled else 0.0),
            distinct_count=column["unique_count"],
            example_values=column["sample_values"],
        )
        for column in raw_profile["columns"]
    ]
    metadata = DatasetMetadataProfile(
        available=True,
        file=metadata_file,
        row_count=rows_profiled,
        rows_profiled=rows_profiled,
        row_limit=request.budget.max_metadata_rows,
        truncated=loaded.truncated,
        column_names=[str(name) for name in dataframe.columns],
        columns=columns,
    )
    label_series = dataframe[request.task.label_column]
    labels = LabelProfile(
        column=request.task.label_column,
        validity="valid",
        distribution=_distribution(label_series),
        distinct_count=int(label_series.nunique(dropna=True)),
        missing_count=int(label_series.isna().sum()),
    )

    split_columns = sorted(
        (str(name) for name in dataframe.columns if str(name).casefold() == "split"),
        key=lambda name: (name.casefold(), name),
    )
    split_field = split_columns[0] if split_columns else None
    splits = SplitProfile(
        detected_field=split_field,
        distribution=(
            _distribution(dataframe[split_field]) if split_field is not None else {}
        ),
    )
    if loaded.truncated:
        limitations.append(
            "Metadata inspection reached max_metadata_rows="
            f"{request.budget.max_metadata_rows}; statistics describe only the bounded prefix."
        )
    return metadata, labels, splits


class DatasetProfiler:
    """Build factual profiles without network, models, or dataset adapters."""

    def profile(self, request: IdentifyRequest, *, seed: int) -> DatasetProfile:
        try:
            root = Path(request.dataset.root).resolve(strict=True)
        except FileNotFoundError as exc:
            raise DatasetRootNotFound(
                f"Dataset root not found: {request.dataset.root}"
            ) from exc
        except (OSError, RuntimeError) as exc:
            raise DatasetRootInvalid(
                f"Dataset root is invalid: {request.dataset.root}"
            ) from exc
        if not root.is_dir():
            raise DatasetRootInvalid(
                f"Dataset root is not a directory: {request.dataset.root}"
            )

        try:
            inspection = inspect_dataset(
                str(root),
                max_entries=request.budget.max_dataset_entries,
            )
        except FileNotFoundError as exc:
            raise DatasetRootNotFound(str(exc)) from exc
        except (NotADirectoryError, ValueError, OSError) as exc:
            raise DatasetRootInvalid(str(exc)) from exc

        warnings = [
            f"Dataset inspection warning: {message}"
            for message in inspection["scan_errors"]
        ]
        limitations: list[str] = []
        if inspection["truncated"]:
            limitations.append(
                "Dataset inspection reached max_dataset_entries="
                f"{request.budget.max_dataset_entries}; structure counts describe only the scanned prefix."
            )

        metadata, labels, splits = _profile_metadata(
            request,
            root,
            warnings,
            limitations,
        )
        references = _sample_references(
            inspection["discovered_files"],
            metadata_file=request.dataset.metadata_file,
            max_samples=request.budget.max_samples,
            seed=seed,
        )
        if not references:
            limitations.append(
                "No representative non-metadata file references were available in the bounded scan."
            )

        return DatasetProfile(
            schema_version="1.0",
            dataset=DatasetProfileDataset(
                name=request.dataset.name,
                root_reference=str(root),
                metadata_file=request.dataset.metadata_file,
                metadata_present=metadata.available,
            ),
            task=DatasetProfileTask(
                task_type=request.task.type,
                label_column=request.task.label_column,
                intended_signal=request.task.intended_signal,
            ),
            structure=DatasetStructureProfile(
                files_scanned=inspection["num_files"],
                directories_scanned=inspection["num_directories"],
                entries_scanned=inspection["entries_scanned"],
                max_entries=inspection["max_entries"],
                truncated=inspection["truncated"],
                file_extension_distribution=inspection["file_extension_counts"],
                top_level_entries=inspection["top_level_entries"],
                max_depth_observed=inspection["max_depth_observed"],
                modality=_modality(
                    inspection["file_extension_counts"],
                    request.dataset.metadata_file,
                ),
            ),
            metadata=metadata,
            labels=labels,
            splits=splits,
            sample_references=references,
            warnings=warnings,
            limitations=limitations,
            profile_metadata=DatasetProfileMetadata(
                deterministic_seed=seed,
                inspection_budget=DatasetInspectionBudget(
                    max_dataset_entries=request.budget.max_dataset_entries,
                    max_metadata_rows=request.budget.max_metadata_rows,
                    max_samples=request.budget.max_samples,
                ),
            ),
        )


def build_dataset_profile(request: IdentifyRequest, *, seed: int) -> DatasetProfile:
    return DatasetProfiler().profile(request, seed=seed)


__all__ = [
    "DatasetProfileError",
    "DatasetProfiler",
    "DatasetRootInvalid",
    "DatasetRootNotFound",
    "LabelColumnNotFound",
    "MetadataNotFound",
    "MetadataParseError",
    "UnsupportedDatasetStructure",
    "build_dataset_profile",
]
