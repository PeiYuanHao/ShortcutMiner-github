from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from shortcutminer.protocol.identify import IdentifyRequest
from tools.common.dataset_inspector import (
    DEFAULT_MAX_DATASET_ENTRIES,
    inspect_dataset,
)
from tools.common.metadata_profiler import (
    DEFAULT_MAX_METADATA_ROWS,
    load_metadata,
    profile_loaded_metadata,
)
from tools.common.sample_selector import select_loaded_metadata


class IdentifyContextError(RuntimeError):
    """Raised when the dataset root prevents context construction."""


class IdentifyDiscoveryContext(BaseModel):
    protocol_version: str
    run_id: str
    seed: int

    dataset_name: str
    task_type: str
    label_column: str

    dataset_inspection: dict
    metadata_profile: dict | None = None
    metadata_available: bool = False

    # Kept for wire compatibility. Every item explicitly identifies itself
    # as a selected metadata row; this is not a sample-content inspection.
    selected_samples: list[dict] = Field(default_factory=list)
    metadata_rows_selected: int = Field(default=0, ge=0)
    samples_inspected: int = Field(default=0, ge=0)

    limitations: list[str] = Field(default_factory=list)


def _resolve_metadata_path(
    dataset_root: str,
    metadata_file: str,
) -> Path:
    root_path = Path(dataset_root).resolve(strict=True)
    candidate = root_path / metadata_file

    try:
        resolved_candidate = candidate.resolve(strict=False)
    except OSError as exc:
        raise ValueError(
            f"Could not resolve configured metadata path: {exc}"
        ) from exc

    try:
        resolved_candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(
            "Configured metadata path resolves outside the dataset root."
        ) from exc

    return resolved_candidate


def _budget_value(
    request: IdentifyRequest,
    field_name: str,
    fallback: int,
) -> int:
    value = getattr(request.budget, field_name, fallback)

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise IdentifyContextError(
            f"Invalid Identify budget {field_name}: {value!r}."
        )

    return value


def _append_dataset_limitations(
    dataset_info: dict,
    limitations: list[str],
) -> None:
    if dataset_info.get("truncated"):
        limitations.append(
            "Dataset inspection reached the configured "
            f"max_dataset_entries={dataset_info['max_entries']}; "
            "counts describe only the scanned prefix."
        )

    for error in dataset_info.get("scan_errors", []):
        limitations.append(f"Dataset inspection limitation: {error}")


def _base_context(
    request: IdentifyRequest,
    seed: int,
    dataset_info: dict,
    metadata_profile: dict | None,
    metadata_available: bool,
    selected_samples: list[dict],
    limitations: list[str],
) -> IdentifyDiscoveryContext:
    return IdentifyDiscoveryContext(
        protocol_version=request.protocol_version,
        run_id=request.run_id,
        seed=seed,
        dataset_name=request.dataset.name,
        task_type=request.task.type,
        label_column=request.task.label_column,
        dataset_inspection=dataset_info,
        metadata_profile=metadata_profile,
        metadata_available=metadata_available,
        selected_samples=selected_samples,
        metadata_rows_selected=len(selected_samples),
        # No tool in this context builder opens or examines sample content.
        samples_inspected=0,
        limitations=limitations,
    )


def build_identify_context(
    request: IdentifyRequest,
    seed: int,
) -> IdentifyDiscoveryContext:
    """
    Build deterministic, bounded discovery evidence for Identify.

    A missing or invalid dataset root is unrecoverable. Metadata failures are
    recorded as limitations so dataset-structure discovery can continue.
    """

    max_dataset_entries = _budget_value(
        request,
        "max_dataset_entries",
        DEFAULT_MAX_DATASET_ENTRIES,
    )
    max_metadata_rows = _budget_value(
        request,
        "max_metadata_rows",
        DEFAULT_MAX_METADATA_ROWS,
    )

    try:
        dataset_info = inspect_dataset(
            request.dataset.root,
            max_entries=max_dataset_entries,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        raise IdentifyContextError(
            f"Dataset root is unavailable: {exc}"
        ) from exc

    limitations: list[str] = []
    _append_dataset_limitations(dataset_info, limitations)

    metadata_profile = None
    metadata_available = False
    selected_samples: list[dict] = []

    if request.dataset.metadata_file is None:
        limitations.append("No metadata file was configured.")
        return _base_context(
            request=request,
            seed=seed,
            dataset_info=dataset_info,
            metadata_profile=metadata_profile,
            metadata_available=metadata_available,
            selected_samples=selected_samples,
            limitations=limitations,
        )

    try:
        metadata_path = _resolve_metadata_path(
            dataset_root=dataset_info["root"],
            metadata_file=request.dataset.metadata_file,
        )
    except (ValueError, OSError) as exc:
        limitations.append(
            f"Configured metadata file was rejected: {exc}"
        )
        return _base_context(
            request=request,
            seed=seed,
            dataset_info=dataset_info,
            metadata_profile=metadata_profile,
            metadata_available=metadata_available,
            selected_samples=selected_samples,
            limitations=limitations,
        )

    try:
        loaded_metadata = load_metadata(
            metadata_path=str(metadata_path),
            max_rows=max_metadata_rows,
        )
    except (
        FileNotFoundError,
        PermissionError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
        OSError,
    ) as exc:
        limitations.append(
            f"Configured metadata file is unavailable or invalid: {exc}"
        )
        return _base_context(
            request=request,
            seed=seed,
            dataset_info=dataset_info,
            metadata_profile=metadata_profile,
            metadata_available=metadata_available,
            selected_samples=selected_samples,
            limitations=limitations,
        )

    metadata_available = True

    try:
        metadata_profile = profile_loaded_metadata(loaded_metadata)
    except (TypeError, ValueError) as exc:
        limitations.append(f"Metadata profiling failed: {exc}")

    if loaded_metadata.truncated:
        limitations.append(
            "Metadata reading reached the configured "
            f"max_metadata_rows={max_metadata_rows}; profiling and "
            "selection use only the bounded prefix."
        )

    try:
        selected_samples = select_loaded_metadata(
            loaded=loaded_metadata,
            max_samples=request.budget.max_samples,
            seed=seed,
            strategy="class_balanced",
            label_column=request.task.label_column,
        )
    except (TypeError, ValueError) as exc:
        limitations.append(f"Class-balanced sampling failed: {exc}")

        try:
            selected_samples = select_loaded_metadata(
                loaded=loaded_metadata,
                max_samples=request.budget.max_samples,
                seed=seed,
                strategy="random",
            )
            limitations.append("Random sampling was used as fallback.")
        except (TypeError, ValueError) as fallback_exc:
            limitations.append(
                f"Metadata row selection failed: {fallback_exc}"
            )

    if selected_samples:
        limitations.append(
            "Selected records are metadata rows only; no dataset sample "
            "content was inspected."
        )

    return _base_context(
        request=request,
        seed=seed,
        dataset_info=dataset_info,
        metadata_profile=metadata_profile,
        metadata_available=metadata_available,
        selected_samples=selected_samples,
        limitations=limitations,
    )
