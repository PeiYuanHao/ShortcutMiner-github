from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from tools.common.metadata_profiler import LoadedMetadata, _normalize_value


def normalize_dataset_reference(value: Any) -> str | None:
    """Return a portable dataset-relative reference or ``None``."""

    if value is None:
        return None

    reference = str(value).strip().replace("\\", "/")
    if not reference:
        return None

    posix_path = PurePosixPath(reference)
    windows_path = PureWindowsPath(reference)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {"", ".", ".."} for part in reference.split("/"))
    ):
        raise ValueError("Sample path must be relative to the dataset root.")

    return posix_path.as_posix()


def resolve_dataset_reference(dataset_root: str, reference: str) -> Path:
    """Resolve one sample reference without permitting dataset escape."""

    normalized = normalize_dataset_reference(reference)
    if normalized is None:
        raise ValueError("Sample path is empty.")

    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    candidate = (root / normalized).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Sample path resolves outside the dataset root.") from exc

    return candidate


def inspect_metadata_fields(
    loaded: LoadedMetadata,
    fields: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return bounded values for requested fields and explicit omissions."""

    missing = [field for field in fields if field not in loaded.dataframe.columns]
    available = [field for field in fields if field in loaded.dataframe.columns]
    records: list[dict[str, Any]] = []

    for row_index, row in loaded.dataframe[available].iterrows():
        records.append(
            {
                "metadata_row_index": int(row_index),
                "record": {
                    field: _normalize_value(row[field]) for field in available
                },
            }
        )

    return records, missing
