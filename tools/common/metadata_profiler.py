from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MAX_METADATA_ROWS = 100_000


@dataclass
class LoadedMetadata:
    path: str
    dataframe: pd.DataFrame
    row_limit: int
    truncated: bool


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    non_null_count: int
    missing_count: int
    unique_count: int
    sample_values: list[Any]


@dataclass
class MetadataProfile:
    path: str
    exists: bool
    num_rows: int
    num_columns: int
    columns: list[dict]
    rows_profiled: int
    row_limit: int
    truncated: bool


def _normalize_value(value: Any) -> Any:
    """Convert pandas/numpy scalar values to JSON-friendly values."""

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass

    return value


def _validate_row_limit(max_rows: int) -> None:
    if (
        not isinstance(max_rows, int)
        or isinstance(max_rows, bool)
        or max_rows < 1
    ):
        raise ValueError("max_rows must be at least 1.")


def _metadata_path(metadata_path: str) -> Path:
    if not isinstance(metadata_path, str) or not metadata_path.strip():
        raise ValueError("Metadata path must be a non-empty path.")

    return Path(metadata_path)


def _validate_readable_csv(metadata_path: str) -> Path:
    path = _metadata_path(metadata_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    path = path.resolve(strict=True)

    if not path.is_file():
        raise ValueError(
            f"Metadata path is not a file: {metadata_path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported metadata format: {path.suffix}"
        )

    return path


def load_metadata(
    metadata_path: str,
    max_rows: int = DEFAULT_MAX_METADATA_ROWS,
) -> LoadedMetadata:
    """Read a bounded CSV prefix once for profiling and row selection."""

    _validate_row_limit(max_rows)
    path = _validate_readable_csv(metadata_path)

    # One look-ahead row tells callers whether the bounded view was
    # truncated. The returned dataframe itself never exceeds the budget.
    dataframe = pd.read_csv(
        path,
        nrows=max_rows + 1,
    )
    truncated = len(dataframe) > max_rows

    if truncated:
        dataframe = dataframe.iloc[:max_rows].copy()

    # A RangeIndex makes the original logical CSV record index explicit and
    # stable for the selection identifiers built downstream.
    dataframe = dataframe.reset_index(drop=True)

    return LoadedMetadata(
        path=str(path),
        dataframe=dataframe,
        row_limit=max_rows,
        truncated=truncated,
    )


def profile_loaded_metadata(
    loaded: LoadedMetadata,
    sample_values_per_column: int = 5,
) -> dict:
    """Profile a previously loaded bounded metadata dataframe."""

    if (
        not isinstance(sample_values_per_column, int)
        or isinstance(sample_values_per_column, bool)
        or sample_values_per_column < 0
    ):
        raise ValueError(
            "sample_values_per_column must be non-negative."
        )

    dataframe = loaded.dataframe
    column_profiles = []

    for column_name in dataframe.columns:
        series = dataframe[column_name]
        unique_values = series.dropna().drop_duplicates()
        sample_values = [
            _normalize_value(value)
            for value in unique_values.iloc[
                :sample_values_per_column
            ]
        ]

        profile = ColumnProfile(
            name=str(column_name),
            dtype=str(series.dtype),
            non_null_count=int(series.notna().sum()),
            missing_count=int(series.isna().sum()),
            unique_count=int(series.nunique(dropna=True)),
            sample_values=sample_values,
        )
        column_profiles.append(asdict(profile))

    result = MetadataProfile(
        path=loaded.path,
        exists=True,
        num_rows=int(len(dataframe)),
        num_columns=int(len(dataframe.columns)),
        columns=column_profiles,
        rows_profiled=int(len(dataframe)),
        row_limit=loaded.row_limit,
        truncated=loaded.truncated,
    )
    return asdict(result)


def profile_metadata(
    metadata_path: str,
    sample_values_per_column: int = 5,
    max_rows: int = DEFAULT_MAX_METADATA_ROWS,
) -> dict:
    """
    Inspect a bounded prefix of CSV metadata deterministically.

    ``num_rows`` and all column statistics describe the bounded view. When
    ``truncated`` is true, they are not full-file statistics.
    """

    _validate_row_limit(max_rows)
    path = _metadata_path(metadata_path)

    if not path.exists():
        result = MetadataProfile(
            path=str(path),
            exists=False,
            num_rows=0,
            num_columns=0,
            columns=[],
            rows_profiled=0,
            row_limit=max_rows,
            truncated=False,
        )
        return asdict(result)

    loaded = load_metadata(
        metadata_path=metadata_path,
        max_rows=max_rows,
    )
    return profile_loaded_metadata(
        loaded=loaded,
        sample_values_per_column=sample_values_per_column,
    )
