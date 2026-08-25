from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


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


def _normalize_value(value: Any) -> Any:
    """
    Convert pandas / numpy scalar values into simple
    JSON-friendly Python values when possible.
    """

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass

    return value


def profile_metadata(
    metadata_path: str,
    sample_values_per_column: int = 5,
) -> dict:
    """
    Inspect basic CSV metadata structure.

    This function performs deterministic descriptive inspection only.
    It does not perform shortcut validation or statistical testing.
    """

    path = Path(metadata_path)

    if not path.exists():
        result = MetadataProfile(
            path=str(path),
            exists=False,
            num_rows=0,
            num_columns=0,
            columns=[],
        )
        return asdict(result)

    if not path.is_file():
        raise ValueError(
            f"Metadata path is not a file: {metadata_path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported metadata format: {path.suffix}"
        )

    df = pd.read_csv(path)

    column_profiles = []

    for column_name in df.columns:
        series = df[column_name]

        sample_values = [
            _normalize_value(value)
            for value in series.dropna().unique()[
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
        path=str(path),
        exists=True,
        num_rows=int(len(df)),
        num_columns=int(len(df.columns)),
        columns=column_profiles,
    )

    return asdict(result)