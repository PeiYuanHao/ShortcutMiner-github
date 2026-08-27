from random import Random
from typing import Any

from tools.common.metadata_profiler import LoadedMetadata, _normalize_value


def _display_value(value: Any) -> str:
    normalized = _normalize_value(value)
    return "<missing>" if normalized is None else str(normalized)


def _group_key(row, group_columns: list[str]) -> str:
    return " | ".join(
        f"{column}={_display_value(row[column])}" for column in group_columns
    )


def _allocate_group_balanced(
    groups: dict[str, list[int]],
    limit: int,
    seed: int,
) -> list[int]:
    keys = sorted(groups)
    shuffled: dict[str, list[int]] = {}
    for offset, key in enumerate(keys):
        indexes = list(groups[key])
        Random(seed + offset).shuffle(indexes)
        shuffled[key] = indexes

    selected: list[int] = []
    while len(selected) < limit:
        progress = False
        for key in keys:
            if shuffled[key] and len(selected) < limit:
                selected.append(shuffled[key].pop())
                progress = True
        if not progress:
            break
    return sorted(selected)


def select_qualitative_samples(
    loaded: LoadedMetadata,
    *,
    max_samples: int,
    seed: int,
    strategy: str,
    group_columns: list[str],
    metadata_fields: list[str],
    sample_id_column: str | None,
    sample_path_column: str | None,
    split_column: str | None = None,
    split: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select reproducible metadata-linked samples for qualitative review."""

    if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples < 1:
        raise ValueError("max_samples must be at least 1.")
    if strategy not in {"random", "group_balanced"}:
        raise ValueError(f"Unsupported qualitative strategy: {strategy}")

    required = list(dict.fromkeys([
        *group_columns,
        *metadata_fields,
        *([sample_id_column] if sample_id_column else []),
        *([sample_path_column] if sample_path_column else []),
        *([split_column] if split_column else []),
    ]))
    missing = [field for field in required if field not in loaded.dataframe.columns]
    if missing:
        raise ValueError("Metadata columns are unavailable: " + ", ".join(missing))

    dataframe = loaded.dataframe
    if split is not None:
        split_columns = (
            [split_column]
            if split_column is not None
            else [
                name
                for name in ("split", "dataset_split")
                if name in dataframe.columns
            ]
        )
        if not split_columns:
            raise ValueError("A split was requested but no split column is available.")
        dataframe = dataframe[
            dataframe[split_columns[0]].astype(str) == str(split)
        ]

    if dataframe.empty:
        raise ValueError("No metadata rows match the qualitative task.")

    groups: dict[str, list[int]] = {}
    for row_index, row in dataframe.iterrows():
        groups.setdefault(_group_key(row, group_columns), []).append(int(row_index))

    limit = min(max_samples, len(dataframe))
    if strategy == "random":
        indexes = [int(index) for index in dataframe.index]
        Random(seed).shuffle(indexes)
        selected_indexes = sorted(indexes[:limit])
    else:
        selected_indexes = _allocate_group_balanced(groups, limit, seed)

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    selected_coverage: dict[str, int] = {}
    for row_index in selected_indexes:
        row = dataframe.loc[row_index]
        if sample_id_column:
            raw_id = _normalize_value(row[sample_id_column])
            if raw_id is None or not str(raw_id).strip():
                raise ValueError(f"Sample ID is missing at metadata row {row_index}.")
            sample_id = str(raw_id).strip()
        else:
            sample_id = f"metadata-row-{row_index:08d}"

        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample ID in selection: {sample_id}")
        seen_ids.add(sample_id)

        group = _group_key(row, group_columns)
        selected_coverage[group] = selected_coverage.get(group, 0) + 1
        path_value = (
            _normalize_value(row[sample_path_column]) if sample_path_column else None
        )
        records.append(
            {
                "sample_id": sample_id,
                "metadata_row_index": row_index,
                "group": group,
                "group_values": {
                    column: _normalize_value(row[column]) for column in group_columns
                },
                "sample_reference": path_value,
                "metadata": {
                    field: _normalize_value(row[field]) for field in metadata_fields
                },
                "selection_reason": strategy,
            }
        )

    available_coverage = {key: len(indexes) for key, indexes in sorted(groups.items())}
    return records, {
        f"available::{key}": count for key, count in available_coverage.items()
    } | {
        f"selected::{key}": count for key, count in sorted(selected_coverage.items())
    }
