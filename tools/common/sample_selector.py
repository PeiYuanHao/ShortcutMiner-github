from random import Random
from typing import Any

import pandas as pd

from tools.common.metadata_profiler import (
    DEFAULT_MAX_METADATA_ROWS,
    LoadedMetadata,
    _normalize_value,
    load_metadata,
)


SUPPORTED_STRATEGIES = {
    "random",
    "class_balanced",
    "stratified",
}


def _random_state(seed: int, offset: int = 0) -> int:
    return (seed + offset) % (2**32)


def _validate_common_inputs(
    dataframe: pd.DataFrame,
    max_samples: int,
    strategy: str,
) -> None:
    if (
        not isinstance(max_samples, int)
        or isinstance(max_samples, bool)
        or max_samples < 1
    ):
        raise ValueError("max_samples must be at least 1.")

    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported sampling strategy: {strategy}"
        )

    if dataframe.empty:
        raise ValueError("Metadata contains no samples.")


def _random_sample(
    dataframe: pd.DataFrame,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    sample_count = min(max_samples, len(dataframe))
    return dataframe.sample(
        n=sample_count,
        random_state=_random_state(seed),
    )


def _balanced_sample(
    dataframe: pd.DataFrame,
    group_column: str,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    if group_column not in dataframe.columns:
        raise ValueError(f"Column not found: {group_column}")

    valid_dataframe = dataframe.dropna(subset=[group_column])

    if valid_dataframe.empty:
        raise ValueError(
            f"No valid values in column: {group_column}"
        )

    groups = list(
        valid_dataframe.groupby(
            group_column,
            sort=True,
        )
    )
    num_groups = len(groups)

    if num_groups == 0:
        return _random_sample(
            valid_dataframe,
            max_samples,
            seed,
        )

    base_per_group = max_samples // num_groups
    remainder = max_samples % num_groups
    allocation_order = list(range(num_groups))
    Random(seed).shuffle(allocation_order)
    groups_with_extra = set(allocation_order[:remainder])
    selected_parts = []

    for index, (_, group_dataframe) in enumerate(groups):
        group_budget = base_per_group + int(index in groups_with_extra)

        if group_budget == 0:
            continue

        selected_parts.append(
            group_dataframe.sample(
                n=min(group_budget, len(group_dataframe)),
                random_state=_random_state(seed, index),
            )
        )

    if not selected_parts:
        return _random_sample(
            valid_dataframe,
            max_samples,
            seed,
        )

    selected_dataframe = pd.concat(selected_parts, axis=0)
    remaining_budget = max_samples - len(selected_dataframe)

    if remaining_budget > 0:
        remaining_dataframe = valid_dataframe.drop(
            index=selected_dataframe.index
        )

        if not remaining_dataframe.empty:
            extra = remaining_dataframe.sample(
                n=min(remaining_budget, len(remaining_dataframe)),
                random_state=_random_state(seed, 1000),
            )
            selected_dataframe = pd.concat(
                [selected_dataframe, extra],
                axis=0,
            )

    return selected_dataframe


def _selected_metadata_records(
    selected: pd.DataFrame,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for row_index, row in selected.sort_index().iterrows():
        normalized_record = {
            str(column): _normalize_value(value)
            for column, value in row.items()
        }
        source_row_index = int(row_index)

        records.append(
            {
                "selection_kind": "metadata_row",
                "selection_id": (
                    f"metadata-row-{source_row_index:08d}"
                ),
                # Zero-based logical record index in the source CSV. The
                # one-based number excludes the header and is included only
                # for human-readable evidence references.
                "metadata_row_index": source_row_index,
                "metadata_row_number": source_row_index + 1,
                "record": normalized_record,
            }
        )

    return records


def select_loaded_metadata(
    loaded: LoadedMetadata,
    max_samples: int,
    seed: int,
    strategy: str = "random",
    label_column: str | None = None,
    stratify_column: str | None = None,
) -> list[dict]:
    """
    Select reproducible metadata records from an already bounded CSV view.

    The returned records are metadata selections, not inspected dataset
    samples. Stable identifiers refer to their original logical CSV rows.
    """

    dataframe = loaded.dataframe
    _validate_common_inputs(
        dataframe=dataframe,
        max_samples=max_samples,
        strategy=strategy,
    )

    if strategy == "random":
        selected = _random_sample(
            dataframe=dataframe,
            max_samples=max_samples,
            seed=seed,
        )
    elif strategy == "class_balanced":
        if label_column is None:
            raise ValueError(
                "label_column is required for class_balanced sampling"
            )

        selected = _balanced_sample(
            dataframe=dataframe,
            group_column=label_column,
            max_samples=max_samples,
            seed=seed,
        )
    else:
        if stratify_column is None:
            raise ValueError(
                "stratify_column is required for stratified sampling"
            )

        selected = _balanced_sample(
            dataframe=dataframe,
            group_column=stratify_column,
            max_samples=max_samples,
            seed=seed,
        )

    return _selected_metadata_records(selected)


def select_samples(
    metadata_path: str,
    max_samples: int,
    seed: int,
    strategy: str = "random",
    label_column: str | None = None,
    stratify_column: str | None = None,
    max_metadata_rows: int = DEFAULT_MAX_METADATA_ROWS,
) -> list[dict]:
    """
    Select reproducible records from a bounded CSV metadata prefix.

    This compatibility entry point reads the CSV once. Context builders that
    also profile metadata should call ``load_metadata`` once and pass the
    result to ``select_loaded_metadata``.
    """

    # Validate cheap caller errors before touching the filesystem.
    if (
        not isinstance(max_samples, int)
        or isinstance(max_samples, bool)
        or max_samples < 1
    ):
        raise ValueError("max_samples must be at least 1.")

    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported sampling strategy: {strategy}"
        )

    loaded = load_metadata(
        metadata_path=metadata_path,
        max_rows=max_metadata_rows,
    )
    return select_loaded_metadata(
        loaded=loaded,
        max_samples=max_samples,
        seed=seed,
        strategy=strategy,
        label_column=label_column,
        stratify_column=stratify_column,
    )
