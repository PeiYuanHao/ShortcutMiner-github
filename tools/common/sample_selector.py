from pathlib import Path

import pandas as pd


SUPPORTED_STRATEGIES = {
    "random",
    "class_balanced",
    "stratified",
}


def _validate_common_inputs(
    df: pd.DataFrame,
    max_samples: int,
) -> None:
    if max_samples < 1:
        raise ValueError(
            "max_samples must be at least 1"
        )

    if df.empty:
        raise ValueError(
            "Metadata contains no samples"
        )


def _random_sample(
    df: pd.DataFrame,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    n = min(max_samples, len(df))

    return df.sample(
        n=n,
        random_state=seed,
    )


def _balanced_sample(
    df: pd.DataFrame,
    group_column: str,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    if group_column not in df.columns:
        raise ValueError(
            f"Column not found: {group_column}"
        )

    valid_df = df.dropna(
        subset=[group_column]
    )

    if valid_df.empty:
        raise ValueError(
            f"No valid values in column: {group_column}"
        )

    groups = list(
        valid_df.groupby(
            group_column,
            sort=True,
        )
    )

    num_groups = len(groups)

    if num_groups == 0:
        return _random_sample(
            valid_df,
            max_samples,
            seed,
        )

    base_per_group = max_samples // num_groups
    remainder = max_samples % num_groups

    selected_parts = []

    for index, (_, group_df) in enumerate(groups):
        group_budget = base_per_group

        if index < remainder:
            group_budget += 1

        if group_budget == 0:
            continue

        group_n = min(
            group_budget,
            len(group_df),
        )

        selected = group_df.sample(
            n=group_n,
            random_state=seed + index,
        )

        selected_parts.append(selected)

    if not selected_parts:
        return _random_sample(
            valid_df,
            max_samples,
            seed,
        )

    selected_df = pd.concat(
        selected_parts,
        axis=0,
    )

    remaining_budget = (
        max_samples - len(selected_df)
    )

    if remaining_budget > 0:
        remaining_df = valid_df.drop(
            index=selected_df.index
        )

        if not remaining_df.empty:
            extra_n = min(
                remaining_budget,
                len(remaining_df),
            )

            extra = remaining_df.sample(
                n=extra_n,
                random_state=seed + 1000,
            )

            selected_df = pd.concat(
                [selected_df, extra],
                axis=0,
            )

    return selected_df


def select_samples(
    metadata_path: str,
    max_samples: int,
    seed: int,
    strategy: str = "random",
    label_column: str | None = None,
    stratify_column: str | None = None,
) -> list[dict]:
    """
    Select a reproducible subset of samples from CSV metadata.

    This function performs deterministic sample selection only.
    It does not inspect samples or perform shortcut validation.
    """

    path = Path(metadata_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Metadata path is not a file: {metadata_path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported metadata format: {path.suffix}"
        )

    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported sampling strategy: {strategy}"
        )

    df = pd.read_csv(path)

    _validate_common_inputs(
        df=df,
        max_samples=max_samples,
    )

    if strategy == "random":
        selected = _random_sample(
            df=df,
            max_samples=max_samples,
            seed=seed,
        )

    elif strategy == "class_balanced":
        if label_column is None:
            raise ValueError(
                "label_column is required "
                "for class_balanced sampling"
            )

        selected = _balanced_sample(
            df=df,
            group_column=label_column,
            max_samples=max_samples,
            seed=seed,
        )

    else:
        if stratify_column is None:
            raise ValueError(
                "stratify_column is required "
                "for stratified sampling"
            )

        selected = _balanced_sample(
            df=df,
            group_column=stratify_column,
            max_samples=max_samples,
            seed=seed,
        )

    return (
        selected
        .sort_index()
        .to_dict(orient="records")
    )