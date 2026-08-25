from pathlib import Path

import pytest

from tools.common.sample_selector import (
    select_samples,
)


def _create_metadata(path: Path) -> None:
    path.write_text(
        "id,label,site\n"
        "1,0,A\n"
        "2,0,A\n"
        "3,0,B\n"
        "4,0,B\n"
        "5,1,A\n"
        "6,1,A\n"
        "7,1,B\n"
        "8,1,B\n",
        encoding="utf-8",
    )


def test_random_sampling(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    result = select_samples(
        metadata_path=str(metadata_path),
        max_samples=4,
        seed=42,
        strategy="random",
    )

    assert len(result) == 4


def test_random_sampling_is_reproducible(
    tmp_path: Path,
):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    result_a = select_samples(
        metadata_path=str(metadata_path),
        max_samples=4,
        seed=42,
        strategy="random",
    )

    result_b = select_samples(
        metadata_path=str(metadata_path),
        max_samples=4,
        seed=42,
        strategy="random",
    )

    assert result_a == result_b


def test_class_balanced_sampling(
    tmp_path: Path,
):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    result = select_samples(
        metadata_path=str(metadata_path),
        max_samples=4,
        seed=42,
        strategy="class_balanced",
        label_column="label",
    )

    labels = [
        sample["label"]
        for sample in result
    ]

    assert labels.count(0) == 2
    assert labels.count(1) == 2


def test_stratified_sampling(
    tmp_path: Path,
):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    result = select_samples(
        metadata_path=str(metadata_path),
        max_samples=4,
        seed=42,
        strategy="stratified",
        stratify_column="site",
    )

    sites = [
        sample["site"]
        for sample in result
    ]

    assert sites.count("A") == 2
    assert sites.count("B") == 2


def test_budget_larger_than_dataset(
    tmp_path: Path,
):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    result = select_samples(
        metadata_path=str(metadata_path),
        max_samples=100,
        seed=42,
        strategy="random",
    )

    assert len(result) == 8


def test_missing_label_column(
    tmp_path: Path,
):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    with pytest.raises(ValueError):
        select_samples(
            metadata_path=str(metadata_path),
            max_samples=4,
            seed=42,
            strategy="class_balanced",
            label_column="does_not_exist",
        )


def test_missing_metadata_file():
    with pytest.raises(FileNotFoundError):
        select_samples(
            metadata_path="missing_123456.csv",
            max_samples=4,
            seed=42,
        )


def test_invalid_strategy(
    tmp_path: Path,
):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    with pytest.raises(ValueError):
        select_samples(
            metadata_path=str(metadata_path),
            max_samples=4,
            seed=42,
            strategy="invalid_strategy",
        )


def test_invalid_budget(
    tmp_path: Path,
):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    with pytest.raises(ValueError):
        select_samples(
            metadata_path=str(metadata_path),
            max_samples=0,
            seed=42,
        )