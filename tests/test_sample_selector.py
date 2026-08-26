from pathlib import Path

import pytest

from tools.common.sample_selector import select_samples


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


def _values(result: list[dict], column: str) -> list:
    return [selection["record"][column] for selection in result]


def test_random_sampling_returns_explicit_metadata_rows(
    tmp_path: Path,
):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    result = select_samples(
        metadata_path=str(metadata_path),
        max_samples=4,
        seed=42,
        strategy="random",
    )

    assert len(result) == 4
    assert all(
        selection["selection_kind"] == "metadata_row"
        for selection in result
    )
    assert all(
        selection["selection_id"]
        == f"metadata-row-{selection['metadata_row_index']:08d}"
        for selection in result
    )
    assert all(
        selection["metadata_row_number"]
        == selection["metadata_row_index"] + 1
        for selection in result
    )


def test_random_sampling_is_reproducible(tmp_path: Path):
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


def test_class_balanced_sampling(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    result = select_samples(
        metadata_path=str(metadata_path),
        max_samples=4,
        seed=42,
        strategy="class_balanced",
        label_column="label",
    )

    labels = _values(result, "label")
    assert labels.count(0) == 2
    assert labels.count(1) == 2


def test_stratified_sampling(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    result = select_samples(
        metadata_path=str(metadata_path),
        max_samples=4,
        seed=42,
        strategy="stratified",
        stratify_column="site",
    )

    sites = _values(result, "site")
    assert sites.count("A") == 2
    assert sites.count("B") == 2


def test_balanced_sampling_uses_seed_when_groups_exceed_budget(
    tmp_path: Path,
):
    metadata_path = tmp_path / "many-groups.csv"
    metadata_path.write_text(
        "id,label\n1,0\n2,1\n3,2\n4,3\n5,4\n",
        encoding="utf-8",
    )

    result = select_samples(
        metadata_path=str(metadata_path),
        max_samples=2,
        seed=42,
        strategy="class_balanced",
        label_column="label",
    )
    repeated = select_samples(
        metadata_path=str(metadata_path),
        max_samples=2,
        seed=42,
        strategy="class_balanced",
        label_column="label",
    )

    assert result == repeated
    assert _values(result, "label") != [0, 1]


def test_budget_larger_than_dataset(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    result = select_samples(
        metadata_path=str(metadata_path),
        max_samples=100,
        seed=42,
        strategy="random",
    )

    assert len(result) == 8
    assert [item["metadata_row_index"] for item in result] == list(
        range(8)
    )


def test_selection_obeys_metadata_row_budget(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    result = select_samples(
        metadata_path=str(metadata_path),
        max_samples=8,
        seed=42,
        strategy="random",
        max_metadata_rows=3,
    )

    assert len(result) == 3
    assert [item["metadata_row_index"] for item in result] == [0, 1, 2]


def test_missing_label_column(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    with pytest.raises(ValueError, match="Column not found"):
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


def test_invalid_strategy(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    with pytest.raises(ValueError, match="Unsupported"):
        select_samples(
            metadata_path=str(metadata_path),
            max_samples=4,
            seed=42,
            strategy="invalid_strategy",
        )


@pytest.mark.parametrize("max_samples", [0, -1, True])
def test_invalid_budget(tmp_path: Path, max_samples):
    metadata_path = tmp_path / "metadata.csv"
    _create_metadata(metadata_path)

    with pytest.raises(ValueError, match="at least 1"):
        select_samples(
            metadata_path=str(metadata_path),
            max_samples=max_samples,
            seed=42,
        )
