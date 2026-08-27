from pathlib import Path

import pytest

from tools.common.metadata_profiler import profile_metadata


def test_profile_metadata(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "id,label,site\n"
        "1,0,A\n"
        "2,0,A\n"
        "3,1,B\n"
        "4,,B\n",
        encoding="utf-8",
    )

    result = profile_metadata(str(metadata_path))

    assert result["exists"] is True
    assert result["num_rows"] == 4
    assert result["rows_profiled"] == 4
    assert result["row_limit"] == 100_000
    assert result["truncated"] is False
    assert result["num_columns"] == 3
    assert [column["name"] for column in result["columns"]] == [
        "id",
        "label",
        "site",
    ]

    label_profile = next(
        column
        for column in result["columns"]
        if column["name"] == "label"
    )
    assert label_profile["non_null_count"] == 3
    assert label_profile["missing_count"] == 1
    assert label_profile["unique_count"] == 2
    assert label_profile["sample_values"] == [0.0, 1.0]


def test_metadata_profile_obeys_row_budget(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "id,label\n1,0\n2,0\n3,1\n4,1\n",
        encoding="utf-8",
    )

    result = profile_metadata(str(metadata_path), max_rows=2)

    assert result["num_rows"] == 2
    assert result["rows_profiled"] == 2
    assert result["row_limit"] == 2
    assert result["truncated"] is True
    label_profile = result["columns"][1]
    assert label_profile["unique_count"] == 1
    assert label_profile["sample_values"] == [0]


def test_exact_metadata_row_budget_is_not_truncated(
    tmp_path: Path,
):
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "id\n1\n2\n",
        encoding="utf-8",
    )

    result = profile_metadata(str(metadata_path), max_rows=2)

    assert result["num_rows"] == 2
    assert result["truncated"] is False


def test_missing_metadata():
    result = profile_metadata(
        "this_metadata_file_should_not_exist_123456.csv"
    )

    assert result["exists"] is False
    assert result["num_rows"] == 0
    assert result["num_columns"] == 0
    assert result["columns"] == []
    assert result["rows_profiled"] == 0
    assert result["truncated"] is False


def test_metadata_path_is_directory(tmp_path: Path):
    with pytest.raises(ValueError):
        profile_metadata(str(tmp_path))


def test_unsupported_metadata_format(tmp_path: Path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported"):
        profile_metadata(str(metadata_path))


def test_sample_values_limit(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "site\nA\nB\nC\nD\n",
        encoding="utf-8",
    )

    result = profile_metadata(
        str(metadata_path),
        sample_values_per_column=2,
    )

    site_profile = result["columns"][0]
    assert site_profile["sample_values"] == ["A", "B"]


@pytest.mark.parametrize("max_rows", [0, -1, True])
def test_invalid_metadata_row_budget(
    tmp_path: Path,
    max_rows,
):
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text("id\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least 1"):
        profile_metadata(str(metadata_path), max_rows=max_rows)


def test_invalid_sample_value_budget(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text("id\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-negative"):
        profile_metadata(
            str(metadata_path),
            sample_values_per_column=-1,
        )
