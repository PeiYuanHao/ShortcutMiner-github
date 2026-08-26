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
    assert result["num_columns"] == 3

    column_names = [
        column["name"]
        for column in result["columns"]
    ]

    assert column_names == [
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


def test_missing_metadata():
    result = profile_metadata(
        "this_metadata_file_should_not_exist_123456.csv"
    )

    assert result["exists"] is False
    assert result["num_rows"] == 0
    assert result["num_columns"] == 0
    assert result["columns"] == []


def test_metadata_path_is_directory(tmp_path: Path):
    with pytest.raises(ValueError):
        profile_metadata(str(tmp_path))


def test_unsupported_metadata_format(tmp_path: Path):
    metadata_path = tmp_path / "metadata.json"

    metadata_path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        profile_metadata(str(metadata_path))


def test_sample_values_limit(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"

    metadata_path.write_text(
        "site\n"
        "A\n"
        "B\n"
        "C\n"
        "D\n",
        encoding="utf-8",
    )

    result = profile_metadata(
        str(metadata_path),
        sample_values_per_column=2,
    )

    site_profile = result["columns"][0]

    assert len(site_profile["sample_values"]) == 2
    assert site_profile["sample_values"] == ["A", "B"]