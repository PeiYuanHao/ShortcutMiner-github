from pathlib import Path

import pytest

from tools.common.dataset_inspector import inspect_dataset


def test_inspect_dataset(tmp_path: Path):
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"

    train_dir.mkdir()
    test_dir.mkdir()

    (train_dir / "sample1.jpg").write_bytes(b"test")
    (train_dir / "sample2.png").write_bytes(b"test")
    (test_dir / "sample3.jpg").write_bytes(b"test")
    (tmp_path / "metadata.csv").write_text(
        "id,label\n1,0\n",
        encoding="utf-8",
    )

    result = inspect_dataset(str(tmp_path))

    assert result["exists"] is True
    assert result["num_files"] == 4
    assert result["num_directories"] == 2

    assert result["file_extensions"] == [
        ".csv",
        ".jpg",
        ".png",
    ]

    assert result["top_level_entries"] == [
        "metadata.csv",
        "test",
        "train",
    ]


def test_missing_dataset():
    result = inspect_dataset(
        "this_path_should_not_exist_123456"
    )

    assert result["exists"] is False
    assert result["num_files"] == 0
    assert result["num_directories"] == 0
    assert result["file_extensions"] == []
    assert result["top_level_entries"] == []


def test_dataset_root_is_file(tmp_path: Path):
    file_path = tmp_path / "dataset.txt"
    file_path.write_text(
        "not a directory",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        inspect_dataset(str(file_path))