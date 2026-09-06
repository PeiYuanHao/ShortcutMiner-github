import os
from pathlib import Path

import pytest

from tools.common import dataset_inspector
from tools.common.dataset_inspector import inspect_dataset


def test_unreadable_root_is_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    real_scandir = os.scandir

    def reject_root(path):
        if Path(path) == dataset_root.resolve():
            raise PermissionError("root is unreadable")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", reject_root)

    with pytest.raises(OSError, match="Could not scan dataset root"):
        inspect_dataset(str(dataset_root))


def test_unreadable_child_is_a_partial_scan_limitation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    child = tmp_path / "child"
    child.mkdir()
    real_scandir = os.scandir

    def reject_child(path):
        if Path(path) == child:
            raise PermissionError("child is unreadable")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", reject_child)

    result = inspect_dataset(str(tmp_path))

    assert result["num_directories"] == 1
    assert result["entries_scanned"] == 1
    assert result["truncated"] is False
    assert len(result["scan_errors"]) == 1
    assert "child is unreadable" in result["scan_errors"][0]


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
    assert result["entries_scanned"] == 6
    assert result["max_entries"] == 10_000
    assert result["truncated"] is False
    assert result["scan_errors"] == []
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


def test_dataset_scan_is_bounded_and_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    nested = tmp_path / "a"
    nested.mkdir()
    (nested / "inside.png").write_bytes(b"test")
    (tmp_path / "b.txt").write_bytes(b"test")
    (tmp_path / "z.jpg").write_bytes(b"test")

    # scandir order is filesystem-dependent. Supply the intended ordered input
    # explicitly; production promises to sort a bounded prefix, not to discover
    # the lexicographically first entries of an unbounded directory.
    real_scandir = os.scandir

    class OrderedScandir:
        def __init__(self, directory):
            with real_scandir(directory) as entries:
                self.entries = sorted(entries, key=lambda entry: entry.name)

        def __enter__(self):
            return iter(self.entries)

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(dataset_inspector.os, "scandir", OrderedScandir)

    result_a = inspect_dataset(str(tmp_path), max_entries=2)
    result_b = inspect_dataset(str(tmp_path), max_entries=2)

    assert result_a == result_b
    assert result_a["entries_scanned"] == 2
    assert result_a["truncated"] is True
    assert result_a["num_directories"] == 1
    assert result_a["num_files"] == 1
    assert result_a["file_extensions"] == [".txt"]
    assert result_a["top_level_entries"] == ["a", "b.txt"]


@pytest.mark.parametrize("names", [
    ["z.jpg", "b.txt", "a.csv", "unread.png"],
    ["b.txt", "z.jpg", "a.csv", "unread.png"],
])
def test_bounded_prefix_is_sorted_without_consuming_rest_of_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: list[str],
):
    requested = []
    classified = []

    class Entry:
        def __init__(self, name):
            self.name = name
            self.path = str(tmp_path / name)

        def is_dir(self, *, follow_symlinks):
            assert follow_symlinks is False
            classified.append(self.name)
            return False

        def is_file(self, *, follow_symlinks):
            assert follow_symlinks is False
            return True

    class Scandir:
        def __enter__(self):
            def entries():
                for name in names:
                    requested.append(name)
                    yield Entry(name)
            return entries()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(dataset_inspector.os, "scandir", lambda _: Scandir())
    result = inspect_dataset(str(tmp_path), max_entries=2)

    assert requested == names[:3]  # Exactly the budget and truncation sentinel.
    assert classified == names[:2]  # The sentinel must not be inspected.
    assert result["top_level_entries"] == ["b.txt", "z.jpg"]
    assert result["file_extensions"] == [".jpg", ".txt"]
    assert result["entries_scanned"] == 2
    assert result["num_files"] == 2
    assert result["num_directories"] == 0
    assert result["truncated"] is True


def test_directory_enumeration_stops_after_budget_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    entries_requested = 0

    class FakeEntry:
        def __init__(self, index: int):
            self.name = f"sample-{index:04d}.jpg"
            self.path = str(tmp_path / self.name)

        def is_dir(self, *, follow_symlinks: bool) -> bool:
            return False

        def is_file(self, *, follow_symlinks: bool) -> bool:
            return True

    class FakeScandir:
        def __init__(self):
            self.index = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal entries_requested

            if self.index == 100:
                raise StopIteration

            entry = FakeEntry(self.index)
            self.index += 1
            entries_requested += 1
            return entry

    monkeypatch.setattr(
        dataset_inspector.os,
        "scandir",
        lambda directory: FakeScandir(),
    )

    result = inspect_dataset(str(tmp_path), max_entries=2)

    assert entries_requested == 3
    assert result["entries_scanned"] == 2
    assert result["truncated"] is True


def test_exact_scan_budget_is_not_marked_truncated(
    tmp_path: Path,
):
    (tmp_path / "a.txt").write_bytes(b"test")
    (tmp_path / "b.txt").write_bytes(b"test")

    result = inspect_dataset(str(tmp_path), max_entries=2)

    assert result["entries_scanned"] == 2
    assert result["truncated"] is False


def test_missing_dataset_is_unrecoverable(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        inspect_dataset(str(tmp_path / "missing"))


def test_empty_dataset_root_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        inspect_dataset("")


def test_dataset_root_is_file(tmp_path: Path):
    file_path = tmp_path / "dataset.txt"
    file_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        inspect_dataset(str(file_path))


@pytest.mark.parametrize("max_entries", [0, -1, True])
def test_invalid_scan_budget(tmp_path: Path, max_entries):
    with pytest.raises(ValueError, match="at least 1"):
        inspect_dataset(str(tmp_path), max_entries=max_entries)
