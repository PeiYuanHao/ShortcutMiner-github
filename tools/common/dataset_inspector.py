from collections import deque
from dataclasses import asdict, dataclass
import os
from pathlib import Path


DEFAULT_MAX_DATASET_ENTRIES = 10_000


@dataclass(frozen=True)
class _DirectoryEntry:
    name: str
    path: str
    kind: str


@dataclass
class DatasetInspection:
    root: str
    exists: bool
    num_files: int
    num_directories: int
    file_extensions: list[str]
    file_extension_counts: dict[str, int]
    discovered_files: list[str]
    top_level_entries: list[str]
    max_depth_observed: int
    entries_scanned: int
    max_entries: int
    truncated: bool
    scan_errors: list[str]


def _validate_root(root: str) -> Path:
    if not isinstance(root, str) or not root.strip():
        raise ValueError("Dataset root must be a non-empty path.")

    root_path = Path(root).resolve(strict=True)

    if not root_path.is_dir():
        raise NotADirectoryError(
            f"Dataset root is not a directory: {root}"
        )

    return root_path


def _append_scan_error(
    scan_errors: list[str],
    message: str,
    error_limit: int,
) -> None:
    if len(scan_errors) < error_limit:
        scan_errors.append(message)


def _read_directory(
    directory: str | Path,
    entry_limit: int,
    scan_errors: list[str],
    error_limit: int,
) -> tuple[list[_DirectoryEntry], bool, OSError | None]:
    """
    Read at most ``entry_limit`` entries plus one truncation sentinel.

    Sorting applies only to the bounded enumeration. This avoids materializing
    an unbounded directory while keeping results reproducible when directory
    enumeration is stable for an unchanged dataset.
    """

    entries: list[_DirectoryEntry] = []
    has_more = False
    scan_failure: OSError | None = None

    try:
        with os.scandir(directory) as iterator:
            for entry in iterator:
                if len(entries) == entry_limit:
                    has_more = True
                    break

                try:
                    if entry.is_dir(follow_symlinks=False):
                        kind = "directory"
                    elif entry.is_file(follow_symlinks=False):
                        kind = "file"
                    else:
                        kind = "other"
                except OSError as exc:
                    kind = "other"
                    _append_scan_error(
                        scan_errors,
                        (
                            "Could not inspect dataset entry "
                            f"{entry.path!r}: {exc}"
                        ),
                        error_limit,
                    )

                entries.append(
                    _DirectoryEntry(
                        name=entry.name,
                        path=entry.path,
                        kind=kind,
                    )
                )
    except OSError as exc:
        scan_failure = exc
        _append_scan_error(
            scan_errors,
            (
                "Could not scan dataset directory "
                f"{str(directory)!r}: {exc}"
            ),
            error_limit,
        )

    entries.sort(key=lambda entry: (entry.name.casefold(), entry.name))
    return entries, has_more, scan_failure


def inspect_dataset(
    root: str,
    max_entries: int = DEFAULT_MAX_DATASET_ENTRIES,
) -> dict:
    """
    Inspect basic dataset structure within an explicit entry budget.

    No directory enumeration retains more than the remaining budget plus one
    sentinel. Symlinked directories are not followed. Counts and extensions
    describe only the scanned prefix when ``truncated`` is true.
    """

    if (
        not isinstance(max_entries, int)
        or isinstance(max_entries, bool)
        or max_entries < 1
    ):
        raise ValueError("max_entries must be at least 1.")

    root_path = _validate_root(root)
    scan_errors: list[str] = []
    directories: deque[str] = deque()
    extensions: set[str] = set()
    extension_counts: dict[str, int] = {}
    discovered_files: list[str] = []
    max_depth_observed = 0
    entries_scanned = 0
    num_files = 0
    num_directories = 0

    root_entries, truncated, root_scan_failure = _read_directory(
        root_path,
        entry_limit=max_entries,
        scan_errors=scan_errors,
        error_limit=max_entries,
    )

    if root_scan_failure is not None:
        raise OSError(
            f"Could not scan dataset root {str(root_path)!r}: "
            f"{root_scan_failure}"
        ) from root_scan_failure

    top_level_entries = [entry.name for entry in root_entries]

    def record_entries(entries: list[_DirectoryEntry]) -> None:
        nonlocal entries_scanned, num_files, num_directories
        nonlocal max_depth_observed

        for entry in entries:
            entries_scanned += 1
            try:
                relative_path = Path(entry.path).relative_to(root_path)
                max_depth_observed = max(
                    max_depth_observed,
                    len(relative_path.parts),
                )
            except ValueError:
                # os.scandir entries should always be descendants of the
                # directory being scanned. Keep the scan bounded and omit an
                # unsafe reference if a platform adapter violates that rule.
                relative_path = None

            if entry.kind == "file":
                num_files += 1
                suffix = Path(entry.name).suffix.lower()
                if suffix:
                    extensions.add(suffix)
                    extension_counts[suffix] = (
                        extension_counts.get(suffix, 0) + 1
                    )
                else:
                    extension_counts["<none>"] = (
                        extension_counts.get("<none>", 0) + 1
                    )
                if relative_path is not None:
                    discovered_files.append(relative_path.as_posix())
            elif entry.kind == "directory":
                num_directories += 1
                directories.append(entry.path)

    record_entries(root_entries)

    while directories and not truncated:
        remaining_entries = max_entries - entries_scanned
        directory = directories.popleft()
        (
            child_entries,
            directory_truncated,
            _child_scan_failure,
        ) = _read_directory(
            directory,
            entry_limit=remaining_entries,
            scan_errors=scan_errors,
            error_limit=max_entries,
        )
        record_entries(child_entries)

        if directory_truncated:
            truncated = True

    result = DatasetInspection(
        root=str(root_path),
        exists=True,
        num_files=num_files,
        num_directories=num_directories,
        file_extensions=sorted(extensions),
        file_extension_counts=dict(sorted(extension_counts.items())),
        discovered_files=discovered_files,
        top_level_entries=top_level_entries,
        max_depth_observed=max_depth_observed,
        entries_scanned=entries_scanned,
        max_entries=max_entries,
        truncated=truncated,
        scan_errors=scan_errors,
    )

    return asdict(result)
