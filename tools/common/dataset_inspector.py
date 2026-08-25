from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DatasetInspection:
    root: str
    exists: bool
    num_files: int
    num_directories: int
    file_extensions: list[str]
    top_level_entries: list[str]


def inspect_dataset(root: str) -> dict:
    """
    Inspect basic dataset structure.

    This function performs deterministic dataset reconnaissance only.
    It does not perform shortcut discovery or validation.
    """

    root_path = Path(root)

    if not root_path.exists():
        result = DatasetInspection(
            root=str(root_path),
            exists=False,
            num_files=0,
            num_directories=0,
            file_extensions=[],
            top_level_entries=[],
        )
        return asdict(result)

    if not root_path.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")

    files = []
    directories = []

    for path in root_path.rglob("*"):
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            directories.append(path)

    extensions = sorted(
        {
            path.suffix.lower()
            for path in files
            if path.suffix
        }
    )

    top_level_entries = sorted(
        path.name
        for path in root_path.iterdir()
    )

    result = DatasetInspection(
        root=str(root_path),
        exists=True,
        num_files=len(files),
        num_directories=len(directories),
        file_extensions=extensions,
        top_level_entries=top_level_entries,
    )

    return asdict(result)