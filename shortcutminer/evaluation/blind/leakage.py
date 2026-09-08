"""Ground-truth firewall checks for blind evaluation infrastructure."""

from __future__ import annotations

import ast
import csv
from pathlib import Path


PRIVATE_NAME_TOKENS = {
    "shortcut", "bias", "spurious", "injected", "target_cue", "ground_truth", "gold"
}
class LeakageBoundaryError(RuntimeError):
    """Raised when public/private challenge isolation is violated."""


def assert_path_isolation(
    *,
    challenge_root: str | Path,
    dataset_root: str | Path,
    hidden_truth_path: str | Path,
) -> None:
    challenge = Path(challenge_root).resolve(strict=True)
    dataset = Path(dataset_root).resolve(strict=True)
    truth = Path(hidden_truth_path).resolve(strict=True)
    try:
        dataset.relative_to(challenge / "public")
        truth.relative_to(challenge / "private")
    except ValueError as exc:
        raise LeakageBoundaryError("challenge public/private roots are not isolated") from exc
    try:
        truth.relative_to(dataset)
    except ValueError:
        pass
    else:
        raise LeakageBoundaryError("hidden truth is inside dataset_root")
    if dataset.is_symlink() or truth.is_symlink():
        raise LeakageBoundaryError("challenge boundary endpoints must not be symlinks")


def assert_public_names_are_neutral(dataset_root: str | Path) -> None:
    root = Path(dataset_root).resolve(strict=True)
    violations = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix().casefold()
        if any(token in relative for token in PRIVATE_NAME_TOKENS):
            violations.append(relative)
    metadata = root / "metadata.csv"
    if metadata.is_file():
        with metadata.open("r", encoding="utf-8", newline="") as stream:
            headers = next(csv.reader(stream), [])
        violations.extend(
            header for header in headers
            if any(token in header.casefold() for token in PRIVATE_NAME_TOKENS)
        )
    if violations:
        raise LeakageBoundaryError("private naming leaked into public data: " + ", ".join(violations))


def production_private_import_violations(repository_root: str | Path) -> list[str]:
    root = Path(repository_root).resolve(strict=True)
    violations = []
    forbidden = "shortcutminer.evaluation.blind"
    source_root = root / "shortcutminer"
    private_namespace = source_root / "evaluation" / "blind"
    for path in source_root.rglob("*.py"):
        if path.is_relative_to(private_namespace):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == forbidden or name.startswith(forbidden + ".") for name in names):
                violations.append(f"{path.relative_to(root).as_posix()}:{node.lineno}")
    return violations


__all__ = [
    "LeakageBoundaryError", "assert_path_isolation", "assert_public_names_are_neutral",
    "production_private_import_violations",
]
