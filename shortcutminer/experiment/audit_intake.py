"""Bounded dataset discovery without models, scientific judgments, or tasks."""

import csv
import io
from pathlib import Path

from shortcutminer.experiment.dataset_profiler import IMAGE_EXTENSIONS, TABULAR_EXTENSIONS, TEXT_EXTENSIONS
from shortcutminer.protocol.audit import AuditDatasetInfo, AuditLabelSource, AuditRequest
from tools.common.dataset_inspector import inspect_dataset


def _csv_header(path: Path, max_bytes: int) -> list[str]:
    # Limit bytes as well as entry counts; a single malformed CSV line can be huge.
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        end = data.rfind(b"\n", 0, max_bytes)
        if end < 0:
            raise ValueError("CSV header exceeds intake byte budget")
        data = data[:end + 1]
    row = next(csv.reader(io.StringIO(data.decode("utf-8-sig")), strict=True), [])
    if not row or any(not value.strip() for value in row) or len(set(row)) != len(row):
        raise ValueError("CSV header is empty or ambiguous")
    return row


def _directory_labels(files: list[str], *, truncated: bool, failed: bool):
    images = [Path(name).parts for name in files if Path(name).suffix.lower() in IMAGE_EXTENSIONS]
    if not images or truncated or failed:
        return None, [], []
    split_names = {"train", "training", "val", "valid", "validation", "test", "testing"}
    if all(len(parts) == 3 and parts[0].casefold() in split_names for parts in images):
        by_split: dict[str, set[str]] = {}
        for split, category, _ in images:
            by_split.setdefault(split, set()).add(category)
        classes = next(iter(by_split.values()))
        if len(classes) >= 2 and all(value == classes for value in by_split.values()):
            return "split_class_directory", sorted(classes), ["split/class/image structure"]
    if all(len(parts) == 2 for parts in images):
        classes = {parts[0] for parts in images}
        generic = split_names | {"images", "samples", "assets", "data", "photos"}
        if len(classes) >= 2 and not any(value.casefold() in generic for value in classes):
            return "class_directory", sorted(classes), ["class/image structure"]
    return None, [], ["Image directories exist, but no unambiguous class layout was recognized."]


def intake_dataset(request: AuditRequest) -> AuditRequest:
    """Return a fresh factual intake; never trust a caller-supplied dataset_info."""
    source = Path(request.dataset_path).expanduser().resolve(strict=True)
    if source.is_file() and source.suffix.lower() not in TABULAR_EXTENSIONS:
        raise ValueError("Audit path must be a dataset directory or a tabular file")
    root = source.parent if source.is_file() else source
    budget = request.intake_budget
    if source.is_file():
        files, scanned, truncated, errors = [source.name], 1, False, []
    else:
        scan = inspect_dataset(str(root), max_entries=budget.max_entries)
        files = sorted(scan["discovered_files"])
        scanned, truncated, errors = scan["entries_scanned"], scan["truncated"], scan["scan_errors"]
    limitations = [f"dataset_scan_error: {error}" for error in errors]
    if truncated:
        limitations.append("dataset_scan_truncated: observations cover only the bounded scan.")
    discovered = [name for name in files if Path(name).suffix.lower() in TABULAR_EXTENSIONS]
    if len(discovered) > budget.max_metadata_files:
        limitations.append("metadata_file_budget_reached: not all metadata sources were inspected.")
    metadata_files = discovered[:budget.max_metadata_files]
    selected = None
    if request.metadata_file:
        path = (root / request.metadata_file).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Explicit metadata must be a file inside the dataset root")
        if path.suffix.lower() != ".csv":
            limitations.append("metadata_format_unsupported: existing workflow reads CSV only.")
        else:
            selected = request.metadata_file
            if selected not in metadata_files:
                metadata_files = [selected, *metadata_files][:budget.max_metadata_files]
    elif not truncated and not errors and len(discovered) == 1 and Path(discovered[0]).suffix.lower() == ".csv":
        selected = discovered[0]
    elif truncated or errors:
        limitations.append("metadata_source_unresolved: incomplete scan cannot establish a unique source.")
    elif len(discovered) > 1:
        limitations.append("metadata_source_unresolved: multiple table files; no source was guessed.")
    elif discovered:
        limitations.append("metadata_format_unsupported: existing workflow reads CSV only.")
    columns = []
    if selected:
        try:
            path = (root / selected).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("Metadata resolves outside the dataset root")
            columns = _csv_header(path, budget.max_header_bytes)
        except (OSError, UnicodeError, ValueError, csv.Error) as exc:
            limitations.append(f"metadata_header_unavailable: {exc}")
            selected = None
    families = set()
    for name in files:
        suffix = Path(name).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            families.add("image")
        elif suffix in TABULAR_EXTENSIONS and name != selected:
            families.add("tabular")
        elif suffix in TEXT_EXTENSIONS:
            families.add("text")
    if not families and discovered:
        families.add("tabular")
    dataset_type = "mixed" if len(families) > 1 else next(iter(families), "unknown")
    kind, classes, possibilities = _directory_labels(files, truncated=truncated, failed=bool(errors))
    possible_columns = [value for value in columns if value.casefold() in {"label", "target", "class"}]
    possibilities = [*(f"metadata column: {value}" for value in possible_columns), *possibilities]
    label = AuditLabelSource(reason="No unambiguous label source was found.", possible_sources=possibilities)
    explicit = request.label_information
    if explicit and explicit.source == "metadata_column":
        if explicit.column in columns and selected:
            label = AuditLabelSource(status="resolved", kind="metadata_column", column=explicit.column,
                                     possible_sources=possibilities, reason="User supplied an exact existing column.")
        else:
            label.reason = "The supplied label column was not found in an available metadata source."
    elif explicit:
        if explicit.source == kind:
            label = AuditLabelSource(status="resolved", kind=kind, classes=classes,
                                     possible_sources=possibilities, reason="User supplied a matching directory layout.")
        else:
            label.reason = "The supplied directory label layout could not be verified."
    elif selected and len(possible_columns) == 1 and kind is None:
        label = AuditLabelSource(status="resolved", kind="metadata_column", column=possible_columns[0],
                                 possible_sources=possibilities, reason="One conventional label column was observed.")
    elif kind and not possible_columns and not discovered:
        label = AuditLabelSource(status="resolved", kind=kind, classes=classes,
                                 possible_sources=possibilities, reason="A conventional class directory layout was observed.")
    if label.status == "unresolved":
        limitations.append(f"label_source_unresolved: {label.reason}")
    info = AuditDatasetInfo(root=str(root), name=source.stem if source.is_file() else source.name or "dataset",
                            dataset_type=dataset_type, metadata_files=metadata_files,
                            selected_metadata_file=selected, metadata_columns=columns, label_source=label,
                            entries_scanned=scanned, truncated=truncated, limitations=limitations)
    return request.model_copy(update={"dataset_info": info})
