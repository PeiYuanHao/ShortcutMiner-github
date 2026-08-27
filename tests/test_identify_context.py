from pathlib import Path

import pytest

from shortcutminer.experiment.identify_context import (
    IdentifyContextError,
    build_identify_context,
)
from shortcutminer.protocol.identify import (
    DatasetSpec,
    IdentifyBudget,
    IdentifyRequest,
    TaskSpec,
)
from tools.common import metadata_profiler


def _create_test_dataset(root: Path) -> None:
    train_dir = root / "train"
    train_dir.mkdir()

    for sample_id in range(1, 9):
        (train_dir / f"sample_{sample_id}.jpg").write_bytes(b"test")

    (root / "metadata.csv").write_text(
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


def _request(
    root: Path,
    *,
    metadata_file: str | None = "metadata.csv",
    label_column: str = "label",
    max_samples: int = 4,
    max_dataset_entries: int = 10_000,
    max_metadata_rows: int = 100_000,
    run_id: str = "test_run",
) -> IdentifyRequest:
    return IdentifyRequest(
        protocol_version="1.0",
        run_id=run_id,
        message_id=f"msg-{run_id}",
        source="orchestrator",
        target="identify",
        message_type="identify_request",
        dataset=DatasetSpec(
            name="test_dataset",
            root=str(root),
            metadata_file=metadata_file,
        ),
        task=TaskSpec(
            type="classification",
            label_column=label_column,
        ),
        budget=IdentifyBudget(
            max_samples=max_samples,
            max_candidates=5,
            max_dataset_entries=max_dataset_entries,
            max_metadata_rows=max_metadata_rows,
        ),
    )


def test_build_identify_context(tmp_path: Path):
    _create_test_dataset(tmp_path)

    context = build_identify_context(
        request=_request(tmp_path),
        seed=42,
    )

    assert context.protocol_version == "1.0"
    assert context.run_id == "test_run"
    assert context.seed == 42
    assert context.dataset_name == "test_dataset"
    assert context.dataset_inspection["exists"] is True
    assert context.metadata_available is True
    assert context.metadata_profile is not None
    assert context.metadata_profile["exists"] is True
    assert context.metadata_rows_selected == 4
    assert len(context.selected_samples) == 4
    assert context.samples_inspected == 0

    labels = [
        selection["record"]["label"]
        for selection in context.selected_samples
    ]
    assert labels.count(0) == 2
    assert labels.count(1) == 2
    assert all(
        selection["selection_kind"] == "metadata_row"
        for selection in context.selected_samples
    )
    assert any(
        "no dataset sample content was inspected" in limitation
        for limitation in context.limitations
    )


def test_context_is_reproducible(tmp_path: Path):
    _create_test_dataset(tmp_path)
    request = _request(tmp_path)

    context_a = build_identify_context(request=request, seed=42)
    context_b = build_identify_context(request=request, seed=42)

    assert context_a.selected_samples == context_b.selected_samples
    assert [
        item["selection_id"] for item in context_a.selected_samples
    ] == [
        item["selection_id"] for item in context_b.selected_samples
    ]


def test_missing_dataset_is_explicit_context_error(tmp_path: Path):
    request = _request(
        tmp_path / "missing_dataset",
        run_id="missing_run",
    )

    with pytest.raises(
        IdentifyContextError,
        match="Dataset root is unavailable",
    ):
        build_identify_context(request=request, seed=42)


def test_dataset_file_is_explicit_context_error(tmp_path: Path):
    dataset_file = tmp_path / "dataset.txt"
    dataset_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(
        IdentifyContextError,
        match="Dataset root is unavailable",
    ):
        build_identify_context(
            request=_request(dataset_file),
            seed=42,
        )


def test_missing_metadata_is_noncritical(tmp_path: Path):
    context = build_identify_context(
        request=_request(tmp_path, metadata_file="missing.csv"),
        seed=42,
    )

    assert context.metadata_profile is None
    assert context.metadata_available is False
    assert context.selected_samples == []
    assert context.samples_inspected == 0
    assert any(
        "unavailable or invalid" in limitation
        for limitation in context.limitations
    )


def test_no_metadata_configured(tmp_path: Path):
    context = build_identify_context(
        request=_request(tmp_path, metadata_file=None),
        seed=42,
    )

    assert context.metadata_profile is None
    assert context.metadata_available is False
    assert context.selected_samples == []
    assert "No metadata file was configured." in context.limitations


def test_sampling_fallback(tmp_path: Path):
    (tmp_path / "metadata.csv").write_text(
        "id,site\n1,A\n2,A\n3,B\n4,B\n",
        encoding="utf-8",
    )

    context = build_identify_context(
        request=_request(
            tmp_path,
            label_column="missing_label",
            max_samples=2,
            run_id="fallback_run",
        ),
        seed=42,
    )

    assert len(context.selected_samples) == 2
    assert context.samples_inspected == 0
    assert any(
        "Class-balanced sampling failed" in limitation
        for limitation in context.limitations
    )
    assert "Random sampling was used as fallback." in context.limitations


def test_dataset_scan_budget_becomes_limitation(tmp_path: Path):
    _create_test_dataset(tmp_path)

    context = build_identify_context(
        request=_request(tmp_path, max_dataset_entries=2),
        seed=42,
    )

    assert context.dataset_inspection["entries_scanned"] == 2
    assert context.dataset_inspection["truncated"] is True
    assert any(
        "max_dataset_entries=2" in limitation
        for limitation in context.limitations
    )


def test_metadata_row_budget_applies_to_profile_and_selection(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)

    context = build_identify_context(
        request=_request(
            tmp_path,
            max_samples=8,
            max_metadata_rows=3,
        ),
        seed=42,
    )

    assert context.metadata_profile is not None
    assert context.metadata_profile["rows_profiled"] == 3
    assert context.metadata_profile["truncated"] is True
    assert context.metadata_rows_selected == 3
    assert {
        item["metadata_row_index"] for item in context.selected_samples
    } <= {0, 1, 2}
    assert any(
        "max_metadata_rows=3" in limitation
        for limitation in context.limitations
    )


def test_context_reads_metadata_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _create_test_dataset(tmp_path)
    original_read_csv = metadata_profiler.pd.read_csv
    read_calls = 0

    def tracked_read_csv(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(
        metadata_profiler.pd,
        "read_csv",
        tracked_read_csv,
    )

    build_identify_context(request=_request(tmp_path), seed=42)

    assert read_calls == 1


def test_unsupported_metadata_format_is_noncritical(
    tmp_path: Path,
):
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")

    context = build_identify_context(
        request=_request(tmp_path, metadata_file="metadata.json"),
        seed=42,
    )

    assert context.metadata_available is False
    assert context.selected_samples == []
    assert any(
        "Unsupported metadata format" in limitation
        for limitation in context.limitations
    )


def test_malformed_metadata_is_noncritical(tmp_path: Path):
    (tmp_path / "metadata.csv").write_text(
        'id,label\n1,"unterminated\n',
        encoding="utf-8",
    )

    context = build_identify_context(
        request=_request(tmp_path),
        seed=42,
    )

    assert context.metadata_available is False
    assert context.metadata_profile is None
    assert context.selected_samples == []
    assert any(
        "unavailable or invalid" in limitation
        for limitation in context.limitations
    )


def test_empty_metadata_selection_failure_is_noncritical(
    tmp_path: Path,
):
    (tmp_path / "metadata.csv").write_text(
        "id,label\n",
        encoding="utf-8",
    )

    context = build_identify_context(
        request=_request(tmp_path),
        seed=42,
    )

    assert context.metadata_available is True
    assert context.metadata_rows_selected == 0
    assert context.samples_inspected == 0
    assert any(
        "Metadata row selection failed" in limitation
        for limitation in context.limitations
    )


def test_metadata_symlink_escape_is_rejected(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    outside_metadata = tmp_path / "outside.csv"
    outside_metadata.write_text("id,label\n1,0\n", encoding="utf-8")

    try:
        (dataset_root / "metadata.csv").symlink_to(outside_metadata)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable in this environment: {exc}")

    context = build_identify_context(
        request=_request(dataset_root),
        seed=42,
    )

    assert context.metadata_available is False
    assert context.selected_samples == []
    assert any(
        "outside the dataset root" in limitation
        for limitation in context.limitations
    )


def test_metadata_parent_escape_is_rejected_in_context(
    tmp_path: Path,
):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (tmp_path / "outside.csv").write_text(
        "id,label\n1,0\n",
        encoding="utf-8",
    )
    request = _request(dataset_root)

    # DatasetSpec rejects this during ordinary construction. Mutating here
    # verifies the context boundary remains safe if a request reaches it from
    # an unvalidated caller or a future protocol adapter.
    request.dataset.metadata_file = "../outside.csv"

    context = build_identify_context(request=request, seed=42)

    assert context.metadata_available is False
    assert context.selected_samples == []
    assert any(
        "outside the dataset root" in limitation
        for limitation in context.limitations
    )
