from pathlib import Path

from shortcutminer.experiment.identify_context import (
    build_identify_context,
)
from shortcutminer.protocol.identify import (
    DatasetSpec,
    IdentifyBudget,
    IdentifyRequest,
    TaskSpec,
)


def _create_test_dataset(
    root: Path,
) -> None:
    train_dir = root / "train"
    train_dir.mkdir()

    for sample_id in range(1, 9):
        (
            train_dir
            / f"sample_{sample_id}.jpg"
        ).write_bytes(b"test")

    metadata = root / "metadata.csv"

    metadata.write_text(
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


def test_build_identify_context(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)

    request = IdentifyRequest(
        run_id="test_run",
        dataset=DatasetSpec(
            name="test_dataset",
            root=str(tmp_path),
            metadata_file="metadata.csv",
        ),
        task=TaskSpec(
            type="classification",
            label_column="label",
        ),
        budget=IdentifyBudget(
            max_samples=4,
            max_candidates=5,
        ),
    )

    context = build_identify_context(
        request=request,
        seed=42,
    )

    assert context.run_id == "test_run"
    assert context.dataset_name == "test_dataset"

    assert (
        context.dataset_inspection["exists"]
        is True
    )

    assert context.metadata_profile is not None
    assert context.metadata_profile["exists"] is True

    assert len(context.selected_samples) == 4

    labels = [
        sample["label"]
        for sample in context.selected_samples
    ]

    assert labels.count(0) == 2
    assert labels.count(1) == 2


def test_context_is_reproducible(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)

    request = IdentifyRequest(
        run_id="test_run",
        dataset=DatasetSpec(
            name="test_dataset",
            root=str(tmp_path),
            metadata_file="metadata.csv",
        ),
        task=TaskSpec(
            type="classification",
            label_column="label",
        ),
        budget=IdentifyBudget(
            max_samples=4,
            max_candidates=5,
        ),
    )

    context_a = build_identify_context(
        request=request,
        seed=42,
    )

    context_b = build_identify_context(
        request=request,
        seed=42,
    )

    assert (
        context_a.selected_samples
        == context_b.selected_samples
    )


def test_missing_dataset(
    tmp_path: Path,
):
    missing_root = (
        tmp_path / "missing_dataset"
    )

    request = IdentifyRequest(
        run_id="missing_run",
        dataset=DatasetSpec(
            name="missing",
            root=str(missing_root),
            metadata_file="metadata.csv",
        ),
        task=TaskSpec(
            type="classification",
            label_column="label",
        ),
    )

    context = build_identify_context(
        request=request,
        seed=42,
    )

    assert (
        context.dataset_inspection["exists"]
        is False
    )

    assert context.selected_samples == []

    assert (
        "Dataset root is not accessible."
        in context.limitations
    )


def test_missing_metadata(
    tmp_path: Path,
):
    request = IdentifyRequest(
        run_id="no_metadata",
        dataset=DatasetSpec(
            name="test_dataset",
            root=str(tmp_path),
            metadata_file=None,
        ),
        task=TaskSpec(
            type="classification",
            label_column="label",
        ),
    )

    context = build_identify_context(
        request=request,
        seed=42,
    )

    assert context.metadata_profile is None
    assert context.selected_samples == []

    assert (
        "No metadata file was configured."
        in context.limitations
    )


def test_sampling_fallback(
    tmp_path: Path,
):
    metadata = tmp_path / "metadata.csv"

    metadata.write_text(
        "id,site\n"
        "1,A\n"
        "2,A\n"
        "3,B\n"
        "4,B\n",
        encoding="utf-8",
    )

    request = IdentifyRequest(
        run_id="fallback_run",
        dataset=DatasetSpec(
            name="test_dataset",
            root=str(tmp_path),
            metadata_file="metadata.csv",
        ),
        task=TaskSpec(
            type="classification",
            label_column="missing_label",
        ),
        budget=IdentifyBudget(
            max_samples=2,
            max_candidates=5,
        ),
    )

    context = build_identify_context(
        request=request,
        seed=42,
    )

    assert len(context.selected_samples) == 2

    assert any(
        "Class-balanced sampling failed"
        in limitation
        for limitation in context.limitations
    )

    assert (
        "Random sampling was used as fallback."
        in context.limitations
    )