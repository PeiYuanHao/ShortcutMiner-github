from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from shortcutminer.experiment.audit_intake import intake_dataset
from shortcutminer.protocol.audit import AuditIntakeBudget, AuditLabelInformation, AuditRequest


def test_single_csv_binds_only_one_conventional_label(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("id,label,device\n1,0,A\n2,1,B\n", encoding="utf-8")
    request = intake_dataset(AuditRequest(dataset_path=str(path)))
    info = request.dataset_info
    assert info.dataset_type == "tabular"
    assert info.root == str(tmp_path.resolve())
    assert info.selected_metadata_file == "data.csv"
    assert info.label_source.status == "resolved"
    assert info.label_source.column == "label"
    assert info.entries_scanned == 1


@pytest.mark.parametrize("header", ["id,score,device", "id,label,target"])
def test_no_arbitrary_or_ambiguous_label_guess(tmp_path, header):
    (tmp_path / "metadata.csv").write_text(header + "\n1,0,A\n", encoding="utf-8")
    info = intake_dataset(AuditRequest(dataset_path=str(tmp_path))).dataset_info
    assert info.label_source.status == "unresolved"
    assert info.label_source.column is None
    assert any("label_source_unresolved" in item for item in info.limitations)


def test_explicit_nonstandard_label_is_exact_and_optional(tmp_path):
    (tmp_path / "table.csv").write_text("id,diagnosis\n1,positive\n", encoding="utf-8")
    for column, expected in [("diagnosis", "resolved"), ("Diagnosis", "unresolved")]:
        request = AuditRequest(dataset_path=str(tmp_path), label_information=
                               AuditLabelInformation(source="metadata_column", column=column))
        info = intake_dataset(request).dataset_info
        assert info.label_source.status == expected


def test_multiple_metadata_sources_require_explicit_selection(tmp_path):
    for name in ["one.csv", "two.csv"]:
        (tmp_path / name).write_text("label\n0\n", encoding="utf-8")
    info = intake_dataset(AuditRequest(dataset_path=str(tmp_path))).dataset_info
    assert info.selected_metadata_file is None
    assert info.label_source.status == "unresolved"
    selected = intake_dataset(AuditRequest(dataset_path=str(tmp_path), metadata_file="two.csv"))
    assert selected.dataset_info.selected_metadata_file == "two.csv"
    assert selected.dataset_info.label_source.column == "label"


@pytest.mark.parametrize("splits,kind", [([""], "class_directory"),
                                         (["train", "test"], "split_class_directory")])
def test_explicit_directory_label_structures_are_recorded(tmp_path, splits, kind):
    for split in splits:
        for label in ["cats", "dogs"]:
            directory = tmp_path / split / label
            directory.mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(directory / "sample.png")
    info = intake_dataset(AuditRequest(dataset_path=str(tmp_path))).dataset_info
    assert info.dataset_type == "image"
    assert info.label_source.status == "resolved"
    assert info.label_source.kind == kind
    assert info.label_source.classes == ["cats", "dogs"]


def test_generic_image_folders_are_not_class_labels(tmp_path):
    for folder in ["images", "samples"]:
        (tmp_path / folder).mkdir()
        Image.new("RGB", (4, 4)).save(tmp_path / folder / "sample.png")
    assert intake_dataset(AuditRequest(dataset_path=str(tmp_path))).dataset_info.label_source.status == "unresolved"


def test_conflicting_sources_remain_unresolved(tmp_path):
    for label in ["cats", "dogs"]:
        (tmp_path / label).mkdir()
        Image.new("RGB", (4, 4)).save(tmp_path / label / "sample.png")
    (tmp_path / "metadata.csv").write_text("label\n0\n", encoding="utf-8")
    request = intake_dataset(AuditRequest(dataset_path=str(tmp_path)))
    assert request.dataset_info.label_source.status == "unresolved"
    assert len(request.dataset_info.label_source.possible_sources) == 2


def test_bounded_scan_never_resolves_from_partial_structure(tmp_path):
    for index in range(5):
        (tmp_path / f"file{index}.csv").write_text("label\n0\n", encoding="utf-8")
    budget = AuditIntakeBudget(max_entries=2, max_metadata_files=1)
    info = intake_dataset(AuditRequest(dataset_path=str(tmp_path), intake_budget=budget)).dataset_info
    assert info.entries_scanned <= 2
    assert info.truncated
    assert len(info.metadata_files) <= 1
    assert info.label_source.status == "unresolved"


def test_csv_header_byte_budget_and_parse_errors_are_limitations(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a" * 100, encoding="utf-8")
    request = AuditRequest(dataset_path=str(path), intake_budget=AuditIntakeBudget(max_header_bytes=16))
    info = intake_dataset(request).dataset_info
    assert info.selected_metadata_file is None
    assert any("byte budget" in item for item in info.limitations)


def test_unsupported_table_format_is_discovered_without_loading(tmp_path):
    path = tmp_path / "data.parquet"
    path.write_bytes(b"not a parquet file")
    info = intake_dataset(AuditRequest(dataset_path=str(path))).dataset_info
    assert info.dataset_type == "tabular"
    assert info.metadata_files == ["data.parquet"]
    assert info.label_source.status == "unresolved"
    assert any("metadata_format_unsupported" in item for item in info.limitations)


def test_invalid_paths_and_traversal(tmp_path):
    with pytest.raises(FileNotFoundError):
        intake_dataset(AuditRequest(dataset_path=str(tmp_path / "missing")))
    with pytest.raises(ValidationError):
        AuditRequest(dataset_path=str(tmp_path), metadata_file="../outside.csv")
    image = tmp_path / "one.png"
    Image.new("RGB", (4, 4)).save(image)
    with pytest.raises(ValueError, match="dataset directory"):
        intake_dataset(AuditRequest(dataset_path=str(image)))


def test_intake_rechecks_caller_supplied_facts(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("label\n0\n", encoding="utf-8")
    request = intake_dataset(AuditRequest(dataset_path=str(path)))
    path.write_text("unknown\n0\n", encoding="utf-8")
    assert intake_dataset(request).dataset_info.label_source.status == "unresolved"
