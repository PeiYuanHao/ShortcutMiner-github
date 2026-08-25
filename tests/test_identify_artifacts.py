import json
from pathlib import Path

import pytest

from shortcutminer.protocol.identify import (
    Candidate,
    CandidateReport,
    Cue,
    DatasetInspectionSummary,
    DatasetSpec,
    IdentifyBudget,
    IdentifyRequest,
    TaskSpec,
)
from shortcutminer.storage.identify_artifacts import (
    IdentifyArtifactError,
    get_identify_run_dir,
    save_identify_input,
    save_identify_output,
)


def _create_request(
    root: Path,
) -> IdentifyRequest:
    return IdentifyRequest(
        run_id="test_run",
        dataset=DatasetSpec(
            name="test_dataset",
            root=str(root),
            metadata_file="metadata.csv",
        ),
        task=TaskSpec(
            type="classification",
            label_column="label",
        ),
        budget=IdentifyBudget(
            max_samples=4,
            max_candidates=2,
        ),
    )


def _create_report() -> CandidateReport:
    return CandidateReport(
        run_id="test_run",
        dataset_summary=DatasetInspectionSummary(
            dataset="test_dataset",
            samples_inspected=4,
            metadata_used=True,
        ),
        candidates=[
            Candidate(
                candidate_id="C001",
                cue=Cue(
                    family="metadata",
                    name="site",
                ),
                description=(
                    "Site may provide an unintended cue."
                ),
                observation=(
                    "Multiple site values were observed."
                ),
                hypothesis=(
                    "Site identity may be spuriously "
                    "associated with the target label."
                ),
                evidence_sources=[
                    "metadata"
                ],
                priority="medium",
            )
        ],
        limitations=[
            "Candidate is unverified."
        ],
    )


def test_get_identify_run_dir(
    tmp_path: Path,
):
    result = get_identify_run_dir(
        runs_root=tmp_path,
        run_id="test_run",
    )

    assert result == (
        tmp_path
        / "test_run"
        / "identify"
    )


def test_save_identify_input(
    tmp_path: Path,
):
    request = _create_request(
        tmp_path
    )

    path = save_identify_input(
        runs_root=tmp_path / "runs",
        request=request,
    )

    assert path.exists()

    assert path == (
        tmp_path
        / "runs"
        / "test_run"
        / "identify"
        / "input.json"
    )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert data["run_id"] == "test_run"

    assert (
        data["dataset"]["name"]
        == "test_dataset"
    )

    assert (
        data["budget"]["max_samples"]
        == 4
    )


def test_save_identify_output(
    tmp_path: Path,
):
    report = _create_report()

    path = save_identify_output(
        runs_root=tmp_path / "runs",
        report=report,
    )

    assert path.exists()

    assert path == (
        tmp_path
        / "runs"
        / "test_run"
        / "identify"
        / "output.json"
    )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert data["run_id"] == "test_run"

    assert (
        data["message_type"]
        == "candidate_report"
    )

    assert len(
        data["candidates"]
    ) == 1

    assert (
        data["candidates"][0]["status"]
        == "unverified"
    )


def test_input_and_output_share_run_directory(
    tmp_path: Path,
):
    runs_root = (
        tmp_path / "runs"
    )

    request = _create_request(
        tmp_path
    )

    report = _create_report()

    input_path = save_identify_input(
        runs_root=runs_root,
        request=request,
    )

    output_path = save_identify_output(
        runs_root=runs_root,
        report=report,
    )

    assert (
        input_path.parent
        == output_path.parent
    )

    assert (
        input_path.parent.name
        == "identify"
    )


def test_existing_output_is_replaced(
    tmp_path: Path,
):
    runs_root = (
        tmp_path / "runs"
    )

    report = _create_report()

    path = save_identify_output(
        runs_root=runs_root,
        report=report,
    )

    first_content = path.read_text(
        encoding="utf-8"
    )

    report.limitations.append(
        "Second write."
    )

    save_identify_output(
        runs_root=runs_root,
        report=report,
    )

    second_content = path.read_text(
        encoding="utf-8"
    )

    assert (
        first_content
        != second_content
    )

    data = json.loads(
        second_content
    )

    assert (
        "Second write."
        in data["limitations"]
    )


def test_empty_run_id_rejected(
    tmp_path: Path,
):
    with pytest.raises(
        IdentifyArtifactError,
        match="run_id",
    ):
        get_identify_run_dir(
            runs_root=tmp_path,
            run_id="",
        )