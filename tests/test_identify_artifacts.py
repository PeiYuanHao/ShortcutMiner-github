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
    JsonObject,
    get_identify_run_dir,
    save_identify_input,
    save_identify_output,
    save_identify_raw_output,
)


def _create_request(
    root: Path,
) -> IdentifyRequest:
    return IdentifyRequest(
        protocol_version="1.0",
        run_id="test_run",
        message_id="identify-request-001",
        source="orchestrator",
        target="identify",
        message_type="identify_request",
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
        protocol_version="1.0",
        run_id="test_run",
        message_id="candidate-report-001",
        source="identify",
        target="executive",
        message_type="candidate_report",
        in_reply_to="identify-request-001",
        artifact_references=[
            "identify/input.json"
        ],
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
                status="unverified",
            )
        ],
        limitations=[
            "Candidate is unverified."
        ],
    )


def _create_agent_payload(
    request: IdentifyRequest,
) -> JsonObject:
    return {
        "request": request.model_dump(
            mode="json"
        ),
        "discovery_context": {
            "seed": 7,
            "selected_samples": [
                "sample-a.jpg",
                "sample-b.jpg",
            ],
            "tool_results": {
                "metadata_used": True,
            },
        },
    }


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
    payload = _create_agent_payload(
        request
    )

    path = save_identify_input(
        runs_root=tmp_path / "runs",
        run_id=request.run_id,
        payload=payload,
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

    assert (
        data["request"]["run_id"]
        == "test_run"
    )

    assert (
        data["request"]["dataset"]["name"]
        == "test_dataset"
    )

    assert (
        data["request"]["budget"]["max_samples"]
        == 4
    )

    assert (
        data["discovery_context"]["seed"]
        == 7
    )

    assert data == payload


def test_save_identify_raw_output(
    tmp_path: Path,
):
    raw_output = {
        "protocol_version": "1.0",
        "run_id": "test_run",
        "source": "identify",
        "message_type": "candidate_report",
        "candidates": [],
        "agent_note": "unvalidated response",
    }

    path = save_identify_raw_output(
        runs_root=tmp_path / "runs",
        run_id="test_run",
        raw_output=raw_output,
    )

    assert path == (
        tmp_path
        / "runs"
        / "test_run"
        / "identify"
        / "raw_output.json"
    )

    assert json.loads(
        path.read_text(
            encoding="utf-8"
        )
    ) == raw_output


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
    payload = _create_agent_payload(
        request
    )

    report = _create_report()

    input_path = save_identify_input(
        runs_root=runs_root,
        run_id=request.run_id,
        payload=payload,
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


@pytest.mark.parametrize(
    "run_id",
    [
        "..",
        "../escape",
        "nested/run",
        r"nested\run",
        ".hidden",
        "CON",
    ],
)
def test_unsafe_run_id_rejected(
    tmp_path: Path,
    run_id: str,
):
    with pytest.raises(
        IdentifyArtifactError,
        match="run_id",
    ):
        get_identify_run_dir(
            runs_root=tmp_path,
            run_id=run_id,
        )


def test_absolute_run_id_rejected(
    tmp_path: Path,
):
    with pytest.raises(
        IdentifyArtifactError,
        match="run_id",
    ):
        get_identify_run_dir(
            runs_root=tmp_path / "runs",
            run_id=str(
                (tmp_path / "escape").resolve()
            ),
        )


def test_repeated_dots_inside_run_id_are_allowed(
    tmp_path: Path,
):
    result = get_identify_run_dir(
        runs_root=tmp_path,
        run_id="run..variant",
    )

    assert result == (
        tmp_path
        / "run..variant"
        / "identify"
    )


def test_save_output_revalidates_report_run_id(
    tmp_path: Path,
):
    report = _create_report().model_copy(
        update={
            "run_id": "../escape"
        }
    )

    with pytest.raises(
        IdentifyArtifactError,
        match="run_id",
    ):
        save_identify_output(
            runs_root=tmp_path / "runs",
            report=report,
        )

    assert not (
        tmp_path
        / "escape"
        / "identify"
        / "output.json"
    ).exists()


def test_identify_directory_cannot_escape_through_symlink(
    tmp_path: Path,
):
    runs_root = tmp_path / "runs"
    outside = tmp_path / "outside"
    runs_root.mkdir()
    outside.mkdir()

    try:
        (runs_root / "linked_run").symlink_to(
            outside,
            target_is_directory=True,
        )

    except OSError:
        pytest.skip(
            "Directory symlinks are not available."
        )

    with pytest.raises(
        IdentifyArtifactError,
        match="runs_root",
    ):
        get_identify_run_dir(
            runs_root=runs_root,
            run_id="linked_run",
        )


def test_directory_creation_failure_is_wrapped(
    tmp_path: Path,
):
    runs_root = tmp_path / "runs"
    runs_root.write_text(
        "not a directory",
        encoding="utf-8",
    )

    with pytest.raises(
        IdentifyArtifactError,
        match="Failed to write artifact",
    ) as exc_info:
        save_identify_input(
            runs_root=runs_root,
            run_id="test_run",
            payload={
                "request": {
                    "run_id": "test_run"
                }
            },
        )

    assert isinstance(
        exc_info.value.__cause__,
        OSError,
    )


def test_atomic_replace_failure_is_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_replace(
        *_args,
        **_kwargs,
    ):
        raise OSError(
            "replace unavailable"
        )

    monkeypatch.setattr(
        "shortcutminer.storage.json_artifacts.os.replace",
        fail_replace,
    )

    expected_path = (
        tmp_path
        / "runs"
        / "test_run"
        / "identify"
        / "raw_output.json"
    )

    with pytest.raises(
        IdentifyArtifactError,
        match="Failed to write artifact",
    ) as exc_info:
        save_identify_raw_output(
            runs_root=tmp_path / "runs",
            run_id="test_run",
            raw_output={
                "run_id": "test_run"
            },
        )

    assert isinstance(
        exc_info.value.__cause__,
        OSError,
    )
    assert not expected_path.exists()
    assert not list(
        expected_path.parent.glob(
            "*.tmp"
        )
    )


def test_non_json_raw_output_is_wrapped(
    tmp_path: Path,
):
    with pytest.raises(
        IdentifyArtifactError,
        match="Failed to write artifact",
    ) as exc_info:
        save_identify_raw_output(
            runs_root=tmp_path / "runs",
            run_id="test_run",
            raw_output={
                "not_json": object(),  # type: ignore[dict-item]
            },
        )

    assert isinstance(
        exc_info.value.__cause__,
        TypeError,
    )
