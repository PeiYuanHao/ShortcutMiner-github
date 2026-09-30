import json
from pathlib import Path

import pytest

from shortcutminer.experiment.identify_context import build_identify_context
from shortcutminer.experiment.identify_executor import (
    IdentifyExecutionError,
    execute_identify,
    validate_candidate_report,
)
from shortcutminer.protocol.identify import (
    CandidateReport,
    DatasetSpec,
    IdentifyBudget,
    IdentifyRequest,
    TaskSpec,
)


class MockIdentifyClient:
    def __init__(self, output: dict):
        self.output = output
        self.calls = []

    def invoke_agent(
        self,
        agent_name: str,
        payload: dict,
        *,
        system_prompt: str | None = None,
    ) -> dict:
        self.calls.append(
            {
                "agent_name": agent_name,
                "payload": payload,
                "system_prompt": system_prompt,
            }
        )

        return self.output


def _create_test_dataset(
    root: Path,
) -> None:
    train_dir = root / "train"
    train_dir.mkdir()

    for sample_id in range(1, 5):
        (
            train_dir
            / f"sample_{sample_id}.jpg"
        ).write_bytes(b"test")

    metadata = root / "metadata.csv"

    metadata.write_text(
        "id,label,site\n"
        "1,0,A\n"
        "2,0,A\n"
        "3,1,B\n"
        "4,1,B\n",
        encoding="utf-8",
    )


def _create_request(
    root: Path,
) -> IdentifyRequest:
    return IdentifyRequest(
        protocol_version="1.0",
        run_id="identify_test_run",
        message_id="identify_test_run:identify:request",
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


def _valid_output() -> dict:
    return {
        "protocol_version": "1.0",
        "run_id": "identify_test_run",
        "message_id": "identify_test_run:identify:report",
        "source": "identify",
        "target": "executive",
        "message_type": "candidate_report",
        "in_reply_to": "identify_test_run:identify:request",
        "artifact_references": [],
        "dataset_summary": {
            "dataset": "test_dataset",
            "samples_inspected": 0,
            "metadata_used": True,
        },
        "candidates": [
            {
                "candidate_id": "C001",
                "cue": {
                    "family": "metadata",
                    "name": "site",
                },
                "description": (
                    "Dataset source site may provide "
                    "an unintended predictive cue."
                ),
                "observation": (
                    "Different site values are present "
                    "in the inspected metadata."
                ),
                "hypothesis": (
                    "Site identity may be spuriously "
                    "associated with the target label."
                ),
                "evidence_sources": [
                    "metadata"
                ],
                "priority": "medium",
                "status": "unverified",
            }
        ],
        "limitations": [
            "The candidate has not been validated.",
            (
                "Selected records are metadata rows only; "
                "no dataset sample content was inspected."
            ),
        ],
    }


def _valid_draft() -> dict:
    candidate = _valid_output()["candidates"][0]
    return {
        "candidates": [{
            "cue": candidate["cue"],
            "description": candidate["description"],
            "observation": candidate["observation"],
            "hypothesis": candidate["hypothesis"],
            "evidence_sources": candidate["evidence_sources"],
            "priority": candidate["priority"],
            "metadata_field_ref": "site",
            "additional_limitations": [],
        }],
        "additional_limitations": ["The candidate has not been validated."],
    }


def test_execute_identify(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)

    request = _create_request(tmp_path)

    client = MockIdentifyClient(_valid_draft())

    report = execute_identify(
        request=request,
        seed=42,
        client=client,
    )

    assert report.run_id == "identify_test_run"
    assert len(report.candidates) == 1
    assert report.candidates[0].status == "unverified"


def test_executor_sends_structured_payload(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)

    request = _create_request(tmp_path)

    client = MockIdentifyClient(_valid_draft())

    execute_identify(
        request=request,
        seed=42,
        client=client,
    )

    assert len(client.calls) == 1

    call = client.calls[0]

    assert call["agent_name"] == "identify"

    payload = call["payload"]

    assert "request" in payload
    assert "discovery_context" in payload
    assert set(payload) == {"request", "discovery_context"}
    assert "Identify Agent Operational Specification" in call["system_prompt"]

    assert (
        payload["request"]["run_id"]
        == "identify_test_run"
    )

    assert (
        payload["discovery_context"]["dataset_name"]
        == "test_dataset"
    )


def test_invalid_agent_output(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)

    request = _create_request(tmp_path)

    client = MockIdentifyClient(
        {
            "this": "is not a CandidateReport"
        }
    )

    with pytest.raises(
        IdentifyExecutionError
    ):
        execute_identify(
            request=request,
            seed=42,
            client=client,
        )


def test_wrong_run_id(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)

    request = _create_request(tmp_path)

    output = _valid_output()
    output["run_id"] = "wrong_run"

    client = MockIdentifyClient(output)

    with pytest.raises(
        IdentifyExecutionError,
        match="required draft schema",
    ):
        execute_identify(
            request=request,
            seed=42,
            client=client,
        )


def test_wrong_reply_message_id(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)
    request = _create_request(tmp_path)
    output = _valid_output()
    output["in_reply_to"] = "another:request"

    with pytest.raises(
        IdentifyExecutionError,
        match="required draft schema",
    ):
        execute_identify(
            request=request,
            seed=42,
            client=MockIdentifyClient(output),
        )


def test_samples_inspected_must_match_context(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)
    request = _create_request(tmp_path)
    output = _valid_output()
    output["dataset_summary"]["samples_inspected"] = 1

    with pytest.raises(
        IdentifyExecutionError,
        match="required draft schema",
    ):
        execute_identify(
            request=request,
            seed=42,
            client=MockIdentifyClient(output),
        )


def test_context_limitations_must_be_preserved(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)
    request = _create_request(tmp_path)
    output = _valid_output()
    output["limitations"] = [
        "The candidate has not been validated."
    ]

    with pytest.raises(
        IdentifyExecutionError,
        match="required draft schema",
    ):
        execute_identify(
            request=request,
            seed=42,
            client=MockIdentifyClient(output),
        )


def test_candidate_ids_must_be_sequential(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)
    request = _create_request(tmp_path)
    output = _valid_output()
    output["candidates"][0]["candidate_id"] = "C002"

    with pytest.raises(
        IdentifyExecutionError,
        match="required draft schema",
    ):
        execute_identify(
            request=request,
            seed=42,
            client=MockIdentifyClient(output),
        )


def test_unavailable_evidence_source_is_rejected(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)
    request = _create_request(tmp_path)
    output = _valid_draft()
    output["candidates"][0]["evidence_sources"] = [
        "sample_inspection"
    ]

    with pytest.raises(
        IdentifyExecutionError,
        match="sample_inspection_unavailable",
    ):
        execute_identify(
            request=request,
            seed=42,
            client=MockIdentifyClient(output),
        )


def test_unsupplied_artifact_reference_is_rejected(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)
    request = _create_request(tmp_path)
    output = _valid_output()
    output["artifact_references"] = ["identify/input.json"]

    with pytest.raises(
        IdentifyExecutionError,
        match="required draft schema",
    ):
        execute_identify(
            request=request,
            seed=42,
            client=MockIdentifyClient(output),
        )


def test_duplicate_evidence_sources_are_rejected(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)
    request = _create_request(tmp_path)
    output = _valid_draft()
    output["candidates"][0]["evidence_sources"] = [
        "metadata",
        "metadata",
    ]

    with pytest.raises(
        IdentifyExecutionError,
        match="required draft schema",
    ):
        execute_identify(
            request=request,
            seed=42,
            client=MockIdentifyClient(output),
        )


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    [
        ("dataset_name", "other_dataset"),
        ("task_type", "regression"),
        ("label_column", "other_label"),
    ],
)
def test_context_identity_must_match_request(
    tmp_path: Path,
    field_name: str,
    wrong_value: str,
):
    _create_test_dataset(tmp_path)
    request = _create_request(tmp_path)
    context = build_identify_context(request=request, seed=42)
    setattr(context, field_name, wrong_value)

    with pytest.raises(IdentifyExecutionError, match=field_name):
        validate_candidate_report(
            report=CandidateReport.model_validate(_valid_output()),
            request=request,
            context=context,
        )


def test_context_metadata_selection_count_is_validated(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)
    request = _create_request(tmp_path)
    context = build_identify_context(request=request, seed=42)
    context.metadata_rows_selected += 1

    with pytest.raises(
        IdentifyExecutionError,
        match="metadata_rows_selected",
    ):
        validate_candidate_report(
            report=CandidateReport.model_validate(_valid_output()),
            request=request,
            context=context,
        )


def test_candidate_budget_enforced(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)

    request = _create_request(tmp_path)

    output = _valid_draft()

    candidate = output["candidates"][0]

    output["candidates"] = [
        {
            **candidate,
            "metadata_field_ref": "site_a",
        },
        {
            **candidate,
            "metadata_field_ref": "site_b",
        },
        {
            **candidate,
            "metadata_field_ref": "site_c",
        },
    ]

    client = MockIdentifyClient(output)

    with pytest.raises(
        IdentifyExecutionError,
        match="candidate_budget_exceeded",
    ):
        execute_identify(
            request=request,
            seed=42,
            client=client,
        )


def test_duplicate_candidate_ids(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)

    request = _create_request(tmp_path)

    output = _valid_draft()

    candidate = output["candidates"][0]

    output["candidates"] = [
        candidate,
        dict(candidate),
    ]

    client = MockIdentifyClient(output)

    with pytest.raises(
        IdentifyExecutionError,
        match="duplicate_metadata_field_selector",
    ):
        execute_identify(
            request=request,
            seed=42,
            client=client,
        )


def test_execute_identify_persists_artifacts(
    tmp_path: Path,
):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    _create_test_dataset(dataset_root)

    request = _create_request(
        dataset_root
    )

    client = MockIdentifyClient(_valid_draft())

    runs_root = tmp_path / "runs"

    report = execute_identify(
        request=request,
        seed=42,
        client=client,
        runs_root=runs_root,
    )

    identify_dir = (
        runs_root
        / request.run_id
        / "identify"
    )

    input_path = (
        identify_dir
        / "input.json"
    )

    output_path = (
        identify_dir
        / "output.json"
    )

    raw_output_path = (
        identify_dir
        / "raw_output.json"
    )

    assert input_path.exists()
    assert raw_output_path.exists()
    assert output_path.exists()

    input_data = json.loads(
        input_path.read_text(
            encoding="utf-8"
        )
    )

    output_data = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    raw_output_data = json.loads(
        raw_output_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        input_data["request"]["run_id"]
        == request.run_id
    )

    assert input_data["discovery_context"]["seed"] == 42
    assert (
        input_data["discovery_context"]["samples_inspected"]
        == 0
    )
    assert raw_output_data == _valid_draft()

    assert (
        output_data["run_id"]
        == report.run_id
    )

    assert (
        output_data["message_type"]
        == "candidate_report"
    )


def test_failed_identify_does_not_persist_output(
    tmp_path: Path,
):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    _create_test_dataset(dataset_root)

    request = _create_request(
        dataset_root
    )

    client = MockIdentifyClient(
        {
            "invalid": "agent output"
        }
    )

    runs_root = tmp_path / "runs"

    with pytest.raises(
        IdentifyExecutionError
    ):
        execute_identify(
            request=request,
            seed=42,
            client=client,
            runs_root=runs_root,
        )

    identify_dir = (
        runs_root
        / request.run_id
        / "identify"
    )

    assert (
        identify_dir
        / "input.json"
    ).exists()

    assert not (
        identify_dir
        / "output.json"
    ).exists()

    assert not (
        identify_dir
        / "raw_output.json"
    ).exists()


def test_success_events_are_persisted(
    tmp_path: Path,
):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    _create_test_dataset(
        dataset_root
    )

    request = _create_request(
        dataset_root
    )

    client = MockIdentifyClient(_valid_draft())

    runs_root = tmp_path / "runs"

    execute_identify(
        request=request,
        seed=42,
        client=client,
        runs_root=runs_root,
    )

    events_path = (
        runs_root
        / request.run_id
        / "events.jsonl"
    )

    events = [
        json.loads(line)
        for line in events_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert [
        event["event"]
        for event in events
    ] == [
        "IDENTIFY_STARTED",
        "IDENTIFY_COMPLETED",
    ]


def test_failure_event_is_persisted(
    tmp_path: Path,
):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    _create_test_dataset(
        dataset_root
    )

    request = _create_request(
        dataset_root
    )

    client = MockIdentifyClient(
        {
            "invalid": "output"
        }
    )

    runs_root = tmp_path / "runs"

    with pytest.raises(
        IdentifyExecutionError
    ):
        execute_identify(
            request=request,
            seed=42,
            client=client,
            runs_root=runs_root,
        )

    events_path = (
        runs_root
        / request.run_id
        / "events.jsonl"
    )

    events = [
        json.loads(line)
        for line in events_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert (
        events[0]["event"]
        == "IDENTIFY_STARTED"
    )

    assert (
        events[-1]["event"]
        == "AGENT_FAILED"
    )
