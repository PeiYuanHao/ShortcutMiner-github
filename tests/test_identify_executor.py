from pathlib import Path

import pytest

import json

from shortcutminer.experiment.identify_executor import (
    IdentifyExecutionError,
    execute_identify,
)
from shortcutminer.protocol.identify import (
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
    ) -> dict:
        self.calls.append(
            {
                "agent_name": agent_name,
                "payload": payload,
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
        run_id="identify_test_run",
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
        "source": "identify",
        "message_type": "candidate_report",
        "dataset_summary": {
            "dataset": "test_dataset",
            "samples_inspected": 4,
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
            "The candidate has not been validated."
        ],
    }


def test_execute_identify(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)

    request = _create_request(tmp_path)

    client = MockIdentifyClient(
        _valid_output()
    )

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

    client = MockIdentifyClient(
        _valid_output()
    )

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
        match="run_id",
    ):
        execute_identify(
            request=request,
            seed=42,
            client=client,
        )


def test_candidate_budget_enforced(
    tmp_path: Path,
):
    _create_test_dataset(tmp_path)

    request = _create_request(tmp_path)

    output = _valid_output()

    candidate = output["candidates"][0]

    output["candidates"] = [
        {
            **candidate,
            "candidate_id": "C001",
        },
        {
            **candidate,
            "candidate_id": "C002",
        },
        {
            **candidate,
            "candidate_id": "C003",
        },
    ]

    client = MockIdentifyClient(output)

    with pytest.raises(
        IdentifyExecutionError,
        match="max_candidates",
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

    output = _valid_output()

    candidate = output["candidates"][0]

    output["candidates"] = [
        candidate,
        dict(candidate),
    ]

    client = MockIdentifyClient(output)

    with pytest.raises(
        IdentifyExecutionError,
        match="duplicate candidate IDs",
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

    client = MockIdentifyClient(
        _valid_output()
    )

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

    assert input_path.exists()
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

    assert (
        input_data["run_id"]
        == request.run_id
    )

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