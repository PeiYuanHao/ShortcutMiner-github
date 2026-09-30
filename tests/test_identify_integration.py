from pathlib import Path

from shortcutminer.experiment.identify_executor import (
    execute_identify,
)
from shortcutminer.protocol.identify import (
    DatasetSpec,
    IdentifyBudget,
    IdentifyRequest,
    TaskSpec,
)


class RecordingAgentClient:
    def __init__(self, output: dict):
        self.output = output
        self.calls: list[dict] = []

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


def _create_dataset(root: Path) -> None:
    train_dir = root / "train"
    train_dir.mkdir()

    for sample_id in range(1, 5):
        (
            train_dir
            / f"sample_{sample_id}.jpg"
        ).write_bytes(b"test")

    metadata_path = root / "metadata.csv"

    metadata_path.write_text(
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
        run_id="integration_test_run",
        message_id="integration_test_run:identify:request",
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


def _agent_report() -> dict:
    return {
        "candidates": [
            {
                "cue": {
                    "family": "metadata",
                    "name": "site",
                },
                "description": (
                    "Dataset site may provide an "
                    "unintended predictive cue."
                ),
                "observation": (
                    "Multiple site values are present "
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
                "metadata_field_ref": "site",
                "additional_limitations": [],
            }
        ],
        "additional_limitations": ["The candidate has not been validated."],
    }


def test_identify_end_to_end_mock(
    tmp_path: Path,
):
    _create_dataset(tmp_path)

    request = _create_request(
        tmp_path
    )

    report_dict = _agent_report()

    client = RecordingAgentClient(report_dict)

    report = execute_identify(
        request=request,
        seed=42,
        client=client,
    )

    # -----------------------------
    # Final CandidateReport
    # -----------------------------

    assert (
        report.run_id
        == "integration_test_run"
    )

    assert (
        report.dataset_summary.dataset
        == "test_dataset"
    )

    assert len(report.candidates) == 1
    assert report.dataset_summary.samples_inspected == 0

    candidate = report.candidates[0]

    assert (
        candidate.candidate_id
        == "C001"
    )

    assert (
        candidate.status
        == "unverified"
    )

    call = client.calls[0]
    assert call["agent_name"] == "identify"
    assert call["payload"]["request"]["run_id"] == "integration_test_run"
    assert call["payload"]["request"]["dataset"]["name"] == "test_dataset"
    assert "discovery_context" in call["payload"]
    assert "selected_samples" in call["payload"]["discovery_context"]
    assert "Identify Agent Operational Specification" in call["system_prompt"]
