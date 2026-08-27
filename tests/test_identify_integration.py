import json
import subprocess
from pathlib import Path

from shortcutminer.experiment.identify_executor import (
    execute_identify,
)
from shortcutminer.openclaw_client import (
    OpenClawClient,
    OpenClawClientConfig,
)
from shortcutminer.openclaw_gateway_cli_transport import (
    OpenClawGatewayCliTransport,
)
from shortcutminer.protocol.identify import (
    DatasetSpec,
    IdentifyBudget,
    IdentifyRequest,
    TaskSpec,
)


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
        "protocol_version": "1.0",
        "run_id": "integration_test_run",
        "message_id": "integration_test_run:identify:report",
        "source": "identify",
        "target": "executive",
        "message_type": "candidate_report",
        "in_reply_to": "integration_test_run:identify:request",
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


def test_identify_end_to_end_mock(
    tmp_path: Path,
):
    _create_dataset(tmp_path)

    request = _create_request(
        tmp_path
    )

    report_dict = _agent_report()

    captured = {}

    def fake_runner(
        command,
        **kwargs,
    ):
        captured["command"] = command
        captured["env"] = kwargs["env"]

        message_index = (
            command.index("--message-file")
            + 1
        )

        message_path = Path(
            command[message_index]
        )

        captured["message"] = (
            message_path.read_text(
                encoding="utf-8"
            )
        )

        cli_output = {
            "payloads": [
                {
                    "text": json.dumps(
                        report_dict
                    )
                }
            ]
        }

        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps(
                cli_output
            ),
            stderr="",
        )

    transport = (
        OpenClawGatewayCliTransport(
            runner=fake_runner
        )
    )

    client = OpenClawClient(
        config=OpenClawClientConfig(
            gateway_url=(
                "ws://openclaw-gateway:18789"
            ),
            token="test-token",
            timeout_seconds=120,
        ),
        transport=transport,
    )

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

    # -----------------------------
    # Official OpenClaw CLI path
    # -----------------------------

    command = captured["command"]

    assert command[0] == "openclaw"
    assert command[1] == "agent"

    assert "--agent" in command
    assert "identify" in command

    assert "--session-key" in command

    assert (
        "shortcutminer:"
        "integration_test_run"
        in command
    )

    assert "--json" in command

    # Must use Gateway runtime.
    assert "--local" not in command

    # -----------------------------
    # Gateway configuration
    # -----------------------------

    assert (
        captured["env"][
            "OPENCLAW_GATEWAY_URL"
        ]
        == "ws://openclaw-gateway:18789"
    )

    assert (
        captured["env"][
            "OPENCLAW_GATEWAY_TOKEN"
        ]
        == "test-token"
    )

    # -----------------------------
    # Discovery context reached Agent
    # -----------------------------

    message = captured["message"]

    assert "integration_test_run" in message
    assert "test_dataset" in message
    assert "discovery_context" in message
    assert "selected_samples" in message
    assert "CandidateReport" in message
    assert "not inspected dataset samples" in message
