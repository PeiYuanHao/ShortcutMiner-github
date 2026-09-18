from pathlib import Path

from shortcutminer.experiment.executive_executor import execute_executive
from shortcutminer.protocol.executive import (
    ExecutiveCapabilities,
    ExecutiveRequest,
    ExecutiveRoutingContext,
)
from shortcutminer.protocol.identify import (
    CandidateReport,
    DatasetInspectionSummary,
    DatasetSpec,
    TaskSpec,
)


class RecordingAgentClient:
    def __init__(self, output: dict):
        self.output = output
        self.calls = []

    def invoke_agent(self, agent_name: str, payload: dict) -> dict:
        self.calls.append({"agent_name": agent_name, "payload": payload})
        return self.output


def test_executive_mock_runtime_integration(tmp_path: Path):
    report = CandidateReport(
        protocol_version="1.0",
        run_id="executive_integration_run",
        message_id="identify:report:empty",
        source="identify",
        target="executive",
        message_type="candidate_report",
        in_reply_to="identify:request:empty",
        artifact_references=[],
        dataset_summary=DatasetInspectionSummary(
            dataset="empty_dataset",
            samples_inspected=0,
            metadata_used=False,
        ),
        candidates=[],
        limitations=["No candidates were identified."],
    )
    request = ExecutiveRequest(
        protocol_version="1.0",
        run_id=report.run_id,
        message_id="executive:request:empty",
        source="orchestrator",
        target="executive",
        message_type="executive_request",
        in_reply_to=report.message_id,
        candidate_report=report,
        routing_context=ExecutiveRoutingContext(
            protocol_version="1.0",
            run_id=report.run_id,
            dataset=DatasetSpec(
                name="empty_dataset",
                root=str(tmp_path / "dataset"),
            ),
            task=TaskSpec(
                type="classification",
                label_column="label",
            ),
            capabilities=ExecutiveCapabilities(),
            artifact_references=[],
            limitations=["No candidates were identified."],
        ),
    )
    output = {
        "decisions": [],
        "limitations": ["No candidates were identified."],
    }
    client = RecordingAgentClient(output)

    plan = execute_executive(request=request, client=client)

    assert plan.tasks == []
    assert client.calls[0]["agent_name"] == "executive"
    assert client.calls[0]["payload"]["request"]["message_type"] == (
        "executive_request"
    )
