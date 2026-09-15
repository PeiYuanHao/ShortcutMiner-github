from pathlib import Path

from shortcutminer.experiment.qualitative_context import build_qualitative_context
from shortcutminer.experiment.qualitative_executor import execute_qualitative
from test_qualitative_context import _dataset, _request
from test_qualitative_executor import _valid_draft


class RecordingAgentClient:
    def __init__(self, output: dict):
        self.output = output
        self.calls = []

    def invoke_agent(self, agent_name: str, payload: dict) -> dict:
        self.calls.append({"agent_name": agent_name, "payload": payload})
        return self.output


def test_qualitative_mock_runtime_integration(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _dataset(dataset_root)
    request = _request(dataset_root)
    context = build_qualitative_context(request)
    client = RecordingAgentClient(_valid_draft(request, context))

    result = execute_qualitative(request, client)

    assert result.payload.candidate_id == "C001"
    assert client.calls[0]["agent_name"] == "qualitative"
    assert client.calls[0]["payload"]["evidence_context"][
        "sample_files_available"
    ] == 4
