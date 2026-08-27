from pathlib import Path

from shortcutminer.experiment.qualitative_context import build_qualitative_context
from shortcutminer.experiment.qualitative_executor import execute_qualitative
from shortcutminer.openclaw_client import OpenClawClient, OpenClawClientConfig
from test_qualitative_context import _dataset, _request
from test_qualitative_executor import _valid_output


class RecordingTransport:
    def __init__(self, output: dict):
        self.output = output
        self.calls = []

    def run_agent(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return self.output


def test_qualitative_mock_runtime_integration(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _dataset(dataset_root)
    request = _request(dataset_root)
    context = build_qualitative_context(request)
    transport = RecordingTransport(_valid_output(request, context))
    client = OpenClawClient(
        config=OpenClawClientConfig(
            gateway_url="ws://mock-gateway",
            token="mock-token",
        ),
        transport=transport,
    )

    result = execute_qualitative(request, client)

    assert result.payload.candidate_id == "C001"
    assert transport.calls[0]["agent_name"] == "qualitative"
    assert transport.calls[0]["session_key"] == (
        "shortcutminer:qualitative_test_run"
    )
    assert transport.calls[0]["payload"]["evidence_context"][
        "sample_files_available"
    ] == 4
