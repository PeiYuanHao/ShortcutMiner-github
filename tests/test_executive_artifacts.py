import json
from pathlib import Path

import pytest

from shortcutminer.protocol.executive import RoutingPlan
from shortcutminer.storage.executive_artifacts import (
    ExecutiveArtifactError,
    get_executive_run_dir,
    save_executive_input,
    save_executive_raw_output,
    save_executive_routing,
)


def _empty_plan(run_id: str = "executive_test_run") -> RoutingPlan:
    return RoutingPlan(
        protocol_version="1.0",
        run_id=run_id,
        message_id="executive:plan:empty",
        source="executive",
        target="orchestrator",
        message_type="routing_plan",
        in_reply_to="executive:request:empty",
        artifact_references=[],
        decisions=[],
        tasks=[],
        limitations=[],
    )


def test_executive_artifact_layout(tmp_path: Path):
    input_path = save_executive_input(
        tmp_path,
        run_id="executive_test_run",
        payload={"request": {"run_id": "executive_test_run"}},
    )
    raw_path = save_executive_raw_output(
        tmp_path,
        run_id="executive_test_run",
        raw_output={"message_type": "routing_plan"},
    )
    routing_path = save_executive_routing(
        tmp_path,
        _empty_plan(),
    )

    expected_dir = tmp_path / "executive_test_run" / "executive"
    assert input_path == expected_dir / "input.json"
    assert raw_path == expected_dir / "raw_output.json"
    assert routing_path == expected_dir / "routing.json"
    assert json.loads(routing_path.read_text("utf-8"))["source"] == "executive"


@pytest.mark.parametrize(
    "run_id",
    ["..", "../escape", "nested/run", r"nested\run", "CON"],
)
def test_unsafe_run_ids_are_rejected(tmp_path: Path, run_id: str):
    with pytest.raises(ExecutiveArtifactError, match="run_id"):
        get_executive_run_dir(tmp_path, run_id)


def test_non_json_output_is_rejected_without_partial_artifact(tmp_path: Path):
    with pytest.raises(ExecutiveArtifactError, match="Failed to write"):
        save_executive_raw_output(
            tmp_path,
            run_id="executive_test_run",
            raw_output={"bad": {1, 2, 3}},
        )

    output_path = (
        tmp_path
        / "executive_test_run"
        / "executive"
        / "raw_output.json"
    )
    assert not output_path.exists()
    assert list(output_path.parent.glob("*.tmp")) == []
