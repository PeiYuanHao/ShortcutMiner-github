import inspect
from importlib import import_module

import pytest

from shortcutminer.agent_client import AgentClient
from shortcutminer.orchestrator import build_workflow
from shortcutminer.runner import run_workflow
from shortcutminer.workflow import graph as graph_module
from shortcutminer.workflow.graph import create_workflow_graph
from shortcutminer.workflow.nodes import IdentifyNodeError
from shortcutminer.workflow.state import ShortcutMinerState


def _initial_state(runs_root: str | None = None) -> ShortcutMinerState:
    identify_config = {"seed": 42}
    if runs_root is not None:
        identify_config["runs_root"] = runs_root
    return ShortcutMinerState(
        run_id="phase1_smoke",
        dataset_info={"name": "synthetic", "root": "unused"},
        task_input={
            "type": "classification",
            "label_column": "label",
            "description": "workflow smoke test",
        },
        identify_config=identify_config,
        errors=[],
        metadata={"phase": 1},
    )


def test_shortcutminer_state_can_be_constructed():
    state = _initial_state()

    assert state["run_id"] == "phase1_smoke"
    assert state["dataset_info"]["name"] == "synthetic"
    assert state["errors"] == []


def test_workflow_graph_compiles():
    workflow = build_workflow()

    assert workflow is not None


def test_workflow_graph_has_expected_topology():
    graph = create_workflow_graph().compile().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert ("__start__", "identify") in edges
    assert ("identify", "executive") in edges
    assert ("executive", "feasibility") in edges
    assert ("feasibility", "quantitative") in edges
    assert ("feasibility", "qualitative") in edges
    assert ("quantitative", "evidence_calibration") in edges
    assert ("qualitative", "evidence_calibration") in edges
    assert ("evidence_calibration", "summarize") in edges
    assert ("quantitative", "summarize") not in edges
    assert ("qualitative", "summarize") not in edges
    assert ("summarize", "__end__") in edges


def test_workflow_without_identify_client_fails_explicitly(tmp_path):
    with pytest.raises(IdentifyNodeError, match="requires an AgentClient"):
        run_workflow(
            _initial_state(str(tmp_path / "runs")),
            agent_client=None,
        )


def test_graph_module_has_no_retired_platform_dependency():
    source = inspect.getsource(graph_module).lower()

    assert "open" + "claw" not in source


def test_core_modules_import_without_retired_platform():
    module_names = [
        "shortcutminer.agent_client",
        "shortcutminer.runner",
        "shortcutminer.orchestrator",
        "shortcutminer.workflow.graph",
        "shortcutminer.experiment.identify_executor",
        "shortcutminer.experiment.executive_executor",
        "shortcutminer.experiment.qualitative_executor",
        "shortcutminer.storage.events",
        "shortcutminer.evaluation.metrics",
    ]

    assert all(import_module(name) is not None for name in module_names)
    assert AgentClient is not None
    assert build_workflow is not None
    assert run_workflow is not None


def test_agent_client_protocol_has_no_runtime_specific_parameters():
    parameters = inspect.signature(AgentClient.invoke_agent).parameters

    assert list(parameters) == [
        "self",
        "agent_name",
        "payload",
        "system_prompt",
    ]
    retired_parameters = {
        "gate" + "way",
        "session" + "_id",
        "session" + "_key",
        "agent" + "_id",
    }
    assert not retired_parameters.intersection(parameters)
