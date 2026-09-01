import inspect
from importlib import import_module

from shortcutminer.agent_client import AgentClient
from shortcutminer.orchestrator import build_workflow
from shortcutminer.runner import run_workflow
from shortcutminer.workflow import graph as graph_module
from shortcutminer.workflow.graph import create_workflow_graph
from shortcutminer.workflow.state import ShortcutMinerState


def _initial_state() -> ShortcutMinerState:
    return ShortcutMinerState(
        run_id="phase1_smoke",
        dataset_info={"name": "synthetic"},
        task_input={"description": "workflow smoke test"},
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
    assert ("executive", "quantitative") in edges
    assert ("executive", "qualitative") in edges
    assert ("quantitative", "summarize") in edges
    assert ("qualitative", "summarize") in edges
    assert ("summarize", "__end__") in edges


def test_placeholder_workflow_completes_minimal_invocation():
    result = run_workflow(_initial_state())

    assert result["identify_result"]["status"] == "placeholder"
    assert result["executive_result"]["status"] == "placeholder"
    assert result["quantitative_result"]["status"] == "placeholder"
    assert result["qualitative_result"]["status"] == "placeholder"
    assert result["summarize_result"]["status"] == "placeholder"
    assert result["metadata"] == {"phase": 1}


def test_graph_module_has_no_openclaw_dependency():
    source = inspect.getsource(graph_module).lower()

    assert "openclaw" not in source


def test_core_modules_import_without_openclaw():
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

    assert list(parameters) == ["self", "agent_name", "payload"]
