"""Construction of the minimal ShortcutMiner LangGraph workflow."""

from langgraph.graph import END, START, StateGraph

from shortcutminer.workflow.nodes import (
    executive_node,
    identify_node,
    qualitative_node,
    quantitative_node,
    summarize_node,
)
from shortcutminer.workflow.state import ShortcutMinerState


def create_workflow_graph() -> StateGraph:
    """Create the uncompiled five-stage workflow graph."""
    graph = StateGraph(ShortcutMinerState)

    graph.add_node("identify", identify_node)
    graph.add_node("executive", executive_node)
    graph.add_node("quantitative", quantitative_node)
    graph.add_node("qualitative", qualitative_node)
    graph.add_node("summarize", summarize_node)

    graph.add_edge(START, "identify")
    graph.add_edge("identify", "executive")
    graph.add_edge("executive", "quantitative")
    graph.add_edge("executive", "qualitative")
    graph.add_edge(["quantitative", "qualitative"], "summarize")
    graph.add_edge("summarize", END)

    return graph


def build_workflow():
    """Compile and return the Phase 1 ShortcutMiner workflow."""
    return create_workflow_graph().compile()
