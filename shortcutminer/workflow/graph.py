"""Construction of the complete five-agent LangGraph workflow."""

from functools import partial

from langgraph.graph import END, START, StateGraph

from shortcutminer.agent_client import AgentClient
from shortcutminer.inspection.semantic import SemanticVisualInspector
from shortcutminer.tooling.protocol import AgentName
from shortcutminer.tooling.runtime import ToolRuntime
from shortcutminer.workflow.nodes import (
    executive_node,
    feasibility_node,
    identify_node,
    qualitative_node,
    quantitative_node,
    summarize_node,
)
from shortcutminer.workflow.state import ShortcutMinerState


def create_workflow_graph(
    identify_client: AgentClient | None = None,
    *,
    agent_client: AgentClient | None = None,
    executive_client: AgentClient | None = None,
    quantitative_client: AgentClient | None = None,
    qualitative_client: AgentClient | None = None,
    summarize_client: AgentClient | None = None,
    tool_runtimes: dict[AgentName, ToolRuntime] | None = None,
    semantic_inspector: SemanticVisualInspector | None = None,
) -> StateGraph:
    """Create the graph while keeping all runtime objects outside state."""
    shared_client = (
        agent_client if agent_client is not None else identify_client
    )

    def select_client(specific: AgentClient | None) -> AgentClient | None:
        return specific if specific is not None else shared_client

    runtimes = tool_runtimes or {}

    graph = StateGraph(ShortcutMinerState)

    graph.add_node(
        "identify",
        partial(
            identify_node,
            client=select_client(identify_client),
            tool_runtime=runtimes.get("identify"),
            semantic_inspector=semantic_inspector,
        ),
    )
    graph.add_node(
        "executive",
        partial(
            executive_node,
            client=select_client(executive_client),
            tool_runtime=runtimes.get("executive"),
        ),
    )
    graph.add_node(
        "quantitative",
        partial(
            quantitative_node,
            client=select_client(quantitative_client),
            tool_runtime=runtimes.get("quantitative"),
        ),
    )
    graph.add_node(
        "feasibility",
        partial(
            feasibility_node,
            quantitative_tool_runtime=runtimes.get("quantitative"),
        ),
    )
    graph.add_node(
        "qualitative",
        partial(
            qualitative_node,
            client=select_client(qualitative_client),
            tool_runtime=runtimes.get("qualitative"),
        ),
    )
    graph.add_node(
        "summarize",
        partial(
            summarize_node,
            client=select_client(summarize_client),
            tool_runtime=runtimes.get("summarize"),
        ),
    )

    graph.add_edge(START, "identify")
    graph.add_edge("identify", "executive")
    graph.add_edge("executive", "feasibility")
    graph.add_edge("feasibility", "quantitative")
    graph.add_edge("feasibility", "qualitative")
    graph.add_edge(["quantitative", "qualitative"], "summarize")
    graph.add_edge("summarize", END)

    return graph


def build_workflow(
    identify_client: AgentClient | None = None,
    checkpointer=None,
    **clients,
):
    """Compile the workflow with injected, runtime-agnostic clients."""
    graph = create_workflow_graph(
        identify_client=identify_client,
        **clients,
    )
    return graph.compile(checkpointer=checkpointer)
