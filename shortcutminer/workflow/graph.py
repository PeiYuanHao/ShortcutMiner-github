"""Construction of the complete five-agent LangGraph workflow."""

from functools import partial

from langgraph.graph import END, START, StateGraph

from shortcutminer.agent_client import AgentClient
from shortcutminer.inspection.semantic import SemanticVisualInspector
from shortcutminer.tooling.protocol import AgentName
from shortcutminer.tooling.runtime import ToolRuntime
from shortcutminer.workflow.nodes import (
    audit_capabilities_node,
    candidate_screening_node,
    evidence_calibration_node,
    executive_node,
    feasibility_node,
    identify_node,
    qualitative_node,
    quantitative_node,
    summarize_node,
)
from shortcutminer.workflow.state import ShortcutMinerState
from shortcutminer.workflow.observability import observe_node
from shortcutminer.workflow.staged_validation import stage_plan_node, stage1_node, promotion_node, stage2_node


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

    def add_node(stage, function):
        graph.add_node(stage, observe_node(stage, function))

    add_node(
        "identify",
        partial(
            identify_node,
            client=select_client(identify_client),
            tool_runtime=runtimes.get("identify"),
            semantic_inspector=semantic_inspector,
        ),
    )
    add_node(
        "executive",
        partial(
            executive_node,
            client=select_client(executive_client),
            tool_runtime=runtimes.get("executive"),
            quantitative_tool_runtime=runtimes.get("quantitative"),
            qualitative_tool_runtime=runtimes.get("qualitative"),
        ),
    )
    add_node("screening", candidate_screening_node)
    add_node("capabilities", partial(audit_capabilities_node,
                   quantitative_tool_runtime=runtimes.get("quantitative"),
                   qualitative_tool_runtime=runtimes.get("qualitative")))
    add_node(
        "quantitative",
        partial(
            stage1_node, function=quantitative_node,
            client=select_client(quantitative_client),
            tool_runtime=runtimes.get("quantitative"),
        ),
    )
    add_node(
        "feasibility",
        partial(
            feasibility_node,
            quantitative_tool_runtime=runtimes.get("quantitative"),
        ),
    )
    add_node(
        "qualitative",
        partial(
            stage1_node, function=qualitative_node,
            client=select_client(qualitative_client),
            tool_runtime=runtimes.get("qualitative"),
        ),
    )
    add_node(
        "evidence_calibration",
        evidence_calibration_node,
    )
    add_node("validation_stage_plan", stage_plan_node)
    add_node("evidence_promotion", promotion_node)
    graph.add_node("stage2_validation", partial(stage2_node,
        quantitative_client=select_client(quantitative_client), qualitative_client=select_client(qualitative_client),
        quantitative_runtime=runtimes.get("quantitative"), qualitative_runtime=runtimes.get("qualitative")))
    add_node(
        "summarize",
        partial(
            summarize_node,
            client=select_client(summarize_client),
            tool_runtime=runtimes.get("summarize"),
        ),
    )

    graph.add_edge(START, "identify")
    graph.add_edge("identify", "screening")
    graph.add_edge("screening", "capabilities")
    graph.add_edge("capabilities", "executive")
    graph.add_edge("executive", "feasibility")
    graph.add_edge("feasibility", "validation_stage_plan")
    graph.add_edge("validation_stage_plan", "quantitative")
    graph.add_edge("validation_stage_plan", "qualitative")
    graph.add_edge(["quantitative", "qualitative"], "evidence_promotion")
    graph.add_edge("evidence_promotion", "stage2_validation")
    graph.add_edge("stage2_validation", "evidence_calibration")
    graph.add_edge("evidence_calibration", "summarize")
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
