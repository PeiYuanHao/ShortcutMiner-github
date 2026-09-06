"""Formal public entry points for new and resumed ShortcutMiner runs."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from shortcutminer.agent_client import AgentClient
from shortcutminer.experiment.real_executor import RealExperimentExecutor
from shortcutminer.experiment.run_manager import generate_run_id
from shortcutminer.experiment.loader import load_run_request
from shortcutminer.orchestrator import (
    execute_run,
    execute_workflow_state,
    resume_workflow_run,
)
from shortcutminer.protocol.run import RunExecutionResult, RunRequest
from shortcutminer.tooling.protocol import AgentName
from shortcutminer.tooling.builtins import build_default_tool_registry
from shortcutminer.tooling.runtime import ToolRuntime
from shortcutminer.workflow.state import ShortcutMinerState


def _clients(
    *,
    agent_client: AgentClient | None,
    identify_client: AgentClient | None,
    executive_client: AgentClient | None,
    quantitative_client: AgentClient | None,
    qualitative_client: AgentClient | None,
    summarize_client: AgentClient | None,
    tool_runtimes: dict[AgentName, ToolRuntime] | None,
) -> dict[str, Any]:
    return {
        "agent_client": agent_client,
        "identify_client": identify_client,
        "executive_client": executive_client,
        "quantitative_client": quantitative_client,
        "qualitative_client": qualitative_client,
        "summarize_client": summarize_client,
        "tool_runtimes": tool_runtimes,
    }


def build_local_tool_runtimes(
    *,
    run_id: str,
    dataset_root: str | Path,
    runs_root: str | Path,
    approved_artifacts: set[str] | None = None,
) -> dict[AgentName, ToolRuntime]:
    """Create least-privilege local runtimes with the real CPU backend."""
    registry = build_default_tool_registry(
        experiment_executor=RealExperimentExecutor()
    )
    return {
        agent: ToolRuntime(
            registry,
            run_id=run_id,
            agent_name=agent,
            dataset_root=dataset_root,
            runs_root=runs_root,
            approved_artifacts=approved_artifacts or set(),
        )
        for agent in (
            "identify",
            "executive",
            "quantitative",
            "qualitative",
            "summarize",
        )
    }


def run(
    request: RunRequest | Mapping[str, Any] | str | Path,
    *,
    agent_client: AgentClient | None = None,
    identify_client: AgentClient | None = None,
    executive_client: AgentClient | None = None,
    quantitative_client: AgentClient | None = None,
    qualitative_client: AgentClient | None = None,
    summarize_client: AgentClient | None = None,
    tool_runtimes: dict[AgentName, ToolRuntime] | None = None,
) -> RunExecutionResult:
    """Validate a formal request, create its run_id, and call Orchestrator."""
    if isinstance(request, RunRequest):
        parsed = request
    elif isinstance(request, (str, Path)):
        parsed = load_run_request(request)
    else:
        parsed = RunRequest.model_validate(request)
    if parsed.run_id is None:
        parsed = parsed.model_copy(update={"run_id": generate_run_id()})
    return execute_run(
        parsed,
        **_clients(
            agent_client=agent_client,
            identify_client=identify_client,
            executive_client=executive_client,
            quantitative_client=quantitative_client,
            qualitative_client=qualitative_client,
            summarize_client=summarize_client,
            tool_runtimes=tool_runtimes,
        ),
    )


def resume_run(
    run_id: str,
    *,
    runs_root: str | Path,
    agent_client: AgentClient | None = None,
    identify_client: AgentClient | None = None,
    executive_client: AgentClient | None = None,
    quantitative_client: AgentClient | None = None,
    qualitative_client: AgentClient | None = None,
    summarize_client: AgentClient | None = None,
    tool_runtimes: dict[AgentName, ToolRuntime] | None = None,
) -> RunExecutionResult:
    """Resume the LangGraph thread associated with an existing run_id."""
    return resume_workflow_run(
        run_id,
        runs_root=runs_root,
        **_clients(
            agent_client=agent_client,
            identify_client=identify_client,
            executive_client=executive_client,
            quantitative_client=quantitative_client,
            qualitative_client=qualitative_client,
            summarize_client=summarize_client,
            tool_runtimes=tool_runtimes,
        ),
    )


def run_workflow(
    initial_state: ShortcutMinerState,
    identify_client: AgentClient | None = None,
    *,
    agent_client: AgentClient | None = None,
    executive_client: AgentClient | None = None,
    quantitative_client: AgentClient | None = None,
    qualitative_client: AgentClient | None = None,
    summarize_client: AgentClient | None = None,
    tool_runtimes: dict[AgentName, ToolRuntime] | None = None,
) -> ShortcutMinerState:
    """Backward-compatible state API routed through the formal Orchestrator."""
    identify_config = initial_state.get("identify_config", {})
    runs_root = (
        identify_config.get("runs_root", "runs")
        if isinstance(identify_config, dict)
        else "runs"
    )
    result = execute_workflow_state(
        initial_state,
        run_id=initial_state["run_id"],
        runs_root=runs_root,
        resume=False,
        **_clients(
            agent_client=agent_client,
            identify_client=identify_client,
            executive_client=executive_client,
            quantitative_client=quantitative_client,
            qualitative_client=qualitative_client,
            summarize_client=summarize_client,
            tool_runtimes=tool_runtimes,
        ),
    )
    return ShortcutMinerState(**result.workflow_state)


__all__ = [
    "build_local_tool_runtimes",
    "resume_run",
    "run",
    "run_workflow",
]
