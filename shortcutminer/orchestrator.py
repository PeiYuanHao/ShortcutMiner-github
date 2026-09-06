"""Bridge the formal run lifecycle to the compiled LangGraph workflow."""

from pathlib import Path
from typing import Any

from shortcutminer.agent_client import AgentClient
from shortcutminer.experiment.dataset_profiler import build_dataset_profile
from shortcutminer.experiment.run_manager import initialize_run, transition_run
from shortcutminer.protocol.dataset_profile import DatasetProfile
from shortcutminer.protocol.identify import IdentifyBudget, IdentifyRequest
from shortcutminer.protocol.run import (
    CheckpointReference,
    RunError,
    RunExecutionResult,
    RunRequest,
)
from shortcutminer.storage.events import append_event
from shortcutminer.storage.dataset_profiles import save_dataset_profile
from shortcutminer.storage.run_store import (
    load_final_result,
    load_run_manifest,
    load_run_record,
    load_workflow_state,
    save_final_result,
    save_workflow_state,
)
from shortcutminer.tooling.protocol import AgentName
from shortcutminer.tooling.runtime import ToolRuntime
from shortcutminer.workflow.checkpoint import (
    checkpoint_config,
    checkpoint_path,
    open_sqlite_checkpointer,
)
from shortcutminer.workflow.graph import build_workflow
from shortcutminer.workflow.state import ShortcutMinerState


class OrchestrationError(RuntimeError):
    """Raised when a formal workflow cannot produce a valid final state."""


def build_initial_workflow_state(
    request: RunRequest,
    run_id: str,
) -> ShortcutMinerState:
    """Translate a validated formal request into serializable graph state."""
    identify_config = dict(request.identify_config)
    identify_config.update(seed=request.seed, runs_root=request.runs_root)
    state = ShortcutMinerState(
        run_id=run_id,
        dataset_info=request.dataset_info.model_dump(mode="json"),
        task_input=request.task_input.model_dump(mode="json"),
        identify_config=identify_config,
        executive_config=dict(request.executive_config),
        quantitative_config=dict(request.quantitative_config),
        qualitative_config=dict(request.qualitative_config),
        summarize_config=dict(request.summarize_config),
        stage_status={},
        errors=[],
        metadata=dict(request.metadata),
    )
    if request.model_runtime is not None:
        state["model_runtime"] = request.model_runtime.model_dump(mode="json")
    return state


def build_dataset_profile_for_run(
    request: RunRequest,
    run_id: str,
) -> DatasetProfile:
    """Run generic intake before the formal LangGraph invocation."""
    identify_request = IdentifyRequest(
        protocol_version="1.0",
        run_id=run_id,
        message_id=request.identify_config.get(
            "message_id", f"{run_id}:identify:request"
        ),
        source="orchestrator",
        target="identify",
        message_type="identify_request",
        dataset=request.dataset_info,
        task=request.task_input,
        budget=IdentifyBudget.model_validate(
            request.identify_config.get("budget", {})
        ),
    )
    return build_dataset_profile(identify_request, seed=request.seed)


def _client_arguments(
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


def _completed_result(
    runs_root: str | Path,
    run_id: str,
) -> RunExecutionResult:
    """Return an already completed run without re-invoking any graph node."""
    return RunExecutionResult(
        run_id=run_id,
        status="COMPLETED",
        final_result=load_final_result(runs_root, run_id),
        workflow_state=load_workflow_state(runs_root, run_id),
        manifest=load_run_manifest(runs_root, run_id),
    )


def execute_workflow_state(
    initial_state: ShortcutMinerState | None,
    *,
    run_id: str,
    runs_root: str | Path,
    resume: bool,
    agent_client: AgentClient | None = None,
    identify_client: AgentClient | None = None,
    executive_client: AgentClient | None = None,
    quantitative_client: AgentClient | None = None,
    qualitative_client: AgentClient | None = None,
    summarize_client: AgentClient | None = None,
    tool_runtimes: dict[AgentName, ToolRuntime] | None = None,
) -> RunExecutionResult:
    """Create or resume a durable run using LangGraph checkpoint semantics."""
    database_path = checkpoint_path(runs_root, run_id)
    reference = CheckpointReference(
        backend="sqlite",
        path="checkpoints.sqlite",
        thread_id=run_id,
    )

    if resume:
        record = load_run_record(runs_root, run_id)
        if record.status == "COMPLETED":
            return _completed_result(runs_root, run_id)
        if not database_path.is_file():
            raise OrchestrationError(
                f"Checkpoint does not exist for resumable run: {run_id}"
            )
        starting_event = "RUN_RESUMED"
    else:
        if initial_state is None:
            raise OrchestrationError("A new run requires initial workflow state.")
        initialize_run(
            initial_state,
            runs_root=runs_root,
            checkpoint=reference,
        )
        serialized_profile = initial_state.get("dataset_profile")
        if serialized_profile is not None:
            profile = DatasetProfile.model_validate(serialized_profile)
            save_dataset_profile(runs_root, run_id, profile)
        starting_event = "RUN_STARTED"

    config = checkpoint_config(run_id)
    workflow = None
    client_arguments = _client_arguments(
        agent_client=agent_client,
        identify_client=identify_client,
        executive_client=executive_client,
        quantitative_client=quantitative_client,
        qualitative_client=qualitative_client,
        summarize_client=summarize_client,
        tool_runtimes=tool_runtimes,
    )
    try:
        transition_run(
            runs_root,
            run_id,
            status="RUNNING",
            resume=resume,
        )
        append_event(runs_root, run_id, starting_event, stage="run")
        with open_sqlite_checkpointer(database_path) as checkpointer:
            workflow = build_workflow(
                checkpointer=checkpointer,
                **client_arguments,
            )
            result_state = workflow.invoke(
                None if resume else initial_state,
                config=config,
            )

        if not isinstance(result_state, dict):
            raise OrchestrationError("LangGraph returned a non-object state.")
        final_result = result_state.get("final_result")
        if not isinstance(final_result, dict):
            raise OrchestrationError(
                "Completed workflow state does not contain final_result."
            )
        save_workflow_state(runs_root, run_id, result_state)
        save_final_result(runs_root, run_id, final_result)
        append_event(runs_root, run_id, "RUN_COMPLETED", stage="run")
        _, manifest = transition_run(
            runs_root,
            run_id,
            status="COMPLETED",
        )
        return RunExecutionResult(
            run_id=run_id,
            status="COMPLETED",
            final_result=final_result,
            workflow_state=result_state,
            manifest=manifest,
        )
    except Exception as exc:
        if workflow is not None and database_path.is_file():
            try:
                with open_sqlite_checkpointer(database_path) as checkpointer:
                    snapshot_workflow = build_workflow(
                        checkpointer=checkpointer,
                        **client_arguments,
                    )
                    snapshot = snapshot_workflow.get_state(config)
                    if isinstance(snapshot.values, dict) and snapshot.values:
                        save_workflow_state(
                            runs_root,
                            run_id,
                            dict(snapshot.values),
                        )
            except Exception as snapshot_exc:
                exc.add_note(
                    f"Could not persist the latest workflow snapshot: {snapshot_exc}"
                )
        error = RunError(
            error_type=type(exc).__name__,
            message=str(exc).strip() or type(exc).__name__,
        )
        try:
            transition_run(
                runs_root,
                run_id,
                status="FAILED",
                error=error,
            )
        except Exception as status_exc:
            exc.add_note(f"Could not mark run FAILED: {status_exc}")
        try:
            append_event(
                runs_root,
                run_id,
                "RUN_FAILED",
                stage="run",
                details=error.model_dump(mode="json"),
            )
        except Exception as event_exc:
            exc.add_note(f"Could not persist RUN_FAILED event: {event_exc}")
        raise


def execute_run(
    request: RunRequest,
    *,
    agent_client: AgentClient | None = None,
    identify_client: AgentClient | None = None,
    executive_client: AgentClient | None = None,
    quantitative_client: AgentClient | None = None,
    qualitative_client: AgentClient | None = None,
    summarize_client: AgentClient | None = None,
    tool_runtimes: dict[AgentName, ToolRuntime] | None = None,
) -> RunExecutionResult:
    """Start a validated formal request through the compiled workflow."""
    if request.run_id is None:
        raise OrchestrationError("Formal RunRequest requires a generated run_id.")
    state = build_initial_workflow_state(request, request.run_id)
    profile = build_dataset_profile_for_run(request, request.run_id)
    state["dataset_profile"] = profile.model_dump(mode="json")
    return execute_workflow_state(
        state,
        run_id=request.run_id,
        runs_root=request.runs_root,
        resume=False,
        **_client_arguments(
            agent_client=agent_client,
            identify_client=identify_client,
            executive_client=executive_client,
            quantitative_client=quantitative_client,
            qualitative_client=qualitative_client,
            summarize_client=summarize_client,
            tool_runtimes=tool_runtimes,
        ),
    )


def resume_workflow_run(
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
    """Resume a persisted graph thread identified by the existing run_id."""
    return execute_workflow_state(
        None,
        run_id=run_id,
        runs_root=runs_root,
        resume=True,
        **_client_arguments(
            agent_client=agent_client,
            identify_client=identify_client,
            executive_client=executive_client,
            quantitative_client=quantitative_client,
            qualitative_client=qualitative_client,
            summarize_client=summarize_client,
            tool_runtimes=tool_runtimes,
        ),
    )


__all__ = [
    "build_dataset_profile_for_run",
    "build_initial_workflow_state",
    "build_workflow",
    "execute_run",
    "execute_workflow_state",
    "resume_workflow_run",
]
