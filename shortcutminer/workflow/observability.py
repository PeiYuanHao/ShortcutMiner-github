"""Best-effort lifecycle facts, independent of graph decisions and Agent output."""

from contextlib import contextmanager
from functools import wraps

from shortcutminer.storage.events import append_event


def emit(runs_root, run_id, event, stage, **details):
    if runs_root is not None:
        try:
            append_event(runs_root, run_id, event, stage=stage, details=details)
        except Exception:
            # Observation must never change execution success or failure.
            pass


@contextmanager
def lifecycle(runs_root, run_id, stage, *, skipped=False, task=None):
    details = ({"task_id": task.task_id, "candidate_id": task.candidate_id,
                "requested_checks": list(task.requested_checks)} if task is not None else {})
    prefix = "TASK" if task is not None else "STAGE"
    emit(runs_root, run_id, f"{prefix}_STARTED", stage, **details)
    try:
        yield
    except BaseException:
        emit(runs_root, run_id, f"{prefix}_FAILED", stage,
             reason="task_execution_failed" if task is not None else "stage_execution_failed", **details)
        raise
    else:
        emit(runs_root, run_id, f"{prefix}_SKIPPED" if skipped else f"{prefix}_COMPLETED", stage, **details)


def observe_node(stage, function):
    @wraps(function)
    def invoke(state):
        if state.get("metadata", {}).get("entrypoint") != "audit":
            return function(state)
        runs_root = state.get("identify_config", {}).get("runs_root")
        skipped = False
        if stage in {"quantitative", "qualitative"}:
            skipped = not any(task["target"] == stage for task in state.get("executive_result", {}).get("tasks", []))
        if stage == "executive":
            skipped = state.get("screening_result", {}).get("promoted") == 0
        with lifecycle(runs_root, state["run_id"], stage, skipped=skipped):
            return function(state)
    return invoke


def observe_task(state, task, function, *args, **kwargs):
    root = kwargs.get("runs_root") if state.get("metadata", {}).get("entrypoint") == "audit" else None
    with lifecycle(root, state["run_id"], task.target, task=task):
        return function(*args, **kwargs)
