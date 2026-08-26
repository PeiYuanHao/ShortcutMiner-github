from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from shortcutminer.protocol.executive import (
    AnalysisTask,
    ExecutiveRequest,
    RoutingPlan,
)
from shortcutminer.storage.events import append_event
from shortcutminer.storage.executive_artifacts import (
    save_executive_input,
    save_executive_raw_output,
    save_executive_routing,
)


class AgentClient(Protocol):
    def invoke_agent(self, agent_name: str, payload: dict) -> dict:
        ...


class ExecutiveExecutionError(RuntimeError):
    """Raised when Executive cannot produce a valid routing plan."""


def build_executive_agent_payload(request: ExecutiveRequest) -> dict:
    return {"request": request.model_dump(mode="json")}


def validate_executive_request(request: ExecutiveRequest) -> None:
    report = request.candidate_report
    context = request.routing_context

    if request.in_reply_to != report.message_id:
        raise ExecutiveExecutionError(
            "ExecutiveRequest in_reply_to does not match CandidateReport."
        )

    if request.message_id == report.message_id:
        raise ExecutiveExecutionError(
            "ExecutiveRequest message_id must identify a new message."
        )

    if report.protocol_version != request.protocol_version:
        raise ExecutiveExecutionError(
            "CandidateReport protocol_version does not match ExecutiveRequest."
        )

    if report.run_id != request.run_id:
        raise ExecutiveExecutionError(
            "CandidateReport run_id does not match ExecutiveRequest."
        )

    if context.protocol_version != request.protocol_version:
        raise ExecutiveExecutionError(
            "ExecutiveRoutingContext protocol_version does not match request."
        )

    if context.run_id != request.run_id:
        raise ExecutiveExecutionError(
            "ExecutiveRoutingContext run_id does not match request."
        )

    if report.dataset_summary.dataset != context.dataset.name:
        raise ExecutiveExecutionError(
            "ExecutiveRoutingContext dataset does not match CandidateReport."
        )

    expected_candidate_ids = [
        f"C{index:03d}" for index in range(1, len(report.candidates) + 1)
    ]
    candidate_ids = [candidate.candidate_id for candidate in report.candidates]
    if candidate_ids != expected_candidate_ids:
        raise ExecutiveExecutionError(
            "CandidateReport candidate IDs must be sequential from C001."
        )

    authoritative_artifacts = set(context.artifact_references)
    if not set(report.artifact_references).issubset(authoritative_artifacts):
        raise ExecutiveExecutionError(
            "CandidateReport artifact references are not authoritative in "
            "ExecutiveRoutingContext."
        )

    missing_limitations = [
        limitation
        for limitation in report.limitations
        if limitation not in context.limitations
    ]
    if missing_limitations:
        raise ExecutiveExecutionError(
            "ExecutiveRoutingContext omitted CandidateReport limitations: "
            + "; ".join(missing_limitations)
        )


def _expected_task_ids(candidate_id: str, route: str) -> list[str]:
    if route == "quantitative":
        return [f"TASK-{candidate_id}-QT"]
    if route == "qualitative":
        return [f"TASK-{candidate_id}-QL"]
    if route == "both":
        return [f"TASK-{candidate_id}-QT", f"TASK-{candidate_id}-QL"]
    return []


def _validate_dependencies(tasks: list[AnalysisTask]) -> None:
    task_ids = {task.task_id for task in tasks}
    graph = {task.task_id: task.dependencies for task in tasks}

    for task in tasks:
        if task.task_id in task.dependencies:
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} depends on itself."
            )
        unknown = [item for item in task.dependencies if item not in task_ids]
        if unknown:
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} has unknown dependencies."
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ExecutiveExecutionError(
                "RoutingPlan contains a cyclic task dependency."
            )
        if task_id in visited:
            return

        visiting.add(task_id)
        for dependency in graph[task_id]:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in graph:
        visit(task_id)


def validate_routing_plan(
    plan: RoutingPlan,
    request: ExecutiveRequest,
) -> None:
    validate_executive_request(request)
    report = request.candidate_report
    context = request.routing_context
    options = context.audit_options
    capabilities = context.capabilities

    if plan.protocol_version != request.protocol_version:
        raise ExecutiveExecutionError(
            "RoutingPlan protocol_version does not match ExecutiveRequest."
        )
    if plan.run_id != request.run_id:
        raise ExecutiveExecutionError(
            "RoutingPlan run_id does not match ExecutiveRequest."
        )
    if plan.in_reply_to != request.message_id:
        raise ExecutiveExecutionError(
            "RoutingPlan in_reply_to does not match ExecutiveRequest."
        )
    if plan.message_id in {request.message_id, report.message_id}:
        raise ExecutiveExecutionError(
            "RoutingPlan message_id must identify a new message."
        )

    authoritative_artifacts = set(context.artifact_references)
    if not set(plan.artifact_references).issubset(authoritative_artifacts):
        raise ExecutiveExecutionError(
            "RoutingPlan contains an unauthoritative artifact reference."
        )

    missing_limitations = [
        limitation
        for limitation in context.limitations
        if limitation not in plan.limitations
    ]
    if missing_limitations:
        raise ExecutiveExecutionError(
            "RoutingPlan omitted routing-context limitations: "
            + "; ".join(missing_limitations)
        )

    if len(plan.decisions) != len(report.candidates):
        raise ExecutiveExecutionError(
            "RoutingPlan must contain exactly one decision per candidate."
        )
    if len(plan.tasks) > options.max_tasks:
        raise ExecutiveExecutionError("RoutingPlan exceeds max_tasks budget.")

    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in report.candidates
    }
    expected_plan_task_ids: list[str] = []

    for candidate, decision in zip(report.candidates, plan.decisions, strict=True):
        if decision.candidate != candidate:
            raise ExecutiveExecutionError(
                "RoutingDecision did not preserve the upstream candidate."
            )
        if decision.priority != candidate.priority:
            raise ExecutiveExecutionError(
                "RoutingDecision priority does not preserve candidate priority."
            )

        expected_ids = _expected_task_ids(candidate.candidate_id, decision.route)
        if decision.task_ids != expected_ids:
            raise ExecutiveExecutionError(
                f"RoutingDecision task IDs do not match route for "
                f"{candidate.candidate_id}."
            )

        if decision.route == "unresolved":
            if decision.status != "unresolved":
                raise ExecutiveExecutionError(
                    "An unresolved route must have unresolved status."
                )
            if not decision.missing_requirements and not decision.limitations:
                raise ExecutiveExecutionError(
                    "An unresolved route must state a missing requirement or "
                    "limitation."
                )
        else:
            if decision.status != "planned":
                raise ExecutiveExecutionError(
                    "An executable route must have planned status."
                )
            if decision.missing_requirements:
                raise ExecutiveExecutionError(
                    "An executable route cannot retain missing requirements."
                )

            required_targets = {
                "quantitative" if task_id.endswith("-QT") else "qualitative"
                for task_id in expected_ids
            }
            if not required_targets.issubset(set(options.allowed_targets)):
                raise ExecutiveExecutionError(
                    "RoutingDecision uses a target prohibited by audit options."
                )
            if not required_targets.issubset(
                set(capabilities.available_targets)
            ):
                raise ExecutiveExecutionError(
                    "RoutingDecision uses an unavailable target agent."
                )

        expected_plan_task_ids.extend(expected_ids)

    actual_task_ids = [task.task_id for task in plan.tasks]
    if actual_task_ids != expected_plan_task_ids:
        raise ExecutiveExecutionError(
            "RoutingPlan tasks must exactly match decision task IDs in order."
        )
    if len(actual_task_ids) != len(set(actual_task_ids)):
        raise ExecutiveExecutionError("RoutingPlan contains duplicate task IDs.")

    message_ids = [plan.message_id]
    for task in plan.tasks:
        candidate = candidate_by_id.get(task.candidate_id)
        if candidate is None:
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} references an unknown candidate."
            )
        if task.in_reply_to != plan.message_id:
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} is not correlated to RoutingPlan."
            )
        if task.run_id != request.run_id:
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} has the wrong run_id."
            )
        if task.message_id in {
            request.message_id,
            report.message_id,
            *message_ids,
        }:
            raise ExecutiveExecutionError(
                "RoutingPlan contains duplicate or reused message IDs."
            )
        message_ids.append(task.message_id)

        expected_suffix = "-QT" if task.target == "quantitative" else "-QL"
        if not task.task_id.endswith(expected_suffix):
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} target does not match task ID."
            )
        if task.task_id != f"TASK-{task.candidate_id}{expected_suffix}":
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} does not match candidate_id."
            )
        if task.hypothesis != candidate.hypothesis:
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} changed candidate hypothesis."
            )
        if task.priority != candidate.priority:
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} changed candidate priority."
            )

        inputs = task.inputs
        if inputs.dataset != context.dataset.name:
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} has the wrong dataset."
            )
        if inputs.dataset_reference != context.dataset.root:
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} invented a dataset reference."
            )
        if inputs.label_column != context.task.label_column:
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} has the wrong label column."
            )
        if (
            inputs.split is not None
            and inputs.split not in capabilities.available_splits
        ):
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} uses an unavailable split."
            )
        if not set(inputs.metadata_fields).issubset(
            set(capabilities.available_metadata_fields)
        ):
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} uses unavailable metadata."
            )
        if not set(inputs.artifact_references).issubset(
            authoritative_artifacts
        ):
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} invented an artifact reference."
            )
        if (
            inputs.sample_limit is not None
            and inputs.sample_limit > options.max_samples_per_task
        ):
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} exceeds the sample budget."
            )
        if inputs.requires_sample_access and not capabilities.sample_access:
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} requires unavailable sample access."
            )
        if not set(inputs.required_model_capabilities).issubset(
            set(capabilities.model_capabilities)
        ):
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} requires unavailable model access."
            )

        available_checks = (
            capabilities.quantitative_checks
            if task.target == "quantitative"
            else capabilities.qualitative_checks
        )
        if not set(task.requested_checks).issubset(set(available_checks)):
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} requests an unavailable check."
            )
        if not set(task.expected_outputs).issubset(
            set(capabilities.available_output_types)
        ):
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} requests an unavailable output."
            )

    if plan.tasks:
        covered_outputs = {
            output
            for task in plan.tasks
            for output in task.expected_outputs
        }
        missing_outputs = [
            output
            for output in options.requested_outputs
            if output not in covered_outputs
        ]
        if missing_outputs:
            raise ExecutiveExecutionError(
                "RoutingPlan does not cover requested outputs: "
                + "; ".join(missing_outputs)
            )

    _validate_dependencies(plan.tasks)


def execute_executive(
    request: ExecutiveRequest,
    client: AgentClient,
    runs_root: str | Path | None = None,
) -> RoutingPlan:
    try:
        if runs_root is not None:
            append_event(
                runs_root=runs_root,
                run_id=request.run_id,
                event="EXECUTIVE_STARTED",
                stage="executive",
            )

        validate_executive_request(request)

        payload = build_executive_agent_payload(request)
        if runs_root is not None:
            save_executive_input(
                runs_root=runs_root,
                run_id=request.run_id,
                payload=payload,
            )

        try:
            raw_output = client.invoke_agent(
                agent_name="executive",
                payload=payload,
            )
        except Exception as exc:
            raise ExecutiveExecutionError(
                f"Executive agent invocation failed: {exc}"
            ) from exc

        if not isinstance(raw_output, dict):
            raise ExecutiveExecutionError(
                "Executive agent returned a non-object response."
            )

        if runs_root is not None:
            save_executive_raw_output(
                runs_root=runs_root,
                run_id=request.run_id,
                raw_output=raw_output,
            )

        try:
            plan = RoutingPlan.model_validate(raw_output)
        except ValidationError as exc:
            raise ExecutiveExecutionError(
                "Executive agent returned an invalid RoutingPlan."
            ) from exc

        validate_routing_plan(plan, request)

        if runs_root is not None:
            save_executive_routing(runs_root=runs_root, plan=plan)
            append_event(
                runs_root=runs_root,
                run_id=request.run_id,
                event="EXECUTIVE_COMPLETED",
                stage="executive",
                details={
                    "candidate_count": len(plan.decisions),
                    "task_count": len(plan.tasks),
                    "unresolved_count": sum(
                        decision.route == "unresolved"
                        for decision in plan.decisions
                    ),
                },
            )

        return plan

    except Exception as exc:
        if runs_root is not None:
            try:
                append_event(
                    runs_root=runs_root,
                    run_id=request.run_id,
                    event="AGENT_FAILED",
                    stage="executive",
                    details={
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            except Exception:
                pass

        if isinstance(exc, ExecutiveExecutionError):
            raise
        raise ExecutiveExecutionError(
            f"Executive execution failed: {exc}"
        ) from exc
