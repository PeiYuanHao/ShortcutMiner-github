from pathlib import Path

from pydantic import ValidationError

from shortcutminer.agent_client import AgentClient
from shortcutminer.experiment.agent_workspace import load_agent_workspace_prompt
from shortcutminer.protocol.executive import (
    AnalysisTask,
    AnalysisTaskInputs,
    ExecutiveAgentDraft,
    ExecutiveDecisionDraft,
    ExecutiveRequest,
    RoutingDecision,
    RoutingPlan,
)
from shortcutminer.experiment.check_requirements import (
    AnalysisTaskSemanticError,
    CHECK_REQUIREMENTS,
    ExecutableCandidateContext,
    build_candidate_contexts,
    checks_require_sample_access,
    expected_outputs_for_checks,
    required_model_capabilities,
    validate_analysis_task_semantics,
)
from shortcutminer.storage.events import append_event
from shortcutminer.storage.executive_artifacts import (
    save_executive_input,
    save_executive_raw_output,
    save_executive_routing,
)
from shortcutminer.tooling.agent_loop import invoke_agent_with_tools
from shortcutminer.tooling.runtime import ToolRuntime


class ExecutiveExecutionError(RuntimeError):
    """Raised when Executive cannot produce a valid routing plan."""


class ExecutiveDraftSemanticError(ExecutiveExecutionError):
    """Raised when a schema-valid Executive draft selects impossible work."""


def build_executive_agent_payload(
    request: ExecutiveRequest,
    candidate_contexts: list[ExecutableCandidateContext] | None = None,
) -> dict:
    contexts = candidate_contexts or build_candidate_contexts(request)
    return {
        "request": request.model_dump(mode="json"),
        "executable_candidate_contexts": [
            item.model_dump(mode="json") for item in contexts
        ],
    }


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

    candidate_ids = [candidate.candidate_id for candidate in report.candidates]
    # Identify still creates sequential IDs. The controller may supply an ordered
    # screening subset; those original IDs must never be renumbered.
    if candidate_ids != sorted(set(candidate_ids)):
        raise ExecutiveExecutionError(
            "Executive candidate IDs must be unique and in original order."
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


def validate_executive_draft(
    draft: ExecutiveAgentDraft,
    request: ExecutiveRequest,
    candidate_contexts: list[ExecutableCandidateContext],
) -> None:
    expected_ids = [
        candidate.candidate_id for candidate in request.candidate_report.candidates
    ]
    actual_ids = [decision.candidate_id for decision in draft.decisions]
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise ExecutiveDraftSemanticError(
            "Executive draft must cover every candidate exactly once in order."
        )
    contexts = {item.candidate_id: item for item in candidate_contexts}
    for decision in draft.decisions:
        quantitative_required = decision.route in {"quantitative", "both"}
        qualitative_required = decision.route in {"qualitative", "both"}
        if (decision.quantitative is not None) != quantitative_required:
            raise ExecutiveDraftSemanticError(
                "Executive draft quantitative branch does not match route."
            )
        if (decision.qualitative is not None) != qualitative_required:
            raise ExecutiveDraftSemanticError(
                "Executive draft qualitative branch does not match route."
            )
        if decision.route == "unresolved":
            if not decision.missing_requirements and not decision.limitations:
                raise ExecutiveDraftSemanticError(
                    "Unresolved Executive draft requires a limitation or missing requirement."
                )
            continue
        if decision.missing_requirements:
            raise ExecutiveDraftSemanticError(
                "Executable Executive draft cannot retain missing requirements."
            )
        context = contexts[decision.candidate_id]
        for target, branch, allowed in (
            (
                "quantitative",
                decision.quantitative,
                context.allowed_quantitative_checks,
            ),
            (
                "qualitative",
                decision.qualitative,
                context.allowed_qualitative_checks,
            ),
        ):
            if branch is None:
                continue
            if not set(branch.requested_checks).issubset(set(allowed)):
                raise ExecutiveDraftSemanticError(
                    f"Executive draft selected unavailable {target} check."
                )
            required = set(request.routing_context.audit_options.required_quantitative_checks)
            if target == "quantitative" and required.issubset(allowed) and not required.issubset(branch.requested_checks):
                raise ExecutiveDraftSemanticError("Executive draft omitted caller-required executable quantitative checks")


def _finalize_task(
    *,
    request: ExecutiveRequest,
    plan_message_id: str,
    candidate,
    candidate_context: ExecutableCandidateContext,
    target: str,
    task_draft,
) -> AnalysisTask:
    checks = task_draft.requested_checks
    needs_samples = checks_require_sample_access(checks)
    metadata_fields = (
        candidate_context.bound_metadata_fields
        if any(
            CHECK_REQUIREMENTS[check].minimum_metadata_fields
            for check in checks
        )
        else []
    )
    if target == "qualitative" and request.routing_context.audit_options.qualitative_metadata_policy == "candidate-bound-v2":
        metadata_fields = list(candidate_context.bound_metadata_fields)
    suffix = "QT" if target == "quantitative" else "QL"
    task_id = f"TASK-{candidate.candidate_id}-{suffix}"
    outputs = expected_outputs_for_checks(
        checks, request.routing_context.capabilities.available_output_types
    )
    if not outputs:
        raise ExecutiveDraftSemanticError(
            "Executive task has no available canonical output type."
        )
    return AnalysisTask(
        protocol_version=request.protocol_version,
        run_id=request.run_id,
        message_id=f"{request.run_id}:executive:{task_id}:task",
        source="executive",
        target=target,
        message_type="analysis_task",
        in_reply_to=plan_message_id,
        task_id=task_id,
        candidate_id=candidate.candidate_id,
        hypothesis=candidate.hypothesis,
        objective=task_draft.objective,
        inputs=AnalysisTaskInputs(
            dataset=request.routing_context.dataset.name,
            dataset_reference=request.routing_context.dataset.root,
            label_column=request.routing_context.task.label_column,
            split=None,
            metadata_fields=metadata_fields,
            artifact_references=[],
            sample_limit=(
                request.routing_context.audit_options.max_samples_per_task
                if needs_samples else None
            ),
            requires_sample_access=needs_samples,
            requires_external_services=False,
            runtime_constraints=(
                request.routing_context.audit_options.runtime_constraints
            ),
            required_model_capabilities=required_model_capabilities(checks),
        ),
        requested_checks=checks,
        expected_outputs=outputs,
        priority=candidate.priority,
        dependencies=[],
        status="planned",
    )


def finalize_executive_draft(
    request: ExecutiveRequest,
    candidate_contexts: list[ExecutableCandidateContext],
    draft: ExecutiveAgentDraft,
) -> RoutingPlan:
    validate_executive_draft(draft, request, candidate_contexts)
    plan_message_id = f"{request.run_id}:executive:routing-plan"
    candidates = {
        item.candidate_id: item for item in request.candidate_report.candidates
    }
    contexts = {item.candidate_id: item for item in candidate_contexts}
    decisions: list[RoutingDecision] = []
    tasks: list[AnalysisTask] = []
    for item in draft.decisions:
        candidate = candidates[item.candidate_id]
        task_ids = _expected_task_ids(item.candidate_id, item.route)
        decisions.append(RoutingDecision(
            candidate=candidate,
            route=item.route,
            routing_reason=item.routing_reason,
            task_ids=task_ids,
            priority=candidate.priority,
            status="unresolved" if item.route == "unresolved" else "planned",
            missing_requirements=item.missing_requirements,
            limitations=item.limitations,
        ))
        for target, task_draft in (
            ("quantitative", item.quantitative),
            ("qualitative", item.qualitative),
        ):
            if task_draft is not None:
                tasks.append(_finalize_task(
                    request=request,
                    plan_message_id=plan_message_id,
                    candidate=candidate,
                    candidate_context=contexts[item.candidate_id],
                    target=target,
                    task_draft=task_draft,
                ))
    plan = RoutingPlan(
        protocol_version=request.protocol_version,
        run_id=request.run_id,
        message_id=plan_message_id,
        source="executive",
        target="orchestrator",
        message_type="routing_plan",
        in_reply_to=request.message_id,
        artifact_references=request.routing_context.artifact_references,
        decisions=decisions,
        tasks=tasks,
        limitations=list(dict.fromkeys([
            *request.routing_context.limitations,
            *draft.limitations,
        ])),
    )
    validate_routing_plan(plan, request, candidate_contexts=candidate_contexts)
    return plan


def validate_routing_plan(
    plan: RoutingPlan,
    request: ExecutiveRequest,
    *,
    candidate_contexts: list[ExecutableCandidateContext] | None = None,
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
    semantic_contexts = {
        item.candidate_id: item
        for item in (candidate_contexts or build_candidate_contexts(request))
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
        if inputs.requires_sample_access and inputs.sample_limit is None:
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} requires sample access without "
                "a sample_limit."
            )
        if inputs.requires_sample_access and not capabilities.sample_access:
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} requires unavailable sample access."
            )
        if (
            inputs.requires_external_services
            and not capabilities.external_services
        ):
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} requires unavailable external "
                "services."
            )
        if inputs.runtime_constraints != options.runtime_constraints:
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} did not preserve runtime "
                "constraints."
            )
        if (
            "no_retraining" in options.runtime_constraints
            and "retraining" in inputs.required_model_capabilities
        ):
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} violates the no_retraining "
                "runtime constraint."
            )
        if (
            "no_external_services" in options.runtime_constraints
            and inputs.requires_external_services
        ):
            raise ExecutiveExecutionError(
                f"AnalysisTask {task.task_id} violates the "
                "no_external_services runtime constraint."
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
        try:
            validate_analysis_task_semantics(
                task,
                semantic_contexts[task.candidate_id],
                capabilities,
            )
        except AnalysisTaskSemanticError as exc:
            raise ExecutiveExecutionError(str(exc)) from exc

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
    workspace_dir: str | Path | None = None,
    tool_runtime: ToolRuntime | None = None,
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

        candidate_contexts = build_candidate_contexts(request)
        payload = build_executive_agent_payload(request, candidate_contexts)
        system_prompt = load_agent_workspace_prompt(
            "executive", workspace_dir
        )
        if runs_root is not None:
            save_executive_input(
                runs_root=runs_root,
                run_id=request.run_id,
                payload=payload,
            )

        try:
            def validate_provider_draft(raw: dict) -> None:
                draft = ExecutiveAgentDraft.model_validate(raw)
                try:
                    finalize_executive_draft(request, candidate_contexts, draft)
                except (ExecutiveDraftSemanticError, ExecutiveExecutionError, ValidationError) as exc:
                    from shortcutminer.tooling.agent_loop import ProviderContractValidationError
                    raise ProviderContractValidationError("executive_context_invalid") from exc

            invocation = invoke_agent_with_tools(
                client,
                "executive",
                payload,
                system_prompt=system_prompt,
                tool_runtime=tool_runtime,
                provider_contract_validator=validate_provider_draft,
            )
            raw_output = invocation.output
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
            draft = ExecutiveAgentDraft.model_validate(raw_output)
        except ValidationError as exc:
            raise ExecutiveExecutionError(
                "Executive agent returned an invalid ExecutiveAgentDraft."
            ) from exc
        plan = finalize_executive_draft(request, candidate_contexts, draft)

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
