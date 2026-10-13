"""Validated execution boundary for the Quantitative Agent."""

from pathlib import Path

from pydantic import ValidationError

from shortcutminer.agent_client import AgentClient
from shortcutminer.experiment.agent_workspace import load_agent_workspace_prompt
from shortcutminer.experiment.check_resolution import resolve_quantitative_checks
from shortcutminer.experiment.execution_contract import (
    ExperimentContractError,
    classify_experiment_availability,
)
from shortcutminer.experiment.quantitative_plan import (
    QuantitativePlanError,
    QuantitativePlanUnmaterializable,
    compile_quantitative_experiment_plan,
    plan_summary,
    primary_mandatory_steps,
)
from shortcutminer.protocol.quantitative import (
    MetricValue,
    QuantitativeAgentDraft,
    QuantitativeExecutionRequirement,
    QuantitativeMeasurement,
    QuantitativeRequest,
    QuantitativeResult,
    QuantitativeResultPayload,
)
from shortcutminer.protocol.quantitative_plan import QuantitativeExperimentPlan
from shortcutminer.storage.events import append_event
from shortcutminer.storage.stage_artifacts import save_stage_json
from shortcutminer.tooling.agent_loop import (
    ProviderContractValidationError,
    RequiredExperimentNotAttempted,
    finalize_tool_call_draft,
    invoke_agent_with_tools,
)
from shortcutminer.tooling.builtins import ExperimentToolOutput
from shortcutminer.tooling.protocol import ToolResult
from shortcutminer.tooling.runtime import ToolRuntime


class QuantitativeExecutionError(RuntimeError):
    """Raised when Quantitative cannot produce a valid result."""


class QuantitativeDraftInvariantError(QuantitativeExecutionError):
    """Raised when a schema-valid draft cites unusable quantitative evidence."""


def build_quantitative_agent_payload(
    request: QuantitativeRequest,
    execution_requirement: QuantitativeExecutionRequirement,
    experiment_plan=None,
) -> dict:
    resources = request.execution_resources
    return {
        "request": request.model_dump(
            mode="json", exclude={"execution_resources"}
        ),
        "execution_requirement": execution_requirement.model_dump(mode="json"),
        "execution_resources_summary": (
            {
                "dataset_name": resources.dataset_name,
                "metadata_file": resources.metadata_file,
                "label_column": resources.label_column,
                "allowed_scientific_metadata_fields": (
                    resources.allowed_scientific_metadata_fields
                ),
                "selected_split": resources.selected_split,
                "available_splits": resources.available_splits,
            }
            if resources is not None else None
        ),
        **(
            {"experiment_plan_summary": plan_summary(experiment_plan)}
            if experiment_plan is not None else {}
        ),
    }


def determine_quantitative_execution_requirement(
    request: QuantitativeRequest,
    tool_runtime: ToolRuntime | None,
    *,
    feasibility_confirmed: bool,
) -> QuantitativeExecutionRequirement:
    """Turn validated feasibility and runtime availability into a safe obligation."""

    validate_quantitative_request(request)
    if not feasibility_confirmed:
        return QuantitativeExecutionRequirement(
            must_attempt_run_experiment=False,
            reason="feasibility_not_confirmed",
            resolved_experiment_types=[],
        )
    resolved_types = resolve_quantitative_checks(request.task.requested_checks)
    if (
        tool_runtime is None
        or not tool_runtime.can_attempt_run_experiment(
            requested_checks=request.task.requested_checks,
            required_model_capabilities=(
                request.task.inputs.required_model_capabilities
            ),
            requires_sample_access=request.task.inputs.requires_sample_access,
            requires_external_services=(
                request.task.inputs.requires_external_services
            ),
            runtime_constraints=request.task.inputs.runtime_constraints,
        )
    ):
        return QuantitativeExecutionRequirement(
            must_attempt_run_experiment=False,
            reason="run_experiment_unavailable",
            resolved_experiment_types=list(resolved_types),
            available_now_experiment_types=[],
            pending_dependency_experiment_types=[],
            unsupported_experiment_types=list(resolved_types),
        )
    available, pending, unsupported = classify_experiment_availability(
        list(resolved_types),
        history=tool_runtime.history,
        explicit_artifacts=set(tool_runtime.explicit_task_artifacts),
        run_id=request.run_id,
        candidate_id=request.candidate.candidate_id,
        task_id=request.task.task_id,
    )
    if not available:
        return QuantitativeExecutionRequirement(
            must_attempt_run_experiment=False,
            reason="run_experiment_unavailable",
            resolved_experiment_types=list(resolved_types),
            available_now_experiment_types=[],
            pending_dependency_experiment_types=pending,
            unsupported_experiment_types=unsupported,
        )
    return QuantitativeExecutionRequirement(
        must_attempt_run_experiment=True,
        reason="task_is_executable",
        resolved_experiment_types=list(resolved_types),
        available_now_experiment_types=available,
        pending_dependency_experiment_types=pending,
        unsupported_experiment_types=unsupported,
    )


def validate_quantitative_request(request: QuantitativeRequest) -> None:
    if request.in_reply_to != request.task.message_id:
        raise QuantitativeExecutionError(
            "QuantitativeRequest in_reply_to does not match AnalysisTask."
        )
    if request.task.target != "quantitative":
        raise QuantitativeExecutionError(
            "QuantitativeRequest contains a non-quantitative task."
        )
    if request.task.run_id != request.run_id:
        raise QuantitativeExecutionError(
            "AnalysisTask run_id does not match QuantitativeRequest."
        )
    if request.task.candidate_id != request.candidate.candidate_id:
        raise QuantitativeExecutionError(
            "AnalysisTask candidate_id does not match Candidate."
        )
    if request.task.hypothesis != request.candidate.hypothesis:
        raise QuantitativeExecutionError(
            "AnalysisTask changed the Candidate hypothesis."
        )


def validate_quantitative_result(
    result: QuantitativeResult,
    request: QuantitativeRequest,
    *,
    tool_results: tuple[ToolResult, ...] = (),
) -> None:
    validate_quantitative_request(request)
    payload = result.payload
    if result.protocol_version != request.protocol_version:
        raise QuantitativeExecutionError("Quantitative protocol_version changed.")
    if result.run_id != request.run_id:
        raise QuantitativeExecutionError("Quantitative result run_id changed.")
    if result.in_reply_to != request.message_id:
        raise QuantitativeExecutionError(
            "QuantitativeResult in_reply_to does not match request."
        )
    if result.message_id in {request.message_id, request.task.message_id}:
        raise QuantitativeExecutionError(
            "QuantitativeResult message_id must identify a new message."
        )
    if payload.task_id != request.task.task_id:
        raise QuantitativeExecutionError("Quantitative result changed task_id.")
    if payload.candidate_id != request.candidate.candidate_id:
        raise QuantitativeExecutionError("Quantitative result changed candidate_id.")
    if payload.hypothesis != request.candidate.hypothesis:
        raise QuantitativeExecutionError("Quantitative result changed hypothesis.")
    allowed_artifacts = set(request.task.inputs.artifact_references)
    for tool_result in tool_results:
        if tool_result.status == "success":
            allowed_artifacts.update(tool_result.artifacts)
    if not set(payload.artifact_references).issubset(allowed_artifacts):
        raise QuantitativeExecutionError(
            "Quantitative result contains an unauthorized artifact reference."
        )
    missing = [
        limitation
        for limitation in request.limitations
        if limitation not in payload.limitations
    ]
    if missing:
        raise QuantitativeExecutionError(
            "Quantitative result omitted upstream limitations: "
            + "; ".join(missing)
        )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


_EXPERIMENT_MATURITY = {
    "statistical_check": ("Q2", "observational"),
    "subgroup_evaluation": ("Q2", "observational"),
    "model_evaluation": ("Q3", "predictive"),
    "intervention": ("Q4", "interventional"),
    "ablation": ("Q4", "interventional"),
    "metric_computation": ("Q1", "descriptive"),
    "task_model_training": ("Q1", "descriptive"),
    "task_model_evaluation": ("Q3", "predictive"),
    "ood_evaluation": ("Q3", "predictive"),
    "counterfactual_intervention": ("Q4", "interventional"),
}


def _selected_evidence(
    draft: QuantitativeAgentDraft,
    request: QuantitativeRequest,
    tool_results: tuple[ToolResult, ...],
    experiment_plan: QuantitativeExperimentPlan | None = None,
) -> list[ToolResult]:
    selected: list[ToolResult] = []
    seen_steps: set[str] = set()
    seen_calls: set[str] = set()
    for result in tool_results:
        if result.tool_name != "run_experiment":
            continue
        tool_call_id = result.tool_call_id
        if result.agent_name != "quantitative":
            raise QuantitativeDraftInvariantError(
                f"evidence ToolResult {tool_call_id} belongs to another agent"
            )
        if result.run_id != request.run_id:
            raise QuantitativeDraftInvariantError(
                f"evidence ToolResult {tool_call_id} belongs to another run"
            )
        if result.candidate_id != request.candidate.candidate_id:
            raise QuantitativeDraftInvariantError(
                f"evidence ToolResult {tool_call_id} belongs to another candidate"
            )
        if result.task_id != request.task.task_id:
            raise QuantitativeDraftInvariantError(
                f"evidence ToolResult {tool_call_id} belongs to another task"
            )
        if result.status != "success":
            continue
        if not result.plan_step_id or not result.planning_check or not result.experiment_type:
            raise QuantitativeDraftInvariantError("successful experiment lacks canonical plan provenance")
        if experiment_plan is not None:
            step = next((item for item in experiment_plan.steps if item.step_id == result.plan_step_id), None)
            if step is None or step.planning_check != result.planning_check or step.experiment_type != result.experiment_type:
                raise QuantitativeDraftInvariantError("ToolResult plan provenance differs from canonical plan")
        if result.plan_step_id in seen_steps or tool_call_id in seen_calls:
            raise QuantitativeDraftInvariantError("duplicate quantitative measurement")
        seen_steps.add(result.plan_step_id)
        seen_calls.add(tool_call_id)
        try:
            output = ExperimentToolOutput.model_validate(result.output)
        except ValidationError as exc:
            raise QuantitativeDraftInvariantError(
                f"evidence ToolResult {tool_call_id} has an invalid typed output"
            ) from exc
        if output.metadata.get("experiment_type") not in {None, result.experiment_type}:
            raise QuantitativeDraftInvariantError("backend experiment type differs from plan")
        selected.append(result)
    if draft.status == "completed" and not selected:
        raise QuantitativeDraftInvariantError(
            "completed quantitative draft requires successful planned evidence"
        )
    if draft.status == "completed" and draft.evidence_direction == "supports" and all(
        ExperimentToolOutput.model_validate(result.output).outcome == "inconclusive"
        or result.experiment_type == "task_model_training"
        for result in selected
    ):
        raise QuantitativeDraftInvariantError("inconclusive backend cannot support a claim")
    return selected


def _measurement(result: ToolResult) -> QuantitativeMeasurement:
    output = ExperimentToolOutput.model_validate(result.output)
    metadata = output.metadata
    details = metadata.get("details")
    lineage = {
        key: value for key, value in (details.items() if isinstance(details, dict) else [])
        if key in {"model_id", "model_hash", "intervention_id", "ood_id", "weights_reference", "model_reference"}
    }
    uncertainty = metadata.get("uncertainty")
    level, strength = _EXPERIMENT_MATURITY[result.experiment_type]
    return QuantitativeMeasurement(
        measurement_id=f"M-{result.tool_call_id}",
        run_id=result.run_id,
        candidate_id=result.candidate_id,
        task_id=result.task_id,
        plan_step_id=result.plan_step_id,
        planning_check=result.planning_check,
        experiment_type=result.experiment_type,
        tool_call_id=result.tool_call_id,
        backend_identity=str(metadata.get("backend") or "unknown_backend"),
        execution_outcome=output.outcome,
        metrics=output.metrics,
        artifact_references=list(dict.fromkeys([*output.artifact_references, *result.artifacts])),
        model_lineage=lineage,
        design_strength=strength,
        uncertainty_metadata=(uncertainty if isinstance(uncertainty, list) else []),
    )


def extract_quantitative_metrics(
    selected_results: list[ToolResult],
) -> dict[str, MetricValue]:
    """Extract typed scalar metrics without interpreting their scientific meaning."""

    outputs = [
        (result.tool_call_id, ExperimentToolOutput.model_validate(result.output))
        for result in selected_results
    ]
    counts: dict[str, int] = {}
    for _, output in outputs:
        for name in output.metrics:
            counts[name] = counts.get(name, 0) + 1
    metrics: dict[str, MetricValue] = {}
    for tool_call_id, output in outputs:
        for name, value in output.metrics.items():
            key = f"{tool_call_id}.{name}" if counts[name] > 1 else name
            metrics[key] = value
    return metrics


def validate_quantitative_execution_obligation(
    draft: QuantitativeAgentDraft,
    request: QuantitativeRequest,
    requirement: QuantitativeExecutionRequirement,
    tool_results: tuple[ToolResult, ...],
) -> None:
    """Enforce status semantics from actual current-task experiment attempts."""

    if not requirement.must_attempt_run_experiment:
        return
    attempts = [
        result for result in tool_results
        if result.tool_name == "run_experiment"
        and result.agent_name == "quantitative"
        and result.run_id == request.run_id
        and result.candidate_id == request.candidate.candidate_id
        and result.task_id == request.task.task_id
    ]
    if not attempts:
        raise QuantitativeDraftInvariantError(
            "required_experiment_not_attempted"
        )
    if any(result.status == "success" for result in attempts) and draft.status != "completed":
        raise QuantitativeDraftInvariantError(
            "A successful required experiment cannot finalize as blocked or failed."
        )


def finalize_quantitative_result(
    draft: QuantitativeAgentDraft,
    request: QuantitativeRequest,
    *,
    tool_results: tuple[ToolResult, ...] = (),
    experiment_plan: QuantitativeExperimentPlan | None = None,
) -> QuantitativeResult:
    """Construct the canonical result from provider reasoning and authoritative tools."""

    selected = _selected_evidence(draft, request, tool_results, experiment_plan)
    if (
        draft.status == "completed"
        and experiment_plan is not None
        and any(
            step.required and step.execution_state in {"ready", "pending_dependency"}
            for step in experiment_plan.steps
        )
    ):
        raise QuantitativeDraftInvariantError("required plan step is unhandled")
    measurements = [_measurement(item) for item in selected]
    execution_limitations: list[str] = []
    for result in selected:
        output = ExperimentToolOutput.model_validate(result.output)
        raw_limitations = output.metadata.get("limitations")
        if isinstance(raw_limitations, list) and all(
            isinstance(item, str) and item.strip() for item in raw_limitations
        ):
            execution_limitations.extend(raw_limitations)
    metrics = (
        extract_quantitative_metrics(selected)
        if draft.status == "completed"
        else {}
    )
    artifacts = _unique(
        [
            *request.task.inputs.artifact_references,
            *(artifact for result in selected for artifact in result.artifacts),
        ]
    )
    limitations = _unique(
        [
            *request.limitations,
            *draft.additional_limitations,
            *execution_limitations,
        ]
    )
    try:
        result = QuantitativeResult(
            protocol_version=request.protocol_version,
            run_id=request.run_id,
            message_id=(
                f"{request.run_id}:quantitative:{request.task.task_id}:result"
            ),
            source="quantitative",
            target="summarize",
            message_type="quantitative_analysis_result",
            in_reply_to=request.message_id,
            payload=QuantitativeResultPayload(
                task_id=request.task.task_id,
                candidate_id=request.candidate.candidate_id,
                hypothesis=request.candidate.hypothesis,
                status=draft.status,
                analysis_type=draft.analysis_type,
                metrics=metrics,
                measurements=measurements,
                evidence_level=(
                    max((_EXPERIMENT_MATURITY[item.experiment_type][0] for item in selected))
                    if draft.status == "completed" else "Q0"
                ),
                evidence_direction=draft.evidence_direction,
                evidence_summary=draft.evidence_summary,
                limitations=limitations,
                artifact_references=artifacts,
                recommended_followup=draft.recommended_followup,
            ),
        )
    except ValidationError as exc:
        raise QuantitativeDraftInvariantError(
            "Quantitative draft cannot be finalized as a canonical result."
        ) from exc
    validate_quantitative_result(result, request, tool_results=tool_results)
    return result


def execute_quantitative(
    request: QuantitativeRequest,
    client: AgentClient,
    runs_root: str | Path | None = None,
    workspace_dir: str | Path | None = None,
    tool_runtime: ToolRuntime | None = None,
    execution_requirement: QuantitativeExecutionRequirement | None = None,
    plan_step_allowlist: frozenset[str] | None = None,
) -> QuantitativeResult:
    """Invoke, validate, and optionally persist one quantitative task."""
    try:
        if runs_root is not None:
            append_event(
                runs_root,
                request.run_id,
                "QUANTITATIVE_STARTED",
                stage="quantitative",
                details={"task_id": request.task.task_id},
            )
        requirement = execution_requirement or determine_quantitative_execution_requirement(
            request,
            tool_runtime,
            feasibility_confirmed=False,
        )
        prompt = load_agent_workspace_prompt("quantitative", workspace_dir)
        if tool_runtime is not None:
            from dataclasses import replace
            tool_runtime.context=replace(tool_runtime.context,construction_resources=(
                request.execution_resources.model_dump(mode='json') if request.execution_resources is not None else None))
            tool_runtime.bind_analysis_task(
                runtime_constraints=request.task.inputs.runtime_constraints,
                requested_checks=request.task.requested_checks,
            )
        if requirement.must_attempt_run_experiment and tool_runtime is not None and plan_step_allowlist is None:
            from shortcutminer.experiment.ood_recipe_selection import selection_pending, select_memory_recipe, EXPENSIVE
            if (request.execution_resources is not None and selection_pending(request.execution_resources)
                    and set(requirement.resolved_experiment_types) & EXPENSIVE):
                request = select_memory_recipe(request, client, tool_runtime, prompt=prompt, runs_root=runs_root)
        experiment_plan = None
        mandatory_step_ids: list[str] = []
        mandatory_results: list[ToolResult] = []
        if requirement.must_attempt_run_experiment:
            if tool_runtime is None:
                raise QuantitativePlanUnmaterializable(
                    "mandatory_quantitative_execution_requires_tool_runtime"
                )
            def compile_current_plan():
                compiled = compile_quantitative_experiment_plan(request, requirement, tool_runtime)
                if plan_step_allowlist is not None:
                    compiled = compiled.model_copy(update={"steps": [step for step in compiled.steps if step.step_id in plan_step_allowlist]})
                return compiled
            experiment_plan = compile_current_plan()
            known_steps = {step.step_id: step for step in experiment_plan.steps}
            for previous in tool_runtime.history:
                if previous.tool_name != "run_experiment" or previous.status != "success":
                    continue
                step = known_steps.get(previous.plan_step_id)
                if (
                    step is None
                    or previous.run_id != request.run_id
                    or previous.candidate_id != request.candidate.candidate_id
                    or previous.task_id != request.task.task_id
                    or previous.planning_check != step.planning_check
                    or previous.experiment_type != step.experiment_type
                    or previous.plan_step_id in mandatory_step_ids
                ):
                    raise QuantitativeExecutionError("restored_quantitative_step_provenance_invalid")
                mandatory_step_ids.append(previous.plan_step_id)
                mandatory_results.append(previous)
            if not mandatory_step_ids and not any(
                step.required and step.execution_state == "ready"
                for step in experiment_plan.steps
            ):
                raise QuantitativePlanUnmaterializable(
                    "no_required_ready_quantitative_plan_step"
                )
            if mandatory_step_ids:
                done = set(mandatory_step_ids)
                experiment_plan = experiment_plan.model_copy(update={
                    "steps": [
                        step.model_copy(update={"execution_state": "completed"})
                        if step.step_id in done else step
                        for step in experiment_plan.steps
                    ]
                })
            base_payload = {
                "request": request.model_dump(
                    mode="json", exclude={"execution_resources"}
                ),
                "execution_requirement": requirement.model_dump(mode="json"),
            }
            while True:
                ready = [
                    step for step in experiment_plan.steps
                    if step.required and step.execution_state == "ready"
                    and step.step_id not in mandatory_step_ids
                ]
                if not ready:
                    break
                for step in ready:
                    call = finalize_tool_call_draft(
                        {"name": "run_experiment", "arguments": {"plan_step_id": step.step_id}},
                        agent_name="quantitative",
                        base_payload=base_payload,
                        tool_runtime=tool_runtime,
                        iteration=0,
                        call_index=len(tool_runtime.history),
                        experiment_plan=experiment_plan,
                    )
                    result = tool_runtime.execute(call)
                    mandatory_step_ids.append(step.step_id)
                    mandatory_results.append(result)
                    if result.status != "success":
                        raise QuantitativePlanUnmaterializable("mandatory_quantitative_backend_execution_failed")
                refreshed = compile_current_plan()
                done = set(mandatory_step_ids)
                experiment_plan = refreshed.model_copy(update={
                    "steps": [
                        step.model_copy(update={"execution_state": "completed"})
                        if step.step_id in done else step
                        for step in refreshed.steps
                    ]
                })
            if runs_root is not None:
                save_stage_json(
                    runs_root,
                    run_id=request.run_id,
                    stage="quantitative",
                    scope=request.task.task_id,
                    filename="experiment_plan.json",
                    payload=experiment_plan,
                )
                save_stage_json(
                    runs_root,
                    run_id=request.run_id,
                    stage="quantitative",
                    scope=request.task.task_id,
                    filename="plan_execution.json",
                    payload={
                        "mandatory_executed_step_ids": mandatory_step_ids,
                        "backend_attempt_count": sum(
                            result.tool_name == "run_experiment"
                            for result in tool_runtime.history
                        ),
                        "successful_tool_result_count": sum(
                            result.tool_name == "run_experiment"
                            and result.status == "success"
                            for result in tool_runtime.history
                        ),
                        "quantitative_agent_call_count": 0,
                    },
                )
            if not mandatory_results or not all(
                result.status == "success" for result in mandatory_results
            ):
                raise QuantitativePlanUnmaterializable(
                    "mandatory_quantitative_backend_execution_failed"
                )
            if any(step.required and step.execution_state == "ready" for step in experiment_plan.steps):
                raise QuantitativePlanUnmaterializable("required_quantitative_step_remains_ready")
        payload = build_quantitative_agent_payload(
            request, requirement, experiment_plan
        )
        if plan_step_allowlist is not None:
            payload["validation_stage"] = {
                "phase": "stage1_cheap_validation",
                "current_requested_checks": list(dict.fromkeys(step.planning_check for step in experiment_plan.steps)) if experiment_plan else [],
                "remaining_checks": "Deferred to the deterministic evidence promotion controller; only current plan steps are executable.",
            }
        if runs_root is not None:
            save_stage_json(
                runs_root,
                run_id=request.run_id,
                stage="quantitative",
                scope=request.task.task_id,
                filename="input.json",
                payload=payload,
            )
            if experiment_plan is not None:
                save_stage_json(
                    runs_root,
                    run_id=request.run_id,
                    stage="quantitative",
                    scope=request.task.task_id,
                    filename="experiment_plan.json",
                    payload=experiment_plan,
                )
                # The initial plan-execution audit was already persisted before
                # the provider boundary; keep the refreshed completed-state plan.
        if tool_runtime is not None:
            tool_runtime.assert_task_scope(
                candidate_id=request.candidate.candidate_id,
                task_id=request.task.task_id,
            )
            if any(
                result.run_id != request.run_id
                or result.agent_name != "quantitative"
                or result.candidate_id != request.candidate.candidate_id
                or result.task_id != request.task.task_id
                for result in tool_runtime.history
            ):
                raise QuantitativeExecutionError(
                    "provider_tool_results_not_task_local"
                )
        if requirement.must_attempt_run_experiment and not any(
            result.status == "success"
            and result.tool_name == "run_experiment"
            and result.run_id == request.run_id
            and result.agent_name == "quantitative"
            and result.candidate_id == request.candidate.candidate_id
            and result.task_id == request.task.task_id
            for result in (tool_runtime.history if tool_runtime is not None else [])
        ):
            raise QuantitativeExecutionError(
                "successful_current_task_run_experiment_required_before_provider"
            )
        invocation = None
        def validate_provider_draft(raw: dict) -> None:
            draft = QuantitativeAgentDraft.model_validate(raw)
            history = tuple(tool_runtime.history) if tool_runtime is not None else ()
            try:
                validate_quantitative_execution_obligation(draft, request, requirement, history)
                finalize_quantitative_result(
                    draft, request, tool_results=history,
                    experiment_plan=experiment_plan,
                )
                if draft.status == "completed" and experiment_plan is not None and any(
                    step.required and step.execution_state in {"ready", "pending_dependency"}
                    for step in experiment_plan.steps
                ):
                    raise QuantitativeDraftInvariantError("required plan step is unhandled")
            except QuantitativeExecutionError as exc:
                raise ProviderContractValidationError("quantitative_context_invalid") from exc
        try:
            invocation = invoke_agent_with_tools(
                client,
                "quantitative",
                payload,
                system_prompt=prompt,
                tool_runtime=tool_runtime,
                experiment_plan=experiment_plan,
                quantitative_request=request,
                provider_contract_validator=validate_provider_draft,
            )
            raw_output = invocation.output
        except RequiredExperimentNotAttempted as exc:
            if runs_root is not None:
                save_stage_json(
                    runs_root,
                    run_id=request.run_id,
                    stage="quantitative",
                    scope=request.task.task_id,
                    filename="execution_obligation.json",
                    payload={
                        "category": exc.category,
                        "execution_requirement": requirement.model_dump(mode="json"),
                        "execution_obligation_retry_count": exc.retry_count,
                    },
                )
            raise QuantitativeExecutionError(exc.category) from exc
        except (ExperimentContractError, QuantitativePlanError) as exc:
            if runs_root is not None:
                save_stage_json(
                    runs_root,
                    run_id=request.run_id,
                    stage="quantitative",
                    scope=request.task.task_id,
                    filename="execution_contract_failure.json",
                    payload={"category": exc.category},
                )
            raise QuantitativeExecutionError(exc.category) from exc
        except Exception as exc:
            raise QuantitativeExecutionError(
                f"Quantitative agent invocation failed: {exc}"
            ) from exc
        if not isinstance(raw_output, dict):
            raise QuantitativeExecutionError(
                "Quantitative agent returned a non-object response."
            )
        try:
            draft = QuantitativeAgentDraft.model_validate(raw_output)
        except ValidationError as exc:
            raise QuantitativeExecutionError(
                "Quantitative agent returned an invalid QuantitativeAgentDraft."
            ) from exc
        validate_quantitative_execution_obligation(
            draft,
            request,
            requirement,
            invocation.tool_results,
        )
        if runs_root is not None:
            if experiment_plan is not None:
                save_stage_json(
                    runs_root,
                    run_id=request.run_id,
                    stage="quantitative",
                    scope=request.task.task_id,
                    filename="plan_execution.json",
                    payload={
                        "mandatory_executed_step_ids": mandatory_step_ids,
                        "backend_attempt_count": len(mandatory_results),
                        "successful_tool_result_count": sum(
                            result.status == "success"
                            for result in mandatory_results
                        ),
                        "quantitative_agent_call_count": (
                            invocation.provider_call_count
                        ),
                    },
                )
            save_stage_json(
                runs_root,
                run_id=request.run_id,
                stage="quantitative",
                scope=request.task.task_id,
                filename="execution_obligation.json",
                payload={
                    "category": None,
                    "execution_requirement": requirement.model_dump(mode="json"),
                    "execution_obligation_retry_count": (
                        invocation.execution_obligation_retry_count
                    ),
                    "provider_tool_draft_count": (
                        invocation.provider_tool_draft_count
                    ),
                    "canonical_tool_call_count": (
                        invocation.canonical_tool_call_count
                    ),
                },
            )
            save_stage_json(
                runs_root,
                run_id=request.run_id,
                stage="quantitative",
                scope=request.task.task_id,
                filename="draft.json",
                payload=draft,
            )
        result = finalize_quantitative_result(
            draft,
            request,
            tool_results=invocation.tool_results,
            experiment_plan=experiment_plan,
        )
        if runs_root is not None:
            save_stage_json(
                runs_root,
                run_id=result.run_id,
                stage="quantitative",
                scope=request.task.task_id,
                filename="result.json",
                payload=result,
            )
            append_event(
                runs_root,
                result.run_id,
                "QUANTITATIVE_COMPLETED",
                stage="quantitative",
                details={
                    "task_id": request.task.task_id,
                    "status": result.payload.status,
                    "evidence_level": result.payload.evidence_level,
                },
            )
        return result
    except Exception as exc:
        if runs_root is not None:
            try:
                if isinstance(exc, QuantitativePlanError):
                    save_stage_json(
                        runs_root,
                        run_id=request.run_id,
                        stage="quantitative",
                        scope=request.task.task_id,
                        filename="execution_contract_failure.json",
                        payload={"category": exc.category},
                    )
                append_event(
                    runs_root,
                    request.run_id,
                    "AGENT_FAILED",
                    stage="quantitative",
                    details={
                        "task_id": request.task.task_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            except Exception as event_exc:
                exc.add_note(f"Could not persist failure event: {event_exc}")
        if isinstance(exc, QuantitativeExecutionError):
            raise
        raise QuantitativeExecutionError(
            f"Quantitative execution failed: {exc}"
        ) from exc
