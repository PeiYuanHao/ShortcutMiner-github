"""Build one Quantitative Agent request from validated Executive state."""

from shortcutminer.experiment.executive_executor import (
    ExecutiveExecutionError,
    validate_routing_plan,
)
from shortcutminer.protocol.executive import ExecutiveRequest, RoutingPlan
from shortcutminer.protocol.quantitative import (
    QuantitativeExecutionResources,
    QuantitativeRequest,
)


class QuantitativeContextError(RuntimeError):
    """Raised when Executive state cannot form a quantitative request."""


def build_quantitative_request(
    *,
    executive_request: ExecutiveRequest,
    routing_plan: RoutingPlan,
    task_id: str,
    message_id: str,
    limitations: list[str] | None = None,
    execution_resource_options: dict | None = None,
) -> QuantitativeRequest:
    """Select one planned quantitative task without redefining it."""
    try:
        validate_routing_plan(routing_plan, executive_request)
    except ExecutiveExecutionError as exc:
        raise QuantitativeContextError(
            f"Executive state is invalid: {exc}"
        ) from exc

    matches = [task for task in routing_plan.tasks if task.task_id == task_id]
    if len(matches) != 1:
        raise QuantitativeContextError(
            "Quantitative task_id must identify exactly one planned task."
        )
    task = matches[0]
    if task.target != "quantitative":
        raise QuantitativeContextError("Selected task is not quantitative.")

    candidates = [
        candidate
        for candidate in executive_request.candidate_report.candidates
        if candidate.candidate_id == task.candidate_id
    ]
    if len(candidates) != 1:
        raise QuantitativeContextError(
            "Quantitative task does not map to exactly one candidate."
        )

    decision_limitations = next(
        decision.limitations
        for decision in routing_plan.decisions
        if decision.candidate.candidate_id == task.candidate_id
    )
    combined_limitations = list(
        dict.fromkeys(
            [
                *executive_request.routing_context.limitations,
                *routing_plan.limitations,
                *decision_limitations,
                *(limitations or []),
            ]
        )
    )
    options = dict(execution_resource_options or {})
    available_fields = list(
        executive_request.routing_context.capabilities.available_metadata_fields
    )
    sample_id_column = options.pop("sample_id_column", None)
    sample_path_column = options.pop("sample_path_column", None)
    split_column = options.pop("split_column", None)
    controlled_fields = {
        task.inputs.label_column,
        sample_id_column,
        sample_path_column,
        split_column,
    } - {None}
    explicitly_scoped = (
        set(task.inputs.metadata_fields)
        if task.inputs.metadata_fields
        else set(available_fields)
    )
    scientific_fields = sorted(explicitly_scoped - controlled_fields)
    configured_max_samples = options.pop("max_samples", 10_000)
    max_samples = task.inputs.sample_limit or configured_max_samples
    execution_resources = QuantitativeExecutionResources(
        dataset_name=executive_request.routing_context.dataset.name,
        metadata_file=executive_request.routing_context.dataset.metadata_file,
        label_column=task.inputs.label_column,
        available_metadata_fields=available_fields,
        allowed_scientific_metadata_fields=scientific_fields,
        sample_id_column=sample_id_column,
        sample_path_column=sample_path_column,
        split_column=split_column,
        selected_split=task.inputs.split,
        available_splits=(
            executive_request.routing_context.capabilities.available_splits
        ),
        authorized_artifact_references=list(dict.fromkeys([
            *executive_request.routing_context.artifact_references,
            *task.inputs.artifact_references,
        ])),
        max_samples=max_samples,
        **options,
    )
    return QuantitativeRequest(
        protocol_version=executive_request.protocol_version,
        run_id=executive_request.run_id,
        message_id=message_id,
        source="orchestrator",
        target="quantitative",
        message_type="quantitative_analysis_request",
        in_reply_to=task.message_id,
        candidate=candidates[0],
        task=task,
        execution_resources=execution_resources,
        limitations=combined_limitations,
    )
