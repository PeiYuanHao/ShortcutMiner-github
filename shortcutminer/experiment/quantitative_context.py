"""Build one Quantitative Agent request from validated Executive state."""

from shortcutminer.experiment.executive_executor import (
    ExecutiveExecutionError,
    validate_routing_plan,
)
from shortcutminer.protocol.executive import ExecutiveRequest, RoutingPlan
from shortcutminer.protocol.quantitative import QuantitativeRequest


class QuantitativeContextError(RuntimeError):
    """Raised when Executive state cannot form a quantitative request."""


def build_quantitative_request(
    *,
    executive_request: ExecutiveRequest,
    routing_plan: RoutingPlan,
    task_id: str,
    message_id: str,
    limitations: list[str] | None = None,
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
        limitations=combined_limitations,
    )
