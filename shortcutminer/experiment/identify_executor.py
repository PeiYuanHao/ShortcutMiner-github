from typing import Protocol

from pydantic import ValidationError

from shortcutminer.experiment.identify_context import (
    IdentifyDiscoveryContext,
    build_identify_context,
)
from shortcutminer.protocol.identify import (
    CandidateReport,
    IdentifyRequest,
)


class AgentClient(Protocol):
    """
    Minimal client contract required by IdentifyExecutor.

    A concrete OpenClaw client can implement this interface later.
    """

    def invoke_agent(
        self,
        agent_name: str,
        payload: dict,
    ) -> dict:
        ...


class IdentifyExecutionError(RuntimeError):
    """Raised when Identify execution cannot produce a valid report."""


def build_identify_agent_payload(
    request: IdentifyRequest,
    context: IdentifyDiscoveryContext,
) -> dict:
    """
    Build the structured payload sent to the Identify Agent.
    """

    return {
        "request": request.model_dump(mode="json"),
        "discovery_context": context.model_dump(mode="json"),
    }


def validate_candidate_report(
    report: CandidateReport,
    request: IdentifyRequest,
) -> None:
    """
    Validate run-level invariants that are not enforced
    directly by the CandidateReport schema.
    """

    if report.run_id != request.run_id:
        raise IdentifyExecutionError(
            "CandidateReport run_id does not match IdentifyRequest."
        )

    if (
        report.dataset_summary.dataset
        != request.dataset.name
    ):
        raise IdentifyExecutionError(
            "CandidateReport dataset does not match IdentifyRequest."
        )

    if (
        report.dataset_summary.samples_inspected
        > request.budget.max_samples
    ):
        raise IdentifyExecutionError(
            "CandidateReport exceeds max_samples budget."
        )

    if (
        len(report.candidates)
        > request.budget.max_candidates
    ):
        raise IdentifyExecutionError(
            "CandidateReport exceeds max_candidates budget."
        )

    candidate_ids = [
        candidate.candidate_id
        for candidate in report.candidates
    ]

    if len(candidate_ids) != len(set(candidate_ids)):
        raise IdentifyExecutionError(
            "CandidateReport contains duplicate candidate IDs."
        )


def execute_identify(
    request: IdentifyRequest,
    seed: int,
    client: AgentClient,
) -> CandidateReport:
    """
    Execute one Identify stage.

    Steps:
    1. Build deterministic discovery context.
    2. Send structured evidence to the Identify Agent.
    3. Parse the returned output as CandidateReport.
    4. Validate run-level invariants.

    This function does not perform shortcut reasoning itself.
    """

    context = build_identify_context(
        request=request,
        seed=seed,
    )

    payload = build_identify_agent_payload(
        request=request,
        context=context,
    )

    try:
        raw_output = client.invoke_agent(
            agent_name="identify",
            payload=payload,
        )

    except Exception as exc:
        raise IdentifyExecutionError(
            f"Identify agent invocation failed: {exc}"
        ) from exc

    try:
        report = CandidateReport.model_validate(
            raw_output
        )

    except ValidationError as exc:
        raise IdentifyExecutionError(
            "Identify agent returned an invalid CandidateReport."
        ) from exc

    validate_candidate_report(
        report=report,
        request=request,
    )

    return report