"""Authoritative candidate-specific prerequisites for Executive checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from shortcutminer.experiment.candidate_binding import CandidateBinding
from shortcutminer.protocol.executive import (
    AnalysisTask,
    ExecutiveCapabilities,
    ExecutiveRequest,
    ModelCapability,
    ValidationTarget,
)
from shortcutminer.protocol.identify import CandidateId, NonEmptyText, StrictProtocolModel


@dataclass(frozen=True)
class CheckRequirement:
    target: ValidationTarget
    minimum_metadata_fields: int = 0
    required_model_capabilities: tuple[ModelCapability, ...] = ()
    requires_sample_access: bool = False
    requires_structured_spec: bool = False
    expected_output: Literal["metrics", "observations"] = "metrics"


CHECK_REQUIREMENTS: dict[str, CheckRequirement] = {
    "association_test": CheckRequirement("quantitative", minimum_metadata_fields=1),
    "subgroup_evaluation": CheckRequirement("quantitative", minimum_metadata_fields=1),
    "model_evaluation": CheckRequirement("quantitative", minimum_metadata_fields=1),
    "intervention": CheckRequirement("quantitative", minimum_metadata_fields=1),
    "ablation": CheckRequirement("quantitative", minimum_metadata_fields=2),
    "metric_computation": CheckRequirement(
        "quantitative", requires_structured_spec=True
    ),
    "task_model_training": CheckRequirement(
        "quantitative",
        required_model_capabilities=("retraining",),
        requires_sample_access=True,
    ),
    "task_model_evaluation": CheckRequirement(
        "quantitative",
        required_model_capabilities=("checkpoint",),
        requires_sample_access=True,
    ),
    "ood_evaluation": CheckRequirement(
        "quantitative",
        required_model_capabilities=("checkpoint", "intervention_inference"),
        requires_sample_access=True,
        requires_structured_spec=True,
    ),
    "counterfactual_intervention": CheckRequirement(
        "quantitative",
        required_model_capabilities=("checkpoint", "intervention_inference"),
        requires_sample_access=True,
        requires_structured_spec=True,
    ),
    "metadata_inspection": CheckRequirement(
        "qualitative", minimum_metadata_fields=1, expected_output="observations"
    ),
    "sample_inspection": CheckRequirement(
        "qualitative", requires_sample_access=True, expected_output="observations"
    ),
    "counterexample_inspection": CheckRequirement(
        "qualitative", requires_sample_access=True, expected_output="observations"
    ),
}


class ExecutableCandidateContext(StrictProtocolModel):
    candidate_id: CandidateId
    binding_status: Literal["resolved", "unresolved"]
    binding_kind: Literal["metadata_field", "non_metadata", "unresolved"]
    bound_metadata_fields: list[NonEmptyText] = Field(default_factory=list)
    allowed_quantitative_checks: list[NonEmptyText] = Field(default_factory=list)
    allowed_qualitative_checks: list[NonEmptyText] = Field(default_factory=list)
    unavailable_checks_with_reasons: dict[NonEmptyText, NonEmptyText] = Field(
        default_factory=dict
    )
    available_output_types: list[NonEmptyText] = Field(default_factory=list)
    sample_access_available: bool
    model_capabilities: list[ModelCapability] = Field(default_factory=list)
    structured_intervention_spec_available: bool = False


class AnalysisTaskSemanticError(ValueError):
    """Raised when a canonical AnalysisTask lacks a check prerequisite."""


def _unavailable_reason(
    check: str,
    requirement: CheckRequirement,
    *,
    binding: CandidateBinding,
    capabilities: ExecutiveCapabilities,
    runtime_constraints: list[str],
    structured_spec_available: bool,
) -> str | None:
    if len(binding.metadata_fields) < requirement.minimum_metadata_fields:
        return f"requires_{requirement.minimum_metadata_fields}_scientific_metadata_fields"
    if requirement.expected_output not in capabilities.available_output_types:
        return f"requires_{requirement.expected_output}_output"
    if requirement.requires_sample_access and not capabilities.sample_access:
        return "requires_sample_access"
    missing = set(requirement.required_model_capabilities) - set(
        capabilities.model_capabilities
    )
    if missing:
        return "requires_model_capabilities:" + ",".join(sorted(missing))
    if "retraining" in requirement.required_model_capabilities and (
        "no_retraining" in runtime_constraints
    ):
        return "blocked_by_no_retraining"
    if requirement.requires_structured_spec and not structured_spec_available:
        return "requires_structured_scientific_spec"
    return None


def build_executable_candidate_context(
    binding: CandidateBinding,
    *,
    capabilities: ExecutiveCapabilities,
    runtime_constraints: list[str],
    structured_spec_available: bool = False,
) -> ExecutableCandidateContext:
    allowed: dict[str, list[str]] = {"quantitative": [], "qualitative": []}
    unavailable: dict[str, str] = {}
    globally_declared = {
        "quantitative": capabilities.quantitative_checks,
        "qualitative": capabilities.qualitative_checks,
    }
    for target in ("quantitative", "qualitative"):
        for check in globally_declared[target]:
            requirement = CHECK_REQUIREMENTS.get(check)
            if requirement is None or requirement.target != target:
                unavailable[check] = "check_requirement_not_registered"
                continue
            reason = _unavailable_reason(
                check,
                requirement,
                binding=binding,
                capabilities=capabilities,
                runtime_constraints=runtime_constraints,
                structured_spec_available=structured_spec_available,
            )
            if reason is None:
                allowed[target].append(check)
            else:
                unavailable[check] = reason
    return ExecutableCandidateContext(
        candidate_id=binding.candidate_id,
        binding_status=binding.status,
        binding_kind=binding.kind,
        bound_metadata_fields=binding.metadata_fields,
        allowed_quantitative_checks=allowed["quantitative"],
        allowed_qualitative_checks=allowed["qualitative"],
        unavailable_checks_with_reasons=unavailable,
        available_output_types=capabilities.available_output_types,
        sample_access_available=capabilities.sample_access,
        model_capabilities=capabilities.model_capabilities,
        structured_intervention_spec_available=structured_spec_available,
    )


def required_model_capabilities(checks: list[str]) -> list[ModelCapability]:
    return list(dict.fromkeys(
        capability
        for check in checks
        for capability in CHECK_REQUIREMENTS[check].required_model_capabilities
    ))


def checks_require_sample_access(checks: list[str]) -> bool:
    return any(CHECK_REQUIREMENTS[check].requires_sample_access for check in checks)


def expected_outputs_for_checks(
    checks: list[str], available_output_types: list[str]
) -> list[str]:
    return list(dict.fromkeys(
        requirement.expected_output
        for check in checks
        if (requirement := CHECK_REQUIREMENTS[check]).expected_output
        in available_output_types
    ))


def validate_analysis_task_semantics(
    task: AnalysisTask,
    candidate_context: ExecutableCandidateContext,
    capabilities: ExecutiveCapabilities,
) -> None:
    allowed = (
        candidate_context.allowed_quantitative_checks
        if task.target == "quantitative"
        else candidate_context.allowed_qualitative_checks
    )
    unavailable = [check for check in task.requested_checks if check not in allowed]
    if unavailable:
        raise AnalysisTaskSemanticError(
            "analysis_task_semantically_incomplete:" + ",".join(unavailable)
        )
    minimum = max(
        CHECK_REQUIREMENTS[check].minimum_metadata_fields
        for check in task.requested_checks
    )
    if len(task.inputs.metadata_fields) < minimum:
        raise AnalysisTaskSemanticError(
            "analysis_task_missing_scientific_metadata_fields"
        )
    if not set(task.inputs.metadata_fields).issubset(
        set(candidate_context.bound_metadata_fields)
    ):
        raise AnalysisTaskSemanticError(
            "analysis_task_metadata_fields_not_candidate_bound"
        )
    expected_capabilities = required_model_capabilities(task.requested_checks)
    if task.inputs.required_model_capabilities != expected_capabilities:
        raise AnalysisTaskSemanticError(
            "analysis_task_model_capabilities_not_canonical"
        )
    needs_samples = checks_require_sample_access(task.requested_checks)
    if task.inputs.requires_sample_access != needs_samples:
        raise AnalysisTaskSemanticError(
            "analysis_task_sample_access_not_canonical"
        )
    if needs_samples and task.inputs.sample_limit is None:
        raise AnalysisTaskSemanticError("analysis_task_sample_limit_missing")
    if not set(task.inputs.required_model_capabilities).issubset(
        set(capabilities.model_capabilities)
    ):
        raise AnalysisTaskSemanticError("analysis_task_model_capability_unavailable")


def build_candidate_contexts(
    request: ExecutiveRequest,
) -> list[ExecutableCandidateContext]:
    context = request.routing_context
    bindings = [
        resolve_candidate_binding_for_request(candidate, request)
        for candidate in request.candidate_report.candidates
    ]
    return [
        build_executable_candidate_context(
            binding,
            capabilities=context.capabilities,
            runtime_constraints=context.audit_options.runtime_constraints,
            structured_spec_available=bool(set(binding.metadata_fields) & set(context.capabilities.structured_intervention_fields)),
        )
        for binding in bindings
    ]


def resolve_candidate_binding_for_request(candidate, request: ExecutiveRequest):
    from shortcutminer.experiment.candidate_binding import resolve_candidate_binding

    context = request.routing_context
    return resolve_candidate_binding(
        candidate,
        available_metadata_fields=context.capabilities.available_metadata_fields,
        label_column=context.task.label_column,
        addressing_fields=context.capabilities.addressing_fields,
        visual_measurement_bindings=context.capabilities.visual_measurement_bindings,
    )


__all__ = [
    "AnalysisTaskSemanticError",
    "CHECK_REQUIREMENTS",
    "CheckRequirement",
    "ExecutableCandidateContext",
    "build_candidate_contexts",
    "build_executable_candidate_context",
    "checks_require_sample_access",
    "expected_outputs_for_checks",
    "required_model_capabilities",
    "validate_analysis_task_semantics",
]
