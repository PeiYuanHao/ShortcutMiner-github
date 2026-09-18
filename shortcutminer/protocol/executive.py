from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator

from shortcutminer.protocol.identify import (
    Candidate,
    CandidateId,
    CandidatePriority,
    CandidateReport,
    DatasetSpec,
    NonEmptyText,
    ProtocolVersion,
    SafeRelativePath,
    SafeRunId,
    StrictProtocolModel,
    TaskSpec,
)


ValidationTarget = Literal["quantitative", "qualitative"]
ValidationRoute = Literal[
    "quantitative",
    "qualitative",
    "both",
    "unresolved",
]
ValidationDepth = Literal["light", "standard", "deep"]
RuntimeConstraint = Literal[
    "no_retraining",
    "no_external_services",
]
ModelCapability = Literal[
    "predictions",
    "logits",
    "features",
    "checkpoint",
    "retraining",
    "intervention_inference",
]
TaskId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^TASK-C(?:00[1-9]|0[1-9][0-9]|[1-9][0-9]{2})-(?:QT|QL)$",
    ),
]


def _reject_duplicates(values: list[str], field_name: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


class ExecutiveAuditOptions(StrictProtocolModel):
    validation_depth: ValidationDepth = "standard"
    max_tasks: int = Field(default=10, ge=1, le=1_998)
    max_samples_per_task: int = Field(default=40, ge=1)
    allowed_targets: list[ValidationTarget] = Field(
        default_factory=lambda: ["quantitative", "qualitative"]
    )
    requested_outputs: list[NonEmptyText] = Field(default_factory=list)
    runtime_constraints: list[RuntimeConstraint] = Field(default_factory=list)

    @field_validator(
        "allowed_targets",
        "requested_outputs",
        "runtime_constraints",
    )
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)


class ExecutiveCapabilities(StrictProtocolModel):
    available_targets: list[ValidationTarget] = Field(default_factory=list)
    available_metadata_fields: list[NonEmptyText] = Field(
        default_factory=list
    )
    addressing_fields: list[NonEmptyText] = Field(default_factory=list)
    available_splits: list[NonEmptyText] = Field(default_factory=list)
    quantitative_checks: list[NonEmptyText] = Field(default_factory=list)
    qualitative_checks: list[NonEmptyText] = Field(default_factory=list)
    model_capabilities: list[ModelCapability] = Field(default_factory=list)
    sample_access: bool = False
    external_services: bool = False
    available_output_types: list[NonEmptyText] = Field(default_factory=list)

    @field_validator(
        "available_targets",
        "available_metadata_fields",
        "addressing_fields",
        "available_splits",
        "quantitative_checks",
        "qualitative_checks",
        "model_capabilities",
        "available_output_types",
    )
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)


class ExecutiveRoutingContext(StrictProtocolModel):
    protocol_version: ProtocolVersion
    run_id: SafeRunId
    dataset: DatasetSpec
    task: TaskSpec
    audit_options: ExecutiveAuditOptions = Field(
        default_factory=ExecutiveAuditOptions
    )
    capabilities: ExecutiveCapabilities = Field(
        default_factory=ExecutiveCapabilities
    )
    artifact_references: list[SafeRelativePath] = Field(default_factory=list)
    limitations: list[NonEmptyText] = Field(default_factory=list)

    @field_validator("artifact_references", "limitations")
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)


class ExecutiveRequest(StrictProtocolModel):
    protocol_version: ProtocolVersion
    run_id: SafeRunId
    message_id: NonEmptyText
    source: Literal["orchestrator"]
    target: Literal["executive"]
    message_type: Literal["executive_request"]
    in_reply_to: NonEmptyText
    candidate_report: CandidateReport
    routing_context: ExecutiveRoutingContext


class AnalysisTaskInputs(StrictProtocolModel):
    dataset: NonEmptyText
    dataset_reference: NonEmptyText
    label_column: NonEmptyText
    split: NonEmptyText | None = None
    metadata_fields: list[NonEmptyText] = Field(default_factory=list)
    artifact_references: list[SafeRelativePath] = Field(default_factory=list)
    sample_limit: int | None = Field(default=None, ge=1)
    requires_sample_access: bool
    requires_external_services: bool
    runtime_constraints: list[RuntimeConstraint]
    required_model_capabilities: list[ModelCapability] = Field(
        default_factory=list
    )

    @field_validator(
        "metadata_fields",
        "artifact_references",
        "runtime_constraints",
        "required_model_capabilities",
    )
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)


class AnalysisTask(StrictProtocolModel):
    protocol_version: ProtocolVersion
    run_id: SafeRunId
    message_id: NonEmptyText
    source: Literal["executive"]
    target: ValidationTarget
    message_type: Literal["analysis_task"]
    in_reply_to: NonEmptyText

    task_id: TaskId
    candidate_id: CandidateId
    hypothesis: NonEmptyText
    objective: NonEmptyText
    inputs: AnalysisTaskInputs
    requested_checks: list[NonEmptyText] = Field(min_length=1)
    expected_outputs: list[NonEmptyText] = Field(min_length=1)
    priority: CandidatePriority
    dependencies: list[TaskId]
    status: Literal["planned"]

    @field_validator(
        "requested_checks",
        "expected_outputs",
        "dependencies",
    )
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)


class RoutingDecision(StrictProtocolModel):
    candidate: Candidate
    route: ValidationRoute
    routing_reason: NonEmptyText
    task_ids: list[TaskId]
    priority: CandidatePriority
    status: Literal["planned", "unresolved"]
    missing_requirements: list[NonEmptyText]
    limitations: list[NonEmptyText]

    @field_validator(
        "task_ids",
        "missing_requirements",
        "limitations",
    )
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)


class RoutingPlan(StrictProtocolModel):
    protocol_version: ProtocolVersion
    run_id: SafeRunId
    message_id: NonEmptyText
    source: Literal["executive"]
    target: Literal["orchestrator"]
    message_type: Literal["routing_plan"]
    in_reply_to: NonEmptyText
    artifact_references: list[SafeRelativePath]

    decisions: list[RoutingDecision]
    tasks: list[AnalysisTask]
    limitations: list[NonEmptyText]

    @field_validator("artifact_references", "limitations")
    @classmethod
    def validate_unique_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)


class ExecutiveTaskDraft(StrictProtocolModel):
    objective: NonEmptyText
    requested_checks: list[NonEmptyText] = Field(min_length=1)

    @field_validator("requested_checks")
    @classmethod
    def validate_checks(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "requested_checks")


class ExecutiveDecisionDraft(StrictProtocolModel):
    candidate_id: CandidateId
    route: ValidationRoute
    routing_reason: NonEmptyText
    quantitative: ExecutiveTaskDraft | None = None
    qualitative: ExecutiveTaskDraft | None = None
    missing_requirements: list[NonEmptyText] = Field(default_factory=list)
    limitations: list[NonEmptyText] = Field(default_factory=list)

    @field_validator("missing_requirements", "limitations")
    @classmethod
    def validate_unique_draft_lists(cls, values: list[str], info) -> list[str]:
        return _reject_duplicates(values, info.field_name)


class ExecutiveAgentDraft(StrictProtocolModel):
    decisions: list[ExecutiveDecisionDraft]
    limitations: list[NonEmptyText] = Field(default_factory=list)

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, values: list[str]) -> list[str]:
        return _reject_duplicates(values, "limitations")
