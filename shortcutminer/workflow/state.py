"""Typed, serializable state carried through the ShortcutMiner workflow."""

from operator import add
from typing import Annotated, Any, Literal, NotRequired, Required, TypedDict


StageStatus = Literal["completed"]


def merge_stage_status(
    left: dict[str, StageStatus],
    right: dict[str, StageStatus],
) -> dict[str, StageStatus]:
    """Merge independent branch status updates at the graph fan-in."""
    return {**left, **right}


class IdentifyWorkflowConfig(TypedDict, total=False):
    seed: Required[int]
    message_id: NotRequired[str]
    budget: NotRequired[dict[str, Any]]
    runs_root: NotRequired[str]
    workspace_dir: NotRequired[str]
    visual_discovery: NotRequired[dict[str, Any]]


class ExecutiveWorkflowConfig(TypedDict, total=False):
    message_id: NotRequired[str]
    audit_options: NotRequired[dict[str, Any]]
    capabilities: NotRequired[dict[str, Any]]
    artifact_references: NotRequired[list[str]]
    limitations: NotRequired[list[str]]
    runs_root: NotRequired[str]
    workspace_dir: NotRequired[str]


class QuantitativeWorkflowConfig(TypedDict, total=False):
    limitations: NotRequired[list[str]]
    runs_root: NotRequired[str]
    workspace_dir: NotRequired[str]


class QualitativeWorkflowConfig(TypedDict, total=False):
    preparation: NotRequired[dict[str, Any]]
    limitations: NotRequired[list[str]]
    runs_root: NotRequired[str]
    workspace_dir: NotRequired[str]


class SummarizeWorkflowConfig(TypedDict, total=False):
    message_id: NotRequired[str]
    runs_root: NotRequired[str]
    workspace_dir: NotRequired[str]


class ShortcutMinerState(TypedDict, total=False):
    """Explicit state envelope for the complete five-agent workflow."""

    run_id: Required[str]
    dataset_info: Required[dict[str, Any]]
    task_input: Required[dict[str, Any]]
    model_runtime: NotRequired[dict[str, Any]]
    identify_config: Required[IdentifyWorkflowConfig]
    executive_config: NotRequired[ExecutiveWorkflowConfig]
    quantitative_config: NotRequired[QuantitativeWorkflowConfig]
    qualitative_config: NotRequired[QualitativeWorkflowConfig]
    summarize_config: NotRequired[SummarizeWorkflowConfig]

    identify_request: NotRequired[dict[str, Any]]
    dataset_profile: NotRequired[dict[str, Any]]
    visual_discovery: NotRequired[dict[str, Any]]
    identify_context: NotRequired[dict[str, Any]]
    identify_result: NotRequired[dict[str, Any]]
    executive_request: NotRequired[dict[str, Any]]
    executive_result: NotRequired[dict[str, Any]]
    quantitative_execution_requirements: NotRequired[dict[str, dict[str, Any]]]
    quantitative_result: NotRequired[dict[str, Any]]
    qualitative_result: NotRequired[dict[str, Any]]
    evidence_ledger: NotRequired[dict[str, Any]]
    summarize_request: NotRequired[dict[str, Any]]
    summarize_result: NotRequired[dict[str, Any]]
    # Backward-compatible export alias. summarize_result is the canonical
    # stage field and summarize_node always writes the same serialized report.
    final_result: NotRequired[dict[str, Any]]

    stage_status: Annotated[dict[str, StageStatus], merge_stage_status]
    errors: Annotated[list[str], add]
    metadata: dict[str, Any]
