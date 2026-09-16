"""Thin LangGraph nodes connecting state to existing agent executors."""

from typing import Any

from pydantic import ValidationError

from shortcutminer.agent_client import AgentClient
from shortcutminer.experiment.executive_context import build_executive_request
from shortcutminer.experiment.executive_executor import execute_executive
from shortcutminer.experiment.identify_context import (
    IdentifyDiscoveryContext,
    build_identify_context_from_profile,
)
from shortcutminer.experiment.dataset_profiler import build_dataset_profile
from shortcutminer.experiment.identify_executor import execute_identify
from shortcutminer.experiment.visual_discovery import VisualDiscoveryProfiler
from shortcutminer.inspection.semantic import SemanticVisualInspector
from shortcutminer.experiment.qualitative_context import build_qualitative_request
from shortcutminer.experiment.qualitative_executor import execute_qualitative
from shortcutminer.experiment.quantitative_context import build_quantitative_request
from shortcutminer.experiment.quantitative_executor import execute_quantitative
from shortcutminer.experiment.summarize_executor import execute_summarize
from shortcutminer.protocol.batches import (
    QualitativeResultBatch,
    QuantitativeResultBatch,
)
from shortcutminer.protocol.executive import (
    ExecutiveAuditOptions,
    ExecutiveCapabilities,
    ExecutiveRequest,
    RoutingPlan,
)
from shortcutminer.protocol.identify import (
    CandidateReport,
    DatasetSpec,
    IdentifyBudget,
    IdentifyRequest,
    TaskSpec,
)
from shortcutminer.protocol.dataset_profile import DatasetProfile
from shortcutminer.protocol.visual_discovery import VisualDiscoveryOptions
from shortcutminer.storage.dataset_profiles import (
    dataset_profile_exists,
    load_dataset_profile,
    save_dataset_profile,
)
from shortcutminer.protocol.qualitative import QualitativePreparationOptions
from shortcutminer.protocol.summarize import SummarizeRequest
from shortcutminer.protocol.evidence import EvidenceLedger
from shortcutminer.evaluation.evidence_ledger import EvidenceLedgerBuilder
from shortcutminer.tooling.runtime import ToolRuntime
from shortcutminer.workflow.state import ShortcutMinerState


class WorkflowNodeError(RuntimeError):
    """Base error for invalid workflow state or missing dependencies."""


class IdentifyNodeError(WorkflowNodeError):
    """Raised when state cannot execute Identify."""


class ExecutiveNodeError(WorkflowNodeError):
    """Raised when state cannot execute Executive."""


class QuantitativeNodeError(WorkflowNodeError):
    """Raised when state cannot execute Quantitative."""


class DownstreamPlanFeasibilityError(WorkflowNodeError):
    """Raised before fan-out when a planned downstream task is not executable."""


class QualitativeNodeError(WorkflowNodeError):
    """Raised when state cannot execute Qualitative."""


class SummarizeNodeError(WorkflowNodeError):
    """Raised when state cannot execute Summarize."""


def _require_client(
    client: AgentClient | None,
    stage: str,
    error_type: type[WorkflowNodeError],
) -> AgentClient:
    if client is None:
        raise error_type(f"{stage} workflow invocation requires an AgentClient.")
    return client


def _stage_config(state: ShortcutMinerState, name: str) -> dict[str, Any]:
    value = state.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _runs_root(state: ShortcutMinerState, config: dict[str, Any]) -> str | None:
    value = config.get("runs_root")
    if value is None:
        identify_config = state.get("identify_config", {})
        if isinstance(identify_config, dict):
            value = identify_config.get("runs_root")
    return value


def build_identify_request_from_state(state: ShortcutMinerState) -> IdentifyRequest:
    """Map the shared workflow state to the existing Identify protocol."""
    try:
        run_id = state["run_id"]
        dataset_info = state["dataset_info"]
        task_input = state["task_input"]
        config = state["identify_config"]
        if not isinstance(dataset_info, dict):
            raise TypeError("dataset_info must be an object")
        if not isinstance(task_input, dict):
            raise TypeError("task_input must be an object")
        if not isinstance(config, dict):
            raise TypeError("identify_config must be an object")
        seed = config["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("identify_config.seed must be an integer")
        return IdentifyRequest(
            protocol_version="1.0",
            run_id=run_id,
            message_id=config.get("message_id", f"{run_id}:identify:request"),
            source="orchestrator",
            target="identify",
            message_type="identify_request",
            dataset=DatasetSpec(
                name=dataset_info["name"],
                root=dataset_info["root"],
                metadata_file=dataset_info.get("metadata_file"),
            ),
            task=TaskSpec(
                type=task_input["type"],
                label_column=task_input["label_column"],
                description=task_input.get("description"),
                intended_signal=task_input.get("intended_signal"),
            ),
            budget=IdentifyBudget.model_validate(config.get("budget", {})),
        )
    except (KeyError, TypeError, ValidationError, ValueError) as exc:
        raise IdentifyNodeError(f"Invalid Identify workflow state: {exc}") from exc


def identify_node(
    state: ShortcutMinerState,
    *,
    client: AgentClient | None,
    tool_runtime: ToolRuntime | None = None,
    semantic_inspector: SemanticVisualInspector | None = None,
) -> dict[str, Any]:
    """Build deterministic context and invoke the Identify executor."""
    runtime = _require_client(client, "Identify", IdentifyNodeError)
    request = build_identify_request_from_state(state)
    config = state["identify_config"]
    runs_root = config.get("runs_root")
    serialized_profile = state.get("dataset_profile")
    if serialized_profile is not None:
        profile = DatasetProfile.model_validate(serialized_profile)
    elif runs_root is not None and dataset_profile_exists(
        runs_root, request.run_id
    ):
        profile = load_dataset_profile(runs_root, request.run_id)
    else:
        profile = build_dataset_profile(request, seed=config["seed"])
        if runs_root is not None:
            save_dataset_profile(runs_root, request.run_id, profile)
    raw_visual_options = config.get("visual_discovery", {})
    if not isinstance(raw_visual_options, dict):
        raise IdentifyNodeError("identify_config.visual_discovery must be an object")
    visual_options_data = dict(raw_visual_options)
    visual_options_data.setdefault("max_samples", request.budget.max_samples)
    try:
        visual_options = VisualDiscoveryOptions.model_validate(visual_options_data)
    except ValidationError as exc:
        raise IdentifyNodeError(f"Invalid visual discovery options: {exc}") from exc
    visual_discovery = None
    if profile.structure.modality in {"image", "mixed"} and visual_options.enabled:
        visual_discovery = VisualDiscoveryProfiler().profile(
            request,
            profile,
            seed=config["seed"],
            options=visual_options,
            runs_root=runs_root,
            semantic_inspector=semantic_inspector,
        )
    context = build_identify_context_from_profile(
        request,
        seed=config["seed"],
        profile=profile,
        visual_discovery=visual_discovery,
    )
    report = execute_identify(
        request=request,
        seed=config["seed"],
        client=runtime,
        runs_root=config.get("runs_root"),
        workspace_dir=config.get("workspace_dir"),
        context=context,
        tool_runtime=tool_runtime,
    )
    output = {
        "identify_request": request.model_dump(mode="json"),
        "dataset_profile": profile.model_dump(mode="json"),
        "identify_context": context.model_dump(mode="json"),
        "identify_result": report.model_dump(mode="json"),
        "stage_status": {"identify": "completed"},
    }
    return output


def build_executive_request_from_state(
    state: ShortcutMinerState,
) -> ExecutiveRequest:
    """Rehydrate Identify state and form validated Executive input."""
    try:
        identify_request = IdentifyRequest.model_validate(state["identify_request"])
        identify_context = IdentifyDiscoveryContext.model_validate(
            state["identify_context"]
        )
        identify_result = CandidateReport.model_validate(state["identify_result"])
        config = _stage_config(state, "executive_config")
        return build_executive_request(
            identify_request=identify_request,
            identify_context=identify_context,
            candidate_report=identify_result,
            message_id=config.get(
                "message_id", f"{identify_request.run_id}:executive:request"
            ),
            audit_options=ExecutiveAuditOptions.model_validate(
                config.get("audit_options", {})
            ),
            capabilities=ExecutiveCapabilities.model_validate(
                config.get("capabilities", {})
            ),
            artifact_references=config.get("artifact_references"),
            limitations=config.get("limitations"),
        )
    except (KeyError, TypeError, ValidationError, ValueError) as exc:
        raise ExecutiveNodeError(
            f"Invalid Executive workflow state: {exc}"
        ) from exc


def executive_node(
    state: ShortcutMinerState,
    *,
    client: AgentClient | None,
    tool_runtime: ToolRuntime | None = None,
) -> dict[str, Any]:
    """Invoke Executive with validated Identify state."""
    runtime = _require_client(client, "Executive", ExecutiveNodeError)
    request = build_executive_request_from_state(state)
    config = _stage_config(state, "executive_config")
    plan = execute_executive(
        request=request,
        client=runtime,
        runs_root=_runs_root(state, config),
        workspace_dir=config.get("workspace_dir"),
        tool_runtime=tool_runtime,
    )
    return {
        "executive_request": request.model_dump(mode="json"),
        "executive_result": plan.model_dump(mode="json"),
        "stage_status": {"executive": "completed"},
    }


def _executive_state(
    state: ShortcutMinerState,
    error_type: type[WorkflowNodeError],
) -> tuple[ExecutiveRequest, RoutingPlan]:
    try:
        return (
            ExecutiveRequest.model_validate(state["executive_request"]),
            RoutingPlan.model_validate(state["executive_result"]),
        )
    except (KeyError, ValidationError, TypeError, ValueError) as exc:
        raise error_type(f"Invalid Executive branch state: {exc}") from exc


def quantitative_node(
    state: ShortcutMinerState,
    *,
    client: AgentClient | None,
    tool_runtime: ToolRuntime | None = None,
) -> dict[str, Any]:
    """Execute every planned task in the quantitative branch."""
    runtime = _require_client(client, "Quantitative", QuantitativeNodeError)
    executive_request, plan = _executive_state(state, QuantitativeNodeError)
    config = _stage_config(state, "quantitative_config")
    results = []
    for task in plan.tasks:
        if task.target != "quantitative":
            continue
        request = build_quantitative_request(
            executive_request=executive_request,
            routing_plan=plan,
            task_id=task.task_id,
            message_id=f"{plan.run_id}:quantitative:{task.task_id}:request",
            limitations=config.get("limitations"),
        )
        results.append(
            execute_quantitative(
                request,
                runtime,
                runs_root=_runs_root(state, config),
                workspace_dir=config.get("workspace_dir"),
                tool_runtime=tool_runtime,
            )
        )
    batch = QuantitativeResultBatch(
        protocol_version=plan.protocol_version,
        run_id=plan.run_id,
        source="quantitative",
        target="summarize",
        message_type="quantitative_result_batch",
        results=results,
    )
    return {
        "quantitative_result": batch.model_dump(mode="json"),
        "stage_status": {"quantitative": "completed"},
    }


def _qualitative_preparation(
    state: ShortcutMinerState,
    task,
    config: dict[str, Any],
) -> QualitativePreparationOptions:
    raw = config.get("preparation", {})
    if not isinstance(raw, dict):
        raise QualitativeNodeError(
            "qualitative_config.preparation must be an object"
        )
    values = dict(raw)
    values.setdefault("seed", state["identify_config"]["seed"])
    values.setdefault("sampling_strategy", "random")
    values.setdefault("group_columns", [])
    if "max_evidence_records" not in values:
        values["max_evidence_records"] = min(
            40,
            task.inputs.sample_limit
            if task.inputs.sample_limit is not None
            else 40,
        )
    try:
        return QualitativePreparationOptions.model_validate(values)
    except ValidationError as exc:
        raise QualitativeNodeError(
            f"Invalid qualitative preparation: {exc}"
        ) from exc


def feasibility_node(state: ShortcutMinerState) -> dict[str, Any]:
    """Validate every routed task deterministically before either branch runs."""

    try:
        executive_request, plan = _executive_state(
            state, DownstreamPlanFeasibilityError
        )
        quantitative_config = _stage_config(state, "quantitative_config")
        qualitative_config = _stage_config(state, "qualitative_config")
        for task in plan.tasks:
            if task.target == "quantitative":
                build_quantitative_request(
                    executive_request=executive_request,
                    routing_plan=plan,
                    task_id=task.task_id,
                    message_id=(
                        f"{plan.run_id}:quantitative:{task.task_id}:request"
                    ),
                    limitations=quantitative_config.get("limitations"),
                )
            elif task.target == "qualitative":
                preparation = _qualitative_preparation(
                    state, task, qualitative_config
                )
                build_qualitative_request(
                    executive_request=executive_request,
                    routing_plan=plan,
                    task_id=task.task_id,
                    message_id=(
                        f"{plan.run_id}:qualitative:{task.task_id}:request"
                    ),
                    preparation=preparation,
                    limitations=qualitative_config.get("limitations"),
                )
            else:
                raise DownstreamPlanFeasibilityError(
                    f"Unsupported downstream target: {task.target}"
                )
    except DownstreamPlanFeasibilityError:
        raise
    except Exception as exc:
        raise DownstreamPlanFeasibilityError(
            f"Downstream plan is not executable: {exc}"
        ) from exc
    return {"stage_status": {"feasibility": "completed"}}


def qualitative_node(
    state: ShortcutMinerState,
    *,
    client: AgentClient | None,
    tool_runtime: ToolRuntime | None = None,
) -> dict[str, Any]:
    """Execute every planned task in the qualitative branch."""
    runtime = _require_client(client, "Qualitative", QualitativeNodeError)
    executive_request, plan = _executive_state(state, QualitativeNodeError)
    config = _stage_config(state, "qualitative_config")
    results = []
    for task in plan.tasks:
        if task.target != "qualitative":
            continue
        request = build_qualitative_request(
            executive_request=executive_request,
            routing_plan=plan,
            task_id=task.task_id,
            message_id=f"{plan.run_id}:qualitative:{task.task_id}:request",
            preparation=_qualitative_preparation(state, task, config),
            limitations=config.get("limitations"),
        )
        results.append(
            execute_qualitative(
                request,
                runtime,
                runs_root=_runs_root(state, config),
                workspace_dir=config.get("workspace_dir"),
                artifact_scope=task.task_id,
                tool_runtime=tool_runtime,
            )
        )
    batch = QualitativeResultBatch(
        protocol_version=plan.protocol_version,
        run_id=plan.run_id,
        source="qualitative",
        target="summarize",
        message_type="qualitative_result_batch",
        results=results,
    )
    return {
        "qualitative_result": batch.model_dump(mode="json"),
        "stage_status": {"qualitative": "completed"},
    }


def build_summarize_request_from_state(
    state: ShortcutMinerState,
) -> SummarizeRequest:
    """Require and validate both branch outputs before fan-in."""
    try:
        identify_result = CandidateReport.model_validate(state["identify_result"])
        executive_result = RoutingPlan.model_validate(state["executive_result"])
        quantitative_result = QuantitativeResultBatch.model_validate(
            state["quantitative_result"]
        )
        qualitative_result = QualitativeResultBatch.model_validate(
            state["qualitative_result"]
        )
        config = _stage_config(state, "summarize_config")
        if state.get("evidence_ledger") is not None:
            evidence_ledger = EvidenceLedger.model_validate(state["evidence_ledger"])
        else:
            evidence_ledger = EvidenceLedgerBuilder(
                apply_bh_fdr=bool(config.get("apply_bh_fdr", True))
            ).build(
                identify_result,
                executive_result,
                quantitative_result,
                qualitative_result,
                runs_root=_runs_root(state, config),
            )
        return SummarizeRequest(
            protocol_version="1.0",
            run_id=state["run_id"],
            message_id=config.get(
                "message_id", f"{state['run_id']}:summarize:request"
            ),
            source="orchestrator",
            target="summarize",
            message_type="summarize_request",
            in_reply_to=executive_result.message_id,
            identify_result=identify_result,
            executive_result=executive_result,
            quantitative_result=quantitative_result,
            qualitative_result=qualitative_result,
            evidence_ledger=evidence_ledger,
        )
    except (KeyError, TypeError, ValidationError, ValueError) as exc:
        raise SummarizeNodeError(
            f"Summarize requires complete branch state: {exc}"
        ) from exc


def evidence_calibration_node(state: ShortcutMinerState) -> dict[str, Any]:
    """Build or reuse the deterministic evidence ledger after branch fan-in."""
    try:
        identify_result = CandidateReport.model_validate(state["identify_result"])
        executive_result = RoutingPlan.model_validate(state["executive_result"])
        quantitative_result = QuantitativeResultBatch.model_validate(state["quantitative_result"])
        qualitative_result = QualitativeResultBatch.model_validate(state["qualitative_result"])
        config = _stage_config(state, "summarize_config")
        ledger = EvidenceLedgerBuilder(
            apply_bh_fdr=bool(config.get("apply_bh_fdr", True))
        ).build(
            identify_result,
            executive_result,
            quantitative_result,
            qualitative_result,
            runs_root=_runs_root(state, config),
        )
        return {
            "evidence_ledger": ledger.model_dump(mode="json"),
            "stage_status": {"evidence_calibration": "completed"},
        }
    except (KeyError, TypeError, ValidationError, ValueError, RuntimeError) as exc:
        raise SummarizeNodeError(f"Evidence calibration requires complete branch state: {exc}") from exc


def summarize_node(
    state: ShortcutMinerState,
    *,
    client: AgentClient | None,
    tool_runtime: ToolRuntime | None = None,
) -> dict[str, Any]:
    """Invoke Summarize after both branch result batches exist."""
    runtime = _require_client(client, "Summarize", SummarizeNodeError)
    request = build_summarize_request_from_state(state)
    config = _stage_config(state, "summarize_config")
    report = execute_summarize(
        request,
        runtime,
        runs_root=_runs_root(state, config),
        workspace_dir=config.get("workspace_dir"),
        tool_runtime=tool_runtime,
    )
    serialized = report.model_dump(mode="json")
    return {
        "evidence_ledger": request.evidence_ledger.model_dump(mode="json") if request.evidence_ledger else None,
        "summarize_request": request.model_dump(mode="json"),
        "summarize_result": serialized,
        "final_result": serialized,
        "stage_status": {"summarize": "completed"},
    }
