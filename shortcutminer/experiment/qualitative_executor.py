from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from shortcutminer.agent_client import AgentClient
from shortcutminer.experiment.agent_workspace import load_agent_workspace_prompt
from shortcutminer.experiment.qualitative_context import (
    QualitativeEvidenceContext,
    build_qualitative_context,
)
from shortcutminer.experiment.qualitative_backend import sample_has_inspection_evidence
from shortcutminer.protocol.qualitative import (
    QualitativeAgentDraft,
    QualitativeInterpretation,
    QualitativeObservation,
    QualitativeRequest,
    QualitativeResult,
    QualitativeResultPayload,
    QualitativeSampling,
)
from shortcutminer.runtime.openai_compatible import canonical_json_hash
from shortcutminer.storage.events import append_event
from shortcutminer.storage.qualitative_artifacts import (
    save_qualitative_draft_invariant_telemetry,
    save_qualitative_input,
    save_qualitative_raw_output,
    save_qualitative_result,
)
from shortcutminer.tooling.agent_loop import invoke_agent_with_tools
from shortcutminer.tooling.runtime import ToolRuntime


MAX_DRAFT_INVARIANT_RETRIES = 1
_RETRYABLE_DRAFT_INVARIANTS = frozenset({"observation_crosses_groups"})


class QualitativeExecutionError(RuntimeError):
    """Raised when Qualitative cannot produce a valid evidence result."""


class QualitativeDraftValidationError(QualitativeExecutionError):
    """A typed scientific-draft invariant failure before finalization."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        observation_index: int | None = None,
        referenced_group_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.observation_index = observation_index
        self.referenced_group_count = referenced_group_count

    def safe_details(self) -> dict[str, int | str]:
        details: dict[str, int | str] = {"category": self.category}
        if self.observation_index is not None:
            details["observation_index"] = self.observation_index
        if self.referenced_group_count is not None:
            details["referenced_group_count"] = self.referenced_group_count
        return details


def _draft_invariant_correction(error: QualitativeDraftValidationError) -> str:
    """Build value-free correction feedback for one retryable draft invariant."""
    if error.category != "observation_crosses_groups":
        raise QualitativeExecutionError(
            "No bounded correction is registered for this draft invariant."
        )
    safe_lines = [
        "Your previous QualitativeAgentDraft was valid JSON and matched the "
        "provider-facing schema, but one observation referenced samples from "
        "more than one authoritative group.",
        "",
        "Each observation must contain sample_refs from exactly one group.",
        "",
        "For a cross-group comparison:",
        "- create separate group-local observations;",
        "- keep the same inspected evidence;",
        "- compare those observations in an interpretation using "
        "supporting_observation_indexes.",
        "",
        "Do not invent or replace samples.",
        "Do not change the candidate or hypothesis.",
        "Do not change scientific conclusions merely to satisfy structure.",
        "Return exactly one revised QualitativeAgentDraft JSON object.",
        "",
        f"Safe failure category: {error.category}",
    ]
    if error.observation_index is not None:
        safe_lines.append(f"Observation index: {error.observation_index}")
    if error.referenced_group_count is not None:
        safe_lines.append(
            f"Referenced group count: {error.referenced_group_count}"
        )
    return "\n".join(safe_lines)


def build_qualitative_agent_payload(
    request: QualitativeRequest,
    context: QualitativeEvidenceContext,
) -> dict:
    return {
        "request": request.model_dump(mode="json"),
        "evidence_context": context.model_dump(mode="json"),
    }


def validate_qualitative_request(request: QualitativeRequest) -> None:
    task = request.task
    candidate = request.candidate
    if request.in_reply_to != task.message_id:
        raise QualitativeExecutionError(
            "QualitativeRequest in_reply_to does not match AnalysisTask."
        )
    if request.message_id == task.message_id:
        raise QualitativeExecutionError(
            "QualitativeRequest message_id must identify a new message."
        )
    if task.target != "qualitative" or not task.task_id.endswith("-QL"):
        raise QualitativeExecutionError(
            "QualitativeRequest must contain a qualitative AnalysisTask."
        )
    if task.status != "planned":
        raise QualitativeExecutionError("Qualitative AnalysisTask is not planned.")
    if task.protocol_version != request.protocol_version:
        raise QualitativeExecutionError(
            "AnalysisTask protocol_version does not match QualitativeRequest."
        )
    if task.run_id != request.run_id:
        raise QualitativeExecutionError(
            "AnalysisTask run_id does not match QualitativeRequest."
        )
    if task.candidate_id != candidate.candidate_id:
        raise QualitativeExecutionError(
            "AnalysisTask candidate_id does not match the candidate."
        )
    if task.hypothesis != candidate.hypothesis:
        raise QualitativeExecutionError(
            "AnalysisTask changed the candidate hypothesis."
        )
    if task.inputs.requires_sample_access and task.inputs.sample_limit is None:
        raise QualitativeExecutionError(
            "Sample access requires an explicit sample_limit."
        )
    if (
        task.inputs.sample_limit is not None
        and request.preparation.max_evidence_records > task.inputs.sample_limit
    ):
        raise QualitativeExecutionError(
            "Qualitative preparation exceeds the AnalysisTask sample_limit."
        )


def validate_qualitative_context(
    context: QualitativeEvidenceContext,
    request: QualitativeRequest,
) -> None:
    validate_qualitative_request(request)
    if context.protocol_version != request.protocol_version:
        raise QualitativeExecutionError(
            "QualitativeEvidenceContext protocol_version does not match request."
        )
    if context.run_id != request.run_id:
        raise QualitativeExecutionError(
            "QualitativeEvidenceContext run_id does not match request."
        )
    if context.task_id != request.task.task_id:
        raise QualitativeExecutionError(
            "QualitativeEvidenceContext task_id does not match request."
        )
    if context.candidate_id != request.candidate.candidate_id:
        raise QualitativeExecutionError(
            "QualitativeEvidenceContext candidate_id does not match request."
        )
    if context.seed != request.preparation.seed:
        raise QualitativeExecutionError("Qualitative evidence seed changed.")
    if context.sampling_strategy != request.preparation.sampling_strategy:
        raise QualitativeExecutionError("Qualitative sampling strategy changed.")
    if context.samples_selected != len(context.selected_samples):
        raise QualitativeExecutionError(
            "Qualitative samples_selected does not match selected_samples."
        )
    if context.samples_selected > request.preparation.max_evidence_records:
        raise QualitativeExecutionError(
            "Qualitative evidence exceeds max_evidence_records."
        )
    available_count = sum(
        sample.sample_available and sample.media_type in {"image", "text"}
        for sample in context.selected_samples
    )
    if context.sample_files_available != available_count:
        raise QualitativeExecutionError(
            "Qualitative sample_files_available is inconsistent."
        )
    sample_ids = [sample.sample_id for sample in context.selected_samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise QualitativeExecutionError(
            "QualitativeEvidenceContext contains duplicate sample IDs."
        )
    inspected_count = sum(
        sample_has_inspection_evidence(sample) for sample in context.selected_samples
    )
    failed_count = sum(
        sample.inspection_status == "failed"
        for sample in context.selected_samples
    )
    if context.samples_inspected != inspected_count:
        raise QualitativeExecutionError(
            "Qualitative samples_inspected is inconsistent."
        )
    if context.failed_inspections != failed_count:
        raise QualitativeExecutionError(
            "Qualitative failed_inspections is inconsistent."
        )
    if context.authoritative_artifact_references != request.task.inputs.artifact_references:
        raise QualitativeExecutionError(
            "QualitativeEvidenceContext changed authoritative artifacts."
        )
    missing_limitations = [
        item for item in request.limitations if item not in context.limitations
    ]
    if missing_limitations:
        raise QualitativeExecutionError(
            "QualitativeEvidenceContext omitted request limitations: "
            + "; ".join(missing_limitations)
        )


def validate_qualitative_draft(
    draft: QualitativeAgentDraft,
    request: QualitativeRequest,
    context: QualitativeEvidenceContext,
) -> None:
    """Validate only scientific choices retained in provider ownership."""
    validate_qualitative_context(context, request)
    selected_by_id = {sample.sample_id: sample for sample in context.selected_samples}
    selected_ids = set(selected_by_id)
    sampled_ids = set(draft.sample_ids)
    if not sampled_ids.issubset(selected_ids):
        raise QualitativeDraftValidationError(
            "unknown_sample_reference",
            "Qualitative draft references a sample outside selected evidence.",
        )
    if request.task.inputs.requires_sample_access:
        unavailable = [
            sample_id for sample_id in draft.sample_ids
            if (
                not selected_by_id[sample_id].sample_available
                or selected_by_id[sample_id].inspection_status == "failed"
                or not sample_has_inspection_evidence(selected_by_id[sample_id])
            )
        ]
        if unavailable:
            raise QualitativeDraftValidationError(
                "unavailable_sample_reference",
                "Qualitative draft references unavailable or uninspected evidence.",
            )
    for observation_index, observation in enumerate(draft.observations):
        if not set(observation.sample_refs).issubset(sampled_ids):
            raise QualitativeDraftValidationError(
                "unknown_sample_reference",
                "Qualitative observation references an unselected draft sample.",
            )
        groups = {selected_by_id[item].group for item in observation.sample_refs}
        if len(groups) != 1:
            raise QualitativeDraftValidationError(
                "observation_crosses_groups",
                "Qualitative observation sample references cross groups.",
                observation_index=observation_index,
                referenced_group_count=len(groups),
            )
    observation_count = len(draft.observations)
    for interpretation in draft.interpretations:
        if any(
            index >= observation_count
            for index in interpretation.supporting_observation_indexes
        ):
            raise QualitativeDraftValidationError(
                "unknown_observation_index",
                "Qualitative interpretation references an unknown observation index.",
            )
    for counterevidence in draft.counterevidence:
        if not set(counterevidence.sample_refs).issubset(sampled_ids):
            raise QualitativeDraftValidationError(
                "unknown_sample_reference",
                "Qualitative counterevidence references an unselected draft sample.",
            )
    permitted_metadata = {
        request.task.inputs.label_column,
        *request.task.inputs.metadata_fields,
    }
    for evidence in draft.metadata_evidence:
        if evidence.field not in permitted_metadata:
            raise QualitativeDraftValidationError(
                "unauthorized_metadata_field",
                "Qualitative draft uses an unauthorized metadata field.",
            )
        if not set(evidence.sample_refs).issubset(sampled_ids):
            raise QualitativeDraftValidationError(
                "unknown_sample_reference",
                "Qualitative metadata evidence references an unselected draft sample.",
            )
    if not context.metadata_available and draft.metadata_evidence:
        raise QualitativeDraftValidationError(
            "metadata_unavailable",
            "Qualitative draft uses metadata evidence when metadata is unavailable.",
        )
    for failure in draft.failures:
        if not set(failure.sample_refs).issubset(selected_ids):
            raise QualitativeDraftValidationError(
                "unknown_sample_reference",
                "Qualitative failure references an unknown sample.",
            )
        if not all(
            _is_allowed_artifact(reference, context)
            for reference in failure.artifact_references
        ):
            raise QualitativeDraftValidationError(
                "unauthorized_artifact_reference",
                "Qualitative failure references an unauthorized artifact.",
            )
    if not all(
        _is_allowed_artifact(reference, context)
        for reference in draft.artifact_references
    ):
        raise QualitativeDraftValidationError(
            "unauthorized_artifact_reference",
            "Qualitative draft references an unauthorized artifact.",
        )


def finalize_qualitative_draft(
    draft: QualitativeAgentDraft,
    request: QualitativeRequest,
    context: QualitativeEvidenceContext,
) -> QualitativeResult:
    """Assemble canonical protocol bookkeeping without changing draft science."""
    validate_qualitative_draft(draft, request, context)
    selected_by_id = {sample.sample_id: sample for sample in context.selected_samples}
    groups: dict[str, int] = {}
    for sample_id in draft.sample_ids:
        group = selected_by_id[sample_id].group
        groups[group] = groups.get(group, 0) + 1
    observation_ids = [
        f"QO-{request.candidate.candidate_id}-{index:03d}"
        for index in range(1, len(draft.observations) + 1)
    ]
    observations = [
        QualitativeObservation(
            observation_id=observation_ids[index],
            group=selected_by_id[item.sample_refs[0]].group,
            sample_refs=item.sample_refs,
            observation=item.observation,
            inspection_quality=item.inspection_quality,
        )
        for index, item in enumerate(draft.observations)
    ]
    interpretations = [
        QualitativeInterpretation(
            interpretation=item.interpretation,
            supporting_observation_ids=[
                observation_ids[index]
                for index in item.supporting_observation_indexes
            ],
            confidence=item.confidence,
        )
        for item in draft.interpretations
    ]
    return QualitativeResult(
        protocol_version=request.protocol_version,
        run_id=request.run_id,
        message_id=(
            f"{request.run_id}:qualitative:{request.task.task_id}:result"
        ),
        source="qualitative",
        target="summarize",
        message_type="qualitative_analysis_result",
        in_reply_to=request.message_id,
        payload=QualitativeResultPayload(
            task_id=request.task.task_id,
            candidate_id=request.candidate.candidate_id,
            candidate_name=request.candidate.cue.name,
            hypothesis=request.candidate.hypothesis,
            evidence_status=draft.evidence_status,
            evidence_summary=draft.evidence_summary,
            sampling=QualitativeSampling(
                strategy=context.sampling_strategy,
                seed=context.seed,
                groups=groups,
                sample_count=len(draft.sample_ids),
                sample_ids=draft.sample_ids,
            ),
            observations=observations,
            interpretations=interpretations,
            counterevidence=draft.counterevidence,
            metadata_evidence=draft.metadata_evidence,
            failures=draft.failures,
            limitations=list(dict.fromkeys([
                *context.limitations,
                *draft.additional_limitations,
            ])),
            artifact_references=list(dict.fromkeys([
                *draft.artifact_references,
                *context.inspection_artifact_references,
            ])),
            recommended_followup=draft.recommended_followup,
        ),
    )


def _is_allowed_artifact(reference: str, context: QualitativeEvidenceContext) -> bool:
    if reference in context.authoritative_artifact_references:
        return True
    path = PurePosixPath(reference.replace("\\", "/"))
    prefix = PurePosixPath(context.allowed_output_artifact_prefix)
    return path != prefix and prefix in path.parents


def _validate_artifact_files(
    result: QualitativeResult,
    context: QualitativeEvidenceContext,
    runs_root: str | Path | None,
) -> None:
    if runs_root is None:
        return
    run_dir = Path(runs_root).resolve(strict=False) / result.run_id
    authoritative = set(context.authoritative_artifact_references)
    references = list(result.payload.artifact_references)
    for failure in result.payload.failures:
        references.extend(failure.artifact_references)
    for reference in dict.fromkeys(references):
        if reference in authoritative:
            continue
        path = (run_dir / reference).resolve(strict=False)
        try:
            path.relative_to(run_dir.resolve(strict=False))
        except ValueError as exc:
            raise QualitativeExecutionError(
                "Qualitative result artifact escaped the run directory."
            ) from exc
        if not path.is_file():
            raise QualitativeExecutionError(
                f"Qualitative result references a missing derived artifact: {reference}"
            )


def validate_qualitative_result(
    result: QualitativeResult,
    request: QualitativeRequest,
    context: QualitativeEvidenceContext,
    *,
    runs_root: str | Path | None = None,
) -> None:
    validate_qualitative_context(context, request)
    payload = result.payload

    if result.protocol_version != request.protocol_version:
        raise QualitativeExecutionError(
            "QualitativeResult protocol_version does not match request."
        )
    if result.run_id != request.run_id:
        raise QualitativeExecutionError("QualitativeResult run_id does not match request.")
    if result.in_reply_to != request.message_id:
        raise QualitativeExecutionError(
            "QualitativeResult in_reply_to does not match request."
        )
    if result.message_id in {request.message_id, request.task.message_id}:
        raise QualitativeExecutionError(
            "QualitativeResult message_id must identify a new message."
        )
    if payload.task_id != request.task.task_id:
        raise QualitativeExecutionError("QualitativeResult changed task_id.")
    if payload.candidate_id != request.candidate.candidate_id:
        raise QualitativeExecutionError("QualitativeResult changed candidate_id.")
    if payload.candidate_name != request.candidate.cue.name:
        raise QualitativeExecutionError("QualitativeResult changed candidate_name.")
    if payload.hypothesis != request.candidate.hypothesis:
        raise QualitativeExecutionError("QualitativeResult changed the hypothesis.")

    selected_by_id = {sample.sample_id: sample for sample in context.selected_samples}
    selected_ids = set(selected_by_id)
    sampled_ids = payload.sampling.sample_ids
    if payload.sampling.strategy != context.sampling_strategy:
        raise QualitativeExecutionError("QualitativeResult changed sampling strategy.")
    if payload.sampling.seed != context.seed:
        raise QualitativeExecutionError("QualitativeResult changed sampling seed.")
    if payload.sampling.sample_count != len(sampled_ids):
        raise QualitativeExecutionError(
            "QualitativeResult sample_count does not match sample_ids."
        )
    if not set(sampled_ids).issubset(selected_ids):
        raise QualitativeExecutionError(
            "QualitativeResult references an unselected sample."
        )
    if request.task.inputs.requires_sample_access:
        unavailable = [
            sample_id
            for sample_id in sampled_ids
            if (
                not selected_by_id[sample_id].sample_available
                or selected_by_id[sample_id].inspection_status == "failed"
                or not sample_has_inspection_evidence(selected_by_id[sample_id])
            )
        ]
        if unavailable:
            raise QualitativeExecutionError(
                "QualitativeResult claims inspection of unavailable sample files."
            )

    expected_groups: dict[str, int] = {}
    for sample_id in sampled_ids:
        group = selected_by_id[sample_id].group
        expected_groups[group] = expected_groups.get(group, 0) + 1
    if payload.sampling.groups != expected_groups:
        raise QualitativeExecutionError(
            "QualitativeResult group counts do not match sampled evidence."
        )

    observation_ids = [item.observation_id for item in payload.observations]
    expected_observation_ids = [
        f"QO-{payload.candidate_id}-{index:03d}"
        for index in range(1, len(observation_ids) + 1)
    ]
    if observation_ids != expected_observation_ids:
        raise QualitativeExecutionError(
            "Qualitative observation IDs must be unique and sequential."
        )
    observation_id_set = set(observation_ids)
    for observation in payload.observations:
        if not set(observation.sample_refs).issubset(set(sampled_ids)):
            raise QualitativeExecutionError(
                "Qualitative observation references an uninspected sample."
            )
        referenced_groups = {
            selected_by_id[sample_id].group
            for sample_id in observation.sample_refs
        }
        if referenced_groups != {observation.group}:
            raise QualitativeExecutionError(
                "Qualitative observation group does not match its referenced "
                "samples."
            )
        referenced_statuses = {
            selected_by_id[sample_id].inspection_status
            for sample_id in observation.sample_refs
        }
        if any(
            not sample_has_inspection_evidence(selected_by_id[sample_id])
            for sample_id in observation.sample_refs
        ):
            raise QualitativeExecutionError(
                "Qualitative observation references a sample without inspection evidence."
            )
        if "failed" in referenced_statuses:
            raise QualitativeExecutionError(
                "Qualitative observation references a failed inspection."
            )
        if (
            observation.inspection_quality == "sufficient"
            and referenced_statuses != {"completed"}
        ):
            raise QualitativeExecutionError(
                "Sufficient observation quality requires completed inspections."
            )
    for interpretation in payload.interpretations:
        if not set(interpretation.supporting_observation_ids).issubset(
            observation_id_set
        ):
            raise QualitativeExecutionError(
                "Qualitative interpretation references an unknown observation."
            )
    for counterevidence in payload.counterevidence:
        if not set(counterevidence.sample_refs).issubset(set(sampled_ids)):
            raise QualitativeExecutionError(
                "Qualitative counterevidence references an uninspected sample."
            )
    permitted_metadata = {
        request.task.inputs.label_column,
        *request.task.inputs.metadata_fields,
    }
    for evidence in payload.metadata_evidence:
        if evidence.field not in permitted_metadata:
            raise QualitativeExecutionError(
                "QualitativeResult uses an unauthorized metadata field."
            )
        if not set(evidence.sample_refs).issubset(set(sampled_ids)):
            raise QualitativeExecutionError(
                "Metadata evidence references an uninspected record."
            )
    if not context.metadata_available and payload.metadata_evidence:
        raise QualitativeExecutionError(
            "Metadata evidence must be empty when qualitative metadata is unavailable."
        )

    for failure in payload.failures:
        if not set(failure.sample_refs).issubset(selected_ids):
            raise QualitativeExecutionError(
                "Qualitative failure references an unknown sample."
            )
        if not all(
            _is_allowed_artifact(reference, context)
            for reference in failure.artifact_references
        ):
            raise QualitativeExecutionError(
                "Qualitative failure references an unauthorized artifact."
            )

    if not all(
        _is_allowed_artifact(reference, context)
        for reference in payload.artifact_references
    ):
        raise QualitativeExecutionError(
            "QualitativeResult references an unauthorized artifact."
        )
    _validate_artifact_files(result, context, runs_root)

    missing_limitations = [
        item for item in context.limitations if item not in payload.limitations
    ]
    if missing_limitations:
        raise QualitativeExecutionError(
            "QualitativeResult omitted evidence-context limitations: "
            + "; ".join(missing_limitations)
        )

    if payload.evidence_status == "insufficient_evidence":
        if not payload.limitations and not payload.failures:
            raise QualitativeExecutionError(
                "insufficient_evidence requires a limitation or failure."
            )
    else:
        if not sampled_ids:
            raise QualitativeExecutionError(
                "A substantive evidence status requires inspected evidence."
            )
        if not payload.observations and not payload.metadata_evidence:
            raise QualitativeExecutionError(
                "A substantive evidence status requires direct evidence."
            )

    if (
        request.task.inputs.requires_sample_access
        and context.samples_inspected == 0
        and payload.evidence_status != "insufficient_evidence"
    ):
        raise QualitativeExecutionError(
            "Unavailable required samples require insufficient_evidence."
        )


def execute_qualitative(
    request: QualitativeRequest,
    client: AgentClient,
    runs_root: str | Path | None = None,
    workspace_dir: str | Path | None = None,
    artifact_scope: str | None = None,
    tool_runtime: ToolRuntime | None = None,
) -> QualitativeResult:
    try:
        if runs_root is not None:
            append_event(
                runs_root=runs_root,
                run_id=request.run_id,
                event="QUALITATIVE_STARTED",
                stage="qualitative",
            )

        validate_qualitative_request(request)
        context = build_qualitative_context(request, runs_root=runs_root)
        validate_qualitative_context(context, request)
        if tool_runtime is not None:
            tool_runtime.approve_samples(
                [
                    sample.sample_reference
                    for sample in context.selected_samples
                    if sample.sample_reference is not None
                ]
            )
            tool_runtime.approve_artifacts(
                context.inspection_artifact_references
            )
        payload = build_qualitative_agent_payload(request, context)
        system_prompt = load_agent_workspace_prompt(
            "qualitative", workspace_dir
        )

        if runs_root is not None:
            save_qualitative_input(
                runs_root,
                run_id=request.run_id,
                payload=payload,
                scope=artifact_scope,
            )

        scientific_input_hash = canonical_json_hash(payload)
        draft_invariant_retry_count = 0
        draft_invariant_attempts: list[dict] = []
        correction: str | None = None
        while True:
            attempt_prompt = system_prompt
            if correction is not None:
                attempt_prompt = f"{system_prompt}\n\n{correction}"
            try:
                invocation = invoke_agent_with_tools(
                    client,
                    "qualitative",
                    payload,
                    system_prompt=attempt_prompt,
                    tool_runtime=tool_runtime,
                )
                raw_output = invocation.output
            except Exception as exc:
                raise QualitativeExecutionError(
                    f"Qualitative agent invocation failed: {exc}"
                ) from exc
            if not isinstance(raw_output, dict):
                raise QualitativeExecutionError(
                    "Qualitative agent returned a non-object response."
                )
            try:
                draft = QualitativeAgentDraft.model_validate(raw_output)
            except ValidationError as exc:
                raise QualitativeExecutionError(
                    "Qualitative agent returned an invalid QualitativeAgentDraft."
                ) from exc
            try:
                result = finalize_qualitative_draft(draft, request, context)
            except QualitativeDraftValidationError as exc:
                draft_invariant_attempts.append({
                    "attempt": len(draft_invariant_attempts) + 1,
                    "scientific_input_hash": scientific_input_hash,
                    "draft_schema_result": "passed",
                    "draft_invariant_result": "failed",
                    "failure": exc.safe_details(),
                })
                can_retry = (
                    exc.category in _RETRYABLE_DRAFT_INVARIANTS
                    and draft_invariant_retry_count
                    < MAX_DRAFT_INVARIANT_RETRIES
                )
                if not can_retry:
                    if runs_root is not None:
                        save_qualitative_draft_invariant_telemetry(
                            runs_root,
                            run_id=request.run_id,
                            scope=artifact_scope,
                            telemetry={
                                "schema_version": "1.0",
                                "run_id": request.run_id,
                                "task_id": request.task.task_id,
                                "candidate_id": request.candidate.candidate_id,
                                "max_draft_invariant_retries": (
                                    MAX_DRAFT_INVARIANT_RETRIES
                                ),
                                "draft_invariant_retry_count": (
                                    draft_invariant_retry_count
                                ),
                                "scientific_input_hash": scientific_input_hash,
                                "attempts": draft_invariant_attempts,
                                "status": "failed",
                            },
                        )
                    raise
                draft_invariant_retry_count += 1
                correction = _draft_invariant_correction(exc)
                if runs_root is not None:
                    append_event(
                        runs_root=runs_root,
                        run_id=request.run_id,
                        event="DRAFT_INVARIANT_RETRY_SCHEDULED",
                        stage="qualitative",
                        details={
                            "retry_index": draft_invariant_retry_count,
                            **exc.safe_details(),
                            "scientific_input_hash": scientific_input_hash,
                        },
                    )
                continue

            draft_invariant_attempts.append({
                "attempt": len(draft_invariant_attempts) + 1,
                "scientific_input_hash": scientific_input_hash,
                "draft_schema_result": "passed",
                "draft_invariant_result": "passed",
                "failure": None,
            })
            if runs_root is not None:
                save_qualitative_raw_output(
                    runs_root,
                    run_id=request.run_id,
                    raw_output=draft.model_dump(mode="json"),
                    scope=artifact_scope,
                )
                save_qualitative_draft_invariant_telemetry(
                    runs_root,
                    run_id=request.run_id,
                    scope=artifact_scope,
                    telemetry={
                        "schema_version": "1.0",
                        "run_id": request.run_id,
                        "task_id": request.task.task_id,
                        "candidate_id": request.candidate.candidate_id,
                        "max_draft_invariant_retries": (
                            MAX_DRAFT_INVARIANT_RETRIES
                        ),
                        "draft_invariant_retry_count": (
                            draft_invariant_retry_count
                        ),
                        "scientific_input_hash": scientific_input_hash,
                        "attempts": draft_invariant_attempts,
                        "status": "passed",
                    },
                )
            break

        validate_qualitative_result(
            result,
            request,
            context,
            runs_root=runs_root,
        )
        if runs_root is not None:
            save_qualitative_result(
                runs_root,
                result,
                scope=artifact_scope,
            )
            append_event(
                runs_root=runs_root,
                run_id=request.run_id,
                event="QUALITATIVE_COMPLETED",
                stage="qualitative",
                details={
                    "candidate_id": result.payload.candidate_id,
                    "evidence_status": result.payload.evidence_status,
                    "samples_inspected": result.payload.sampling.sample_count,
                    "observation_count": len(result.payload.observations),
                    "counterevidence_count": len(result.payload.counterevidence),
                },
            )
        return result
    except Exception as exc:
        if runs_root is not None:
            try:
                append_event(
                    runs_root=runs_root,
                    run_id=request.run_id,
                    event="AGENT_FAILED",
                    stage="qualitative",
                    details={
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            except Exception:
                pass
        if isinstance(exc, QualitativeExecutionError):
            raise
        raise QualitativeExecutionError(
            f"Qualitative execution failed: {exc}"
        ) from exc
