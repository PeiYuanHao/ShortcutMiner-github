from pathlib import Path, PurePosixPath
from typing import Protocol

from pydantic import ValidationError

from shortcutminer.experiment.qualitative_context import (
    QualitativeEvidenceContext,
    build_qualitative_context,
)
from shortcutminer.protocol.qualitative import (
    QualitativeRequest,
    QualitativeResult,
)
from shortcutminer.storage.events import append_event
from shortcutminer.storage.qualitative_artifacts import (
    save_qualitative_input,
    save_qualitative_raw_output,
    save_qualitative_result,
)


class AgentClient(Protocol):
    def invoke_agent(self, agent_name: str, payload: dict) -> dict:
        ...


class QualitativeExecutionError(RuntimeError):
    """Raised when Qualitative cannot produce a valid evidence result."""


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
    available_count = sum(sample.sample_available for sample in context.selected_samples)
    if context.sample_files_available != available_count:
        raise QualitativeExecutionError(
            "Qualitative sample_files_available is inconsistent."
        )
    sample_ids = [sample.sample_id for sample in context.selected_samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise QualitativeExecutionError(
            "QualitativeEvidenceContext contains duplicate sample IDs."
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
            if not selected_by_id[sample_id].sample_available
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
        and context.sample_files_available == 0
        and payload.evidence_status != "insufficient_evidence"
    ):
        raise QualitativeExecutionError(
            "Unavailable required samples require insufficient_evidence."
        )


def execute_qualitative(
    request: QualitativeRequest,
    client: AgentClient,
    runs_root: str | Path | None = None,
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
        context = build_qualitative_context(request)
        validate_qualitative_context(context, request)
        payload = build_qualitative_agent_payload(request, context)

        if runs_root is not None:
            save_qualitative_input(
                runs_root,
                run_id=request.run_id,
                payload=payload,
            )

        try:
            raw_output = client.invoke_agent("qualitative", payload)
        except Exception as exc:
            raise QualitativeExecutionError(
                f"Qualitative agent invocation failed: {exc}"
            ) from exc
        if not isinstance(raw_output, dict):
            raise QualitativeExecutionError(
                "Qualitative agent returned a non-object response."
            )

        if runs_root is not None:
            save_qualitative_raw_output(
                runs_root,
                run_id=request.run_id,
                raw_output=raw_output,
            )
        try:
            result = QualitativeResult.model_validate(raw_output)
        except ValidationError as exc:
            raise QualitativeExecutionError(
                "Qualitative agent returned an invalid QualitativeResult."
            ) from exc

        validate_qualitative_result(
            result,
            request,
            context,
            runs_root=runs_root,
        )
        if runs_root is not None:
            save_qualitative_result(runs_root, result)
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
