"""Serializable branch outputs used at the LangGraph fan-in boundary."""

from typing import Literal

from pydantic import model_validator

from shortcutminer.protocol.identify import (
    ProtocolVersion,
    SafeRunId,
    StrictProtocolModel,
)
from shortcutminer.protocol.qualitative import QualitativeResult
from shortcutminer.protocol.quantitative import QuantitativeResult


class QuantitativeResultBatch(StrictProtocolModel):
    protocol_version: ProtocolVersion
    run_id: SafeRunId
    source: Literal["quantitative"]
    target: Literal["summarize"]
    message_type: Literal["quantitative_result_batch"]
    results: list[QuantitativeResult]

    @model_validator(mode="after")
    def validate_result_correlations(self):
        _validate_batch_results(self, "quantitative")
        return self


class QualitativeResultBatch(StrictProtocolModel):
    protocol_version: ProtocolVersion
    run_id: SafeRunId
    source: Literal["qualitative"]
    target: Literal["summarize"]
    message_type: Literal["qualitative_result_batch"]
    results: list[QualitativeResult]

    @model_validator(mode="after")
    def validate_result_correlations(self):
        _validate_batch_results(self, "qualitative")
        return self


def _validate_batch_results(batch, branch: str) -> None:
    """Keep persisted fan-in batches internally correlated and unambiguous."""
    message_ids: list[str] = []
    task_ids: list[str] = []
    for result in batch.results:
        if result.protocol_version != batch.protocol_version:
            raise ValueError(f"{branch} result protocol_version differs from batch")
        if result.run_id != batch.run_id:
            raise ValueError(f"{branch} result run_id differs from batch")
        message_ids.append(result.message_id)
        task_ids.append(result.payload.task_id)
    if len(message_ids) != len(set(message_ids)):
        raise ValueError(f"{branch} result message_ids must be unique")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"{branch} result task_ids must be unique")
