"""Deterministic binding from discovered candidates to executable inputs."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from shortcutminer.protocol.identify import (
    Candidate,
    CandidateId,
    NonEmptyText,
    StrictProtocolModel,
)


BindingStatus = Literal["resolved", "unresolved"]
BindingKind = Literal["metadata_field", "non_metadata", "unresolved"]


class CandidateBinding(StrictProtocolModel):
    candidate_id: CandidateId
    status: BindingStatus
    kind: BindingKind
    metadata_fields: list[NonEmptyText] = Field(default_factory=list)
    source: Literal[
        "exact_metadata_field_ref",
        "exact_candidate_cue_name",
        "candidate_non_metadata_family",
        "unresolved_candidate_cue",
    ]
    limitation: NonEmptyText | None = None


def resolve_candidate_binding(
    candidate: Candidate,
    *,
    available_metadata_fields: list[str],
    label_column: str,
    addressing_fields: list[str] | tuple[str, ...] | set[str] = (),
) -> CandidateBinding:
    """Resolve an explicit exact selector, or a historical exact cue name."""

    if candidate.cue.family != "metadata":
        return CandidateBinding(
            candidate_id=candidate.candidate_id,
            status="resolved",
            kind="non_metadata",
            source="candidate_non_metadata_family",
        )

    forbidden = {label_column, *addressing_fields}
    selected_field = candidate.metadata_field_ref or candidate.cue.name
    if selected_field in available_metadata_fields and selected_field not in forbidden:
        return CandidateBinding(
            candidate_id=candidate.candidate_id,
            status="resolved",
            kind="metadata_field",
            metadata_fields=[selected_field],
            source=(
                "exact_metadata_field_ref"
                if candidate.metadata_field_ref is not None
                else "exact_candidate_cue_name"
            ),
        )
    return CandidateBinding(
        candidate_id=candidate.candidate_id,
        status="unresolved",
        kind="unresolved",
        source="unresolved_candidate_cue",
        limitation=(
            "Metadata candidate selector did not exactly identify one authorized "
            "non-addressing metadata field."
        ),
    )


__all__ = [
    "BindingKind",
    "BindingStatus",
    "CandidateBinding",
    "resolve_candidate_binding",
]
