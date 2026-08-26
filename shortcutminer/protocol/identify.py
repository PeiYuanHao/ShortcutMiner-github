from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)


class StrictProtocolModel(BaseModel):
    """Shared validation policy for Identify protocol objects."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

ProtocolVersion = Literal["1.0"]

CueFamily = Literal[
    "appearance",
    "spatial",
    "context",
    "geometry",
    "acquisition",
    "metadata",
    "other",
]

EvidenceSource = Literal[
    "dataset_structure",
    "metadata",
    "sample_inspection",
    "image_statistics",
]

CandidatePriority = Literal[
    "high",
    "medium",
    "low",
]

CandidateStatus = Literal[
    "unverified",
]


def _validate_run_id(value: str) -> str:
    if len(value) > 128:
        raise ValueError("run_id must contain at most 128 characters")

    if not value[0].isascii() or not value[0].isalnum():
        raise ValueError("run_id must begin with an ASCII letter or digit")

    if not all(
        character.isascii()
        and (character.isalnum() or character in "._-")
        for character in value
    ):
        raise ValueError(
            "run_id may contain only ASCII letters, digits, '.', '_', and '-'"
        )

    windows_stem = value.split(".", maxsplit=1)[0].upper()
    reserved_windows_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    if windows_stem in reserved_windows_names:
        raise ValueError("run_id must not be a reserved filesystem name")

    if value.endswith((".", " ")):
        raise ValueError("run_id must not end with a dot or space")

    return value


SafeRunId = Annotated[NonEmptyText, AfterValidator(_validate_run_id)]


def _validate_safe_relative_path(value: str) -> str:
    if "\x00" in value:
        raise ValueError("path must not contain a NUL character")

    normalized = value.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(value)

    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    ):
        raise ValueError("path must be relative")

    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            "path must not contain empty, current-directory, or parent-directory segments"
        )

    if any(part.endswith((".", " ")) for part in parts):
        raise ValueError("path segments must not end with a dot or space")

    return value


SafeRelativePath = Annotated[
    NonEmptyText,
    AfterValidator(_validate_safe_relative_path),
]

CandidateId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^C(?:00[1-9]|0[1-9][0-9]|[1-9][0-9]{2})$",
    ),
]


class DatasetSpec(StrictProtocolModel):
    name: NonEmptyText
    root: NonEmptyText
    metadata_file: SafeRelativePath | None = None


class TaskSpec(StrictProtocolModel):
    type: NonEmptyText
    label_column: NonEmptyText
    description: NonEmptyText | None = None
    intended_signal: NonEmptyText | None = None


class IdentifyBudget(StrictProtocolModel):
    max_samples: int = Field(default=40, ge=1)
    max_candidates: int = Field(default=5, ge=1, le=999)
    max_dataset_entries: int = Field(default=10_000, ge=1)
    max_metadata_rows: int = Field(default=100_000, ge=1)


class IdentifyRequest(StrictProtocolModel):
    protocol_version: ProtocolVersion
    run_id: SafeRunId
    message_id: NonEmptyText
    source: Literal["orchestrator"]
    target: Literal["identify"]
    message_type: Literal["identify_request"]

    dataset: DatasetSpec
    task: TaskSpec

    budget: IdentifyBudget = Field(default_factory=IdentifyBudget)


class Cue(StrictProtocolModel):
    family: CueFamily
    name: NonEmptyText


class Candidate(StrictProtocolModel):
    candidate_id: CandidateId

    cue: Cue

    description: NonEmptyText
    observation: NonEmptyText
    hypothesis: NonEmptyText

    evidence_sources: list[EvidenceSource] = Field(min_length=1)

    priority: CandidatePriority
    status: CandidateStatus


class DatasetInspectionSummary(StrictProtocolModel):
    dataset: NonEmptyText
    samples_inspected: int = Field(ge=0)
    metadata_used: bool


class CandidateReport(StrictProtocolModel):
    protocol_version: ProtocolVersion
    run_id: SafeRunId
    message_id: NonEmptyText
    source: Literal["identify"]
    target: Literal["executive"]
    message_type: Literal["candidate_report"]
    in_reply_to: NonEmptyText
    artifact_references: list[SafeRelativePath]

    dataset_summary: DatasetInspectionSummary

    candidates: list[Candidate]

    limitations: list[NonEmptyText]
