from typing import Literal

from pydantic import BaseModel, Field


CueFamily = Literal[
    "appearance",
    "spatial",
    "context",
    "geometry",
    "acquisition",
    "metadata",
    "other",
]

CandidatePriority = Literal[
    "high",
    "medium",
    "low",
]

CandidateStatus = Literal[
    "unverified",
]


class DatasetSpec(BaseModel):
    name: str
    root: str
    metadata_file: str | None = None


class TaskSpec(BaseModel):
    type: str
    label_column: str


class IdentifyBudget(BaseModel):
    max_samples: int = Field(default=40, ge=1)
    max_candidates: int = Field(default=5, ge=1)


class IdentifyRequest(BaseModel):
    protocol_version: str = "1.0"
    run_id: str

    dataset: DatasetSpec
    task: TaskSpec

    budget: IdentifyBudget = Field(
        default_factory=IdentifyBudget
    )


class Cue(BaseModel):
    family: CueFamily
    name: str


class Candidate(BaseModel):
    candidate_id: str

    cue: Cue

    description: str
    observation: str
    hypothesis: str

    evidence_sources: list[str] = Field(
        default_factory=list
    )

    priority: CandidatePriority
    status: CandidateStatus = "unverified"


class DatasetInspectionSummary(BaseModel):
    dataset: str
    samples_inspected: int = Field(ge=0)
    metadata_used: bool


class CandidateReport(BaseModel):
    protocol_version: str = "1.0"
    run_id: str

    source: Literal["identify"] = "identify"
    message_type: Literal[
        "candidate_report"
    ] = "candidate_report"

    dataset_summary: DatasetInspectionSummary

    candidates: list[Candidate]

    limitations: list[str] = Field(
        default_factory=list
    )