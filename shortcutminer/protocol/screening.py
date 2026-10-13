"""Early filtering contracts; screening decisions are never shortcut verdicts."""

from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, StringConstraints, model_validator

from shortcutminer.protocol.identify import CandidateId, NonEmptyText, SafeRunId, StrictProtocolModel


UnitInterval = Annotated[FiniteFloat, Field(ge=0, le=1)]
NonnegativeCost = Annotated[FiniteFloat, Field(ge=0)]
Fingerprint = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class CandidateScreeningPolicy(StrictProtocolModel):
    screening_version: Literal["v1"] = "v1"
    min_confidence: UnitInterval = 0.5
    min_samples: int = Field(default=1, ge=1)
    max_promoted: int | None = Field(default=None, ge=0)
    max_candidate_cost: NonnegativeCost | None = None
    max_total_cost: NonnegativeCost | None = None


class CandidateScreeningSignals(StrictProtocolModel):
    """Optional public pre-validation signals, not results of future experiments.

    Costs use relative work units, not prices or runtime predictions. The current
    CandidateReport has no confidence field; missing confidence stays unavailable.
    """

    confidence_by_candidate: dict[CandidateId, UnitInterval] = Field(default_factory=dict)
    cost_by_candidate: dict[CandidateId, NonnegativeCost] = Field(default_factory=dict)


class CandidateScreeningResult(StrictProtocolModel):
    candidate_id: CandidateId
    decision: Literal["promote", "screened_out"]
    confidence: UnitInterval | None = None
    sample_availability: int = Field(ge=0)
    estimated_validation_cost: NonnegativeCost
    reason: list[NonEmptyText] = Field(min_length=1)
    screening_version: Literal["v1"] = "v1"


class CandidateScreeningReport(StrictProtocolModel):
    run_id: SafeRunId
    screening_version: Literal["v1"] = "v1"
    input_hash: Fingerprint
    result_hash: Fingerprint
    candidates_generated: int = Field(ge=0)
    promoted: int = Field(ge=0)
    screened_out: int = Field(ge=0)
    results: list[CandidateScreeningResult]

    @model_validator(mode="after")
    def validate_counts(self):
        ids = [item.candidate_id for item in self.results]
        if len(set(ids)) != len(ids):
            raise ValueError("Screening candidate IDs must be unique")
        if self.candidates_generated != len(self.results):
            raise ValueError("Screening generated count does not match results")
        if self.promoted != sum(item.decision == "promote" for item in self.results):
            raise ValueError("Screening promoted count does not match decisions")
        if self.screened_out != self.candidates_generated - self.promoted:
            raise ValueError("Screening screened-out count does not match decisions")
        return self
