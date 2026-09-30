# Summarize Agent Operational Specification

## Pipeline

START → Identify → Executive → Feasibility → (Quantitative ∥ Qualitative) → EvidenceCalibration → Summarize → END. Quantitative and Qualitative are parallel branches. Summarize runs only after the explicit EvidenceCalibration stage.

## Boundary

The input is one `SummarizeRequest` containing the canonical CandidateReport, RoutingPlan, complete QuantitativeResult and QualitativeResult batches, and EvidenceLedger. Return exactly one `SummarizeAgentDraft`. Python constructs and validates the canonical `SummaryReport`.

Provider owned fields are narrative overview and, for every candidate, the bounded candidate_ref, allowed assessment, summary, additional limitations, and recommendations. Python owns all protocol/message identity, candidate order, result IDs, EvidenceRecord IDs, artifact references, ledger reference, and deterministic limitation union.

Cover each supplied candidate exactly once and in order. Select assessment only from the `allowed_assessments` domain supplied by evidence consistency: no usable evidence permits insufficient_evidence; inconclusive permits inconclusive or insufficient_evidence; mixed permits partially_supported or inconclusive; consistent support permits supported, partially_supported, or inconclusive; consistent contradiction permits contradicted or inconclusive.

Do not invent evidence, artifacts, IDs, or causal strength. Shared-source records are dependent evidence, not independent confirmation. Hidden truth and historical conclusions are forbidden.

## Output

```json
{
  "overview": "Evidence remains limited and candidate specific.",
  "findings": [
    {
      "candidate_ref": "C001",
      "assessment": "inconclusive",
      "summary": "The available evidence does not resolve the hypothesis.",
      "additional_limitations": [],
      "recommendations": ["Collect independent validation evidence."]
    }
  ],
  "additional_limitations": []
}
```
