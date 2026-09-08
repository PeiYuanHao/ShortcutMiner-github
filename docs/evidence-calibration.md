# Evidence calibration and uncertainty

ShortcutMiner calibrates evidence after Quantitative and Qualitative complete and
before Summarize. `EvidenceLedgerBuilder` is deterministic infrastructure. It does
not call an LLM, run an experiment, alter upstream evidence, discover candidates,
or make a final shortcut judgment.

## Contracts and persistence

`shortcutminer/protocol/evidence.py` defines `EvidenceRecord`,
`CandidateEvidenceProfile`, and `EvidenceLedger`. Every candidate from
`CandidateReport` receives a ledger entry even when it has no task, all work is
blocked, or every inspection fails. The artifact is JSON-serializable and stored
at:

```text
runs/<run_id>/evidence/ledger.json
```

Its fingerprint covers CandidateReport, RoutingPlan, QuantitativeResultBatch, and
QualitativeResultBatch. Matching inputs reuse the existing artifact. Any change
to those inputs causes a rebuild, preventing stale evidence from crossing a
resume boundary.

## Threshold audit classification

The Phase 11 audit classifies threshold-like values rather than treating every
numeric bound as a scientific decision rule:

- scientific decision thresholds: none are used by the production calibration,
  experiment, workflow, or protocol paths to confirm a shortcut;
- numerical/statistical implementation thresholds: confidence-level validation,
  finite-value checks, category validity, and mathematical requirements;
- resource/safety budgets: `max_rows`, `max_categories`, `max_pixels`, minimum
  estimable group size, bootstrap/permutation/solver limits, and file budgets;
- test-only expectations: fixture tolerances and assertions remain confined to
  tests;
- documentation/examples: illustrative effect sizes or p-values are not runtime
  decisions.

The regression audit parses production syntax to detect conditional mappings from
p-values, effect sizes, Cramer's V, accuracy gaps, deltas, or scores to supported,
confirmed, shortcut, or risk verdicts. Resource limits and descriptor-definition
bounds remain legal because they constrain execution or define a measurement,
not a scientific conclusion.

## Independent dimensions

The ledger deliberately separates:

```text
raw metric != evidence direction
evidence direction != evidence confidence
evidence confidence != shortcut risk
shortcut risk != confirmation probability
```

There is no weighted shortcut score, numeric risk score, shortcut probability,
or universal p-value/effect-size/accuracy-gap cutoff. Directions come only from
validated Agent results. The builder groups and preserves them; it never invents
a direction from a number.

Evidence kinds distinguish association, subgroup, predictive probe,
intervention, ablation, metric, qualitative metadata, visual/spatial observation,
counterevidence, and failure. Design strength is descriptive, observational,
predictive, or interventional. This describes the research design, not whether a
candidate is supported. Q0-Q4 retain the same meaning: not tested/failed,
descriptive, observational, predictive, and intervention-style evidence maturity.
They are not confidence percentages, risk levels, or final verdicts.

Execution states are `usable`, `scientifically_inconclusive`, `failed`, and
`blocked`. Failure or blockage reduces coverage and is never contradiction.
Scientific inconclusiveness is not evidence of absence. Qualitative inspection
quality likewise remains independent from evidence direction.

## Uncertainty

An uncertainty record states whether an interval is available. Available
intervals carry the metric, method, confidence level, lower/upper bounds,
standard error, iteration count, seed, and resampling unit. Unavailable intervals
carry a reason; no interval is fabricated.

The CPU backend uses deterministic percentile bootstrap where the retained data
supports it:

- association effect sizes: sample rows;
- subgroup accuracy, worst-group accuracy, and group gap: prediction rows while
  preserving each row's group identity;
- probe accuracy, balanced accuracy, baseline accuracy, and improvement: held-out
  prediction rows without repeated model fitting;
- intervention effect delta: paired sample rows;
- allowlisted prediction metrics: prediction rows.

Bootstrap iterations are schema-bounded to at most 1,000 (default 200), and the
confidence level defaults to 95% with a valid range of `[0.8, 1.0)`. CI exclusion
of any value never changes evidence direction. Interval width describes
precision, not importance or risk.

## Multiple testing

Benjamini-Hochberg FDR correction is optional and deterministic. Formal p-values
for related tests within one candidate form
`<candidate_id>:related_statistical_tests`; candidates are not pooled into one
global family. Every test retains its raw p-value and records adjusted p-value,
method, family ID, and family size. When no defensible family is available, the
builder performs no implicit correction. Adjusted p-values never trigger a
shortcut verdict.

## Coverage, provenance, and dependence

Coverage records requested, attempted, usable, failed, and inconclusive work plus
available sample, valid, missing, selected, and inspected counts. Provenance links
run, candidate, task, result, dataset/input family, shared-source identifiers, and
validated safe relative artifacts. Missing or mismatched artifacts are recorded
as limitations and are not treated as inspected evidence.

Records with the same `input_family` are explicitly grouped. Association, probe,
and intervention evidence derived from the same field/rows therefore remain
visible as dependent corroboration rather than being described as independent
confirmations. No statistical dependence model is implied.

## Consistency and summary guard

Logical aggregation uses only upstream directions:

- supports only: `consistent_support`;
- contradicts only: `consistent_contradiction`;
- both: `mixed`;
- only scientifically inconclusive work: `inconclusive`;
- failed/blocked/no records: `no_usable_evidence`.

The Summary validator prevents obvious overclaim: support requires supporting
evidence, contradiction requires contradictory evidence, all-failed evidence must
remain insufficient, all-inconclusive evidence must remain unresolved, and mixed
evidence cannot be collapsed into an unqualified binary verdict. It does not use
a formula to choose a final assessment; Summarize still synthesizes the complete
record while preserving both branches, uncertainty, failures, limitations, and
shared-source caveats.
