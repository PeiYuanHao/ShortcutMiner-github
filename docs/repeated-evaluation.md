# Repeated Blind Evaluation

Repeated runs measure observed stability because `temperature: 0` does not make a remote model deterministic. `RepeatedBlindAuditRunner` executes sequentially by default and assigns every repeat an independent run ID, repeat index, runtime sampling seed, and optional model seed under one frozen evaluation snapshot.

The persisted matrix reports raw quantities rather than an invented overall consistency score:

- hidden-truth discovery count, completed-run denominator, and frequency;
- exact, partial, incorrect, and unknown localization counts and frequencies;
- supported, partially supported, inconclusive, contradicted, and insufficient-evidence final-assessment distributions;
- clean-control candidate, partial-final, and strong-final false positives per run, plus the fraction with any strong final false positive;
- run success and separate provider, schema, budget, Identify, Executive, Quantitative, Qualitative, and Summarize failure counts.

Each outcome can reference its `RunModelTelemetry` and result artifact. A failed repeat is retained. With `continue_on_run_failure: true`, later repeats still run; runtime failures are not recoded as scientific findings.

The matrix is checkpointed after every repeat under `evaluation/repeated/`. Resume validates the full strict configuration and skips every already recorded repeat, so completed model calls, semantic calls, experiments, and inspections are not reproduced. The runner is intentionally synchronous and bounded in this phase; it is infrastructure for later evaluation, not a formal benchmark result.
