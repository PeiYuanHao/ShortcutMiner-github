# Quantitative Agent Memory Policy

## Purpose

This file contains stable operating principles only. It is not an editable run log,
cache index, task ledger, evidence store or recovery journal.

Workspace Markdown is read-only behavioral configuration. MEMORY.md contains stable rules only, never per-run records. Each invocation uses only its explicit request and prepared context. Workflow State, Storage, Artifacts, Run Manifest and Events hold run-specific state; checkpoint/resume is managed by infrastructure, not by conversation continuity.

## Stable Rules

- Preserve candidate_id, task_id and the assigned hypothesis in QuantitativeResult.
- Q0 means no valid test; Q1 descriptive measurement; Q2 association testing; Q3 model-behavior testing; Q4 intervention testing. Levels describe methods, not guaranteed positive findings.
- Use the exact evidence_direction values: supports, mixed, contradicts, inconclusive, not_tested.
- Only valid, traceable measurements justify completed. Missing evidence is blocked/Q0; actual failed execution is failed/Q0. Preserve limitations and never replace undefined metrics with zero.
- Keep metric definitions, units, denominators, sample counts, uncertainty and threshold provenance explicit.
- Never infer dataset mappings, split encodings or model preprocessing from a familiar name.
- Reused evidence must be explicitly supplied with matching provenance; do not claim cached results were newly computed.
- Never change seeds, groups, thresholds, samples or metrics to obtain confirmation.
- Current invocation has no execution tools. An analysis plan or artifact path is not numerical evidence.

## State and Provenance Boundary

Do not write current run identifiers, dataset mappings, candidate records, metrics,
observations, artifact paths, failures, progress, temporary conclusions or secrets
into this file. Even verified run-specific information belongs in runtime records,
not persistent Agent instructions.

Preserve relevant provenance and uncertainty in the canonical structured result.
The control layer validates and persists the result and events. The Agent does not
independently read prior runs, mutate workflow state, update this file or restore
itself from stored conversation.

On resume, use the explicit inputs supplied by infrastructure. Missing inputs
cannot be reconstructed from presumed memory.

## Permanent Rule

Remember how to assess evidence, never what a previous audit supposedly found.
