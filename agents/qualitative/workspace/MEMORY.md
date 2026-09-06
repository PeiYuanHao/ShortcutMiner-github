# Qualitative Agent Memory Policy

## Purpose

This file contains stable operating principles only. It is not an editable run log,
cache index, task ledger, evidence store or recovery journal.

Workspace Markdown is read-only behavioral configuration. MEMORY.md contains stable rules only, never per-run records. Each invocation uses only its explicit request and prepared context. Workflow State, Storage, Artifacts, Run Manifest and Events hold run-specific state; checkpoint/resume is managed by infrastructure, not by conversation continuity.

## Stable Rules

- Preserve task_id, candidate_id, candidate.cue.name and the assigned hypothesis in QualitativeResult.
- Use only evidence_context and its selected_samples; do not resample or replace records.
- A metadata row or available path is not an inspected image. Visual claims require actual visible evidence.
- Separate observations, interpretations, metadata evidence and counterevidence.
- Use sequential QO-Cnnn-nnn observation IDs and concrete sample_refs in results, never as stored records here.
- Preserve the supplied seed and strategy; report actual inspected sample IDs and corresponding group counts.
- Use supported, partially_supported, inconclusive, contradicted or insufficient_evidence.
- A few examples do not establish dataset-wide prevalence or model reliance.
- Missing tools, samples and failed inspection require failures/limitations, not fabricated observations or negative scientific conclusions.
- Cross-branch synthesis belongs to Summarize; do not wait for Quantitative.
- Never transfer assumptions about a dataset, candidate or acquisition site from another invocation.

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
