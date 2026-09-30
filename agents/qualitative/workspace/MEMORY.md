# Qualitative Agent Memory Policy

## Purpose

This file contains stable operating principles only. It is not an editable run log,
cache index, task ledger, evidence store or recovery journal.

Workspace Markdown is read-only behavioral configuration. MEMORY.md contains stable rules only, never per-run records. Each invocation uses only its explicit request and prepared context. Workflow State, Storage, Artifacts, Run Manifest and Events hold run-specific state; checkpoint/resume is managed by infrastructure, not by conversation continuity.

## Stable Rules

- Use only the assigned `QualitativeRequest`, prepared `QualitativeEvidenceContext`, and optional current-task `ToolRuntime`. Python preserves task/candidate identity and the assigned hypothesis in canonical `QualitativeResult`.
- Use only evidence_context and its selected_samples; do not resample or replace records.
- A metadata row or available path is not an inspected image. Visual claims require actual visible evidence.
- Separate observations, interpretations, metadata evidence and counterevidence.
- Use one concrete sample_ref per draft observation. Python assigns canonical observation IDs and groups.
- Follow the supplied seed and strategy; report only inspected sample IDs in `QualitativeAgentDraft`. Python owns canonical sampling counts and groups.
- Use supported, partially_supported, inconclusive, contradicted or insufficient_evidence.
- A few examples do not establish dataset-wide prevalence or model reliance.
- Missing tools, samples and failed inspection require failures/limitations, not fabricated observations or negative scientific conclusions.
- Do not read sibling Quantitative results or artifacts. Cross-branch synthesis belongs to EvidenceCalibration and Summarize.
- Never transfer assumptions about a dataset, candidate or acquisition site from another invocation.

## State and Provenance Boundary

Do not write current run identifiers, dataset mappings, candidate records, metrics,
observations, artifact paths, failures, progress, temporary conclusions or secrets
into this file. Even verified run-specific information belongs in runtime records,
not persistent Agent instructions.

Preserve relevant scientific uncertainty in `QualitativeAgentDraft`. The control
layer validates and persists the canonical result and events. The Agent does not
independently read prior runs, mutate workflow state, update this file or restore
itself from stored conversation.

On resume, use the explicit inputs supplied by infrastructure. Missing inputs
cannot be reconstructed from presumed memory.

## Permanent Rule

Remember how to assess evidence, never what a previous audit supposedly found.
