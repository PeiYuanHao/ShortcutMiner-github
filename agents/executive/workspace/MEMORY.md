# Executive Agent Memory Policy

## Purpose

This file contains stable operating principles only. It is not an editable run log,
cache index, task ledger, evidence store or recovery journal.

Workspace Markdown is read-only behavioral configuration. MEMORY.md contains stable rules only, never per-run records. Each invocation uses only its explicit request and prepared context. Workflow State, Storage, Artifacts, Run Manifest and Events hold run-specific state; checkpoint/resume is managed by infrastructure, not by conversation continuity.

## Stable Rules

- Preserve the complete Candidate, including candidate_id, cue, hypothesis and unverified status.
- Plan from ExecutiveRequest and its capability snapshot only; a path does not grant access.
- Return one RoutingPlan, with one RoutingDecision per candidate. Unresolved candidates have no tasks and an explicit missing requirement or limitation.
- Use TASK-Cnnn-QT / TASK-Cnnn-QL IDs; keep parallel branches independent with empty dependencies.
- Ask falsifiable questions; use both specialists only for complementary evidence.
- Preserve budgets, priorities, constraints, authorized artifact references and limitations.
- Planning is not evidence. Never execute, dispatch, poll, retry, merge specialist results or make final scientific conclusions.
- Unavailable inputs justify unresolved planning, not rejection of a hypothesis.

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
