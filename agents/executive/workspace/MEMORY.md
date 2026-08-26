# Executive Agent — MEMORY.md

## Purpose

The Executive Agent memory stores only the minimum coordination state required to plan, route, and track shortcut-validation tasks within the current audit run.

Memory exists to preserve:

* routing continuity;
* task traceability;
* execution progress;
* unresolved dependencies;
* references to downstream results.

Memory must not become an uncontrolled store of scientific conclusions.

The current one-shot Executive planning executor does not maintain mutable task lifecycle memory. Its durable, authoritative output is the validated `RoutingPlan`; downstream status and retry memory become applicable only after the deterministic execution skeleton is connected.

---

# 1. Memory Scope

Executive memory is primarily **run-scoped**.

All stored state must be associated with the current:

```text
run_id
```

Do not mix information from different audit runs.

Do not reuse candidate states, task states, or artifacts from another dataset unless explicitly provided through the current runtime context.

---

# 2. What Should Be Remembered

The Executive Agent should remember the following information for each active run.

## Run State

```text
run_id
dataset reference
audit options
available constraints
overall execution status
```

## Candidate State

For each candidate:

```text
candidate_id
shortcut_type
description
priority
route
routing reason
status
```

## Task State

For each generated task:

```text
task_id
candidate_id
target agent
objective
requested checks
priority
dependencies
execution status
result references
artifact references
```

## Failure State

When relevant:

```text
failure type
failure reason
missing requirements
retry eligibility
rerouting history
```

---

# 3. Suggested Memory Structure

A conceptual memory representation may resemble:

```json
{
  "run_id": "<run_id>",
  "status": "active",
  "candidates": {
    "SC-001": {
      "shortcut_type": "hospital_source",
      "priority": "high",
      "route": "quantitative",
      "routing_reason": "Hospital metadata can be directly measured against labels.",
      "status": "running",
      "tasks": [
        "TASK-SC-001-QT"
      ]
    }
  },
  "tasks": {
    "TASK-SC-001-QT": {
      "candidate_id": "SC-001",
      "target": "quantitative",
      "status": "running",
      "result_refs": [],
      "artifact_refs": []
    }
  }
}
```

This structure is illustrative.

The canonical runtime representation is defined by ShortcutMiner infrastructure and protocol code.

---

# 4. Candidate Memory

For every candidate received from Identify, preserve its original identity.

At minimum, retain:

```text
candidate_id
shortcut_type
original description
priority
```

The Executive Agent may additionally store:

```text
route
routing reason
task IDs
current status
```

Do not overwrite the original candidate description with an Executive reinterpretation.

If additional interpretation is required, store it separately as a routing note.

Example:

```json
{
  "candidate_id": "SC-003",
  "shortcut_type": "background_context",
  "routing_note": "The candidate may require both measurable and semantic evidence."
}
```

---

# 5. Routing Memory

Once a routing decision is made, remember:

```text
candidate_id
selected route
reason
decision timestamp or sequence position, if available
```

Valid routes are:

```text
quantitative
qualitative
both
unresolved
```

If routing later changes, preserve the history rather than silently replacing the original decision.

Example:

```json
{
  "candidate_id": "SC-006",
  "routing_history": [
    {
      "route": "quantitative",
      "reason": "Initial metadata-based validation plan."
    },
    {
      "route": "qualitative",
      "reason": "Required metadata were unavailable."
    }
  ]
}
```

---

# 6. Task Memory

Every task created by the Executive Agent must remain linked to its candidate.

Maintain:

```text
task_id
candidate_id
target agent
objective
requested outputs
status
```

Recommended relationship:

```text
run_id
  ↓
candidate_id
  ↓
task_id
```

Never remember a task without its `candidate_id`.

Never use phrases such as:

```text
the previous task
that analysis
the second experiment
```

as the only task reference.

Always preserve explicit identifiers.

---

# 7. Execution Status Memory

Recommended task states:

```text
planned
queued
running
completed
failed
blocked
cancelled
```

Recommended candidate states:

```text
received
planned
dispatched
running
partially_completed
completed
blocked
unresolved
```

Memory should reflect the most recently confirmed state.

Do not infer completion without explicit downstream or execution-layer evidence.

---

# 8. Result Memory

The Executive Agent should remember **references to results**, not unnecessarily duplicate full results.

Prefer:

```json
{
  "task_id": "TASK-SC-001-QT",
  "status": "completed",
  "result_refs": [
    "RESULT-SC-001-QT"
  ],
  "artifact_refs": [
    "ART-SC-001-001"
  ]
}
```

Avoid storing entire:

* tables;
* plots;
* logs;
* large JSON outputs;
* image sets;
* model predictions.

These belong in artifact storage.

Executive memory should preserve pointers to them.

---

# 9. Artifact Memory

Remember artifact identifiers only when they are needed for later coordination or Summarize handoff.

Useful fields include:

```text
artifact_id
candidate_id
task_id
artifact_type
artifact reference or path
producer agent
```

Do not invent artifact references.

Do not assume an artifact exists simply because a task requested it.

Only store confirmed artifacts.

---

# 10. Failure Memory

Failures must remain visible.

For failed or blocked tasks, remember:

```text
candidate_id
task_id
failure_type
reason
retry_eligible
missing_requirements
```

Example:

```json
{
  "candidate_id": "SC-004",
  "task_id": "TASK-SC-004-QT",
  "status": "blocked",
  "failure_type": "missing_input",
  "reason": "hospital_id metadata is unavailable",
  "retry_eligible": false
}
```

Do not erase failure history after rerouting.

Failure information may be important for the final audit report.

---

# 11. Retry Memory

When a task is retried, preserve retry state.

Example:

```json
{
  "task_id": "TASK-SC-002-QT",
  "attempt": 2,
  "previous_failures": [
    {
      "attempt": 1,
      "failure_type": "execution_error"
    }
  ]
}
```

Do not allow memory loss to cause repeated identical retries.

Before retrying, check whether the same failure has already occurred.

---

# 12. Dependency Memory

If tasks have genuine dependencies, preserve them explicitly.

Example:

```json
{
  "task_id": "TASK-SC-007-QL",
  "dependencies": [
    "TASK-SC-007-QT"
  ]
}
```

Independent Quantitative and Qualitative tasks should normally have:

```json
{
  "dependencies": []
}
```

Do not create dependencies merely because tasks belong to the same candidate.

---

# 13. Summarize Handoff Memory

Before handoff to Summarize, the Executive Agent should be able to reconstruct:

```text
candidate_id
shortcut_type
route
routing reason
task IDs
task statuses
result references
artifact references
failure information
```

This is the minimum coordination context needed for downstream synthesis.

Executive memory should not contain a pre-written final scientific conclusion unless such a conclusion was explicitly returned by a downstream agent as evidence.

---

# 14. What Must Not Be Stored

Do not store unsupported beliefs such as:

```text
SC-001 is definitely a shortcut.
The dataset is biased.
The model relies on hospital identity.
```

unless these statements are explicitly represented as downstream evidence with provenance.

Do not convert speculation into persistent memory.

---

# 15. No Hidden Scientific State

The Executive Agent must not maintain hidden conclusions that are absent from structured execution records.

If a decision affects downstream behavior, it should be represented explicitly through:

```text
routing decision
task definition
status update
result reference
failure record
```

Important state must remain inspectable.

---

# 16. Cross-Run Isolation

Never carry these fields across unrelated runs:

```text
candidate IDs
task IDs
route decisions
artifact references
execution results
failure states
dataset-specific observations
```

A new `run_id` represents a new audit context unless the runtime explicitly declares continuation.

---

# 17. Long-Term Memory Policy

The Executive Agent should not normally maintain long-term scientific memory across datasets.

Do not permanently learn rules such as:

```text
all medical datasets should use both routes
hospital metadata always implies shortcut risk
background cues should always be high priority
```

Routing must depend on the evidence and constraints of the current run.

Reusable system rules belong in:

```text
AGENTS.md
IDENTITY.md
SOUL.md
TOOLS.md
```

not in runtime memory.

---

# 18. Memory Update Principle

Update memory only when new structured information becomes available.

Examples:

```text
candidate received
routing decision made
task dispatched
task status changed
result returned
artifact confirmed
failure occurred
reroute performed
candidate resolved
```

Do not rewrite memory repeatedly when nothing has changed.

---

# 19. Memory Consistency

Before updating state, verify that identifiers match.

For example:

```text
result.run_id == current run_id
result.candidate_id == expected candidate_id
result.task_id == dispatched task_id
```

If identifiers conflict, do not merge the result into memory.

Record the inconsistency as an execution problem.

---

# 20. Recovery Behavior

If the Executive Agent restarts or loses conversational context, reconstruct coordination state from structured memory and execution records.

Recovery should rely on:

```text
run_id
candidate records
task records
result references
artifact references
status records
```

Do not rely on natural-language conversation history when structured state is available.

---

# 21. Memory Priority

When memory capacity is constrained, retain information in this order:

```text
1. run_id
2. candidate_id and original candidate identity
3. routing decision
4. task_id and target agent
5. execution status
6. failure information
7. result references
8. artifact references
9. routing notes
```

Large raw outputs should always be stored externally as artifacts rather than consuming Executive memory.

---

# 22. Final Rule

Executive memory exists to answer:

```text
What candidate are we validating?
How was it routed?
What task was created?
Who is responsible for it?
What is its current state?
Where is the evidence?
What failed or remains unresolved?
```

If a piece of information does not help answer one of these questions, it probably does not belong in Executive memory.

Keep memory:

> minimal, structured, run-scoped, traceable, and evidence-preserving.
