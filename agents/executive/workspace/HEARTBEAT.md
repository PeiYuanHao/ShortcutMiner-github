# Executive Agent — HEARTBEAT.md

## 1. Purpose

The Executive Agent heartbeat defines the periodic checks performed while an audit run is active.

The heartbeat is not used to discover new shortcut candidates.

The current Executive planning executor does not run heartbeats or receive downstream status. This file defines future coordination behavior for the deterministic execution skeleton. During the current one-shot invocation, return a `RoutingPlan` only and never fabricate any state described here.

It is used to verify that:

* received candidates have been routed;
* dispatched tasks are still traceable;
* blocked tasks are visible;
* completed downstream results are not lost;
* the Executive Agent does not enter uncontrolled retry or planning loops.

---

# 2. Heartbeat Principle

Each heartbeat should answer:

```text
What is currently pending?
What is blocked?
What has completed?
What requires an Executive decision?
```

The Executive Agent should perform only lightweight coordination checks during heartbeat execution.

Do not perform full quantitative or qualitative analysis inside a heartbeat.

---

# 3. Heartbeat Checklist

On each heartbeat:

1. Confirm the active `run_id`.
2. Inspect all candidates received from Identify.
3. Confirm that every candidate has a routing state.
4. Inspect dispatched task states.
5. Detect completed downstream tasks.
6. Detect failed or blocked tasks.
7. Check whether any candidate has missing downstream evidence.
8. Determine whether a retry, reroute, or unresolved state is justified.
9. Preserve all candidate and task identifiers.
10. Update the execution state without changing scientific conclusions.

---

# 4. Candidate State Check

Every candidate should remain in one of the following coordination states:

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

Example:

```json
{
  "candidate_id": "SC-001",
  "route": "both",
  "status": "partially_completed",
  "tasks": [
    {
      "task_id": "TASK-SC-001-QT",
      "status": "completed"
    },
    {
      "task_id": "TASK-SC-001-QL",
      "status": "running"
    }
  ]
}
```

Do not mark the candidate `completed` until all required tasks for its selected route are finished or explicitly resolved.

---

# 5. Task State Check

Each dispatched task should use a clear execution state.

Recommended states:

```text
planned
queued
running
completed
failed
blocked
cancelled
```

The Executive Agent should not invent task completion.

A task is `completed` only when the responsible downstream agent or execution layer reports successful completion.

---

# 6. Quantitative Task Check

For each Quantitative task, verify that the result can be linked to:

```text
run_id
candidate_id
task_id
```

If the task completed, confirm that expected outputs or artifact references are present.

If required outputs are missing, do not silently mark the task successful.

Example:

```json
{
  "task_id": "TASK-SC-001-QT",
  "status": "blocked",
  "reason": "Quantitative execution completed but required metric artifacts were not returned."
}
```

---

# 7. Qualitative Task Check

For each Qualitative task, verify that:

* the expected inspection was performed;
* observations are associated with the correct candidate;
* representative evidence or artifact references are available when required;
* the result is not detached from its original hypothesis.

Do not reinterpret qualitative observations during the heartbeat.

The heartbeat verifies execution state, not final scientific meaning.

---

# 8. Dual-Route Candidates

For candidates routed to `both`, Quantitative and Qualitative tasks should normally remain independent.

Example:

```text
SC-003
├── TASK-SC-003-QT
└── TASK-SC-003-QL
```

Possible state:

```json
{
  "candidate_id": "SC-003",
  "route": "both",
  "status": "partially_completed",
  "quantitative_status": "completed",
  "qualitative_status": "running"
}
```

Do not block one independent task merely because the other has not completed.

---

# 9. Blocked Task Handling

A task may become blocked because of:

* missing metadata;
* unavailable dataset artifacts;
* unavailable model access;
* unsupported tool requirements;
* malformed input;
* execution failure;
* resource constraints;
* downstream agent failure.

When a blocked state is detected, record:

```text
candidate_id
task_id
failure type
reason
missing requirement
retry eligibility
```

Example:

```json
{
  "candidate_id": "SC-004",
  "task_id": "TASK-SC-004-QT",
  "status": "blocked",
  "failure_type": "missing_input",
  "reason": "Required hospital metadata are unavailable.",
  "missing_requirements": [
    "hospital_id"
  ],
  "retry_eligible": false
}
```

---

# 10. Retry Policy

Do not retry automatically without justification.

A retry may be appropriate when:

* execution failed because of a transient error;
* required artifacts became available;
* a malformed task can be corrected without changing the original hypothesis;
* the execution layer explicitly supports safe retry.

A retry is normally not appropriate when:

* required data do not exist;
* the requested analysis is fundamentally unsupported;
* the budget is exhausted;
* the candidate itself is too ambiguous to test.

Avoid repeated retries of the same failure.

---

# 11. Rerouting Policy

The heartbeat may identify that the original route cannot be executed.

Example:

```text
Original route:
quantitative

Problem:
Required structured metadata do not exist.

Possible alternative:
qualitative
```

Rerouting is allowed only when the alternative route can test the same original candidate without silently changing its meaning.

Record the reason for rerouting.

Example:

```json
{
  "candidate_id": "SC-006",
  "previous_route": "quantitative",
  "new_route": "qualitative",
  "reason": "Structured background metadata are unavailable, but the suspected cue can still be inspected visually."
}
```

Do not reroute merely to force a task to succeed.

---

# 12. Unresolved Candidates

Mark a candidate `unresolved` when no valid executable route remains.

Example reasons:

```text
missing required data
unsupported analysis
insufficient candidate definition
budget restriction
tool restriction
unavailable model access
```

Unresolved candidates must remain visible to Summarize.

Never silently drop them.

---

# 13. Completion Detection

A candidate is ready for downstream reporting when:

### Quantitative route

```text
required quantitative task completed
```

### Qualitative route

```text
required qualitative task completed
```

### Both route

```text
required quantitative task completed
AND
required qualitative task completed
```

### Unresolved route

```text
unresolved reason recorded
```

The Executive Agent should not determine the final shortcut-risk conclusion during heartbeat execution.

---

# 14. Summarize Readiness

The heartbeat should check whether the audit has enough execution information to notify or hand off to Summarize.

A candidate is ready when the following information is available:

```text
candidate_id
shortcut_type
route
task_ids
execution states
result references
artifact references
failure information, if any
```

Example:

```json
{
  "candidate_id": "SC-001",
  "route": "both",
  "status": "completed",
  "task_ids": [
    "TASK-SC-001-QT",
    "TASK-SC-001-QL"
  ],
  "result_refs": [
    "RESULT-SC-001-QT",
    "RESULT-SC-001-QL"
  ]
}
```

---

# 15. Run-Level Completion

An Executive run is coordination-complete when every received candidate is in one of:

```text
completed
unresolved
cancelled
```

There must be no candidate left silently in:

```text
received
planned
queued
running
partially_completed
```

unless the audit itself is still actively executing.

---

# 16. Heartbeat Output

When the heartbeat produces a state update, prefer a compact structured representation.

Example:

```json
{
  "run_id": "<run_id>",
  "agent": "executive",
  "heartbeat_status": "active",
  "candidate_summary": {
    "total": 5,
    "completed": 2,
    "running": 2,
    "blocked": 1,
    "unresolved": 0
  },
  "pending_actions": [
    {
      "candidate_id": "SC-003",
      "action": "wait_for_qualitative_result"
    },
    {
      "candidate_id": "SC-004",
      "action": "resolve_blocked_task"
    }
  ]
}
```

Do not generate large natural-language reports for routine heartbeat checks.

---

# 17. No-Op Heartbeat

If no coordination action is required, return a minimal no-op state.

Example:

```json
{
  "run_id": "<run_id>",
  "agent": "executive",
  "heartbeat_status": "ok",
  "action_required": false
}
```

Do not create unnecessary tasks simply because a heartbeat occurred.

---

# 18. Prohibited Heartbeat Behavior

During heartbeat execution, the Executive Agent must NOT:

* discover new shortcut candidates;
* modify candidate meaning;
* silently change shortcut types;
* fabricate downstream results;
* perform full quantitative experiments;
* perform full qualitative inspection;
* create duplicate tasks;
* repeatedly retry permanent failures;
* restart completed tasks without reason;
* mix different `run_id` contexts;
* declare scientific conclusions unsupported by downstream evidence.

---

# 19. Heartbeat Decision Order

Use the following order:

```text
Confirm run
    ↓
Check candidate states
    ↓
Check task states
    ↓
Collect completed results
    ↓
Detect failures or blocks
    ↓
Decide retry / reroute / unresolved
    ↓
Check Summarize readiness
    ↓
Update coordination state
```

---

# 20. Final Rule

The Executive heartbeat exists to maintain:

```text
progress
traceability
failure visibility
execution consistency
```

It must not become an additional reasoning loop.

When there is nothing meaningful to coordinate, do nothing.

In the current unconnected planning stage, heartbeat execution itself is unavailable. Do not emit a heartbeat object in place of the required `RoutingPlan`.
