<!-- TODO: add content. -->
# MEMORY.md

## Purpose

This file defines what the **Qualitative Agent** should remember during a ShortcutMiner run.

The memory exists to support consistent, traceable qualitative analysis across multiple candidate inspections without repeatedly reconstructing the same context.

Memory must support the task.

Memory must not silently become a source of new evidence.

---

## 1. Memory Principle

The Qualitative Agent should remember information that helps maintain continuity across the current audit.

Useful memory includes:

```text
active run context
candidate definitions
sample groups
artifact references
inspection history
observed patterns
counterexamples
limitations
cross-agent context
completed qualitative results
```

Memory should never override direct evidence.

If remembered information conflicts with newly inspected evidence, prefer the new evidence and record the discrepancy.

---

## 2. Memory Scope

Memory should primarily be scoped by:

```text
run_id
candidate_id
dataset_id
```

Do not mix evidence from unrelated runs or datasets.

A pattern observed in one dataset must not automatically be transferred to another dataset.

A result from a previous candidate must not be treated as evidence for a new candidate unless the relationship is explicitly established.

---

## 3. Run-Level Memory

For each active audit run, remember:

```yaml
run:
  run_id:
  dataset_id:
  dataset_name:
  task_scope:
  active_candidates:
  completed_candidates:
  available_artifacts:
  known_constraints:
```

This information helps maintain consistency across multiple qualitative tasks.

---

## 4. Candidate Memory

For every candidate assigned to the Qualitative Agent, maintain a structured record.

Recommended structure:

```yaml
candidate:
  candidate_id:
  candidate_name:
  hypothesis:
  suspected_cue:
  target_label:
  comparison_groups:
  requested_checks:
  status:
```

Possible status values:

```text
pending
active
blocked
completed
```

Do not modify the original upstream hypothesis in memory.

If your interpretation differs from the original hypothesis, store it separately.

---

## 5. Evidence Memory

Remember only evidence that was actually inspected or provided by a trusted upstream artifact.

Recommended structure:

```yaml
evidence:
  sample_refs:
    - ...

  artifact_refs:
    - ...

  metadata_refs:
    - ...

  observations:
    - observation_id:
      candidate_id:
      sample_refs:
      observation:

  counterevidence:
    - counterevidence_id:
      candidate_id:
      sample_refs:
      description:
```

Do not store imagined or inferred samples as evidence.

---

## 6. Observation Memory

Each meaningful qualitative observation should have a stable local identifier.

Example:

```yaml
observation_id: QO-001
candidate_id: C-003
sample_refs:
  - S102
  - S145
observation: >
  Both inspected samples contain a dark border near the image boundary.
```

Stable observation identifiers make it easier for later reasoning and downstream agents to reference the same evidence.

---

## 7. Interpretation Memory

Interpretations must be stored separately from observations.

Example:

```yaml
interpretation_id: QI-001
candidate_id: C-003
supports:
  - QO-001
interpretation: >
  The dark border may reflect an acquisition or preprocessing difference.
confidence: medium
```

Never store an interpretation as if it were a direct fact.

---

## 8. Counterevidence Memory

Counterevidence must remain first-class memory.

Do not allow supporting evidence to overwrite contradictory evidence.

Example:

```yaml
counterevidence_id: QC-002
candidate_id: C-003
sample_refs:
  - S214
description: >
  A sample from the comparison group also contains the same dark border.
```

Counterexamples should remain available when the final qualitative status is assigned.

---

## 9. Sample Selection Memory

Remember which samples have already been inspected.

Recommended structure:

```yaml
sample_history:
  candidate_id:
  inspected:
    - sample_id:
      group:
      reason_selected:
      inspection_status:
```

This prevents unnecessary repeated inspection and helps determine whether group coverage is adequate.

Possible `reason_selected` values include:

```text
representative
random
high_score
low_score
counterexample
shortcut_aligned
shortcut_conflicting
failure_case
metadata_match
```

---

## 10. Group Coverage Memory

For comparative analysis, remember current sample coverage.

Example:

```yaml
coverage:
  candidate_id: C-003

  groups:
    label_A_source_X:
      inspected: 8

    label_A_source_Y:
      inspected: 8

    label_B_source_X:
      inspected: 8

    label_B_source_Y:
      inspected: 6
```

This helps identify underrepresented groups before a conclusion is finalized.

---

## 11. Metadata Memory

Metadata should be remembered with provenance and interpretation boundaries.

Example:

```yaml
metadata:
  field: hospital_id
  documented_meaning: acquisition hospital
  source: metadata_schema
  values:
    - H1
    - H2
```

For opaque identifiers:

```yaml
metadata:
  field: site_id
  documented_meaning: unknown
  values:
    - site_1
    - site_2
```

Do not invent a semantic meaning for undocumented fields.

---

## 12. Artifact Memory

Remember upstream and generated artifact references.

Examples:

```text
candidate files
sample manifests
metadata tables
quantitative plots
inspection outputs
visualization files
experiment summaries
```

Recommended structure:

```yaml
artifacts:
  - artifact_id:
    type:
    source_agent:
    path_or_ref:
    candidate_id:
    description:
```

Use artifact references instead of copying large artifact contents into memory when possible.

---

## 13. Cross-Agent Memory

The Qualitative Agent may remember relevant upstream results.

For example:

```yaml
upstream_context:
  candidate_id: C-003

  identify:
    candidate_description:
    evidence_refs:

  executive:
    routing_reason:
    requested_checks:

  quantitative:
    status:
    key_result:
    artifact_refs:
```

This information provides context.

Do not rewrite upstream results to match your qualitative conclusion.

---

## 14. Quantitative Result Memory

When Quantitative results are available, remember only what is necessary for qualitative comparison.

For example:

```yaml
quantitative_context:
  candidate_id: C-003
  association_detected: true
  reported_strength: strong
  relevant_artifact:
```

Do not reproduce large statistical outputs in qualitative memory unless necessary.

The Quantitative Agent remains the authoritative source for numerical results.

---

## 15. Qualitative Result Memory

After completing a candidate, store a compact final result.

Example:

```yaml
result:
  candidate_id: C-003
  evidence_status: partially_supported

  evidence_summary: >
    A dark border is frequently visible in the target subgroup,
    but several comparison samples also contain the same pattern.

  key_observations:
    - QO-001
    - QO-004

  key_counterevidence:
    - QC-002

  limitations:
    - limited source-Y samples

  result_artifact:
```

This allows downstream or later tasks to reference the completed analysis without re-running it unnecessarily.

---

## 16. Failure Memory

Operational failures should be remembered during the run.

Example:

```yaml
failure:
  candidate_id: C-004
  type: missing_metadata
  artifact_ref:
  description:
  retry_attempted: true
  resolved: false
```

This prevents repeated attempts to access known-unavailable resources.

---

## 17. Blocked-State Memory

For blocked candidates, remember:

```yaml
blocked:
  candidate_id:
  reason:
  completed_work:
  missing_requirement:
  possible_followup:
```

If the missing requirement later becomes available, analysis may resume from the recorded state.

Do not restart the entire task unless necessary.

---

## 18. Completed-State Memory

Once a candidate is completed, mark it clearly.

```yaml
candidate_id: C-003
status: completed
```

A completed candidate should not be automatically re-analyzed.

Re-analysis is appropriate only when:

```text
new evidence becomes available
the upstream hypothesis changes
the Executive Agent explicitly requests re-analysis
a previous analysis is discovered to be invalid
```

---

## 19. Memory Update Rules

Update memory after meaningful events, including:

```text
new task received
artifact loaded
sample inspected
new observation recorded
counterexample found
comparison completed
candidate blocked
candidate completed
new upstream evidence received
```

Do not update memory after trivial internal reasoning steps.

Memory should remain compact enough to be useful.

---

## 20. Memory Conflict Rules

If memory conflicts with current evidence:

1. do not silently overwrite the conflict;
2. identify the old remembered claim;
3. record the new evidence;
4. prefer direct, verifiable evidence;
5. update the interpretation when necessary.

Example:

```text
Previous memory:
Border appears unique to group A.

New evidence:
Three group-B samples also contain the border.

Updated interpretation:
Border is more common in group A but is not group-exclusive.
```

---

## 21. Memory Compression

When many samples have been inspected, do not store long natural-language descriptions for every sample unless necessary.

Prefer structured summaries.

Instead of:

```text
S001 has a border.
S002 has a border.
S003 has a border.
S004 does not have a border.
...
```

prefer:

```yaml
group_A:
  inspected: 20
  border_present: 16
  border_absent: 4

representative_refs:
  present:
    - S001
    - S008
  absent:
    - S004
```

Retain enough representative references to preserve traceability.

---

## 22. Memory Must Not Become Evidence

A remembered claim is not automatically evidence.

For example:

```text
Memory:
Candidate C-003 previously appeared visually plausible.
```

This cannot be cited as direct support for another candidate.

Use the original evidence references.

Memory exists to point back to evidence, not replace it.

---

## 23. No Cross-Dataset Contamination

Never transfer qualitative assumptions between datasets.

For example:

```text
Camelyon17 may contain hospital-specific acquisition differences.
```

does not imply that another medical dataset contains the same pattern.

Every dataset must be inspected independently.

Prior domain knowledge may guide attention but cannot count as evidence.

---

## 24. Persistent Behavioral Memory

The Qualitative Agent should consistently preserve the following principles across tasks:

```text
inspect before concluding
separate observation from interpretation
preserve counterexamples
do not fabricate evidence
do not infer causality from appearance
maintain sample-level traceability
respect agent boundaries
```

These are stable operating principles rather than candidate-specific evidence.

---

## 25. Memory Cleanup

At the end of an audit run:

Retain:

```text
final candidate results
important artifact references
key observations
key counterexamples
documented limitations
failure records relevant to reproducibility
```

Discard or compress:

```text
temporary inspection notes
duplicate observations
redundant sample descriptions
resolved operational details
intermediate reasoning that is not needed for traceability
```

The retained memory should allow another agent or researcher to understand what was inspected and why the conclusion was reached.

---

## 26. Final Memory Rule

The Qualitative Agent should remember:

```text
what was assigned
what was inspected
what was observed
what contradicted the hypothesis
what remained uncertain
where the evidence came from
what conclusion was returned
```

It should not remember unsupported assumptions as facts.

A useful memory is not the largest possible memory.

A useful memory is the smallest reliable record that preserves continuity, evidence provenance, and reproducibility.
