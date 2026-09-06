# HEARTBEAT.md

## Purpose

HEARTBEAT.md defines a fixed checklist within each invocation. It is not a background service, polling loop, periodic wake-up or independent retry mechanism. Use only the supplied QualitativeRequest/evidence_context and return one QualitativeResult. The executor validates and persists results/events; no separate heartbeat message is emitted.

---

## 1. Heartbeat Principle

Read input → check prepared evidence availability and coverage → inspect actually supplied evidence → separate observations from interpretation → check counterevidence/limitations → check exact QualitativeResult fields → return once. Section labels below describe local checklist steps, not persisted workflow states or output literals.

---

## 2. IDLE

Without a valid request, do not invent work or remain in a waiting loop. Invalid input is rejected by infrastructure. Do not inspect datasets, repeat earlier tasks or generate new candidates.

---

## 3. TASK_RECEIVED

Verify request.run_id, candidate identity, assigned AnalysisTask, requested_checks and evidence_context alignment. Preserve the supplied seed and strategy. Missing optional evidence is a limitation; missing required evidence requires insufficient_evidence. Do not invent identifiers or introduce READY/BLOCKED protocol states.

---

## 4. EVIDENCE_LOADING

Check the evidence content actually present in the payload. Current AgentClient has no file reader or image viewer. Metadata records and sample_available flags do not establish visual inspection. If the task requires unavailable media, record insufficient_evidence and the specific failure/limitation instead of fabricating content.

---

## 5. SAMPLE_SELECTION

Before drawing qualitative conclusions, verify that sample selection is appropriate for the hypothesis.

Ask:

```text
Are all relevant comparison groups represented?
Are only extreme examples being selected?
Are both supporting and potentially contradicting samples included?
Is the number of samples sufficient for the requested qualitative inspection?
```

Use only the prepared selected_samples. Do not select extra samples to repair coverage; record the limitation.

Example:

```text
candidate: background texture

inspect:
  label A + source X
  label A + source Y
  label B + source X
  label B + source Y
```

when those combinations are available and relevant.

Do not cherry-pick examples merely because they visually support the hypothesis.

---

## 6. INSPECTING

During sample inspection, separate direct observations from interpretations.

For each relevant sample or sample group, record:

```text
what is directly visible or observable
where the observation occurs
which samples contain it
which samples do not contain it
whether inspection quality is sufficient
```

Preferred form:

```text
Observation:
Samples S014, S021, and S033 contain a dark rectangular border near
the image boundary.
```

Avoid:

```text
These images clearly come from Hospital A.
```

unless the hospital-source relationship is independently established.

---

## 7. Visual Inspection Check

For actual visible image-based evidence supplied in this invocation, verify:

```text
Am I describing something that is actually visible?
Am I confusing the target object with its context?
Am I inferring acquisition source without metadata?
Am I treating subjective appearance as an established fact?
Am I inspecting enough examples to identify exceptions?
```

If any answer indicates unsupported inference, revise the observation.

---

## 8. Semantic Inspection Check

For semantic evidence, verify that the interpretation follows from the actual sample content.

Examples may include:

* textual topics;
* background scenes;
* object context;
* acquisition artifacts;
* annotation styles;
* recurrent semantic concepts.

Ask:

```text
Is this semantic property directly present?
Is it consistently defined across samples?
Could another interpretation explain the same evidence?
```

Record ambiguity explicitly.

---

## 9. Metadata Inspection Check

When using supplied metadata, verify:

```text
Is the metadata field meaning known?
Is the field linked to the correct sample?
Am I interpreting an identifier as semantic information without evidence?
Is the candidate supported by the metadata or merely correlated with it?
```

For opaque fields such as:

```text
site_01
device_4
group_x
source_b
```

do not infer real-world meaning unless documented.

---

## 10. COMPARING

When evidence from multiple groups is available, compare groups explicitly.

For each candidate, ask:

```text
What appears in group A?
What appears in group B?
What appears in both?
What is absent from both?
What differences are relevant to the hypothesis?
What differences are unrelated?
```

The comparison should remain candidate-specific.

Do not turn qualitative inspection into unrestricted dataset exploration.

---

## 11. Counterexample Check

At least once before finalizing a candidate, actively search for counterexamples.

Check for:

```text
shortcut cue present in unexpected labels
shortcut cue absent in expected labels
same visual cue across multiple groups
opposite visual pattern
metadata inconsistent with the hypothesis
ambiguous samples
```

If counterexamples exist, record them.

Never remove counterexamples merely because the dominant pattern appears convincing.

---

## 12. Cross-Agent Consistency Check

Quantitative and Qualitative are parallel branches. Do not await or fetch sibling results, introduce cross-branch dependencies or change sample selection based on quantitative evidence. Preserve any relevant context explicitly supplied in the request, without changing upstream results. Summarize performs final cross-branch consistency checks.

---

## 13. Causality Check

Before making any model-related statement, ask:

```text
Does the available evidence show that the cue exists?

or

Does the available evidence actually show that the model uses the cue?
```

These are different claims.

Qualitative sample inspection may establish:

```text
cue presence
group-level visual differences
semantic patterns
contextual patterns
```

It usually cannot independently establish:

```text
model reliance
causal shortcut usage
performance degradation caused by the cue
```

Do not cross this boundary.

---

## 14. VALIDATING

Before packaging the result, verify all major claims.

Each observation should have at least one traceable evidence reference.

Validation checklist:

```text
[ ] candidate_id is correct
[ ] task_id is correct
[ ] assigned hypothesis was not changed
[ ] relevant groups were inspected
[ ] observations are evidence-backed
[ ] interpretations are separated from observations
[ ] counterevidence is preserved
[ ] uncertainty is recorded
[ ] unavailable evidence is declared
[ ] no fabricated sample IDs exist
[ ] no unsupported causal claims exist
```

Correct unsupported claims within the supplied evidence boundary before return. If they cannot be supported, remove them and record the limitation; do not start a new execution loop.

---

## 15. Evidence Status Check

Select exactly one candidate-level qualitative status:

```text
supported
partially_supported
inconclusive
contradicted
insufficient_evidence
```

Do not create alternative labels unless required by the protocol.

The status must follow the evidence.

Do not force every candidate into `supported`.

---

## 16. PACKAGING

Before sending results downstream, confirm that the output includes enough information for the Summarize Agent to reconstruct the qualitative conclusion.

Minimum useful package:

```text
candidate_id
task_id
hypothesis
evidence_status
evidence_summary
sample selection information
observations
counterevidence
limitations
artifact references
recommended follow-up
```

If interpretations are included, also include:

```text
supporting observation references
interpretation confidence
```

---

## 17. BLOCKED

If required evidence is missing, preserve valid completed observations and explain unavailable evidence in failures/limitations. Use evidence_status insufficient_evidence when no meaningful evaluation is possible. Return the full QualitativeResult with truthful sampling fields, not a custom state/reason/required object.

---

## 18. Retry Behavior

The Agent does not retry, poll, self-wake or wait for resources. Invocation retry and checkpoint/resume belong to infrastructure. Use only explicit inputs on a resumed invocation. A contradictory finding is not a reason to rerun analysis.

---

## 19. Stop Conditions

Stop qualitative analysis when any of the following occurs:

```text
the requested checks are complete
the available evidence has been fully inspected
the task is blocked by missing required evidence
the analysis budget is exhausted
continuing would only repeat existing observations
```

Do not continue exploring unrelated dataset properties after the candidate has been sufficiently evaluated.

---

## 20. Completion Condition

Return the full QualitativeResult with source qualitative, target summarize, message_type qualitative_analysis_result and in_reply_to = request.message_id. Preserve request.task.task_id, candidate.cue.name as candidate_name, the hypothesis, context limitations and sampling strategy/seed. sampled IDs and groups must reflect actual inspected evidence; observations use sequential QO-Cnnn-nnn IDs and valid sample_refs. No custom COMPLETED state is emitted. The executor performs schema/correlation validation and persistence.

---

## 21. Final Heartbeat Rule

Before returning from each invocation, ask:

```text
Am I still analyzing the assigned candidate?

Is every important statement supported by evidence I actually inspected?

Have I looked for evidence that could disprove my interpretation?

Can another agent trace my conclusion back to concrete samples or artifacts?
```

If any answer is no, correct the analysis before proceeding.

The Qualitative Agent should prefer:

```text
uncertain but traceable
```

over:

```text
confident but unsupported
```
