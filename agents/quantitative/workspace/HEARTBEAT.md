# Quantitative Agent Invocation Checklist

## Purpose

HEARTBEAT.md is a fixed checklist for each invocation, not a background loop,
idle service, progress emitter, polling mechanism or independent retry controller.
Return exactly one QuantitativeResult; there is no separate heartbeat schema.

## 1. Intake and Scope

- Read request as QuantitativeRequest; preserve run_id, candidate_id, task_id and hypothesis.
- Check request.in_reply_to against task.message_id and the quantitative target.
- Read requested_checks, expected_outputs, inputs and limitations without enlarging scope.
- Quantitative and Qualitative are parallel branches; do not wait for sibling evidence.
- No active valid request means no autonomous work. Invalid input is rejected by the executor.

## 2. Evidence Availability

A path, model name or requested metric is not evidence that a computation occurred.
Current AgentClient has no bound execution, file-reading or artifact-writing tools.
Use only explicitly supplied, traceable measurements. If required measurements are
unavailable, return blocked/Q0/not_tested with empty metrics and explicit limitations.
Do not perform mental or invented experiments to fill a missing tool result.

## 3. Hypothesis and Method Check

Before accepting numerical evidence, verify:

- what is being measured;
- which variables, groups and splits are compared;
- which metric or intervention addresses the hypothesis;
- whether the evidence is descriptive, associative, model-level or interventional;
- whether assumptions and comparison conditions are valid.

## 4. Reproducibility Check

Preserve available dataset/split identity, sample counts, method, seed, checkpoint,
preprocessing and configuration provenance in the result's supported fields.
Do not invent absent provenance. A cached result must not be described as fresh.

## 5. Numerical Validity Check

Reject fabricated, undefined or invalid measurements. Check:

- NaN and infinity;
- empty groups or zero-sized splits;
- impossible probabilities or metric ranges;
- unexpected missing predictions and failed trials;
- sample ID alignment and actual denominators.

Accuracy, AUROC and probability values must lie in their valid ranges.
An undefined single-class AUROC must not be replaced with zero.

## 6. Sample and Repetition Check

Preserve small subgroup sizes, missing classes, invalid contingency tables and
insufficient sample limitations. For supplied repeated-trial results, verify which
trials actually succeeded; do not imply complete replication from a partial set.

## 7. Evidence and Failure Check

Use evidence_level Q0, Q1, Q2, Q3 or Q4, and evidence_direction supports, mixed,
contradicts, inconclusive or not_tested. Level describes evidence type, not certainty.

Only valid measurements support status completed (Q1–Q4). Unavailable evidence
requires blocked/Q0; actual failed execution requires failed/Q0. Both require
nonempty limitations. Scientific contradiction is not execution failure.
Do not retry or recommend repeated tests merely to obtain a positive result.

## 8. Output and Persistence Check

- Return the exact QuantitativeResult envelope/payload in AGENTS.md.
- Set source quantitative, target summarize, message_type quantitative_analysis_result.
- Set in_reply_to to request.message_id and use a new message_id.
- Preserve request limitations verbatim.
- Use a flat metrics mapping and string lists for limitations/recommended_followup.
- Reference only artifacts supplied in task.inputs.artifact_references.
- Do not claim file creation; the executor validates and persists results/events.

## 9. Return

Finish after the single structured response. Workspace documents remain read-only.
Checkpoint/resume and invocation failures belong to infrastructure. Never fabricate
success to maintain workflow progress.
