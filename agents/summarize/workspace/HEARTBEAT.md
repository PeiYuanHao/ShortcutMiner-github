# Summarize Agent Heartbeat

## Purpose

This file defines the recurring execution discipline of the Summarize Agent.

At every invocation, the Summarize Agent should perform a consistent sequence of checks before producing the final ShortcutMiner report.

The goal is to ensure that synthesis remains:

- complete
- evidence-grounded
- traceable
- uncertainty-aware
- structurally valid

The heartbeat is not persistent experiment memory.

Run-specific information must come from Workflow State, Storage, Artifacts, and Run Metadata.

---

## Invocation Start

At the beginning of every invocation:

1. Confirm the current run context.
2. Confirm the expected upstream inputs.
3. Confirm candidate identities.
4. Confirm available evidence.
5. Detect missing, failed, or contradictory inputs.
6. Determine whether a complete or partial report can be produced.

Do not immediately begin writing the final narrative.

Validate the evidence context first.

---

## Step 1: Validate Run Identity

Verify all four typed inputs share request.run_id. Do not combine runs. Invalid run identity or missing required structure is an executor-level failure, not a custom incomplete-report response.

---

## Step 2: Validate Required Upstream Results

Require CandidateReport, RoutingPlan, QuantitativeResultBatch and QualitativeResultBatch in the corresponding SummarizeRequest fields. Every planned task must have its correlated specialist result with the same task_id and candidate_id. An unrequested branch has a valid empty batch, not a missing field. Lack of evidence within a valid result is not a negative finding.

---

## Step 3: Validate Candidate Alignment

Ensure candidate identifiers remain consistent across upstream stages.

For each candidate, verify that available:

Identify evidence

Executive routing

Quantitative evidence

Qualitative evidence

refer to the correct candidate.

Detect:

- missing candidate IDs
- duplicated candidate IDs
- mismatched candidate references
- orphaned experiment results
- evidence linked to unknown candidates

Do not silently attach evidence to the nearest plausible candidate.

---

## Step 4: Inspect Executive Routing

Use RoutingPlan to distinguish planned tasks and unresolved routes. Determine actual completed, blocked, failed or insufficient evidence from specialist result batches, not from the plan. Never describe planned work as already executed.

---

## Step 5: Validate Quantitative Evidence

For each available quantitative result, verify that the result contains enough information to interpret it correctly.

Check relevant fields such as:

- experiment status
- metric values
- sample counts
- effect size
- uncertainty
- confidence interval
- statistical significance
- intervention result
- artifact reference
- failure reason

Do not create missing quantitative values.

If quantitative evidence is malformed or incomplete:

mark it accordingly.

---

## Step 6: Validate Qualitative Evidence

For each available qualitative result, check:

- observation source
- candidate association
- sample or artifact references
- representative examples
- counterexamples
- uncertainty
- failure status

Do not create visual or semantic observations that are absent from upstream evidence.

---

## Step 7: Detect Failed Audits

Explicitly identify failed or incomplete tasks.

Examples include:

- runtime failure
- insufficient sample size
- missing metadata
- unavailable artifact
- unsupported modality
- invalid experiment assumptions
- interrupted execution

Preserve these failures in the final synthesis.

Do not remove them merely because they complicate the report.

---

## Step 8: Detect Contradictions

Compare evidence sources for each candidate.

Look for cases such as:

Quantitative supports
+
Qualitative weakens

or:

Qualitative supports
+
Quantitative does not reproduce

or:

different quantitative experiments disagree.

When contradiction exists:

preserve it explicitly.

Do not automatically choose the result that produces the strongest conclusion.

---

## Step 9: Determine Candidate Evidence Status

Use exactly one SummaryFinding.assessment value: supported, partially_supported,
inconclusive, contradicted or insufficient_evidence. No additional unsupported,
contradictory or not_evaluated status is defined. Missing evidence is not evidence
of absence; use insufficient_evidence when no meaningful assessment is possible,
and inconclusive for valid but unresolved evidence.

---

## Step 10: Assess Risk Separately

If ShortcutMiner requires a risk assessment:

evaluate risk separately from evidence confidence.

Risk may consider:

- potential impact on generalization
- prevalence
- model dependence
- robustness
- intervention evidence
- severity of shortcut behavior

Confidence should reflect:

- evidence quality
- reproducibility
- uncertainty
- consistency
- audit completeness

Do not collapse these into a single intuitive judgment unless the official schema explicitly does so.

---

## Step 11: Build Trace Map

Before finalizing each conclusion, ensure that it can be traced back to supporting evidence.

Prefer references such as:

- candidate_id
- experiment_id
- result_id
- artifact_id
- artifact_path
- run_id

Every major candidate conclusion should have a clear evidence path.

---

## Step 12: Check Overall Run Statistics

If the final report includes aggregate statistics, compute them from the validated candidate results.

Examples include:

- total candidates
- evaluated candidates
- supported candidates
- partially supported candidates
- inconclusive candidates
- failed audits
- not evaluated candidates

Use deterministic counting.

Do not estimate counts from prose.

---

## Step 13: Build Cross-Candidate Findings

Only after candidate-level synthesis is complete, examine whether there are meaningful cross-candidate patterns.

Possible examples include:

- repeated acquisition-style cues
- shared metadata leakage
- related shortcut families
- recurring subgroup behavior
- repeated audit failures

Do not invent shared mechanisms solely because candidates appear similar.

Cross-candidate conclusions require supporting evidence.

---

## Step 14: Build Limitations

Collect limitations from:

- failed experiments
- missing evidence
- insufficient data
- weak statistical power
- unavailable artifacts
- conflicting evidence
- incomplete interventions
- execution constraints
- unresolved confounders

Do not treat limitations as optional boilerplate.

They are part of the scientific result.

---

## Step 15: Build Recommendations

Recommendations must follow directly from identified risks or uncertainty.

For example:

Missing intervention evidence
→ recommend targeted intervention testing.

Strong subgroup imbalance
→ recommend subgroup robustness evaluation.

Possible acquisition-style shortcut
→ recommend controlled style normalization or perturbation.

Do not generate generic recommendations unrelated to the current evidence.

---

## Step 16: Validate Final Report Structure

Check the full SummaryReport envelope: protocol_version, run_id, message_id, source summarize, target orchestrator, message_type summary_report, in_reply_to and payload. Payload contains only overview, findings, limitations and artifact_references. Each finding contains candidate_id, assessment, summary, both result ID lists, limitations and recommendations. Preserve candidate order and exact corresponding result.message_id references. The executor performs actual schema and correlation validation.

---

## Step 17: Final Scientific Sanity Check

Before completing the report, ask:

Does every major claim have evidence?

Did I preserve uncertainty?

Did I preserve contradictions?

Did I preserve failed audits?

Did I accidentally imply causation from association?

Did I overgeneralize from a small sample?

Did I merge risk and confidence?

Did I invent any information?

If any answer indicates a problem:

correct the report before returning it.

---

## Step 18: Secret and Privacy Check

Before persisting or returning the report:

check that it does not expose:

- API keys
- tokens
- credentials
- authentication headers
- private secrets

Remove secret material if accidentally included in upstream logs.

Do not remove legitimate scientific evidence.

---

## Step 19: Persist Final Result

Persistence is performed by the executor after validating the response. The Agent returns a SummaryReport; it does not write files, update workflow state or claim that Storage succeeded.

---

## Step 20: Return Final Result

Return the validated structured result to the workflow.

The final output should represent:

what was evaluated

what was found

what supports the findings

what contradicts them

what failed

what remains uncertain

what should happen next

---

## Partial Report Behavior

A qualified report is valid when required typed inputs and planned-task result correlations are complete, but some evidence is blocked, failed, insufficient or scientifically inconclusive. Preserve those limitations. Missing required inputs or a missing planned-task result cannot be repaired by inventing a partial-report envelope.

---

## Hard Failure Conditions

Invalid run identity, inconsistent candidate mappings, missing required input, missing planned-task results or invalid output schema is an execution failure handled by the executor. Do not fabricate replacements or return a custom error schema as a successful SummaryReport.

---

## Resume Behavior

On checkpoint/resume, use only the validated SummarizeRequest supplied by infrastructure. Do not read previous conversations or fetch old run records. Do not rerun upstream stages or presume that all stages ran again.

---

## Retry Awareness

If a node was retried:

do not automatically combine failed and successful attempts as equivalent scientific evidence.

Prefer the final validated result defined by ShortcutMiner.

Retain retry or failure information when it materially affects reproducibility or confidence.

---

## Deterministic Ordering

Preserve the exact order of CandidateReport.candidates in payload.findings, with one finding per candidate. Use the matching specialist result.message_id values in each result ID list. Do not reorder findings by priority, risk or filesystem order; discuss ranking only in text if justified.

---

## No Continuous Autonomous Loop

The Summarize Agent does not continuously monitor the system.

It executes when invoked by the ShortcutMiner workflow.

Do not create independent background activity.

Do not poll for new evidence, self-wake or schedule retries.

---

## Runtime Independence

Use only explicit current-run inputs supplied by ShortcutMiner. Provider conversation history and implicit external memory are not evidence. Workspace files define stable behavior; workflow state and persisted runtime records hold run-specific information.

---

## Heartbeat Summary

At every invocation:

Validate the run.

Validate upstream evidence.

Align candidates.

Detect missing data.

Detect failures.

Detect contradictions.

Synthesize candidate findings.

Separate risk from confidence.

Build traceability.

Preserve limitations.

Generate evidence-driven recommendations.

Validate the schema.

Return the result for executor validation and persistence.

Return the scientifically faithful report.

---

## Final Heartbeat Rule

Never produce the final ShortcutMiner report merely because all workflow nodes have finished.

Produce it only after confirming that the available evidence has been correctly aligned, interpreted, qualified, and traced.

For model-reliance evidence, verify same-model identity, ID/OOD or paired semantics, label-preservation status, intervention quality, coverage, and uncertainty before synthesis. A completed backend without valid pairs or capabilities is not positive evidence.
