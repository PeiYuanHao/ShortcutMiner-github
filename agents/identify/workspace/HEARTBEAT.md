# Identify Agent Runtime Health Policy

## 1. Purpose

The heartbeat verifies that the Identify Agent can continue the current run safely and can produce a valid result.

Heartbeat behavior concerns runtime health only.

It must not perform shortcut discovery itself.

---

## 2. Required Runtime Checks

During execution, verify that the following conditions remain valid:

* the top-level `{request, discovery_context}` payload exists;
* the request envelope is complete and uses protocol `1.0`;
* `protocol_version`, `run_id`, dataset identity, task type, and label column agree across both objects;
* all four configured budgets are known;
* prepared evidence availability and limitations are explicit;
* the final output can be represented as a valid `CandidateReport`.

---

## 3. Prepared Evidence Accessibility

Before relying on evidence, verify that it is actually present and marked usable in `discovery_context`.

Possible failures include:

```text
dataset root unavailable
metadata file missing
permission denied
unsupported file format
sample decoding failure
```

The Python control layer, not the Agent, attempts dataset access. Do not retry a failed access or call inspection tools directly. A configured path is not evidence, and a selected metadata row is not an inspected sample.

---

## 4. Budget Health

Track execution against the configured limits.

Verify that:

```text
reported_samples_inspected == context.samples_inspected <= budget.max_samples
candidates_returned <= budget.max_candidates
dataset entries prepared <= budget.max_dataset_entries
metadata rows processed <= budget.max_metadata_rows
```

If a limit is reached, stop the corresponding exploration instead of exceeding the budget.

Budget exhaustion is not a runtime failure by itself.

---

## 5. Tool Health

When the context records that a control-layer operation failed:

1. record the failure;
2. determine whether the task can continue using remaining evidence;
3. avoid fabricating the missing result;
4. include the limitation in the final report when relevant.

A single non-critical preparation failure does not necessarily require aborting the run. The Agent does not directly rerun the failed tool.

---

## 6. Recoverable Conditions

The Identify Agent may continue when:

* some samples are unreadable;
* optional metadata are missing;
* a subset of files is unavailable;
* one evidence source fails;
* fewer candidates can be produced than requested.

Continue only when remaining evidence is sufficient for defensible discovery.

---

## 7. Non-Recoverable Conditions

The run should be treated as unable to complete normally when essential context is missing or unusable.

Examples include:

* no valid `{request, discovery_context}` payload;
* missing `run_id`;
* dataset root is missing, is not a directory, or cannot be inspected at all;
* target label cannot be determined;
* required input is structurally invalid;
* a valid `CandidateReport` cannot be produced.

Missing metadata, partial directory errors, bounded scan truncation, selection failure, or zero candidate-supporting evidence is recoverable when the dataset root was successfully inspected and the payload remains valid. Identify can then complete in a degraded state with explicit limitations and zero candidates.

Do not compensate for non-recoverable failures by inventing information.

---

## 8. Output Validation

Before completion, verify that the final result:

* explicitly uses `protocol_version = "1.0"`;
* preserves the input `run_id`;
* has a non-empty `message_id`;
* uses `source = "identify"`;
* uses `target = "executive"`;
* uses `message_type = "candidate_report"`;
* sets `in_reply_to` to the request `message_id`;
* explicitly contains safe run-relative `artifact_references`;
* reports exactly the inspected-sample count recorded by the context;
* reports whether metadata were actually used;
* contains no more than the configured candidate limit;
* uses unique candidate identifiers sequentially from `C001`;
* uses only allowed cue families;
* uses only allowed priorities;
* explicitly sets every candidate status to `unverified`;
* preserves every context limitation verbatim and explicitly records any additional relevant limitations, including an empty list when none apply.

The output must conform to the current `CandidateReport` protocol.

---

## 9. Candidate Integrity Check

Before returning the report, confirm that no candidate:

* claims validation;
* claims statistical significance;
* claims causal evidence;
* claims model reliance;
* contains fabricated observations;
* contains an evidence source that was not used;
* duplicates another candidate without a distinct mechanism.

If such a candidate exists, correct or remove it before completion.

---

## 10. Completion State

Identify may complete successfully when:

```text
valid request
+
valid prepared context
+
budget-compliant inspection
+
candidate construction completed
+
valid CandidateReport
```

A successful run may contain:

```text
zero candidates
```

if no defensible shortcut hypothesis was discovered.

Successful execution means that discovery completed correctly.

It does not mean that a shortcut was found or validated.

---

## 11. Degraded Completion

When discovery can complete but important evidence is missing, return the best valid `CandidateReport` supported by available evidence and record the limitation.

Examples:

```text
Metadata were unavailable.
Only 18 of 40 requested samples could be decoded.
Only the training split was accessible.
```

Do not hide degraded execution conditions.

If the root is usable but optional evidence is unavailable and no evidence supports a candidate, degraded completion means an explicit zero-candidate report with `samples_inspected = 0`, truthful `metadata_used`, and every prepared limitation. Do not turn that evidence shortage into a fabricated candidate. A wholly unavailable dataset root fails earlier during context construction instead.

---

## 12. Failure Reporting

When completion is impossible:

* fail explicitly;
* preserve the current `run_id` when available;
* report the reason;
* do not generate synthetic candidates merely to satisfy the output contract.

Runtime failure is preferable to fabricated scientific evidence.

---

## 13. Heartbeat Boundary

The heartbeat must not:

* generate shortcut candidates;
* inspect additional samples for discovery;
* directly invoke Python discovery tools;
* change candidate priority;
* perform statistical validation;
* make routing decisions;
* alter experiment configuration.

Its purpose is only to determine whether the current Identify execution remains healthy, degraded, or unable to complete.

---

## 14. Permanent Rule

> **Fail visibly when execution is invalid, degrade transparently when evidence is incomplete, and never fabricate progress.**
