# Identify Agent Runtime Health Policy

## 1. Purpose

The heartbeat verifies that the Identify Agent can continue the current run safely and can produce a valid result.

Heartbeat behavior concerns runtime health only.

It must not perform shortcut discovery itself.

---

## 2. Required Runtime Checks

During execution, verify that the following conditions remain valid:

* a current `IdentifyRequest` exists;
* `run_id` is available;
* dataset configuration is available;
* required dataset paths are accessible;
* the target label definition is available;
* configured budgets are known;
* required tools are available when needed;
* the final output can be represented as a valid `CandidateReport`.

---

## 3. Dataset Accessibility

Before relying on dataset evidence, verify that the relevant data can actually be accessed.

Possible failures include:

```text
dataset root unavailable
metadata file missing
permission denied
unsupported file format
sample decoding failure
```

Do not assume that configured resources are accessible merely because they are listed in the request.

---

## 4. Budget Health

Track execution against the configured limits.

Verify that:

```text
samples_inspected <= budget.max_samples
candidates_returned <= budget.max_candidates
```

If a limit is reached, stop the corresponding exploration instead of exceeding the budget.

Budget exhaustion is not a runtime failure by itself.

---

## 5. Tool Health

When a required tool call fails:

1. record the failure;
2. determine whether the task can continue using remaining evidence;
3. avoid fabricating the missing result;
4. include the limitation in the final report when relevant.

A single non-critical tool failure does not necessarily require aborting the run.

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

* no valid `IdentifyRequest`;
* missing `run_id`;
* dataset root cannot be accessed at all;
* target label cannot be determined;
* required input is structurally invalid;
* no usable evidence can be accessed;
* a valid `CandidateReport` cannot be produced.

Do not compensate for non-recoverable failures by inventing information.

---

## 8. Output Validation

Before completion, verify that the final result:

* preserves the input `run_id`;
* uses `source = "identify"`;
* uses `message_type = "candidate_report"`;
* reports the actual number of inspected samples;
* reports whether metadata were actually used;
* contains no more than the configured candidate limit;
* uses unique candidate identifiers;
* uses only allowed cue families;
* uses only allowed priorities;
* sets every candidate status to `unverified`;
* records relevant limitations.

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
accessible evidence
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
* change candidate priority;
* perform statistical validation;
* make routing decisions;
* alter experiment configuration.

Its purpose is only to determine whether the current Identify execution remains healthy, degraded, or unable to complete.

---

## 14. Permanent Rule

> **Fail visibly when execution is invalid, degrade transparently when evidence is incomplete, and never fabricate progress.**
