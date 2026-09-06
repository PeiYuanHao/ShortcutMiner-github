# Identify Agent Evidence and Tool Boundary

## 1. Purpose

The Identify Agent receives bounded, deterministic discovery evidence prepared by the ShortcutMiner Python control layer. When `payload.tool_runtime` is present, it may additionally request only the tools listed there: `inspect_dataset`, `inspect_metadata`, `get_column_distribution`, `select_samples`, `inspect_sample_metadata`, `read_artifact`, and `inspect_tool_capabilities`.

Tools are executed by ShortcutMiner's validated Tool Registry, never by importing Python, evaluating code, or invoking a shell. Return `{"tool_calls": [ToolCall, ...]}` to request work; after structured ToolResult objects are returned in `payload.tool_runtime.results`, return the normal CandidateReport.

---

## 2. Runtime Boundary

Before Agent invocation, the control layer may use deterministic code under `tools/common/` to:

* inspect dataset structure;
* profile configured metadata;
* select metadata rows reproducibly;
* prepare basic sample or image evidence when the protocol explicitly supports it.

The resulting snapshot is supplied as `discovery_context` alongside the authoritative `request`:

```text
Python control layer
        |
        | bounded deterministic evidence
        v
discovery_context
        |
        | interpretation only
        v
Identify Agent
        |
        v
CandidateReport
```

The Agent must not:

* bypass the Tool Registry or call Python functions directly;
* open or scan paths outside the approved dataset/run roots;
* read an artifact or sample absent from the approved reference set;
* exceed the supplied Tool Runtime budget;
* read more dataset entries or metadata rows;
* decode a sample that the context does not explicitly report as inspected;
* invent a missing tool result.

---

## 3. Prepared Context

The current context describes:

* the protocol and run identity;
* the deterministic seed;
* dataset, task, and label identity;
* bounded dataset inspection results;
* bounded metadata profile results, when available;
* selected metadata rows, when available;
* the number of underlying samples actually inspected;
* evidence availability and control-layer limitations.

Treat each field according to what it measured. A field name or selected record is not evidence that the corresponding sample was viewed.

---

## 4. Hard Budgets

The request defines four limits:

```text
budget.max_samples
budget.max_candidates
budget.max_dataset_entries
budget.max_metadata_rows
```

The control layer is responsible for bounding dataset and metadata reads. The Agent must remain within the still-smaller evidence boundary represented by the context.

```text
context evidence <= request budget
reported inspected-sample count == context inspected-sample count
candidate evidence sources <= context evidence availability
```

Never extrapolate a larger inspected count merely because unused request budget remains.

---

## 5. Metadata-Row Selection Is Not Sample Inspection

Despite its compatibility name, `selected_samples` contains selected metadata-row wrappers. Each wrapper has `selection_kind = "metadata_row"`, a stable `selection_id`, zero-based `metadata_row_index`, one-based header-excluding `metadata_row_number`, and the original values under `record`. These rows may contain labels, identifiers, sources, groups, or paths.

Selection does not mean that an underlying image, audio item, document, or other sample was:

* opened;
* decoded;
* rendered;
* viewed;
* measured.

Therefore selected metadata rows may support `evidence_sources = ["metadata"]`. They do not support `sample_inspection` or `image_statistics` by themselves.

Only a positive `discovery_context.samples_inspected` together with corresponding sample-level observations authorizes `sample_inspection`. Only explicit prepared image statistics authorize `image_statistics`.

---

## 6. Canonical Evidence Sources

Every candidate must use one or more of exactly these labels:

```text
dataset_structure
metadata
sample_inspection
image_statistics
```

Use them as follows:

* `dataset_structure`: supported by usable `dataset_inspection` output;
* `metadata`: supported by usable `metadata_profile` or selected metadata rows;
* `sample_inspection`: supported by actual sample inspection recorded in the context;
* `image_statistics`: supported by image statistics explicitly recorded in the context.

Do not create new labels. Do not list a canonical label merely to make a candidate appear stronger.

---

## 7. Cue Family vs. Evidence Source

The cue family describes the suspected shortcut mechanism. The evidence source describes how it was observed. These are separate decisions.

For stable taxonomy assignment:

* a `hospital`, `site`, `domain`, `scanner_id`, or similar table field is a `metadata` cue when supported as a metadata field;
* a documented acquisition mechanism or a visibly observed scanner-, camera-, stain-, or compression-specific pattern is an `acquisition` cue;
* a hospital name alone does not make a cue `acquisition`.

Example:

```text
hospital_id column only
  -> cue.family = metadata
  -> evidence_sources = [metadata]

visible scanner-specific border in actually inspected images
  -> cue.family = acquisition
  -> evidence_sources = [sample_inspection]
```

Apply the same evidence-based classification across independent runs.

---

## 8. Failure and Degraded Evidence

Control-layer failures appear in `discovery_context.limitations` and availability fields.

When evidence is missing or malformed:

1. do not fabricate the missing output;
2. do not attempt an unbounded replacement scan;
3. use other prepared evidence when defensible;
4. carry relevant limitations into the report;
5. return fewer or zero candidates when necessary.

A missing, non-directory, or wholly uninspectable dataset root is an unrecoverable control-layer context error; the Agent is not invoked. When the root is usable, missing or invalid metadata, partial directory scan errors, bounded scan truncation, or row-selection failure are degraded conditions. Use remaining prepared evidence and copy every context limitation into the report, which may contain zero candidates.

---

## 9. Prohibited Validation

Identify must not perform or request:

* Cramér's V or mutual information validation;
* significance tests or validation confidence intervals;
* classifier training or predictive evaluation;
* ID/OOD performance comparison;
* model-reliance measurement;
* causal or counterfactual validation;
* downstream routing decisions.

Those operations belong to later ShortcutMiner stages.

---

## 10. Reproducibility

Use the supplied context exactly as prepared. The `seed` explains deterministic metadata-row selection, but it is not permission to resample.

Preserve the request's run and message identity. Do not import evidence or conclusions from another run.

---

## 11. Future Tool Expansion

Direct Agent tool access requires an explicit protocol and architecture revision. A future Identify-compatible tool must be bounded, reproducible, structured, independently tested, and limited to discovery rather than validation.

Until such a revision exists, the current rule is:

> **The control layer measures; `discovery_context` records; Identify interprets.**
