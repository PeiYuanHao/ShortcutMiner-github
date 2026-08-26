# ShortcutMiner Communication Protocol

## 1. Current Scope

Protocol version `1.0` currently defines the complete canonical contract for the Identify stage:

```text
Orchestrator
    |
    | IdentifyRequest + IdentifyDiscoveryContext
    v
Identify Agent
    |
    | CandidateReport
    v
Executive
```

The executable Pydantic contract is defined in `shortcutminer/protocol/identify.py`. The prepared context is defined in `shortcutminer/experiment/identify_context.py`. This document describes the same wire contract and its cross-object invariants.

Routing, quantitative, qualitative, and summary message bodies will be added as their agents are implemented. They must follow the same explicit envelope principles, but no incomplete placeholder schema for those stages is part of protocol `1.0` today.

---

## 2. Validation Policy

Every Identify protocol model uses these rules:

* unknown fields are forbidden at every nesting level;
* JSON types are strict and are not silently coerced;
* text is stripped of surrounding whitespace;
* required text must remain non-empty after stripping;
* `protocol_version` is required and must equal the string `"1.0"` exactly;
* message routing fields are required in serialized input and output;
* output control fields are never supplied by schema defaults.

For example, numeric text such as `"40"` is not accepted for an integer budget, and `1` is not accepted for a boolean.

---

## 3. Agent Invocation Payload

The object sent to Identify has exactly two top-level members:

```json
{
  "request": {},
  "discovery_context": {}
}
```

`request` is the authoritative configuration and communication envelope. `discovery_context` is the immutable evidence snapshot prepared by the Python control layer for that request.

The Identify Agent does not directly invoke the Python inspection, profiling, or selection tools. It interprets the supplied snapshot.

---

## 4. IdentifyRequest

### 4.1 Complete example

```json
{
  "protocol_version": "1.0",
  "run_id": "waterbirds__model-a__seed42__r01",
  "message_id": "msg_identify_0001",
  "source": "orchestrator",
  "target": "identify",
  "message_type": "identify_request",
  "dataset": {
    "name": "waterbirds",
    "root": "/datasets/waterbirds",
    "metadata_file": "metadata/metadata.csv"
  },
  "task": {
    "type": "classification",
    "label_column": "y",
    "description": "Classify the bird category.",
    "intended_signal": "Bird morphology."
  },
  "budget": {
    "max_samples": 40,
    "max_candidates": 5,
    "max_dataset_entries": 10000,
    "max_metadata_rows": 100000
  }
}
```

### 4.2 Envelope fields

| Field | Type | Rule |
| --- | --- | --- |
| `protocol_version` | string | Required; exactly `"1.0"`. |
| `run_id` | string | Required safe single-path run identifier. |
| `message_id` | string | Required non-empty request correlation identifier. |
| `source` | string | Required; exactly `"orchestrator"`. |
| `target` | string | Required; exactly `"identify"`. |
| `message_type` | string | Required; exactly `"identify_request"`. |

`run_id` must be 1–128 characters, begin with an ASCII letter or digit, and contain only ASCII letters, digits, `.`, `_`, and `-`. It must not be a Windows reserved filesystem name or end in a dot or space. Slashes, backslashes, absolute paths, parent traversal, whitespace, and non-ASCII path characters are rejected.

`message_id` is opaque to Identify. It must be preserved for response correlation through `CandidateReport.in_reply_to`.

### 4.3 DatasetSpec

| Field | Type | Rule |
| --- | --- | --- |
| `name` | string | Required non-empty dataset name. |
| `root` | string | Required non-empty control-layer dataset root. |
| `metadata_file` | string or null | Optional safe path relative to `root`. |

`metadata_file` must not be absolute, drive-qualified, empty, or contain empty, `.` or `..` path segments. Both `/` and `\` separators are checked so a path cannot bypass validation by targeting a different deployment operating system.

The dataset root itself may be an absolute deployment path because it selects the explicitly configured dataset boundary. The relative metadata path must stay inside that boundary.

### 4.4 TaskSpec

| Field | Type | Rule |
| --- | --- | --- |
| `type` | string | Required non-empty task type. |
| `label_column` | string | Required non-empty target column. |
| `description` | string or null | Optional non-empty task description. |
| `intended_signal` | string or null | Optional non-empty description of the intended semantic signal. |

The optional fields provide discovery context. They do not override the structured dataset, task type, or target column.

### 4.5 IdentifyBudget

| Field | Type | Default | Rule |
| --- | --- | --- | --- |
| `max_samples` | integer | `40` | At least 1; bounds metadata-row selection and actual sample inspection. |
| `max_candidates` | integer | `5` | From 1 through 999; bounds returned candidates and matches the `C001`–`C999` identifier space. |
| `max_dataset_entries` | integer | `10000` | At least 1; bounds entries retained and counted by dataset traversal. |
| `max_metadata_rows` | integer | `100000` | At least 1; bounds the metadata rows retained, profiled, and selected. |

The budget object may be omitted for backward-compatible programmatic construction, in which case all four documented defaults apply. If the object or any value is serialized explicitly, it must satisfy the strict type and range rules.

---

## 5. IdentifyDiscoveryContext

### 5.1 Complete shape

The control layer produces these fields:

```json
{
  "protocol_version": "1.0",
  "run_id": "waterbirds__model-a__seed42__r01",
  "seed": 42,
  "dataset_name": "waterbirds",
  "task_type": "classification",
  "label_column": "y",
  "dataset_inspection": {},
  "metadata_profile": {},
  "metadata_available": true,
  "selected_samples": [
    {
      "selection_kind": "metadata_row",
      "selection_id": "metadata-row-00000000",
      "metadata_row_index": 0,
      "metadata_row_number": 1,
      "record": {
        "y": 0,
        "place": 1,
        "img_filename": "001.jpg"
      }
    }
  ],
  "metadata_rows_selected": 1,
  "samples_inspected": 0,
  "limitations": []
}
```

`protocol_version`, `run_id`, `dataset_name`, `task_type`, and `label_column` must agree with the request. `seed` is the deterministic row-selection seed.

`dataset_inspection` contains the bounded result of dataset reconnaissance. `metadata_profile` is either the bounded profile or `null`. `metadata_available` states whether configured metadata could be used.

### 5.2 Selected metadata rows

`selected_samples` is retained as a compatibility field name, but every current item is a metadata-row wrapper:

| Field | Meaning |
| --- | --- |
| `selection_kind` | Always `"metadata_row"`. |
| `selection_id` | Stable selection identifier such as `metadata-row-00000000`. |
| `metadata_row_index` | Zero-based logical CSV record index. |
| `metadata_row_number` | One-based logical CSV record number, excluding the header. |
| `record` | Original selected metadata values. |

The following invariant is exact:

```text
metadata_rows_selected == len(selected_samples)
```

Selecting a metadata row does not open, decode, render, view, or measure the referenced sample. It therefore does not increment `samples_inspected`.

In the current Identify control path:

```text
samples_inspected == 0
```

Consequently, current prepared rows can support `metadata` evidence but cannot support `sample_inspection` or `image_statistics`. Those labels remain reserved for a future context that explicitly carries such evidence and a truthful positive inspection count.

### 5.3 Context limits

The preparation layer enforces:

```text
dataset entries retained and counted <= request.budget.max_dataset_entries
metadata rows retained and profiled <= request.budget.max_metadata_rows
metadata_rows_selected <= request.budget.max_samples
samples_inspected <= request.budget.max_samples
```

The context may be smaller than the request budget because the dataset is smaller, evidence is unavailable, or a bounded operation failed. A budget is permission to prepare up to a limit; it is not evidence that the limit was consumed.

---

## 6. CandidateReport

### 6.1 Complete example

```json
{
  "protocol_version": "1.0",
  "run_id": "waterbirds__model-a__seed42__r01",
  "message_id": "msg_candidate_report_0001",
  "source": "identify",
  "target": "executive",
  "message_type": "candidate_report",
  "in_reply_to": "msg_identify_0001",
  "artifact_references": [],
  "dataset_summary": {
    "dataset": "waterbirds",
    "samples_inspected": 0,
    "metadata_used": true
  },
  "candidates": [
    {
      "candidate_id": "C001",
      "cue": {
        "family": "metadata",
        "name": "background_category"
      },
      "description": "Background category may encode label information.",
      "observation": "Prepared metadata rows contain background categories.",
      "hypothesis": "Background category may be spuriously associated with the target label.",
      "evidence_sources": ["metadata"],
      "priority": "medium",
      "status": "unverified"
    }
  ],
  "limitations": [
    "Selected metadata rows were not decoded as samples."
  ]
}
```

### 6.2 Envelope and traceability

| Field | Type | Rule |
| --- | --- | --- |
| `protocol_version` | string | Required; exactly `"1.0"`; equal to request. |
| `run_id` | string | Required safe run ID; equal to request. |
| `message_id` | string | Required non-empty new report identifier. |
| `source` | string | Required; exactly `"identify"`. |
| `target` | string | Required; exactly `"executive"`. |
| `message_type` | string | Required; exactly `"candidate_report"`. |
| `in_reply_to` | string | Required; exactly the request `message_id`. |
| `artifact_references` | array of strings | Required, possibly empty; known safe run-relative paths only. |

Artifact references must use the same safe-relative-path rules as `metadata_file`. Absolute paths and any `..` traversal are rejected. References are relative to the current run directory. An Agent must not invent a path; when no authoritative artifact reference is supplied, it returns an explicit empty array. The current invocation payload supplies none, so the current executor requires this array to be empty.

### 6.3 DatasetInspectionSummary

| Field | Type | Rule |
| --- | --- | --- |
| `dataset` | string | Required; equal to request dataset name. |
| `samples_inspected` | integer | Required, non-negative, and truthfully bounded. |
| `metadata_used` | boolean | Required strict boolean. |

The inspected count obeys the hard upper bound:

```text
report.dataset_summary.samples_inspected
    == discovery_context.samples_inspected
    <= request.budget.max_samples
```

`metadata_used` may be true only when metadata were available in the context and were actually used in discovery. It is false when prepared metadata were absent or inaccessible.

### 6.4 Candidate

Every candidate explicitly contains:

| Field | Rule |
| --- | --- |
| `candidate_id` | `C001`–`C999` format; unique and sequential from `C001` within the report. |
| `cue.family` | One canonical cue family. |
| `cue.name` | Non-empty normalized cue name. |
| `description` | Non-empty human-readable description. |
| `observation` | Non-empty statement limited to current prepared evidence. |
| `hypothesis` | Non-empty uncertainty-aware shortcut hypothesis. |
| `evidence_sources` | Non-empty list of canonical evidence labels. |
| `priority` | Exactly `high`, `medium`, or `low`. |
| `status` | Required explicitly; exactly `unverified`. |

Candidate count must not exceed `request.budget.max_candidates`. Candidate IDs must not repeat.

### 6.5 Cue families

The canonical families are:

```text
appearance
spatial
context
geometry
acquisition
metadata
other
```

Institution and site concepts are classified by supporting evidence to prevent cross-run drift:

* a `hospital`, `site`, `domain`, `scanner_id`, or similar table field is `metadata`;
* a documented capture mechanism or an actually visible scanner-, camera-, stain-, or compression-specific cue is `acquisition`.

The word “hospital” alone does not imply `acquisition`.

### 6.6 Evidence sources

The only allowed evidence labels are:

```text
dataset_structure
metadata
sample_inspection
image_statistics
```

Each label must correspond to evidence actually available in `discovery_context`. Selected metadata rows authorize `metadata` only. A candidate cannot use an unsupported label, an empty evidence list, or repeat the same label.

### 6.7 Limitations

`limitations` is a required array. Each item must be non-empty after stripping. Use an explicit empty array when no limitation applies.

Every string in `discovery_context.limitations` must be preserved verbatim in the report. The Agent may add further honest limitations.

---

## 7. Cross-Object Invariants

Schema validation establishes object shape. The Identify execution boundary additionally enforces semantic relationships:

```text
request.protocol_version == context.protocol_version == report.protocol_version
request.run_id == context.run_id == report.run_id
request.dataset.name == context.dataset_name == report.dataset_summary.dataset
request.task.type == context.task_type
request.task.label_column == context.label_column
report.in_reply_to == request.message_id
report.message_id != request.message_id
len(report.candidates) <= request.budget.max_candidates
report.samples_inspected == context.samples_inspected <= request.budget.max_samples
context.metadata_rows_selected == len(context.selected_samples)
```

Candidate IDs are exactly `C001`, `C002`, ... in report order. Candidate evidence sources and dataset-summary claims must not exceed the evidence availability represented by the context. Metadata evidence requires `dataset_summary.metadata_used = true`; sample-inspection evidence requires a positive context inspection count; the current context never authorizes image-statistics evidence. All context limitations must be carried into the report.

---

## 8. Degraded Completion and Failure

The dataset root is the minimum evidence boundary. If it is missing, is not a directory, or cannot be inspected at all, context construction raises an explicit error and the Identify Agent is not invoked. This is an unrecoverable stage failure.

Once the root has been inspected successfully, missing or rejected metadata, metadata parse failure, partial directory scan errors, bounded scan truncation, sampling failure, or zero selected rows are degraded evidence conditions rather than automatic stage failures.

When the request and context remain structurally valid, Identify should return the best defensible report, which may contain:

```json
{
  "dataset_summary": {
    "samples_inspected": 0,
    "metadata_used": false
  },
  "candidates": [],
  "limitations": ["Configured metadata file is unavailable or invalid."]
}
```

The excerpt above illustrates degraded content; the actual report still includes every required envelope and traceability field.

Protocol failure is also appropriate when the invocation payload is invalid, identities conflict, required control fields are missing, or the Agent output cannot form a valid `CandidateReport`. Degraded evidence must never be hidden by omitted context limitations, fabricated observations, sample counts, evidence labels, or candidates.

---

## 9. Persistence and Reproducibility

A reproducible Identify run is defined by the exact request, exact prepared context, accessible dataset state, control-layer implementation, and model behavior.

Persisting only `IdentifyRequest` is insufficient to reconstruct the evidence shown to the Agent. Run artifacts retain the exact `{request, discovery_context}` invocation payload, including seed, selected metadata-row wrappers, scan truncation state, limitations, and relevant artifact references. The unvalidated Agent response is retained separately from the validated canonical report so schema or semantic failures remain diagnosable.

Artifact storage is run-local and must not become hidden cross-run Agent memory.

---

## 10. Protocol Evolution

Protocol `1.0` is strict: adding a new field to an existing object requires a deliberate protocol update because unknown fields are rejected.

When the remaining agents are implemented, each message will define:

* an explicit version, run ID, message ID, source, target, and message type;
* request/response correlation where applicable;
* explicit artifact references;
* a strict stage-specific body;
* cross-object and budget invariants;
* a documented degraded/failure boundary.

Until those contracts exist, the complete canonical contract in this document applies only to Identify. It does not claim that placeholder schemas for later agents are already valid or stable.
