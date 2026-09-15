# ShortcutMiner Communication Protocol

## 1. Current Scope

Protocol version `1.0` defines canonical contracts for the five-agent workflow:

```text
Orchestrator
    |
    | IdentifyRequest + IdentifyDiscoveryContext
    v
Identify Agent
    |
    | CandidateReport
    v
Executive Agent
    |
    | RoutingPlan + AnalysisTask messages
    v
Orchestrator
    |
    +--> QuantitativeRequest --> QuantitativeResult --+
    |                                               |
    +--> QualitativeRequest --> QualitativeAgentDraft
    |                         --> QualitativeResult ---+
                                                    |
                                                    v
                                             Summarize Agent
                                                    |
                                                    v
                                               SummaryReport
```

The executable Pydantic contracts are defined in
`shortcutminer/protocol/identify.py`, `shortcutminer/protocol/executive.py`,
`shortcutminer/protocol/quantitative.py`,
`shortcutminer/protocol/qualitative.py`, and
`shortcutminer/protocol/summarize.py`. Deterministic preparation and
cross-stage construction live in the corresponding `shortcutminer/experiment/`
modules. This document describes the same wire contracts and their cross-object
invariants.

Parallel branch result batches are defined in
`shortcutminer/protocol/batches.py` and form the explicit fan-in payload for
Summarize.

---

## 2. Validation Policy

Every implemented protocol model uses these rules:

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

When the remaining stage contracts are implemented, each message will define:

* an explicit version, run ID, message ID, source, target, and message type;
* request/response correlation where applicable;
* explicit artifact references;
* a strict stage-specific body;
* cross-object and budget invariants;
* a documented degraded/failure boundary.

Quantitative and Summarize contracts are implemented as strict Pydantic models.
Their detailed field-level reference can be expanded in a later documentation
phase; their executable schemas are authoritative.

---

## 11. Executive Planning Protocol

### 11.1 Current execution boundary

Executive converts a validated Identify `CandidateReport` into a routing plan:

```text
CandidateReport + authoritative routing context
        |
        v
Executive Agent
        |
        v
RoutingPlan
  |- RoutingDecision[]
  `- AnalysisTask[]
        |
        v
deterministic Orchestrator (future skeleton integration)
```

The current Executive executor plans only. It does not dispatch tasks, wait for downstream results, run heartbeat coordination, or claim task completion. Those lifecycle operations belong to deterministic infrastructure and remain outside this stage until the main execution skeleton is connected.

### 11.2 ExecutiveRequest

The invocation payload contains exactly one `request` object. Its complete envelope is:

```json
{
  "protocol_version": "1.0",
  "run_id": "waterbirds__model-a__seed42__r01",
  "message_id": "msg_executive_request_0001",
  "source": "orchestrator",
  "target": "executive",
  "message_type": "executive_request",
  "in_reply_to": "msg_candidate_report_0001",
  "candidate_report": {},
  "routing_context": {}
}
```

`candidate_report` is the complete, validated Identify output. It is not rewritten into a second candidate representation. In particular, Executive protocol `1.0` uses the Identify IDs `C001` through `C999`, not a separate `SC-001` namespace.

`in_reply_to` must equal `candidate_report.message_id`. Protocol version and run ID must agree across the request, report, and routing context. The request `message_id` must be new.

### 11.3 ExecutiveRoutingContext

The deterministic control layer supplies:

```json
{
  "protocol_version": "1.0",
  "run_id": "waterbirds__model-a__seed42__r01",
  "dataset": {
    "name": "waterbirds",
    "root": "/datasets/waterbirds",
    "metadata_file": "metadata.csv"
  },
  "task": {
    "type": "classification",
    "label_column": "y"
  },
  "audit_options": {
    "validation_depth": "standard",
    "max_tasks": 10,
    "max_samples_per_task": 40,
    "allowed_targets": ["quantitative", "qualitative"],
    "requested_outputs": ["metrics", "observations"],
    "runtime_constraints": ["no_retraining"]
  },
  "capabilities": {
    "available_targets": ["quantitative", "qualitative"],
    "available_metadata_fields": ["y", "place"],
    "available_splits": ["train", "val", "test"],
    "quantitative_checks": ["association_test"],
    "qualitative_checks": ["representative_sample_inspection"],
    "model_capabilities": ["predictions"],
    "sample_access": true,
    "external_services": false,
    "available_output_types": ["metrics", "tables", "observations"]
  },
  "artifact_references": ["identify/output.json"],
  "limitations": []
}
```

Capabilities are permissions and availability facts, not suggestions. Executive must not request a target, check, metadata field, split, model capability, output type, sample access, external-service access, or artifact reference absent from this context.

`max_tasks` is from 1 through 1998, matching at most two tasks for each possible Identify candidate. `max_samples_per_task` is a hard positive limit. Any task that requires sample access must therefore provide a non-null `sample_limit` within that bound. `validation_depth` is exactly `light`, `standard`, or `deep`; it never grants unlimited execution.

`runtime_constraints` contains only canonical protocol tokens. Version `1.0` supports `no_retraining` and `no_external_services`. Every planned task must copy the complete list so deterministic downstream infrastructure receives the same restrictions. A task under `no_retraining` cannot request the `retraining` model capability, and a task under `no_external_services` must set `requires_external_services` to false. External-service use is also rejected unless `capabilities.external_services` is true.

The context builder copies dataset and task identity from the Identify request, verifies that capability metadata fields occurred in the bounded Identify profile, and carries forward Identify limitations. It does not rescan the dataset.

### 11.4 RoutingPlan

Executive returns exactly one object:

```json
{
  "protocol_version": "1.0",
  "run_id": "waterbirds__model-a__seed42__r01",
  "message_id": "msg_routing_plan_0001",
  "source": "executive",
  "target": "orchestrator",
  "message_type": "routing_plan",
  "in_reply_to": "msg_executive_request_0001",
  "artifact_references": [],
  "decisions": [],
  "tasks": [],
  "limitations": []
}
```

The target is `orchestrator` because the Agent decides what should run while deterministic infrastructure owns physical dispatch, concurrency, retries, persistence, and lifecycle management.

The plan message ID must be distinct from both upstream message IDs. Every routing-context limitation must appear verbatim in plan limitations. Plan artifact references must be a subset of the authoritative context references.

### 11.5 RoutingDecision

There is exactly one decision for every input candidate, in input order:

```json
{
  "candidate": {},
  "route": "both",
  "routing_reason": "Measurable and semantic evidence are complementary.",
  "task_ids": ["TASK-C001-QT", "TASK-C001-QL"],
  "priority": "high",
  "status": "planned",
  "missing_requirements": [],
  "limitations": []
}
```

`candidate` is the complete original `Candidate` and must match it exactly. `priority` must also preserve the upstream value. The four routes are:

| Route | Exact task IDs | Status |
| --- | --- | --- |
| `quantitative` | `TASK-<candidate_id>-QT` | `planned` |
| `qualitative` | `TASK-<candidate_id>-QL` | `planned` |
| `both` | QT then QL tasks | `planned` |
| `unresolved` | Empty list | `unresolved` |

An executable decision cannot retain missing requirements. An unresolved decision must state at least one missing requirement or limitation. No candidate may be dropped merely because it cannot be executed.

### 11.6 AnalysisTask

Every planned task is itself a dispatchable protocol message:

```json
{
  "protocol_version": "1.0",
  "run_id": "waterbirds__model-a__seed42__r01",
  "message_id": "msg_task_C001_QT",
  "source": "executive",
  "target": "quantitative",
  "message_type": "analysis_task",
  "in_reply_to": "msg_routing_plan_0001",
  "task_id": "TASK-C001-QT",
  "candidate_id": "C001",
  "hypothesis": "Background category may be associated with the label.",
  "objective": "Measure the label-background association.",
  "inputs": {
    "dataset": "waterbirds",
    "dataset_reference": "/datasets/waterbirds",
    "label_column": "y",
    "split": "train",
    "metadata_fields": ["y", "place"],
    "artifact_references": [],
    "sample_limit": null,
    "requires_sample_access": false,
    "requires_external_services": false,
    "runtime_constraints": ["no_retraining"],
    "required_model_capabilities": []
  },
  "requested_checks": ["association_test"],
  "expected_outputs": ["metrics", "tables"],
  "priority": "high",
  "dependencies": [],
  "status": "planned"
}
```

Task IDs are deterministic and unique. QT tasks target `quantitative`; QL tasks target `qualitative`. The hypothesis and priority are copied exactly from the candidate. Dataset reference and label column are copied from routing context. Every other input and requested operation must remain within declared capabilities and budgets.

Tasks produced for `both` are independent by default. Dependencies are allowed only when genuine, must refer to tasks in the same plan, and must be acyclic. A planned task is not a dispatched, running, or completed task.

### 11.7 Cross-object invariants

The Executive boundary enforces:

```text
request/report/context/plan protocol_version and run_id agree
request.in_reply_to == candidate_report.message_id
plan.in_reply_to == request.message_id
one decision per candidate, in order
decision.candidate == upstream candidate
decision.priority == candidate.priority
task IDs exactly match route
plan tasks exactly match all decision task IDs, in order
len(tasks) <= audit_options.max_tasks
task target, suffix, candidate_id, hypothesis, and priority agree
task inputs/checks/outputs <= routing_context capabilities
task sample_limit <= max_samples_per_task
sample access implies a non-null bounded sample_limit
task runtime_constraints == audit_options.runtime_constraints
no_retraining excludes the retraining model capability
external-service use requires capability and is excluded by no_external_services
all artifact references <= authoritative references
all task dependencies exist and form an acyclic graph
```

Unknown fields and type coercion are rejected recursively, just as in Identify.

### 11.8 Failure and persistence

Invalid identity, correlation, candidate preservation, capability, budget, traceability, or schema is a protocol failure. Missing data or unavailable capabilities are not reasons to fabricate a task; Executive returns a visible `unresolved` decision when the request itself remains valid.

With run storage enabled, Executive persists:

```text
runs/<run_id>/executive/input.json
runs/<run_id>/executive/raw_output.json
runs/<run_id>/executive/routing.json
```

`input.json` is the exact invocation payload, `raw_output.json` is the unvalidated Agent object, and `routing.json` is written only after schema and semantic validation. Events record `EXECUTIVE_STARTED`, `EXECUTIVE_COMPLETED`, or `AGENT_FAILED` without claiming downstream execution.

---

## 12. Qualitative Evidence Protocol

### 12.1 Execution boundary

Qualitative consumes exactly one validated Executive task whose target is
`qualitative` and whose ID ends in `-QL`. The deterministic control layer
prepares evidence before invoking the Agent:

```text
Candidate + AnalysisTask + preparation options
        |
        v
bounded metadata load and split filter
        |
        v
random or group-balanced sample selection
        |
        v
dataset-contained sample-reference manifest
        |
        v
Qualitative Agent
        |
        v
validated QualitativeAgentDraft
        |
        v
deterministically finalized QualitativeResult
```

Preparation only resolves and records evidence. It does not claim to visually
inspect samples and does not transform source data.

### 12.2 QualitativeRequest

The request envelope is:

```json
{
  "protocol_version": "1.0",
  "run_id": "waterbirds__model-a__seed42__r01",
  "message_id": "msg_qualitative_request_C001",
  "source": "orchestrator",
  "target": "qualitative",
  "message_type": "qualitative_analysis_request",
  "in_reply_to": "msg_task_C001_QL",
  "candidate": {},
  "task": {},
  "metadata_file": "metadata.csv",
  "preparation": {
    "sampling_strategy": "group_balanced",
    "seed": 42,
    "sample_id_column": "sample_id",
    "sample_path_column": "img_filename",
    "split_column": "split",
    "group_columns": ["y", "place"],
    "max_evidence_records": 40,
    "max_metadata_rows": 100000
  },
  "limitations": []
}
```

The embedded candidate is the unmodified Identify candidate. The embedded task
is the unmodified Executive `AnalysisTask`. Its hypothesis, candidate ID,
dataset reference, requested checks, expected outputs, runtime constraints, and
sample limit remain authoritative. Preparation columns must be present in the
task's authorized label or metadata fields. `max_evidence_records` cannot
exceed the task's `sample_limit` when one is present.

### 12.3 QualitativeEvidenceContext

The invocation payload contains exactly:

```json
{
  "request": {},
  "evidence_context": {}
}
```

The evidence context records selected sample IDs, metadata row indexes, group
values, relevant metadata, dataset-relative sample references, file
availability, media type, byte size, available and selected group coverage,
the sampling seed and strategy, authoritative upstream artifacts, an allowed
candidate-scoped derived-artifact prefix, and every preparation limitation.

Sample paths must be relative to the configured dataset root. Absolute paths,
drive-qualified paths, `.`/`..` traversal, missing files, and unknown formats
are not exposed as valid inspectable sample evidence. Missing metadata, missing
groups, selection failure, or zero accessible samples produce a bounded degraded
context so the Agent can return `insufficient_evidence` instead of fabricating a
result.

### 12.4 QualitativeAgentDraft

The provider returns only scientific choices. It does not reproduce protocol,
request/candidate identity, sampling bookkeeping, limitations already present in
the context, observation IDs, or observation groups:

```json
{
  "sample_ids": ["S017", "S204"],
  "evidence_status": "partially_supported",
  "evidence_summary": "The cue is visible with several counterexamples.",
  "observations": [
    {
      "sample_refs": ["S017"],
      "observation": "A bounded visible property was observed.",
      "inspection_quality": "sufficient"
    }
  ],
  "interpretations": [
    {
      "interpretation": "The difference is descriptive rather than causal.",
      "supporting_observation_indexes": [0],
      "confidence": "low"
    }
  ],
  "counterevidence": [],
  "metadata_evidence": [],
  "failures": [],
  "additional_limitations": [],
  "artifact_references": [],
  "recommended_followup": []
}
```

The provider-facing schema is strict and forbids extra fields. Sample IDs and
every evidence reference are checked against the prepared context before
finalization. An observation may reference only one authoritative group, and
interpretation indexes must refer to existing observations.

An observation is group-local evidence: every observation's `sample_refs` must
belong to exactly one authoritative group. Cross-group analysis is expressed by
separate group-local observations plus an interpretation that cites their
indexes. Python never splits or rewrites an observation. If an otherwise valid
draft violates this structure, the executor may issue at most one value-free
draft-invariant correction for `observation_crosses_groups`; this retry is
tracked independently from JSON/schema and transport retries. The correction
contains only the typed category, observation index, and referenced group count,
never the previous draft, sample/group values, observation text, or private
truth.

### 12.5 Canonical QualitativeResult

The deterministic control layer combines the validated draft with the request
and evidence context to construct:

```json
{
  "protocol_version": "1.0",
  "run_id": "waterbirds__model-a__seed42__r01",
  "message_id": "msg_qualitative_result_C001",
  "source": "qualitative",
  "target": "summarize",
  "message_type": "qualitative_analysis_result",
  "in_reply_to": "msg_qualitative_request_C001",
  "payload": {
    "task_id": "TASK-C001-QL",
    "candidate_id": "C001",
    "candidate_name": "background_category",
    "hypothesis": "Background category may be associated with the label.",
    "evidence_status": "partially_supported",
    "evidence_summary": "The cue is visible with several counterexamples.",
    "sampling": {
      "strategy": "group_balanced",
      "seed": 42,
      "groups": {"y=0 | place=0": 1, "y=1 | place=1": 1},
      "sample_count": 2,
      "sample_ids": ["S017", "S204"]
    },
    "observations": [],
    "interpretations": [],
    "counterevidence": [],
    "metadata_evidence": [],
    "failures": [],
    "limitations": [],
    "artifact_references": [],
    "recommended_followup": []
  }
}
```

The five evidence statuses are `supported`, `partially_supported`,
`inconclusive`, `contradicted`, and `insufficient_evidence`. Interpretation
confidence is `high`, `medium`, or `low` and is explicitly not statistical
confidence.

Observations use candidate-scoped sequential IDs such as `QO-C001-001` and
must cite selected, inspected sample IDs. Interpretations cite existing
observation IDs. Counterevidence and metadata evidence cite selected records;
metadata fields must be authorized by the task. Failure categories are
canonical and remain distinct from negative scientific evidence.

`sampling.strategy` and `sampling.seed` come from the context; `sample_count`
and `groups` are derived exactly from the draft's validated `sample_ids`.
Observation IDs are generated sequentially and groups are derived from sample
references. Authoritative context limitations are unioned with the provider's
`additional_limitations` without allowing the provider to remove them.

### 12.6 Cross-object invariants

The Qualitative boundary enforces:

```text
request/task/context/result protocol_version and run_id agree
request.in_reply_to == task.message_id
result.in_reply_to == request.message_id
result.payload.task_id == request.task.task_id
candidate ID, cue name, and hypothesis are preserved
task target == qualitative and task ID suffix == -QL
preparation record budget <= task sample_limit
result sampling strategy and seed == evidence context
result sample_count == len(sample_ids)
result sample IDs <= selected evidence-context sample IDs
sample-access tasks may cite only available sample files
reported group counts exactly match cited sample groups
observation IDs are candidate-scoped and sequential
observations/counterevidence/metadata evidence cite known samples
interpretations cite known observations
metadata fields <= task-authorized fields
artifact references <= upstream references or candidate output prefix
every evidence-context limitation is preserved verbatim
substantive statuses require direct evidence
insufficient_evidence requires a visible limitation or failure
```

These rules provide structural traceability. They do not turn a qualitative
observation into proof that a model uses the cue; the workspace contract keeps
dataset-level evidence and model reliance separate.

### 12.7 Persistence and failure

With run storage enabled, Qualitative persists:

```text
runs/<run_id>/qualitative/input.json
runs/<run_id>/qualitative/raw_output.json
runs/<run_id>/qualitative/draft_invariant_telemetry.json
runs/<run_id>/qualitative/result.json
runs/<run_id>/qualitative/artifacts/<candidate_id>/...
```

`input.json` preserves the exact request and deterministic evidence context.
`raw_output.json` contains only a schema-valid provider draft; invalid or
non-object responses are not persisted. `result.json` is written only after
draft validation, deterministic finalization, and canonical semantic validation.
`draft_invariant_telemetry.json` contains only retry counts, stable scientific
input hashes, pass/fail states, and bounded value-free failure metadata.
Derived artifact files are
required to exist before a stored result may cite them. Events record
`QUALITATIVE_STARTED`, `QUALITATIVE_COMPLETED`, or `AGENT_FAILED`.
