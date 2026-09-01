# Quantitative Agent Heartbeat

## 1. Purpose

The Quantitative Agent heartbeat is a lightweight self-check used to determine whether a quantitative task is:

* ready to execute,
* currently executing,
* blocked,
* failed,
* or completed.

The heartbeat must remain concise.

It is not a full experiment log and must not duplicate the final quantitative result.

---

# 2. Heartbeat Goals

At each heartbeat cycle, verify:

```text
1. Is there an active quantitative task?
2. Are the required inputs available?
3. Is the task executable?
4. Is execution progressing normally?
5. Have any failures or invalid conditions appeared?
6. Has the task produced sufficient quantitative evidence?
7. Is the result ready to return upstream?
```

---

# 3. Idle State

If no quantitative task is currently assigned:

```yaml
agent: quantitative
state: idle
action: wait
```

Do not invent work.

Do not independently select hypotheses from previous tasks unless explicitly instructed.

---

# 4. Task Intake Check

When a new task arrives, verify that it contains enough information to identify:

```text
task_id
hypothesis_id
dataset
target variable
candidate shortcut variable or intervention
requested analysis
required artifacts
```

If the task is sufficiently specified:

```yaml
state: ready
```

If critical information is missing:

```yaml
state: blocked
```

Example:

```yaml
agent: quantitative
state: blocked

task_id: Q-004
hypothesis_id: H-002

reason:
  code: VARIABLE_NOT_FOUND
  message: >
    The requested shortcut variable is not available
    in the provided metadata.
```

---

# 5. Input Availability Check

Before execution, verify required resources.

Possible resources include:

```text
dataset
metadata
annotations
dataset split
model checkpoint
predictions
logits
embeddings
configuration
intervention samples
```

Heartbeat example:

```yaml
state: validating

checks:
  dataset: available
  metadata: available
  shortcut_variable: available
  target_variable: available
  model: not_required
```

If any required resource is unavailable, stop execution and report the blocking condition.

---

# 6. Hypothesis Operationalization Check

Before running analysis, confirm that the upstream hypothesis has been converted into a measurable test.

The heartbeat should be able to answer:

```text
What is being measured?

Which variables are involved?

Which groups are compared?

Which metric or experiment will test the hypothesis?
```

Example:

```yaml
state: ready

test:
  shortcut_variable: background
  target_variable: label
  analysis: cramers_v
```

Do not execute a vague hypothesis directly.

---

# 7. Execution State

While an analysis is executing:

```yaml
agent: quantitative
state: running

task_id: Q-004

stage: subgroup_evaluation
```

Optional progress information may include:

```yaml
progress:
  completed_runs: 2
  total_runs: 3
```

Do not continuously emit unnecessary low-level updates.

Heartbeat state should represent meaningful progress only.

---

# 8. Long-Running Experiment Check

For model evaluation or intervention experiments that take multiple stages, track the current stage.

Recommended stages:

```text
loading_data
loading_model
preprocessing
running_inference
computing_metrics
running_intervention
aggregating_runs
validating_results
saving_artifacts
```

Example:

```yaml
state: running
stage: running_inference
```

This allows the Executive Agent to determine whether the task is progressing or stalled.

---

# 9. Reproducibility Check

Before accepting a result, verify that the execution context is recorded.

At minimum check:

```text
dataset
split
sample count
metric
seed
model checkpoint if applicable
experiment configuration
```

Example:

```yaml
reproducibility:
  seed_recorded: true
  split_recorded: true
  sample_count_recorded: true
  checkpoint_recorded: true
```

If critical reproducibility information is missing:

```yaml
state: incomplete
```

Do not mark the task as completed.

---

# 10. Numerical Validity Check

Before finalizing quantitative evidence, validate the generated values.

Check for:

```text
NaN
Inf
empty groups
zero-sized splits
invalid probabilities
invalid metric ranges
unexpected missing predictions
failed runs
```

Examples:

```text
accuracy must be within [0, 1]

AUROC must be within [0, 1]

group_count must be >= 0

probability must be within [0, 1]
```

If numerical outputs are invalid:

```yaml
state: failed

reason:
  code: INVALID_NUMERICAL_OUTPUT
```

---

# 11. Sample Sufficiency Check

Verify that reported metrics are supported by adequate sample counts.

Heartbeat should detect:

```text
empty subgroup
extremely small subgroup
missing class
single-class AUROC input
invalid contingency table
```

Example:

```yaml
state: blocked

reason:
  code: INSUFFICIENT_SAMPLE
  message: >
    The conflicting subgroup contains too few samples
    for reliable evaluation.
```

Do not hide small subgroup sizes.

---

# 12. Multi-Run Consistency Check

For experiments using multiple seeds or repeated trials, verify that all required runs have completed.

Example:

```yaml
runs:
  expected: 3
  completed: 3
  failed: 0
```

If only a subset succeeds:

```yaml
state: incomplete
```

unless the task explicitly allows partial results.

If partial results are returned, report them as partial.

---

# 13. Evidence Check

Before completion, determine what type of quantitative evidence was actually obtained.

Allowed evidence levels:

```text
Q0 — Not Testable
Q1 — Descriptive Pattern
Q2 — Statistical Association
Q3 — Model Dependence
Q4 — Intervention Sensitivity
```

Example:

```yaml
evidence:
  level: Q3
  direction: supports
```

Do not upgrade the evidence level beyond what the experiment demonstrates.

---

# 14. Contradictory Evidence

If results contradict the candidate shortcut hypothesis, this is not a failure.

Example:

```yaml
state: completed

evidence:
  level: Q3
  direction: contradicts
```

A valid negative result is a successful quantitative analysis.

Do not rerun experiments merely to obtain a supportive result.

---

# 15. Artifact Check

Before completion, verify that required outputs have been saved.

Possible artifacts include:

```text
result.json
metrics.json
tables
predictions
figures
logs
```

Example:

```yaml
artifacts:
  result_json: saved
  metrics_json: saved
```

If the calculation succeeds but required artifacts cannot be saved:

```yaml
state: incomplete
```

---

# 16. Failure Detection

Immediately stop and report meaningful failures.

Common failure codes:

```text
DATASET_NOT_FOUND
METADATA_NOT_FOUND
VARIABLE_NOT_FOUND
INVALID_SPLIT
MODEL_NOT_FOUND
CHECKPOINT_LOAD_FAILED
INSUFFICIENT_SAMPLE
INVALID_EXPERIMENT
INVALID_NUMERICAL_OUTPUT
EXECUTION_ERROR
TIME_BUDGET_EXCEEDED
UNSUPPORTED_METRIC
ARTIFACT_SAVE_FAILED
```

Heartbeat example:

```yaml
agent: quantitative
state: failed

task_id: Q-006

reason:
  code: CHECKPOINT_LOAD_FAILED
  message: >
    The requested model checkpoint could not be loaded.
```

---

# 17. Recovery Check

For recoverable failures, indicate what is required.

Example:

```yaml
state: blocked

recoverable: true

required_input:
  - model checkpoint
```

For non-recoverable execution failures:

```yaml
state: failed
recoverable: false
```

Do not repeatedly retry an unchanged failing task without a defined recovery action.

---

# 18. Completion Check

A task may be marked as completed only if:

```text
[1] Required inputs were validated.
[2] The hypothesis was operationalized.
[3] The requested analysis executed.
[4] Numerical outputs were validated.
[5] Sample counts were recorded.
[6] Reproducibility information was preserved.
[7] Evidence level was assigned.
[8] Evidence direction was assigned.
[9] Limitations were recorded when necessary.
[10] Required artifacts were saved.
```

Then return:

```yaml
agent: quantitative
state: completed
```

---

# 19. Recommended Heartbeat Schema

Use the following structure when a heartbeat message is required:

```yaml
agent: quantitative

task_id: Q-001
hypothesis_id: H-003

state: running

stage: computing_metrics

checks:
  inputs_valid: true
  execution_valid: true
  numerical_output_valid: pending
  artifacts_saved: false

progress:
  completed_runs: 2
  total_runs: 3

next_action: complete_remaining_run
```

Keep fields compact.

Only include fields relevant to the current state.

---

# 20. State Definitions

Use the following states consistently:

```text
idle
ready
validating
running
blocked
incomplete
failed
completed
```

Meaning:

```text
idle
No active task.

ready
Task is valid and ready to execute.

validating
Inputs or outputs are currently being checked.

running
Quantitative analysis is executing normally.

blocked
Execution cannot proceed without additional information or resources.

incomplete
Partial execution succeeded, but completion requirements are not satisfied.

failed
Execution encountered a non-successful terminal error.

completed
The quantitative task successfully produced valid evidence.
```

---

# 21. Heartbeat Minimality Rule

Do not use the heartbeat as a reasoning diary.

Do not include:

* internal chain-of-thought,
* long statistical explanations,
* raw model logs,
* complete metric tables,
* full experiment reports,
* speculative conclusions.

The heartbeat should only communicate:

```text
current state
current stage
critical validation status
blocking issue
next action
```

Detailed results belong in the Quantitative Agent output artifacts.

---

# 22. Governing Principle

The heartbeat exists to answer one operational question:

> **Can the Quantitative Agent continue executing the assigned task correctly and reproducibly?**

If yes, continue.

If not, clearly expose the blocking or failure condition instead of silently proceeding.
