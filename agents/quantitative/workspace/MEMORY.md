# Quantitative Agent Memory

## 1. Purpose

The Quantitative Agent memory stores only information that is useful for maintaining **consistent, reproducible, and efficient quantitative auditing across tasks**.

Memory is not a raw experiment log.

Memory should preserve stable knowledge such as:

* validated dataset structure,
* reusable metric definitions,
* confirmed variable mappings,
* successful experiment configurations,
* known failure conditions,
* artifact conventions,
* previously verified quantitative results that may be reused.

Do not store unnecessary transient execution details.

---

# 2. Memory Principle

Store information only when it is likely to improve future quantitative tasks.

A useful memory item should answer at least one of the following:

```text
Will this prevent repeated work?

Will this prevent inconsistent metrics?

Will this prevent repeating a known failure?

Will this preserve reproducibility?

Will this help interpret future tasks consistently?
```

If not, do not store it.

---

# 3. What Should Be Remembered

The Quantitative Agent may remember the following categories.

## 3.1 Dataset Structure

Store verified dataset information such as:

```text
dataset name
dataset version
available splits
metadata schema
label column
domain column
environment column
group attributes
sample identifier
known subgroup definitions
```

Example:

```yaml
dataset:
  name: Waterbirds

verified_schema:
  sample_id: img_id
  target: y
  background: place
  split: split
```

Only store mappings that were actually verified.

---

# 4. Variable Mappings

Store confirmed mappings between semantic concepts and dataset fields.

Example:

```yaml
variable_mapping:
  concept: class_label
  field: y
```

or:

```yaml
variable_mapping:
  concept: background_type
  field: place
```

This avoids repeatedly rediscovering the same schema.

Never infer a mapping solely from a column name without validating it.

---

# 5. Split Definitions

Store verified split conventions when they are stable.

Example:

```yaml
split_mapping:
  0: train
  1: validation
  2: test
```

This is especially important when datasets encode splits numerically.

Never reuse a split mapping from one dataset for another.

---

# 6. Group Definitions

Store subgroup definitions that have been explicitly validated.

Example:

```yaml
groups:
  aligned:
    condition: label == background

  conflicting:
    condition: label != background
```

or:

```yaml
groups:
  hospital_0:
    domain: 0
```

Group definitions must be linked to the relevant dataset.

Do not create globally shared subgroup definitions when semantics differ across datasets.

---

# 7. Metric Definitions

Store canonical metric definitions used by ShortcutMiner.

Example:

```yaml
metric:
  name: shortcut_accuracy_gap

  definition: >
    Accuracy on shortcut-aligned samples minus
    accuracy on shortcut-conflicting samples.
```

Another example:

```yaml
metric:
  name: prediction_flip_rate

  definition: >
    Fraction of samples whose predicted class changes
    after the controlled shortcut intervention.
```

Metric definitions should remain stable once established.

---

# 8. Statistical Procedure Memory

Store reusable statistical procedures when they have been validated.

Examples:

```text
Cramér's V implementation
bootstrap confidence interval procedure
subgroup aggregation method
multi-seed aggregation method
AUROC handling
missing-value policy
```

Example:

```yaml
procedure:
  name: multi_seed_aggregation

  aggregation:
    center: mean
    dispersion: standard_deviation
```

Do not silently change these procedures between comparable experiments.

---

# 9. Threshold Memory

Store thresholds only when their source is explicit.

Example:

```yaml
threshold:
  metric: some_metric
  value: 0.25
  source: benchmark_protocol_v1
```

Never store arbitrary thresholds as universal rules.

Threshold provenance must always be preserved.

Allowed sources may include:

```text
benchmark protocol
experiment configuration
validated calibration
paper-defined evaluation rule
user-defined criterion
```

---

# 10. Model Evaluation Configuration

Store stable evaluation settings for reusable models.

Possible fields:

```text
architecture
checkpoint identifier
input resolution
normalization
decision threshold
class mapping
evaluation mode
```

Example:

```yaml
model:
  id: resnet18_waterbirds_seed42

  checkpoint: checkpoints/resnet18_seed42.pt

  preprocessing:
    resolution: 224
    normalization: imagenet
```

Do not store temporary checkpoint paths that are no longer valid unless they are required for provenance.

---

# 11. Experiment Configuration Memory

Store reusable experiment configurations that may be required for comparable future runs.

Example:

```yaml
experiment:
  name: background_swap_test

  preserved_region:
    type: roi

  intervention:
    region: background

  metric:
    - prediction_flip_rate
    - confidence_delta
```

This helps ensure that repeated experiments use equivalent procedures.

---

# 12. Reproducibility Memory

Store reproducibility conventions such as:

```text
default random seed policy
number of repeated runs
artifact directory structure
naming conventions
result schema version
```

Example:

```yaml
reproducibility:
  default_seeds:
    - 42
    - 3407
    - 2026
```

Only treat these as defaults when they have been explicitly established by project configuration.

---

# 13. Artifact Conventions

Remember where quantitative outputs should be stored.

Recommended convention:

```text
results/
└── quantitative/
    └── <task_id>/
        ├── result.json
        ├── metrics.json
        ├── tables/
        ├── predictions/
        ├── figures/
        └── logs/
```

Remember naming rules that allow outputs to be traced to:

```text
task_id
hypothesis_id
dataset
model
seed
```

---

# 14. Result Schema Memory

Maintain consistency in the structure of quantitative outputs.

Preferred result fields include:

```text
task_id
hypothesis_id
dataset
split
sample_count
analysis_type
metrics
evidence_level
evidence_direction
limitations
reproducibility
artifacts
```

If the schema evolves, preserve a version identifier.

Example:

```yaml
schema_version: quantitative_result_v1
```

---

# 15. Known Valid Results

Previously validated results may be remembered when they are likely to be reused.

Example:

```yaml
validated_result:
  dataset: Waterbirds
  split: train
  metric: cramers_v
  value: 0.8673
```

A remembered value must also preserve enough provenance to determine whether reuse is valid.

At minimum:

```text
dataset
version if relevant
split
variables
metric definition
experiment configuration
```

Do not reuse a numerical result when the current task uses different conditions.

---

# 16. Cached Result Policy

Memory may point to cached quantitative artifacts.

Example:

```yaml
cache:
  task_signature: waterbirds_train_y_place_cramers_v
  artifact: results/quantitative/Q-001/result.json
```

Before reuse, verify that the current task matches:

```text
dataset
dataset version
split
variables
metric
configuration
```

If all relevant conditions match:

```yaml
execution_source: cache
```

Otherwise recompute.

Never present cached evidence as newly executed evidence.

---

# 17. Known Failure Patterns

Remember recurring failures that can prevent wasted execution.

Example:

```yaml
known_issue:
  dataset: example_dataset

  condition:
    metric: auroc
    subgroup: domain_4

  problem:
    single_class_target

  consequence:
    auroc_not_defined
```

Another example:

```yaml
known_issue:
  model: model_A

  problem:
    checkpoint_requires_different_class_mapping
```

Known issues should help validate tasks before expensive execution begins.

---

# 18. Recovery Knowledge

Memory may preserve successful recovery procedures for recurring operational issues.

Example:

```yaml
recovery:
  error: CHECKPOINT_LOAD_FAILED

  cause:
    state_dict_contains_module_prefix

  verified_fix:
    remove_module_prefix_before_loading
```

Only store recovery procedures that have been successfully verified.

Do not store speculative fixes as confirmed solutions.

---

# 19. Invalid Assumptions Memory

Remember assumptions that were previously shown to be incorrect.

Example:

```yaml
invalid_assumption:
  dataset: Dataset-X

  assumption:
    split_column_1_means_test

  corrected_mapping:
    1: validation
```

This is important because repeated schema assumptions can silently corrupt quantitative results.

---

# 20. Evidence Interpretation Memory

Store stable interpretations only when they are part of an explicit project protocol.

For example:

```yaml
evidence_levels:
  Q0: not_testable
  Q1: descriptive_pattern
  Q2: statistical_association
  Q3: model_dependence
  Q4: intervention_sensitivity
```

These definitions should remain consistent across tasks.

Do not store unsupported claims such as:

```text
Q4 means shortcut definitively proven.
```

Evidence levels describe evidence type, not absolute certainty.

---

# 21. Evidence Direction Memory

Maintain the standardized directions:

```text
supports
contradicts
inconclusive
```

Do not create unnecessary synonyms such as:

```text
positive
negative
maybe
weak_yes
```

Stable labels make downstream aggregation easier.

---

# 22. What Must Not Be Remembered

Do not store:

* chain-of-thought,
* hidden reasoning,
* temporary scratch calculations,
* arbitrary guesses,
* unverified dataset mappings,
* unsupported thresholds,
* partial metrics from failed runs,
* misleading intermediate values,
* duplicate logs,
* irrelevant system messages,
* large raw datasets,
* full prediction arrays unless specifically needed as artifacts.

Memory should remain compact and operational.

---

# 23. Temporary vs Persistent Information

Differentiate temporary task state from reusable memory.

Temporary:

```text
currently loading model
run 2 of 3 executing
GPU memory currently at 80%
temporary file path
current progress percentage
```

Do not preserve these after task completion.

Persistent:

```text
validated dataset schema
metric definition
verified preprocessing
known checkpoint format
known dataset issue
canonical artifact convention
```

These may be retained.

---

# 24. Confidence in Memory

Each important memory entry should distinguish verified information from uncertain information.

Preferred states:

```text
verified
provisional
deprecated
```

Example:

```yaml
memory_status: verified
```

A provisional item must not be treated as ground truth.

---

# 25. Memory Provenance

Important memory entries should preserve their origin when possible.

Example:

```yaml
source:
  task_id: Q-014
  artifact: results/quantitative/Q-014/result.json
```

or:

```yaml
source:
  configuration: experiments/waterbirds.yaml
```

This allows future tasks to verify where a remembered fact came from.

---

# 26. Memory Update Rule

When new verified information conflicts with stored memory:

1. do not silently merge incompatible values;
2. determine whether the difference is caused by dataset version, configuration, or previous error;
3. preserve the new verified value;
4. mark outdated information as deprecated when appropriate.

Example:

```yaml
memory_status: deprecated
```

Do not continue using known-outdated mappings.

---

# 27. Dataset Isolation Rule

Memory must remain dataset-aware.

Never apply information such as:

```text
column mapping
class encoding
split encoding
group definition
normalization
```

from one dataset to another unless explicitly verified.

A memory entry should always contain enough context to identify its scope.

---

# 28. Model Isolation Rule

Likewise, model-specific settings must remain attached to the model or checkpoint they belong to.

Do not assume that two models share:

```text
class ordering
input normalization
decision threshold
architecture details
preprocessing
```

even when they were trained on the same dataset.

---

# 29. Hypothesis Memory

Previously tested hypotheses may be remembered using their identifier.

Example:

```yaml
hypothesis:
  id: H-007

  candidate:
    background_type

  quantitative_status:
    tested

  strongest_evidence:
    level: Q3
    direction: supports

  result_reference:
    task_id: Q-021
```

This helps avoid unnecessary duplicate analyses.

However, a hypothesis may need to be retested when:

```text
dataset changes
model changes
split changes
intervention changes
configuration changes
```

---

# 30. Negative Result Memory

Negative results are valuable and should be preserved.

Example:

```yaml
hypothesis:
  id: H-011

  evidence:
    level: Q2
    direction: contradicts
```

Do not discard contradictory evidence simply because it does not support shortcut detection.

Remembering negative results prevents repeated confirmation attempts.

---

# 31. Inconclusive Result Memory

Inconclusive analyses may also be preserved when they explain why stronger conclusions were impossible.

Example:

```yaml
hypothesis:
  id: H-013

  evidence:
    direction: inconclusive

  limitation:
    subgroup_sample_size_too_small
```

This can help Executive avoid repeatedly scheduling the same unresolvable test.

---

# 32. Memory Retrieval Priority

When beginning a task, consult memory in this order:

```text
1. Dataset schema
2. Variable mappings
3. Known task/hypothesis results
4. Model configuration
5. Metric definitions
6. Known failures
7. Cached artifacts
```

Use memory to reduce redundant work.

Do not allow memory to override current verified input.

Current task data has priority over stale memory.

---

# 33. Memory Verification Rule

Before using remembered information in a new experiment, ask:

```text
Does the dataset match?

Does the dataset version match?

Does the split match?

Does the variable definition match?

Does the model match?

Does the metric definition match?

Does the experiment configuration match?
```

If material conditions differ, do not directly reuse the remembered result.

---

# 34. Suggested Memory Structure

A practical memory structure may use:

```yaml
datasets: {}

variables: {}

models: {}

metrics: {}

procedures: {}

hypotheses: {}

validated_results: {}

known_issues: {}

recovery_patterns: {}

artifact_conventions: {}

protocols: {}
```

Keep the structure machine-readable where possible.

---

# 35. Example Memory Entry

```yaml
datasets:

  waterbirds:

    status: verified

    schema:
      target: y
      background: place
      split: split

    split_mapping:
      0: train
      1: validation
      2: test

metrics:

  cramers_v:
    status: verified
    role: categorical_association

hypotheses:

  H-003:

    candidate:
      shortcut: background
      target: label

    strongest_quantitative_evidence:
      level: Q2
      direction: supports

    source:
      task_id: Q-001
      artifact: results/quantitative/Q-001/result.json
```

---

# 36. Memory Integrity Rules

Always:

```text
store verified reusable information
preserve dataset/model scope
preserve provenance
distinguish cached from fresh results
keep negative results
deprecate outdated information
```

Never:

```text
store fabricated values
store unverified assumptions as facts
reuse incompatible cached results
erase contradictory evidence
convert temporary execution state into permanent memory
allow stale memory to override current validated inputs
```

---

# 37. Governing Principle

The Quantitative Agent memory exists to make future analysis:

> **more consistent, more reproducible, and less redundant.**

Remember verified quantitative knowledge.

Forget operational noise.

And never allow memory to replace validation.
