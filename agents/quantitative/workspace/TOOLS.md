# Quantitative Agent Tools

## Current Invocation Boundary

The executable contract is QuantitativeRequest → QuantitativeResult, defined in
shortcutminer/protocol/quantitative.py. The payload contains request and, when
configured, a provider-neutral `tool_runtime` catalogue/result history. Permitted
tools are `inspect_dataset`, `inspect_metadata`, `get_column_distribution`,
`select_samples`, `inspect_sample_metadata`, `read_artifact`, `read_run_trace`,
`run_experiment`, and `inspect_tool_capabilities`. There is no Python, shell,
dynamic import, source-editing, package-installation or unrestricted network tool.

Use only explicitly supplied, traceable evidence. Without actual quantitative
measurements, return blocked with Q0, not_tested, empty metrics, and explicit
limitations/recommended_followup. Never manufacture measurements from hypotheses.
The executor validates and persists the result and events. Do not call other
agents, update workspace/state, retry tasks or control checkpoint/resume.

The method guidance and numerical examples below are scientific reference material,
not real run results or alternative output schemas. Apply a method only through a
listed authorized tool with the necessary evidence. Return
`{"tool_calls": [ToolCall, ...]}` to request execution, then use the structured
ToolResult history to form the normal QuantitativeResult. Recommendations do not
authorize execution. Exact output fields are defined in AGENTS.md.

## 1. Purpose

This file defines how the Quantitative Agent should use available tools during shortcut auditing.

The Quantitative Agent uses tools to:

* inspect structured dataset metadata,
* execute statistical analyses,
* evaluate model predictions,
* run controlled quantitative experiments,
* generate reproducible metrics,
* save machine-readable artifacts,
* and verify quantitative results.

Tools are execution mechanisms.

They do not replace experimental reasoning.

The governing rule is:

> **Choose the simplest valid tool that can test the assigned hypothesis reproducibly.**

---

# 2. Tool Philosophy

The Quantitative Agent should not use tools merely because they are available.

Every tool call must serve a defined analytical purpose.

Before using a tool, determine:

```text
What hypothesis is being tested?

What information is required?

What operation is necessary?

What output is expected?

How will the result be validated?
```

Avoid unnecessary tool chains.

Prefer direct and reproducible execution.

---

# 3. Tool Categories

Current callable tools are exactly the entries supplied in `payload.tool_runtime.tools`; the list above is the maximum Quantitative permission set. The method catalogue does not grant any additional filesystem, Python, provider, network, package-installation or artifact-writing access. Experiments must use `run_experiment`, which delegates to the configured ExperimentExecutor.

---

# 4. Workspace File Access

The agent may inspect files that are relevant to the assigned quantitative task.

Typical files include:

```text
CSV
JSON
JSONL
YAML
Parquet
NumPy arrays
PyTorch tensors
prediction files
experiment configurations
model checkpoints
metadata tables
```

Before reading a large file, determine whether the entire file is actually required.

Prefer targeted inspection when possible.

---

# 5. Dataset Inspection

Dataset inspection tools should be used to understand the structure of the data before quantitative analysis.

Typical checks include:

```text
column names
data types
number of rows
missing values
unique values
class distribution
split distribution
group distribution
```

Example inspection objectives:

```text
Does the metadata contain the shortcut variable?

Which column stores the label?

How are train/validation/test splits encoded?

Are subgroup identifiers available?
```

Never infer these properties without checking when the files are available.

---

# 6. Metadata Analysis

Metadata is often the cheapest source of shortcut evidence.

Prefer metadata-based analysis before expensive model execution when the hypothesis can be evaluated directly from annotations.

Possible operations include:

```text
groupby
value_counts
crosstab
conditional frequency
contingency table
correlation
mutual information
Cramér's V
chi-square
odds ratio
```

Example:

```text
label × background
```

may be tested without loading image tensors or a neural network.

---

# 7. Python Execution

Python is the default environment for quantitative computation when available.

Use Python for:

```text
data loading
data validation
statistical tests
metric computation
subgroup analysis
model evaluation
controlled experiments
result aggregation
artifact generation
```

Preferred libraries may include:

```text
numpy
pandas
scipy
scikit-learn
torch
torchvision
statsmodels
```

Use only libraries required for the task.

Do not add heavy dependencies unnecessarily.

---

# 8. Reusable Code First

If the repository already contains a validated implementation of a metric or experiment, prefer reusing it rather than creating a parallel implementation.

Before writing new code, check for relevant modules such as:

```text
metrics/
analysis/
evaluation/
experiments/
utils/
```

Reuse reduces metric inconsistency.

However, do not reuse code blindly.

Verify that its implementation matches the current quantitative definition.

---

# 9. Statistical Tool Selection

Select statistical procedures according to the variable types.

## Categorical vs Categorical

Possible tools:

```text
contingency table
chi-square test
Cramér's V
mutual information
odds ratio
```

## Continuous vs Categorical

Possible tools:

```text
group mean
group median
effect size
t-test
Mann-Whitney U
ANOVA
Kruskal-Wallis
AUROC
```

## Continuous vs Continuous

Possible tools:

```text
Pearson correlation
Spearman correlation
regression
distribution comparison
```

Do not use a statistical method solely because it is familiar.

Use a method appropriate to the data and hypothesis.

---

# 10. Cramér's V

For categorical shortcut-label association, Cramér's V may be used when appropriate.

The implementation should be based on a contingency table.

Record:

```text
variables
sample count
contingency table
chi-square statistic
Cramér's V
```

If bias correction is used, record that explicitly.

Do not silently switch between corrected and uncorrected definitions across experiments.

---

# 11. Mutual Information

Mutual information may be used to measure dependence between variables.

When using it, record:

```text
variable encoding
normalization if applied
estimator
sample count
```

Do not compare mutual-information values across incompatible estimation procedures without disclosure.

---

# 12. Chi-Square Tests

Before performing a chi-square test, check whether assumptions are reasonably satisfied.

Inspect:

```text
contingency table
expected cell counts
sample size
```

If assumptions are seriously violated, report the limitation or select another valid procedure.

Do not mechanically report a p-value without checking the table.

---

# 13. Model Loading Tools

Load a model only when model-level evidence is required.

Before model loading, verify:

```text
architecture
checkpoint path
class count
class ordering
input shape
normalization
device requirements
```

When loading a checkpoint, record the exact checkpoint used.

Example:

```yaml
model:
  architecture: resnet18
  checkpoint: checkpoints/model_seed42.pt
```

Never silently substitute another checkpoint.

---

# 14. Device Selection

Model execution may use:

```text
CPU
CUDA GPU
other supported accelerators
```

Prefer available acceleration when it materially improves execution.

However, device selection must not change evaluation semantics.

Record device information when relevant to reproducibility.

Do not fail an otherwise small analysis simply because a GPU is unavailable if CPU execution is feasible within constraints.

---

# 15. Inference Mode

When evaluating trained models:

```text
use evaluation mode
disable gradient computation
apply the exact expected preprocessing
preserve class mapping
```

For PyTorch-style execution this usually means behavior equivalent to:

```text
model.eval()
no gradient tracking
```

Do not evaluate using training-mode behavior unless the experiment explicitly requires it.

---

# 16. Prediction Files

If predictions or logits are already available and verified, prefer using them over rerunning inference.

Possible reusable artifacts include:

```text
sample_id
ground_truth
prediction
confidence
logits
group
domain
seed
```

Before reuse, verify that predictions match:

```text
dataset
split
model
checkpoint
preprocessing
```

If valid:

```yaml
execution_source: cached_predictions
```

Do not pretend cached predictions were newly generated.

---

# 17. Subgroup Evaluation

Use subgroup tools to partition evaluation samples according to the assigned hypothesis.

Possible subgroup definitions include:

```text
shortcut aligned
shortcut conflicting
domain
hospital
environment
class
background
acquisition source
```

For every subgroup metric, record:

```text
group name
sample count
metric value
```

Example:

```yaml
aligned:
  n: 2200
  accuracy: 0.94

conflicting:
  n: 180
  accuracy: 0.61
```

Never report subgroup performance without subgroup counts.

---

# 18. Worst-Group Evaluation

When relevant, compute worst-group performance.

Define the groups explicitly.

Do not report:

```text
worst-group accuracy
```

without specifying the grouping rule.

Example:

```text
groups = label × background
```

must be recorded alongside the metric.

---

# 19. Distribution-Shift Tools

Use distribution comparison tools when investigating domain or split shifts.

Possible comparisons include:

```text
train vs validation
train vs test
ID vs OOD
hospital A vs hospital B
domain 0 vs domain 1
```

Possible metrics include:

```text
conditional frequency difference
total variation distance
Jensen-Shannon divergence
Wasserstein distance
class distribution difference
shortcut prevalence change
```

Use only metrics appropriate to the data type.

---

# 20. Intervention Tools

Interventions should be used only when the task explicitly requires controlled manipulation.

Examples include:

```text
cue removal
cue randomization
background replacement
context swap
color modification
position randomization
ROI-preserving transformation
```

Before intervention execution, preserve:

```text
original sample identifier
intervention type
intervention parameters
random seed
output path
```

The original data must remain unchanged.

---

# 21. Intervention Pairing

Controlled interventions should preserve sample pairing.

Whenever possible, compare:

```text
original sample
        ↕
intervened version of same sample
```

This enables metrics such as:

```text
prediction flip
confidence delta
logit delta
```

Avoid comparing unrelated samples when a paired experiment is possible.

---

# 22. Randomization Tools

When using randomized transformations:

```text
set random seed
record random seed
preserve sampled parameters
```

For example:

```yaml
intervention:
  type: background_randomization
  seed: 42
```

Repeated experiments should use explicitly defined seeds.

Do not rely on uncontrolled randomness.

---

# 23. Multi-Seed Execution

When repeated trials are requested, use the specified seeds.

Store individual-run outputs before aggregation.

Example:

```text
seed_42
seed_3407
seed_2026
```

Then compute:

```text
mean
standard deviation
confidence interval if required
```

Do not discard individual seed results after aggregation unless storage policy requires it.

---

# 24. Metric Validation

Every metric must pass basic sanity checks.

Examples:

```text
0 <= accuracy <= 1
0 <= AUROC <= 1
0 <= flip_rate <= 1
sample_count >= 0
probability sum approximately valid
```

Also check:

```text
NaN
Inf
missing groups
empty arrays
shape mismatch
duplicate sample identifiers
```

A computation that returns a number is not automatically valid.

---

# 25. AUROC Safety

AUROC requires appropriate label diversity.

Before computing AUROC, ensure the evaluated subset contains the required classes.

If a subgroup contains only one class:

```text
AUROC is undefined
```

Report the condition.

Do not replace it with zero or another fabricated value.

---

# 26. Confidence Intervals

When confidence intervals are required, use an explicitly defined method.

Possible methods include:

```text
bootstrap
normal approximation
Wilson interval
```

Record:

```text
method
number of resamples
confidence level
seed
```

Do not mix interval procedures across comparable experiments without documentation.

---

# 27. Bootstrap Tools

When bootstrap estimation is used:

```text
set seed
record number of resamples
preserve sampling unit
```

If data contains grouped or paired observations, choose the bootstrap unit carefully.

Do not automatically resample individual rows when this would break the structure of the data.

---

# 28. Visualization Tools

Figures are optional supporting artifacts.

Use visualization only when it improves interpretation or downstream reporting.

Possible quantitative figures include:

```text
bar charts
group performance plots
distribution plots
confidence interval plots
scatter plots
intervention effect plots
```

Do not substitute plots for machine-readable metrics.

Every important numerical result should still be saved separately.

---

# 29. Table Generation

Prefer tables for structured quantitative evidence.

Useful tables include:

```text
contingency tables
subgroup metrics
split comparisons
multi-seed results
intervention results
```

Save reusable tables as formats such as:

```text
CSV
JSON
Parquet
```

rather than only as rendered images.

---

# 30. Artifact Writing

Artifact persistence belongs to the executor and Tool Runtime. Return QuantitativeResult; reference only supplied task inputs or artifact paths returned by successful ToolResult objects. Do not choose a new results directory or claim an unreported artifact was written.

---

# 31. Result JSON

The stored result.json is the executor-persisted QuantitativeResult with the exact envelope and payload in AGENTS.md. There is no separate result schema. Numerical evidence belongs in flat payload.metrics, explanatory context in evidence_summary/limitations, and follow-up in a list of strings.

---

# 32. Metrics JSON

`metrics.json` should contain exact numerical outputs in machine-readable form.

Example:

```json
{
  "cramers_v": 0.8673,
  "aligned_accuracy": 0.9421,
  "conflicting_accuracy": 0.6178,
  "accuracy_gap": 0.3243
}
```

Avoid embedding formatted strings such as:

```text
"94.21%"
```

when the raw numerical value can be stored.

---

# 33. Logging

Logs should capture information useful for debugging and reproducibility.

Useful log content includes:

```text
task start
configuration loaded
dataset loaded
model loaded
number of samples
analysis stages
warnings
errors
artifact paths
completion status
```

Do not use logs as the primary result format.

---

# 34. Command-Line Tools

Repository scripts may be invoked through command-line execution when appropriate.

Preferred pattern:

```text
python script.py --config ...
```

rather than manually editing constants before each run.

CLI-driven experiments improve reproducibility.

When possible, experiment parameters should come from explicit configuration or arguments.

---

# 35. Configuration Files

Prefer configuration-driven execution for repeated experiments.

Typical configuration files:

```text
YAML
JSON
TOML
```

Example:

```yaml
dataset:
  name: user_dataset
  split: train

analysis:
  metric: cramers_v

seed: 42
```

Avoid scattering experiment parameters across source code.

---

# 36. Configuration Validation

Before execution, verify configuration values.

Check:

```text
required keys
valid paths
valid split names
supported metrics
valid seeds
model compatibility
```

Fail early when configuration is invalid.

---

# 37. Repository Safety

Do not modify unrelated repository files while executing a quantitative task.

Avoid:

```text
changing agent workspace files
rewriting global configuration unexpectedly
deleting existing results
editing source datasets
```

Only modify files required by the assigned task.

---

# 38. Dataset Protection

Original datasets are read-only by default.

Never:

```text
overwrite source images
rewrite original metadata
change labels
move dataset files
delete samples
```

If transformed data is required, store it separately.

Example:

```text
data/
  raw/
  processed/
```

or another project-defined structure.

---

# 39. Checkpoint Protection

Model checkpoints are read-only evidence artifacts.

Do not overwrite them during evaluation.

If a converted checkpoint is required, save a separate derived copy with clear provenance.

---

# 40. Path Handling

Avoid hard-coded machine-specific paths when repository-relative paths are available.

Prefer:

```text
data/...
results/...
checkpoints/...
```

over:

```text
E:\...
/home/user/...
```

inside reusable project code.

Machine-specific root paths may come from environment configuration.

---

# 41. Environment Variables

The control layer resolves environment/configuration before invocation. The Agent must not read environment variables, credentials or machine-local paths; use the supplied request.

---

# 42. Dependency Discipline

No installation tools are available in the current invocation. If execution requires an unavailable dependency, report it in limitations/recommended_followup. Dependency changes require a separate authorized engineering task.

---

# 43. Error Handling

Tool failures must be converted into structured quantitative failures when relevant.

Examples:

```text
file missing
invalid schema
out-of-memory
checkpoint mismatch
unsupported format
statistical test failure
```

Do not return raw stack traces as the final agent response.

Capture diagnostic details in logs when useful.

---

# 44. GPU Out-of-Memory Handling

If model inference fails due to GPU memory:

Possible valid recovery actions include:

```text
reduce batch size
disable unnecessary gradient storage
use mixed precision if validated
use CPU if feasible
```

Do not change:

```text
input resolution
model architecture
evaluation subset
```

without explicit disclosure, because these may alter the experiment.

---

# 45. Runtime Budget

Respect task runtime constraints.

If a task includes:

```yaml
max_runtime_seconds: 300
```

do not silently start a much larger experiment.

When possible, use cheap validation before expensive execution.

If the requested experiment cannot reasonably fit the budget, return a full
QuantitativeResult with status blocked, evidence_level Q0 and explicit limitations.
There is no incomplete status in this protocol.

---

# 46. Sampling

Sampling is allowed only when justified by the task or computational budget.

If sampling is used, record:

```text
sampling method
original sample count
sampled count
seed
stratification variables
```

Never silently replace full-dataset evaluation with a sample.

---

# 47. Temporary Files

Temporary execution artifacts should be clearly separated from final evidence.

Temporary files may include:

```text
intermediate tensors
temporary transformed samples
temporary caches
```

Remove them when safe if they are not required for reproducibility.

Do not delete final evidence artifacts.

---

# 48. Cache Policy

Caching may be used to reduce repeated computation.

Valid cache targets include:

```text
predictions
embeddings
processed metadata
intervention outputs
```

Cache keys should reflect relevant experiment conditions.

For example:

```text
dataset
split
checkpoint
preprocessing
seed
intervention
```

Do not reuse cache entries when these conditions differ materially.

---

# 49. Tool Failure vs Hypothesis Failure

A valid test that contradicts the hypothesis is status completed with evidence_direction contradicts and its actual Q1–Q4 level. An execution failure is failed/Q0; no available test is blocked/Q0/not_tested. Never confuse scientific negative evidence with missing computation.

---

# 50. Tool Output Validation

Never trust tool output blindly.

After execution, check:

```text
Did the expected number of rows appear?

Were all required groups present?

Did model inference cover every intended sample?

Do metrics fall in valid ranges?

Were artifact files successfully written?
```

Tool completion does not imply experiment validity.

---

# 51. Provenance Tracking

Every important tool-produced artifact should be traceable.

Preserve:

```text
task_id
candidate_id
dataset
split
model
seed
configuration
generation step
```

This provenance may be stored inside:

```text
result.json
artifact metadata
logs
```

---

# 52. Tool Selection Priority

When multiple approaches are possible, prefer approximately this order:

```text
1. Existing validated artifact
2. Existing repository utility
3. Direct metadata computation
4. Lightweight Python analysis
5. Cached model predictions
6. Fresh model inference
7. Controlled intervention
8. Expensive repeated experiment
```

This is a preference, not a rigid rule.

Use the method that provides sufficient evidence with the lowest unnecessary cost.

---

# 53. Cross-Agent Tool Boundaries

The Quantitative Agent should not use tools primarily to perform another agent's role.

Do not use tools to:

```text
perform unrestricted shortcut discovery
coordinate the entire agent workflow
generate the final audit report
perform broad qualitative example inspection
```

If a quantitative task reveals information relevant to another agent, report it through the structured result.

---

# 54. Executive Interaction

Do not send tool results or status updates to Executive. Return the canonical QuantitativeResult addressed to summarize; infrastructure validates it and retains correlation to the planned task.

---

# 55. Summarize Interaction

Summarize should receive stable machine-readable artifacts.

Prioritize:

```text
result.json
metrics.json
tables
```

over raw execution logs.

Figures may be included as supporting evidence.

---

# 56. Security and Secrets

Never expose:

```text
API keys
access tokens
passwords
private credentials
```

through logs or artifacts.

Use environment variables or approved secret-management mechanisms when required.

---

# 57. External Resources

Do not access external resources unless they are required and permitted by the task.

Possible legitimate purposes include:

```text
obtaining an explicitly configured dataset
retrieving a documented model checkpoint
installing an approved dependency
```

Avoid unnecessary network dependencies in reproducible experiments.

Prefer local, versioned resources when available.

---

# 58. Network Download Validation

If an external dataset or checkpoint is downloaded, verify when possible:

```text
source
version
filename
file size
checksum
```

Record provenance.

Do not assume that a file with the expected name is the expected version.

---

# 59. Tool Minimality

Avoid workflows such as:

```text
load full images
run neural network
extract embeddings
train classifier
generate plots
```

when the assigned question can be answered by:

```text
read metadata
compute contingency table
compute Cramér's V
```

Tool use should follow the evidence requirement, not technological complexity.

---

# 60. Tool Escalation Rule

Escalate tool usage only when stronger evidence is justified.

Example:

```text
metadata analysis shows association
        ↓
model predictions show subgroup gap
        ↓
controlled intervention measures sensitivity
```

Each stage should add information.

Do not escalate simply to make the audit appear more sophisticated.

---

# 61. Preferred Tool Workflow

A standard quantitative task should approximately follow:

```text
receive task
    ↓
read configuration
    ↓
inspect required files
    ↓
validate schema
    ↓
load relevant data
    ↓
execute selected metric/test
    ↓
validate outputs
    ↓
save artifacts
    ↓
return structured result
```

Model tasks add:

```text
load checkpoint
    ↓
validate preprocessing
    ↓
run inference
```

Intervention tasks add:

```text
generate controlled variant
    ↓
pair original and intervention
    ↓
compare predictions
```

---

# 62. Example: Dataset-Level Tool Flow

Hypothesis:

```text
Background is correlated with label.
```

Preferred flow:

```text
read metadata
    ↓
verify label/background columns
    ↓
filter requested split
    ↓
compute contingency table
    ↓
compute conditional frequencies
    ↓
compute Cramér's V
    ↓
validate sample counts
    ↓
save metrics
```

No model should be loaded unless additional model-level evidence is requested.

---

# 63. Example: Model-Level Tool Flow

Hypothesis:

```text
The classifier relies on background.
```

Possible flow:

```text
load predictions or model
    ↓
verify sample IDs
    ↓
construct aligned/conflicting groups
    ↓
compute group accuracy
    ↓
compute accuracy gap
    ↓
validate group counts
    ↓
save results
```

This supports model-dependence analysis.

---

# 64. Example: Intervention Tool Flow

Hypothesis:

```text
Changing background while preserving the central object
changes model predictions.
```

Possible flow:

```text
load paired samples
    ↓
validate ROI preservation
    ↓
run inference on originals
    ↓
run inference on interventions
    ↓
pair predictions by sample ID
    ↓
compute flip rate
    ↓
compute confidence delta
    ↓
save paired results
```

Do not interpret the experiment if pair matching fails.

---

# 65. Final Tool Rules

Always:

```text
use the simplest sufficient tool
validate files before analysis
reuse validated implementations
control randomness
record configuration
validate numerical outputs
preserve original data
save machine-readable results
track provenance
respect computational budgets
```

Never:

```text
run expensive experiments without purpose
fabricate tool outputs
hide execution failures
silently change evaluation settings
overwrite source datasets
overwrite checkpoints
reuse incompatible caches
report raw logs as evidence
trust metrics without validation
```

---

# 66. Governing Principle

The Quantitative Agent's tools exist to convert a defined shortcut hypothesis into reliable numerical evidence.

The final rule is:

> **Use tools to strengthen evidence, not to increase complexity. Every computation must be necessary, traceable, and reproducible.**

# 67. Model-Reliance Experiment Catalogue

Use `task_model_training` only when `retraining` is declared and `no_retraining` is absent. Use `task_model_evaluation` with an approved safe `model_reference`. Use `ood_evaluation` and `counterfactual_intervention` only when `checkpoint` and `intervention_inference` are available. The Agent supplies dataset columns, split/partition, budgets, a public Candidate-derived `target_cue`, an allowlisted transformation object, and an explicit label-preservation status. The backend owns safe image decoding, fitting, persistence, transformation, inference, pairing, metrics, and bootstrap computation.

The preferred deep sequence is train/load → ID evaluation → OOD evaluation → paired counterfactual evaluation. Each later request must use the model reference returned by the earlier tool result. Do not retrain for OOD or counterfactual work. Detailed predictions and pair rows remain in approved artifacts; keep the final `QuantitativeResult.metrics` flat.
