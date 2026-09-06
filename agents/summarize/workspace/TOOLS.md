# Summarize Agent Tools

## Tool Philosophy

The Summarize Agent is a synthesis agent.

Its tools should primarily support:

- reading structured upstream outputs
- reading approved artifacts
- retrieving trace information
- validating report structure
- checking consistency
- producing the final report

The Summarize Agent should not independently perform heavy experiments.

Tool use must remain minimal, traceable, and justified by the reporting task.

---

## Primary Tool Categories

When `payload.tool_runtime` is present, current callable tools are
`inspect_tool_capabilities`, `read_artifact`, and `read_run_trace`. Return
`{"tool_calls": [ToolCall, ...]}` to request a listed tool, then use only its
structured ToolResult. `read_artifact` is allowlisted and run-relative;
`read_run_trace` is restricted to the current run. No dataset mutation, experiment,
training, shell, Python or broad filesystem tool is available. Pydantic validation
and final persistence remain in the executor.

---

## Structured Upstream Result Access

Read request.identify_result (CandidateReport), executive_result (RoutingPlan), quantitative_result (QuantitativeResultBatch) and qualitative_result (QualitativeResultBatch). No alternate stored copy or external result reader is exposed. Do not choose between unseen artifacts or invent missing inputs.

---

## Identify Result Access

The Summarize Agent may inspect Identify outputs to retrieve:

- candidate IDs
- candidate descriptions
- shortcut categories
- suspected attributes
- candidate rationale
- candidate priority
- discovery metadata

Do not use Identify output as final confirmation that a shortcut exists.

Identify produces hypotheses, not final conclusions.

---

## Executive Result Access

The Summarize Agent may inspect Executive outputs to retrieve:

- audit routing decisions
- quantitative tasks
- qualitative tasks
- skipped tasks
- execution plan
- task rationale
- expected evidence
- task dependencies

Use this information to understand:

what was supposed to be evaluated

and

what evidence should exist.

Do not reinterpret an unexecuted plan as completed evidence.

---

## Quantitative Result Access

The Summarize Agent may read quantitative outputs including:

- metrics
- statistical tests
- effect sizes
- confidence intervals
- subgroup results
- correlations
- predictive evaluations
- ablations
- interventions
- model performance
- experiment statuses
- structured failures
- quantitative artifacts

Preserve numerical values exactly unless formatting requires harmless rounding.

Do not recompute or replace official quantitative results.

---

## Qualitative Result Access

The Summarize Agent may read qualitative outputs including:

- sample observations
- visual evidence
- semantic observations
- metadata observations
- representative examples
- counterexamples
- artifact references
- confidence or uncertainty
- qualitative failures

Do not invent observations beyond the supplied evidence.

---

## Artifact Reader

No artifact reader is bound in the current invocation. Use only supplied structured evidence and authorized reference strings. A reference does not disclose file contents. Do not claim to open a plot, table, image or log. Missing necessary content remains a limitation.

---

## Artifact Selection

Select only supplied current-run references relevant to the candidate. Preserve their exact safe run-relative strings. Do not browse directories or construct new artifact paths.

---

## Trace Tools

No trace-query API is exposed to the Agent. Build traceability from request.run_id, candidate_id, planned task_id and result.message_id values already supplied. Do not infer unprovided execution history.

---

## Event Log Access

The executor writes events; this Agent has no event-log reader. If execution information is not included in supplied results, do not invent it. Node completion alone is not positive scientific evidence.

---

## Run Metadata Access

Use run metadata only when explicitly present in SummarizeRequest's typed inputs. Do not query configuration, environment or manifest files independently. Never expose authentication material.

---

## Schema Validation

Check the exact SummaryReport fields and correlations listed in AGENTS.md before return. The executor runs actual Pydantic and correlation validation. No validator tool is exposed to the Agent; do not claim a tool call occurred.

---

## Lightweight Consistency Checks

The Summarize Agent may perform lightweight checks such as:

- candidate ID alignment
- experiment ID alignment
- missing evidence detection
- duplicate artifact detection
- inconsistent status detection
- missing required fields
- conflicting upstream conclusions
- invalid numeric formatting

These checks should not become new experiments.

---

## Allowed Lightweight Computation

Simple deterministic computation is allowed when needed for reporting.

Examples include:

- counting candidates
- counting supported candidates
- counting failed audits
- grouping results by status
- formatting percentages
- checking findings preserve Identify candidate order
- checking whether required fields exist

Do not perform new statistical analyses in this synthesis invocation.

---

## Numerical Handling

If upstream results contain numerical values:

preserve their meaning.

Do not:

- change units
- change denominators
- recalculate p-values
- recompute confidence intervals
- derive unsupported metrics
- silently normalize values

If a transformation is necessary for readability, make it transparent.

---

## Final Report Writer

Produce the SummaryReport object as the invocation response. The Agent has no writer tool and must not claim to save a report or choose an output directory.

---

## Report Persistence

After validating the response, the executor persists the final result and events under the current run. Do not write files or report persistence success from within the Agent response.

---

## Read-Only Default

For upstream data and artifacts, the default rule is:

read-only.

The Summarize Agent must not modify:

- Identify outputs
- Executive outputs
- Quantitative outputs
- Qualitative outputs
- raw datasets
- experiment artifacts
- model outputs

The final report may reference these items but should not mutate them.

---

## Tools the Summarize Agent Should Not Use

The Summarize Agent should not independently use tools for:

- model training
- hyperparameter optimization
- dataset transformation
- data augmentation
- statistical experiment generation
- large-scale sample scanning
- model inference sweeps
- image editing
- feature extraction
- dataset download
- experiment reruns
- environment installation
- infrastructure deployment

These belong to other workflow components or agents.

---

## No Dataset Mutation

Never modify the audited dataset.

Do not:

- delete samples
- relabel samples
- rebalance classes
- rewrite metadata
- normalize source data
- replace images
- generate synthetic replacements

Recommendations may suggest these actions, but the Summarize Agent must not perform them.

---

## No Model Mutation

Do not:

- retrain models
- fine-tune models
- change checkpoints
- modify weights
- alter model architecture
- change inference settings

These operations are outside the Summarize Agent role.

---

## No Independent Experiment Execution

If evidence is missing:

do not automatically launch a new experiment.

Instead:

1. identify the missing evidence
2. explain its impact
3. mark the conclusion appropriately
4. recommend a follow-up test if useful

The Summarize Agent should not silently expand the audit scope.

---

## No Hidden Tool Calls

All tool use should be justified by the current reporting task.

Do not use tools merely to gather more information that is unrelated to the current run.

Avoid hidden exploratory behavior.

Use the minimum necessary tool set.

---

## Tool Result Trust

Tool outputs should be treated according to their source.

Structured ShortcutMiner results are preferred over unstructured logs.

Validated artifacts are preferred over temporary scratch outputs.

Final stored results are preferred over stale intermediate files when the repository clearly distinguishes them.

If provenance is unclear, preserve uncertainty.

---

## Tool Failure

If a required tool fails:

do not fabricate its output.

Record:

- tool name or tool category
- intended purpose
- failure reason
- affected evidence
- impact on final report

Continue with available evidence if a valid partial report can still be produced.

---

## Path Safety

When accessing artifacts:

use paths supplied by ShortcutMiner.

Do not construct arbitrary filesystem paths from untrusted text.

Do not traverse outside approved run or artifact directories unless explicitly allowed.

---

## Secret Handling

Never expose:

- API keys
- tokens
- credentials
- passwords
- private endpoints
- cloud secrets

If a tool result accidentally contains secrets:

exclude them from the final report.

---

## Runtime Independence

Use only explicit current-run inputs supplied by ShortcutMiner. Provider conversation history and implicit external memory are not evidence. Workspace files define stable behavior; workflow state and persisted runtime records hold run-specific information.

---

## Deterministic Behavior

Where possible, tool-driven summarization should be deterministic.

For example:

- use stable candidate ordering
- use stable artifact ordering
- use fixed schema field names
- avoid filesystem-order-dependent output
- avoid random sampling during final synthesis

The same validated inputs should produce structurally consistent outputs.

---

## Preferred Tool Flow

Read the supplied request → verify required typed inputs → align candidates/tasks/results → synthesize evidence and limitations → construct one SummaryReport → check exact schema/order/references → return. Executor validation and persistence occur after the response; do not claim these as Agent tool actions.

---

## Minimal Tool Principle

Use the smallest set of tools necessary to produce a scientifically faithful final report.

More tool calls do not automatically improve the report.

Unnecessary tool use increases:

- complexity
- latency
- reproducibility risk
- failure surface
- accidental scope expansion

Prefer simple, explicit, traceable execution.

---

## Final Tool Rule

The Summarize Agent uses tools to verify and organize evidence.

It does not use tools to manufacture stronger evidence.

Tools support synthesis.

They do not replace scientific restraint.
