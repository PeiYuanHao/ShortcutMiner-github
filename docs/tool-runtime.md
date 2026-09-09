# ShortcutMiner Tool Runtime

Phase 7 adds a provider-independent tool boundary without changing the five-Agent
schemas or LangGraph topology.

## Execution boundary

`ToolCall` and `ToolResult` are strict Pydantic contracts. Executors optionally
receive a `ToolRuntime`; runtime objects remain closure dependencies and are never
written into LangGraph State or checkpoints. When enabled, the Agent payload adds:

```json
{
  "tool_runtime": {
    "tools": [],
    "results": [],
    "remaining_calls": 8
  }
}
```

An Agent requests work by returning `{"tool_calls": [ToolCall]}`. ShortcutMiner
validates identity, permission and arguments, executes through `ToolRegistry`,
validates and serializes the output, persists a `ToolResult`, and invokes the Agent
again. A normal Agent result ends the loop and continues through its existing
schema validator. `max_tool_calls` and per-call timeout bound the loop. The hardened
model runtime also applies total, per-Agent, transport-retry, schema-retry, and
optional provider-reported token budgets, so both sides of the loop are bounded.

The core protocol contains no OpenAI, LangChain, AutoGen, CrewAI, or provider SDK
types. A future provider-native function-calling adapter belongs in the model client.

## Registered tools and permissions

| Tool | Identify | Executive | Quantitative | Qualitative | Summarize |
|---|---:|---:|---:|---:|---:|
| `inspect_tool_capabilities` | yes | yes | yes | yes | yes |
| `inspect_dataset` | yes | no | yes | yes | no |
| `inspect_metadata` | yes | no | yes | yes | no |
| `get_column_distribution` | yes | no | yes | yes | no |
| `select_samples` | yes | no | yes | yes | no |
| `inspect_sample_metadata` | yes | no | yes | yes | no |
| `inspect_sample` | no | no | no | yes | no |
| `read_artifact` | yes | yes | yes | yes | yes |
| `read_run_trace` | no | yes | yes | yes | yes |
| `run_experiment` | no | no | yes | no | no |

This matrix is enforced by the registry, not trusted model output.

## Reused implementations

Visual Discovery is prepared before Identify invocation and is not a new raw-image Agent tool. Identify's registered Tool Runtime permissions remain unchanged. The control layer places the strict, bounded `VisualDiscoveryProfile` in `discovery_context`; Identify interprets that precomputed evidence and cannot reopen images, expand the selected set, or bypass the registry. The image descriptor implementation is shared with Qualitative so identical samples and options use the same definitions.

The optional `SemanticVisualInspector` is a separately injected control-layer capability, not a generally callable tool. It supports a deterministic mock and an explicitly configured provider-backed implementation; no network call occurs by default. Its model-call and image budgets are independent of Tool Runtime permissions.

- `inspect_dataset` wraps `tools.common.dataset_inspector.inspect_dataset`.
- `inspect_metadata` wraps `tools.common.metadata_profiler.profile_metadata`.
- `get_column_distribution` reuses the bounded metadata loader.
- `select_samples` wraps `tools.common.sample_selector.select_samples`.
- `inspect_sample_metadata` wraps the existing qualitative evidence manifest.
- `inspect_sample` resolves one already approved sample and computes only requested
  allowlisted local descriptors within file, pixel, frame, and text budgets.
- `read_artifact` and `read_run_trace` use approved run-relative storage paths.
- `run_experiment` delegates to the `ExperimentExecutor` protocol. The mock backend
  remains available for tests; configured local live runs use the CPU-first
  `RealExperimentExecutor`.

## Real quantitative experiments

`run_experiment.parameters` is a union of strict, experiment-specific schemas;
unknown fields are forbidden. In particular, code, command, and script parameters
are not accepted. Quantitative explicitly supplies resolved metadata fields—candidate
display names are never fuzzily mapped to columns.

The static backend registry supports:

| Experiment type | Current controlled implementation |
|---|---|
| `statistical_check` | categorical contingency/chi-square/Cramér's V, numeric-group tests/eta-squared, numeric Pearson and Spearman association, optional bounded permutation |
| `subgroup_evaluation` | label distributions, or overall/per-group prediction accuracy when explicit predictions are supplied |
| `model_evaluation` | bounded logistic-regression probe with explicit or deterministic stratified holdout |
| `intervention` | seeded metadata shuffle, optionally within explicit splits |
| `ablation` | out-of-sample probe comparison after removing one explicit metadata feature |
| `metric_computation` | allowlisted accuracy or balanced accuracy |

Budgets cap rows, categories, permutations, probe input fields, and solver
iterations. Statistical schemas also cap percentile-bootstrap iterations at 1,000,
use a configurable confidence level in `[0.8, 1.0)`, and record the seed and
resampling unit. Association effects use sample-row resampling; subgroup and
prediction metrics use prediction rows; probes bootstrap held-out predictions
without retraining. Metrics for which the required paired predictions are not
retained report uncertainty as unavailable rather than silently retraining.
Sparse, constant, single-class, and insufficient samples produce a
successful `outcome: inconclusive`; parse/access failures produce a failed
ToolResult. The backend never assigns evidence levels/directions or applies a
threshold that confirms a shortcut.

## File and artifact safety

Dataset tools receive their root from `ToolExecutionContext`; it is not selected by
the Agent. Metadata and sample arguments are safe relative paths resolved beneath
that root. Artifact reads must both resolve beneath `runs_root/run_id` and appear in
the runtime's explicit allowlist. Absolute paths, drive-qualified paths, traversal,
unapproved artifacts, and unsupported formats are rejected.

`inspect_sample_metadata` exposes only file metadata/signature status.
`inspect_sample` is separately restricted to Qualitative and returns deterministic
image descriptors or bounded text facts for an already control-layer-approved
reference. Supported descriptor names are `dimensions`, `intensity`, `rgb_stats`,
`entropy`, `edge_density`, and `spatial_stats`; unknown names fail schema validation.
It provides no semantic vision, OCR, arbitrary transformation, shortcut decision,
shell, dynamic import, or network capability.

Prepared qualitative artifacts live under
`qualitative/<task_id>/inspection.json` and optional `contact_sheet.png`. They are
added to the validated QualitativeResult artifact references. Runtime approval is
dynamic and limited to the selected sample references and generated task artifacts.

## Trace and failures

Tool results are written to
`runs/<run_id>/tools/<agent_name>/<tool_call_id>.json`. Existing `events.jsonl`
receives `TOOL_STARTED`, `TOOL_COMPLETED`, `TOOL_FAILED`, or `TOOL_REJECTED`.
Unknown tools, permissions and invalid inputs raise their named boundary errors.
Execution exceptions and timeouts produce serializable failed `ToolResult` objects.
A scientifically inconclusive experiment remains a successful tool execution whose
typed output has `outcome: inconclusive`.

Real experiment evidence is additionally stored as
`runs/<run_id>/experiments/<task_id>/<experiment_type>-<parameter_hash>.json`.
The artifact contains bounded structured details such as contingency tables,
group summaries, and uncertainty contracts while the Tool output keeps scalar
metrics and exposes uncertainty through metadata. A completed matching
artifact is validated and reused on retry/resume rather than recomputed.
