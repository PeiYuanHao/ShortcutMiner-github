# Real API blind development pilot

ShortcutMiner Phase 16 is a **DEVELOPMENT PILOT — NOT FINAL PAPER RESULTS**. It runs one
frozen cloud text runtime and one frozen cloud multimodal runtime through a six-challenge,
hidden-truth audit. The scientific question is whether the five Agents discover and validate
shortcuts without receiving the answer, while clean controls expose false confirmations.

The pre-implementation audit and its current resolutions are recorded in
[phase16-readiness-audit.md](phase16-readiness-audit.md).

## Scope and provider neutrality

The text runtime drives Identify, Executive, Quantitative, Qualitative, and Summarize. The
semantic VLM only makes bounded visible-content observations; it does not propose candidates,
assess shortcut probability, or make final claims. The two OpenAI-compatible runtime
configurations may use different providers, models, endpoints, and credential environment
variables. Provider and model identity live only in configuration and immutable snapshots.

Phase 16 is API-only. It does not install a local LLM/VLM, add retrieval, create a shortcut
knowledge base, fine-tune a model, run an ablation, or compare providers.

## Workflow

Use the native Windows conda environment `shortcutminer` from the repository root.
This historical development pilot is separate from formal E1–E5. Credentials for local
runs are read at runtime from Git-ignored `.env.formal` by API CLI commands. The
acceptance helper disables loading. Keep credentials out of
configuration, source, logs, and command arguments. Presence-only inspection:

```powershell
conda activate shortcutminer
python -m shortcutminer pilot credentials --config configs/experiments/real_blind_pilot.example.yaml
```

The result contains only `SET` or `MISSING`; it never includes credential content, fragments,
lengths, or hashes.

1. `python -m shortcutminer pilot dry-run --config <pilot.yaml>` generates four positive
   challenges and two clean controls, freezes the protocol, workspaces, backends, models,
   seeds, and budgets, and prints the maximum call envelope. It makes no provider calls.
2. `python -m shortcutminer pilot preflight --config <pilot.yaml>` requires both configured
   real providers. The text endpoint returns strict JSON. The VLM receives `sample_a.png` and
   `sample_b.png` and must distinguish a central circle from a central square.
3. `pilot legacy-task-negative-replay` selects the persisted `TASK-C002-QT` explicitly and proves
   that its malformed association task is rejected. `pilot historical-candidate-binding-replay`
   reads only the historical public CandidateReport and proves that the ambiguous
   `site_device_batch` cue remains unresolved with no fuzzy fallback or generated task.
   `pilot current-contract-task-offline` uses a separate synthetic public engineering fixture to
   exercise exact binding, deterministic Executive finalization, semantic validation,
   Quantitative plan compilation, `RealExperimentExecutor`, and quantitative finalization with
   zero provider calls.
4. `python -m shortcutminer pilot canary --config <pilot.yaml>` runs one full real-provider
   workflow. The canary is excluded from scientific metrics.
5. `python -m shortcutminer pilot run --config <pilot.yaml>` runs the frozen matrix
   sequentially. `pilot resume` skips every already-attempted matrix cell, so completed paid
   calls are never repeated to improve a result.
6. `python -m shortcutminer pilot report --config <pilot.yaml>` regenerates `report.md` from
   saved immutable results.

## Challenge suite and firewall

Public IDs are only `pilot_001` through `pilot_006`. The suite contains metadata association,
spatial appearance, acquisition style, and semantic background positives, plus two seeded
mixed visual/metadata clean controls. Positive correlations are strong but imperfect. Each
challenge stores pixels and public metadata below `public/dataset`; hidden truth remains below
`private` and is opened only by the evaluator after the workflow completes.

Production requests include the public dataset, task, deterministic visual profile, and bounded
VLM observations. They never contain the private path, truth family, target field, injection
strength, affected sample IDs, or expected answer. The runner does not insert model-reliance
experiments: Executive and Quantitative must select task-model, OOD, or counterfactual work.

## Freeze, budgets, and resume

`snapshot.json` contains the git commit, five workspace hashes, protocol and backend hashes,
secret-free text/VLM runtime snapshots, provider-combination hash, temperatures, retry/schema
policies, candidate and visual budgets, global call limits, challenge configuration, and seeds.
It never contains API keys, authorization headers, or image payloads. Any frozen-field or
workspace change requires a new pilot ID.

Executive selects scientific checks from a candidate-specific executable set. Exact metadata
scope is owned by `CandidateBindingResolver`, and the deterministic Executive finalizer constructs
the canonical task. A deterministic
`QuantitativeExperimentPlan` compiler converts that scope and the frozen execution resources into
canonical `ExperimentToolInput` steps. When execution is mandatory, the control layer executes
the first ready required check group through Tool Runtime and the real backend before the first
Quantitative provider call. The provider therefore starts with real measurements and interprets
evidence; it does not construct paths, field names, experiment identity, or dependency state.

The live provider schema for an optional follow-up `run_experiment` call contains only
`plan_step_id`. It may select a currently ready precompiled step, but cannot alter its canonical
input. Unknown, completed, and dependency-pending IDs are rejected with typed errors. The legacy
raw draft remains an internal/offline compatibility boundary and is not exposed in the real
provider path. The snapshot freezes the plan protocol/version/compiler, resolver, finalizer,
backend, workspace, and the complete ten-experiment classification.

Quantitative mutable authority is scoped to one `AnalysisTask`. Each task forks a clean
`ToolRuntime` from shared registry/backend infrastructure, while history, call IDs, generated
artifacts, dependency availability, and tool-call budget remain task-local. Explicit upstream
artifacts are distinct from current-task generated artifacts. Model-reliance steps resolve a model
only from a successful training result matching the current run, candidate, and task, or from a
model artifact explicitly assigned to that task. Sibling task results never enter the provider
payload. Resume restoration applies the same run/candidate/task filter.

Before another paid Canary, run the zero-provider real-backend isolation replay:

```powershell
python scripts/phase16_task_isolation_replay.py --root runs/phase16-task-isolation-offline-gate
```

It executes two independent training/evaluation tasks and one exact-field association task,
requires five successful `real_cpu_v1` results, verifies same-task model lineage, and reports zero
provider calls and zero provider-visible cross-task results.

Experiment availability is staged as `available_now`, `pending_dependency`, and `unsupported`.
Task-model evaluation remains pending until a successful training step has produced an approved
model artifact. OOD and counterfactual work without an explicit deterministic transformation is
classified as `needs_scientific_spec`, not guessed by the provider. Pending dependencies and
missing scientific specifications are not reported as schema failures and do not block an earlier
ready requested check.

Before a new full run, the runner reserves a conservative per-run envelope against global text
and VLM call caps. It stops creating new runs when the remaining envelope is insufficient,
preserves completed work, and marks the pilot incomplete. Provider and schema retries follow
only the frozen runtime policy. A failed provider is never replaced automatically.

## Metrics and failures

Saved per-run blind results retain Recall@1/3, exact and partial localization, candidate versus
final clean-control false positives, validation funnels, final assessment distributions, and
failure reasons. Model telemetry records provider/model, latency, retry and schema-retry calls,
request hashes, token usage when the provider reports it, and limitations when usage is absent.
Runtime failures are separated from scientific misses. Poor discovery performance is a result,
not a protocol failure and has no arbitrary pass threshold.

Structured-output retries append one deterministic serialization-only correction without
including the rejected response. The scientific payload hash remains unchanged, while each
materially different retry body receives its own request hash. Invocation records retain only
safe response diagnostics such as finish reason, content length, categorical first/last token
classes, and JSON parse status; raw response content and reasoning content are never persisted.

The initial Phase 16 development pilot uses the provider's non-thinking mode for every
structured Text Agent invocation. High-thinking mode exhausted the configured output budget
before producing protocol JSON during pre-pilot compatibility testing. This is a frozen runtime
compatibility choice, not a post-hoc scientific result adjustment. In non-thinking mode,
temperature `0.0` is sent as an effective request parameter, but it is not a guarantee of
deterministic output.

Phase status is `BLOCKED` when real configuration, preflight, canary, or the required successful
runs are missing. It is `FAIL` for a protocol breach such as hidden-truth leakage, mock use,
snapshot drift, unsafe resume, or invalid evaluation. It is `PASS` only after all twelve frozen
real-provider scientific runs complete under the protocol; PASS does not mean publication-ready
performance.

## Credential security and limitations

Put credentials only in the local Git-ignored `.env.formal` file under the environment variable
names selected by `api_key_env`; do not paste them into configuration, prompts, CLI arguments, or
shell history. The first configuration uses `SHORTCUTMINER_TEXT_API_KEY` and
`SHORTCUTMINER_VLM_API_KEY`. The API CLI loads them into its process only while the command runs;
no automatic credential loading occurs in acceptance. Pilot artifacts are local and ignored by Git.

This small synthetic suite measures harness behavior and early discovery reliability. Future
semantic backends may include local VLMs and retrieval-augmented domain knowledge, but those are
future work and intentionally absent here.
