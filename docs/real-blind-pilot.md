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

1. `python -m shortcutminer pilot dry-run --config <pilot.yaml>` generates four positive
   challenges and two clean controls, freezes the protocol, workspaces, backends, models,
   seeds, and budgets, and prints the maximum call envelope. It makes no provider calls.
2. `python -m shortcutminer pilot preflight --config <pilot.yaml>` requires both configured
   real providers. The text endpoint returns strict JSON. The VLM receives `sample_a.png` and
   `sample_b.png` and must distinguish a central circle from a central square.
3. `python -m shortcutminer pilot canary --config <pilot.yaml>` runs one full real-provider
   workflow. The canary is excluded from scientific metrics.
4. `python -m shortcutminer pilot run --config <pilot.yaml>` runs the frozen matrix
   sequentially. `pilot resume` skips every already-attempted matrix cell, so completed paid
   calls are never repeated to improve a result.
5. `python -m shortcutminer pilot report --config <pilot.yaml>` regenerates `report.md` from
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

Phase status is `BLOCKED` when real configuration, preflight, canary, or the required successful
runs are missing. It is `FAIL` for a protocol breach such as hidden-truth leakage, mock use,
snapshot drift, unsafe resume, or invalid evaluation. It is `PASS` only after all twelve frozen
real-provider scientific runs complete under the protocol; PASS does not mean publication-ready
performance.

## Credential security and limitations

Put credentials only in the environment variables named by `api_key_env`; do not paste them into
configuration or prompts. Example names are `SHORTCUTMINER_TEXT_API_KEY` and
`SHORTCUTMINER_VLM_API_KEY`. Pilot artifacts are local and ignored by Git.

This small synthetic suite measures harness behavior and early discovery reliability. Future
semantic backends may include local VLMs and retrieval-augmented domain knowledge, but those are
future work and intentionally absent here.
