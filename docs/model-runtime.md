# Model Runtime

ShortcutMiner keeps the five Agents behind the minimal provider-neutral `AgentClient` contract. The production adapter currently targets explicitly configured OpenAI-compatible `/chat/completions` endpoints; it does not claim compatibility with every provider or model.

## Configuration and isolation

A flat model configuration is shared by all five Agents by default. `ModelRuntimeConfigSet` adds optional per-Agent overrides for Identify, Executive, Quantitative, Qualitative, and Summarize without changing the graph topology. Credentials are referenced only by environment-variable name. API keys, authorization headers, full prompt payloads, image base64 data, and raw HTTP response envelopes are never written to model-call artifacts.

The runtime snapshot freezes provider, model, sanitized endpoint identity, temperature, output limit, timeout, retry policy, call budget, optional seed, and declared capabilities. Its stable hash changes when those settings change. A model seed is sent only when the configuration explicitly declares provider support; `temperature: 0` and a seed still do not promise bitwise provider determinism.

## Invocation records and telemetry

Every attempted provider invocation writes a distinct record under `runs/<run_id>/model_calls/`. Records include canonical SHA-256 request, prompt, payload, and response hashes; timestamps; monotonic latency; attempt and retry state; schema validity; failure metadata; and provider-reported usage. Token counts are never estimated. If any call omits usage, aggregate token totals are unavailable and the limitation is explicit.

`model_calls/telemetry.json` aggregates calls, successes, failures, transport retries, schema retries, rate limits, timeouts, provider 5xx failures, invalid JSON, schema failures, model budgets, latency, and per-Agent breakdowns. Runtime failures remain distinct from scientific outcomes: a timeout is not an inconclusive experiment, and a schema failure is not evidence against a shortcut.

## Failure and retry policy

HTTP 401, 403, 429, 408, 5xx, other provider rejections, DNS/connection errors, timeouts, malformed envelopes, invalid JSON, protocol-schema failures, budget exhaustion, and cancellation have separate statuses. Only configured rate limits, timeouts, and 5xx failures receive bounded transport retries. Successful HTTP responses with invalid JSON or invalid Agent protocols receive a separately bounded schema retry. Every retry is a real counted call with its own latency and usage record.

Both the Tool Runtime budget and the ModelCallBudget constrain a tool loop. Exceeding either boundary is an explicit runtime failure and never a scientific negative result.

## CLI

Use `--model-config` with `run` or API `resume`. A flat file shares one client; a `shared` plus `per_agent` file selects explicit overrides. Use `--semantic-model-config` only for an image-capable endpoint. Network calls remain opt-in, and automated tests use fake transports unless the real-provider smoke environment flags are explicitly enabled.

## Value-free protocol rejection diagnostics

A failed invocation records `schema_diagnostics.issues`, including `error_type`
(the rejected rule) and `field_path`. Existing Pydantic issues remain capped at
8, with their total count preserved. Typed provider-contract failures now record
allowlisted rule names such as `invalid_metadata_field_selector` and bounded
schema paths such as `candidates.0.metadata_field_ref`. Other validator ValueError
or TypeError failures record only their type and `<root>`; unrecognized rule or
field metadata is replaced with generic markers. Exception text, rejected values,
raw model replies, reasoning content, and credentials are never copied into these
diagnostics. `MODEL_INVOCATION_FAILED` events carry the same safe projection.

These diagnostic additions change records and safe failure messages only.
Validation conditions, routing, budgets, and retry counts remain unchanged.
Other Agents retain their existing correction. Identify retries receive the
safe typed rule and field path as well as the authoritative contract below.

Read `runs/<run_id>/model_calls/<invocation_id>.json`, or find the corresponding
failure event in `events.jsonl`. A record may now show, for example:

```json
{"total_schema_issue_count": 1, "issues": [{"field_path": "candidates.0.metadata_field_ref", "error_type": "invalid_metadata_field_selector"}]}
```

This example illustrates the new format; it is not a diagnosis of an earlier run.
Historical failures with null diagnostics cannot be reconstructed from response
hashes. Earlier failed Waterbirds artifacts were removed at the user's request
on 2026-10-05; their diagnostics are not claimed to have been reconstructed.

## Identify output contract

`load_identify_workspace_prompt()` appends a compact output contract derived
deterministically from `IdentifyAgentDraft.model_json_schema()`. The extraction
resolves schema references and preserves object structure, required fields,
enum values, and applicable structural constraints. The executable protocol
remains the single source of truth; no validator or scientific rule is changed.

The Identify workspace documents the seven existing cue families, required
candidate fields (including `description`), and a generic schema-valid example.
Its taxonomy descriptions contain no dataset-specific answers.

For Identify alone, the existing schema-retry correction is followed by the same
derived contract and the instruction: "Correct only protocol conformance. Do not
change the scientific interpretation merely to satisfy validation." The original
scientific request remains intact. The correction does not include rejected
values, provider reasoning, or scientific answers. Other Agents retain their
existing retry prompts, and the configured retry count is unchanged.

Offline regression tests cover invalid families, missing descriptions, valid
objects, schema-derived prompt metadata, generic taxonomy, and preservation of
the original request during retry. No real-provider invocation is required.
Earlier failed trials were removed at the user's request on 2026-10-05.
This hardening did not reconstruct missing historical diagnostics.

Identify invocations also append the current allowed metadata selector names,
derived from the same `_public_metadata_fields()` function used by validation.
These are quoted addressing identifiers, not scientific answers. Non-metadata
cues must use null; non-null selectors must be unique, authorized, and supported
by metadata evidence. No selector is repaired or inferred in Python.
