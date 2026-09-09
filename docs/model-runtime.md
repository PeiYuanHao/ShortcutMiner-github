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
