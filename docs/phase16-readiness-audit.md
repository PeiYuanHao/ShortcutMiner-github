> Historical phase record. Current deployment uses native Windows conda `shortcutminer`;
> follow `deployment.md` and `formal-experiment-environment.md` for current commands.

# Phase 16 readiness audit

This audit was performed before adding the Phase 16 pilot layer. “Current resolution” records
what the implementation now enforces.

1. **Real text API configuration:** `OpenAICompatibleModelConfig` (or a shared/per-Agent
   `ModelRuntimeConfigSet`) supplies provider adapter, model, base URL, credential environment
   variable, timeout, temperature, token limit, capabilities, retry policy, and call budget.
2. **Real VLM API configuration:** the same strict single-runtime model config is passed to
   `OpenAICompatibleSemanticVisualInspector` and must declare image-input capability.
3. **Two base URLs:** yes. Text and semantic clients are separately constructed.
4. **Two credential environment variables:** yes. Each runtime independently freezes only its
   `api_key_env` name; values are never serialized.
5. **Text provider neutrality:** the Agent/workflow contract is `AgentClient`; the available
   cloud adapter speaks an OpenAI-compatible wire protocol. Provider/model choice is config,
   not Agent logic or workflow topology.
6. **Semantic provider neutrality:** visual discovery consumes the
   `SemanticVisualInspector` protocol. The available cloud implementation is an
   OpenAI-compatible adapter and keeps observation-only safety boundaries.
7. **Real-provider marking:** concrete cloud clients and semantic inspectors declare
   `runtime_kind="real_provider"`. Invocation telemetry separately uses the functional kinds
   `text_agent` and `semantic_vlm`; semantic discovery artifacts retain `real_provider`.
8. **Runtime snapshot completeness:** the pre-existing `ModelRuntimeSnapshot` freezes provider,
   model, normalized endpoint, credential-variable name, temperature, token limit, timeout,
   retry/schema policy, call/token budgets, seed, and capabilities without a secret. The pilot
   now freezes independent text and VLM snapshots plus their stable combination identity.
9. **Real AgentClient in blind execution:** the production Runner already accepts real shared or
   per-Agent clients. The blind evaluator is post-workflow. The pilot now owns the bridge from a
   public blind challenge through the real workflow and then into the private evaluator.
10. **Repeated calls on resume:** completed LangGraph runs return persisted results without node
    calls, and the older repeated runner skips every stored repeat outcome. The pilot adds a
    frozen whole-matrix identity and never retries any attempted scientific matrix cell.
11. **CLI entry point:** `run` and `resume` already supported explicit real model config; semantic
    config was a separate option. Dedicated `pilot dry-run/preflight/canary/run/resume/report`
    commands are now present.
12. **Real-provider opt-in:** real calls were already explicit through the API CLI; offline mock
    was a separate command. Pilot commands now fail closed unless both configured runtimes pass
    real-provider checks and preflight.
13. **Mock/fake risk before the pilot layer:** programmatic callers could wire a mock client,
    mock semantic inspector, or injected fake transport into general workflows. That remains
    useful for offline tests. Scientific pilot execution now rejects non-real runtime markers,
    rejects injected non-network transport for the bundled adapter, uses the real CPU experiment
    backend, and scans scientific artifacts for forbidden mock/fake identities.
14. **Missing orchestration before Phase 16:** there was no formal pilot schema, dual-runtime
    freeze, provider-combination identity, two-image preflight, fixed six-challenge suite,
    pre-pilot cost envelope, real canary gate, global pilot budget, pilot-level resume record,
    aggregate failure taxonomy, or pilot report. These components now live in
    `shortcutminer/evaluation/pilot/`.
15. **Quantitative resource ownership:** provider-facing experiment drafts contain only
    scientific choices. Dataset-relative paths, identity, label/addressing columns, seeds,
    budgets, and model artifact references are injected from strict authoritative resources and
    validated against a frozen ten-experiment ownership matrix.
16. **Dependency-aware execution:** resolved experiment types are separated into currently
    available, dependency-pending, and unsupported sets. Deterministic resource and dependency
    failures do not consume a provider correction retry.
