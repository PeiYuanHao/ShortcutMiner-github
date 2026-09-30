# PHASE 16 SYSTEM CONTRACT CLOSURE: PASS

## Final cross-contract review (R1–R3)

The subsequent code review identified three additional readiness blockers. All three are closed in the same closure branch. No real Provider or Canary was called.

| ID | Closure | Explicit regression |
|---|---|---|
| R1 | Canonical `Candidate.metadata_field_ref` retains the context-validated exact public selector. `resolve_candidate_binding` uses it exclusively when present; historical candidates without it retain exact `cue.name` matching. Non-metadata candidates cannot carry a selector. | `test_identify_executor.py::test_exact_metadata_selector_survives_finalization_and_binding` |
| R2 | All seven loaded Quantitative workspace files assign evidence maturity and design strength to deterministic Python. The Provider emits only `QuantitativeAgentDraft` scientific fields and never selects a Q level. | `test_workspace_consistency.py::test_full_quantitative_prompt_respects_provider_draft_ownership` |
| R3 | All seven loaded Qualitative workspace files describe `QualitativeRequest` plus `QualitativeEvidenceContext` and optional current-task tools as input, `QualitativeAgentDraft` as Provider output, and canonical `QualitativeResult` as Python output. Sibling Quantitative results and artifacts are unavailable during the parallel branch. | `test_workspace_consistency.py::test_full_qualitative_prompt_respects_parallel_draft_and_tool_boundary` |

The full concatenated workspace prompts are checked semantically, including draft schema ownership and permitted tool names. Final cross-contract readiness blockers: **0**.

This is an offline contract qualification, not a Full Real Canary or real-provider readiness result. DeepSeek calls, Qwen calls, Canary runs, and formal Pilot runs in this closure: **0**. Baseline before closure: `8ab6fcec8951ab22ad4aac8ef5842f54a1af3eab`.

## Original blocker closure

The prior audit and `evaluation/phase16_system_contract_matrix.json` contained exactly seven readiness blockers. Their identities are retained below.

| ID | Violated invariant and readiness impact | Closure | Regression |
|---|---|---|---|
| B1 | Quantitative Provider owned evidence call IDs and Q maturity, so it could invent provenance or strength. | `protocol/quantitative.py` removes those draft fields; `tooling/agent_loop.py`, `tooling/protocol.py`, and `experiment/quantitative_executor.py` carry exact plan step/check/type through canonical calls and results and derive Q maturity from the deterministic experiment registry. | `test_quantitative_finalizer.py`, `test_quantitative_experiment_plan.py` |
| B2 | One result collapsed training and evaluation into one evidence record. | `QuantitativeMeasurement[]` retains each successful planned ToolResult; `evaluation/evidence_ledger.py` emits one quantitative EvidenceRecord per measurement. | `test_quantitative_finalizer.py::test_training_and_evaluation_remain_separate_canonical_measurements`, `test_phase16_three_candidate_chain.py` |
| B3 | Predictable Executive and Quantitative context errors were first rejected after Provider invocation returned. | All five executors run their context finalizer as a `provider_contract_validator` inside the shared bounded schema retry path. A machine-readable classification is in `evaluation/phase16_post_provider_failure_inventory.json`. | `test_phase16_five_agent_contract_retry.py`, executor invariant suites |
| B4 | A timed-out experiment worker could write authoritative output late. | `tooling/registry.py` runs the backend under a temporary runs root and publishes only validated completed artifacts before the deadline. Published paths are checked against both staging and run roots. | `test_tool_runtime.py::test_timed_out_experiment_cannot_publish_late_or_restore_authority`, backend security suites |
| B5 | Pilot-wide VLM images/tokens and live retry caps were incomplete. | `runtime/openai_compatible.py` shares live Pilot call/image/token counters across clients; `evaluation/pilot/runner.py` reconstructs them from persisted invocation records. Missing usage under a hard token cap blocks the next issued call. | `test_phase16_live_budget.py`, `test_real_blind_pilot.py` |
| B6 | A failed Full Real Canary lacked an immutable attempt and same-snapshot retry refusal. | `evaluation/pilot/runner.py` writes a running attempt before the first Provider call and a separate immutable terminal record, refuses another attempt for the same pilot/snapshot, and reserves `canary.json` for a completed gate. Snapshot identity includes dirty state, runtime/config hashes, and tool permission matrix hash. | `test_phase16_canary_lifecycle.py` |
| B7 | Adversarial, resume, failed-canary replay, and three-candidate production coverage was incomplete. | Added the three-candidate OpenAI-compatible fake-transport production chain, six graph interruption/resume boundaries, a mid Quantitative DAG failure/resume regression, budget and Canary adversarial tests, and the post-Provider inventory. Existing Qualitative artifact resume and timeout resume tests remain active. | `test_phase16_three_candidate_chain.py`, `test_phase16_resume_stages.py`, `test_quantitative_task_isolation.py`, `test_phase16_canary_lifecycle.py`, `test_tool_runtime.py` |

**B1–B7: CLOSED. Remaining real-provider readiness blockers found by this offline scan: 0.**

## Ownership and flow

The production graph is `Identify → Executive → Feasibility → (Quantitative ∥ Qualitative) → EvidenceCalibration → Summarize`. Summarize requires the already-calibrated EvidenceLedger. All five Agents emit bounded Provider drafts. Python owns canonical IDs, task inputs, plan steps, measurements, artifacts, Q maturity, limitations, and report envelopes. Static schema and context validation occur inside the generic invocation boundary; duplicate finalizer checks after it are deterministic assertions.

The quantitative identity chain is `AnalysisTask → QuantitativeExperimentPlan → PlanStep → ToolCall → ToolResult → QuantitativeMeasurement → EvidenceRecord`. Each measurement carries run/candidate/task, exact plan step and check, experiment type, call ID, backend identity, outcome, metrics, artifacts, model lineage where present, deterministic design strength, and uncertainty. Required ready steps are exhausted in dependency order before a completed draft can finalize. A resumed task reuses exact successful same-task steps and retries only incomplete steps. Provider interpretation remains scientific judgment; execution bookkeeping remains Python-owned.

Mutable ToolRuntime history, model/artifact authority, sample allowlists, and dependency recovery are task-scoped. The shared run tool counter is distinct from the Executive task cap and per-task tool cap. Only Executive and Summarize may read the run trace. Exact metadata binding is enforced; no fuzzy mapping or hidden-truth access is used in scientific execution.

The experiment timeout boundary holds writes in a temporary run root. A timeout returns a failed ToolResult and a later backend completion cannot publish to the authoritative run. Path checks reject symlink escape. Pilot budget accounting counts issued retries, actual VLM image inputs, known tokens, and persisted usage on resume. When `max_total_tokens` is set and an issued response lacks reliable usage, further calls are refused; unavailable usage is never treated as zero. Full Real Canary attempts have immutable running and terminal records under `canary_attempts/`; `canary.json` is only the completed gate for the frozen snapshot. No real Canary was run for this audit.

## Offline verification

The three-candidate workflow exercised DatasetProfiler, VisualDiscovery, Identify/Executive drafts and finalizers, binding, Feasibility, full Quantitative DAG, task-isolated Qualitative, EvidenceCalibration, and Summarize through `OpenAICompatibleClient` with a fake transport and `RealExperimentExecutor`. It validated SummaryReport, EvidenceLedger, exact measurement-to-record correlation, and no duplicate Provider execution on completed-run resume. The Phase 15 full-path smoke also exercises a fake semantic visual transport.

Resume coverage includes interruption before Executive, Feasibility, Quantitative, Qualitative, EvidenceCalibration, and Summarize; mid Quantitative training/evaluation failure; Qualitative artifact reuse; and late timeout output refusal. Adversarial coverage includes missing/wrong/duplicate plan provenance, skipped required steps, invented artifact claims, Provider-selected Q fields, inconclusive support, cross-task authority, budget retries, missing token usage, failed Canary immutability, and same-snapshot replay refusal.

Final gate results are recorded in `evaluation/phase16_system_contract_matrix.json`. The persistence sentinel uses only an offline fake transport and finds zero API keys, Authorization values, raw Provider bodies, reasoning content, or base64 image payloads in persisted files. `docker compose config` was not run. No real Provider or formal Pilot activity occurred.
