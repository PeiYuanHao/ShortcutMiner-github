# Phase 17 audit runtime progress

`shortcutminer audit DATASET_PATH` enables progress by default. Use `--no-progress`
for CI/scripts/debugging; explicit `--progress` remains compatible. Audit resume
also enables progress by default unless `--no-progress` is supplied. Existing
`run`/`mock` commands remain opt-in with `--progress`.

```sh
shortcutminer audit tests/fixtures/generic_dataset_a
shortcutminer audit ./dataset --no-progress
shortcutminer resume my-audit --runtime openai-compatible --model-config configs/my-model.yaml
python -m shortcutminer mock --run-id progress-demo --progress
```

Progress goes to stderr; stdout retains started/completed JSON and normal exit
codes. TTY uses Rich Live. Non-TTY uses changed structured `[ShortcutMiner]` lines,
without ANSI escapes, animation, or Rich markup interpretation.

## One read-only monitoring projection

The existing `monitoring/progress.py`, `event_listener.py`, and `renderer.py` are
extended, not replaced by another subsystem. Sources are the current run's
`events.jsonl`, bounded DatasetProfile/visual discovery artifacts, original
CandidateReport, screening/capabilities caches, Executive routing tasks,
QuantitativeExperimentPlan, scoped result artifacts, ToolResult correlation
headers, exported workflow state, and committed LangGraph SQLite checkpoints.
Checkpoint connections use `mode=ro`; no saver setup or checkpoint writes occur.
Only fixed current-run paths and canonical task IDs select artifacts. Raw provider
responses/drafts, evaluator truth, source dataset files, and arbitrary artifact
references are not followed by the monitor.

The stable AuditStage projection contains audit_intake, dataset_profiling,
visual_discovery, identify, candidate_screening, capability_discovery, executive,
feasibility, quantitative, qualitative, evidence_calibration, summarize, and
audit_report. Statuses are pending/running/completed/skipped/failed. Exactly the
five existing Agents are shown separately, with observed calls and elapsed time.
No task in a validation branch means skipped, even if the workflow's branch
returns a completed empty batch. Visual discovery is skipped when not enabled or
when the profiled modality does not use that stage.

All generated candidates remain visible, including screened_out decisions and
categorical reasons. Promoted candidates display their actual AnalysisTask IDs,
requested checks, PlanStep IDs, experiment types, and correlated tool statuses.
PlanStep status is never inferred from list position. Screening counts show
generated/promoted/screened_out; no token, GPU, money, or cost-saving estimate is
invented. A screened-out hypothesis is not a negative scientific verdict.

## Observation events and protection

The control layer adds STAGE_STARTED/COMPLETED/SKIPPED/FAILED and
TASK_STARTED/COMPLETED/FAILED only where lifecycle coverage was missing. Existing
Agent/provider/tool events remain primary sources. Existing Tool events now also
carry ToolCall's already-authoritative candidate/task/PlanStep/check/experiment
identifiers; no ToolCall or result contract changed. The audit CLI exclusively
reserves a fresh run and stores the original user AuditRequest before intake,
then completes the same artifact after bounded intake. An existing run cannot
be overwritten. Intake failures retain their partial user input and events.

Observation emission is best effort and cannot change graph decisions, provider
retry, budgets, tool permissions, isolation, execution, evidence, or scientific
thresholds. No Agent workspace, prompt, or provider schema receives progress
fields or callbacks. The monitor itself performs no writes.

## Telemetry, weighting, and recovery

Provider counts independently reconcile started events with unique invocation
records, including retries but excluding budget rejections that issued no call.
Separate counters prevent a race where telemetry is visible before its matching
started event is polled. Text/VLM calls, VLM image count, prompt/completion/total
usage, latency, tool count, and elapsed wall time are displayed. Missing usage or
unobserved telemetry is N/A. Finalized usage covers persisted calls and can lag
in-flight calls. If a failed call lacks usage, cumulative tokens remain N/A after
successful resume. No values are estimated from prompt lengths or configuration.

Progress is explicitly coarse: intake/profile/visual 10%, Identify 15%,
screening/capabilities 10%, Executive 10%, validation 35%, calibration 10%, and
summary/report 10%. Validation uses actual planned task count and completed task
or PlanStep fractions across both branches. Adding tasks may change the fraction;
it is not an ETA, a scientific score, or an execution input. RUN_COMPLETED confirms
100%. Audit Report currently means persistence of existing final_result.json;
this phase does not introduce audit_report.json or audit_report.md exports.

A new listener replays history and reconstructs caches, tasks, plans, scoped
results, ToolResults, telemetry, and checkpoint values before following updates.
Completed stages and partial validation progress survive resume. Old ToolResults
cannot overwrite a newer retry's PlanStep status. Concurrent pending checkpoint
writes supply already-committed branch state. Monitoring does not re-run intake,
plan tasks, make model/tool calls, or rewrite damaged artifacts.

Failures retain the last pipeline/task/step state and artifact directory. Only
allowlisted failure categories are shown; raw errors, Authorization, credentials,
base64, provider bodies, and reasoning content are never rendered. CLI failure
output is sanitized and does not print stack traces. Partial JSONL lines wait for
a newline; malformed complete lines are ignored. Monitoring/render failures do
not change execution success. Elapsed wall time includes gaps between attempts;
unlogged crash time cannot be inferred.

## Offline acceptance

Use the repository .venv. The helper launches the installed audit CLI with only
`tests/fixtures/generic_dataset_a`, a loopback fake provider configured via the
existing model environment setting, and real CPU backends:

```powershell
.\.venv\Scripts\python.exe scripts/phase17_progress_acceptance.py
.\.venv\Scripts\python.exe -m pytest tests/test_progress_state.py tests/test_progress_renderer.py tests/test_audit_progress.py tests/test_monitoring.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

The helper preserves captured stdout/stderr alongside the audit artifacts and
reports fake versus real provider calls. It contacts no DeepSeek/Qwen service.
Screened-out, skipped-branch, multi-step, failure, checkpoint-only recovery,
telemetry race, secret isolation, control-field firewall, and read-only behavior
are covered by regression tests.
