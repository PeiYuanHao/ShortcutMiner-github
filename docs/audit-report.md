# Audit report export (schema 1.0)

Every completed dataset audit automatically writes `audit_report.json`,
`audit_report.md` and `audit_report.technical.md` beside the existing
`final_result.json`. JSON is canonical; both Markdown views render that same
validated report. The default `audit_report.md` is a concise English reader-facing
view: conclusion counts, candidate status, actually completed checks, paired
accuracy changes and recorded uncertainty, plus material completion limitations.
It does not expand prediction arrays, per-image references or repeated limitations.
Common cue names have plain-English display labels; unknown names remain unchanged.
No new model call or scientific assessment is introduced. The CLI keeps its JSON stdout
events and prints the report locations to stderr, including with `--no-progress`.

The builder is a Python controller after the existing five Agent workflow.
EvidenceLedger and SummaryReport retain scientific authority. Findings and
evidence records are copied from finalized protocols, with no recalibration,
risk upgrades, or new shortcut verdicts.

## Canonical sections

| Section | Recorded source |
| --- | --- |
| `schema_version` | Export envelope version, `1.0` |
| `run` | RunRecord ID, status, persisted timestamps and interval |
| `dataset` | Completed AuditRequest, bounded profile and authorized capabilities |
| `runtime` | Persisted runtime snapshots, invocation accounting and telemetry |
| `discovery` | All original Identify hypotheses, unverified; absent confidence stays null |
| `screening` | Full deterministic decisions, reasons, counts, version and hashes |
| `validation` | RoutingPlan, AnalysisTasks, typed PlanSteps, feasibility and actual tool headers/parameters |
| `quantitative`, `qualitative` | Finalized branch result batches |
| `evidence` | Existing EvidenceLedger, including its fingerprint and provenance |
| `findings` | Existing SummaryReport payload and exact assessment vocabulary |
| `cost` | Recorded calls, images, tokens, retries, tool events and stage/task timings |
| `reproducibility` | Recorded run/sampling seeds, budgets, model snapshot IDs, screening/capability/evidence identities |
| `limitations` | Deduplicated upstream text with every source location |
| `artifacts` | Safe run-relative references, without embedded file contents |

`screened_out` is a controller decision, separate from `contradicted` or any
other SummaryReport assessment. The technical Markdown prominently explains:

> Screened-out candidates were not experimentally disproved; they were terminated before expensive validation under the configured screening policy.

An example shape (values abbreviated):

```json
{
  "schema_version": "1.0",
  "run": {"run_id": "audit-example", "mode": "audit", "status": "COMPLETED"},
  "discovery": {"candidates_generated": 2, "candidates": []},
  "screening": {"promoted": 1, "screened_out": 1, "results": []},
  "validation": {"routing": {}, "plans": {}, "tools": []},
  "evidence": {"fingerprint": "existing-ledger-fingerprint", "candidates": []},
  "findings": {"findings": []},
  "reproducibility": {"seed": 42, "git_commit": null}
}
```

The technical Markdown retains the previous full rendering: Run Summary, Dataset, Shortcut Candidate Discovery, Candidate
Screening, Validation Plan, Quantitative Evidence, Qualitative Evidence, Evidence
Synthesis, Final Findings, Runtime and Cost, Model Runtime, Limitations,
Reproducibility, and Artifact Index sections. Records retain their identifiers:
candidate → task → PlanStep/measurement → EvidenceRecord → SummaryFinding.

## Availability and safety

Missing values are JSON null / technical Markdown N/A; the concise view says
unrecorded rather than inventing a value. Planned checks do not count as executed
checks in the concise view. An executed probe with no final confirmation is
explicitly labelled; duplicate planned operations across paired candidates are
flagged as non-independent mechanism validation. "Supported" is displayed as
evidence support, never upgraded to a claim of model-reliant shortcut confirmation.
The builder does not query git,
re-read datasets, hash the full dataset, capture a new environment snapshot,
infer model configuration from current environment variables, estimate savings,
or invent seeds. Historical git/version/environment/tool-budget information not
captured upstream remains unavailable. CPU/backend identities are retained where
the existing measurement provenance records them.

Metadata counts describe metadata rows, not guaranteed decoded samples. A
truncated metadata scan exposes `rows_profiled` and `metadata_truncated`, while
the complete `sample_count` remains null. Run elapsed time uses the persisted
Runner start/completion interval. Stage/task timings retain their recorded
intervals, including failed attempts; export timing is excluded so rebuilding
the report cannot change its own inputs.

Only fixed canonical files and typed invocation references are read. Raw model
responses, raw reasoning, dataset example rows and evaluator files are not
exported. Snapshot export excludes API-key environment variable names. Untrusted
text/metrics containing credential markers, private truth or embedded image
blobs are withheld (`[redacted]`); ordinary finalized scientific text is preserved.
Artifact references outside the run, unsafe symlinks, and raw/private references
are excluded. No secret environment values are read.

## Rebuild and resume

```python
from shortcutminer.experiment.audit_report import (
    build_audit_report, export_audit_report, render_audit_markdown,
    render_audit_technical_markdown,
)

report = build_audit_report("runs", "audit-example")  # Read-only
export_audit_report("runs", "audit-example")  # Replaces the three exports atomically per file
markdown = render_audit_markdown(report)  # Reads only the canonical model
technical = render_audit_technical_markdown(report)
```

Completed-run resume rebuilds missing or existing reports using persisted
authority and invokes no Agent, screening or experiment. Repeat exports are
byte-identical when those artifacts are unchanged. If report I/O fails after
scientific completion, the finalized completed run remains recoverable; retry
completed-run resume or the export API after resolving the I/O problem.
Partial failed-run reports are outside this step.

Historical reports are not migrated automatically. To preview the concise view
without changing retained scientific evidence, load the existing canonical JSON
as `AuditReport`, render it to `audit_report.readable.md`, and point its technical
link to the historical `audit_report.md`. This adds a display derivative only;
the old Markdown, canonical JSON, SummaryReport and source snapshot stay intact.

## Offline manual acceptance

```powershell
.venv\Scripts\python.exe scripts/phase17_progress_acceptance.py --runs-root runs/phase17-step5-acceptance --run-id step5-example
```

This runs the installed `shortcutminer audit tests/fixtures/generic_dataset_a`
command using a loopback fake provider and real CPU validation. It checks both
reports, their scientific authority, model accounting, security and idempotence.
Use a fresh run ID. No real DeepSeek/Qwen request is made.
