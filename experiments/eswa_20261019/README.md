# Bounded single-device audit study

This directory contains the source of an isolated Windows/Conda study harness.
It adds no hidden dataset truth to the core Agent-facing audit API. Raw inputs,
private planting records, result files, provider calls, checkpoints and delivery
archives are not included in the public repository.

## Scope and interpretation

| Component | Source | Meaning |
| --- | --- | --- |
| Source preparation | `prepare_inputs.py`, `prepare_pet.py` | Public/private split, source identity, generator masks and exact checkpoint-import checks. Evaluator private truth must never be supplied to discovery. |
| Frozen comparison | `prepare_formal_stage.py`, `formal_queue.py`, `run_one.py`, `compact_roles.py` | Full, serialized single-controller and fixed-policy audit adapters sharing declared resources and scientific gates. |
| Fixed candidate checks | `stage_extensions.py`, `extension_queue.py`, `run_extension.py`, `extension_policy.py` | Replayed discovery and measurements within a block; selected text-feedback masking or fixed routing. This is not another independent discovery experiment. |
| Real-data/evaluator diagnostics | `pet_queue.py`, `run_pet.py`, `pet_local_diagnostics.py`, `pet_local_ops.py`, `synthetic_specificity.py` | Agent attempts and evaluator-only segmentation/exact-support diagnostics remain different evidence categories. |
| Conditional seeds | `seed_replication.py` | Additional classifiers only after the original expansion gate permits them; not additional Agent replications. |
| Resources and recovery | `usage_budget.py`, `resource_policy.py`, `monitor.py`, queue/recovery helpers | Known usage, unknown reservations, fail-closed limits, receipts and explicit recovery identities. Reservations are not invoices. |
| Local export | `export_analysis.py`, `gen_fig_results.py`, `finalize_delivery.py` | Tables, conditional figures, local reports and lossless content-addressed packaging from supplied local records. No empirical outcomes are embedded in the reporting source. |

Observation, association, model sensitivity and nuisance attribution are distinct.
No-op consistency and successful execution do not establish semantic preservation.
Use actual fidelity checks and independent candidate review. Shared source images,
model seeds and image-level bootstrap intervals are not independent Agent audits.
Do not retry valid nulls or deferrals to obtain a preferred outcome.

## Prerequisites

Use the [root installation instructions](../../README.md#installation) and the
locked native Windows Python 3.11 environment. The study scripts use one local
GPU and configured text/VLM providers only when their live stages are invoked.
Offline tests and read-only reporting do not require provider credentials.

The dated preparation/queue scripts retain explicit local paths, original study
identifiers, deadline/budget rules and external protocol-file dependencies. Review
them before reuse; a clean checkout alone is not an executable copy of the private
study inputs. A new study needs its own output root, authorized local data,
generation plan and independently frozen schedule. Do not copy old result IDs
or label a new source snapshot as the old experiment.

`development_queue.py`, `remaining_development.py` and `repair_cached_training.py`
retain bounded historical recovery mechanisms. Their one-shot flags are not fresh
run commands. Inspect actual receipts and processes before any recovery. Never
launch duplicate queues, raise budgets without the declared scope rule, or replay
completed discovery just to overcome an unfavorable result.

## Offline verification

From the repository root in the existing environment:

```powershell
python -X utf8 -m pytest experiments/eswa_20261019 -q
```

Tests cover input boundaries, cache identity, protocol/resource guards, text
masking, pairing, local operators, usage deduplication and archive restoration.
Test fixtures are synthetic; a passing suite is not a scientific success claim.

## Reporting from local artifacts

`export_analysis.py` expects the canonical files produced by this study, checks
receipt increments and source hashes, and refuses an existing output version.
Raw missing values stay missing. Human adjudication is pending until real human
labels are supplied. `monitor.py --once --compact` reads the local runtime state;
it does not call providers. A monitor is not evidence that a training process is
actually making progress; retain explicit execution/process checks.

Publication plots are optional. Install their dependencies in a separate local
environment instead of changing the frozen experimental environment:

```powershell
python -m venv artifacts/plot-env
.\artifacts\plot-env\Scripts\python.exe -m pip install -r experiments/eswa_20261019/requirements-plot.txt
.\artifacts\plot-env\Scripts\python.exe experiments/eswa_20261019/gen_fig_results.py --delivery results_eswa_single/delivery/YOUR_EXPORT_VERSION
```

The plotter reads the supplied export and writes vector PDF plus 300 dpi PNG.
It makes no model calls and trains no classifiers. Verify plot captions against
the supplied local records before using them in a manuscript.

`finalize_delivery.py --delivery PATH --package` packages local evidence with a
file-path manifest and SHA256-addressed deduplication. It excludes credential and
ephemeral files, streams every archived object for integrity verification, and
includes `RESTORE.py`. Packaging does not publish or upload the bundle. Keep it
local unless its contents have been explicitly authorized for release.

## Historical identities

The source published here can include reporting/documentation improvements made
after data collection. Canonical comparisons must use their separately retained
frozen source archives and profiles, never the latest repository head by default.
Human approvals, global novelty, causality, complete provider billing and GPU
kernel-active time must not be inferred from implementation availability.
