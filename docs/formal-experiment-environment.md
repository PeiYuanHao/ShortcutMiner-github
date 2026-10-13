# Formal experiment environment: native Windows conda

The official target is Windows AMD64, conda `shortcutminer`, Python 3.11.16.
The supported terminals are CMD and PowerShell. Scientific architecture,
five Agents, workflow, screening, evidence promotion, schemas, and thresholds are
unchanged. No E1–E5 or real-provider calls are part of environment acceptance.

## Exact environment and locking

The conda interpreter is `E:/Anaconda_envs/envs/shortcutminer/python.exe`.
`requirements/conda-win64.explicit.txt` records exact win64 conda package builds and
MD5 identities. The JSON manifest also records conda package SHA-256 where available.

Keep the current conda build tools: pip 26.2.1, setuptools 83.0.0, wheel 0.47.0.
The eleven scientific direct pins remain in `requirements/core.in` without upgrades
or downgrades. Dependency resolution runs in this native conda environment, with
constraints preserving existing installed dependency versions; host `.venv` is not
a resolution source. Conflicts must stop resolution rather than changing scientific pins.

`requirements/formal-win64-py311.lock` freezes all resolved runtime wheels and
SHA-256 hashes. `requirements/build-tools-win64.lock` freezes build tools and their
transitive dependencies. The standalone root `requirements.txt` lists the complete
71-package set as exact versions with Windows wheel SHA-256 hashes, without `-r`
references to other files. Install using
`--require-hashes --no-deps`; never resolve compatibility ranges for formal execution.
The local source is run through `python -m shortcutminer` from the repository root.

## GPU

Top-level GPU pins are torch 2.9.1+cu128 and torchvision 0.24.1+cu128 from the official
PyTorch CUDA 12.8 Windows wheel channel, without torchaudio. Windows wheels embed CUDA
libraries; the selected Windows wheels supply their runtime dependencies.
Actual selected wheels and dependency metadata are recorded in the native resolution.
GPU availability requires a separate successful local tensor check; resolution alone
is not GPU acceptance. Hardware: RTX 4060 Laptop GPU, 8188 MiB, driver 596.49.

The existing Stage-2 `real_cpu_v1` / `reference_image_linear` backend runs on CPU.
Installing GPU PyTorch does not change this scientific execution backend.

## Paths and restart/resume

| Purpose | Frozen Windows root |
| --- | --- |
| Code | `E:/ShortcutMiner-github` |
| All run artifacts | `E:/ShortcutMiner-github/runs` |
| Source datasets | `E:/ShortcutMiner-Experiments` |
| Runtime/config files | `E:/ShortcutMiner-github/configs` |

All audits, checkpoints, telemetry, intermediates, and reports must use the same
explicit runs root; resume preserves that root and the run ID. Native files remain
accessible after process exit/restart. Datasets and configs are treated as read-only;
this is a local operational policy, not a mounted filesystem guarantee. Do not
blindly resume historical checkpoints with different absolute paths.

## Efficiency and resources

Host: Intel Core i9-14900HX, 24 cores / 32 logical processors, 15.62 GiB RAM.
Native execution uses host resources without additional hard resource caps.
A short synthetic NumPy matrix benchmark compared 1, 2, 4, 8, 16, 24, and 32 threads.
The fastest observed median selected 24 threads (about 0.0182s vs 0.0539s at 2 threads).
This is a speed-priority candidate, not a claim of globally optimal experiment speed.
Set OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, NUMEXPR_NUM_THREADS to 24
before imports. The acceptance wrapper applies these values to its child process.
For explicit local launch in PowerShell:

```powershell
$env:OMP_NUM_THREADS="24"
$env:MKL_NUM_THREADS="24"
$env:OPENBLAS_NUM_THREADS="24"
$env:NUMEXPR_NUM_THREADS="24"
```
CMD equivalent:

```cmd
set OMP_NUM_THREADS=24
set MKL_NUM_THREADS=24
set OPENBLAS_NUM_THREADS=24
set NUMEXPR_NUM_THREADS=24
```

Before actual audits, set the same values explicitly in the local launch process.

Local capacity acceptance on representative actual workloads remains mandatory before
E1. Adjust and re-freeze only before E1; then use the same values for E1–E5. This synthetic
benchmark is not capacity acceptance and does not introduce extra experiment parallelism.

## Network and secrets

Offline acceptance uses a fake loopback provider, strips real-provider credential and
opt-in variables, and disables LangSmith tracing. Native Windows networking is not
OS-disabled. No OS network-isolation guarantee is made.
Formal provider access is an explicit local launch, with credentials injected only at
runtime from ignored `.env.formal`. API CLI commands load the two credential variables
for the lifetime of that command and restore the caller environment afterward.
No Windows user/system environment variables are written. Existing explicit process
credentials take precedence. Offline acceptance and tests disable file loading via
`SHORTCUTMINER_DISABLE_FORMAL_ENV=1`; the acceptance wrapper never loads this file. Never record key values, fragments, lengths, or hashes.
`SHORTCUTMINER_TEXT_API_KEY` and `SHORTCUTMINER_VLM_API_KEY` must stay out of source,
configs, locks, manifests, logs, and audit reports.

## Timeouts

Dependency installation acceptance budget: 60 minutes. Full pytest wrapper timeout:
30 minutes. Fake-provider smoke wrapper timeout: 5 minutes. Deadline failure records no
PASS and terminates only the launched process tree. A dependency download/resolution
failure leaves the freeze blocked; no automatic pin changes are allowed.
Existing provider timeout (60s in model examples) and tool timeout (30s default) are
preserved. Formal long audits receive no new whole-run timeout. There is no build step for local execution.

## Source identity and gates

`configs/environment/formal_experiment_environment.json` records HEAD, dirty state,
source tree hash, exact conda/package/lock/config identities, hardware, thread candidate,
paths, network policy, and timeout policy. Source hashing excludes ignored files,
`.env*`, and the manifest directory to avoid self-reference; configs and locks are
separately hashed. Finalization writes a receipt with the tracked manifest hash.

The user explicitly revoked the clean-commit prerequisite on 2026-10-04 and authorized experiments from the current source snapshot. The Windows manifest records `source_policy.mode=user_authorized_source_snapshot`, the actual dirty status and source tree hash; execution archives retain the exact source used. No automatic commit is made. Capacity acceptance remains workload-specific. See `docs/waterbirds-paper-validation-run.md` for the completed native Waterbirds case.

The following clean-commit route remains available as an optional stricter source identity:

A candidate can describe a dirty tree. Under the optional clean-commit route, formal E1–E5 require a user-created clean Git
commit plus the Windows runtime/conda/environment/config/lock identities. No automated
commit is made. `finalize` records the clean identity but leaves the capacity gate pending.

```powershell
conda activate shortcutminer
python -m pip install --require-hashes --no-deps -r requirements.txt
python scripts/formal_environment.py check
python scripts/formal_environment.py pytest
python scripts/formal_environment.py smoke
# Record the configured source identity policy; no automatic commit:
python scripts/formal_environment.py finalize
```

Checks inspect frozen versions and file hashes and run `pip check`. They do not start
formal experiments. Receipts and local resolution evidence live under ignored `runs`.
