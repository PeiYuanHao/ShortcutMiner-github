# Windows local deployment

Use Windows AMD64 and conda environment `shortcutminer` (Python 3.11.16).
Commands support both CMD and PowerShell from `E:\ShortcutMiner-github`.
The root `requirements.txt` is a standalone text file containing all 71 exact package
versions and Windows wheel hashes. The separate `requirements` directory holds
supplementary Windows wheel/provenance locks, not the file to pass to `pip -r`.

CMD: `cd /d E:\ShortcutMiner-github`

PowerShell: `Set-Location E:\ShortcutMiner-github`

```powershell
conda activate shortcutminer
python -m pip install --require-hashes --no-deps -r requirements.txt
python scripts/formal_environment.py check
python scripts/formal_environment.py pytest
python scripts/formal_environment.py smoke
```

The exact interpreter currently resides at
`E:\Anaconda_envs\envs\shortcutminer\python.exe`. If activation is unavailable,
invoke that interpreter directly:

CMD:

```cmd
"E:\Anaconda_envs\envs\shortcutminer\python.exe" scripts\formal_environment.py check
```

PowerShell:

```powershell
& "E:\Anaconda_envs\envs\shortcutminer\python.exe" scripts\formal_environment.py check
```

Use `python -m shortcutminer` so the CLI uses the
same interpreter. Do not install from compatibility ranges for formal runs.

Recreating the conda base on Windows uses the exact package export:

```powershell
conda create --name shortcutminer --file requirements/conda-win64.explicit.txt
conda activate shortcutminer
python -m pip install --require-hashes --no-deps -r requirements.txt
```

The create command is for a missing environment, not the existing one. Package
retrieval requires internet access; offline acceptance uses a fake loopback provider
and strips provider credentials. This does not disable Windows networking.

## Persistent paths

| Purpose | Windows path |
| --- | --- |
| Source | `E:/ShortcutMiner-github` |
| Runs/checkpoints/reports/telemetry | `E:/ShortcutMiner-github/runs` |
| Source datasets (treat as read-only) | `E:/ShortcutMiner-Experiments` |
| Frozen configs (treat as read-only) | `E:/ShortcutMiner-github/configs` |

Pass `--runs-root E:/ShortcutMiner-github/runs` explicitly for audit/mock/resume.
Resume must use the same root, run ID, source identity, and frozen configuration.
Artifacts survive process exit or Windows restart. Dataset/config read-only policy is procedural, not an OS sandbox. Historical
checkpoints with different absolute roots need separate path review.

## Credentials and formal execution

API CLI commands automatically read the repository-root, Git-ignored `.env.formal`
for `SHORTCUTMINER_TEXT_API_KEY` and `SHORTCUTMINER_VLM_API_KEY`. Values are injected
only into the current process while the command runs, then removed. Existing explicit
process credentials take precedence. No Windows user/system environment variables
are written. UTF-8/BOM, quoted values, comments, and CRLF are supported; shell commands
and variable interpolation are never executed. Only these two key names are loaded.
Mock commands, offline acceptance, and unit tests disable this loading.

Presence-only verification (same command in CMD and PowerShell; no API request):

```text
conda activate shortcutminer
python -m shortcutminer pilot credentials --config configs/experiments/real_blind_pilot.example.yaml
```

The result reports only SET/MISSING, never values. Do not place keys in configuration, source, requirements, manifests, logs, reports,
or command arguments. Acceptance never
calls real providers. Credentials alone do not authorize a formal run.

Before E1, complete local capacity acceptance and manually create a clean Git
commit. Record the final source/config/environment/lock identity with:

```powershell
python scripts/formal_environment.py finalize
```

This records an identity receipt under `runs/environment-freeze-windows` and does
not run E1–E5. Source or environment changes require reviewed re-freezing before
experiments. See [formal environment](formal-experiment-environment.md).
