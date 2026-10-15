# E4 ablation workspace

This directory keeps E4 experiment code, configurations and documentation separate
from the main ShortcutMiner implementation and previous experiment entries.
The user's E4 design will define the actual ablation matrix; no variants or
experiments have been selected or launched by creating this scaffold.

## Layout

```text
experiments/e4_ablation/       # version-controlled experiment workspace
  scripts/                    # launch, collect and compare experiment results
  variants/                   # E4-specific adapters and component switches
  configs/                    # baseline and per-variant experiment configurations
  tests/                      # tests for the E4 adapters and comparison code
  docs/                       # protocol, run instructions and interpretation

runs/e4_ablation/             # ignored local outputs, outside the code directory
  <experiment-id>/            # configs/snapshots, logs, checkpoints and reports
```

## Working conventions

- Implement E4-specific launchers and variants here. Reuse the main
  `shortcutminer` package; avoid copying and editing a second framework tree.
- Keep the main framework defaults and existing experiment entry points intact.
  If a shared hook is needed, make it explicit and backward compatible, with a
  test that the unchanged default path retains its behavior.
- Record component changes in explicit per-variant configurations. The actual
  comparison settings and budgets come from the user's E4 design.
- Use a distinct experiment ID for each configuration. Put generated outputs
  under `runs/e4_ablation/<experiment-id>/`, never in the source/config folders
  or an earlier experiment's run directory. Resume the same configuration in its
  existing run directory rather than replacing its identity.
- Keep credentials only in the repository's ignored local `.env.formal`, loaded
  through the existing runtime boundary. Do not copy credentials into this
  workspace, configurations or reports.
- Use the existing Windows Conda environment `shortcutminer` and PowerShell/CMD.
  E4 reports should be in English. Do not commit datasets, weights or run outputs.

There is intentionally no runnable experiment command yet: this is an isolated
workspace scaffold, ready for the E4 protocol already designed by the user.
