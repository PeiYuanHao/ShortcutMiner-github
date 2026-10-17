# E4 ablation workspace

This directory keeps E4 experiment code, configurations and documentation separate
from the main ShortcutMiner implementation and previous experiment entries.
The current user-approved E4 scope contains E2-A and E2-B only, with Single-Agent,
Fixed-Pipeline, w/o Qualitative, w/o Quantitative and w/o Adaptive Routing.
Full references the retained historical experiments. Counterevidence ablation,
E2-C and E3 are excluded from the revised E4 comparison. Earlier pilot/matrix
artifacts and legacy variant implementations are preserved for provenance.
See [reduced recovery protocol](docs/reduced-recovery.md).
See the [historical matrix protocol](docs/protocol.md) for original definitions,
common budgets, metric limitations and the earlier four-input study.

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

Run scripts live in `scripts/`; controller adapters live in `variants/`.
Generated progress, checkpoints and results live under `runs/e4_ablation/`.
The existence of these entry points is not evidence that the matrix completed.

## Boundary with the main framework

E4 has separate entry points, variant policies, protocol encoding and tests; it
reuses the main framework rather than maintaining a second copy of the five
Agents. The shared `shortcutminer/experiment/autonomous_construction.py` exposes
optional construction hooks: `disable_qualitative=False` and
`seek_counterexamples=True`. Existing callers retain the original defaults.
The E4 descriptor codec is opt-in through `--protocol-revision`; `legacy` remains
the default. No E4 switch changes the scientific thresholds or training budget
of the ordinary experiment entry points.

The Git backup contains source, tests and documentation. Local provider secrets,
datasets, model weights, checkpoints and run reports remain ignored. Retained
experiments keep their original frozen source archives; a later backup commit
does not retroactively change their source identities.
