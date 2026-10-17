# Experiment organization

This directory is for isolated study-specific code. Shared framework behavior
remains in `shortcutminer/`, reusable research entry points in
`infrastructure/experiments/`, and raw outputs under ignored `runs/` directories.

| Study area | Entry or documentation | Scope |
| --- | --- | --- |
| Known-shortcut image validation | [Waterbirds runner](../scripts/run_waterbirds_shortcutminer.py), [retained run](../docs/waterbirds-shortcutminer-core-run.md), [Colored-MNIST runner](../scripts/run_colorized_mnist_shortcutminer.py) | Dataset-specific protocols; distinguish them from open-ended blind discovery. |
| Agent-owned spatial discovery | [Spatial image entry](../infrastructure/experiments/run_blind_spatial_image_audit.py), [E2-A protocol](../docs/e2-a-isolated-retest-20261009.md) | Training-only localization and construction, then framework-owned validation. |
| Candidate-complete image audits | [Generic entry](../infrastructure/experiments/run_candidate_complete_image_audit.py), [E3 evidence summary](../docs/experiments/e3-texture-recheck.md) | Low-cost checks for measurable candidates, reviewed interventions and conditional model probes. |
| E4 comparisons and ablations | [Dedicated workspace](e4_ablation/README.md), [current scope](e4_ablation/docs/reduced-recovery.md) | E2-A/B, five selected variants; historical Full references, eight retained cells and two protocol-repaired E2-A baseline reruns. Earlier larger matrix remains preserved. |
| Controlled evaluation | [Blind evaluation](../docs/blind-evaluation.md), [repeated evaluation](../docs/repeated-evaluation.md) | Public/private truth separation, scoring and aggregation infrastructure. |

Use `runs/e4_ablation/<experiment-id>/` for E4 outputs. Other studies should use
distinct, descriptive run roots and IDs. Do not move an old run to make it appear
to use a new source identity. Preserve canonical reports and historical configs.

Data generation/private scoring code must remain outside Agent-facing inputs.
Framework ablations, classifier baselines, mock smoke checks and complete
ShortcutMiner runs are different comparisons and must be labelled accordingly.
This index is organizational; it does not declare the E1–E5 evaluation complete.
