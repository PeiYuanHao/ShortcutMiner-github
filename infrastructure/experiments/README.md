# Shared image-audit entry points

These scripts compose existing ShortcutMiner execution paths. They are not
separate replacements for the five-Agent framework. Run them from the repository
root in the Windows Conda environment `shortcutminer`.

| Entry | Purpose |
| --- | --- |
| `run_candidate_complete_image_audit.py` | Candidate-bound measurements, cheap checks, reviewed Agent-selected interventions, conditional model validation and final reporting. |
| `run_autonomous_image_audit.py` | Earlier generic single-construction path; retained for compatible protocols. |
| `run_blind_spatial_image_audit.py` | Training-only Agent localization and spatial quality review using the generic entry contract. |
| `run_e2_spatial_quality.py` | Historical, dataset/path-specific E2 engineering entry; not a portable generic launcher. |

The candidate-complete entry supports `--representative-review` for the newer
class-representative discovery and training-construction-review policies.
`--isolated-discovery` excludes learned methods from previous runs. Defaults of
earlier entry points are retained for their historical configurations.

Use the [dataset-neutral example](../../configs/experiments/image-audit.example.json)
and [reviewer guide](../../docs/reproducibility.md) for new image audits. A live
launch calls real providers and can train GPU models. `--help` only shows syntax.
Never supply private truth, expected cues or pre-generated OOD to blind discovery.

New E4-specific launchers and variants belong in
[experiments/e4_ablation](../../experiments/e4_ablation/README.md).
