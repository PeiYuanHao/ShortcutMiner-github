# Experiment configurations

[image-audit.example.json](image-audit.example.json) is a dataset-neutral example
for `infrastructure/experiments/run_candidate_complete_image_audit.py`. It copies
the existing ResNet18/ResNet50 training envelope, three seeds and paired-probe
policy from `synthetic-corner-patch-shortcutminer-memory-v4.json`; it does not
resolve or upgrade dependencies or change those scientific values.

The example intentionally excludes dataset roots, manuscript paths, expected
shortcut fields, pre-generated OOD splits and dataset-specific intervention
recipes. The launcher supplies addressing/split fields from the public input
views and lets the Agents propose candidate-bound interventions.

Review provider YAML settings, hardware resources, task and data splits before
freezing a new configuration. The example's 2 pp effect boundary and 5% family
budget are declared choices, not universal scientific constants. Model training
expands conditionally rather than forcing six profiles for every candidate.

Other files retain their dataset/protocol-specific meanings. Do not repurpose an
old frozen file or rename a historical result as a new comparison. New E4
configurations belong in
[experiments/e4_ablation/configs](../../experiments/e4_ablation/configs).
