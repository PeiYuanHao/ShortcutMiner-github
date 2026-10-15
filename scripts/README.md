# Supporting scripts

Primary user commands are documented in the [repository README](../README.md).
Shared Agent-owned image audit launchers live in
[infrastructure/experiments](../infrastructure/experiments/README.md), and new E4
code belongs in [its own workspace](../experiments/e4_ablation/README.md).

| Group | Files | Usage boundary |
| --- | --- | --- |
| Dataset-specific framework runners | `run_waterbirds_shortcutminer.py`, `run_colorized_mnist_shortcutminer.py` | Read the matching protocol/configuration; live mode calls providers. |
| Background/model reference experiments | `run_waterbirds_background_validation.py`, `run_waterbirds_paper.py` | Inspect the documented scope; model-only evidence is not automatically a complete framework result. |
| Environment tooling | `formal_environment.py`, `formal_environment_lock.py` | A recorded source/environment freeze is installation-specific; refreshing is not historical reproduction. |
| Acceptance and recovery checks | `phase16_provider_security_preflight.py`, `phase16_task_isolation_replay.py`, `phase17_progress_acceptance.py`, `phase17_promotion_acceptance.py` | Development checks with their own documented conditions, not paper experiment launchers. |
| Historical empty scaffolds | `run_experiment.py`, `run_matrix.py`, `aggregate_results.py`, `health_check.py`, `validate_config.py` | Currently placeholders, not working launch or aggregation commands. |

Historical names and paths remain unchanged for source-manifest compatibility.
New experiment code should use the dedicated study workspace rather than adding
another top-level script here. Generated reports, logs, weights and credentials
belong outside this source directory.
