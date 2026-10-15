# Documentation

Start with the [repository README](../README.md) for the method and supported
entry points, then use the [reproducibility guide](reproducibility.md) to inspect
or repeat a particular audit. Individual documents describe different protocol
versions; the configuration and source snapshot attached to a run determine
which behavior actually executed.

## Installation, execution and reports

| Document | Contents |
| --- | --- |
| [Reproducibility](reproducibility.md) | Reviewer checklist, public input format, run identity, output interpretation and recovery. |
| [Windows deployment](deployment.md) | Native Conda setup, credential loading and acceptance tools. |
| [Environment record](formal-experiment-environment.md) | Recorded Windows dependency/resource freeze; source-specific rather than a promise that every later checkout matches it. |
| [Model runtime](model-runtime.md) | Provider configuration, capabilities and invocation contracts. |
| [Audit reports](audit-report.md) | Structured, concise and technical reports, and historical display derivatives. |
| [Progress monitoring](runtime-progress.md) | Event-based execution and recovery monitoring. |

## Method and implementation

| Area | Documentation |
| --- | --- |
| Five-agent execution | [Architecture](architecture.md), [communication protocol](protocol.md), [tool runtime](tool-runtime.md) |
| Candidate discovery | [Visual discovery](visual-discovery.md), [semantic vision](semantic-vision.md), [literature and validation catalog](shortcut-literature-and-validation.md) |
| Measurement and construction | [Candidate-complete validation](candidate-complete-validation.md), [measurement alignment](intervention-measurement-alignment.md), [representative observation and quality review](representative-discovery-and-construction-review.md) |
| Model evidence | [Model reliance](model-reliance.md), [OOD/counterfactual validation](ood-counterfactual.md), [local background construction](local-background-ood.md), [segmentation](segmentation.md) |
| Budgets and stopping | [Evidence promotion](evidence-promotion.md), [adaptive paired stopping](adaptive-paired-early-stop.md), [confidence sequence](paired-confidence-sequence-v1.md) |
| Synthesis and learning | [Evidence calibration](evidence-calibration.md), [OOD method memory](quantitative-ood-memory.md), [contribution/evidence checklist](paper-revision-contributions.md) |

Some older implementation documents describe the earlier CPU reference backend
or an earlier construction catalog. They remain valid records of that version;
current executable entry points are indexed under
[infrastructure/experiments](../infrastructure/experiments/README.md).

## Experiments and evaluation

- [Experiment workspace index](../experiments/README.md): shared entries,
  dataset-specific runners and isolated E4 code.
- [E3 retained texture audit](experiments/e3-texture-recheck.md): verified local
  six-model development recheck, including its partially supported conclusion.
- [Waterbirds retained run](waterbirds-shortcutminer-core-run.md): scoped
  metadata-assisted background experiment and retained local evidence.
- [E2-A isolated retest protocol](e2-a-isolated-retest-20261009.md): Agent-owned
  spatial construction and post-run hidden-truth separation.
- [Blind evaluation infrastructure](blind-evaluation.md) and
  [repeated evaluation](repeated-evaluation.md): scoring and aggregation protocols.

Full run bundles are local, Git-ignored artifacts. A link to a protocol or a local
run ID is not a public release of its weights, datasets or predictions. Unit tests,
known-shortcut rediscovery, blind discovery, negative controls and development
rechecks should remain separate evidence categories.

## Development history

- [Archived development README](archive/development-readme.md): former homepage
  with Phase 16/17 implementation chronology and obsolete status statements.
- [Phase 6 validation](phase6-validation.md),
  [Phase 16 readiness](phase16-readiness-audit.md) and
  [system contract audit](phase16-system-contract-audit.md): historical checks.
- [Real-provider development pilot](real-blind-pilot.md) and
  [reviewer-driven validation notes](reviewer-driven-validation.md): development
  protocols and revision context, not current aggregate paper results.

Historical files retain their original paths to preserve existing links and
recorded source manifests. They are not the primary installation instructions.
