# Retained E3 texture/edge audit

This is a curated summary of the completed five-Agent run
`case-e3-representative-v1`, retained locally under
`runs/e3-representative-20261014/`. It is not the original Summarize output or a
release of the complete run bundle. The [machine-readable summary](e3-texture-recheck.json)
records the measurements, model profiles, source identity and canonical hashes
used to check the numbers below.

## Task and execution

The dataset was Oxford-IIIT Pet, with 37 breed classes: 3,311 train, 369 validation
and 3,669 ID images. Agent-facing inputs used neutral image IDs and opaque numeric
labels. The dataset identity, previous audit results and an expected shortcut were
not supplied to the Agents. Because these source images had already been used in
development, this recheck is **not an independent first-time blind experiment**.

The numeric discovery scan decoded all 3,311 training images. Semantic inspection
sampled four per class (148 total), after which Identify elected to stop its
observation-budget expansion. Later candidate-specific Qualitative checks had
their own recorded sampling budgets; these are separate coverage figures.

Identify proposed five candidates. Quantitative selected a frozen edge-density
measurement for C003 and a low-pass frequency intervention, cutoff 0.15, from
training-only evidence. Actual training pairs were reviewed before held-out
evaluation. A central masking proposal for C001 was rejected for damaging task
content; it was not promoted to expensive model validation.

## Conditional expansion and independent confirmation

At 256 pilot pairs, the ResNet18/seed-42 accuracy drop was 8.98 pp and its
sequential lower bound was 2.03 pp, exceeding the predeclared 2 pp boundary.
The existing rule therefore expanded to the frozen two-architecture/three-seed
budget. It did not force that budget for every candidate.

| Architecture | Seeds | Training stopped at epochs | Best checkpoints |
| --- | --- | --- | --- |
| ResNet18 | 42, 43, 44 | 17, 20, 30 | 7, 10, 20 |
| ResNet50 | 42, 43, 44 | 26, 20, 23 | 16, 10, 13 |

The final confirmation used **2,169 paired images**, disjoint from all 1,500
reserved probe sources. Source-ID overlap was verified as zero. All six model
weight hashes and final prediction sample sets were checked.

| Architecture | Original accuracy (%) | After intervention (%) | Accuracy drop (pp) |
| --- | ---: | ---: | ---: |
| ResNet18 | 88.44 ± 0.59 | 81.24 ± 0.60 | 7.21 ± 0.23 |
| ResNet50 | 92.52 ± 0.28 | 85.91 ± 0.40 | 6.61 ± 0.51 |

Values are mean ± sample standard deviation across three seeds. All rows reuse
the same final image pairs; six profiles are not six independent datasets.
The recorded aggregate model accuracy changed from 92.67% to 85.98%, a drop of
6.69 pp with a 95% paired percentile-bootstrap interval of [5.53, 7.89] pp
(200 bootstrap iterations). The pilot confidence sequence and final bootstrap
interval are different procedures with different scopes.

## Actual final assessments

| Candidate | Hypothesis | Summarize assessment |
| --- | --- | --- |
| C001 | Center-region brightness and color balance | Inconclusive |
| C002 | File size and image dimensions | Inconclusive |
| C003 | Texture and edge complexity | **Partially supported** |
| C004 | Background and scene context | Inconclusive |
| C005 | Watermark and text markers | Insufficient evidence |

C001, C002 and C004 completed low-cost checks but had no accepted model
intervention in this run. C005 lacked an executable watermark measurement.
These outcomes do not establish that their cues are absent.

C003 demonstrates a stable model response to the selected low-pass operation.
The operation changes both background and animal texture; fine coat texture may
be legitimate information for breed classification. A single smoothing operation
without a specificity control does not isolate nuisance texture from valid task
information or generic perturbation sensitivity. The actual assessment therefore
remains **partially_supported**, not clean causal shortcut confirmation.

## Provenance and availability

The run used an author-authorized dirty snapshot based on commit
`34c3e34ad8d249fad933f373c01a9bb998799276`; later repository commits must not be
substituted for that execution identity. The recorded source archive SHA-256 is
`099ed360460a2748f2952916eabb85bc27d4da54cb0c403a6ea4ca258f5cacdb`.
A hashed runtime adapter clarified nullable measurement-plan JSON encoding after
a schema rejection. It did not coerce outputs or change scientific gates.

Original canonical reports and SummaryReport remain unchanged. A separate
`audit_report.readable.md` corrects a presentation omission of final paired metrics.
Some raw limitations inherited from discovery refer to the earlier train-only,
pre-training stage; the final model and held-out facts above were checked against
the completed execution records.

The full weights, images, source archive, predictions and runtime traces are
local and Git-ignored. This summary provides an inspectable evidence record; a
repository clone alone is not sufficient to reproduce the complete historical
run. No new experiment was launched to write this document.
