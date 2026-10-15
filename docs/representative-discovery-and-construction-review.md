# Representative discovery and training construction review

The candidate-complete entry enables these changes only with
`--representative-review`. Existing entry defaults, old construction proposal
schemas and receipts remain unchanged. Historical results are not recertified.

## Observation coverage: class-representative-v1

Deterministic numeric descriptors cover the public training input, bounded at
10,000 images, 40 label groups and 500 MB of inspected image bytes. Actual decoded
coverage, omitted groups and byte failures remain explicit. Numeric group
statistics are descriptive, not shortcut certification. Large per-image numeric
arrays remain in canonical artifacts; Identify receives group summaries and the
actual semantic sample subset rather than an unbounded context.

Semantic inspection starts with four deterministic shuffled samples per class.
Identify makes a training-only coverage decision to stop or expand to eight and
twelve per class. Later looks add new images only. Decisions include unresolved
observation questions and limitations, and are not significance tests. Batches,
source identities and decisions are cached under the discovery run's
`identify/representative-coverage`. Missing or failed semantic inspection is
reported as unavailable, never evidence of no shortcuts. Numeric and semantic
coverage must be reported separately; scanning pixels is not semantic review.

## Construction quality: training-construction-review-v2

Quantitative declares mechanism scope, the part of the hypothesis actually
tested, remaining confounds and training-only selection criteria. Natural
subject color, scene color and acquisition color cast are not interchangeable.
Scalar neutralization does not automatically isolate the mechanism.

Previews include eight seed-2026 random training images and four artifact-stress
extremes: channel-ratio extremes for a ratio operation, brightness extremes for
other operations. The frozen primary descriptor is measured separately on every
preview; stress selection does not imply primary-descriptor coverage. Diagnostics
add newly boundary-valued RGB components, RGB
change, per-pixel luminance change and horizontal luminance-gradient change.
These are descriptive diagnostics, with no new scientific cutoffs.
Qualitative reviews every preview and reports task preservation, mechanism
specificity, artifact risks and whether to revise/defer/accept a scoped probe.
Damaged or unsupported constructions cannot reach acceptance, even if an
Executive were willing to accept them. Revisions use the existing three-attempt
cap. Uncertain semantics may support a limited diagnostic with disclosed limits,
not a causal claim or a label-preservation certificate.

For a frozen RGB-mean ratio, the new calibration branch uses
`bounded_channel_ratio`: a training-fitted center, an Agent-selected bounded
log-ratio shift, per-pixel float RGB luminance preservation, and chroma compression
into the RGB gamut. Eight-bit rounding introduces at most 0.5 luminance units
under the operator's RGB weights. This is not perceptual luminance invariance.
The measured ratio may only partly move to its target. Subject and background
colors both change; actual target measurement and visual review remain required.
Legacy `channel_ratio` operations are unchanged.

Background-only hypotheses require the separately authorized SAM/ViTMatte
construction path and its masks/quality evidence. This generic entry does not
silently substitute a whole-image operation or advertise unavailable segmentation
as executable background isolation. Such cases remain explicit deferrals.

## Scientific boundaries

No changes to five-Agent ordering, association logic, EvidenceLedger, promotion,
the 2 pp primary effect, 1/2/5 pp sensitivity, 5% family error budget, 1,500 paired
probe maximum, model budgets or conditional early stopping. No held-out outcomes
are used to select operators or strength. A new run on already inspected E3 data
is a development recheck, not an independent first-time blind confirmation. A
positive result is not guaranteed or manufactured by increasing corruption.

## Retained E3 development recheck

`runs/e3-representative-20261014/case-e3-representative-v1` completed the full
five-Agent workflow. Numeric inspection decoded 3,311 training images; Identify
stopped semantic sampling at four per class (148 images). Central masking was
rejected as damaging task content. The new C003 texture/edge candidate selected
a training-reviewed low-pass operation, cutoff 0.15. The new bounded color-ratio
tool was not selected in this run; its execution evidence comes from tests.

The paired pilot expanded at 256 pairs under the unchanged 2 pp boundary.
ResNet18 and ResNet50, seeds 42/43/44, completed conditionally. Final evaluation
used 2,169 source-disjoint held-out pairs, with zero overlap with reserved probe
IDs. Mean drops were 7.21 ± 0.23 pp and 6.61 ± 0.51 pp across the respective
three seeds. The aggregate model drop was 6.69 pp, 95% paired bootstrap interval
[5.53, 7.89] pp. These rows share samples and are not independent replications.

The actual Summarize assessment is **partially_supported**. One low-pass operation
does not distinguish nuisance texture from legitimate breed texture or generic
smoothing sensitivity. This is not a clean causal shortcut confirmation or an
independent blind-discovery result. `audit_report.readable.md` corrects a
post-completion presentation omission of final paired metrics; original canonical
reports and SummaryReport remain unchanged, with provenance in execution-review.
