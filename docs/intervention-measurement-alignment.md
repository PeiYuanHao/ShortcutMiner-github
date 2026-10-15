# Measurement–intervention alignment

The E3 run `case-e3-v2` found five hypotheses but confirmed no model-reliant
shortcut. Its brightness probe executed successfully: 1500 pairs, an accuracy
drop of 0.0093333 and confidence sequence bounds [-0.0057461, 0.0257433]. At the
frozen 0.02 boundary this remains unresolved. Pair-budget exhaustion is neither
a transport/execution failure nor evidence of absence. The 0.05 sensitivity
stopped for low effect; 0.01 and 0.02 remained unresolved. Do not tune the primary
effect boundary or the probe budget after seeing these results.

The separate engineering defect was that C003's frozen red/blue channel-mean
ratio was assigned the same luminance normalization as C002. Pixel changes and
even incidental changes to the color ratio do not make this a targeted primary
chromaticity intervention. The historical result remains inconclusive; it must
not be recertified or treated as independent color validation.

## Opt-in execution guard

`select_autonomous_construction(..., measurement_binding=...)` enables
`measurement-intervention-alignment-v1`. The candidate-complete entry passes its
already frozen primary measurement. Legacy callers omit the argument and retain
their original proposal identity, selection path and receipt format. Historical
source snapshots and reports are never migrated automatically.

Before calibration or model training:

1. Bind the same frozen candidate and descriptor/ratio.
2. Check supported operation types against the existing descriptor axes:
   chromaticity uses global color or calibrated channel-ratio operations; photometry uses luminance,
   brightness or contrast; texture uses registered blur/frequency/noise/encoding
   operations; spatial statistics use spatial operations; acquisition uses
   encoding/resolution operations. Equal RGB scaling is not a primary color test.
   Unknown or mixed measurement axes remain available, without an invented
   deterministic family assignment, subject to measured change and visual review.
3. Measure the exact primary descriptor on the actual original/changed training
   previews. No measurable change triggers revision or deferral, not a negative
   shortcut verdict. This uses the registered descriptor's numerical precision;
   it introduces no new scientific effect threshold.
4. Deliver target-change diagnostics to Qualitative and Executive alongside
   actual paired pixels. Neither a compatible operation nor a nonzero descriptor
   change certifies semantic preservation, mechanism removal or model reliance.

Invalid proposals receive explicit training-only feedback inside the existing
three-attempt maximum. No fallback automatically selects a transformation. The
Quantitative Agent still selects the operation and parameters, and may defer.
No held-out pixels, labels for calibration, model predictions, private truth or
desired accuracy loss enter these checks.

Successful receipts contain the binding, operation-axis check and per-preview
target values. The derived-view builder validates alignment against the frozen
measurement before admitting the receipt. Resume preserves the contract hash and
preview hashes; changing contracts requires a new run rather than editing an old
receipt.

## Calibrated channel-ratio constructor

The aligned proposal schema adds `channel_ratio_calibration`, selected by the
Quantitative Agent. It chooses `normalize_mean` or `reflect_log_ratio` and a
pooled median/mean strategy. The exact numerator/denominator channels come from
the already frozen RGB-mean ratio; Python fits the center in log-ratio space
using training images only. Per-image reciprocal gains neutralize the ratio
toward that center or reflect it in log space. The numeric center cannot be
supplied directly in an aligned proposal. No dependencies or scientific effect
thresholds were added.

Geometry and the uninvolved channel are preserved. Brightness and subject color
are not guaranteed invariant. Gamut clipping/quantization can leave residual
ratios; actual measured values and paired visual review remain required. Zero
channel means make this constructor unavailable, rather than silently adding an
arbitrary epsilon. Legacy `ConstructionProposal` retains its previous schema;
the extended `AlignedConstructionProposal` is used only for the opt-in path.

Real training-only E3 acceptance after this addition: Quantitative chose
red/blue `normalize_mean`; 3311 training images fitted center ratio
1.1730878402677594. In 12 inspected pairs, the primary target changed in 11,
and the ratio range narrowed from [0.89183596, 1.89779162] to
[1.14962304, 1.17859229]. Qualitative reported `supported` for the scoped visual
change and Executive accepted it. Their limitations explicitly state that label
preservation and model reliance remain unverified. The earlier guard-only
acceptance attempt was deferred and is retained separately. These observations
validate constructor execution, not shortcut dependence.

## Remaining limits

Reliable localized text/logo measurement is still unavailable in this entry.
Texture/color differences may be legitimate task information, and an arbitrary
corruption-induced accuracy loss is not shortcut proof. Limited construction
budgets and missing tools must remain explicit deferrals. Confidence sequence
assumptions, instance independence, probe/final separation, model expansion and
scientific thresholds are unchanged.

Training-only replanning on previously audited E3 inputs is engineering
acceptance, not a new blind experiment or a final paper result. Fresh complete
audits must freeze the new source and configuration and retain their own model,
paired evidence and SummaryReport.
