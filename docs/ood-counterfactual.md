# OOD and Counterfactual Validation

OOD evaluation and counterfactual evaluation answer different questions.

- OOD compares model behavior across distributions or controlled challenge sets.
- Counterfactual evaluation pairs each original sample with a cue-modified version and runs the same model on both.

ShortcutMiner records `comparison_mode` as `paired` or `independent`; it does not guess. OOD construction changes a Candidate-derived suspected cue while intending to preserve explicitly recorded invariants such as task-label semantics. The manifest records distinct OOD/intervention IDs, the transformation, parameters, seed, input/output references, selected/usable/failed counts and pair coverage, affected samples, model hash, segmentation provider/version, quality, failures, warnings, limitations and fingerprint. Outputs are run-local beneath `interventions/<task-id>/`; source data is never overwritten.

## Transformations

The allowlist includes global channel scaling/red-blue swapping, grayscale, desaturation and color replacement; bounded brightness and contrast; Gaussian blur; seeded Gaussian noise; bounded JPEG recompression; downsample/up-sample resolution changes; fixed/custom normalized spatial masking and replacement; bounded spatial relocation; and mask-assisted background replacement. Parameters are strict and bounded. The engine never optimizes a transformation to maximize model error and never accepts code or free-form image operations.

## Pair integrity and validity

Each `CounterfactualPair` preserves original sample ID/reference, modified reference, label, model ID/hash, transformation, target cue, seed, optional mask, label-preservation status, warnings, limitations and quality. Label preservation can be guaranteed by synthetic construction, supported by a masked design, user-declared, unverified, or failed. Real-dataset background edits are not automatically guaranteed label-preserving.

Quality fields report decode success, dimensions, bytes/format, mask coverage, pixel-change fraction, and foreground/background change fractions. They are measurements, not validity thresholds. Corrupt samples become retained failed pairs; usable pairs may still yield a completed or scientifically inconclusive experiment.

## Metrics and uncertainty

ID/OOD outputs include accuracy, balanced accuracy and their deltas. Paired outputs additionally include pair coverage, prediction flip count/rate, original/counterfactual accuracy, true-class confidence delta, true-class logit delta, and margin delta when supported. The same model hash is mandatory throughout. Paired bootstrap resamples whole pairs; independent comparison resamples the two distributions separately. Neither accuracy drop nor prediction flip is automatically mapped to `supports` or causal proof: artifacts, changed task information and instability remain alternative explanations.
