# Blind Shortcut Discovery Evaluation

ShortcutMiner's blind evaluation framework measures whether the production audit
workflow can discover a shortcut without receiving the answer in advance. It is
evaluation infrastructure under `shortcutminer/evaluation/blind/`; production
workflow, runtime, experiment, tool, protocol, and storage modules must not import
it.

This phase establishes a reproducible evaluation harness. The included controlled
smoke fixtures demonstrate that the harness can generate, isolate, match, score,
and aggregate challenges. They are not evidence of real-world ShortcutMiner
performance.

## Public/private firewall

Each generated challenge has two physically separate roots:

```text
<challenge-id>/
├── challenge.json
├── public/
│   └── dataset/
│       ├── metadata.csv
│       └── images/item-*.png
└── private/
    └── hidden_truth.json
```

Only `public/dataset/` may be supplied as the production `dataset_root`.
`hidden_truth.json` is loaded explicitly by `BlindEvaluator` only after all
production outputs have completed. The evaluator consumes immutable structured
outputs and verifies that their content hashes do not change during scoring.

The hidden truth reference and its content hash are held by the evaluation
challenge descriptor, outside the public dataset. Private fields such as shortcut
family, target field or region, injection mechanism, affected sample IDs, and
strength never enter the dataset profile, Agent prompts, tool inputs, workflow
state, manifest, evidence ledger, or summary report. Public filenames, directory
names, and metadata headers are checked for answer-revealing names. Runtime tools
remain constrained to the public dataset and approved run artifacts.

Because the public challenge ID becomes part of the dataset path, IDs containing
challenge-family or answer-revealing terms are rejected as well.

The leakage suite runs the same deterministic production workflow three times
with private truth present, removed, and replaced. The complete production output
hash must be identical in all three cases. It also statically checks that production
modules do not import the blind evaluator namespace.

## Challenge protocol and generation

All evaluation records use strict Pydantic contracts with unknown fields rejected.
The public contract contains only a neutral challenge ID, generator version,
public dataset reference, metadata filename, task description, runtime sampling
seed, and public dataset hash. The private contract records the controlled truth,
including precise localization and affected samples.

`BlindChallengeGenerator` is deterministic for a fixed configuration and seed.
It creates small, CPU-only classification datasets with neutral public names in
five families:

- `metadata_association`: a label-associated neutral device field;
- `spatial_visual`: a label-associated corner pattern;
- `acquisition_quality`: a label-associated brightness transformation;
- `mixed`: one real metadata shortcut plus irrelevant image-size, brightness,
  site, and batch variation;
- `clean_control`: no injected shortcut, but realistic class imbalance, missing
  numeric metadata, multiple sites/devices/batches, and balanced image-size and
  brightness variation.

Injection strength is explicit (`weak`, `moderate`, or `strong`) and stored only in
private truth. A public dataset hash covers sorted relative filenames and file
bytes, not machine-specific filesystem metadata. A private hash covers canonical
truth JSON. Generation refuses to overwrite a non-empty challenge directory.

## Candidate matching and adjudication

Candidate-to-truth matching is deliberately conservative and deterministic. A
match requires a compatible cue family and the exact controlled target field,
spatial region, or acquisition transformation to appear in the candidate or its
structured evidence. There are no embeddings, edit-distance rules, substring
guessing, or other fuzzy matching.

Family-compatible candidates without exact localization are `ambiguous`; they do
not count as discoveries until adjudicated. Explicit manual adjudications can
override deterministic outcomes. The multi-rater interface reports item-level
counts, raw agreement, consensus where one exists, and two-rater Cohen's kappa
when it is defined. Clean controlled challenges are exhaustive with respect to
the generator, so unmatched candidates are false-positive candidates; a
non-exhaustive future challenge can instead label them `potential_novel`.

## Metrics and failure analysis

Metrics keep discovery, validation, and final claims separate:

- truth recall, recall at 1, 3, and the configured candidate limit;
- assessed-candidate precision, unmatched and ambiguous candidate counts;
- exact/partial/incorrect/unknown localization and exact localization rate;
- the discovery-to-routing-to-quantitative/qualitative-to-usable-evidence-to-final
  support funnel;
- candidate false positives, supported final false positives, and separately
  counted partially supported false positives;
- miss taxonomy for non-discovery, routing failure, inconclusive evidence, and
  summary underclaim or overclaim.

Positive challenges and clean controls are always reported separately. Recall is
undefined for clean controls rather than being represented as zero. Candidate
false positives are exploratory hypotheses that do not match controlled truth;
final false positives are stronger `supported` summary findings on a clean
control. These rates must not be conflated.

Optional downstream impact accepts measured before/after, out-of-distribution,
group-gap, or intervention metrics. When no downstream model study is supplied,
the result records an explicit unavailable reason rather than inventing a number.

## Frozen snapshots, seeds, and batch uncertainty

Every run records an `EvaluationSnapshot` at
`evaluation/snapshots/<snapshot-id>.json`. Its identity freezes:

- the Git commit;
- every file in all five Agent workspaces;
- production message-contract files;
- quantitative, qualitative, and evidence-calibration backend versions;
- sanitized runtime/model configuration;
- runtime sampling seed and evaluation bootstrap seed;
- candidate limit, validation depth, and budget settings.

Source-tree runs resolve the full Git commit directly from `.git` without invoking
an external command. Packaged environments, including the repository's Docker
image where `.git` is intentionally excluded, must supply the full 40- or 64-digit
hexadecimal commit explicitly; an unknown or abbreviated revision is rejected.

Secrets and absolute paths are removed. Private truth is never an input to the
snapshot. The snapshot ID deliberately excludes the per-challenge public hash so
one frozen production configuration can identify a batch; each saved per-run
snapshot still records and verifies its own public dataset hash.
When supplied, the frozen model-runtime snapshot ID and sanitized snapshot hash
are recorded explicitly. A model runtime configuration change therefore changes
the evaluation snapshot identity.

Batch aggregation refuses mixed snapshot IDs and reports positive and clean
strata independently. Percentile bootstrap intervals reuse the Phase 11
uncertainty implementation and resample independent challenges, never samples
inside one challenge. An interval is available only with at least two independent
challenges in that stratum; otherwise the result records why it is unavailable.

## Artifacts and claims boundary

Per-run scores are written to `evaluation/blind_result.json`. This result contains
truth identifiers and the verified private hash needed for traceability, but not
the hidden truth specification, injection mechanism, strength, or affected sample
list. Batch results reference per-run artifacts by safe relative paths.

The framework does not download datasets or models, change Hugging Face assets,
run comparison baselines, or claim scientific effectiveness. A credible discovery
claim requires a separately designed blind batch with fixed configuration,
independent challenges, predefined adjudication, adequate uncertainty, and honest
reporting of misses and clean-control false positives.

`RepeatedBlindAuditRunner` adds sequential, checkpointed repetitions under the
same frozen snapshot. It retains runtime failures independently and reports raw
discovery, localization, final-assessment, false-positive, and runtime-reliability
distributions. See [repeated-evaluation.md](repeated-evaluation.md).
