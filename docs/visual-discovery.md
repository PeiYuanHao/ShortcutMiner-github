# Visual Discovery for Identify

Visual Discovery performs real, bounded image-content discovery before the Identify Agent is invoked. The control layer decodes approved public images on CPU and produces a strict `VisualDiscoveryProfile`; Identify may interpret that profile to propose an explicitly unverified candidate. The backend itself never constructs a candidate, applies a shortcut threshold, runs a hypothesis test, or returns a validation verdict.

## Execution boundary

```text
IdentifyRequest + DatasetProfile
              |
              v
   VisualDiscoveryProfiler (REAL)
              |
              v
   VisualDiscoveryProfile artifact
              |
              v
       Identify Agent reasoning
              |
              v
     CandidateReport (unverified)
```

This is deterministic infrastructure, not a sixth Agent. The default path is offline and performs no semantic-model or network call.

## Sampling and path resolution

Sampling is seeded, duplicate-free, and bounded by both `VisualDiscoveryOptions.max_samples` and `IdentifyBudget.max_samples`.

- When metadata, a valid label column, and an explicitly configured `sample_path_column` are available, `auto` uses class-balanced sampling. Without a usable label it uses random sampling.
- Sample paths are never guessed from names such as `image`, `file`, `filename`, or `path`. Only an explicitly configured field is resolved.
- Without metadata or an explicit path field, the profiler performs a bounded scan beneath the approved dataset root for BMP, GIF, JPEG, PNG, TIFF, or WebP files.
- A metadata-free directory name is not treated as a label. The profile records `label_grouping_unavailable`, so no label-based group difference is fabricated.

## Canonical image descriptors

Identify and Qualitative use the same `image_descriptor_v1` implementation and the same fixed regions. Recorded facts can include dimensions, aspect ratio, original mode and channels, grayscale mean and standard deviation, RGB channel means and standard deviations, HSV-style saturation mean and standard deviation, 256-bin grayscale Shannon entropy, edge density, file size, and spatial statistics.

The center is the central 50% of width and height. The border is the union of the outer 20% on all sides. Top, bottom, left, and right are fixed outer 20% strips; each corner is a fixed 20% by 20% region. Intensity and RGB values are normalized to `[0, 1]`. Edge density is the fraction of pixels with an adjacent horizontal or vertical absolute grayscale gradient greater than `32/255`.

## Descriptive summaries and ranking

For an explicitly grouped selection, the profiler reports per-group descriptor means and population standard deviations. For each pair of groups it may report:

```text
raw difference = mean(group A) - mean(group B)
absolute difference = abs(raw difference)
pooled descriptive scale = sqrt((std(A)^2 + std(B)^2) / 2)
standardized descriptive difference = raw difference / pooled descriptive scale
```

The standardized value is omitted when the pooled scale is zero. Output is bounded to the configured top N. Ordering uses absolute standardized magnitude when defined and otherwise absolute raw difference; this is screening salience for context compression, not shortcut likelihood or statistical significance. There are no p-values, formal tests, fixed discovery cutoffs, or automatic candidate decisions.

## Coverage and inspected-sample semantics

The profile separately records requested, selected, available, decoded, failed, truncated, bytes inspected, and selected-per-group coverage. `IdentifyDiscoveryContext.samples_inspected` equals successfully decoded samples only. A selected metadata row, missing file, corrupt file, unsafe path, signature mismatch, over-budget file, or skipped sample is not counted as inspected.

## Decode and filesystem safety

Image access is constrained to safe relative references beneath the approved dataset root. Symlink escapes are rejected. Decode is limited by per-file bytes, total inspection bytes, pixel count, and frame policy. Extension and decoded format must agree; corrupt, truncated, decompression-bomb, unsupported-mode, and multi-frame conditions fail or degrade safely. Pixels, PIL objects, NumPy arrays, absolute host paths, runtime capability objects, and hidden truth never enter workflow state or artifacts.

## Optional semantic contract

`SemanticVisualInspector` is a provider-neutral optional protocol. A deterministic mock verifies selected-sample scoping and structured observation integration for categories such as foreground object, background context, text or marker, acquisition artifact, environment, accessory, and spatial context.

When explicitly configured, the provider-backed semantic inspector sends only the already approved sample set through the hardened multimodal runtime and returns strict visible observations. If no capability is injected or a VLM call fails, the profile records `semantic.available = false` and retains deterministic descriptors. Low-level descriptors cannot reliably distinguish semantic differences such as grass versus an indoor room, so Identify must not infer them from filenames. See [semantic-vision.md](semantic-vision.md).

## Persistence and resume

When run storage is enabled, the profile is stored at:

```text
runs/<run_id>/identify/visual_discovery.json
```

The request fingerprint covers the public dataset profile, bounded identities of selected public files, task, seed, descriptor version, options and semantic capability identity. A matching artifact is reused without decoding images again. A malformed or incompatible artifact fails explicitly instead of silently mixing evidence.

## Scientific scope

Real: image decoding, low-level descriptors, coverage, label-aware descriptive summaries, metadata-free scanning, artifact persistence, resume reuse, and an optional configured OpenAI-compatible semantic observation path.

Tested offline: mock and fake-provider semantic inspection, schema retries, scoping, safety limits, fallback, and telemetry.

Not implemented or claimed: provider-independent universal VLM support, model training, model-reliance testing, ID/OOD evaluation, causal claims, counterfactual validation, or a shortcut verdict. Segmentation is not part of discovery because it would be an intervention or localization aid for a later explicitly designed validation stage, not evidence that a shortcut exists.
