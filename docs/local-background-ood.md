# Local background intervention for Quantitative

Native Windows conda `shortcutminer`, Python 3.11.16; Transformers **5.14.1**.
Existing torch 2.9.1+cu128 / torchvision 0.24.1+cu128 and existing scientific
package versions are preserved. All newly required packages and Windows wheel
hashes are included in `requirements.txt` and `requirements/formal-win64-py311.lock`.

```powershell
conda activate shortcutminer
python -m pip install --no-deps -r requirements.txt
python scripts/run_waterbirds_shortcutminer.py --config configs/experiments/waterbirds-shortcutminer-background-formal.json
```

The second Python command only verifies original images and the frozen data
view. To start a NEW full live five-Agent run with fresh quantitative training:

```powershell
python scripts/run_waterbirds_shortcutminer.py --live --config configs/experiments/waterbirds-shortcutminer-background-formal.json
```

The default configuration now uses this paired background protocol. Do not
resume the previously completed v7 run with the new configuration: its config,
training source, independent evaluation and summary remain historical artifacts.
This change does not relabel its official-test results as constructed OOD.

## What the tool executes

`background_ood_specs.place` is caller-frozen and scoped to the exactly bound
candidate field. The deterministic plan compiler creates the required
`ood_evaluation` step after the current task's training dependency. Providers
can request only a compiled step ID; they cannot change model paths, hashes,
object prompts or intervention parameters. The completed measurement is
finalized into Quantitative evidence, Ledger and Summarize through the normal
workflow. It is not a separate classifier baseline script.

The local, safetensors-only pipeline is:

1. Grounding DINO locates `bird.` from pixels. No label, place value, official
   foreground mask or oracle box is passed to the detector.
2. SAM ViT-B produces foreground masks using those detected boxes. Multiple
   detections are merged; low predicted mask quality or no detections fails
   explicitly, without a heuristic fallback.
3. ViTMatte refines the SAM-derived trimap. Padding is cropped off, not resized
   into the image. Its alpha protection region is united with SAM's region,
   then expanded by the frozen 8-pixel margin.
4. The intervention engine keeps all protected pixels exactly unchanged and
   replaces pixels outside that region with RGB(127,127,127), saving lossless
   PNGs at original dimensions. This is background neutralization, not a
   natural background swap. Official `test` images remain in the data view,
   but this paired measurement uses the held-out `id` images only (959 in the
   full protocol). The existing input manifest describes available original
   partitions; `background_ood_specs` specifies the intervention actually run.
5. Every ResNet18/50 seed is evaluated on the same original/modified image
   pairs. Report per-seed logits, predictions, original/counterfactual accuracy,
   signed accuracy drop, flip rate, paired bootstrap intervals, architecture
   means and sample standard deviations. Both sides are inferred once per
   member and retained for all statistics.

Models and all required processor/tokenizer files have SHA-256 identities in
`configs/experiments/waterbirds-shortcutminer-background-formal.json`:

- SAM: `E:/Beyond Post-hoc/sam-vit-base`
- ViTMatte: `E:/Beyond Post-hoc/vitmatte-small-composition-1k`
- DINO: the author's existing `D:/hf_cache/hub/models--IDEA-Research--grounding-dino-base/snapshots/12bdfa3120f3e7ec7b434d90674b3396eccf88eb`

Model loading uses `local_files_only=True`; no model download, remote custom
code, arbitrary module, or pickle checkpoint is selected by providers. Full
recipe, source and mask hashes, detector/SAM scores, protected mask, modified
image, coverage, quality checks and construction failures are stored under the
current run's `interventions/<task>/ood/<fingerprint>/` directory. Source image
content is not changed. Missing files or changed hashes fail before execution.

## Evidence limits and acceptance

Preservation status is **unverified**. Zero change inside a predicted protection
mask proves pixel preservation there; it does not prove that the mask contains
the entire true bird or that semantic labels are preserved. Independent mask
review, failure coverage and additional background conditions remain needed for
stronger scientific claims. Never optimize mask thresholds against classifier
accuracy or force an expected performance drop. Failed pairs remain explicit;
usable-subset metrics cannot be presented as full-dataset coverage.

Native-runtime checks:

- Four images spanning label/background combinations passed automatic
  DINO/SAM/ViTMatte inference and protected-pixel preservation, with no oracle
  masks. Receipt: `runs/local-segmentation-review-20261004/automatic-results.json`.
- Six fresh tiny GPU classifier models (two architectures, three seeds, one
  epoch) and four real intervention pairs per member passed the full five-Agent
  graph with fixture Agent responses. Receipt:
  `runs/local-segmentation-review-20261004/quantitative-gpu-acceptance.json`.

These are integration acceptance artifacts, **not formal paper results**.
No live provider was called for this acceptance.

Full formal paired validation completed on 2026-10-05:
`runs/waterbirds-shortcutminer-paired-20261005-v1`. All five real Agents ran,
six fresh models trained, 954 intervention pairs succeeded and 5 failed.
Independent verification passed. See the run's `results-overview.md` and original
`audit_report.md`. Predicted foreground/label preservation remains independently
unverified. Earlier superseded results were deleted at the user's request;
the current run, input view and successful acceptance evidence remain.
