# E2: blind discovery with autonomous counterfactual validation

## Roles and blinding

The private dataset generator only selects examples, splits data, implants the
shortcut, and records hidden truth. It must not generate OOD images, paired
counterfactual images, pristine comparison images, pair indexes, intervention
recipes, target regions, or expected verdicts for ShortcutMiner.

The discovery input contains training images, task labels, opaque sample IDs,
relative image paths, and a neutral task description. Training/validation/ID
resources contain the same ordinary schema. Validation and ID resources must
not be opened by the intervention planner before its proposal is frozen.
Hidden truth, generation code, masks, and researcher previews remain private.

After discovery, Quantitative proposes the candidate-specific validation:
operation, region/mask strategy, parameters, invariants, processing controls,
semantic-preservation checks, and limitations. Its signed proposal and evidence
references must be persisted before held-out inference. The executor implements
that proposal on approved input images and saves new paired artifacts under the
run directory. It must never recover original content or complementary images
from private generation resources.

A versioned, generic tool catalog or memory is allowed. Supplying an available
tool is not choosing the experiment. The controller must not select a watermark
removal recipe, supply its coordinates, prescribe a color swap, or silently
substitute a fixed operation based on the dataset identity. A proposal can be
revised using training-only construction diagnostics within a frozen budget;
model outcomes and hidden truth must never be used for revision. Unsupported
operations or failed preservation checks must defer with an explicit reason.

## What remains researcher-controlled

Resource budgets, model families/seeds, training protocol, confidence procedure,
the minimum meaningful effect, and maximum selection/revision budget are frozen
evaluation machinery. They are not claims of Agent-invented methodology.
Quantitative selects a candidate-specific intervention through that machinery.

The existing early-stop policy can remain: reserve disjoint probe and final
confirmation groups, expand only under the frozen rule, and report all failures.
Do not force a positive result. Pixel preservation outside a region is a
structural check; it does not guarantee semantic label preservation underneath
an occluded region.

Hidden truth is opened only by a separate external scorer after the production
workflow completes. Candidate matching, localization, duplicate mechanisms,
false positives, and model-dependence findings are recorded separately. Truth
scoring must not rewrite the original Agent outputs or feed back into that run.

## E2-A data version

Private generator:
`E:\datasets\E2-A\e2-private\case-a-v2\prepare_dataset.py`.

- `e2-case-a-input-v2`: 3,000 training images for discovery.
- `e2-case-a-training-v2`: train 3,000 / val 600 / ID 800 ordinary images.
- `e2-private\case-a-v2`: generation code, hidden truth, researcher previews,
  hashes, and dataset integrity checks; never an Agent input.

There is no `ood` split and no paired counterfactual export. The generator only
implements the cat/dog 90/10 watermark implantation; intervention ownership is
reserved for ShortcutMiner. The version is separate to preserve prior results.

## Prior result status and current execution gap

`e2-a-shortcutminer-validation-20261007-v1` used pre-generated paired OOD.
It remains evidence of blind cue discovery followed by external OOD validation,
but is **not** an E2 autonomous-intervention result. Preserve its artifacts;
exclude it from tables measuring autonomous intervention construction.

`runs/e2-a-control/run_complete.py` is a legacy external-pair entry point. Do not
run it for the revised E2 protocol. Its frozen candidate replay and provided-pair
binding must not be reused as a new autonomous proposal.

The v2 trial entry is `runs/e2-a-control/run_autonomous_v2.py`, preceded by
`run_discovery_v2.py` for fresh blind discovery. Quantitative proposes a typed
generic spatial mask/replacement, region, fill, and seed from training evidence,
or defers. The full proposal is frozen before held-out access. Construction
diagnostics check training pixels and a bounded region, then the production
workflow trains and creates its own paired images through InterventionEngine.
The input guard accepts only train/val/ID and rejects external-pair recipes.

The spatial trial checks exact no-op re-encoding logits and unchanged pixels
outside the selected region. It does not reconstruct hidden content or certify
semantic preservation inside the region. Report that limitation and retain
qualitative counterexamples; a performance drop alone is not causal proof.
This opt-in path does not change the frozen recipes of prior datasets.
# Training-only spatial construction optimization (spatial-quality-v3.1)

This opt-in revision addresses a construction gap: changing some pixels does not
establish that the operation covers the target localized by the Agent. The
Quantitative Agent first declares a normalized `target_box` from training discovery
and observations in a separate localization call, before seeing operation presets.
The subsequent operation-selection call must retain that frozen localization.
Grid column/row indices are explicitly converted to normalized x/y coordinates.
Tool coordinates
are explicit: a corner preset spans only 25% of each image dimension. No preset
meaning, legacy memory recipe, scientific effect threshold or early-stop rule changes.

Before any task-model training or held-out access, deterministic checks require
full coverage of the declared box, an intervention area at most half the image,
identical protected pixels, and a nontrivial change across the training preview.
Records include actual pixel bounds, uncovered target pixels, source hashes and
original/mask/transformed previews. A failed proposal returns training-only
geometry feedback to the Agent for at most two operation revisions. Candidate,
descriptor and target localization remain frozen during repairs; shrinking the
target to pass is rejected. Exhaustion defers execution without a fallback.
Resume validates proposal/input identities and training structural checks.

Coverage is **not** certification of correct localization, complete cue removal,
label preservation or artifact-matched controls. These remain explicit limitations.
The checker does not read hidden truth, model accuracy, validation/ID images or
earlier validation results. Spatial operations sent to the shared construction
checker without a declared target now fail instead of silently receiving generic
color-recipe acceptance. Existing non-spatial dispatch remains unchanged.

Windows entry point, using the existing frozen v2 discovery only:

```powershell
& 'E:\Anaconda_envs\envs\shortcutminer\python.exe' infrastructure/experiments/run_e2_spatial_quality.py --proposal-only
```

This loads credentials through the existing `.env.formal` loader and asks the real
Quantitative Agent for a training-only proposal. Output goes to
`runs/e2-a-v31-quality-control`; previous v2 proposals and results are preserved.
No model training or held-out inference occurs with `--proposal-only`.
An explicit `--development-validation` flag enables the subsequent existing audit
pipeline under a separate run ID. Because v2 held-out results have already been
observed, reuse is labelled development validation, not fresh blind confirmatory
E2 evidence. A fresh confirmatory evaluation needs unseen held-out data and a
proposal frozen before access.
