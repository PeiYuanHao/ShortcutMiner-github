# E2-B independent-source Agent-blind test v3

Frozen before this run's validation access on 2026-10-09. This is a follow-up after development on v2; the researcher knows the mechanism family. It is not evidence of globally novel discovery or an entirely unadapted study. Agent inputs remain blinded to the planting rules and prior outcomes.

## Data and isolation

The private generator selects 2000 source images excluding all v2 byte/pixel hashes. The unchanged planting parameters and seed remain in `E:/datasets/E2-B/e2_truth_v3`; no OOD is generated there. Public CSV remains `img_id,img_filename,y,split`. Public discovery contains 1200 train images; model input contains train 1200, val 300, ID 500. File/pixel disjointness does not establish geographic independence.

The Agent sees `e2-case-b-input-v3` only during discovery. Construction/calibration/review use training pixels. Held-out images are opened only after acceptance. Use a new runs root and `--isolated-discovery` to exclude learned memory and old run artifacts. Generic cited literature memory/search remain available; no dataset-specific rule or previous failure is placed in the prompts.

## Method correction

Quantitative compares cue neutralization and reversal using training observations, selects one primary supported operation and explains its purpose. Python fits the numerical target/pivot label-blind from training data. Actual training pairs expose changed means, standard deviations, centered structure difference, chroma, and clipping diagnostics. Mean ordering/range checks distinguish neutralization from reversal without adding a scientific threshold. Qualitative examines actual images and Executive authorizes a scoped diagnostic. There is no forced reflection, hidden-parameter injection, model-outcome-based recipe search, or guarantee of a positive verdict.

Saved paired logits now supply auxiliary true-class margin changes for the pilot and full per-model paired reports. These are descriptive sensitivity diagnostics, not calibrated probability or substitutes for accuracy-drop certification. Legacy probe records without logits retain their existing results.

## Frozen training/statistical profile

Reuse `configs/experiments/synthetic-corner-patch-shortcutminer-memory-v4.json` solely for its training, promotion and statistical envelope. Its dataset-specific OOD fields are discarded by the generic entry. ResNet-18/ResNet-50, seeds 42/43/44, epoch/patience settings, primary delta 0.02, family alpha 0.05, sensitivity 0.01/0.02/0.05, and paired-betting-cs-v1 stay unchanged. There are at most six model profiles, conditional on the pilot gate. No total long-run experiment timeout is added. VLM output budget is 8192 tokens.

## Execution

```powershell
& 'E:\Anaconda_envs\envs\shortcutminer\python.exe' infrastructure/experiments/run_autonomous_image_audit.py --input 'E:\datasets\E2-B\e2-case-b-input-v3' --training-input 'E:\datasets\E2-B\e2-case-b-training-v3' --profile configs/experiments/synthetic-corner-patch-shortcutminer-memory-v4.json --runs-root 'E:\ShortcutMiner-github\runs\e2-b-v3-isolated-20261009' --run-id case-b-v3 --task 'Binary aerial-image classification with observed labels zero and one. Discover candidate cues from public training images; validation and ID images are reserved for evaluation. No expected mechanism, private rule, OOD recipe or result is supplied.' --isolated-discovery --validation
```

Credentials load only from `.env.formal`. Retain the source snapshot and frozen requests, discovery, construction receipts, training weights/manifests, paired predictions, stopping decision, SummaryReport and explicit failure records. Score hidden truth only outside the completed Agent workflow. An inconclusive result remains inconclusive.
