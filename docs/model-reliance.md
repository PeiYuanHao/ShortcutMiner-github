# Model Reliance

ShortcutMiner separates five claims: a dataset association exists; a cue predicts labels in a lightweight probe; a task model exhibits particular behavior; that same model changes under a controlled OOD shift; and paired predictions change when the suspected cue alone is modified. Only the latter three concern an audited task model. None automatically confirms a shortcut.

## Task-model boundary

`TaskModelAdapter` is separate from the LLM `AgentClient`. The production registry is static and currently contains only `ReferenceImageLinearClassifier`. It decodes bounded local images, resizes deterministically to 16×16 RGB, flattens values to `[0,1]`, standardizes using training statistics, and fits scikit-learn logistic regression on CPU. `reference_image_features_v1` fixes this representation. It is a runtime reference model for controlled fixtures, not a paper baseline, CNN, ViT, or main backbone.

The adapter declares predictions, probabilities, logits, training, checkpoint, image-input, and intervention-inference capabilities. `PredictionArtifactAdapter` declares predictions only and cannot retrain or infer modified images. `ExternalTaskModelAdapter` is a future contract; no arbitrary external model implementation is registered.

## Training and persistence

`task_model_training` accepts an explicit metadata file, sample ID/path/label columns, an optional train/validation/test split, bounded sample and iteration budgets, seed, and the allowlisted backend/version. Without a split, the runtime creates a deterministic stratified holdout and records it. Training IDs are disjoint from validation and test IDs.

Run-local models use:

```text
models/<model-id>/model.json
models/<model-id>/weights.npz
models/<model-id>/training_manifest.json
```

JSON records the backend/version, feature contract, classes, input identity, split IDs, configuration, convergence and metrics. NPZ stores only numeric arrays and is loaded with `allow_pickle=False`; pickle, joblib, arbitrary Torch checkpoints, module paths, commands, dynamic imports and downloads are forbidden. Dataset bytes, selected training samples and configuration form the training fingerprint. Matching artifacts resume without fitting again; identity mismatch fails rather than silently overwriting or retraining.

## Evaluation

`task_model_evaluation` loads an approved model from the same run and evaluates either an explicit split or the generated training-manifest test partition. It reports count, accuracy, balanced accuracy, per-sample predictions, probabilities/logits when available, and prediction-row bootstrap uncertainty. OOD and counterfactual backends never call `fit`.

Executive must declare `retraining` for training, `checkpoint` for task-model evaluation, and `checkpoint` plus `intervention_inference` for OOD/counterfactual work. `no_retraining` is bound into the Quantitative tool runtime and blocks training even if an Agent requests it. Missing capabilities are blocked scope, not contradictory evidence.

## Claims

The backend returns measurements and artifact identity only. There is no reliance score or magic threshold. Quantitative interprets the design and observations; Evidence Ledger preserves model/intervention provenance; Summarize must distinguish association, predictive evidence, model behavior, OOD response, and intervention evidence.
