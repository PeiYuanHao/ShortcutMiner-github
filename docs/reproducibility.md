# Reproducibility and reviewer guide

This guide separates running the current code from reproducing an earlier
experiment. A new checkout, a passing test suite and a completed audit do not
retroactively certify a historical finding.

## 1. Identify the evidence being inspected

Use the [experiment index](../experiments/README.md) to locate the appropriate
runner or workspace. For a retained finding, identify its run ID, canonical
SummaryReport, input/split identities, execution profile and original source
snapshot. Record whether it is known-shortcut rediscovery, Agent-blind discovery,
a negative control, a development recheck, or a model-only baseline.

Source code and dependency locks are version controlled. Raw data, local provider
credentials, model weights and full run directories are excluded from Git.
Therefore a repository clone alone does not contain every experimental artifact.
Retained local run IDs in documentation are provenance references, not downloadable
artifact URLs. No complete public paper-results bundle is declared here.

## 2. Recreate the recorded Windows environment

From the repository root, in PowerShell or CMD:

```text
conda create --name shortcutminer --file requirements/conda-win64.explicit.txt
conda activate shortcutminer
python -m pip install --require-hashes --no-deps -r requirements.txt
python -m pip check
python -c "import sys, torch, torchvision; print(sys.version); print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
```

Create the environment only when it does not already exist. Run the package from
the repository root with `python -m shortcutminer`. The wheel lock targets Windows
AMD64 / Python 3.11; it is not a Linux lock. Optional segmentation assets must be
obtained and configured separately. Initial dependency and pretrained-weight
retrieval can require network access.

The strict `scripts/formal_environment.py check` compares a particular recorded
environment/source manifest, including source/config hashes and an absolute
Conda prefix. It may correctly reject a newer checkout or another installation
path. Do not silently refresh that manifest to make a historical run pass. Review
and freeze a separate execution identity for a new study; retain older manifests.
This documentation cleanup does not rewrite environment or historical run hashes.

## 3. Check infrastructure without real providers

```text
python -X utf8 -m shortcutminer mock --runs-root runs/smoke --progress
python -X utf8 -m pytest -q
```

The mock command uses synthetic inputs and mock agents. It does not load provider
credentials or establish a shortcut discovery result. Tests may generate ignored
temporary/run files. The strict acceptance wrapper has additional source-freeze
requirements described above; it is distinct from this portable smoke command.

## 4. Supply public data without the answer key

The candidate-complete image entry expects two directories, each with a CSV whose
header is exactly:

```csv
img_id,img_filename,y,split
```

| Input | Contents |
| --- | --- |
| Discovery root | Train rows/images only. |
| Training root | Matching train rows/images plus val and id rows/images. |

Sample IDs must be unique, image paths must be safe relative paths, and train
records must match exactly between the two views. Use disjoint source images
across splits. Exact pixel deduplication does not prove physical-instance or
scene independence; stronger grouping must be supported by actual source records.

For blind evaluation, keep planting rules, watermark states, hidden coordinates,
private truth and post-run scoring outside both public inputs. Do not prescribe
an OOD operation or expected shortcut in the task description. Evaluation OOD is
constructed by the authorized quantitative path after a training-only proposal
has been reviewed and frozen. The held-out images are not used to choose a
successful operation or perturbation strength.

## 5. Freeze configuration before execution

The dataset-neutral [image profile](../configs/experiments/image-audit.example.json)
copies the existing model/training and paired-probe settings. It supplies a
concrete example, not an objectively optimal protocol or an E4 ablation design.
It contains no dataset root, shortcut field, planting ratio or predefined OOD.

Review the model endpoint YAML files, data task, resource requirements and profile.
Keep API keys only in ignored repository-root `.env.formal`. Existing live entry
points load credentials into their command process. Use a fresh run ID and an
isolated run root. Follow the live command in the [README](../README.md#run-an-audit).

The supplied model envelope is ResNet18/ResNet50, seeds 42/43/44. The paired gate
uses a 2 pp meaningful-drop boundary, 5% candidate-family budget, descriptive
1/2/5 pp sensitivity and at most 1,500 reserved probe groups, subject to available
held-out data and the configured reserve fraction. Training expansion is
conditional. Fewer trained profiles after stopping or deferral are valid outcomes.
These settings are predeclared protocol choices, not thresholds learned from
the same held-out results. See [confidence-sequence assumptions](paired-confidence-sequence-v1.md).

## 6. Read the actual execution evidence

| Evidence | What to check |
| --- | --- |
| Source identity and snapshot | Actual commit/dirty snapshot, archive hash and original configuration. |
| Discovery and construction records | Candidate ownership, approved input coverage, operator selection, calibration, actual paired previews and limitations. |
| Screening and promotion | Why each candidate stopped, deferred or received expensive validation. |
| Training profiles | Actually trained architectures/seeds, validation-selected checkpoints and weight hashes. |
| Paired predictions and images | Source/label pairing, usable/failed counts, controls where applicable, and probe/final separation. |
| Uncertainty | Effect direction, sampling unit, method, multiplicity scope and assumptions. |
| EvidenceLedger and SummaryReport | Evidence consistency, final assessments and unresolved confounds. |
| Events and telemetry | Provider retries, failures, executed tasks, runtime and measured cost. |

Mean ± seed standard deviation is not a confidence interval. Multiple models on
the same paired images do not multiply the independent sample count. Two ResNet
architectures do not establish diversity across unrelated model families.

The E3 [evidence summary](experiments/e3-texture-recheck.md) illustrates a full
conditional model matrix and a **partially supported** finding. Stable sensitivity
to smoothing does not isolate nuisance texture from valid breed information.

## 7. Resume and preserve history

General CLI runs expose `python -m shortcutminer resume --help`. Research entries
revalidate frozen inputs/configuration/source when invoked for the same run.
Resume must preserve the original run root, run ID, source and configuration.
An incompatible source or construction requires a new run, not a patched receipt.
Completed-run display exports should not be confused with new model execution.

Keep canonical scientific files unchanged. A presentation correction can be saved
as a separately identified `audit_report.readable.md`, with its renderer/hash and
relationship to the canonical report recorded. Do not replace a previous finding
with the result of a newer code version under the same identity.
