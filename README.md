# ShortcutMiner

**Multi-agent discovery and controlled validation of dataset shortcuts.**

ShortcutMiner audits local datasets by proposing shortcut hypotheses, planning
candidate-specific checks, constructing interventions, measuring model responses,
and producing an evidence-linked report. It combines five research agents with
a deterministic execution layer built on LangGraph.

[Method](#method) · [Installation](#installation) · [Run an audit](#run-an-audit) ·
[Reproducibility](docs/reproducibility.md) · [Documentation](docs/README.md) ·
[Experiments](experiments/README.md)

## Method

| Agent | Responsibility |
| --- | --- |
| **Identify** | Propose hypotheses from approved training samples, descriptive statistics and optional bounded literature retrieval. |
| **Executive** | Select feasible candidate-specific validation paths within the declared capabilities and budgets. |
| **Quantitative** | Select executable measurements and interventions, and interpret statistical and model-level evidence. |
| **Qualitative** | Inspect approved samples and actual intervention pairs for cue changes, artifacts and task-content damage. |
| **Summarize** | Synthesize supporting evidence, counterevidence and limitations into the final structured report. |

Screening, feasibility checks, evidence promotion, tool permissions, checkpoints
and evidence calibration are deterministic controllers, not additional agents.
Agents select from typed, authorized tools rather than executing arbitrary code.

The image-audit workflow separates training-only discovery and construction from
held-out evaluation. Quantitative selects an intervention; numerical calibration,
where supported, uses training pixels only. Qualitative reviews real image pairs
and Executive authorizes the scoped diagnostic before held-out access.

The framework provides:

- **Open-ended candidate discovery:** visual observations, independent descriptor
  axes and a literature-backed method catalog, with optional bounded retrieval.
- **Candidate-bound validation:** usable hypotheses receive low-cost checks even
  when a reliable intervention cannot be constructed. Missing capabilities remain
  explicit deferrals.
- **Resource-aware model validation:** sequential paired probes can stop, defer or
  authorize further model training. Two architectures and three seeds are a
  conditional maximum in the supplied profile, not compulsory work for every cue.
- **Auditable execution:** strict message contracts, source/config/input hashes,
  saved weights and predictions, uncertainty records, SQLite checkpoints and
  candidate-level evidence provenance.
- **Evidence-gated method memory:** eligible completed runs can contribute generic
  validation methods to future runs. Hidden truth and same-run outcomes are not
  returned to that run's discovery stage.

See [architecture](docs/architecture.md), [candidate-complete validation](docs/candidate-complete-validation.md)
and [construction review](docs/representative-discovery-and-construction-review.md).

## Installation

The recorded experiment environment is **Windows AMD64, Python 3.11.16**, using
the Conda environment `shortcutminer`. Commands below run from the repository
root in PowerShell or CMD.

```text
git clone https://github.com/PeiYuanHao/ShortcutMiner-github.git
cd ShortcutMiner-github
conda create --name shortcutminer --file requirements/conda-win64.explicit.txt
conda activate shortcutminer
python -m pip install --require-hashes --no-deps -r requirements.txt
python -m pip check
```

Skip environment creation if `shortcutminer` already exists. The root
`requirements.txt` is the complete, version-pinned Windows wheel lock with
SHA-256 hashes; `requirements/` contains supplementary provenance locks.
The GPU packages are PyTorch `2.9.1+cu128` and torchvision `0.24.1+cu128`.
CPU-only smoke checks do not require an available GPU; task-model GPU experiments
require successful local CUDA acceptance. Package/model retrieval and live
providers require network access.

Run the offline workflow demo without provider credentials:

```text
python -X utf8 -m shortcutminer mock --runs-root runs/smoke --progress
```

This checks execution infrastructure using mock agents; it is not a scientific
dataset audit. See [deployment](docs/deployment.md) for environment acceptance and
[reproducibility](docs/reproducibility.md) for source-specific freeze checks.

## Run an audit

### Provider setup

Copy `.env.example` to the Git-ignored `.env.formal`, and set only the required
provider credentials there:

```dotenv
SHORTCUTMINER_TEXT_API_KEY=YOUR_TEXT_PROVIDER_KEY
SHORTCUTMINER_VLM_API_KEY=YOUR_VISUAL_PROVIDER_KEY
```

Configure endpoint, model identifier and declared capabilities in the model YAML
files under [configs/models](configs/models). Live entry points load the two keys
from `.env.formal` for the command's process; no Windows system environment setup
is required. Do not put key values in YAML, source code, reports or command lines.

### General dataset intake

```text
python -X utf8 -m shortcutminer audit DATASET_ROOT --metadata-file metadata.csv --label-column y --model-config configs/models/pilot_text.example.yaml --semantic-model-config configs/models/pilot_vlm.example.yaml --runs-root runs/audits
```

This entry discovers available validation capabilities. It does not automatically
provide an image-training protocol or a paired intervention for every dataset.
Directory intake and missing labels may produce explicit limitations; CSV-backed
image paths and labels are the supported input for the model-validation entry.

### Agent-owned image intervention and model validation

Prepare separate public discovery and training inputs, each with `metadata.csv`:

```csv
img_id,img_filename,y,split
sample-0001,images/train/sample-0001.png,0,train
```

The discovery input contains **train only**. The training input contains
**train/val/id**; its train rows and images must match discovery exactly. Paths
are relative to the corresponding input root. Validation selects checkpoints;
reserved probe and final-confirmation sources remain separate. No OOD images,
expected shortcut or hidden truth are supplied to discovery.

The following command **calls real providers and may train GPU models**:

```text
python -X utf8 infrastructure/experiments/run_candidate_complete_image_audit.py --input PUBLIC_DISCOVERY_ROOT --training-input PUBLIC_TRAINING_ROOT --profile configs/experiments/image-audit.example.json --runs-root runs/image-audits --run-id audit-001 --task "Image classification; y is the target class ID." --isolated-discovery --representative-review --validation
```

Use a fresh run ID. Review the example profile's budgets, model settings and
resource requirements before freezing a run. It contains no dataset-specific
intervention. `--isolated-discovery` excludes learned prior-run methods;
`--representative-review` enables bounded class-representative observation and
training-only intervention-quality review. Local segmentation-backed background
construction has separate prerequisites; see [background OOD](docs/local-background-ood.md).

## Outputs and interpretation

Completed audit runs preserve:

| Artifact | Purpose |
| --- | --- |
| `audit_report.md` | Concise English view of finalized findings and executed checks. |
| `audit_report.technical.md` | Detailed methods, limitations and evidence references. |
| `audit_report.json` | Structured report built from persisted scientific records. |
| `summarize/result.json` | The actual Summarize Agent output. |
| `evidence/ledger.json` | Evidence direction, uncertainty, coverage and provenance. |
| Models, predictions, interventions and checkpoints | Execution evidence and recovery inputs, where those steps executed. |

General CLI audits write under their run directory; image research entries also
save separately identified discovery, construction and derived-input directories.
Historical display corrections may be retained as `audit_report.readable.md`
without replacing the original reports.

Dataset association, model sensitivity and a nuisance shortcut are distinct
claims. A performance drop after intervention may also reflect damage to valid
task information. `partially_supported`, `inconclusive` and `insufficient_evidence`
are substantive outcomes; workflow completion is not shortcut confirmation.
For a concrete example with six model/seed profiles, see the
[retained E3 texture audit](docs/experiments/e3-texture-recheck.md).

## Repository map

| Directory | Role |
| --- | --- |
| `shortcutminer/` | Core contracts, execution, runtime, storage and LangGraph workflow. |
| `agents/` | Five agent workspaces and operating instructions. |
| `tools/` | Bounded data inspection and experiment tools. |
| `configs/` and `requirements/` | Runtime examples, scientific profiles and dependency locks. |
| `infrastructure/experiments/` | Shared image-audit research entry points. |
| `experiments/` | Isolated experiment workspaces, including E4 ablations. |
| `evaluation/` | Controlled evaluation and blind-scoring infrastructure. |
| `tests/` | Contract, execution, security, statistical and recovery tests. |
| `scripts/` | Dataset-specific runners, acceptance utilities and historical scaffolds. |
| `docs/` | Method, reproduction, evidence summaries and historical documentation. |
| `runs/` | Git-ignored local outputs; E4 outputs use `runs/e4_ablation/`. |

Raw datasets, credentials, checkpoints and full run outputs are not distributed
in this source repository. Dataset roots and optional local model assets must be
provided separately. The [experiment index](experiments/README.md) distinguishes
retained evidence from experiment scaffolds; E4 configurations and results are
not claimed by the workspace alone.

## Development checks

```text
python -X utf8 -m pytest -q
```

Tests establish implementation behavior rather than paper-level discovery
performance. Reproduce scientific findings from their original source snapshot,
configuration, data splits and evidence bundle. Do not silently rerun a historical
checkpoint with current defaults. Start with the
[reviewer guide](docs/reproducibility.md) and [documentation index](docs/README.md).
