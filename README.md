# ShortcutMiner

ShortcutMiner is a multi-agent framework for automated dataset shortcut discovery, validation, and evidence synthesis.

ShortcutMiner is under active research revision and experimental development. The repository currently provides a tested orchestration, contract, storage, tool, and local-container foundation; it is not a finished experimental evaluation or a release-ready research artifact.

## Overview

Dataset shortcuts are unintended signals that a model can exploit instead of the intended task signal. ShortcutMiner organizes an audit into five traceable stages:

- discover potential shortcut candidates;
- plan suitable validation paths;
- execute quantitative and qualitative audits;
- combine supporting and counter-evidence;
- produce a structured, traceable report.

The deterministic Python control layer owns execution, validation, storage, checkpoints, and tool permissions. Agent outputs cannot directly change workflow control or access unrestricted runtime capabilities.

## Architecture

```text
Dataset / Audit Request
          |
          v
        Runner
          |
          v
Generic Dataset Intake
          |
          v
    DatasetProfile
          |
          v
     Orchestrator
          |
          v
       LangGraph
          |
          v
       Identify
          |
          v
      Executive
       /      \
      v        v
Quantitative  Qualitative
      \        /
       v      v
 EvidenceLedgerBuilder
          |
          v
       Summarize
          |
          v
     SummaryReport
```

Runner and Orchestrator are deterministic control components, not agents. ShortcutMiner has exactly five research agents. LangGraph provides the typed execution graph: Executive fans out to Quantitative and Qualitative. At the join, deterministic evidence calibration builds a threshold-free `EvidenceLedger` before Summarize runs; it is infrastructure, not a sixth Agent.

The production control path accepts user-provided local datasets through a generic intake contract; dataset names are not selected from a benchmark whitelist. Intake performs bounded, deterministic inspection of supported local structures and CSV metadata, emits a strict `DatasetProfile`, and gives that factual profile to Identify as untrusted evidence. It does not infer or rank shortcut candidates. Unknown or unsupported structures are reported through warnings and limitations rather than being treated as a known benchmark.

For image or mixed datasets, Identify can also receive a real `VisualDiscoveryProfile`: safely decoded low-level descriptors, explicit coverage, and label-aware descriptive summaries produced by bounded CPU infrastructure. Metadata-free image folders use deterministic bounded sampling without inferring labels from directory names. Descriptor ranking is only screening salience, never statistical validation or a shortcut verdict. Optional provider-backed semantic vision adds strictly scoped visible observations while leaving candidate generation with Identify. See [Visual Discovery](docs/visual-discovery.md) and [Semantic Vision](docs/semantic-vision.md).

See [Architecture](docs/architecture.md) for component boundaries and persistence behavior.

## Five Agents

| Agent | Responsibility | Formal output |
|---|---|---|
| Identify | Discover evidence-grounded shortcut candidates | `CandidateReport` |
| Executive | Validate candidates and produce an executable routing plan | `RoutingPlan` |
| Quantitative | Interpret controlled experiment and statistical evidence | `QuantitativeResult` |
| Qualitative | Inspect approved samples, metadata, and artifacts | `QualitativeResult` |
| Summarize | Synthesize upstream evidence and limitations | `SummaryReport` |

Detailed operating behavior is maintained in each Agent workspace and the [communication protocol](docs/protocol.md).

## Workflow State and Contracts

All inter-Agent communication uses strict, serializable schemas. The workflow does not rely on hidden sessions, provider-specific memory, function objects, file handles, or model clients in checkpoint state.

The principal output contracts are:

```text
DatasetProfile
CandidateReport
RoutingPlan
QuantitativeResult
QualitativeResult
EvidenceLedger
SummaryReport
```

Each message carries explicit run and stage correlation data. Executors validate schema and cross-object invariants before results enter workflow state or Storage. SQLite checkpoints support local single-machine resume.

## Tool Runtime

ShortcutMiner owns a provider-independent Tool Runtime and Tool Registry. The current registry contains:

```text
inspect_tool_capabilities
inspect_dataset
inspect_metadata
get_column_distribution
select_samples
inspect_sample_metadata
inspect_sample
read_artifact
read_run_trace
run_experiment
```

The Registry performs registration, schema publication, input/output validation, permission enforcement, bounded execution, and structured result creation. Runtime tracing is written through the existing Storage and Event systems. Tool permissions are enforced by ShortcutMiner rather than delegated to model output.

See [Tool Runtime](docs/tool-runtime.md) for the exact permission matrix and file-access boundary.

## Current Tool Permission Model

- **Identify** can inspect the approved dataset, metadata, distributions, and samples to support candidate discovery.
- **Executive** can inspect available capabilities and approved run information, but cannot run experiments.
- **Quantitative** is the only Agent permitted to call `run_experiment`; it cannot execute arbitrary shell or Python code.
- **Qualitative** alone can call `inspect_sample` for allowlisted, bounded local
  descriptors of already selected samples; it can also inspect approved metadata
  and artifacts without mutating upstream data.
- **Summarize** can read approved artifacts and run traces for report synthesis, but cannot run experiments or mutate datasets.

All file access is root-constrained, relative-path validated, allowlisted where appropriate, and bounded by runtime budgets.

## Generic Dataset Intake

A formal YAML/JSON run request can describe a new dataset without an adapter:

```yaml
run_id: my-audit
runs_root: runs
dataset_info:
  name: my_dataset
  root: /data/my_dataset
  metadata_file: metadata.csv  # optional
task_input:
  type: classification
  label_column: label
seed: 42
identify_config:
  budget:
    max_dataset_entries: 10000
    max_metadata_rows: 100000
    max_samples: 40
```

The current profiler recognizes image-, tabular-, and text-like structures using deterministic file-extension heuristics. CSV is the currently supported metadata parser. Metadata-free intake remains valid, but the profile explicitly records that labels are unavailable when no other supported label source exists. Each run persists the result at `dataset/profile.json`; resume validates and reuses that profile instead of rescanning.

## Experiment Runtime

`ExperimentExecutor` is the typed boundary between `run_experiment` and scientific execution. `MockExperimentExecutor` remains available for deterministic unit tests. `RealExperimentExecutor` is the local CPU backend used by configured live CLI runs.

The real backend supports bounded metadata/tabular statistical association, subgroup label or prediction evaluation, out-of-sample logistic-regression probes, metadata permutation, metadata-feature probe ablation, and an allowlisted prediction metric computation. Every experiment uses an experiment-specific strict parameter schema and writes a reusable artifact under `experiments/<task_id>/`. Supported metrics carry bounded, seeded percentile-bootstrap uncertainty; unavailable intervals include an explicit reason. Statistics, raw/adjusted p-values, effect sizes, predictive accuracy, and intervention deltas are evidence—not automatic shortcut decisions.

The evidence ledger is stored at `evidence/ledger.json`. It preserves direction,
execution state, coverage, uncertainty, provenance, artifacts, failures,
counterevidence, and shared input families separately. Benjamini-Hochberg FDR is
applied transparently within candidate-level related-test families while retaining
raw p-values. Neither intervals nor adjusted p-values trigger a shortcut verdict.
See [Evidence calibration](docs/evidence-calibration.md).

The quantitative backend is CSV-first and CPU-only. It does not perform large
neural-model retraining, image intervention, or arbitrary Python/shell execution.

## Blind Evaluation Infrastructure

The repository includes an evaluation-only blind challenge framework for
open-ended shortcut discovery. It generates deterministic metadata, spatial,
acquisition, mixed, and clean-control challenges with a strict physical firewall
between the public dataset and private ground truth. Private truth is loaded only
after the production workflow has completed, and leakage tests require identical
production outputs when that truth is present, absent, or replaced.

The framework provides exact candidate-to-truth matching, optional manual and
multi-rater adjudication, discovery/localization/funnel/false-positive metrics,
snapshot-frozen batch aggregation, and independent-challenge bootstrap intervals.
It also provides a sequential, resumable repeated-audit matrix with raw discovery,
localization, final-assessment, false-positive, and runtime-reliability distributions.
Controlled smoke fixtures validate the infrastructure only; they are not a
ShortcutMiner performance claim. See [Blind shortcut discovery evaluation](docs/blind-evaluation.md)
and [Repeated Blind Evaluation](docs/repeated-evaluation.md).

## Qualitative Evidence Runtime

Qualitative preparation supports deterministic `random` sampling without metadata
or grouping, and `group_balanced` sampling only when explicit group columns are
configured. CSV-backed tasks may inspect linked image/text samples or selected
tabular rows; metadata-free datasets use a bounded, root-constrained scan of
supported files.

The real local image backend uses Pillow and NumPy on CPU to compute dimensions,
mode/channels, intensity and RGB summaries, 256-bin grayscale entropy, deterministic
edge density, and fixed center/border/side/corner statistics. It handles corrupt,
oversized, decompression-bomb, truncated, and multi-frame inputs as explicit partial
or failed evidence rather than crashing the task. Text previews and scanning are
bounded by task budgets, and all dataset-derived strings are marked as untrusted.

Each persisted task writes
`qualitative/<task_id>/inspection.json` and, when inspectable images exist, a fixed
bounded `contact_sheet.png`. Matching artifacts are validated and reused on retry
or resume. The backend produces sample facts, group summaries, and descriptor
differences only; the Qualitative Agent interprets evidence status and Summarize
performs final synthesis.

Current limits are deliberate: provider-backed semantic vision is optional and
observation-only; there is no OCR guarantee, image editing intervention, segmentation,
or neural visual representation benchmark. Agents receive bounded descriptors and
structured observations, never unrestricted local files.

## Repository Structure

```text
ShortcutMiner-github/
├── agents/                       # Five ShortcutMiner workspaces
├── configs/
│   ├── datasets/                 # Evaluation/example descriptors, not a runtime whitelist
│   ├── experiments/              # Research and local-run configurations
│   └── models/                   # Explicit model endpoint configurations
├── data/                         # Dataset-mount guidance; no raw datasets
├── docs/                         # Architecture, protocol, tools, and Docker docs
├── infrastructure/docker/        # Container image definition
├── scripts/                      # Acceptance helpers and future research entry points
├── shortcutminer/
│   ├── experiment/               # Stage executors and experiment boundary
│   ├── protocol/                 # Structured contracts
│   ├── runtime/                  # AgentClient implementations
│   ├── storage/                  # Run, event, artifact, and checkpoint persistence
│   ├── tooling/                  # Tool protocol, registry, runtime, and built-ins
│   ├── workflow/                 # LangGraph state, nodes, graph, and checkpoint
│   ├── agent_client.py
│   ├── cli.py
│   ├── orchestrator.py
│   └── runner.py
├── tests/                        # Unit, contract, security, and workflow tests
├── tools/                        # Reusable bounded data/evidence operations
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

Some research-oriented files remain explicit placeholders. The tested entry point is `python -m shortcutminer`, not the placeholder matrix or aggregation scripts.

## Agent Workspaces

Every Agent owns the following files under `agents/<agent>/workspace/`:

```text
AGENTS.md
IDENTITY.md
SOUL.md
USER.md
TOOLS.md
MEMORY.md
HEARTBEAT.md
```

They are loaded in this stable order:

```text
AGENTS -> IDENTITY -> SOUL -> USER -> TOOLS -> MEMORY -> HEARTBEAT
```

These documents define ShortcutMiner's own stable Agent behavior, responsibilities, evidence limits, and tool policy. They are loaded explicitly by the workspace loader and are included in consistency tests.

## Docker: Recommended Validation Path

The current supported environment is local Docker Desktop or Docker Engine. From the repository root:

```bash
docker compose build
docker compose run --rm shortcutminer pytest -q
```

Run the complete offline synthetic five-Agent workflow:

```bash
docker compose run --rm shortcutminer python -m shortcutminer mock --run-id local-smoke --runs-root /runs
docker compose run --rm shortcutminer python scripts/container_acceptance.py local-smoke
```

The default Compose network mode is `none`. The mock workflow requires no credentials, real dataset, model API, or external network. Build-time access is needed only to obtain the base image and Python dependencies.

Compose mounts its named run volume at `/runs`, `./data` read-only at `/data`, and `./configs` read-only at `/config`. Do not use `docker compose down -v` unless deleting the local run volume is intentional.

See [Local Docker Usage](docs/deployment.md) for persistence and resume checks.

## Optional Local Python Development

Python 3.11 or newer is required. Conda and `venv` are both suitable:

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

## Current Development Validation

Run the current Phase 10 validation with:

```bash
python -m pytest -q
docker compose build
docker compose run --rm shortcutminer pytest -q
```

Latest verified results:

```text
Host:   406 passed, 6 skipped
Docker: 412 passed
```

Automatic tests use deterministic mock Agent clients plus synthetic data for the
real quantitative and qualitative backends. They do not call real LLM APIs or run
expensive training.

## Project Status

Implemented and tested:

- five-Agent LangGraph orchestration;
- structured Agent and run contracts;
- deterministic workspace loading;
- AgentClient abstraction and offline mock client;
- Tool Runtime, Registry, budgets, and per-Agent permissions;
- real CPU quantitative experiments for bounded tabular/metadata evidence;
- real CPU qualitative image/text/tabular evidence with sample traceability,
  descriptor comparisons, contact sheets, and resume reuse;
- run, artifact, event, and final-report storage;
- SQLite checkpoint and resume behavior;
- local Docker build, mock execution, and container tests.

Not yet implemented as production research capability:

- provider-native multimodal/VLM qualitative interpretation;
- provider-native LLM function/tool-call adapters;
- the full ICDM experimental evaluation;
- large-scale dataset/model/seed experiment matrices.

## Datasets and Research Assets

Large raw datasets are not committed to GitHub. Waterbirds, Camelyon17, Colorized-MNIST, and other real datasets must be stored locally and supplied through a configured dataset root, normally as a read-only Docker volume. Small synthetic fixtures used by tests remain in the repository where appropriate.

Run outputs, checkpoints, raw artifacts, credentials, and local dataset directories are ignored by Git. Research and experiment assets may be maintained separately in the future; no external asset repository is declared yet.

## Development Scope

The repository is currently focused on validating the local research architecture before beginning full experiments. Contributions should preserve the separation between Agent reasoning, structured contracts, deterministic execution, tool permissions, and durable evidence. Large-scale research claims should only be added after their corresponding real backends and evaluations exist.
