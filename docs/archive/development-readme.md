> Historical documentation, archived during the reviewer-facing repository cleanup on 2026-10-15.
> Installation guidance, implementation status and phase-specific statements below may be obsolete.
> Use the [current README](../../README.md) and [documentation index](../README.md).
> Relative links were adjusted for this archive location. Original README SHA-256: `b69e2a7a60c91e5a92e7b48ead9151a8ae309284b4672730d14fcd160b7e51be`.

# ShortcutMiner

## Dataset audit entrance (Phase 17 Step 1)

Use the native Windows conda environment `shortcutminer` and install the standalone
`requirements.txt` with `python -m pip install --no-deps -r requirements.txt`.
Configure the text model once using an existing model-runtime JSON/YAML file; API
credentials are loaded automatically from repository-root `.env.formal` into the
current CLI process only. No Windows system environment setup is needed:

```powershell
$env:SHORTCUTMINER_MODEL_CONFIG = "configs/models/pilot_text.example.yaml"
shortcutminer audit ./dataset
```

The equivalent module command is `python -m shortcutminer audit ./dataset`.
No experiment configuration, shortcut name, metadata-field list, or task JSON is
required. `--model-config` can override the model settings for an invocation.
Optional inputs include `--description`, `--metadata-file` (relative to the dataset
root), `--label-column` (an exact column), or `--label-source class_directory` /
`--label-source split_class_directory`. `--runs-root`, `--run-id`, `--progress`, and
`--semantic-model-config` reuse existing execution facilities.

The entry builds an `AuditRequest`, performs bounded local intake, constructs the
existing `RunRequest`, and calls the existing Runner/LangGraph. It never constructs
AnalysisTask, ExperimentPlan, or quantitative/qualitative tasks. Identify and
Executive retain discovery and task-generation ownership. Other CLI commands remain
available for internal runs, pilots, regression tests, and recovery.

Intake inspects at most 10,000 filesystem entries, records at most eight metadata
files, and reads at most 65,536 bytes plus one truncation sentinel for a selected
CSV header. Symlink directories are not followed. A unique CSV source with one
conventional `label`, `target`, or `class` column can be bound automatically; multiple
sources/label columns and incomplete scans remain unresolved unless the user supplies
an exact source. Other table formats are discovered but the existing workflow reader
supports CSV only. Image, tabular, text, mixed, and unknown layouts are recorded.

Complete `class/image` and `split/class/image` layouts are recorded as directory
label sources; generic image folders and inconsistent layouts remain unresolved.
This records observed conventions, not a claim about the intended scientific target.
Step 1 does not synthesize directory-label metadata: the existing profiler still
needs CSV labels. This boundary is recorded as `directory_labels_not_bound`. Without
a resolved CSV label column, metadata remains recorded in AuditRequest but is not
passed as executable metadata, because the existing profiler requires a label column.
An internal unavailable-label marker satisfies TaskSpec without guessing a real field.
Intake limitations enter the task description, Executive context, and run metadata.

Each CLI audit persists `runs/<run_id>/audit_request.json` and the existing run
manifest, checkpoints, evidence, and `final_result.json`. Existing resume uses those
persisted inputs. Step 3 adds automatic validation capabilities below;
`audit_report.json` / `audit_report.md` remain later steps. Step 1 does not claim that
an entry-only run validated shortcuts. No changes to the five Agent workspaces or
Phase 16 evidence/tool contracts are required.

## Candidate Screening Gate (Phase 17 Step 2)

The Step 2 graph runs `Identify -> Screening -> Executive -> Feasibility ->
(Quantitative || Qualitative) -> EvidenceCalibration -> Summarize`. Screening is a
deterministic Python controller, not a sixth Agent. It makes no model/tool calls and
does not read dataset files, hidden truth, or future validation results. Its inputs
are the original CandidateReport, public DatasetProfile, and optional public initial
confidence/cost signals. Relative cost units estimate initial inspection work;
they are not API prices, training plans, or predicted runtimes.

The default v1 policy screens out candidates with known initial confidence below
0.5, unavailable required metadata, or insufficient public sample/evidence access.
Missing confidence stays `null`: the current Identify contract has no confidence
field, and priority is never converted into a probability. Defaults do not impose
an additional candidate/cost cap. Optional internal `identify_config.candidate_screening`
contains only `policy` and `signals`; no additional CLI inputs are required.
Policy can set `min_confidence`, `min_samples`, `max_promoted`, `max_candidate_cost`,
and `max_total_cost`. Signals accept `confidence_by_candidate` and
`cost_by_candidate` from public pre-validation observations only. Future experiment
results cannot be supplied as screening inputs. Candidate confidence generation
and automatic experiment planning are outside Step 2.

Original Candidate IDs and hypotheses remain unchanged. Executive receives only
`promote` candidates. When none are promoted, Python uses the existing Executive
finalizer to produce an empty plan, without Executive/validation provider calls.
The full original candidate report remains available to evidence aggregation and
Summarize. `screened_out` means validation was deferred under the current policy;
it never means contradicted, rejected, or confirmed. Scientific assessments still
require the existing evidence checks.

Decisions and categorical reasons (`weak_signal`, `insufficient_metadata`,
`insufficient_samples`, `unsupported_validation`, `resource_budget`) persist in
`runs/<run_id>/screening/result.json`. Canonical input/result hashes and v1 policy
identity allow matching decisions to be reused after interruption; changed inputs,
foreign IDs, and corrupt hashes are rejected. Checkpointed stages are not rerun.
An older checkpoint resuming at Executive passes through the same controller if
screening state is absent; already-executed legacy stages are not retroactively changed.

The final `SummaryReport` in `final_result.json` includes a Python-owned
`payload.candidate_screening` section with `candidates_generated`, `promoted`,
`screened_out`, and all per-candidate reasons. Screened-out candidates retain empty
evidence records and an evidence-based `insufficient_evidence` finding; their
separate screening decision explains why validation did not run. No Agent workspace,
CandidateReport, ToolCall, EvidenceLedger, or quantitative/qualitative result contract
changes are needed.

## Automatic audit capabilities (Phase 17 Step 3)

The same graph now runs `Identify -> Screening -> Capabilities -> Executive ->
Feasibility -> (Quantitative || Qualitative) -> EvidenceCalibration -> Summarize`.
Capabilities is a deterministic Python controller, not an Agent or a second
workflow. Legacy configured run/pilot requests pass through it unchanged. For
`shortcutminer audit DATASET_PATH`, it derives Executive input from completed
AuditRequest intake, the bounded public DatasetProfile, promoted candidates,
registered tool permissions, and configured backend declarations. No user field
list, requested checks, AnalysisTask JSON, or experiment configuration is needed.

Executive receives promoted hypotheses, a public DatasetProfile, and generated
ExecutiveCapabilities: exact available metadata fields, sample/image access,
available validation methods, supported executable experiment types, and explicit
limitations. It still emits ExecutiveAgentDraft; the existing Python finalizer
alone produces canonical candidate/task IDs, scientific field bindings, and
runtime constraints. Registered `run_experiment` without an installed executor
does not advertise quantitative validation.

Metadata authority is the intersection of intake-confirmed exact headers and
bounded profile columns. Reserved private truth fields are excluded; arbitrary
workflow metadata, hidden evaluator files, runtime histories, and future results
never authorize fields. Label, ID, path, and split columns remain addressing/control
inputs, excluded from scientific candidate binding. An unambiguous conventional
ID/path column can supply execution addressing; ambiguous columns remain unbound
with a limitation. Candidate prose never invents a field such as `site_device_batch`.

Resolved CSV labels and scientific metadata can enable association, subgroup,
metadata-probe, and metadata-intervention checks on installed CPU backends.
Image/text sample access enables existing qualitative inspection; metadata-free
datasets use the existing bounded random sampling path. Directory labels remain
recorded intake facts without synthesizing a label table. Model training and
structured OOD/counterfactual checks are not advertised without verified model
inputs and scientific specifications. These limitations propagate to Executive
and the final evidence report; no unsupported validation is manufactured.

Generated authority and public input/result hashes are checkpointed and persisted
as `runs/<run_id>/capabilities/result.json`. Recovery revalidates dataset, candidate,
tool/backend authority and hashes, then reuses matching context. A changed backend
or corrupt cached context fails explicitly. Building capabilities makes no tool,
provider, or source-data calls and creates no validation tasks.

Phase 16 development now includes a provider-neutral real API blind-pilot protocol with separate
text and semantic-VLM runtimes, strict preflight, immutable snapshots, a six-challenge hidden-truth
suite, global budgets, resumable sequential execution, and reporting. This is code readiness only;
no real scientific pilot is claimed until configured providers pass preflight and all frozen runs
complete. See [docs/real-blind-pilot.md](../../docs/real-blind-pilot.md).

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
       Screening
          |
          v
     Capabilities
          |
          v
      Executive
          |
          v
      Feasibility
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

Runner, Orchestrator, Candidate Screening, and Capability Builder are deterministic control components, not agents. ShortcutMiner has exactly five research agents. LangGraph builds audit capability context for promoted candidates before Executive, then checks task feasibility before fan-out to Quantitative and Qualitative. At the join, deterministic evidence calibration builds a threshold-free `EvidenceLedger` before Summarize runs.

The production control path accepts user-provided local datasets through a generic intake contract; dataset names are not selected from a benchmark whitelist. Intake performs bounded, deterministic inspection of supported local structures and CSV metadata, emits a strict `DatasetProfile`, and gives that factual profile to Identify as untrusted evidence. It does not infer or rank shortcut candidates. Unknown or unsupported structures are reported through warnings and limitations rather than being treated as a known benchmark.

For image or mixed datasets, Identify can also receive a real `VisualDiscoveryProfile`: safely decoded low-level descriptors, explicit coverage, and label-aware descriptive summaries produced by bounded CPU infrastructure. Metadata-free image folders use deterministic bounded sampling without inferring labels from directory names. Descriptor ranking is only screening salience, never statistical validation or a shortcut verdict. Optional provider-backed semantic vision adds strictly scoped visible observations while leaving candidate generation with Identify. See [Visual Discovery](../../docs/visual-discovery.md) and [Semantic Vision](../../docs/semantic-vision.md).

See [Architecture](../../docs/architecture.md) for component boundaries and persistence behavior.

Audit and audit resume enable progress by default; use `--no-progress` to disable it.
Add `--progress` to CLI `run` or `mock` commands for the same monitor. TTY uses Rich
Live; redirected output uses plain structured lines. Candidates, screening reasons,
tasks, PlanSteps, five-Agent status, and recorded telemetry are reconstructed from
existing events, artifacts, and read-only checkpoints. See
[Runtime progress monitoring](../../docs/runtime-progress.md) for usage and replay behavior.

## Five Agents

| Agent | Responsibility | Formal output |
|---|---|---|
| Identify | Discover evidence-grounded shortcut candidates | `CandidateReport` |
| Executive | Choose candidate-specific validation intent and checks | `ExecutiveAgentDraft`; Python finalizes `RoutingPlan` |
| Quantitative | Interpret controlled experiment and statistical evidence | `QuantitativeResult` |
| Qualitative | Inspect approved samples, metadata, and artifacts | `QualitativeResult` |
| Summarize | Synthesize upstream evidence and limitations | `SummaryReport` |

Detailed operating behavior is maintained in each Agent workspace and the [communication protocol](../../docs/protocol.md).

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

See [Tool Runtime](../../docs/tool-runtime.md) for the exact permission matrix and file-access boundary.

## Current Tool Permission Model

- **Identify** can inspect the approved dataset, metadata, distributions, and samples to support candidate discovery.
- **Executive** can inspect available capabilities and approved run information, but cannot run experiments.
- **Quantitative** is the only Agent permitted to request an optional `run_experiment`, and can do
  so only by selecting a ready deterministic `plan_step_id`; it cannot construct raw experiment
  parameters or execute arbitrary shell or Python code. Mandatory ready evidence is executed by
  the control layer before Quantitative interprets it.
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
  root: E:/ShortcutMiner-Experiments/my_dataset
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

The real backend supports bounded metadata/tabular statistical association, subgroup label or prediction evaluation, out-of-sample logistic-regression probes, metadata permutation, metadata-feature probe ablation, allowlisted prediction metrics, and explicit task-model training/evaluation, OOD, and paired counterfactual experiments. The real task-model reliance backend currently includes a bounded CPU reference image classifier and deterministic image interventions. The reference classifier is infrastructure for controlled validation, not a paper baseline or arbitrary deep-network runner. Every experiment uses a strict schema and reusable, input-fingerprinted artifacts. Measurements and uncertainty are evidence—not automatic shortcut decisions.

The evidence ledger is stored at `evidence/ledger.json`. It preserves direction,
execution state, coverage, uncertainty, provenance, artifacts, failures,
counterevidence, and shared input families separately. Benjamini-Hochberg FDR is
applied transparently within candidate-level related-test families while retaining
raw p-values. Neither intervals nor adjusted p-values trigger a shortcut verdict.
See [Evidence calibration](../../docs/evidence-calibration.md).

The quantitative backend remains CSV-indexed, offline and CPU-only. It never executes arbitrary Python/shell, arbitrary checkpoints, dynamic model code, or network downloads. Safe reference weights use JSON+NPZ rather than executable serialization. See [Model reliance](../../docs/model-reliance.md), [OOD and counterfactual validation](../../docs/ood-counterfactual.md), and [Segmentation providers](../../docs/segmentation.md).

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
ShortcutMiner performance claim. See [Blind shortcut discovery evaluation](../../docs/blind-evaluation.md)
and [Repeated Blind Evaluation](../../docs/repeated-evaluation.md).

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

Provider-backed semantic vision remains optional and observation-only, with no OCR guarantee. Deterministic image interventions are executed only by the typed backend; Agents receive bounded descriptors, structured observations and approved artifact references, never unrestricted local files. Provided masks support CPU background replacement, while SAM and neural visual representation benchmarks remain optional/unavailable.

## Repository Structure

```text
ShortcutMiner-github/
├── agents/                       # Five ShortcutMiner workspaces
├── configs/
│   ├── datasets/                 # Evaluation/example descriptors, not a runtime whitelist
│   ├── experiments/              # Research and local-run configurations
│   └── models/                   # Explicit model endpoint configurations
├── data/                         # Dataset-mount guidance; no raw datasets
├── docs/                         # Architecture, protocol, tools, and deployment docs
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
├── .env.example
├── .gitignore
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

## Windows Local Experiment Environment

Formal experiments use native Windows AMD64 and the conda environment `shortcutminer`,
Python 3.11.16. Commands below work in both CMD and PowerShell from the repository
root. `requirements.txt` is a standalone file containing all 71 exact package versions
and Windows wheel hashes; the separate `requirements` folder contains provenance locks.

```powershell
conda activate shortcutminer
python -m pip install --require-hashes --no-deps -r requirements.txt
python scripts/formal_environment.py check
python scripts/formal_environment.py pytest
python scripts/formal_environment.py smoke
```

The standalone requirements and supplementary Windows locks include exact versions
and SHA-256 hashes. Scientific direct versions remain
in `requirements/core.in`. See [deployment](../../docs/deployment.md) and the
[formal environment manifest](../../docs/formal-experiment-environment.md) for paths, resources,
credentials, and the clean-commit/capacity gates before E1. Acceptance uses fake providers;
it never starts E1–E5 or contacts a real provider.

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
- native Windows conda environment, mock execution, and local acceptance checks.

Not yet implemented as production research capability:

- provider-native multimodal/VLM qualitative interpretation;
- provider-native LLM function/tool-call adapters;
- the full ICDM experimental evaluation;
- large-scale dataset/model/seed experiment matrices.

## Datasets and Research Assets

Large raw datasets are not committed to GitHub. Waterbirds, Camelyon17, Colorized-MNIST, and other real datasets must be stored locally and supplied through a configured dataset root, under `E:/ShortcutMiner-Experiments`; treat source datasets as read-only. Small synthetic fixtures used by tests remain in the repository where appropriate.

Run outputs, checkpoints, raw artifacts, credentials, and local dataset directories are ignored by Git. Research and experiment assets may be maintained separately in the future; no external asset repository is declared yet.

## Development Scope

The repository is currently focused on validating the local research architecture before beginning full experiments. Contributions should preserve the separation between Agent reasoning, structured contracts, deterministic execution, tool permissions, and durable evidence. Large-scale research claims should only be added after their corresponding real backends and evaluations exist.

### Final audit reports (Phase17 Step5)

Completed `shortcutminer audit DATASET_PATH` runs automatically export `audit_report.json` and `audit_report.md`, preserving the original `final_result.json`. The Python report builder aggregates persisted scientific results and runtime parameters without additional Agent or model calls. See [Audit report schema and reproducibility](../../docs/audit-report.md).

### Evidence promotion (Phase17 Step6)

Audit runs execute bounded Stage-1 checks before an evidence-only promotion controller can authorize expensive Stage-2 checks. The same five Agents and existing execution contracts are reused. The default policy has no scientific thresholds and defers expensive validation; a formal experiment policy must be determined and frozen separately. See [classification, policy, recovery and offline acceptance](../../docs/evidence-promotion.md).
