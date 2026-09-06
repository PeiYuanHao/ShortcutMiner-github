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
       Summarize
          |
          v
     SummaryReport
```

Runner and Orchestrator are deterministic control components, not agents. ShortcutMiner has exactly five research agents. LangGraph provides the typed execution graph: Executive fans out to Quantitative and Qualitative, and Summarize runs after both branches join.

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
CandidateReport
RoutingPlan
QuantitativeResult
QualitativeResult
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
- **Qualitative** can inspect approved samples, metadata, artifacts, and structured experiment output without mutating upstream data.
- **Summarize** can read approved artifacts and run traces for report synthesis, but cannot run experiments or mutate datasets.

All file access is root-constrained, relative-path validated, allowlisted where appropriate, and bounded by runtime budgets.

## Experiment Runtime

`ExperimentExecutor` is the typed boundary between `run_experiment` and a scientific execution backend. The validated test backend is currently `MockExperimentExecutor`, which is deterministic, offline, and schema-identical to a future real backend.

Real statistical checks, probes, model evaluations, interventions, ablations, and subgroup evaluations are not yet implemented. Placeholder modules under `tools/quantitative/`, `shortcutminer/evaluation/`, and selected scripts are retained for future research development and must not be interpreted as completed backends.

## Repository Structure

```text
ShortcutMiner-github/
├── agents/                       # Five ShortcutMiner workspaces
├── configs/
│   ├── datasets/                 # Dataset descriptors
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

Latest repository-refresh validation:

```text
Host:   346 passed, 3 skipped
Docker: 349 passed
```

Automatic tests use deterministic mock clients and tools. They do not call real LLM APIs or run expensive training.

## Project Status

Implemented and tested:

- five-Agent LangGraph orchestration;
- structured Agent and run contracts;
- deterministic workspace loading;
- AgentClient abstraction and offline mock client;
- Tool Runtime, Registry, budgets, and per-Agent permissions;
- run, artifact, event, and final-report storage;
- SQLite checkpoint and resume behavior;
- local Docker build, mock execution, and container tests.

Not yet implemented as production research capability:

- real quantitative experiment backends;
- real visual or multimodal qualitative integration;
- provider-native LLM function/tool-call adapters;
- the full ICDM experimental evaluation;
- large-scale dataset/model/seed experiment matrices.

## Datasets and Research Assets

Large raw datasets are not committed to GitHub. Waterbirds, Camelyon17, Colorized-MNIST, and other real datasets must be stored locally and supplied through a configured dataset root, normally as a read-only Docker volume. Small synthetic fixtures used by tests remain in the repository where appropriate.

Run outputs, checkpoints, raw artifacts, credentials, and local dataset directories are ignored by Git. Research and experiment assets may be maintained separately in the future; no external asset repository is declared yet.

## Development Scope

The repository is currently focused on validating the local research architecture before beginning full experiments. Contributions should preserve the separation between Agent reasoning, structured contracts, deterministic execution, tool permissions, and durable evidence. Large-scale research claims should only be added after their corresponding real backends and evaluations exist.
