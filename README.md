# ShortcutMiner

ShortcutMiner is a multi-agent framework for automatically discovering, routing, validating, and reporting potential dataset shortcuts.

The framework contains five specialized agents:

* **Identify** — discovers potential shortcut candidates from dataset information.
* **Executive** — evaluates candidates and determines the appropriate validation route.
* **Quantitative** — performs statistical and model-based quantitative validation.
* **Qualitative** — collects sample-level, visual, and metadata-based qualitative evidence.
* **Summarize** — integrates evidence and generates the final shortcut audit report.

ShortcutMiner uses **OpenClaw** as the agent runtime. A deterministic Python control layer is responsible for experiment execution, agent scheduling, configuration management, artifact storage, logging, and reproducibility.

> **Project Status**
>
> ShortcutMiner is currently undergoing a repository and runtime redesign.
> The current goal is to establish a stable and reproducible execution framework before migrating and refining the full ICDM experimental pipeline.

---

## Architecture

The logical agent workflow is:

```text
Dataset / Audit Task
        │
        ▼
     Identify
        │
        ▼
    Executive
     /      \
    ▼        ▼
Quantitative  Qualitative
     \        /
      \      /
       ▼    ▼
      Summarize
          │
          ▼
     Audit Report
```

ShortcutMiner contains exactly **five research agents**.

`Runner` and `Orchestrator` are deterministic Python components used to execute this workflow reliably. They are **not additional agents**.

The system therefore separates:

```text
Agent reasoning
      │
      ▼
Structured decisions
      │
      ▼
Deterministic execution
      │
      ▼
Tools / Artifacts / Results
```

---

## Repository Structure

```text
ShortcutMiner/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .env.example
│
├── agents/
│   ├── identify/
│   ├── executive/
│   ├── quantitative/
│   ├── qualitative/
│   └── summarize/
│
├── shortcutminer/
│   ├── runner.py
│   ├── orchestrator.py
│   ├── openclaw_client.py
│   ├── context.py
│   ├── protocol/
│   ├── experiment/
│   ├── storage/
│   └── evaluation/
│
├── configs/
│   ├── base.yaml
│   ├── datasets/
│   ├── models/
│   └── experiments/
│
├── tools/
│   ├── common/
│   ├── quantitative/
│   └── qualitative/
│
├── infrastructure/
│   ├── docker/
│   └── openclaw/
│
├── scripts/
├── tests/
├── data/
├── runs/
├── results/
└── docs/
```

### Directory Summary

| Directory         | Purpose                                                                   |
| ----------------- | ------------------------------------------------------------------------- |
| `agents/`         | OpenClaw workspaces for the five agents                                   |
| `shortcutminer/`  | Core deterministic execution framework                                    |
| `configs/`        | Dataset, model, and experiment configurations                             |
| `tools/`          | Deterministic tools callable by agents                                    |
| `infrastructure/` | Docker and OpenClaw deployment configuration                              |
| `scripts/`        | User-facing experiment entry points                                       |
| `tests/`          | Unit, integration, and mock pipeline tests                                |
| `data/`           | Dataset preparation instructions                                          |
| `runs/`           | Raw experiment records                                                    |
| `results/`        | Aggregated research results                                               |
| `docs/`           | Detailed architecture, protocol, experiment, and deployment documentation |

---

# Agents

All five agents are stored under:

```text
agents/
```

Each agent has an independent OpenClaw workspace.

Example:

```text
agents/identify/workspace/
├── AGENTS.md
├── IDENTITY.md
├── SOUL.md
├── USER.md
├── TOOLS.md
├── MEMORY.md
└── HEARTBEAT.md
```

The same structure is used for:

```text
identify
executive
quantitative
qualitative
summarize
```

### Workspace Files

`AGENTS.md`

Defines the agent's main operational behavior, including:

* responsibilities
* expected inputs
* workflow
* output format
* communication rules
* failure behavior
* prohibited actions

`IDENTITY.md`

Defines the permanent role and identity of the agent.

`SOUL.md`

Defines stable behavioral principles.

`TOOLS.md`

Defines which tools the agent can use and how those tools should be used.

`MEMORY.md`

Defines memory behavior.

Formal experiments should avoid uncontrolled cross-run memory contamination.

`USER.md`

Defines how task and upstream user context should be interpreted.

`HEARTBEAT.md`

Defines optional runtime health-check behavior.

---

# Core Framework

The deterministic execution layer is implemented in:

```text
shortcutminer/
```

## `runner.py`

Runs a single ShortcutMiner experiment.

Its responsibilities include:

```text
Load configuration
        │
        ▼
Create run
        │
        ▼
Initialize storage
        │
        ▼
Execute pipeline
        │
        ▼
Persist artifacts
        │
        ▼
Finalize run status
```

---

## `orchestrator.py`

Executes the five-agent workflow.

Conceptually:

```text
Identify
   │
   ▼
Executive
 /      \
▼        ▼
Quant.  Qual.
 \        /
  \      /
   ▼    ▼
  Summarize
```

The **Executive Agent makes routing decisions**.

The Orchestrator only ensures that those decisions are executed reliably.

The current Executive `1.0` implementation stops at a validated routing plan:
it preserves every Identify candidate, selects `quantitative`, `qualitative`,
`both`, or `unresolved`, and emits complete downstream task messages.

The Qualitative `1.0` stage can consume one planned qualitative task, prepare a
bounded metadata-linked sample manifest, invoke the Qualitative Agent, validate
its sample-level evidence, and persist a result for Summarize. Physical
multi-task dispatch and result tracking remain part of the deferred Orchestrator
skeleton.

---

## `openclaw_client.py`

Provides the interface between ShortcutMiner and the OpenClaw runtime.

OpenClaw-specific behavior should be isolated in this module whenever possible.

Typical responsibilities include:

* invoking agents
* creating or resolving sessions
* sending structured requests
* retrieving outputs
* checking execution status
* handling timeouts
* handling runtime errors

This separation prevents OpenClaw runtime changes from affecting the entire ShortcutMiner codebase.

---

## `context.py`

Maintains run-level context.

Typical fields include:

```text
run_id
dataset
model
seed
repeat
artifact paths
agent session keys
experiment configuration
```

Each experimental run receives an independent context.

---

# Communication Protocol

Structured communication is defined under:

```text
shortcutminer/protocol/
```

```text
protocol/
├── identify.py
├── executive.py
├── messages.py
├── schemas.py
├── artifacts.py
└── states.py
```

ShortcutMiner should avoid using unrestricted natural-language messages as the only control mechanism between agents.

Agent communication should contain explicit fields such as:

```text
protocol_version
run_id
message_id
source
target
message_type
stage-specific body fields
artifact references
```

Example:

```json
{
  "protocol_version": "1.0",
  "run_id": "waterbirds__model-a__seed42__r01",
  "message_id": "msg_00001",
  "source": "identify",
  "target": "executive",
  "message_type": "candidate_report",
  "in_reply_to": "msg_identify_request_00001",
  "artifact_references": [],
  "dataset_summary": {
    "dataset": "waterbirds",
    "samples_inspected": 0,
    "metadata_used": true
  },
  "candidates": [],
  "limitations": []
}
```

Expected structured entities include:

```text
Candidate
CandidateReport
ExecutiveRequest
RoutingDecision
RoutingPlan
AnalysisTask
QuantitativeResult
QualitativeResult
SummaryReport
ArtifactReference
```

Detailed protocol definitions belong in:

```text
docs/protocol.md
```

---

# Configuration

Experiment configuration is stored under:

```text
configs/
```

```text
configs/
├── base.yaml
├── datasets/
├── models/
└── experiments/
```

The main design goal is:

> Changing datasets, seeds, models, or experiment repetitions should not require rewriting the five agent workspaces.

---

## Base Configuration

Shared default settings are stored in:

```text
configs/base.yaml
```

Examples include:

```text
timeout
retry policy
artifact root
logging settings
default execution budget
```

---

## Dataset Configuration

Each dataset has its own configuration file.

```text
configs/datasets/
├── waterbirds.yaml
├── colored_mnist.yaml
└── synthetic_patch.yaml
```

Example:

```yaml
name: waterbirds
root: /datasets/waterbirds

metadata_file: metadata.csv
label_column: y
```

Raw datasets are **not stored in this Git repository**.

---

## Model Configuration

Different LLM backends are configured under:

```text
configs/models/
```

Example:

```yaml
provider: openai
model: example-model
temperature: 0
max_tokens: 4096
```

API keys must not be stored in these YAML files.

---

## Experiment Configuration

Experiment matrices are defined under:

```text
configs/experiments/
```

The repository initially contains:

```text
smoke_test.yaml
icdm_main.yaml
```

Example experiment matrix:

```yaml
datasets:
  - waterbirds
  - colored_mnist
  - synthetic_patch

models:
  - model_a
  - model_b
  - model_c

seeds:
  - 42
  - 3407
  - 2026

repeats: 3
```

The framework expands the configuration into independent runs automatically.

---

# Tools

Agent-callable deterministic tools are stored under:

```text
tools/
```

```text
tools/
├── common/
├── quantitative/
└── qualitative/
```

The general design principle is:

```text
Agent decides what should be analyzed
                │
                ▼
Python tool performs the computation
                │
                ▼
Agent interprets the returned evidence
```

LLMs should not manually perform calculations when deterministic implementations are available.

Tool ownership is stage-specific. In the current Identify `1.0` integration,
the deterministic control layer runs the bounded dataset and metadata
preparation plan before invocation and sends the resulting immutable
`discovery_context` to the Agent. Identify interprets that evidence but does
not directly rerun the Python tools. Moving inspection choices into the Agent
requires an explicit runtime-tool binding and protocol revision so budgets and
run isolation remain enforceable.

---

## Common Tools

```text
tools/common/
```

Examples:

* file reading
* JSON / CSV handling
* dataset metadata inspection
* dataset summary extraction

---

## Quantitative Tools

```text
tools/quantitative/
```

Planned functionality includes:

* association analysis
* Cramér's V
* mutual information
* correlation analysis
* significance tests
* confidence intervals
* accuracy
* AUC
* ID/OOD evaluation
* intervention-based evaluation

---

## Qualitative Tools

```text
tools/qualitative/
```

Implemented functionality includes:

* bounded, deterministic random or group-balanced sampling
* metadata-field and split validation
* dataset-contained sample-reference resolution
* representative evidence manifests with group coverage
* explicit unavailable/unsupported-sample limitations

The control layer does not invent visual observations. It selects and verifies
the evidence boundary; the Qualitative Agent inspects the referenced samples
and returns traceable observations, interpretations, counterevidence, failures,
and limitations under the strict `QualitativeResult` contract.

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd ShortcutMiner
```

---

## 2. Create a Python Environment

Recommended:

```bash
conda create -n shortcutminer python=3.11
conda activate shortcutminer
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Install ShortcutMiner in editable mode:

```bash
pip install -e .
```

---

## 4. Configure Environment Variables

Copy the environment template:

```bash
cp .env.example .env
```

Configure the required runtime credentials.

Example:

```env
OPENCLAW_GATEWAY_URL=
OPENCLAW_TOKEN=

OPENAI_API_KEY=
```

Never commit the real `.env` file.

---

# Dataset Preparation

Raw datasets are not committed to GitHub.

Dataset preparation instructions are stored in:

```text
data/README.md
```

The recommended server-side layout is:

```text
/srv/shortcutminer/
├── datasets/
├── runtime/
└── runs/
```

Example:

```text
/srv/shortcutminer/datasets/
├── waterbirds/
├── colored_mnist/
└── synthetic_patch/
```

Dataset paths are defined through:

```text
configs/datasets/
```

Whenever possible, source datasets should be mounted into containers as **read-only** directories.

---

# OpenClaw

ShortcutMiner uses OpenClaw as the runtime environment for the five agents.

The complete OpenClaw source tree is **not copied into this repository**.

ShortcutMiner-specific OpenClaw configuration is stored under:

```text
infrastructure/openclaw/
```

```text
infrastructure/openclaw/
├── openclaw.json.template
├── agents.example.json
└── version.lock
```

`version.lock` is used to record the OpenClaw version or container image used for formal experiments.

Formal experiments should use a pinned OpenClaw version rather than an uncontrolled `latest` version.

---

# Docker Deployment

Docker-related files are stored under:

```text
infrastructure/docker/
```

```text
infrastructure/docker/
├── Dockerfile
└── docker-compose.yml
```

The intended server architecture is:

```text
Server
│
├── ShortcutMiner container
│
├── OpenClaw runtime
│
├── persistent datasets
│
├── persistent run storage
└── runtime/cache storage
```

Persistent experiment results must not depend on the temporary container filesystem.

Detailed deployment instructions belong in:

```text
docs/deployment.md
```

---

# Running ShortcutMiner

## Health Check

Before running experiments:

```bash
python scripts/health_check.py
```

The health check should verify:

* OpenClaw availability
* model API availability
* dataset paths
* runtime directories
* output write permissions
* required configuration

---

## Configuration Validation

Validate configuration before starting a formal run:

```bash
python scripts/validate_config.py
```

---

## Smoke Test

A smoke test should always be executed before large experiment matrices.

```bash
python scripts/run_matrix.py \
    --config configs/experiments/smoke_test.yaml
```

The smoke test should contain only:

```text
1 dataset
× 1 model
× 1 seed
× 1 repeat
```

Its purpose is to verify that the complete pipeline is operational.

---

## Single Experiment

Example:

```bash
python scripts/run_experiment.py \
    --dataset waterbirds \
    --model model_a \
    --seed 42
```

---

## Experiment Matrix

Run the complete configured experiment matrix with:

```bash
python scripts/run_matrix.py \
    --config configs/experiments/icdm_main.yaml
```

For example:

```text
3 datasets
×
3 models
×
3 seeds
×
3 repeats

= 81 independent runs
```

Each run must remain isolated from the others.

---

# Experiment Runs

Every experiment receives a unique `run_id`.

Example:

```text
waterbirds__model-a__seed42__r01
```

Raw run artifacts are stored under:

```text
runs/
```

Example:

```text
runs/
└── waterbirds__model-a__seed42__r01/
    ├── manifest.json
    ├── resolved_config.yaml
    ├── status.json
    ├── events.jsonl
    │
    ├── identify/
    │   ├── input.json
    │   ├── raw_output.json
    │   └── output.json
    │
    ├── executive/
    │   ├── input.json
    │   ├── raw_output.json
    │   └── routing.json
    │
    ├── quantitative/
    │   ├── input.json
    │   ├── result.json
    │   └── artifacts/
    │
    ├── qualitative/
    │   ├── input.json
    │   ├── raw_output.json
    │   ├── result.json
    │   └── artifacts/
    │
    └── summarize/
        ├── input.json
        └── report.json
```

The exact internal artifact structure may evolve as the protocol is finalized.

---

# Run Manifest

Each experiment should generate:

```text
manifest.json
```

Example:

```json
{
  "run_id": "waterbirds__model-a__seed42__r01",
  "dataset": "waterbirds",
  "model": "model-a",
  "seed": 42,
  "repeat": 1,

  "shortcutminer_commit": "git-commit",
  "openclaw_version": "pinned-version",

  "started_at": "timestamp",
  "finished_at": "timestamp",

  "status": "completed"
}
```

The manifest makes each experimental result traceable to its exact execution environment.

---

# Event Logging

Important runtime events are stored in:

```text
events.jsonl
```

Possible events include:

```text
RUN_CREATED

IDENTIFY_STARTED
IDENTIFY_COMPLETED

EXECUTIVE_STARTED
EXECUTIVE_COMPLETED

QUANTITATIVE_STARTED
QUANTITATIVE_COMPLETED

QUALITATIVE_STARTED
QUALITATIVE_COMPLETED

SUMMARIZE_STARTED
SUMMARIZE_COMPLETED

AGENT_FAILED
RETRY_REQUESTED

RUN_COMPLETED
RUN_FAILED
```

This log is used for debugging, failure analysis, and experiment traceability.

---

# Results

ShortcutMiner distinguishes between **raw experiment runs** and **aggregated research results**.

## Raw Runs

Stored under:

```text
runs/
```

These files may contain:

* complete agent inputs and outputs
* intermediate artifacts
* execution logs
* statistical files
* qualitative evidence
* runtime metadata

Raw runs may become large and are normally excluded from Git.

---

## Aggregated Results

Generate aggregated results with:

```bash
python scripts/aggregate_results.py
```

Aggregated outputs are written to:

```text
results/
├── tables/
├── figures/
└── summaries/
```

Examples:

```text
results/tables/main_results.csv
results/tables/llm_comparison.csv
results/tables/ablation.csv

results/figures/

results/summaries/main_summary.json
```

Small aggregated research outputs may be committed to Git.

The intended data flow is:

```text
runs/
  │
  ▼
Evaluation
  │
  ▼
Aggregation
  │
  ▼
results/
```

---

# Evaluation

Cross-run evaluation is implemented under:

```text
shortcutminer/evaluation/
```

Framework-level evaluation may include:

* detection performance
* routing performance
* execution success rate
* completion rate
* failure rate
* runtime
* token usage
* cost
* seed variance
* model variance
* dataset variance

Dataset-specific metrics may additionally be produced by quantitative tools.

---

# Testing

Run the test suite with:

```bash
pytest
```

Important tests include:

```text
tests/
├── test_protocol.py
├── test_config.py
├── test_runner.py
├── test_storage.py
└── test_pipeline_mock.py
```

## Mock Pipeline

`test_pipeline_mock.py` is especially important during the current redesign.

It should test:

```text
Mock Identify
      │
      ▼
Mock Executive
   /        \
  ▼          ▼
Mock Quant. Mock Qual.
   \          /
    \        /
     ▼      ▼
    Mock Summarize
```

without requiring:

* OpenClaw
* external LLM APIs
* real model inference

The deterministic pipeline should work correctly before real agents are connected.

---

# Development Workflow

The current recommended development order is:

```text
Repository structure
        │
        ▼
Communication protocol
        │
        ▼
Runner / Orchestrator
        │
        ▼
Storage and manifests
        │
        ▼
Mock pipeline
        │
        ▼
OpenClaw integration
        │
        ▼
Five agent workspaces
        │
        ▼
Smoke test
        │
        ▼
Full experiment matrix
```

The previous implementation may be used as a reference, but new code should follow the architecture of the current repository instead of directly reproducing the previous local directory structure.

---

# Collaboration Guidelines

When contributing to ShortcutMiner, keep changes within the appropriate layer whenever possible.

### Modify agent behavior

```text
agents/<agent>/workspace/
```

### Modify structured communication

```text
shortcutminer/protocol/
```

### Modify pipeline execution

```text
shortcutminer/runner.py
shortcutminer/orchestrator.py
```

### Modify OpenClaw integration

```text
shortcutminer/openclaw_client.py
infrastructure/openclaw/
```

### Add or modify datasets

```text
configs/datasets/
```

### Add or modify LLM configurations

```text
configs/models/
```

### Modify experiment matrices

```text
configs/experiments/
```

### Add deterministic analysis functions

```text
tools/
```

### Modify cross-run metrics

```text
shortcutminer/evaluation/
```

Avoid placing unrelated logic inside agent prompts when it can be implemented as reusable Python code.

---

# Git Policy

The repository should contain the information required to reproduce ShortcutMiner experiments.

The following files should be committed:

```text
source code
agent workspaces
configuration files
tests
documentation
Docker configuration
OpenClaw configuration templates
small aggregated results
```

The following files should normally **not** be committed:

```text
.env
API keys
authentication tokens
raw datasets
OpenClaw runtime sessions
cache files
temporary files
large raw experiment runs
private credentials
```

---

# Documentation

Detailed design documentation is maintained separately:

```text
docs/
├── architecture.md
├── protocol.md
├── experiment.md
└── deployment.md
```

### `architecture.md`

Explains:

* system components
* Runner and Orchestrator
* OpenClaw integration
* five-agent execution flow
* storage architecture

### `protocol.md`

Explains:

* message schema
* agent input/output contracts
* `run_id`
* `message_id`
* artifact references
* pipeline states

### `experiment.md`

Explains:

* datasets
* random seeds
* LLM configurations
* repeated trials
* evaluation metrics
* experiment matrix

### `deployment.md`

Explains:

* server setup
* Docker
* OpenClaw
* persistent volumes
* runtime storage
* experiment startup and shutdown

---

# Current Development Priorities

The current development priorities are:

1. Finalize repository structure.
2. Finalize the structured agent communication protocol.
3. Implement the deterministic Runner and Orchestrator.
4. Implement persistent run storage and manifests.
5. Pass the complete mock pipeline.
6. Integrate OpenClaw.
7. Rebuild the five agent workspaces against the new protocol.
8. Pass a single smoke-test experiment.
9. Expand to multi-dataset, multi-seed, and multi-model experiments.

Large-scale experiments should not begin until the smoke-test pipeline is stable.

---

# Summary

The responsibilities of the main directories can be summarized as:

```text
agents/
    Defines how the five agents behave.

shortcutminer/
    Executes the five-agent system reliably.

configs/
    Defines what experiment should be executed.

tools/
    Provides deterministic operations for the agents.

infrastructure/
    Defines how ShortcutMiner and OpenClaw are deployed.

scripts/
    Provides user-facing execution commands.

tests/
    Verifies that the framework works correctly.

data/
    Explains how datasets should be prepared.

runs/
    Records everything that happened in each experiment.

results/
    Stores aggregated research outputs.

docs/
    Explains the detailed system design.
```

In short:

> **Git stores how to run the system.
> Datasets define what is audited.
> Runs record what happened.
> Results summarize what was learned.**
