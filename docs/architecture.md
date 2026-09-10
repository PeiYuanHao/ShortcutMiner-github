# ShortcutMiner Architecture

ShortcutMiner is an experimental five-Agent research framework with a deterministic Python control plane. This document describes the currently implemented local architecture; it does not claim that the real scientific experiment backends are complete.

## Execution flow

```text
Dataset / Audit Request
          |
          v
        Runner
          |
          v
    Dataset Intake
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
          |
          v
 Storage / Events / Checkpoint
```

Runner and Orchestrator are not Agents. Runner validates a run request and exposes run/resume entry points. Orchestrator owns run lifecycle and invokes the compiled LangGraph. The graph contains five Agent-backed nodes, an Executive fan-out, and a Quantitative/Qualitative fan-in before Summarize. `EvidenceLedgerBuilder` runs deterministically at that fan-in inside the control layer; it is not an Agent and does not execute or interpret new experiments.

Dataset Intake and Visual Discovery are deterministic infrastructure, not additional Agents. `DatasetProfiler` performs a root-constrained, bounded scan with no network, LLM, arbitrary code execution, or dataset-specific adapter. Its strict `DatasetProfile` records structure, inferred modality, bounded metadata facts, label validity/distribution, explicit split data, safe representative file references, warnings, limitations, and the inspection budget. For image or mixed data, `VisualDiscoveryProfiler` can then decode bounded public samples and compute the same canonical low-level descriptors used by Qualitative.

```text
DatasetProfiler -> DatasetProfile -> VisualDiscoveryProfiler -> VisualDiscoveryProfile -> Identify
```

Both profiler outputs contain facts only. Visual group differences and ranking are descriptive context compression, not a hypothesis test, candidate, or shortcut judgment. A provider-neutral semantic-inspection capability can be injected as either a deterministic mock or a bounded provider-backed multimodal runtime. It emits observations only and does not alter the five-Agent topology.

Identify receives the dataset profile and, when applicable, the visual profile inside its discovery context. Every dataset-derived name, path, column, example value, filename, and visual observation is explicitly untrusted evidence, so dataset content cannot become an instruction. Identify alone reasons from those facts and produces the `CandidateReport`.

## Five Agent boundaries

| Agent | Input role | Validated output |
|---|---|---|
| Identify | Dataset/task description and bounded discovery context | `CandidateReport` |
| Executive | Candidate report and declared runtime capabilities | `RoutingPlan` |
| Quantitative | Quantitative tasks selected by the routing plan | `QuantitativeResult` batch |
| Qualitative | Qualitative tasks and bounded evidence context | `QualitativeResult` batch |
| Summarize | Routing plan and both result batches | `SummaryReport` |

Every executor loads its own workspace, invokes an injected `AgentClient`, validates the returned protocol object, and persists stage artifacts. The deterministic control layer, rather than Agent prose, decides graph transitions and artifact locations.

## State and contracts

The shared workflow state contains serializable run data and validated Agent outputs. Runtime objects such as model clients and Tool Runtime instances are closure dependencies and do not enter state or checkpoints.

Protocol definitions live under `shortcutminer/protocol/`. The main cross-stage contracts are `CandidateReport`, `RoutingPlan`, `QuantitativeResult`, `QualitativeResult`, and `SummaryReport`. They carry explicit run, candidate, task, source, target, and message correlations where applicable.

`DatasetProfile` is the pre-Agent intake contract. It is serializable and stored in workflow/checkpoint state as plain JSON; filesystem objects, clients, handles, and runtime instances never enter the state.

## AgentClient and workspaces

`shortcutminer/agent_client.py` defines the minimal provider-independent AgentClient boundary. `shortcutminer/runtime/` contains deterministic demo/mock implementations and an explicitly configured OpenAI-compatible JSON client. The real client records secret-free per-call artifacts, usage when supplied, monotonic latency, classified failures, bounded retries, budgets, and a frozen runtime snapshot. Automatic tests use only offline clients. See [model-runtime.md](model-runtime.md).

Each Agent workspace lives under `agents/<agent>/workspace/`. The workspace loader reads `AGENTS.md`, `IDENTITY.md`, `SOUL.md`, `USER.md`, `TOOLS.md`, `MEMORY.md`, and `HEARTBEAT.md` in that order. These files are ShortcutMiner configuration and do not hold hidden runtime state.

## Tool Runtime

The optional Tool Runtime remains separate from AgentClient. When an Agent returns a typed `ToolCall`, its executor routes the request through the ShortcutMiner Tool Registry. The Registry validates the call identity, Agent permission, arguments, output, budget, and timeout. The resulting `ToolResult` is persisted and returned to the AgentClient for the next bounded iteration.

The runtime exposes dataset and metadata inspection, deterministic sample selection, controlled artifact/trace reads, sample metadata inspection, and a typed ExperimentExecutor entry point. It exposes no arbitrary shell, dynamic import, or unrestricted filesystem reader. The exact permission matrix is documented in [tool-runtime.md](tool-runtime.md).

Qualitative additionally receives `inspect_sample`, a Qualitative-only tool for an
already approved dataset-relative reference. Its strict allowlist exposes the same
bounded descriptor families used by evidence preparation, never an arbitrary path
or user-supplied computation.

## Storage and resume

All durable run data is rooted under `runs_root/run_id`:

```text
manifest.json
events.jsonl
final_result.json
checkpoints.sqlite
dataset/profile.json
identify/
executive/
quantitative/
qualitative/
evidence/ledger.json
summarize/
tools/
experiments/
model_calls/
evaluation/repeated/
```

Stage inputs, raw outputs, validated outputs, and ToolResults are stored as JSON artifacts. Events record run, stage, and tool lifecycle changes. The SQLite LangGraph checkpointer uses the run ID as its thread ID and supports local single-machine resume without serializing clients or tool implementations.

The dataset profile is written once before Identify invocation. Visual discovery is separately stored at `identify/visual_discovery.json`; its fingerprint covers the public dataset identity, selected public-file content, task, seed, descriptor version, budgets/options, and semantic capability. A matching visual artifact is reused without decoding again. An incompatible or malformed stored profile fails explicitly rather than silently mixing evidence from different settings. The Evidence Ledger uses a deterministic fingerprint over the candidate report, routing plan, and both result batches. A matching ledger is reused; any upstream identity or content change rebuilds it.

## Evidence calibration boundary

The versioned Evidence Ledger contains one entry for every Identify candidate,
including candidates with no task or only blocked/failed work. Records keep
research design, execution state, evidence direction, coverage, uncertainty,
artifact provenance, and shared input sources as separate fields. Logical
consistency aggregation can describe support, contradiction, mixed,
inconclusive, or no usable evidence, but it never derives direction from a raw
metric threshold. The Summarize validator rejects impossible overclaims such as
support with no supporting record or a substantive verdict when all work failed.

## Experiment boundary

`run_experiment` delegates to the `ExperimentExecutor` Protocol. `MockExperimentExecutor` remains the isolated test backend. `RealExperimentExecutor` uses a static registry of CPU implementations for association, subgroup analysis, lightweight out-of-sample probes, metadata permutation, metadata-feature ablation, allowlisted prediction metrics, task-model training/evaluation, OOD evaluation, and paired counterfactual intervention.

Agent and backend responsibilities remain separate: Executive defines a scoped AnalysisTask; Quantitative requests an approved experiment type with exact resolved fields; deterministic code validates parameters and computes evidence; Quantitative interprets the returned measurements; Summarize combines evidence. The backend never emits a final shortcut judgment or evidence level.

Real experiment artifacts are stored under `experiments/<task_id>/` and keyed by experiment type plus a deterministic parameter-and-input fingerprint. Inputs remain dataset-relative or explicitly approved run-relative references. The bounded, network-free, CPU-only task-model path provides an allowlisted reference image classifier with safe JSON+NPZ persistence. OOD and counterfactual experiments reuse the exact trained model identity and apply only typed deterministic image transformations. They do not retrain on OOD data or infer a causal verdict from metric magnitude.

## Qualitative evidence boundary

The qualitative control path supports metadata-linked group-balanced selection and
metadata-free random selection. Both are deterministic, duplicate-free, and bounded.
Selected images and UTF-8 text are resolved beneath the approved dataset root;
tabular-only tasks retain selected row evidence without inventing visual content.

`qualitative_backend.py` uses Pillow and NumPy locally on CPU. It emits only
descriptor facts: dimensions and channels, brightness/contrast, RGB statistics,
histogram entropy, mathematically defined edge density, and fixed fractional spatial
regions. Group summaries and pairwise differences are descriptive. They never carry
an evidence status, shortcut score, model-reliance claim, or causal conclusion.

Each task may persist:

```text
qualitative/<task_id>/inspection.json
qualitative/<task_id>/contact_sheet.png
```

The structured artifact records request identity, sampling and coverage (including
failures), descriptor definitions, per-sample evidence, group comparisons, warnings,
limitations, and safe run-relative references. A retry validates schema, run,
candidate, task, sampling options, request limitations, and referenced files before
reuse; mismatch fails explicitly. The contact sheet has a fixed bounded layout and
only sample/group/label annotations.

Dataset content remains quoted untrusted input. The Agent receives descriptors and
artifact references through its normal text payload, not arbitrary filesystem access
or provider-native image input. Semantic recognition, OCR, VLM inference, and neural
representations remain outside this phase. Image interventions are available only
through the separate typed experiment backend; Qualitative itself remains
observation-only.

## Runtime scope

The validated operating target is local Docker Desktop or Docker Engine with a persistent run volume and read-only dataset/configuration mounts. No web service, distributed queue, remote control plane, or GPU runtime is part of the current architecture.
