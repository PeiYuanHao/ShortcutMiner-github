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
          |
          v
 Storage / Events / Checkpoint
```

Runner and Orchestrator are not Agents. Runner validates a run request and exposes run/resume entry points. Orchestrator owns run lifecycle and invokes the compiled LangGraph. The graph contains five Agent-backed nodes, an Executive fan-out, and a Quantitative/Qualitative fan-in before Summarize.

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

## AgentClient and workspaces

`shortcutminer/agent_client.py` defines the minimal provider-independent AgentClient boundary. `shortcutminer/runtime/` contains the deterministic demo/mock implementations and an explicitly configured OpenAI-compatible JSON client. Automatic tests use only offline clients.

Each Agent workspace lives under `agents/<agent>/workspace/`. The workspace loader reads `AGENTS.md`, `IDENTITY.md`, `SOUL.md`, `USER.md`, `TOOLS.md`, `MEMORY.md`, and `HEARTBEAT.md` in that order. These files are ShortcutMiner configuration and do not hold hidden runtime state.

## Tool Runtime

The optional Tool Runtime remains separate from AgentClient. When an Agent returns a typed `ToolCall`, its executor routes the request through the ShortcutMiner Tool Registry. The Registry validates the call identity, Agent permission, arguments, output, budget, and timeout. The resulting `ToolResult` is persisted and returned to the AgentClient for the next bounded iteration.

The runtime exposes dataset and metadata inspection, deterministic sample selection, controlled artifact/trace reads, sample metadata inspection, and a typed ExperimentExecutor entry point. It exposes no arbitrary shell, dynamic import, or unrestricted filesystem reader. The exact permission matrix is documented in [tool-runtime.md](tool-runtime.md).

## Storage and resume

All durable run data is rooted under `runs_root/run_id`:

```text
manifest.json
events.jsonl
final_result.json
checkpoints.sqlite
identify/
executive/
quantitative/
qualitative/
summarize/
tools/
```

Stage inputs, raw outputs, validated outputs, and ToolResults are stored as JSON artifacts. Events record run, stage, and tool lifecycle changes. The SQLite LangGraph checkpointer uses the run ID as its thread ID and supports local single-machine resume without serializing clients or tool implementations.

## Experiment boundary

`run_experiment` delegates to the `ExperimentExecutor` Protocol. Phase 7 provides a deterministic `MockExperimentExecutor` for contract and workflow testing. Real statistical, model-evaluation, intervention, and ablation backends remain future research work.

## Runtime scope

The validated operating target is local Docker Desktop or Docker Engine with a persistent run volume and read-only dataset/configuration mounts. No web service, distributed queue, remote control plane, or GPU runtime is part of the current architecture.
