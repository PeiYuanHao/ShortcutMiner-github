"""Human-readable report generation from immutable pilot results."""

from __future__ import annotations

from pathlib import Path

from shortcutminer.evaluation.pilot.protocol import PilotResult, PilotSnapshot


def render_pilot_report(result: PilotResult, snapshot: PilotSnapshot) -> str:
    aggregate = result.aggregate
    workspaces = "\n".join(
        f"- {item.agent_name}: `{item.sha256}`" for item in snapshot.workspace_hashes
    )
    failures = "\n".join(
        f"- {name}: {count}" for name, count in sorted(aggregate.failure_taxonomy.items())
    ) or "- None recorded"
    return f"""# Phase 16 Real API Blind Pilot

> **DEVELOPMENT PILOT — NOT FINAL PAPER RESULTS**

## Pilot identity

- Pilot: `{result.pilot_id}`
- Status: `{result.status}`
- Phase conclusion: `{result.phase_conclusion}`
- Snapshot: `{result.snapshot_id}`
- Provider combination: `{result.provider_combination_id}`
- Frozen git commit: `{snapshot.git_commit}`

## Provider combination

- Text: `{snapshot.text_runtime.provider}` / `{snapshot.text_runtime.model}`
- Semantic VLM: `{snapshot.semantic_runtime.provider}` / `{snapshot.semantic_runtime.model}`
- Text temperature: `{snapshot.text_runtime.temperature}`
- VLM temperature: `{snapshot.semantic_runtime.temperature}`

## Challenge suite

- Challenges: {result.challenge_count} (4 positive, 2 clean controls)
- Repeats: {result.repeat_count}
- Planned full runs: {aggregate.planned_runs}
- Completed full runs: {aggregate.completed_runs}
- Failed full runs: {aggregate.failed_runs}

## Positive results

- Truth discoveries: {aggregate.positive_truth_discovered}/{aggregate.positive_truth_count}
- Recall@1: {aggregate.recall_at_1}
- Recall@3: {aggregate.recall_at_3}
- Exact localizations: {aggregate.exact_localization_count}
- Partial localizations: {aggregate.partial_localization_count}

## Clean-control results

- Candidate false positives: {aggregate.clean_candidate_false_positives}
- Partial final false positives: {aggregate.clean_partial_final_false_positives}
- Strong final false positives: {aggregate.clean_strong_final_false_positives}

## Validation funnel and model reliance

Per-run blind results retain discovery, routing, quantitative, qualitative, usable-evidence,
and final-support counts. Model-reliance experiments are Agent-selected; the pilot runner does
not insert task-model, OOD, or counterfactual calls.

## Semantic VLM behavior

Preflight must distinguish a circle image from a square image using neutrally named files.
Scientific semantic observations and their contribution remain in each run artifact.

## Runtime reliability and usage

- Text calls: {aggregate.total_text_calls}
- VLM calls: {aggregate.total_vlm_calls}
- Retry calls: {aggregate.retry_calls}
- Schema retries: {aggregate.schema_retries}
- Prompt tokens: {aggregate.prompt_tokens}
- Completion tokens: {aggregate.completion_tokens}
- Total tokens: {aggregate.total_tokens}
- Usage complete: {aggregate.usage_complete}
- Total provider latency (ms): {aggregate.total_latency_ms}

## Failure analysis

{failures}

## Frozen workspaces

{workspaces}

## Scientific limitations

This is a small synthetic development pilot. It tests a blinded, provider-neutral scientific
protocol and records model behavior honestly; it is not a performance threshold, provider
comparison, ablation, or final-paper experiment. Future semantic backends may include local
VLMs and retrieval-augmented domain knowledge, but neither is part of Phase 16.
"""


def save_pilot_report(result: PilotResult, snapshot: PilotSnapshot, pilot_root: str | Path) -> Path:
    path = Path(pilot_root) / "report.md"
    path.write_text(render_pilot_report(result, snapshot), encoding="utf-8")
    return path


__all__ = ["render_pilot_report", "save_pilot_report"]
