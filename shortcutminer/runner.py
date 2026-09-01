"""Minimal runner entry point for the Phase 1 LangGraph workflow."""

from shortcutminer.orchestrator import build_workflow
from shortcutminer.workflow.state import ShortcutMinerState


def run_workflow(initial_state: ShortcutMinerState) -> ShortcutMinerState:
    """Invoke the placeholder workflow once and return its final state."""
    return build_workflow().invoke(initial_state)
