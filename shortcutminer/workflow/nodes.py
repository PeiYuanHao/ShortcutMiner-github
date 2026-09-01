"""Placeholder nodes for the Phase 1 LangGraph workflow."""

from typing import Any

from shortcutminer.workflow.state import ShortcutMinerState


def _placeholder_result(node_name: str) -> dict[str, str]:
    return {
        "node": node_name,
        "status": "placeholder",
    }


def identify_node(state: ShortcutMinerState) -> dict[str, Any]:
    """Record that the Identify placeholder was reached."""
    # TODO: Phase 2 migration - connect the existing Identify business logic.
    return {"identify_result": _placeholder_result("identify")}


def executive_node(state: ShortcutMinerState) -> dict[str, Any]:
    """Record that the Executive placeholder was reached."""
    # TODO: Phase 3 migration - connect the existing Executive business logic.
    return {"executive_result": _placeholder_result("executive")}


def quantitative_node(state: ShortcutMinerState) -> dict[str, Any]:
    """Record that the Quantitative placeholder was reached."""
    # TODO: Phase 4 migration - connect the existing Quantitative business logic.
    return {"quantitative_result": _placeholder_result("quantitative")}


def qualitative_node(state: ShortcutMinerState) -> dict[str, Any]:
    """Record that the Qualitative placeholder was reached."""
    # TODO: Phase 4 migration - connect the existing Qualitative business logic.
    return {"qualitative_result": _placeholder_result("qualitative")}


def summarize_node(state: ShortcutMinerState) -> dict[str, Any]:
    """Record that the Summarize placeholder was reached."""
    # TODO: Phase 5 migration - connect the existing Summarize business logic.
    return {"summarize_result": _placeholder_result("summarize")}
