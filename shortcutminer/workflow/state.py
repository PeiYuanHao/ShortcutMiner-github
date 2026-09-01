"""Shared state carried through the ShortcutMiner workflow."""

from operator import add
from typing import Annotated, Any, NotRequired, Required, TypedDict


class ShortcutMinerState(TypedDict, total=False):
    """Minimal state envelope for the five-stage ShortcutMiner workflow."""

    run_id: Required[str]
    dataset_info: Required[dict[str, Any]]
    task_input: Required[dict[str, Any]]

    identify_result: NotRequired[Any]
    executive_result: NotRequired[Any]
    quantitative_result: NotRequired[Any]
    qualitative_result: NotRequired[Any]
    summarize_result: NotRequired[Any]

    errors: Annotated[list[str], add]
    metadata: dict[str, Any]
