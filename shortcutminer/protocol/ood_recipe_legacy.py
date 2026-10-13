"""Bounded Quantitative memory selection, distinct from experiment parameters."""
from typing import Literal
from pydantic import Field, model_validator
from shortcutminer.protocol.identify import NonEmptyText, StrictProtocolModel


class OODRecipeChoice(StrictProtocolModel):
    recipe_id: NonEmptyText
    field: NonEmptyText
    rationale: NonEmptyText
    supporting_tool_call_ids: list[NonEmptyText] = Field(default_factory=list)
