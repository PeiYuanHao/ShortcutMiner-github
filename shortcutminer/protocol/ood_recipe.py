"""Bounded Quantitative memory selection, distinct from experiment parameters."""
from typing import Literal
from pydantic import Field, model_validator
from shortcutminer.protocol.identify import NonEmptyText, StrictProtocolModel
from shortcutminer.protocol.segmentation import LocalForegroundSpec


class OODRecipeSelectionPolicy(StrictProtocolModel):
    catalog_version: Literal['ood-memory-v1', 'ood-memory-v2'] = 'ood-memory-v1'
    fields: list[NonEmptyText] = Field(min_length=1)
    construction_protocol: Literal['legacy-v1', 'quality-v2'] = 'legacy-v1'
    max_quality_revisions: int = Field(default=2, ge=0, le=3)
    local_foreground_tools: dict[NonEmptyText, LocalForegroundSpec] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def preserve_existing_explicit_quality_configs(cls, value):
        if isinstance(value, dict) and 'construction_protocol' not in value:
            # v2 configs explicitly opted into revisions/tools before the named
            # switch existed. Plain v1 configs keep the legacy selection path.
            if value.get('catalog_version') == 'ood-memory-v2' or 'max_quality_revisions' in value or 'local_foreground_tools' in value:
                return {**value, 'construction_protocol': 'quality-v2'}
        return value

    @model_validator(mode='after')
    def unique_fields(self):
        if len(self.fields) != len(set(self.fields)):
            raise ValueError('Recipe authorization fields must be unique')
        if self.construction_protocol == 'legacy-v1' and (self.catalog_version != 'ood-memory-v1' or self.local_foreground_tools):
            raise ValueError('Legacy selection requires the frozen v1 catalog without local tool bindings')
        return self


class OODRecipeChoice(StrictProtocolModel):
    recipe_id: NonEmptyText
    field: NonEmptyText
    rationale: NonEmptyText
    supporting_tool_call_ids: list[NonEmptyText] = Field(default_factory=list)
    local_foreground_tool: NonEmptyText | None = None
    object_prompt: NonEmptyText | None = Field(default=None, max_length=120)
