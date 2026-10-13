"""Caller-frozen local foreground models; no provider-chosen code or downloads."""
from typing import Literal, Annotated
from pydantic import Field, StringConstraints, model_validator
from shortcutminer.protocol.identify import StrictProtocolModel, NonEmptyText

SHA256 = Annotated[str, StringConstraints(pattern=r'^[0-9a-f]{64}$')]

class LocalModelFiles(StrictProtocolModel):
    directory: NonEmptyText
    files_sha256: dict[NonEmptyText, SHA256]

class LocalForegroundSpec(StrictProtocolModel):
    sam: LocalModelFiles
    grounding_dino: LocalModelFiles
    vitmatte: LocalModelFiles | None = None
    transformers_version: Literal['5.14.1'] = '5.14.1'
    object_prompt: NonEmptyText = 'bird.'
    box_threshold: float = Field(default=.25, gt=0, lt=1)
    text_threshold: float = Field(default=.25, gt=0, lt=1)
    max_boxes: int = Field(default=4, ge=1, le=16)
    detector_shortest_edge: int = Field(default=512, ge=128, le=800)
    detector_longest_edge: int = Field(default=768, ge=256, le=1333)
    minimum_sam_score: float = Field(default=.8, gt=0, lt=1)
    trimap_kernel: int = Field(default=13, ge=3, le=31)
    alpha_protection_threshold: float = Field(default=.05, gt=0, lt=1)
    protection_margin: int = Field(default=8, ge=0, le=32)
    minimum_coverage: float = Field(default=.005, gt=0, lt=1)
    maximum_coverage: float = Field(default=.95, gt=0, lt=1)

    @model_validator(mode='after')
    def valid_morphology(self):
        if self.trimap_kernel % 2 != 1 or self.minimum_coverage >= self.maximum_coverage or self.detector_shortest_edge > self.detector_longest_edge:
            raise ValueError('Odd trimap kernel and ordered coverage bounds required')
        return self
