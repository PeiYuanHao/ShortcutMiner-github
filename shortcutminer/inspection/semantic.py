"""Provider-neutral optional semantic image-inspection boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Protocol

from shortcutminer.protocol.visual_discovery import (
    SemanticInspectionRequest,
    SemanticObservation,
)


class SemanticVisualInspector(Protocol):
    """Optional runtime capability; implementations receive only public samples."""

    @property
    def provider_name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def inspect(
        self,
        *,
        dataset_root: str | Path,
        request: SemanticInspectionRequest,
    ) -> list[SemanticObservation]: ...


class MockSemanticVisualInspector:
    """Deterministic contract fixture; it performs no model or network call."""

    provider_name = "mock_semantic_visual_inspector"
    version = "mock_semantic_v1"

    def __init__(
        self,
        observations: Mapping[str, Sequence[SemanticObservation | dict]],
    ) -> None:
        self._observations = {
            sample_id: [
                item
                if isinstance(item, SemanticObservation)
                else SemanticObservation.model_validate(item)
                for item in items
            ]
            for sample_id, items in observations.items()
        }
        self.calls: list[SemanticInspectionRequest] = []

    def inspect(
        self,
        *,
        dataset_root: str | Path,
        request: SemanticInspectionRequest,
    ) -> list[SemanticObservation]:
        root = Path(dataset_root).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError("semantic inspection dataset root is not a directory")
        self.calls.append(request.model_copy(deep=True))
        allowed = {
            (sample.sample_id, sample.sample_reference) for sample in request.samples
        }
        output: list[SemanticObservation] = []
        for sample in request.samples:
            for observation in self._observations.get(sample.sample_id, []):
                if (observation.sample_id, observation.sample_reference) not in allowed:
                    raise ValueError(
                        "mock semantic observation is outside the approved sample request"
                    )
                output.append(observation.model_copy(deep=True))
        return deepcopy(output)


__all__ = ["MockSemanticVisualInspector", "SemanticVisualInspector"]
