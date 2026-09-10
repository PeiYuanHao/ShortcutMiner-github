"""Provider-neutral interface for audited classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from shortcutminer.protocol.model_reliance import (
    TaskModelArtifact,
    TaskModelCapabilities,
)


class TaskModelAdapter(Protocol):
    model_id: str
    model_kind: str
    capabilities: TaskModelCapabilities

    def fit(self, paths: list[Path], labels: list[str], *, seed: int, max_iterations: int) -> None: ...
    def predict(self, paths: list[Path]) -> list[str]: ...
    def predict_proba(self, paths: list[Path]) -> np.ndarray: ...
    def predict_logits(self, paths: list[Path]) -> np.ndarray: ...
    def save_safe_artifact(self, **kwargs) -> TaskModelArtifact: ...
    @classmethod
    def load_safe_artifact(cls, model_path: Path, **kwargs) -> tuple["TaskModelAdapter", TaskModelArtifact]: ...


class ExternalTaskModelAdapter(Protocol):
    """Future extension point; no arbitrary external adapter is registered by default."""

    model_id: str
    capabilities: TaskModelCapabilities

    @property
    def available(self) -> bool: ...


__all__ = ["ExternalTaskModelAdapter", "TaskModelAdapter"]
