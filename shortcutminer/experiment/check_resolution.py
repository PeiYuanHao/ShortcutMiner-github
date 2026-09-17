"""Deterministic translation from Executive checks to experiment backends."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import get_args

from shortcutminer.experiment.tool_executor import ExperimentType


QUANTITATIVE_CHECK_RESOLUTION_VERSION = "quantitative_check_resolution_v1"

EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES = MappingProxyType({
    "association_test": ("statistical_check",),
    "task_model_training": ("task_model_training",),
    "task_model_evaluation": ("task_model_evaluation",),
    "ood_evaluation": ("ood_evaluation",),
    "counterfactual_intervention": ("counterfactual_intervention",),
})

QUANTITATIVE_EXPERIMENT_TYPES = frozenset(get_args(ExperimentType))


class UnsupportedQuantitativeCheck(ValueError):
    """Raised when Executive emits an unmapped quantitative planning check."""

    category = "unsupported_quantitative_check"

    def __init__(self, check: str) -> None:
        self.check = check
        super().__init__(f"Unsupported quantitative planning check: {check}")


def resolve_quantitative_checks(
    requested_checks: list[str] | tuple[str, ...],
) -> tuple[ExperimentType, ...]:
    """Resolve checks in stable input/mapping order without fuzzy matching."""

    resolved: list[ExperimentType] = []
    seen: set[str] = set()
    for check in requested_checks:
        targets = EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES.get(check)
        if targets is None:
            raise UnsupportedQuantitativeCheck(check)
        for target in targets:
            if target not in QUANTITATIVE_EXPERIMENT_TYPES:
                raise RuntimeError(
                    f"Mapped quantitative experiment type is unsupported: {target}"
                )
            if target not in seen:
                resolved.append(target)
                seen.add(target)
    return tuple(resolved)


def _mapping_hash() -> str:
    encoded = json.dumps(
        {
            "version": QUANTITATIVE_CHECK_RESOLUTION_VERSION,
            "mapping": [
                [check, list(targets)]
                for check, targets in EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES.items()
            ],
        },
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


QUANTITATIVE_CHECK_RESOLUTION_HASH = _mapping_hash()


__all__ = [
    "EXECUTIVE_CHECK_TO_EXPERIMENT_TYPES",
    "QUANTITATIVE_CHECK_RESOLUTION_HASH",
    "QUANTITATIVE_CHECK_RESOLUTION_VERSION",
    "QUANTITATIVE_EXPERIMENT_TYPES",
    "UnsupportedQuantitativeCheck",
    "resolve_quantitative_checks",
]
