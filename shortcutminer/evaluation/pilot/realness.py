"""Fail-closed scientific-runtime realness checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FORBIDDEN_SCIENTIFIC_RUNTIME_NAMES = (
    "MockAgentClient", "MockSemanticVisualInspector", "FakeTransport",
    "MockExperimentExecutor", "DemoAgentClient", "controlled Agent",
)


class PilotRealnessError(RuntimeError):
    """Raised when a scientific run is wired to a fake or mock component."""


def assert_real_provider(component: Any, *, role: str) -> None:
    name = f"{type(component).__module__}.{type(component).__name__}"
    if getattr(component, "runtime_kind", None) != "real_provider":
        raise PilotRealnessError(f"{role} must declare runtime_kind=real_provider")
    if any(token.casefold() in name.casefold() for token in FORBIDDEN_SCIENTIFIC_RUNTIME_NAMES):
        raise PilotRealnessError(f"{role} uses forbidden scientific runtime {name}")
    nested = getattr(component, "client", None)
    if nested is not None:
        assert_real_provider(nested, role=f"{role} client")


def audit_scientific_artifacts(root: str | Path) -> list[str]:
    findings: list[str] = []
    base = Path(root)
    if not base.exists():
        return findings
    for path in base.rglob("*.json"):
        try:
            serialized = json.dumps(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        for token in FORBIDDEN_SCIENTIFIC_RUNTIME_NAMES:
            if token.casefold() in serialized.casefold():
                findings.append(f"{path.relative_to(base).as_posix()}: {token}")
    return findings


__all__ = [
    "FORBIDDEN_SCIENTIFIC_RUNTIME_NAMES", "PilotRealnessError",
    "assert_real_provider", "audit_scientific_artifacts",
]
