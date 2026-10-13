"""Run-level export envelope; scientific protocols remain the authority."""

from typing import Any, Literal

from shortcutminer.protocol.identify import StrictProtocolModel


class AuditReport(StrictProtocolModel):
    schema_version: Literal["1.0"] = "1.0"
    run: dict[str, Any]
    dataset: dict[str, Any]
    runtime: dict[str, Any]
    discovery: dict[str, Any]
    screening: dict[str, Any]
    validation: dict[str, Any]
    quantitative: list[dict[str, Any]]
    qualitative: list[dict[str, Any]]
    evidence: dict[str, Any]
    findings: dict[str, Any]
    cost: dict[str, Any]
    reproducibility: dict[str, Any]
    limitations: list[dict[str, Any]]
    artifacts: dict[str, Any]
    evidence_promotion: dict[str, Any] | None = None
