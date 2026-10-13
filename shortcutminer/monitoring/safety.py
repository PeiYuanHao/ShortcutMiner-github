"""Small allowlisted presentation boundary: never render raw errors or payloads."""

import re


def safe_text(value, limit=160):
    if not isinstance(value, str):
        return "N/A"
    if re.search(r"authorization|bearer\s|api[_ -]?key|\bsk-[a-z0-9]|base64|data:|reasoning|secret", value, re.I):
        return "[redacted]"
    return " ".join("".join(char for char in value if char.isprintable()).split())[:limit]


def failure_reason(details):
    # Free-form exception messages and Provider response bodies are never shown.
    allowed = {"backend_execution_failed", "stage_execution_failed", "task_execution_failed",
               "weak_signal", "insufficient_metadata", "unsupported_validation", "resource_budget",
               "insufficient_samples", "low_priority", "authentication_failed", "authorization_failed",
               "rate_limited", "timed_out", "transport_failed", "provider_5xx", "provider_rejected",
               "invalid_response_envelope", "invalid_json", "protocol_schema_failed", "budget_exceeded", "cancelled"}
    for value in (details.get("reason"), details.get("category"), details.get("failure_category")):
        if isinstance(value, str) and value in allowed:
            return value
    return "backend_execution_failed" if "error" in details else "stage_execution_failed"
