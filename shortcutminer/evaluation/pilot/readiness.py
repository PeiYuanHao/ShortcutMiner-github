"""Secret-free completeness checks for Phase 16 real-provider configuration."""

from __future__ import annotations

import re

from shortcutminer.evaluation.pilot.protocol import PilotConfig


_PLACEHOLDER = re.compile(
    r"<[^<>]+>|placeholder|replace[-_ ]?with|change[-_ ]?me",
    re.IGNORECASE,
)


def _is_placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER.search(value))


def real_provider_configuration_blockers(config: PilotConfig) -> list[str]:
    """Return public field labels for unresolved real-runtime placeholders."""
    checks = (
        ("TEXT_MODEL_ID", config.text_runtime.model),
        ("TEXT_API_BASE_URL", config.text_runtime.base_url),
        ("VLM_MODEL_ID", config.semantic_runtime.model),
        ("VLM_API_BASE_URL", config.semantic_runtime.base_url),
    )
    return [name for name, value in checks if _is_placeholder(value)]


__all__ = ["real_provider_configuration_blockers"]
