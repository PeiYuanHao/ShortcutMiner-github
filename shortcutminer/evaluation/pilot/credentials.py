"""Secret-safe credential presence checks for configured pilot runtimes."""

from __future__ import annotations

import os
from typing import Literal

from shortcutminer.evaluation.pilot.protocol import PilotConfig


CredentialPresence = Literal["SET", "MISSING"]


def credential_readiness(config: PilotConfig) -> dict[str, CredentialPresence]:
    """Report only presence; never return, transform, hash, or measure a value."""
    names = (
        config.text_runtime.api_key_env,
        config.semantic_runtime.api_key_env,
    )
    return {
        name: "SET" if bool(os.environ.get(name)) else "MISSING"
        for name in dict.fromkeys(names)
    }


__all__ = ["CredentialPresence", "credential_readiness"]
