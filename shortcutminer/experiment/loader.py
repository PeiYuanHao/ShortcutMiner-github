"""Load a formal Runner request from JSON or YAML configuration."""

import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.protocol.run import RunRequest


class RunConfigError(RuntimeError):
    """Raised when a run configuration cannot be loaded or validated."""


def _load_config_object(path: str | Path) -> dict:
    """Read a local JSON/YAML object without interpreting its schema."""
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            value = json.loads(text)
        elif config_path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(text)
        else:
            raise RunConfigError(
                "Run configuration must use .json, .yaml, or .yml."
            )
    except RunConfigError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise RunConfigError(
            f"Unable to load run configuration: {config_path}"
        ) from exc
    if not isinstance(value, dict):
        raise RunConfigError("Run configuration must contain an object.")
    return value


def load_run_request(path: str | Path) -> RunRequest:
    """Read one local config file and apply the formal RunRequest schema."""
    value = _load_config_object(path)
    try:
        return RunRequest.model_validate(value)
    except ValidationError as exc:
        raise RunConfigError("Run configuration is invalid.") from exc


def load_model_runtime_config(
    path: str | Path,
) -> OpenAICompatibleModelConfig:
    """Load safe model settings; credentials remain environment references."""
    value = _load_config_object(path)
    try:
        return OpenAICompatibleModelConfig.model_validate(value)
    except ValidationError as exc:
        raise RunConfigError("Model runtime configuration is invalid.") from exc


__all__ = [
    "RunConfigError",
    "load_model_runtime_config",
    "load_run_request",
]
