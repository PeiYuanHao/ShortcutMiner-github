"""Offline provider-response persistence boundary preflight for Phase 16."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from shortcutminer.inspection.semantic_provider import (
    OpenAICompatibleSemanticVisualInspector,
)
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.protocol.visual_discovery import (
    SemanticInspectionRequest,
    SemanticInspectionSample,
)
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient
from shortcutminer.storage.model_calls import load_run_model_telemetry


RUN_ID = "phase16-provider-security-offline"
_RAW_SENTINEL = "RAW_PROVIDER_SENTINEL_7F21"
_REASONING_SENTINEL = "REASONING_SENTINEL_91AC"
_AUTH_SENTINEL = "AUTH_SENTINEL_44D2"
_BASE64_SENTINEL = "BASE64_SENTINEL_C130"


class _SentinelTransport:
    """Hold unsafe provider data in memory while exercising production parsing."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.image_input_verified = False
        self.image_data_url: str | None = None

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout: float,
    ) -> object:
        self.calls.append({
            "url": url,
            "headers": dict(headers),
            "body": body,
            "timeout": timeout,
        })
        if headers.get("Authorization") != f"Bearer {_AUTH_SENTINEL}":
            raise RuntimeError("offline transport did not receive the in-memory credential")

        messages = body.get("messages")
        is_multimodal = False
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict) or part.get("type") != "image_url":
                        continue
                    image_url = part.get("image_url")
                    url_value = image_url.get("url") if isinstance(image_url, dict) else None
                    if isinstance(url_value, str) and url_value.startswith("data:image/png;base64,"):
                        is_multimodal = True
                        self.image_data_url = url_value
        self.image_input_verified = self.image_input_verified or is_multimodal

        content: dict[str, Any]
        if is_multimodal:
            content = {
                "schema_version": "1.0",
                "observations": [{
                    "sample_id": "sample-a",
                    "sample_reference": "sample.png",
                    "observation_type": "foreground_object",
                    "region": "center",
                    "description": "a small blue square",
                    "confidence": "high",
                    "limitations": [],
                }],
            }
        else:
            content = {"ok": True}

        # These unsafe fields deliberately arrive in the same response envelope
        # used by a real OpenAI-compatible provider. Production persistence must
        # retain only safe derived telemetry, never these values.
        return {
            "choices": [{
                "message": {
                    "content": json.dumps(content),
                    "reasoning_content": _REASONING_SENTINEL,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
            "provider_debug": _RAW_SENTINEL,
            "authorization_echo": _AUTH_SENTINEL,
            "base64_echo": _BASE64_SENTINEL,
        }


def _config(*, image_input: bool) -> OpenAICompatibleModelConfig:
    return OpenAICompatibleModelConfig.model_validate({
        "model": "offline-security-sentinel",
        "base_url": "https://offline.invalid/v1",
        "api_key_env": "SHORTCUTMINER_OFFLINE_SENTINEL_KEY",
        "capabilities": {
            "supports_image_input": image_input,
            "supports_json_object": True,
        },
        "retry_policy": {"max_attempts": 1},
    })


def _scan(root: Path) -> dict[str, int]:
    markers = {
        "raw_provider_body": _RAW_SENTINEL.encode(),
        "reasoning": _REASONING_SENTINEL.encode(),
        "authorization": _AUTH_SENTINEL.encode(),
        "base64": _BASE64_SENTINEL.encode(),
    }
    hits = {name: 0 for name in markers}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes()
        for name, marker in markers.items():
            hits[name] += data.count(marker)
    return hits


def run_preflight(root: Path) -> dict[str, Any]:
    """Exercise text and image requests without any external provider call."""
    root.mkdir(parents=True, exist_ok=True)
    transport = _SentinelTransport()
    runs_root = root / "runs"

    text_client = OpenAICompatibleClient(
        _config(image_input=False),
        api_key=_AUTH_SENTINEL,
        transport=transport,
        runs_root=runs_root,
    )
    text_result = text_client.invoke_agent("identify", {"run_id": RUN_ID})
    if text_result != {"ok": True}:
        raise RuntimeError("offline text response did not complete")

    dataset = root / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (20, 80, 220)).save(dataset / "sample.png")
    vision_client = OpenAICompatibleClient(
        _config(image_input=True),
        api_key=_AUTH_SENTINEL,
        transport=transport,
        runs_root=runs_root,
    )
    observations = OpenAICompatibleSemanticVisualInspector(vision_client).inspect(
        dataset_root=dataset,
        request=SemanticInspectionRequest(
            run_id=RUN_ID,
            dataset_name="offline-security-fixture",
            samples=[SemanticInspectionSample(
                sample_id="sample-a",
                sample_reference="sample.png",
            )],
        ),
    )
    if len(observations) != 1 or not transport.image_input_verified:
        raise RuntimeError("offline multimodal request did not traverse the image path")

    telemetry = load_run_model_telemetry(runs_root, RUN_ID)
    telemetry_ok = bool(
        telemetry is not None
        and telemetry.total_calls == 2
        and telemetry.successful_calls == 2
        and telemetry.usage_complete
        and telemetry.total_tokens == 36
        and telemetry.total_model_latency_ms >= 0
        and len(telemetry.invocation_references) == 2
    )
    snapshot_flags = [
        json.loads(path.read_text(encoding="utf-8")).get("raw_response_persistence")
        for path in (runs_root / RUN_ID / "model_calls").glob(
            "runtime_snapshot-*.json"
        )
    ]
    raw_provider_body_persistence = not bool(
        snapshot_flags and all(value is False for value in snapshot_flags)
    )
    hits = _scan(root)
    image_payload_hits = 0
    base64_prefix_hits = 0
    for path in root.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            if transport.image_data_url is not None:
                image_payload_hits += data.count(transport.image_data_url.encode())
            base64_prefix_hits += data.count(b"base64,")
    passed = (
        not raw_provider_body_persistence
        and telemetry_ok
        and len(transport.calls) == 2
        and all(value == 0 for value in hits.values())
        and image_payload_hits == 0
        and base64_prefix_hits == 0
    )
    if not passed:
        raise RuntimeError("provider persistence security preflight failed")
    return {
        "status": "passed",
        "raw_provider_body_persistence": False,
        "sentinel_hits": hits,
        "image_payload_hits": image_payload_hits,
        "base64_prefix_hits": base64_prefix_hits,
        "sanitized_telemetry_persisted": True,
        "image_input_verified": True,
        "offline_transport_calls": len(transport.calls),
        "provider_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
