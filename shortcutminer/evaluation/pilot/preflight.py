"""Minimal real text/VLM preflight with two content-distinguishable images."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Literal

from PIL import Image, ImageDraw

from shortcutminer.evaluation.pilot.protocol import PilotPreflightResult, PreflightCheck
from shortcutminer.inspection.semantic import SemanticVisualInspector
from shortcutminer.protocol.identify import StrictProtocolModel
from shortcutminer.protocol.visual_discovery import SemanticInspectionRequest
from shortcutminer.runtime.openai_compatible import ModelRuntimeError, OpenAICompatibleClient
from shortcutminer.storage.json_artifacts import atomic_write_json


class _TextPreflightResponse(StrictProtocolModel):
    status: Literal["ok"]


def _write_assets(root: Path) -> Path:
    asset_root = root / "preflight_assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    specs = (("sample_a.png", "circle"), ("sample_b.png", "square"))
    for name, shape in specs:
        image = Image.new("RGB", (64, 64), (35, 95, 210))
        draw = ImageDraw.Draw(image)
        if shape == "circle":
            draw.ellipse((18, 18, 46, 46), fill=(250, 220, 30))
        else:
            draw.rectangle((18, 18, 46, 46), fill=(235, 45, 170))
        image.save(asset_root / name, format="PNG")
    return asset_root


def _usage_since(client: OpenAICompatibleClient, start: int) -> bool:
    records = client.invocations[start:]
    return bool(records) and all(item.usage.available for item in records)


def _retry_counts(client: OpenAICompatibleClient, start: int) -> tuple[int, int]:
    records = client.invocations[start:]
    return (
        sum(item.schema_retry for item in records),
        sum(item.transport_retry for item in records),
    )


def _image_input_verified(client: OpenAICompatibleClient, start: int) -> bool:
    records = client.invocations[start:]
    expected = {
        ("sample-a", "sample_a.png"),
        ("sample-b", "sample_b.png"),
    }
    observed = {
        (image.sample_id, image.sample_reference)
        for record in records
        if record.runtime_kind == "semantic_vlm"
        for image in record.image_inputs
    }
    return bool(records) and expected <= observed


class RealProviderPreflight:
    def __init__(self, text_client: OpenAICompatibleClient,
                 semantic_inspector: SemanticVisualInspector) -> None:
        self.text_client = text_client
        self.semantic_inspector = semantic_inspector

    def run(self, *, pilot_id: str, pilot_root: str | Path) -> PilotPreflightResult:
        root = Path(pilot_root)
        root.mkdir(parents=True, exist_ok=True)
        if getattr(self.text_client, "runtime_kind", None) != "real_provider":
            raise TypeError("text preflight requires runtime_kind=real_provider")
        if getattr(self.semantic_inspector, "runtime_kind", None) != "real_provider":
            raise TypeError("semantic preflight requires runtime_kind=real_provider")
        text_start = len(self.text_client.invocations)
        start = time.monotonic()
        try:
            value = self.text_client.invoke_agent_validated(
                "identify",
                {"run_id": f"{pilot_id}-text-preflight", "request": "Return exactly {status: ok}."},
                system_prompt="Return one strict JSON object with status equal to ok.",
                validator=lambda item: _TextPreflightResponse.model_validate(item),
            )
            _TextPreflightResponse.model_validate(value)
            text = PreflightCheck(
                name="text_structured_json", success=True,
                provider=self.text_client.config.provider, model=self.text_client.config.model,
                runtime_kind="real_provider", latency_ms=(time.monotonic() - start) * 1000,
                usage_available=_usage_since(self.text_client, text_start), message="HTTP, credentials, model, JSON, schema, latency, and telemetry succeeded.",
            )
        except Exception as exc:
            failure = exc.failure.category if isinstance(exc, ModelRuntimeError) and exc.failure else type(exc).__name__
            text = PreflightCheck(
                name="text_structured_json", success=False,
                provider=self.text_client.config.provider, model=self.text_client.config.model,
                runtime_kind="real_provider", latency_ms=(time.monotonic() - start) * 1000,
                usage_available=_usage_since(self.text_client, text_start), failure_category=failure,
                message=f"Text preflight failed: {type(exc).__name__}: {exc}",
            )

        text_schema_retries, text_transport_retries = _retry_counts(
            self.text_client, text_start,
        )
        text_records = self.text_client.invocations[text_start:]
        text_final = text_records[-1] if text_records else None
        text = text.model_copy(update={
            "finish_reason": (
                text_final.response_diagnostics.finish_reason
                if text_final is not None
                and text_final.response_diagnostics is not None else None
            ),
            "completion_tokens": (
                text_final.usage.completion_tokens if text_final is not None else None
            ),
            "reasoning_tokens": (
                text_final.usage.reasoning_tokens if text_final is not None else None
            ),
            "schema_retry_count": text_schema_retries,
            "transport_retry_count": text_transport_retries,
        })
        if not text.success:
            semantic = PreflightCheck(
                name="semantic_two_image_content", success=False,
                provider=getattr(self.semantic_inspector, "provider_name", "unknown"),
                model=getattr(self.semantic_inspector, "model_name", "unknown"),
                runtime_kind="real_provider", latency_ms=0.0,
                usage_available=False, failure_category="blocked_by_text_preflight",
                message="VLM preflight was not attempted because text preflight failed.",
            )
            result = PilotPreflightResult(
                pilot_id=pilot_id, status="blocked", text=text, semantic_vlm=semantic,
                image_input_verified=False, image_content_distinguishable=False,
                image_names=["sample_a.png", "sample_b.png"],
                completed_at=datetime.now(timezone.utc).isoformat(),
                limitations=[
                    "Scientific pilot execution is blocked until both configured real providers pass preflight."
                ],
            )
            atomic_write_json(
                root / "preflight.json", result.model_dump(mode="json"),
                error_type=RuntimeError,
            )
            return result

        asset_root = _write_assets(root)
        semantic_client = getattr(self.semantic_inspector, "client", None)
        semantic_start = len(semantic_client.invocations) if semantic_client is not None else 0
        start = time.monotonic()
        descriptions: dict[str, str] = {}
        semantic_error: Exception | None = None
        try:
            observations = self.semantic_inspector.inspect(
                dataset_root=asset_root,
                request=SemanticInspectionRequest(
                    run_id=f"{pilot_id}-vlm-preflight", dataset_name="preflight",
                    samples=[
                        {"sample_id": "sample-a", "sample_reference": "sample_a.png"},
                        {"sample_id": "sample-b", "sample_reference": "sample_b.png"},
                    ],
                ),
            )
            image_input_verified = (
                semantic_client is not None
                and _image_input_verified(semantic_client, semantic_start)
            )
            if not image_input_verified:
                raise ValueError(
                    "VLM telemetry did not verify both image inputs in the multimodal request"
                )
            for item in observations:
                descriptions[item.sample_id] = f"{descriptions.get(item.sample_id, '')} {item.description}".strip().casefold()
            a = descriptions.get("sample-a", "")
            b = descriptions.get("sample-b", "")
            distinguishable = bool(
                a and b and a != b
                and any(token in a for token in ("circle", "circular", "round"))
                and any(token in b for token in ("square", "rectangle", "rectangular"))
            )
            if not distinguishable:
                raise ValueError("VLM observations did not distinguish the central circle from the square")
        except Exception as exc:
            semantic_error = exc
            distinguishable = False
            image_input_verified = (
                semantic_client is not None
                and _image_input_verified(semantic_client, semantic_start)
            )
        semantic_usage = (
            _usage_since(semantic_client, semantic_start) if semantic_client is not None else False
        )
        semantic_records = (
            semantic_client.invocations[semantic_start:]
            if semantic_client is not None else []
        )
        semantic_final = semantic_records[-1] if semantic_records else None
        semantic_schema_retries, semantic_transport_retries = (
            _retry_counts(semantic_client, semantic_start)
            if semantic_client is not None else (0, 0)
        )
        semantic = PreflightCheck(
            name="semantic_two_image_content", success=semantic_error is None,
            provider=getattr(self.semantic_inspector, "provider_name", "unknown"),
            model=getattr(self.semantic_inspector, "model_name", "unknown"),
            runtime_kind="real_provider", latency_ms=(time.monotonic() - start) * 1000,
            usage_available=semantic_usage,
            finish_reason=(
                semantic_final.response_diagnostics.finish_reason
                if semantic_final is not None
                and semantic_final.response_diagnostics is not None
                else None
            ),
            completion_tokens=(
                semantic_final.usage.completion_tokens
                if semantic_final is not None else None
            ),
            reasoning_tokens=(
                semantic_final.usage.reasoning_tokens
                if semantic_final is not None else None
            ),
            schema_retry_count=semantic_schema_retries,
            transport_retry_count=semantic_transport_retries,
            failure_category=type(semantic_error).__name__ if semantic_error else None,
            message=(
                "Both image inputs were recorded in the multimodal request and produced content-distinguishable structured observations."
                if semantic_error is None else
                f"VLM preflight failed: {type(semantic_error).__name__}: {semantic_error}"
            ),
        )
        status = "passed" if text.success and semantic.success and distinguishable else "blocked"
        result = PilotPreflightResult(
            pilot_id=pilot_id, status=status, text=text, semantic_vlm=semantic,
            image_input_verified=image_input_verified,
            image_content_distinguishable=distinguishable,
            image_names=["sample_a.png", "sample_b.png"],
            completed_at=datetime.now(timezone.utc).isoformat(),
            limitations=[] if status == "passed" else ["Scientific pilot execution is blocked until both configured real providers pass preflight."],
        )
        atomic_write_json(root / "preflight.json", result.model_dump(mode="json"), error_type=RuntimeError)
        return result


__all__ = ["RealProviderPreflight"]
