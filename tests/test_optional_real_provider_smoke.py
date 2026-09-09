"""Opt-in network smoke tests; skipped unless explicitly enabled."""

import os
from pathlib import Path

from PIL import Image
import pytest

from shortcutminer.inspection.semantic_provider import OpenAICompatibleSemanticVisualInspector
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.protocol.visual_discovery import (
    SemanticInspectionRequest,
    SemanticInspectionSample,
)
from shortcutminer.runtime.openai_compatible import OpenAICompatibleClient


def _real_config(*, vision: bool) -> OpenAICompatibleModelConfig:
    prefix = "SHORTCUTMINER_REAL_VLM" if vision else "SHORTCUTMINER_REAL_TEXT"
    required = {
        "model": os.environ.get(f"{prefix}_MODEL"),
        "base_url": os.environ.get(f"{prefix}_BASE_URL"),
        "api_key_env": os.environ.get(f"{prefix}_API_KEY_ENV"),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        pytest.skip(f"missing opt-in provider settings: {', '.join(missing)}")
    return OpenAICompatibleModelConfig(
        **required,
        max_tokens=512,
        capabilities={"supports_image_input": vision},
    )


@pytest.mark.skipif(
    os.environ.get("SHORTCUTMINER_REAL_PROVIDER_TEST") != "1"
    and os.environ.get("SHORTCUTMINER_REAL_TEXT_SMOKE") != "1",
    reason="real text-provider smoke is opt-in",
)
def test_optional_real_text_provider_smoke():
    result = OpenAICompatibleClient(_real_config(vision=False)).invoke_agent(
        "identify",
        {"run_id": "optional-real-text-smoke", "instruction": "Return an empty JSON object."},
        system_prompt="Return only one JSON object.",
    )
    assert isinstance(result, dict)


@pytest.mark.skipif(
    os.environ.get("SHORTCUTMINER_REAL_VLM_TEST") != "1"
    and os.environ.get("SHORTCUTMINER_REAL_VLM_SMOKE") != "1",
    reason="real semantic-VLM smoke is opt-in",
)
def test_optional_real_semantic_vlm_smoke(tmp_path: Path):
    Image.new("RGB", (8, 8), (30, 120, 60)).save(tmp_path / "tiny.png")
    inspector = OpenAICompatibleSemanticVisualInspector(
        OpenAICompatibleClient(_real_config(vision=True), runs_root=tmp_path / "runs")
    )
    output = inspector.inspect(
        dataset_root=tmp_path,
        request=SemanticInspectionRequest(
            run_id="optional-real-vlm-smoke",
            dataset_name="tiny-public-smoke",
            samples=[SemanticInspectionSample(sample_id="tiny", sample_reference="tiny.png")],
        ),
    )
    assert isinstance(output, list)
