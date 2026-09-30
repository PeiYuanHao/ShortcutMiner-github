"""Fake transport tests for live Pilot call, image, and token accounting."""

import json
from pathlib import Path

import pytest

from shortcutminer.evaluation.pilot.runner import PilotRunner
from shortcutminer.inspection.semantic_provider import OpenAICompatibleSemanticVisualInspector
from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.protocol.model_runtime import ModelImageInput
from shortcutminer.runtime.openai_compatible import (
    ModelRuntimeError, OpenAICompatibleClient, PilotInvocationBudgetState,
)
from test_real_blind_pilot import _config


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def post_json(self, _url, *, headers, body, timeout):
        self.calls += 1
        return self.responses.pop(0)


def _response(value, *, usage=True):
    result = {"choices": [{"message": {"content": json.dumps(value)}}]}
    if usage:
        result["usage"] = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    return result


def _client(transport, *, image=False, runs_root=None):
    return OpenAICompatibleClient(
        OpenAICompatibleModelConfig(
            model="offline", base_url="https://offline.invalid/v1",
            api_key_env="PHASE16_FAKE_KEY", temperature=0,
            capabilities={"supports_image_input": image},
            model_call_budget={"max_total_calls": 100, "max_calls_per_agent": 100,
                               "max_schema_retries": 2},
        ),
        api_key="offline", transport=transport, runs_root=runs_root,
        sleeper=lambda _: None,
    )


def _state(*, text_per_run=1, vlm_per_run=1, images=2, tokens=None):
    return PilotInvocationBudgetState(
        max_text_calls=20, max_vlm_calls=20, max_vlm_images=20,
        max_text_per_run=text_per_run, max_vlm_per_run=vlm_per_run,
        max_vlm_images_per_run=images, max_tokens=tokens,
    )


def _image(index):
    return ModelImageInput(
        sample_id=f"S{index}", sample_reference=f"images/{index}.png",
        image_sha256="a" * 64, mime_type="image/png", width=4, height=4,
    )


def _multimodal(client, count):
    return client.invoke_multimodal(
        run_id="budget-run", payload_for_hash={"run_id": "budget-run"},
        system_prompt="offline", validator=lambda value: (
            None if value == {"ok": True} else (_ for _ in ()).throw(ValueError("context"))
        ),
        content_parts=[{"type": "text", "text": "offline"}, *[
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
            for _ in range(count)
        ]],
        image_inputs=[_image(index) for index in range(count)],
    )


def test_text_schema_retry_uses_same_live_run_cap():
    transport = FakeTransport(_response({"invalid": True}), _response({"ok": True}))
    client = _client(transport)
    client._pilot_budget_state = _state()
    with pytest.raises(ModelRuntimeError) as error:
        client.invoke_agent_validated(
            "identify", {"run_id": "budget-run"}, system_prompt=None,
            validator=lambda value: (
                None if value == {"ok": True} else (_ for _ in ()).throw(ValueError("context"))
            ),
        )
    assert error.value.failure.category == "budget_exceeded"
    assert transport.calls == 1
    assert client._pilot_budget_state.text_calls == 1


def test_vlm_retry_and_actual_multi_image_count_hit_live_caps():
    transport = FakeTransport(_response({"invalid": True}), _response({"ok": True}))
    client = _client(transport, image=True)
    client._pilot_budget_state = _state(images=2)
    with pytest.raises(ModelRuntimeError) as error:
        _multimodal(client, 2)
    assert error.value.failure.category == "budget_exceeded"
    assert transport.calls == 1
    assert client._pilot_budget_state.vlm_images == 2

    second_transport = FakeTransport(_response({"ok": True}))
    second = _client(second_transport, image=True)
    second._pilot_budget_state = _state(images=1)
    with pytest.raises(ModelRuntimeError):
        _multimodal(second, 2)
    assert second_transport.calls == 0


def test_missing_token_usage_blocks_next_call_under_hard_cap(tmp_path: Path):
    transport = FakeTransport(_response({"ok": True}, usage=False), _response({"ok": True}))
    client = _client(transport, runs_root=tmp_path)
    client._pilot_budget_state = _state(text_per_run=3, tokens=100)
    assert client.invoke_agent("identify", {"run_id": "budget-run"}) == {"ok": True}
    with pytest.raises(ModelRuntimeError) as error:
        client.invoke_agent("identify", {"run_id": "budget-run"})
    assert error.value.failure.category == "budget_exceeded"
    assert transport.calls == 1
    assert client._pilot_budget_state.usage_incomplete


def test_pilot_resume_reconstructs_issued_calls_and_usage(tmp_path: Path):
    config = _config(tmp_path)
    root = Path(config.artifact_root) / config.pilot_id / "runs"
    text = _client(FakeTransport(_response({"ok": True})), runs_root=root)
    semantic = _client(FakeTransport(), image=True, runs_root=root)
    runner = PilotRunner(
        config, text_client=text,
        semantic_inspector=OpenAICompatibleSemanticVisualInspector(semantic),
    )
    runner._seed_global_call_caps(None)
    text.invoke_agent("identify", {"run_id": "resume-budget"})

    resumed_text = _client(FakeTransport(), runs_root=root)
    resumed_semantic = _client(FakeTransport(), image=True, runs_root=root)
    resumed = PilotRunner(
        config, text_client=resumed_text,
        semantic_inspector=OpenAICompatibleSemanticVisualInspector(resumed_semantic),
    )
    resumed._seed_global_call_caps(None)
    state = resumed_text._pilot_budget_state
    assert state is resumed_semantic._pilot_budget_state
    assert state.text_calls == 1
    assert state.by_run["resume-budget"]["text"] == 1
    assert state.known_tokens == 5
