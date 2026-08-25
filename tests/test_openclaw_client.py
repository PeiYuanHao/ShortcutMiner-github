import pytest

from shortcutminer.openclaw_client import (
    OpenClawClient,
    OpenClawClientConfig,
    OpenClawClientError,
)


class MockTransport:
    def __init__(
        self,
        response: dict | None = None,
        error: Exception | None = None,
    ):
        self.response = response or {}
        self.error = error
        self.calls = []

    def run_agent(
        self,
        *,
        gateway_url: str,
        token: str,
        agent_name: str,
        session_key: str,
        payload: dict,
        timeout_seconds: int,
    ) -> dict:
        self.calls.append(
            {
                "gateway_url": gateway_url,
                "token": token,
                "agent_name": agent_name,
                "session_key": session_key,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )

        if self.error is not None:
            raise self.error

        return self.response


def _payload() -> dict:
    return {
        "request": {
            "run_id": "waterbirds__model-a__seed42__r01"
        },
        "discovery_context": {
            "dataset_name": "waterbirds"
        },
    }


def test_invoke_agent():
    transport = MockTransport(
        response={
            "run_id": "waterbirds__model-a__seed42__r01"
        }
    )

    client = OpenClawClient(
        config=OpenClawClientConfig(
            gateway_url="ws://openclaw-gateway:18789",
            token="test-token",
            timeout_seconds=120,
        ),
        transport=transport,
    )

    result = client.invoke_agent(
        agent_name="identify",
        payload=_payload(),
    )

    assert result["run_id"] == (
        "waterbirds__model-a__seed42__r01"
    )

    assert len(transport.calls) == 1

    call = transport.calls[0]

    assert call["agent_name"] == "identify"

    assert call["session_key"] == (
        "shortcutminer:"
        "waterbirds__model-a__seed42__r01"
    )


def test_session_key_is_reproducible():
    key_a = OpenClawClient._build_session_key(
        agent_name="identify",
        run_id="run_001",
    )

    key_b = OpenClawClient._build_session_key(
        agent_name="identify",
        run_id="run_001",
    )

    assert key_a == key_b


def test_missing_request():
    transport = MockTransport()

    client = OpenClawClient(
        config=OpenClawClientConfig(
            gateway_url="ws://gateway:18789",
            token="token",
        ),
        transport=transport,
    )

    with pytest.raises(
        OpenClawClientError,
        match="request",
    ):
        client.invoke_agent(
            agent_name="identify",
            payload={},
        )


def test_missing_run_id():
    transport = MockTransport()

    client = OpenClawClient(
        config=OpenClawClientConfig(
            gateway_url="ws://gateway:18789",
            token="token",
        ),
        transport=transport,
    )

    with pytest.raises(
        OpenClawClientError,
        match="run_id",
    ):
        client.invoke_agent(
            agent_name="identify",
            payload={
                "request": {}
            },
        )


def test_transport_failure():
    transport = MockTransport(
        error=RuntimeError(
            "gateway unavailable"
        )
    )

    client = OpenClawClient(
        config=OpenClawClientConfig(
            gateway_url="ws://gateway:18789",
            token="token",
        ),
        transport=transport,
    )

    with pytest.raises(
        OpenClawClientError,
        match="gateway unavailable",
    ):
        client.invoke_agent(
            agent_name="identify",
            payload=_payload(),
        )


def test_from_env(monkeypatch):
    monkeypatch.setenv(
        "OPENCLAW_GATEWAY_URL",
        "ws://openclaw-gateway:18789",
    )

    monkeypatch.setenv(
        "OPENCLAW_GATEWAY_TOKEN",
        "secret-token",
    )

    monkeypatch.setenv(
        "OPENCLAW_TIMEOUT_SECONDS",
        "60",
    )

    transport = MockTransport()

    client = OpenClawClient.from_env(
        transport=transport
    )

    assert client.config.gateway_url == (
        "ws://openclaw-gateway:18789"
    )

    assert client.config.token == "secret-token"
    assert client.config.timeout_seconds == 60