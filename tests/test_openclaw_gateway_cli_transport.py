import json
import subprocess

import pytest

from shortcutminer.openclaw_gateway_cli_transport import (
    OpenClawGatewayCliTransport,
    OpenClawGatewayCliTransportError,
)


def _candidate_report() -> dict:
    return {
        "protocol_version": "1.0",
        "run_id": "test_run",
        "source": "identify",
        "message_type": "candidate_report",
        "dataset_summary": {
            "dataset": "test_dataset",
            "samples_inspected": 4,
            "metadata_used": True,
        },
        "candidates": [],
        "limitations": [],
    }


def test_run_agent():
    report = _candidate_report()

    cli_output = {
        "payloads": [
            {
                "text": json.dumps(report)
            }
        ],
        "meta": {},
    }

    calls = []

    def fake_runner(
        command,
        **kwargs,
    ):
        calls.append(
            {
                "command": command,
                "kwargs": kwargs,
            }
        )

        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps(cli_output),
            stderr="",
        )

    transport = (
        OpenClawGatewayCliTransport(
            runner=fake_runner
        )
    )

    result = transport.run_agent(
        gateway_url=(
            "ws://openclaw-gateway:18789"
        ),
        token="test-token",
        agent_name="identify",
        session_key=(
            "shortcutminer:test_run"
        ),
        payload={
            "request": {
                "run_id": "test_run"
            }
        },
        timeout_seconds=120,
    )

    assert result == report
    assert len(calls) == 1

    command = calls[0]["command"]

    assert command[0] == "openclaw"
    assert command[1] == "agent"

    assert "--agent" in command
    assert "identify" in command

    assert "--session-key" in command
    assert "shortcutminer:test_run" in command

    assert "--message-file" in command
    assert "--json" in command

    # It must NEVER run embedded.
    assert "--local" not in command

    env = calls[0]["kwargs"]["env"]

    assert env[
        "OPENCLAW_GATEWAY_URL"
    ] == "ws://openclaw-gateway:18789"

    assert env[
        "OPENCLAW_GATEWAY_TOKEN"
    ] == "test-token"


def test_parse_payloads():
    report = _candidate_report()

    envelope = {
        "payloads": [
            {
                "text": json.dumps(report)
            }
        ]
    }

    result = (
        OpenClawGatewayCliTransport
        ._parse_output(
            json.dumps(envelope)
        )
    )

    assert result == report


def test_executive_message_uses_routing_contract():
    message = OpenClawGatewayCliTransport._build_message(
        {"request": {"run_id": "test_run"}},
        agent_name="executive",
    )

    assert "Executive routing task" in message
    assert "RoutingPlan" in message
    assert "do not claim that tasks were dispatched" in message


def test_parse_nested_gateway_payload():
    report = _candidate_report()

    envelope = {
        "result": {
            "payloads": [
                {
                    "text": json.dumps(
                        report
                    )
                }
            ]
        }
    }

    result = (
        OpenClawGatewayCliTransport
        ._parse_output(
            json.dumps(envelope)
        )
    )

    assert result == report


def test_empty_output():
    with pytest.raises(
        OpenClawGatewayCliTransportError,
        match="empty",
    ):
        (
            OpenClawGatewayCliTransport
            ._parse_output("")
        )


def test_invalid_cli_json():
    with pytest.raises(
        OpenClawGatewayCliTransportError,
        match="invalid JSON",
    ):
        (
            OpenClawGatewayCliTransport
            ._parse_output(
                "not-json"
            )
        )


def test_invalid_agent_json():
    envelope = {
        "payloads": [
            {
                "text": "not-json"
            }
        ]
    }

    with pytest.raises(
        OpenClawGatewayCliTransportError,
        match="valid JSON object",
    ):
        (
            OpenClawGatewayCliTransport
            ._parse_output(
                json.dumps(envelope)
            )
        )


def test_cli_failure():
    def fake_runner(
        command,
        **kwargs,
    ):
        return subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr="gateway unavailable",
        )

    transport = (
        OpenClawGatewayCliTransport(
            runner=fake_runner
        )
    )

    with pytest.raises(
        OpenClawGatewayCliTransportError,
        match="gateway unavailable",
    ):
        transport.run_agent(
            gateway_url="ws://gateway:18789",
            token="token",
            agent_name="identify",
            session_key="shortcutminer:test",
            payload={
                "request": {
                    "run_id": "test"
                }
            },
            timeout_seconds=120,
        )
