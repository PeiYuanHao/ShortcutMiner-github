import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


class OpenClawGatewayCliTransportError(RuntimeError):
    """Raised when the official OpenClaw Gateway CLI call fails."""


class OpenClawGatewayCliTransport:
    """
    Transport implemented using the official Gateway-backed
    `openclaw agent` CLI command.

    ShortcutMiner does not implement the OpenClaw WebSocket
    protocol itself. OpenClaw owns Gateway connection,
    authentication, session routing, and protocol compatibility.
    """

    def __init__(
        self,
        runner: Callable = subprocess.run,
        executable: str = "openclaw",
    ) -> None:
        self.runner = runner
        self.executable = executable

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
        message = self._build_message(payload)

        message_path = self._write_message_file(
            message
        )

        try:
            command = [
                self.executable,
                "agent",
                "--agent",
                agent_name,
                "--session-key",
                session_key,
                "--message-file",
                str(message_path),
                "--timeout",
                str(timeout_seconds),
                "--json",
            ]

            env = os.environ.copy()

            env["OPENCLAW_GATEWAY_URL"] = (
                gateway_url
            )

            env["OPENCLAW_GATEWAY_TOKEN"] = (
                token
            )

            try:
                completed = self.runner(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    env=env,
                    timeout=timeout_seconds + 30,
                    check=False,
                )

            except subprocess.TimeoutExpired as exc:
                raise OpenClawGatewayCliTransportError(
                    "OpenClaw CLI process timed out."
                ) from exc

            if completed.returncode != 0:
                stderr = (
                    completed.stderr.strip()
                    if completed.stderr
                    else "unknown OpenClaw CLI error"
                )

                raise OpenClawGatewayCliTransportError(
                    f"OpenClaw CLI failed: {stderr}"
                )

            return self._parse_output(
                completed.stdout
            )

        finally:
            message_path.unlink(
                missing_ok=True
            )

    @staticmethod
    def _build_message(
        payload: dict,
    ) -> str:
        """
        Build the structured message sent to the configured agent.
        """

        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )

        return (
            "Execute the current ShortcutMiner Identify task "
            "according to your workspace instructions.\n\n"
            "The INPUT object has two required fields: "
            "request is the authoritative IdentifyRequest, and "
            "discovery_context is deterministic, read-only evidence "
            "prepared for this run. Selected metadata records are "
            "not inspected dataset samples. Use the context's "
            "samples_inspected value exactly.\n\n"
            "Return exactly one CandidateReport JSON object. "
            "Do not add Markdown fences or explanatory prose.\n\n"
            "INPUT:\n"
            f"{payload_json}"
        )

    @staticmethod
    def _write_message_file(
        message: str,
    ) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".md",
            delete=False,
        ) as file:
            file.write(message)

            return Path(file.name)

    @staticmethod
    def _parse_output(
        stdout: str,
    ) -> dict:
        if not stdout or not stdout.strip():
            raise OpenClawGatewayCliTransportError(
                "OpenClaw CLI returned empty output."
            )

        try:
            envelope = json.loads(stdout)

        except json.JSONDecodeError as exc:
            raise OpenClawGatewayCliTransportError(
                "OpenClaw CLI returned invalid JSON."
            ) from exc

        text = (
            OpenClawGatewayCliTransport
            ._extract_agent_text(envelope)
        )

        if text is None:
            raise OpenClawGatewayCliTransportError(
                "OpenClaw CLI response contained no agent text."
            )

        try:
            result = json.loads(
                text.strip()
            )

        except json.JSONDecodeError as exc:
            raise OpenClawGatewayCliTransportError(
                "Identify Agent did not return "
                "a valid JSON object."
            ) from exc

        if not isinstance(result, dict):
            raise OpenClawGatewayCliTransportError(
                "Identify Agent response "
                "must be a JSON object."
            )

        return result

    @staticmethod
    def _extract_agent_text(
        envelope: dict,
    ) -> str | None:
        """
        Extract visible assistant text from the official
        `openclaw agent --json` envelope.
        """

        payloads = envelope.get(
            "payloads"
        )

        if isinstance(payloads, list):
            texts = [
                item["text"]
                for item in payloads
                if isinstance(item, dict)
                and isinstance(
                    item.get("text"),
                    str,
                )
                and item["text"].strip()
            ]

            if texts:
                return "\n".join(texts)

        # Compatibility with the raw Gateway result
        # when preserved by the CLI envelope.
        result = envelope.get(
            "result"
        )

        if isinstance(result, dict):
            payloads = result.get(
                "payloads"
            )

            if isinstance(
                payloads,
                list,
            ):
                texts = [
                    item["text"]
                    for item in payloads
                    if isinstance(item, dict)
                    and isinstance(
                        item.get("text"),
                        str,
                    )
                    and item["text"].strip()
                ]

                if texts:
                    return "\n".join(texts)

        return None
