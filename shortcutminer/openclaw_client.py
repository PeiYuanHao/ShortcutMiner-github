import os
from dataclasses import dataclass
from typing import Protocol


class OpenClawClientError(RuntimeError):
    """Raised when communication with OpenClaw fails."""


class OpenClawTransport(Protocol):
    """
    Transport contract used by OpenClawClient.

    A real cloud Gateway transport will implement
    this interface later.
    """

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
        ...


@dataclass(frozen=True)
class OpenClawClientConfig:
    gateway_url: str
    token: str
    timeout_seconds: int = 120


class OpenClawClient:
    """
    ShortcutMiner interface to the OpenClaw runtime.

    Network and Gateway protocol details are delegated
    to OpenClawTransport.
    """

    def __init__(
        self,
        config: OpenClawClientConfig,
        transport: OpenClawTransport,
    ) -> None:
        self.config = config
        self.transport = transport

    @classmethod
    def from_env(
        cls,
        transport: OpenClawTransport,
    ) -> "OpenClawClient":
        gateway_url = os.getenv(
            "OPENCLAW_GATEWAY_URL",
            "",
        ).strip()

        token = (
            os.getenv("OPENCLAW_GATEWAY_TOKEN")
            or os.getenv("OPENCLAW_TOKEN")
            or ""
        ).strip()

        timeout_raw = os.getenv(
            "OPENCLAW_TIMEOUT_SECONDS",
            "120",
        )

        if not gateway_url:
            raise OpenClawClientError(
                "OPENCLAW_GATEWAY_URL is not configured."
            )

        if not token:
            raise OpenClawClientError(
                "OpenClaw Gateway token is not configured."
            )

        try:
            timeout_seconds = int(timeout_raw)
        except ValueError as exc:
            raise OpenClawClientError(
                "OPENCLAW_TIMEOUT_SECONDS must be an integer."
            ) from exc

        if timeout_seconds < 1:
            raise OpenClawClientError(
                "OPENCLAW_TIMEOUT_SECONDS must be positive."
            )

        config = OpenClawClientConfig(
            gateway_url=gateway_url,
            token=token,
            timeout_seconds=timeout_seconds,
        )

        return cls(
            config=config,
            transport=transport,
        )

    def invoke_agent(
        self,
        agent_name: str,
        payload: dict,
    ) -> dict:
        request = payload.get("request")

        if not isinstance(request, dict):
            raise OpenClawClientError(
                "Agent payload does not contain a valid request."
            )

        run_id = request.get("run_id")

        if not isinstance(run_id, str) or not run_id.strip():
            raise OpenClawClientError(
                "Agent payload does not contain a valid run_id."
            )

        session_key = self._build_session_key(
            agent_name=agent_name,
            run_id=run_id,
        )

        try:
            result = self.transport.run_agent(
                gateway_url=self.config.gateway_url,
                token=self.config.token,
                agent_name=agent_name,
                session_key=session_key,
                payload=payload,
                timeout_seconds=self.config.timeout_seconds,
            )

        except Exception as exc:
            raise OpenClawClientError(
                f"OpenClaw agent execution failed: {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise OpenClawClientError(
                "OpenClaw agent returned a non-object response."
            )

        return result

    @staticmethod
    def _build_session_key(
        agent_name: str,
        run_id: str,
    ) -> str:
        return f"shortcutminer:{run_id}"