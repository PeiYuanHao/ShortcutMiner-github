"""Hardened provider-neutral client for OpenAI-compatible chat APIs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import ValidationError

from shortcutminer.protocol.model import OpenAICompatibleModelConfig
from shortcutminer.protocol.model_runtime import (
    ModelEndpointIdentity,
    ModelFailure,
    ModelImageInput,
    ModelInvocationRecord,
    ModelResponseDiagnostics,
    ModelRuntimeSnapshot,
    ModelSchemaDiagnostics,
    ModelSchemaIssue,
    ModelUsage,
)
from shortcutminer.storage.events import append_event
from shortcutminer.storage.model_calls import save_model_invocation, save_runtime_snapshot


class ModelRuntimeError(RuntimeError):
    """Raised when a configured model runtime cannot return structured output."""

    def __init__(self, message: str, *, failure: ModelFailure | None = None) -> None:
        super().__init__(message)
        self.failure = failure


class ProviderResponseJSONError(RuntimeError):
    """The HTTP body was not a valid provider JSON envelope."""


class JsonHttpTransport(Protocol):
    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout: float,
    ) -> object: ...


class UrllibJsonTransport:
    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        timeout: float,
    ) -> object:
        request = Request(
            url,
            data=json.dumps(body, allow_nan=False).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderResponseJSONError(
                "Model endpoint returned a non-JSON response envelope."
            ) from exc


class UrllibStreamingJsonTransport:
    """Assemble SSE deltas without accepting an interrupted partial response."""

    def post_json(self, url, *, headers, body, timeout):
        request = Request(url, data=json.dumps(body, allow_nan=False).encode('utf-8'),
                          headers=dict(headers), method='POST')
        choices = {}
        envelope = {}
        done = False
        with urlopen(request, timeout=timeout) as response:
            if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                return json.loads(response.read())
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if data == b'[DONE]':
                    done = True
                    break
                chunk = json.loads(data)
                if chunk.get('error'):
                    raise ProviderResponseJSONError('Streaming endpoint returned an error envelope')
                for key in ('id', 'model', 'created', 'system_fingerprint', 'usage'):
                    if chunk.get(key) is not None:
                        envelope[key] = chunk[key]
                for choice in chunk.get('choices', []):
                    index = choice['index']
                    item = choices.setdefault(index, {'index': index, 'message': {'role': 'assistant', 'content': ''}, 'finish_reason': None})
                    delta = choice.get('delta', {})
                    for field in ('content', 'reasoning_content', 'refusal'):
                        if delta.get(field) is not None:
                            item['message'][field] = item['message'].get(field, '') + delta[field]
                    if delta.get('tool_calls'):
                        raise ProviderResponseJSONError('Unexpected tool calls in structured streaming response')
                    if choice.get('finish_reason') is not None:
                        item['finish_reason'] = choice['finish_reason']
        if not done or not choices or any(item['finish_reason'] is None for item in choices.values()):
            raise ProviderResponseJSONError('Streaming response ended before completion')
        return {**envelope, 'object': 'chat.completion', 'choices': [choices[i] for i in sorted(choices)]}


class ModelBudgetState:
    """Run-scoped counters shared by clients with per-Agent configurations."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.calls_total = 0
        self.calls_by_agent: dict[str, int] = {}
        self.known_total_tokens = 0
        self.unknown_usage_seen = False


class PilotInvocationBudgetState:
    """Shared live pilot limits, including retries and submitted images."""

    def __init__(self, *, max_text_calls: int, max_vlm_calls: int,
                 max_vlm_images: int, max_text_per_run: int,
                 max_vlm_per_run: int, max_vlm_images_per_run: int,
                 max_tokens: int | None = None) -> None:
        self.lock = Lock()
        self.max_text_calls = max_text_calls
        self.max_vlm_calls = max_vlm_calls
        self.max_vlm_images = max_vlm_images
        self.max_text_per_run = max_text_per_run
        self.max_vlm_per_run = max_vlm_per_run
        self.max_vlm_images_per_run = max_vlm_images_per_run
        self.max_tokens = max_tokens
        self.text_calls = 0
        self.vlm_calls = 0
        self.vlm_images = 0
        self.by_run: dict[str, dict[str, int]] = {}
        self.known_tokens = 0
        self.usage_incomplete = False

    def reserve(self, run_id: str, runtime_kind: str, images: int,
                *, restore: bool = False) -> bool:
        with self.lock:
            counts = self.by_run.setdefault(run_id, {"text": 0, "vlm": 0, "images": 0})
            if not restore and self.max_tokens is not None and (
                self.usage_incomplete or self.known_tokens >= self.max_tokens
            ):
                return False
            if runtime_kind == "semantic_vlm":
                if (self.vlm_calls >= self.max_vlm_calls
                    or counts["vlm"] >= self.max_vlm_per_run
                    or self.vlm_images + images > self.max_vlm_images
                    or counts["images"] + images > self.max_vlm_images_per_run):
                    return False
                self.vlm_calls += 1
                self.vlm_images += images
                counts["vlm"] += 1
                counts["images"] += images
            else:
                if (self.text_calls >= self.max_text_calls
                    or counts["text"] >= self.max_text_per_run):
                    return False
                self.text_calls += 1
                counts["text"] += 1
            return True

    def observe_usage(self, usage: ModelUsage) -> None:
        with self.lock:
            if usage.available:
                self.known_tokens += usage.total_tokens or 0
            else:
                self.usage_incomplete = True


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def text_hash(value: str | None) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def build_runtime_snapshot(
    config: OpenAICompatibleModelConfig,
) -> ModelRuntimeSnapshot:
    parsed = urlsplit(config.base_url)
    endpoint = ModelEndpointIdentity(
        scheme=parsed.scheme,
        host=parsed.netloc,
        path=parsed.path or "/",
    )
    payload = {
        "provider": config.provider,
        "model": config.model,
        "endpoint": endpoint.model_dump(mode="json"),
        "api_key_env": config.api_key_env,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout_seconds": config.timeout,
        "retry_policy": config.retry_policy.model_dump(mode="json"),
        "model_call_budget": config.model_call_budget.model_dump(mode="json"),
        "model_seed": (
            config.model_seed if config.capabilities.supports_model_seed else None
        ),
        "capabilities": config.capabilities.model_dump(mode="json"),
        "reasoning": (
            config.reasoning.model_dump(mode="json")
            if config.reasoning is not None else None
        ),
        "structured_output_mode": (
            "json_object" if config.capabilities.supports_json_object else "none"
        ),
        "raw_response_persistence": False,
    }
    return ModelRuntimeSnapshot(
        snapshot_id=f"model-runtime-{canonical_json_hash(payload)[:20]}",
        **payload,
    )


def _usage(response: object) -> ModelUsage:
    if not isinstance(response, dict) or not isinstance(response.get("usage"), dict):
        return ModelUsage(available=False)
    usage = response["usage"]
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (prompt, completion, total)
    )
    if not valid:
        return ModelUsage(available=False)
    prompt_details = usage.get("prompt_tokens_details", {})
    completion_details = usage.get("completion_tokens_details", {})
    cached = prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None
    reasoning = completion_details.get("reasoning_tokens") if isinstance(completion_details, dict) else None
    return ModelUsage(
        available=True,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=cached if isinstance(cached, int) and cached >= 0 else None,
        reasoning_tokens=(
            reasoning if isinstance(reasoning, int) and reasoning >= 0 else None
        ),
    )


def _response_diagnostics(response: object) -> ModelResponseDiagnostics:
    choice: object = None
    message: object = None
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
    if isinstance(choice, dict):
        message = choice.get("message")
        raw_finish_reason = choice.get("finish_reason")
    else:
        raw_finish_reason = None
    finish_reason = (
        raw_finish_reason
        if raw_finish_reason in {"stop", "length", "content_filter", "tool_calls"}
        else "missing" if raw_finish_reason is None else "other"
    )
    if not isinstance(message, dict) or "content" not in message:
        return ModelResponseDiagnostics(
            finish_reason=finish_reason,
            content_present=False,
            content_length=0,
            content_whitespace_only=False,
            content_first_token_class="empty",
            content_last_token_class="empty",
            json_parse_status="content_missing",
        )
    content = message.get("content")
    if not isinstance(content, str):
        return ModelResponseDiagnostics(
            finish_reason=finish_reason,
            content_present=False,
            content_length=0,
            content_whitespace_only=False,
            content_first_token_class="empty",
            content_last_token_class="empty",
            json_parse_status="content_not_string",
        )
    stripped = content.strip()
    if not stripped:
        first_class = last_class = "empty"
    else:
        first_class = (
            "markdown_fence" if stripped.startswith("```")
            else "object_open" if stripped.startswith("{")
            else "array_open" if stripped.startswith("[")
            else "other"
        )
        last_class = (
            "markdown_fence" if stripped.endswith("```")
            else "object_close" if stripped.endswith("}")
            else "array_close" if stripped.endswith("]")
            else "other"
        )
    try:
        parsed = json.loads(content)
        parse_status = "valid_object" if isinstance(parsed, dict) else "valid_non_object"
    except json.JSONDecodeError:
        parse_status = "invalid_json"
    return ModelResponseDiagnostics(
        finish_reason=finish_reason,
        content_present=bool(content),
        content_length=len(content),
        content_whitespace_only=bool(content) and not bool(stripped),
        content_first_token_class=first_class,
        content_last_token_class=last_class,
        json_parse_status=parse_status,
    )


_SCHEMA_CORRECTION = (
    "The previous model response could not be accepted because it was not valid "
    "JSON or did not satisfy the required protocol schema. Return exactly one JSON "
    "object matching the required output contract. Do not include Markdown fences, "
    "commentary, or explanatory text. Preserve the original task and evidence; do "
    "not change scientific conclusions merely to satisfy validation."
)

_SCHEMA_GUIDANCE = {
    "missing": "required field is missing",
    "extra_forbidden": "remove the extra field",
    "literal_error": "use one of the exact allowed schema literals",
    "int_type": "field must be an integer",
    "string_type": "field must be a string",
    "list_type": "field must be a JSON array",
    "dict_type": "field must be a JSON object",
    "string_pattern_mismatch": "follow the exact identifier pattern defined by the protocol",
    "too_short": "satisfy the protocol minimum length",
}


def _response_kind(
    output: dict[str, Any] | None,
    validator: Callable[[dict[str, Any]], None] | None,
) -> str:
    if output is not None and "tool_calls" in output:
        return "tool_call"
    if output is not None and validator is not None:
        return "final_result"
    return "unknown"


def _safe_schema_diagnostics(exc: ValidationError) -> ModelSchemaDiagnostics:
    """Whitelist value-free Pydantic metadata and bound persisted issue volume."""
    raw_errors = exc.errors(include_input=False, include_context=False)
    issues: list[ModelSchemaIssue] = []
    for error in raw_errors[:8]:
        error_type = str(error.get("type", "validation_error"))
        if re.fullmatch(r"[a-z0-9_]{1,64}", error_type) is None:
            error_type = "validation_error"
        location = list(error.get("loc", ()))
        safe_parts: list[str] = []
        for index, part in enumerate(location):
            if isinstance(part, int) and part >= 0:
                safe_parts.append(str(part))
            elif isinstance(part, str):
                if error_type == "extra_forbidden" and index == len(location) - 1:
                    safe_parts.append("<extra_field>")
                elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", part):
                    safe_parts.append(part)
                else:
                    safe_parts.append("<field>")
            else:
                safe_parts.append("<field>")
        issues.append(ModelSchemaIssue(
            field_path=".".join(safe_parts) or "<root>",
            error_type=error_type,
        ))
    return ModelSchemaDiagnostics(
        total_schema_issue_count=len(raw_errors),
        issues=issues,
    )


_SAFE_CONTRACT_RULES = frozenset({
    "candidate_budget_exceeded", "duplicate_metadata_field_selector",
    "metadata_unavailable", "sample_inspection_unavailable", "image_statistics_unavailable",
    "invalid_metadata_field_selector", "metadata_selector_without_source",
    "executive_context_invalid", "quantitative_context_invalid",
    "candidate_coverage_mismatch", "assessment_overclaim",
    "unknown_sample_reference", "unavailable_sample_reference", "unauthorized_artifact_reference",
    "comparison_single_group", "unknown_observation_index", "unknown_comparison_index",
    "unauthorized_metadata_field",
})


def _safe_contract_diagnostics(exc: Exception) -> ModelSchemaDiagnostics | None:
    """Record typed rule metadata, never exception messages or rejected values."""
    # Lazy import avoids a runtime/tool-loop import cycle during initialization.
    from shortcutminer.tooling.agent_loop import ProviderContractValidationError
    if isinstance(exc, ProviderContractValidationError):
        rule = (exc.category if isinstance(exc.category, str) and exc.category in _SAFE_CONTRACT_RULES
                else "provider_contract_rejected")
        path = exc.field_path or "<root>"
        if not isinstance(path, str) or re.fullmatch(
            r"(?:candidates|findings)(?:\.[0-9]{1,6})?(?:\.(?:metadata_field_ref|evidence_sources|assessment))?",
            path,
        ) is None:
            path = "<root>" if not exc.field_path else "<field>"
    elif isinstance(exc, TypeError):
        rule, path = "validator_type_error", "<root>"
    elif isinstance(exc, ValueError):
        rule, path = "validator_value_error", "<root>"
    else:
        return None
    return ModelSchemaDiagnostics(total_schema_issue_count=1,
                                  issues=[ModelSchemaIssue(field_path=path, error_type=rule)])


def _schema_correction(diagnostics: ModelSchemaDiagnostics | None) -> str:
    if diagnostics is None or not diagnostics.issues:
        return _SCHEMA_CORRECTION
    lines = [
        "The previous JSON was syntactically valid but failed the required protocol schema.",
        "Schema issues:",
    ]
    for issue in diagnostics.issues:
        guidance = _SCHEMA_GUIDANCE.get(issue.error_type)
        suffix = f" ({guidance})" if guidance else ""
        lines.append(f"- {issue.field_path}: {issue.error_type}{suffix}")
    lines.extend([
        "Correct only protocol conformance.",
        "Do not change the scientific interpretation merely to satisfy validation.",
        "Preserve the original evidence, observations, candidate, hypothesis, sample identities and conclusions.",
        "Return exactly one valid JSON object.",
    ])
    return "\n".join(lines)


def _agent_schema_correction(agent_name: str, diagnostics: ModelSchemaDiagnostics | None) -> str:
    correction = _schema_correction(diagnostics)
    if agent_name == "qualitative":
        correction += ("\nEvery counterevidence and metadata_evidence item must cite at least one actual "
            "selected sample ID. General caveats without a sample belong in additional_limitations. "
            "Preserve genuine counterevidence. Do not invent sample references or change scientific "
            "interpretation to satisfy the protocol. Comparisons require different authoritative groups.")
    if agent_name == "identify":
        from shortcutminer.runtime.identify_output_contract import identify_output_contract
        correction += (
            "\n\nCorrect only protocol conformance.\n"
            "Do not change the scientific interpretation merely to satisfy validation.\n"
            "Follow the authoritative Identify contract in the original system message and repeated below.\n"
            + identify_output_contract()
        )
    return correction


def _http_failure(status: int) -> ModelFailure:
    if status == 401:
        category, retryable = "authentication_failed", False
    elif status == 403:
        category, retryable = "authorization_failed", False
    elif status == 429:
        category, retryable = "rate_limited", True
    elif status == 408:
        category, retryable = "timed_out", True
    elif 500 <= status <= 599:
        category, retryable = "provider_5xx", True
    else:
        category, retryable = "provider_rejected", False
    return ModelFailure(
        category=category,
        message=f"Model endpoint returned HTTP {status}.",
        http_status=status,
        retryable=retryable,
    )


def _exception_failure(exc: Exception) -> ModelFailure:
    if isinstance(exc, ModelRuntimeError) and exc.failure is not None:
        return exc.failure
    if isinstance(exc, HTTPError):
        return _http_failure(exc.code)
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return ModelFailure(
            category="timed_out",
            message="Model endpoint request timed out.",
            retryable=True,
        )
    if isinstance(exc, ProviderResponseJSONError):
        return ModelFailure(
            category="invalid_response_envelope",
            message="Model endpoint returned an invalid JSON envelope.",
            retryable=False,
        )
    if isinstance(exc, (URLError, ConnectionError, OSError)):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return ModelFailure(
                category="timed_out",
                message="Model endpoint request timed out.",
                retryable=True,
            )
        return ModelFailure(
            category="transport_failed",
            message="Model endpoint transport failed.",
            retryable=False,
        )
    return ModelFailure(
        category="transport_failed",
        message=f"Model endpoint transport failed: {type(exc).__name__}.",
        retryable=False,
    )


class OpenAICompatibleClient:
    """Invoke configured text or multimodal OpenAI-compatible endpoints."""

    runtime_kind = "real_provider"

    def __init__(
        self,
        config: OpenAICompatibleModelConfig,
        *,
        api_key: str | None = None,
        transport: JsonHttpTransport | None = None,
        runs_root: str | Path | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
        budget_state: ModelBudgetState | None = None,
    ) -> None:
        self.config = config
        self._explicit_api_key = api_key
        self._transport = transport or UrllibJsonTransport()
        self._runs_root = Path(runs_root) if runs_root is not None else None
        self._sleeper = sleeper
        self._monotonic = monotonic
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self._lock = Lock()
        self._budget_state = budget_state or ModelBudgetState()
        self._pilot_budget_state: PilotInvocationBudgetState | None = None
        self.invocations: list[ModelInvocationRecord] = []
        self.runtime_snapshot = build_runtime_snapshot(config)

    def _api_key(self) -> str:
        key = self._explicit_api_key
        if key is None:
            key = os.environ.get(self.config.api_key_env)
        if not key or not key.strip():
            failure = ModelFailure(
                category="authentication_failed",
                message=(
                    "Model credential environment variable is not set: "
                    f"{self.config.api_key_env}"
                ),
                retryable=False,
            )
            raise ModelRuntimeError(failure.message, failure=failure)
        return key.strip()

    @staticmethod
    def _run_id(payload: dict[str, Any]) -> str:
        request = payload.get("request")
        value = (
            request.get("run_id")
            if isinstance(request, dict)
            else payload.get("run_id")
        )
        return value if isinstance(value, str) and value else "untracked-model-run"

    def _reserve_call(self, agent_name: str, *, run_id: str,
                      runtime_kind: str, image_count: int) -> ModelFailure | None:
        state = self._budget_state
        with state.lock:
            budget = self.config.model_call_budget
            used_by_agent = state.calls_by_agent.get(agent_name, 0)
            if (
                state.calls_total >= budget.max_total_calls
                or used_by_agent >= budget.max_calls_per_agent
            ):
                return ModelFailure(
                    category="budget_exceeded",
                    message="Model call budget was exceeded before provider invocation.",
                    retryable=False,
                )
            if (
                budget.max_total_tokens is not None
                and (state.unknown_usage_seen
                     or state.known_total_tokens >= budget.max_total_tokens)
            ):
                return ModelFailure(
                    category="budget_exceeded",
                    message="Known model token budget was exhausted.",
                    retryable=False,
                )
            if self._pilot_budget_state is not None and not self._pilot_budget_state.reserve(
                run_id, runtime_kind, image_count
            ):
                return ModelFailure(
                    category="budget_exceeded",
                    message="Pilot live model budget was exhausted before provider invocation.",
                    retryable=False,
                )
            state.calls_total += 1
            state.calls_by_agent[agent_name] = used_by_agent + 1
            return None

    def _backoff(self, retry_index: int) -> float:
        policy = self.config.retry_policy
        return min(
            policy.max_backoff_seconds,
            policy.initial_backoff_seconds
            * (policy.backoff_multiplier ** max(0, retry_index - 1)),
        )

    def _transport_retry_allowed(
        self, failure: ModelFailure, retry_count: int
    ) -> bool:
        policy = self.config.retry_policy
        budget = self.config.model_call_budget
        if retry_count >= min(
            policy.max_attempts - 1, budget.max_transport_retries
        ):
            return False
        return bool(
            (failure.category == "rate_limited" and policy.retry_on_rate_limit)
            or (failure.category == "timed_out" and policy.retry_on_timeout)
            or (failure.category == "provider_5xx" and policy.retry_on_5xx)
        )

    def _record(
        self,
        *,
        run_id: str,
        agent_name: str,
        runtime_kind: str,
        attempt: int,
        payload_hash: str,
        system_prompt_hash: str,
        request_hash: str,
        response_hash: str | None,
        started_at: str,
        latency_ms: float,
        status: str,
        failure: ModelFailure | None,
        usage: ModelUsage,
        schema_valid: bool | None,
        transport_retry: bool,
        schema_retry: bool,
        response_diagnostics: ModelResponseDiagnostics | None = None,
        response_kind: str = "unknown",
        schema_diagnostics: ModelSchemaDiagnostics | None = None,
        limitations: list[str] | None = None,
        image_inputs: list[ModelImageInput] | None = None,
    ) -> ModelInvocationRecord:
        invocation_id = f"{run_id}-{agent_name}-{uuid.uuid4().hex[:16]}"
        record = ModelInvocationRecord(
            invocation_id=invocation_id,
            run_id=run_id,
            agent_name=agent_name,
            runtime_kind=runtime_kind,
            provider=self.config.provider,
            model=self.config.model,
            attempt=attempt,
            request_hash=request_hash,
            system_prompt_hash=system_prompt_hash,
            payload_hash=payload_hash,
            response_hash=response_hash,
            started_at=started_at,
            completed_at=self._wall_clock().isoformat(),
            latency_ms=round(max(0.0, latency_ms), 6),
            status=status,
            failure_category=failure.category if failure else None,
            http_status=failure.http_status if failure else None,
            retryable=failure.retryable if failure else False,
            usage=usage,
            schema_valid=schema_valid,
            transport_retry=transport_retry,
            schema_retry=schema_retry,
            structured_output_mode=(
                "json_object"
                if self.config.capabilities.supports_json_object else "none"
            ),
            response_kind=response_kind,
            response_diagnostics=response_diagnostics,
            schema_diagnostics=schema_diagnostics,
            image_inputs=image_inputs or [],
            limitations=limitations or [],
        )
        with self._lock:
            self.invocations.append(record)
        state = self._budget_state
        if status != "budget_exceeded":
            with state.lock:
                if usage.available:
                    state.known_total_tokens += usage.total_tokens or 0
                else:
                    state.unknown_usage_seen = True
        if self._pilot_budget_state is not None and status != "budget_exceeded":
            self._pilot_budget_state.observe_usage(usage)
        if self._runs_root is not None:
            save_runtime_snapshot(self._runs_root, run_id, self.runtime_snapshot)
            record = save_model_invocation(
                self._runs_root,
                record,
                runtime_snapshot_ids=[self.runtime_snapshot.snapshot_id],
            )
            with self._lock:
                for index, existing in enumerate(self.invocations):
                    if existing.invocation_id == record.invocation_id:
                        self.invocations[index] = record
                        break
            event = (
                "MODEL_INVOCATION_COMPLETED"
                if status == "completed"
                else "MODEL_BUDGET_EXCEEDED"
                if status == "budget_exceeded"
                else "MODEL_INVOCATION_FAILED"
            )
            append_event(
                self._runs_root,
                run_id,
                event,
                stage=agent_name,
                details={
                    "invocation_id": record.invocation_id,
                    "status": record.status,
                    "retryable": record.retryable,
                    **({"schema_diagnostics": schema_diagnostics.model_dump(mode="json")}
                       if schema_diagnostics is not None else {}),
                },
            )
        return record

    @staticmethod
    def _parse_response(response: object) -> dict[str, Any]:
        try:
            content = response["choices"][0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            failure = ModelFailure(
                category="invalid_response_envelope",
                message="Model response envelope is missing message content.",
                retryable=False,
            )
            raise ModelRuntimeError(failure.message, failure=failure) from exc
        if not isinstance(content, str):
            failure = ModelFailure(
                category="invalid_response_envelope",
                message="Model response content must be a JSON string.",
                retryable=False,
            )
            raise ModelRuntimeError(failure.message, failure=failure)
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            failure = ModelFailure(
                category="invalid_json",
                message="Model response content is not valid JSON.",
                retryable=False,
            )
            raise ModelRuntimeError(failure.message, failure=failure) from exc
        if not isinstance(result, dict):
            failure = ModelFailure(
                category="invalid_json",
                message="Model response content must decode to an object.",
                retryable=False,
            )
            raise ModelRuntimeError(failure.message, failure=failure)
        return result

    def _invoke_structured(
        self,
        *,
        agent_name: str,
        runtime_kind: str,
        run_id: str,
        payload_for_hash: dict[str, Any],
        system_prompt: str | None,
        messages: list[dict[str, Any]],
        validator: Callable[[dict[str, Any]], None] | None,
        image_inputs: list[ModelImageInput] | None = None,
    ) -> dict[str, Any]:
        if not agent_name.strip():
            raise ModelRuntimeError("agent_name must not be empty.")
        payload_digest = canonical_json_hash(payload_for_hash)
        prompt_digest = text_hash(system_prompt)
        transport_retries = 0
        schema_retries = 0
        previous_schema_diagnostics: ModelSchemaDiagnostics | None = None
        attempt = 0
        while True:
            attempt += 1
            attempt_messages = list(messages)
            if schema_retries:
                attempt_messages.append({
                    "role": "user",
                    "content": _agent_schema_correction(agent_name, previous_schema_diagnostics),
                })
            body: dict[str, Any] = {
                "model": self.config.model,
                "messages": attempt_messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            if isinstance(self._transport, UrllibStreamingJsonTransport):
                body.update(stream=True, stream_options={"include_usage": True})
            if self.config.capabilities.supports_json_object:
                body["response_format"] = {"type": "json_object"}
            if self.config.reasoning is not None:
                if self.config.reasoning.wire_format == "enable_thinking":
                    body["enable_thinking"] = self.config.reasoning.mode == "enabled"
                else:
                    body["thinking"] = {"type": self.config.reasoning.mode}
                if self.config.reasoning.effort is not None:
                    body["reasoning_effort"] = self.config.reasoning.effort
            if (
                self.config.model_seed is not None
                and self.config.capabilities.supports_model_seed
            ):
                body["seed"] = self.config.model_seed
            request_digest = canonical_json_hash(body)
            budget_failure = self._reserve_call(
                agent_name,
                run_id=run_id,
                runtime_kind=runtime_kind,
                image_count=len(image_inputs or []),
            )
            started = self._wall_clock().isoformat()
            start_tick = self._monotonic()
            if budget_failure is not None:
                self._record(
                    run_id=run_id,
                    agent_name=agent_name,
                    runtime_kind=runtime_kind,
                    attempt=attempt,
                    payload_hash=payload_digest,
                    system_prompt_hash=prompt_digest,
                    request_hash=request_digest,
                    response_hash=None,
                    started_at=started,
                    latency_ms=(self._monotonic() - start_tick) * 1000,
                    status=budget_failure.category,
                    failure=budget_failure,
                    usage=ModelUsage(available=False),
                    schema_valid=None,
                    transport_retry=transport_retries > 0,
                    schema_retry=schema_retries > 0,
                    image_inputs=image_inputs,
                )
                raise ModelRuntimeError(budget_failure.message, failure=budget_failure)
            if self._runs_root is not None:
                append_event(
                    self._runs_root,
                    run_id,
                    "MODEL_INVOCATION_STARTED",
                    stage=agent_name,
                    details={"attempt": attempt, "runtime_kind": runtime_kind},
                )
            try:
                response = self._transport.post_json(
                    f"{self.config.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key()}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": "ShortcutMiner/0.1",
                    },
                    body=body,
                    timeout=self.config.timeout,
                )
            except Exception as exc:
                failure = _exception_failure(exc)
                can_retry = self._transport_retry_allowed(failure, transport_retries)
                self._record(
                    run_id=run_id,
                    agent_name=agent_name,
                    runtime_kind=runtime_kind,
                    attempt=attempt,
                    payload_hash=payload_digest,
                    system_prompt_hash=prompt_digest,
                    request_hash=request_digest,
                    response_hash=None,
                    started_at=started,
                    latency_ms=(self._monotonic() - start_tick) * 1000,
                    status=failure.category,
                    failure=failure.model_copy(update={"retryable": can_retry}),
                    usage=ModelUsage(available=False),
                    schema_valid=None,
                    transport_retry=transport_retries > 0,
                    schema_retry=schema_retries > 0,
                    image_inputs=image_inputs,
                )
                if can_retry:
                    transport_retries += 1
                    if self._runs_root is not None:
                        append_event(
                            self._runs_root,
                            run_id,
                            "MODEL_RETRY_SCHEDULED",
                            stage=agent_name,
                            details={
                                "retry_kind": "transport",
                                "retry_index": transport_retries,
                                "failure_category": failure.category,
                            },
                        )
                    self._sleeper(self._backoff(transport_retries))
                    continue
                raise ModelRuntimeError(failure.message, failure=failure) from exc

            usage = _usage(response)
            response_digest = canonical_json_hash(response)
            response_diagnostics = _response_diagnostics(response)
            result: dict[str, Any] | None = None
            kind = "unknown"
            try:
                result = self._parse_response(response)
                kind = _response_kind(result, validator)
                if validator is not None:
                    validator(result)
            except (ModelRuntimeError, ValidationError, ValueError, TypeError) as exc:
                pydantic_diagnostics = (
                    _safe_schema_diagnostics(exc)
                    if isinstance(exc, ValidationError) else None
                )
                schema_diagnostics = (pydantic_diagnostics if pydantic_diagnostics is not None
                                      else _safe_contract_diagnostics(exc))
                if isinstance(exc, ModelRuntimeError) and exc.failure is not None:
                    failure = exc.failure
                else:
                    message = "Model JSON did not satisfy the required protocol schema."
                    if pydantic_diagnostics is None and schema_diagnostics is not None:
                        issue = schema_diagnostics.issues[0]
                        message += f" Rule: {issue.error_type}; field: {issue.field_path}."
                    failure = ModelFailure(
                        category="protocol_schema_failed",
                        message=message,
                        retryable=False,
                    )
                can_schema_retry = (
                    schema_retries
                    < self.config.model_call_budget.max_schema_retries
                )
                recorded_failure = failure.model_copy(
                    update={"retryable": can_schema_retry}
                )
                self._record(
                    run_id=run_id,
                    agent_name=agent_name,
                    runtime_kind=runtime_kind,
                    attempt=attempt,
                    payload_hash=payload_digest,
                    system_prompt_hash=prompt_digest,
                    request_hash=request_digest,
                    response_hash=response_digest,
                    started_at=started,
                    latency_ms=(self._monotonic() - start_tick) * 1000,
                    status=failure.category,
                    failure=recorded_failure,
                    usage=usage,
                    schema_valid=False,
                    transport_retry=transport_retries > 0,
                    schema_retry=schema_retries > 0,
                    response_diagnostics=response_diagnostics,
                    response_kind=kind,
                    schema_diagnostics=schema_diagnostics,
                    image_inputs=image_inputs,
                )
                if can_schema_retry:
                    schema_retries += 1
                    # New contract diagnostics are observability only: preserve the
                    # existing retry prompt, budgets, and scientific validation.
                    previous_schema_diagnostics = schema_diagnostics if agent_name == "identify" else pydantic_diagnostics
                    if self._runs_root is not None:
                        append_event(
                            self._runs_root,
                            run_id,
                            "MODEL_RETRY_SCHEDULED",
                            stage=agent_name,
                            details={
                                "retry_kind": "schema",
                                "retry_index": schema_retries,
                                "failure_category": failure.category,
                            },
                        )
                    continue
                raise ModelRuntimeError(failure.message, failure=failure) from exc

            limitations = []
            if not usage.available:
                limitations.append(
                    "Provider token usage was unavailable; token totals were not guessed and a configured token budget cannot be fully enforced."
                )
            self._record(
                run_id=run_id,
                agent_name=agent_name,
                runtime_kind=runtime_kind,
                attempt=attempt,
                payload_hash=payload_digest,
                system_prompt_hash=prompt_digest,
                request_hash=request_digest,
                response_hash=response_digest,
                started_at=started,
                latency_ms=(self._monotonic() - start_tick) * 1000,
                status="completed",
                failure=None,
                usage=usage,
                schema_valid=True,
                transport_retry=transport_retries > 0,
                schema_retry=schema_retries > 0,
                response_diagnostics=response_diagnostics,
                response_kind=kind,
                limitations=limitations,
                image_inputs=image_inputs,
            )
            return result

    def invoke_agent(
        self,
        agent_name: str,
        payload: dict[str, Any],
        *,
        system_prompt: str | None = None,
    ) -> object:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
        )
        return self._invoke_structured(
            agent_name=agent_name,
            runtime_kind="text_agent",
            run_id=self._run_id(payload),
            payload_for_hash=payload,
            system_prompt=system_prompt,
            messages=messages,
            validator=None,
        )

    def invoke_agent_validated(
        self,
        agent_name: str,
        payload: dict[str, Any],
        *,
        system_prompt: str | None,
        validator: Callable[[dict[str, Any]], None],
    ) -> object:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
        )
        return self._invoke_structured(
            agent_name=agent_name,
            runtime_kind="text_agent",
            run_id=self._run_id(payload),
            payload_for_hash=payload,
            system_prompt=system_prompt,
            messages=messages,
            validator=validator,
        )

    def invoke_multimodal(
        self,
        *,
        run_id: str,
        payload_for_hash: dict[str, Any],
        system_prompt: str,
        content_parts: list[dict[str, Any]],
        validator: Callable[[dict[str, Any]], None],
        image_inputs: list[ModelImageInput] | None = None,
        agent_name: str = "semantic_visual",
    ) -> dict[str, Any]:
        if not self.config.capabilities.supports_image_input:
            failure = ModelFailure(
                category="provider_rejected",
                message="Configured model capability does not allow image input.",
                retryable=False,
            )
            raise ModelRuntimeError(failure.message, failure=failure)
        submitted_images = sum(part.get("type") == "image_url" for part in content_parts)
        if submitted_images != len(image_inputs or []):
            raise ModelRuntimeError(
                "Submitted image count differs from safe image telemetry."
            )
        return self._invoke_structured(
            agent_name=agent_name,
            runtime_kind="semantic_vlm",
            run_id=run_id,
            payload_for_hash=payload_for_hash,
            system_prompt=system_prompt,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts},
            ],
            validator=validator,
            image_inputs=image_inputs,
        )


__all__ = [
    "JsonHttpTransport",
    "ModelRuntimeError",
    "ModelBudgetState",
    "OpenAICompatibleClient",
    "ProviderResponseJSONError",
    "UrllibJsonTransport",
    "build_runtime_snapshot",
    "canonical_json_hash",
    "text_hash",
]
