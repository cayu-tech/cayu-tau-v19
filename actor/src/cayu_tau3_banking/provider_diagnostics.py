"""Opt-in private OpenAI transport diagnostics, never authentication/wire logs.

Requests contain conversation data: enable only for local diagnostics. Transport
headers, URLs, arbitrary metadata and successful response text are not recorded.
Bounded, redacted provider explanations are private diagnostic data, not logs.
The wrapper neither retries requests nor changes events or error propagation.
"""

import asyncio
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping
from contextlib import aclosing, contextmanager
from pathlib import Path
from uuid import uuid4

from cayu.providers import capture_provider_errors
from cayu.vaults import SecretRedactor

_IDENTITIES = frozenset(
    {
        "usage_limit_reached",
        "rate_limit_exceeded",
        "rate_limit_error",
        "insufficient_quota",
        "billing_hard_limit_reached",
        "model_not_found",
        "invalid_request_error",
        "invalid_value",
        "invalid_prompt",
        "unsupported_model",
        "server_error",
        "internal_error",
        "authentication_error",
        "permission_error",
        "context_length_exceeded",
        "previous_response_not_found",
        "bad_request",
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "WriteError",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "LocalProtocolError",
        "RequestError",
        "SseEventLimitError",
        "SseEventTimeoutError",
    }
)
_REQUEST_KEYS = frozenset(
    {
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "text",
        "max_output_tokens",
        "stream",
        "store",
        "include",
    }
)


def safe_error_fields(value: Mapping) -> dict:
    """Project only known identities and valid numeric status; no free-form strings."""
    result = {}
    for key in ("type", "code"):
        field = value.get(key)
        result[key] = (
            field
            if isinstance(field, str) and field in _IDENTITIES
            else ("unrecognized" if field is not None else None)
        )
    status = value.get("status_code")
    if type(status) is int and 100 <= status <= 599:
        result["status_code"] = status
    retry = value.get("retry_after_s")
    if type(retry) in (int, float) and math.isfinite(retry) and retry >= 0:
        result["retry_after_s"] = retry
    return result


class DiagnosticOpenAITransport:
    def __init__(self, base, directory: Path, *, redactor: SecretRedactor | None = None):
        self.base = base
        # This agent uses a public, fictional benchmark. Other workloads must
        # supply their own registered secrets; capture is not PII anonymization.
        self.redactor = redactor if redactor is not None else SecretRedactor()
        self.capture_failures = 0
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="provider-", dir=directory))

    def _save(self, name, value):
        descriptor = os.open(self.directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, allow_nan=False)

    def record_generation_recovery(self, value):
        """Preserve typed deadline/cleanup evidence before runtime-owned retry."""
        self._save(f"{uuid4().hex}.recovery.json", {
            "schema": "cayu-tau3.local-generation-recovery.v1", **value,
        })

    async def stream_response_events(self, **kwargs):
        if self.capture_failures:
            raise RuntimeError(
                "Provider diagnostic capture failed; repair it before another request."
            )
        capture_id = uuid4().hex
        started = time.monotonic()
        payload = kwargs["payload"]
        headers = kwargs.get("headers", {})
        credentials = []
        for name, value in headers.items():
            if isinstance(value, str) and value.strip():
                # The subscription transport's public client identifier is not
                # a credential. Registering it masks our public tool names and
                # makes captured requests unusable for diagnosis. Exempt only
                # this exact protocol pair, not arbitrary originator values or
                # the same string supplied through an authentication header.
                if name.lower() == "originator" and value == "cayu":
                    continue
                credentials.append(value)
                if name.lower() in {"authorization", "proxy-authorization"}:
                    _, separator, token = value.partition(" ")
                    if separator and token.strip():
                        credentials.append(token.strip())
        redactor = self.redactor.merged_with(SecretRedactor(credentials))
        # Serialize before dispatch so caller mutation cannot alter this snapshot.
        snapshot = json.loads(json.dumps({k: v for k, v in payload.items() if k in _REQUEST_KEYS}))
        redacted_snapshot = redactor.redact_json_values(snapshot)
        await asyncio.to_thread(
            self._save,
            f"{capture_id}.request.json",
            {
                "schema": "cayu-tau3.provider-request.v1",
                "capture_id": capture_id,
                "boundary": "prepared payload before transport; allowlisted fields only",
                "payload": redacted_snapshot,
                "payload_redacted": redacted_snapshot != snapshot,
                "omitted_field_count": len(set(payload) - _REQUEST_KEYS),
            },
        )

        totals = dict(records_attempted=0, records_written=0, sink_failures=0, records_dropped=0)

        def save_error(record):
            if totals["records_written"] >= 32:
                totals["records_dropped"] += 1
                raise RuntimeError("Per-request diagnostic record limit exceeded.")
            record["sequence"] += totals["records_attempted"]
            suffix = "exception" if record["boundary"] == "transport_error" else "error"
            self._save(f"{capture_id}.{record['sequence']}.{suffix}.json", record)

        @contextmanager
        def capture_step():
            # Never hold a ContextVar token across an async-generator yield:
            # its next read/close may be driven by a different task/context.
            with capture_provider_errors(
                save_error, redactor=redactor, capture_id=capture_id
            ) as capture:
                try:
                    yield capture
                finally:
                    for key in totals:
                        totals[key] += getattr(capture, key)

        exception_active = False
        try:
            async with aclosing(self.base.stream_response_events(**kwargs)) as stream:
                while True:
                    with capture_step() as capture:
                        try:
                            event = await anext(stream)
                        except StopAsyncIteration:
                            break
                        # Bundled HTTP capture already has the real response
                        # headers. Custom transports can only expose event data.
                        if (
                            isinstance(event.get("type"), str)
                            and event["type"] in {"error", "response.failed"}
                            and capture.records_attempted == 0
                        ):
                            response = event.get("response")
                            container = response if isinstance(response, Mapping) else event
                            error = container.get("error", container)
                            capture.record_error(
                                error if isinstance(error, Mapping) else {},
                                boundary="stream_error",
                                provider_request_id=container.get("request_id"),
                                request_id_source=(
                                    "body"
                                    if container.get("request_id") is not None
                                    else "unavailable"
                                ),
                            )
                    yield event
        except BaseException as error:
            exception_active = True
            if isinstance(error, Exception) and not totals["records_attempted"]:
                with capture_step() as capture:
                    fields = safe_error_fields(
                        {
                            "type": getattr(error, "error_type", None),
                            "code": getattr(error, "error_code", None),
                            "status_code": getattr(error, "status_code", None),
                            "retry_after_s": getattr(error, "retry_after_s", None),
                        }
                    )
                    capture.record_error(fields, boundary="transport_error")
            raise
        finally:
            self.capture_failures += totals["sink_failures"] + totals["records_dropped"]
            try:
                self._save(
                    f"{capture_id}.capture.json",
                    {
                        "schema": "cayu-tau3.provider-capture.v1",
                        "capture_id": capture_id,
                        "elapsed_seconds": time.monotonic() - started,
                        **totals,
                    },
                )
            except OSError:
                self.capture_failures += 1
                if not exception_active:
                    raise RuntimeError(
                        "Provider diagnostic status could not be persisted."
                    ) from None

    async def aclose(self):
        await self.base.aclose()
