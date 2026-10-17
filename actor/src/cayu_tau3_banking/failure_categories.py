"""Content-free failure categories, separate from authoritative benchmark grades.

Only typed errors and our exact wrapper envelope are recognized. An arbitrary
timeout is not evidence of budget exhaustion; provider failure is not proof of a
framework defect. Unknown cases stay unknown instead of guessing from prose.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

_CATEGORIES = {
    "RuntimeLimitError": "runtime_limit",
    "RuntimeLimitReached": "runtime_limit",
    "AdmissionStallError": "admission_stall",
    "PlanningStallError": "planning_stall",
    "RecallSourceUnavailable": "retrieval_unavailable",
    "RuntimeCallbackTimeoutError": "adapter_deadline",
    "ExecutionDeadlineExceeded": "execution_deadline",
    "RuntimeShutdownError": "cleanup_failure",
    "ReadError": "provider_transport",
    "ConnectError": "provider_transport",
    "RemoteProtocolError": "provider_transport",
    "ProviderConnectionError": "provider_transport",
    "APIConnectionError": "provider_transport",
    "RateLimitError": "provider_rate_limit",
    "AuthenticationError": "provider_authentication",
    "PermissionDeniedError": "provider_authentication",
    "ServiceUnavailableError": "provider_unavailable",
    "ModelStreamDeadlineError": "provider_stream_timeout",
    "LocalGenerationStreamTimeout": "provider_stream_timeout",
    # The runtime may preserve only the provider envelope in its terminal
    # exception. Keep that typed failure visible even when the transport
    # subtype is available only in private provider diagnostics.
    "ModelProviderError": "provider_error",
    "ModelAttemptFailed": "provider_error",
}
_PROVIDER_CODES = {
    "usage_limit_reached": "provider_quota",
    "insufficient_quota": "provider_quota",
    "billing_hard_limit_reached": "provider_quota",
    "rate_limit_exceeded": "provider_rate_limit",
    "service_unavailable_error": "provider_unavailable",
    "server_is_overloaded": "provider_unavailable",
}
_WRAPPER_PREFIX = "Runtime callback failed ("
_WRAPPER_SUFFIX = "); conversation stopped."


def classify_failure_info(info: object) -> str:
    """Project saved τ exception metadata without exposing messages or tracebacks."""
    if not isinstance(info, Mapping):
        return "unclassified"
    info = cast(Mapping[str, object], info)
    name = info.get("error_type")
    if name == "RuntimeCallbackExecutionError":
        message = info.get("error")
        if isinstance(message, str) and message.startswith(_WRAPPER_PREFIX) and message.endswith(_WRAPPER_SUFFIX):
            name = message[len(_WRAPPER_PREFIX) : -len(_WRAPPER_SUFFIX)]
    # A typed terminal stream deadline outranks an earlier recovered overload
    # response. Never infer final cause from the last transport error file alone.
    if isinstance(name, str) and name in {"ModelStreamDeadlineError", "LocalGenerationStreamTimeout"}:
        return "provider_stream_timeout"
    # Provider diagnostics are emitted in a separate, structured envelope.
    # Preserve their category when Tau has reduced the terminal exception to a
    # generic ModelProviderError/RuntimeCallbackExecutionError.
    provider_error = info.get("provider_error")
    if isinstance(provider_error, Mapping):
        error = cast(Mapping[str, object], provider_error).get("error")
        if isinstance(error, Mapping):
            for field in ("code", "type"):
                code = error.get(field)
                if isinstance(code, str) and code in _PROVIDER_CODES:
                    return _PROVIDER_CODES[code]
        # A bare 429 proves throttling, not specifically exhausted quota.
        # Explicit provider codes above take precedence over HTTP status.
        status = provider_error.get("http_status_code")
        if status == 429:
            return "provider_rate_limit"
        if status in (500, 502, 503, 504):
            return "provider_unavailable"
    evidence = info.get("failure_evidence")
    evidence_types = ()
    if isinstance(evidence, Mapping):
        raw_types = evidence.get("exception_types")
        if isinstance(raw_types, list | tuple):
            evidence_types = tuple(value for value in raw_types if isinstance(value, str))
    if not isinstance(name, str):
        return "unclassified"
    category = _CATEGORIES.get(name)
    # Prefer a specific structured cause over a generic provider wrapper.
    for evidence_type in evidence_types:
        evidence_category = _CATEGORIES.get(evidence_type)
        if evidence_category is not None and not (
            category == "provider_error" and evidence_category == "provider_error"
        ):
            return evidence_category
    if category is not None:
        return category
    return "unclassified"


def classify_exception(error: BaseException) -> str:
    evidence = getattr(error, "failure_evidence", None)
    return classify_failure_info(
        {
            "error_type": type(error).__name__,
            "error": str(error) if type(error).__name__ == "RuntimeCallbackExecutionError" else "",
            "failure_evidence": (
                evidence.model_dump(mode="json", exclude_none=True)
                if evidence is not None
                else None
            ),
        }
    )
