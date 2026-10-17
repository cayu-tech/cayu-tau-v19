"""Bounded transport recovery; Cayu retains ownership of request retries."""

import asyncio
from contextlib import aclosing

from cayu.providers import OpenAISubscriptionProvider
from cayu.providers._stream_cleanup import (
    _local_http_cleanup_observer,
    _LocalHttpCleanupObserver,
)
from cayu.providers.base import (
    ModelProviderError,
    ModelStreamDeadlineError,
    ModelStreamEvent,
    ModelStreamEventType,
    _terminal_preserving_provider_stream,
)
from cayu.providers.openai import build_openai_payload

QUOTA_CODES = frozenset({"usage_limit_reached", "insufficient_quota", "billing_hard_limit_reached"})
CLEANUP_RECEIPT_WAIT_SECONDS = 1.0


def _local_generation_only(request):
    payload = build_openai_payload(request, stream=True, reasoning_state="inline")
    return (
        payload.get("store") is False
        and not request.hosted_tools
        and not any(k in payload for k in ("previous_response_id", "conversation", "background"))
        and all(t.get("type") == "function" for t in payload.get("tools", []))
    )


class RecoveringSubscriptionProvider(OpenAISubscriptionProvider):
    async def runtime_stream(self, request):
        """Let Cayu retry an unfinished, locally settled pure generation.

        The pinned framework's cleanup observer is chained, not replaced: its
        durable receipt must publish before recovery is allowed. Socket closure
        alone says nothing about remote effects, hence the separate no-hosted-
        tools/no-continuation contract. Banking tools execute only after a model
        attempt completes; Cayu discards partial calls on a failed attempt.
        """
        local_only = _local_generation_only(request)
        parent = _local_http_cleanup_observer.get()
        receipt_ready = asyncio.Event()
        receipt = None

        async def published(evidence, succeeded):
            nonlocal receipt
            if parent is not None:
                parent.expired(evidence)
                await parent.closed(succeeded=succeeded)
                publication = parent.confirm_expiry()
                if publication is not None:
                    await publication
            receipt = (evidence.payload(), succeeded)
            receipt_ready.set()

        observer = _LocalHttpCleanupObserver(published)
        completed = False
        failure = None
        failure_event = None
        async with aclosing(super().runtime_stream(request)) as stream:
            try:
                while True:
                    # The runtime may advance/close generators in different
                    # task contexts. Never retain a ContextVar token over yield.
                    token = _local_http_cleanup_observer.set(observer)
                    try:
                        event = await anext(stream)
                    except StopAsyncIteration:
                        return
                    finally:
                        _local_http_cleanup_observer.reset(token)
                    if event.type is ModelStreamEventType.COMPLETED:
                        completed = True
                    if (event.type is ModelStreamEventType.ERROR
                            and event.payload.get("provider_error_type") == "ModelStreamDeadlineError"):
                        failure_event = event
                        break
                    yield event
            except ModelStreamDeadlineError as exc:
                failure = exc
        payload = failure.error_payload_fields() if failure is not None else failure_event.payload
        semantic = payload.get("provider_error_code") == "provider_stream_semantic_idle_timeout"
        if local_only and not completed and semantic:
            if not receipt_ready.is_set():
                # Timeout/caller cancellation remains terminal; no detached
                # generation, close or retry is started here.
                try:
                    async with asyncio.timeout(CLEANUP_RECEIPT_WAIT_SECONDS):
                        await receipt_ready.wait()
                except TimeoutError:
                    pass
            matched = (
                parent is not None and receipt is not None and receipt[1] is True
                and all(payload.get(k) == v for k, v in receipt[0].items())
            )
            if matched:
                record = getattr(self.transport, "record_generation_recovery", None)
                if record is not None:
                    record({
                        "original_failure": payload,
                        "local_http_cleanup": "succeeded",
                        "scope": "unfinished_local_function_generation",
                        "retry_owner": "cayu_runtime",
                    })
                error = ModelProviderError(
                    "Unfinished local generation stalled; matching HTTP cleanup confirmed.",
                    provider=self.name, error_type="LocalGenerationStreamTimeout",
                    error_code="settled_local_generation_timeout", retryable=True,
                    retry_after_s=2,
                )
                yield ModelStreamEvent.error(str(error), cause=error)
                return
        if failure is not None:
            raise failure
        yield failure_event

    @_terminal_preserving_provider_stream
    async def stream(self, request):
        # Only local function calls are allowed: runtime dispatches them, not the
        # remote provider. Never widen retries for hosted tools or continuations.
        local_only = _local_generation_only(request)
        completed = False
        async with aclosing(super().stream(request)) as stream:
            async for event in stream:
                if event.type is ModelStreamEventType.COMPLETED:
                    completed = True
                if (
                    event.type is ModelStreamEventType.ERROR
                    and any(event.payload.get(key) in QUOTA_CODES
                            for key in ("provider_error_type", "provider_error_code"))
                ):
                    # An exhausted allowance cannot recover inside a request's
                    # short retry window. Preserve the provider error as infra.
                    event = ModelStreamEvent(
                        type=event.type,
                        payload={**event.payload, "retryable": False,
                                 "quota_recovery": "requires_capacity_or_reset"},
                    )
                elif (
                    local_only
                    and not completed
                    and event.type is ModelStreamEventType.ERROR
                    and event.payload.get("provider_error_type")
                    in {"RemoteProtocolError", "ReadError", "ReadTimeout", "service_unavailable_error"}
                    and event.payload.get("status_code") in (None, 200)
                    and event.payload.get("retryable") is False
                    and not event.payload.get("stream_cleanup_failed")
                ):
                    event = ModelStreamEvent(
                        type=event.type,
                        payload={**event.payload, "retryable": True,
                                 "transport_recovery": "local_function_generation"},
                    )
                yield event
