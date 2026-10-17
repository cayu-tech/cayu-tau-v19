"""Credential-free regression for completion followed by a stalled HTTP body.

Completed responses close cleanly under the integrated framework; an unfinished
response still times out. Neither identifies the historical remote cause.
"""

import asyncio
import json

import httpx
import pytest
from cayu import Message
from cayu.providers import HttpxOpenAITransport, ModelRequest, OpenAIProvider
from cayu.providers.base import ModelStreamDeadlineError, ModelStreamEventType
from cayu.providers.deadlines import ProviderStreamDeadlines
from cayu.providers.openai_subscription import (
    OpenAISubscriptionCredentials,
    OpenAISubscriptionProvider,
)


@pytest.mark.parametrize("stalled,terminal", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize("subscription", [False, True])
def test_response_then_http_eof_or_stall(stalled, subscription, terminal):
    async def run():
        event_payload = {
            "type": "response.completed",
            "response": {
                "id": "resp_probe",
                "model": "test-model",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_probe",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                    }
                ],
            },
        }
        if not terminal:
            event_payload = {
                "type": "response.created",
                "response": {
                    "id": "resp_probe",
                    "model": "test-model",
                    "status": "in_progress",
                    "output": [],
                },
            }

        class Body(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield ("data: " + json.dumps(event_payload) + "\n\n").encode()
                if stalled:
                    await asyncio.Event().wait()

            async def aclose(self):
                self.closed = True

        body = Body()
        transport = HttpxOpenAITransport()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=body))
        ) as client:
            transport._client._client = client

            class StaticAuth:
                async def credentials(self):
                    return OpenAISubscriptionCredentials(
                        access_token="synthetic-access",
                        refresh_token="synthetic-refresh",
                        expires_at=2_000_000_000,
                        account_id="synthetic-account",
                    )

            options = dict(
                transport=transport,
                stream_deadlines=ProviderStreamDeadlines(
                    semantic_progress_timeout_s=0.1,
                    transport_idle_timeout_s=2,
                    protocol_idle_timeout_s=2,
                    absolute_stream_timeout_s=3,
                ),
            )
            provider = (
                OpenAISubscriptionProvider(auth=StaticAuth(), **options)
                if subscription
                else OpenAIProvider(api_key="synthetic-not-a-credential", **options)
            )
            events = []
            error = None
            try:
                async with asyncio.timeout(5):
                    async for event in provider.runtime_stream(
                        ModelRequest(
                            model="test-model", messages=[Message.text("user", "Synthetic probe")]
                        )
                    ):
                        events.append(event)
            except ModelStreamDeadlineError as exc:
                error = exc
            assert sum(e.type == ModelStreamEventType.COMPLETED for e in events) == int(terminal)
            assert body.closed
            if stalled and not terminal:
                assert error is not None
                assert error.error_payload_fields()["provider_last_progress_kind"] != "terminal"
            else:
                assert error is None
            await provider.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("subscription", [False, True])
def test_disconnect_after_http_success_is_explicitly_nonretryable(subscription):
    """The task-100 boundary: a larger unknown-attempt cap cannot recover this."""
    from cayu.runtime._model_errors import model_provider_error_from_payload
    from cayu.runtime.retry_policy import RetryPolicy, retry_decision

    async def run():
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"type":"response.created","response":{"id":"r","status":"in_progress","output":[]}}\n\n'
                raise httpx.RemoteProtocolError("synthetic disconnect")

        class StaticAuth:
            async def credentials(self):
                return OpenAISubscriptionCredentials(
                    access_token="synthetic-access",
                    refresh_token="synthetic-refresh",
                    expires_at=2_000_000_000,
                    account_id="synthetic-account",
                )

        transport = HttpxOpenAITransport()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Body()))
        ) as client:
            transport._client._client = client
            provider = (
                OpenAISubscriptionProvider(auth=StaticAuth(), transport=transport)
                if subscription
                else OpenAIProvider(api_key="synthetic-key", transport=transport)
            )
            events = [
                event
                async for event in provider.runtime_stream(
                    ModelRequest(
                        model="test",
                        messages=[Message.text("user", "Summarize fictional dialogue.")],
                    )
                )
            ]
            errors = [event for event in events if event.type is ModelStreamEventType.ERROR]
            assert len(errors) == 1
            assert not any(event.type is ModelStreamEventType.COMPLETED for event in events)
            error = model_provider_error_from_payload(
                errors[0].payload, fallback_provider=provider.name
            )
            assert error is not None and error.retryable is False
            assert error.error_type == "RemoteProtocolError"
            for cap in (1, 2):
                decision = retry_decision(
                    policy=RetryPolicy(max_attempts=2, max_unknown_attempts=cap),
                    attempt=1,
                    error=str(error),
                    retryable=error.retryable,
                    status_code=error.status_code,
                )
                assert decision.retry is False
            await provider.aclose()

    asyncio.run(run())
