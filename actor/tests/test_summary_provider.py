import asyncio
import json

import httpx
import pytest
from cayu import Message
from cayu.providers import HttpxOpenAITransport, ModelRequest, OpenAIProvider
from cayu.providers.base import ModelStreamEventType
from cayu.runtime._model_errors import model_provider_error_from_payload
from cayu.runtime.retry_policy import retry_decision

from cayu_tau3_banking.compaction import banking_compactor
from cayu_tau3_banking.summary_provider import ReadOnlySummaryProvider


@pytest.mark.parametrize("failures", [1, 4])
def test_real_compactor_retries_http_body_loss_but_stops_at_four(failures):
    from uuid import uuid4

    from cayu import AgentSpec
    from cayu.context.base import CompactionRequest
    from cayu.providers.base import ModelProviderError
    from cayu.sessions.base import Session

    async def run():
        calls = []

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"type":"response.created","response":{"id":"r","status":"in_progress","output":[]}}\n\n'
                raise httpx.RemoteProtocolError("synthetic disconnect")

        def handle(request):
            calls.append(json.loads(request.content))
            if len(calls) <= failures:
                return httpx.Response(200, stream=Body())
            event = {
                "type": "response.completed",
                "response": {
                    "id": "resp_ok",
                    "model": "test",
                    "status": "completed",
                    "output": [
                        {
                            "id": "msg_ok",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Source facts preserved.",
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
                },
            }

            class SuccessBody(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield b'data: {"type":"response.output_text.delta","delta":"Source facts preserved."}\n\n'
                    yield ("data: " + json.dumps(event) + "\n\n").encode()

            return httpx.Response(200, stream=SuccessBody())

        transport = HttpxOpenAITransport()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            transport._client._client = client
            provider = OpenAIProvider(api_key="synthetic-key", transport=transport)
            compactor = banking_compactor(provider, "test")
            session = Session(
                id="summary-test",
                agent_name="a",
                provider_name="openai",
                model="test",
                invocation={
                    "origin": {"trust": "unattributed"},
                    "root_invocation_id": str(uuid4()),
                    "root_session_id": "summary-test",
                    "source": "sdk_run",
                },
            )
            request = CompactionRequest(
                session=session,
                agent=AgentSpec(name="a", model="test"),
                existing_summary="Previous checkpoint.",
                messages=[Message.text("user", "New source facts.")],
            )
            before = request.model_dump(mode="json")
            if failures == 4:
                with pytest.raises(ModelProviderError):
                    await compactor.compact(request)
            else:
                result = await compactor.compact(request)
                assert result.summary == "Source facts preserved."
            assert len(calls) == (4 if failures == 4 else 2)
            assert calls[0] == calls[1]
            assert "tools" not in calls[0] and calls[0]["store"] is False
            assert request.model_dump(mode="json") == before
            await provider.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("status", [200, 401, 429])
def test_only_interrupted_summary_body_is_reclassified(status):
    async def run():
        calls = []

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"type":"response.created","response":{"id":"r","status":"in_progress","output":[]}}\n\n'
                raise httpx.RemoteProtocolError("synthetic disconnect")

        def handle(request):
            calls.append(json.loads(request.content))
            return (
                httpx.Response(200, stream=Body())
                if status == 200
                else httpx.Response(status, json={"error": {"message": "synthetic error"}})
            )

        transport = HttpxOpenAITransport()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            transport._client._client = client
            actor = OpenAIProvider(api_key="synthetic-key", transport=transport)
            compactor = banking_compactor(actor, "test")
            assert isinstance(compactor.provider, ReadOnlySummaryProvider)
            request = ModelRequest(model="test", messages=[Message.text("user", "Summarize.")])
            events = [event async for event in compactor.provider.runtime_stream(request)]
            (error_event,) = [event for event in events if event.type is ModelStreamEventType.ERROR]
            error = model_provider_error_from_payload(
                error_event.payload, fallback_provider=actor.name
            )
            assert error is not None
            if status == 200:
                assert error.retryable is True
                assert error.error_type == "RemoteProtocolError"
                assert error_event.payload["summary_recovery"] == "interrupted_read_only_generation"
                for attempt, expected in ((1, True), (3, True), (4, False)):
                    assert (
                        retry_decision(
                            policy=compactor.retry_policy,
                            attempt=attempt,
                            error=str(error),
                            retryable=error.retryable,
                        ).retry
                        is expected
                    )
            else:
                assert "summary_recovery" not in error_event.payload
                assert error.status_code == status
            assert len(calls) == 1  # The adapter itself never retries.
            # The actor still has the original fail-closed policy.
            original = [event async for event in actor.runtime_stream(request)]
            if status == 200:
                assert (
                    next(e for e in original if e.type is ModelStreamEventType.ERROR).payload[
                        "retryable"
                    ]
                    is False
                )
            await actor.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "options,tools",
    [
        ({"openai": {"tools": [{"type": "web_search"}]}}, []),
        ({"openai": {"store": True}}, []),
        ({}, [{"name": "write", "description": "Write", "input_schema": {"type": "object"}}]),
    ],
)
def test_effect_capabilities_rejected_before_dispatch(options, tools):
    async def run():
        transport = HttpxOpenAITransport()

        def unexpected(request):
            raise AssertionError("Unsafe request reached HTTP transport")

        async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
            transport._client._client = client
            actor = OpenAIProvider(api_key="synthetic-key", transport=transport)
            summary = ReadOnlySummaryProvider(actor)
            request = ModelRequest(
                model="test", messages=[Message.text("user", "Act.")], options=options, tools=tools
            )
            with pytest.raises(ValueError):
                _ = [event async for event in summary.runtime_stream(request)]
            await actor.aclose()

    asyncio.run(run())
