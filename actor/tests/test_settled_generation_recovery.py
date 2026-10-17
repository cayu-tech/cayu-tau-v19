"""Real HTTP-stream deadlines: only settled local generations can be retried."""
import asyncio
import json

import httpx
import pytest
from cayu import Message, RetryPolicy
from cayu.providers import HttpxOpenAITransport, ModelRequest
from cayu.providers.base import ModelStreamDeadlineError
from cayu.providers.deadlines import ProviderStreamDeadlines
from cayu.providers.openai_subscription import OpenAISubscriptionCredentials
from tau2.data_model.message import UserMessage

from cayu_tau3_banking.provider_diagnostics import DiagnosticOpenAITransport
from cayu_tau3_banking.runtime_agent import RuntimeCallbackExecutionError
from cayu_tau3_banking.subscription_recovery import RecoveringSubscriptionProvider


def frame(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


class Auth:
    async def credentials(self):
        return OpenAISubscriptionCredentials(access_token="synthetic-access", refresh_token="synthetic-refresh",
            expires_at=2_000_000_000, account_id="synthetic-account")


def setup_provider(tmp_path, *, fail_close=False, delay_close=0, always_stall=False, partial_tool=False):
    requests = []
    closed = []

    class Body(httpx.AsyncByteStream):
        def __init__(self, index):
            self.index = index

        async def __aiter__(self):
            if self.index == 0 or always_stall:
                yield frame({"type": "response.created", "response": {"id": "r0", "status": "in_progress", "output": []}})
                if partial_tool:
                    item = {"id": "fc0", "type": "function_call", "call_id": "discarded", "name": "host_action", "arguments": ""}
                    yield frame({"type": "response.output_item.added", "output_index": 0, "item": item})
                    yield frame({"type": "response.function_call_arguments.delta", "item_id": "fc0", "output_index": 0, "delta": '{"value":1}'})
                    yield frame({"type": "response.function_call_arguments.done", "item_id": "fc0", "output_index": 0, "arguments": '{"value":1}'})
                yield frame({"type": "response.reasoning_summary_text.delta", "delta": "synthetic"})
                await asyncio.Event().wait()
            else:
                yield frame({"type": "response.output_text.delta", "delta": "Recovered."})
                yield frame({"type": "response.completed", "response": {"id": "r1", "model": requests[self.index]["model"], "status": "completed", "output": [{"id": "m1", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "Recovered.", "annotations": []}]}]}})

        async def aclose(self):
            if delay_close and self.index == 0:
                await asyncio.sleep(delay_close)
            if fail_close:
                raise RuntimeError("synthetic close failure")
            closed.append(self.index)

    def handler(request):
        index = len(requests)
        requests.append(json.loads(request.content))
        return httpx.Response(200, stream=Body(index))

    base = HttpxOpenAITransport()
    base._client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = DiagnosticOpenAITransport(base, tmp_path)
    provider = RecoveringSubscriptionProvider(auth=Auth(), transport=transport,
        stream_deadlines=ProviderStreamDeadlines(semantic_progress_timeout_s=0.1,
            protocol_idle_timeout_s=2, transport_idle_timeout_s=2, absolute_stream_timeout_s=3))
    return provider, requests, closed


@pytest.mark.parametrize("delay", [0, 0.2])
def test_real_runtime_recovers_stall_without_dispatching_partial_tool(tmp_path, delay):
    from test_runtime_agent import make_agent
    provider, requests, closed = setup_provider(tmp_path, delay_close=delay, partial_tool=True)
    agent, _, _ = make_agent(provider, retry_policy=RetryPolicy(max_attempts=2))
    state = agent.get_init_state([])
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("hello"), state)
        assert reply.content == "Recovered."
        assert not reply.tool_calls
        assert len(requests) == 2
        assert closed == [0, 1]
        receipts = list(tmp_path.glob("*/*.recovery.json"))
        assert len(receipts) == 1
        receipt = json.loads(receipts[0].read_text())
        assert receipt["original_failure"]["provider_error_code"] == "provider_stream_semantic_idle_timeout"
        assert receipt["local_http_cleanup"] == "succeeded"
        # The second request is the same logical generation, without an invented
        # tool result or partial action from the abandoned attempt.
        assert requests[0]["input"] == requests[1]["input"]
    finally:
        agent.stop(state=state)
        asyncio.run(provider.aclose())


def test_failed_close_remains_terminal(tmp_path):
    from test_runtime_agent import make_agent
    provider, requests, closed = setup_provider(tmp_path, fail_close=True)
    agent, _, _ = make_agent(provider, retry_policy=RetryPolicy(max_attempts=10))
    state = agent.get_init_state([])
    try:
        with pytest.raises(RuntimeCallbackExecutionError):
            agent.generate_next_message(UserMessage.text("hello"), state)
        assert len(requests) == 1
        assert not closed
        assert not list(tmp_path.glob("*/*.recovery.json"))
    finally:
        agent.stop(state=state)
        asyncio.run(provider.aclose())


def test_stalls_obey_runtime_attempt_budget(tmp_path):
    from test_runtime_agent import make_agent
    provider, requests, _ = setup_provider(tmp_path, always_stall=True)
    agent, _, _ = make_agent(provider, retry_policy=RetryPolicy(max_attempts=2))
    state = agent.get_init_state([])
    try:
        with pytest.raises(RuntimeCallbackExecutionError):
            agent.generate_next_message(UserMessage.text("hello"), state)
        assert len(requests) == 2
        assert len(list(tmp_path.glob("*/*.recovery.json"))) == 2
    finally:
        agent.stop(state=state)
        asyncio.run(provider.aclose())


def test_opaque_transport_without_cleanup_proof_does_not_gain_retry(tmp_path, monkeypatch):
    monkeypatch.setattr("cayu_tau3_banking.subscription_recovery.CLEANUP_RECEIPT_WAIT_SECONDS", 0.05)
    class Opaque:
        async def stream_response_events(self, **kwargs):
            yield {"type": "response.created", "response": {"id": "r", "status": "in_progress", "output": []}}
            await asyncio.Event().wait()
        async def aclose(self):
            pass
    async def run():
        provider = RecoveringSubscriptionProvider(auth=Auth(), transport=Opaque(),
            stream_deadlines=ProviderStreamDeadlines(semantic_progress_timeout_s=0.1))
        with pytest.raises(ModelStreamDeadlineError):
            async for _ in provider.runtime_stream(ModelRequest(model="test", messages=[Message.text("user", "hello")])):
                pass
        await provider.aclose()
    asyncio.run(run())


def test_terminal_deadline_outweighs_earlier_overload():
    from cayu_tau3_banking.failure_categories import classify_failure_info
    info = {"error_type": "RuntimeCallbackExecutionError", "error": "Runtime callback failed (ModelStreamDeadlineError); conversation stopped.",
        "provider_error": {"error": {"code": "server_is_overloaded"}, "http_status_code": 200}}
    assert classify_failure_info(info) == "provider_stream_timeout"


def test_hosted_and_continuation_payloads_are_ineligible():
    from cayu.providers.hosted import OpenAIWebSearch

    from cayu_tau3_banking.subscription_recovery import _local_generation_only
    plain = dict(model="test", messages=[Message.text("user", "hello")])
    assert _local_generation_only(ModelRequest(**plain))
    assert not _local_generation_only(ModelRequest(**plain, hosted_tools=(OpenAIWebSearch(),)))
    assert not _local_generation_only(ModelRequest(**plain, options={"openai": {"conversation": "c0"}}))
    # The serializer already refuses caller overrides for stored/background
    # responses and server continuations, before any network request.
    for key, value in [("store", True), ("background", True), ("previous_response_id", "r0")]:
        with pytest.raises(ValueError):
            _local_generation_only(ModelRequest(**plain, options={"openai": {key: value}}))


def test_missing_runtime_receipt_publication_prevents_recovery(tmp_path):
    async def run():
        provider, requests, _ = setup_provider(tmp_path)
        try:
            with pytest.raises(ModelStreamDeadlineError):
                async for _ in provider.runtime_stream(ModelRequest(model="test", messages=[Message.text("user", "hello")])):
                    pass
            assert len(requests) == 1
            assert not list(tmp_path.glob("*/*.recovery.json"))
        finally:
            await provider.aclose()
    asyncio.run(run())


def test_caller_cancellation_while_cleanup_pending_is_preserved(tmp_path):
    from cayu.providers._stream_cleanup import (
        _local_http_cleanup_observer,
        _LocalHttpCleanupObserver,
    )
    async def run():
        provider, requests, _ = setup_provider(tmp_path, delay_close=0.4)
        async def publish(evidence, succeeded):
            pass
        async def consume():
            token = _local_http_cleanup_observer.set(_LocalHttpCleanupObserver(publish))
            try:
                async for _ in provider.runtime_stream(ModelRequest(model="test", messages=[Message.text("user", "hello")])):
                    pass
            finally:
                _local_http_cleanup_observer.reset(token)
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.25)
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.4)
            assert len(requests) == 1
            assert not list(tmp_path.glob("*/*.recovery.json"))
        finally:
            await provider.aclose()
    asyncio.run(run())


def test_failed_durable_publication_prevents_recovery(tmp_path):
    from cayu.providers._stream_cleanup import (
        _local_http_cleanup_observer,
        _LocalHttpCleanupObserver,
    )
    async def run():
        provider, requests, _ = setup_provider(tmp_path)
        async def publish(evidence, succeeded):
            raise OSError("synthetic durable event publication failure")
        token = _local_http_cleanup_observer.set(_LocalHttpCleanupObserver(publish))
        try:
            with pytest.raises((ModelStreamDeadlineError, OSError)):
                async for _ in provider.runtime_stream(ModelRequest(model="test", messages=[Message.text("user", "hello")])):
                    pass
            assert len(requests) == 1
            assert not list(tmp_path.glob("*/*.recovery.json"))
        finally:
            _local_http_cleanup_observer.reset(token)
            await provider.aclose()
    asyncio.run(run())
