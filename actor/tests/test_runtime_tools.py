from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    InMemorySessionStore,
    Message,
    ResumeRequest,
    RunRequest,
    SQLiteSessionStore,
    ToolResultPart,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec

from cayu_tau3_banking.runtime_tools import HostToolBroker, HostToolProtocolError

SPEC = ToolSpec(
    name="host_action",
    input_schema={
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    },
)


def context(session_id="session"):
    return ToolContext(session_id=session_id, idempotency_key="runtime-call-key")


def run(scenario):
    async def bounded():
        async with asyncio.timeout(20):
            return await scenario()

    return asyncio.run(bounded())


async def exchange(broker, tool, *, result=None):
    task = asyncio.create_task(tool.run(context(), {"value": 7}))
    try:
        call = await broker.next_call()
        broker.resolve(
            session_id=call.session_id,
            request_id=call.request_id,
            tool_name=call.tool_name,
            result=ToolResult(content="ok") if result is None else result,
        )
        return call, await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_result_and_arguments_are_detached_and_private():
    async def scenario():
        broker = HostToolBroker()
        call, result = await exchange(broker, broker.tool(SPEC))
        arguments = call.arguments
        arguments["value"] = 99
        assert call.arguments == {"value": 7}
        assert "value" not in repr(call)
        assert result.content == "ok"
        with pytest.raises(HostToolProtocolError, match="active delivered"):
            broker.resolve(
                session_id=call.session_id,
                request_id=call.request_id,
                tool_name=call.tool_name,
                result=ToolResult(content="ok"),
            )
        broker.close()

    run(scenario)


@pytest.mark.parametrize("field", ["session_id", "request_id", "tool_name"])
def test_wrong_result_identity_does_not_settle_or_consume_call(field):
    async def scenario():
        broker = HostToolBroker()
        task = asyncio.create_task(broker.tool(SPEC).run(context(), {"value": 7}))
        try:
            call = await broker.next_call()
            identity = dict(
                session_id=call.session_id, request_id=call.request_id, tool_name=call.tool_name
            )
            wrong = {**identity, field: "wrong"}
            with pytest.raises(HostToolProtocolError, match="active delivered"):
                broker.resolve(**wrong, result=ToolResult(content="forged"))
            assert not task.done()
            broker.resolve(**identity, result=ToolResult(content="authoritative"))
            assert (await task).content == "authoritative"
        finally:
            broker.close()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario)


@pytest.mark.parametrize("arguments", [{}, {"value": True}, {"value": "7"}, {"value": 7, "x": 1}])
def test_invalid_arguments_never_reach_host(arguments):
    async def scenario():
        broker = HostToolBroker()
        result = await broker.tool(SPEC).run(context(), arguments)
        assert result.is_error
        assert result.structured == {"error": "invalid_arguments"}
        assert not broker._pending
        assert not broker._queued
        broker.close()

    run(scenario)


def test_error_result_is_preserved():
    async def scenario():
        broker = HostToolBroker()
        _, result = await exchange(
            broker, broker.tool(SPEC), result=ToolResult(content="no", is_error=True)
        )
        assert result.is_error
        assert result.content == "no"
        broker.close()

    run(scenario)


@pytest.mark.parametrize("delivered", [False, True])
def test_cancellation_removes_queued_and_delivered_calls(delivered):
    async def scenario():
        broker = HostToolBroker(max_pending=1)
        task = asyncio.create_task(broker.tool(SPEC).run(context(), {"value": 7}))
        try:
            if delivered:
                call = await broker.next_call()
            else:
                await broker._available.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not broker._queued
            assert not broker._pending
            if delivered:
                with pytest.raises(HostToolProtocolError, match="active delivered"):
                    broker.resolve(
                        session_id=call.session_id,
                        request_id=call.request_id,
                        tool_name=call.tool_name,
                        result=ToolResult(content="late"),
                    )
            # Cancellation must return capacity and leave no stale queue entries.
            await exchange(broker, broker.tool(SPEC))
        finally:
            broker.close()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario)


def test_close_wakes_receiver_and_cancels_unresolved_execution():
    async def scenario():
        broker = HostToolBroker()
        task = asyncio.create_task(broker.tool(SPEC).run(context(), {"value": 7}))
        await broker.next_call()
        receiver = asyncio.create_task(broker.next_call())
        broker.close()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(HostToolProtocolError, match="closed"):
            await receiver
        assert not broker._pending
        broker.close()

    run(scenario)


def test_pending_capacity_is_bounded():
    async def scenario():
        broker = HostToolBroker(max_pending=1)
        tool = broker.tool(SPEC)
        task = asyncio.create_task(tool.run(context(), {"value": 7}))
        try:
            await broker.next_call()
            with pytest.raises(HostToolProtocolError, match="capacity"):
                await tool.run(context(), {"value": 8})
        finally:
            broker.close()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario)


class _LocalTool(Tool):
    spec = ToolSpec(name="local_action", input_schema={"type": "object"})

    async def run(self, ctx, args):
        return ToolResult(content="local-result")


def test_cancelled_call_is_not_delivered_before_coroutine_cleanup():
    async def scenario():
        broker = HostToolBroker()
        tool = broker.tool(SPEC)
        first = asyncio.create_task(tool.run(context(), {"value": 7}))
        await broker._available.wait()
        first.cancel()
        second = asyncio.create_task(tool.run(context(), {"value": 8}))
        try:
            # Do not yield to the cancelled coroutine's cleanup before receiving.
            call = await broker.next_call()
            assert call.arguments == {"value": 8}
            broker.resolve(
                session_id=call.session_id,
                request_id=call.request_id,
                tool_name=call.tool_name,
                result=ToolResult(content="second"),
            )
            assert (await second).content == "second"
            with pytest.raises(asyncio.CancelledError):
                await first
        finally:
            broker.close()
            await asyncio.gather(first, second, return_exceptions=True)

    run(scenario)


def test_malformed_result_does_not_consume_call():
    async def scenario():
        broker = HostToolBroker()
        task = asyncio.create_task(broker.tool(SPEC).run(context(), {"value": 7}))
        try:
            call = await broker.next_call()
            identity = dict(
                session_id=call.session_id, request_id=call.request_id, tool_name=call.tool_name
            )
            with pytest.raises(ValueError):
                broker.resolve(**identity, result=ToolResult.model_construct(is_error="false"))
            assert not task.done()
            broker.resolve(**identity, result=ToolResult(content="ok"))
            assert (await task).content == "ok"
        finally:
            broker.close()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario)


def test_broker_rejects_use_from_a_different_event_loop():
    broker = HostToolBroker()

    async def first():
        await exchange(broker, broker.tool(SPEC))
        broker.close()

    async def second():
        with pytest.raises(HostToolProtocolError, match="another event loop"):
            await broker.next_call()

    run(first)
    run(second)


def test_cancelling_receiver_does_not_cancel_the_next_host_action():
    async def scenario():
        broker = HostToolBroker()
        receiver = asyncio.create_task(broker.next_call())
        # Run the receiver until it is waiting, without a wall-clock sleep.
        await asyncio.sleep(0)
        with pytest.raises(HostToolProtocolError, match="Only one"):
            await broker.next_call()
        receiver.cancel()
        with pytest.raises(asyncio.CancelledError):
            await receiver
        await exchange(broker, broker.tool(SPEC))
        broker.close()

    run(scenario)


@pytest.mark.parametrize("invalid", [True, 0, -1, 257, 1.5, "16"])
def test_broker_rejects_invalid_capacity(invalid):
    with pytest.raises(ValueError, match="max_pending"):
        HostToolBroker(max_pending=invalid)


def test_host_tools_cannot_claim_cayu_workspace_mutations():
    with pytest.raises(ValueError, match="workspace mutation"):
        HostToolBroker().tool(
            SPEC.model_copy(update={"workspace_mutation": True, "parallel_safe": False})
        )


class _Provider(ModelProvider):
    name = "runtime-test"

    def __init__(self):
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            for call_id, name, arguments in (
                ("local-1", "local_action", {}),
                ("host-1", "host_action", {"value": 7}),
                ("host-2", "host_action", {"value": 8}),
            ):
                yield ModelStreamEvent.tool_call(id=call_id, name=name, arguments=arguments)
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        else:
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})


@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "sqlite"])
def test_cayu_owns_mixed_rounds_and_preserves_history_across_customer_turns(tmp_path, persistent):
    async def collect(stream):
        return [event async for event in stream]

    async def scenario():
        store = (
            SQLiteSessionStore(tmp_path / "runtime.sqlite3")
            if persistent
            else InMemorySessionStore()
        )
        broker = HostToolBroker()
        provider = _Provider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="agent", model="scripted"),
            tools=[_LocalTool(), broker.tool(SPEC)],
        )
        task = asyncio.create_task(
            collect(
                app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="session",
                        messages=[Message.text("user", "first")],
                    )
                )
            )
        )
        try:
            async with asyncio.timeout(15):
                ids = []
                for value in (7, 8):
                    call = await broker.next_call()
                    assert call.arguments == {"value": value}
                    ids.append(call.idempotency_key)
                    assert len(provider.requests) == 1
                    assert (await store.load_checkpoint("session"))["pending_tool_round"]
                    broker.resolve(
                        session_id=call.session_id,
                        request_id=call.request_id,
                        tool_name=call.tool_name,
                        result=ToolResult(content=f"host-result-{value}"),
                    )
                events = await task
                assert len(set(ids)) == 2
                assert any(event.type == EventType.SESSION_COMPLETED for event in events)
                assert len(provider.requests) == 2
                results = [
                    part
                    for message in provider.requests[1].messages
                    for part in message.content
                    if isinstance(part, ToolResultPart)
                ]
                assert [part.content for part in results] == [
                    "local-result",
                    "host-result-7",
                    "host-result-8",
                ]
                assert not (await store.load_checkpoint("session")).get("pending_tool_round")
                await collect(
                    app.resume(
                        ResumeRequest(
                            session_id="session", messages=[Message.text("user", "second")]
                        )
                    )
                )
                assert len(provider.requests) == 3
                texts = [
                    part.text
                    for message in provider.requests[2].messages
                    for part in message.content
                    if part.type == "text"
                ]
                assert "first" in texts
                assert "second" in texts
                assert "done" in texts
                assert (
                    sum(
                        isinstance(part, ToolResultPart)
                        for message in provider.requests[2].messages
                        for part in message.content
                    )
                    == 3
                )
        finally:
            broker.close()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if persistent:
                await store.close()

    run(scenario)


def test_stopping_runtime_with_a_delivered_host_call_does_not_advance_the_model():
    async def scenario():
        broker = HostToolBroker()
        provider = _Provider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="agent", model="scripted"),
            tools=[_LocalTool(), broker.tool(SPEC)],
        )
        events = []

        async def drain():
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="session",
                    messages=[Message.text("user", "first")],
                )
            ):
                events.append(event)

        task = asyncio.create_task(drain())
        try:
            call = await broker.next_call()
            task.cancel()
            broker.close()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(provider.requests) == 1
            assert not broker._pending
            assert not broker._queued
            assert not any(event.type == EventType.SESSION_COMPLETED for event in events)
            with pytest.raises(HostToolProtocolError, match="active delivered"):
                broker.resolve(
                    session_id=call.session_id,
                    request_id=call.request_id,
                    tool_name=call.tool_name,
                    result=ToolResult(content="late success"),
                )
        finally:
            task.cancel()
            broker.close()
            await asyncio.gather(task, return_exceptions=True)

    run(scenario)
