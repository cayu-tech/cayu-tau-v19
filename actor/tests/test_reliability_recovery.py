import asyncio

import pytest
from cayu import Message, ModelStreamEvent, ScriptedModelProvider
from cayu.providers import ModelRequest, OpenAISubscriptionProvider
from cayu.providers.base import ModelStreamEventType
from tau2.data_model.message import UserMessage

from cayu_tau3_banking.compaction import banking_compactor
from cayu_tau3_banking.runtime_agent import RuntimeCallbackExecutionError
from cayu_tau3_banking.subscription_recovery import RecoveringSubscriptionProvider
from cayu_tau3_banking.summary_provider import ReadOnlySummaryProvider


@pytest.mark.parametrize("kind,completed,cleanup,expected", [
    ("RemoteProtocolError", False, False, True),
    ("ReadError", False, False, True),
    ("service_unavailable_error", False, False, True),
    ("RemoteProtocolError", True, False, False),
    ("RemoteProtocolError", False, True, False),
    ("AuthenticationError", False, False, False),
    ("invalid_request_error", False, False, False),
])
def test_subscription_transport_retry_scope(monkeypatch, kind, completed, cleanup, expected):
    async def source(self, request):
        if completed:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
        yield ModelStreamEvent(type=ModelStreamEventType.ERROR, payload={
            "error": "synthetic", "provider_error_type": kind,
            "retryable": False, "stream_cleanup_failed": cleanup,
        })
    monkeypatch.setattr(OpenAISubscriptionProvider, "stream", source)

    async def run():
        provider = RecoveringSubscriptionProvider()
        request = ModelRequest(model="test", messages=[Message.text("user", "hello")])
        events = [e async for e in provider.stream(request)]
        assert events[-1].payload["retryable"] is expected
        assert isinstance(banking_compactor(provider, "test").provider, ReadOnlySummaryProvider)
        await provider.aclose()
    asyncio.run(run())


def test_planning_stall_returns_benchmark_stop_not_infrastructure(monkeypatch):
    from test_runtime_agent import make_agent
    agent, _, _ = make_agent(ScriptedModelProvider([]))
    state = agent.get_init_state([])
    def fail(message):
        raise RuntimeCallbackExecutionError("PlanningStallError")
    monkeypatch.setattr(state, "advance", fail)
    reply, same = agent.generate_next_message(UserMessage.text("continue"), state)
    assert same is state
    assert agent.is_stop(reply)
    assert reply.raw_data["agent_termination"] == "planning_stall"
    assert "cayu_runtime_usage" in reply.raw_data
    assert not reply.tool_calls
    agent.stop(state=state)


def test_provider_failure_is_not_converted_to_agent_stop(monkeypatch):
    from test_runtime_agent import make_agent
    agent, _, _ = make_agent(ScriptedModelProvider([]))
    state = agent.get_init_state([])
    def fail(message):
        raise RuntimeCallbackExecutionError("ModelProviderError")
    monkeypatch.setattr(state, "advance", fail)
    with pytest.raises(RuntimeCallbackExecutionError):
        agent.generate_next_message(UserMessage.text("continue"), state)
    agent.stop(state=state)


@pytest.mark.parametrize("wrapped", [False, True])
def test_single_search_unavailable_is_a_repairable_tool_result(tmp_path, monkeypatch, wrapped):
    from cayu.memory.recall import RecallSourceUnavailable
    from test_runtime_banking import assembly, call, text

    from cayu_tau3_banking.agent_memory import BankingMemorySession
    from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL

    original = BankingMemorySession.invoke
    async def invoke(self, name, arguments, **kwargs):
        if name == KNOWLEDGE_SEARCH_TOOL:
            if wrapped:
                try:
                    raise RecallSourceUnavailable("knowledge", "timeout")
                except RecallSourceUnavailable as error:
                    raise ValueError("safe wrapper") from error
            raise RecallSourceUnavailable("knowledge", "timeout")
        return await original(self, name, arguments, **kwargs)
    monkeypatch.setattr(BankingMemorySession, "invoke", invoke)
    agent, provider, _ = assembly(tmp_path, [
        call(KNOWLEDGE_SEARCH_TOOL, {"query": "purchase limit"}),
        text("I could not retrieve that policy."),
    ], planning_mode="outcomes", advice_mode="lean", controller_mode="advisory")
    state = agent.get_init_state([])
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("What is the purchase limit?"), state)
        assert reply.content == "I could not retrieve that policy."
        assert len(provider.requests) == 2
        assert "Knowledge retrieval is unavailable" in str(provider.requests[-1].messages)
    finally:
        agent.stop(state=state)


def test_subscription_compactor_recovers_interrupted_generation(monkeypatch):
    from uuid import uuid4

    from cayu import AgentSpec
    from cayu.context.base import CompactionRequest
    from cayu.sessions.base import Session

    calls = []
    async def source(self, request):
        calls.append(request)
        if len(calls) == 1:
            yield ModelStreamEvent(type=ModelStreamEventType.ERROR, payload={
                "error": "synthetic", "provider_error_type": "RemoteProtocolError",
                "retryable": False,
            })
        else:
            yield ModelStreamEvent.text_delta("Facts retained.")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
    monkeypatch.setattr(OpenAISubscriptionProvider, "stream", source)

    async def run():
        provider = RecoveringSubscriptionProvider()
        compactor = banking_compactor(provider, "test")
        session = Session(id="summary", agent_name="a", provider_name=provider.name, model="test",
            invocation={"origin": {"trust": "unattributed"}, "root_invocation_id": str(uuid4()),
                        "root_session_id": "summary", "source": "sdk_run"})
        result = await compactor.compact(CompactionRequest(session=session,
            agent=AgentSpec(name="a", model="test"), messages=[Message.text("user", "Facts.")]))
        assert result.summary == "Facts retained."
        assert len(calls) == 2
        assert not any(r.tools for r in calls)
        await provider.aclose()
    asyncio.run(run())


def test_runtime_retries_transport_before_dispatching_local_tools(monkeypatch):
    from cayu import RetryPolicy
    from test_runtime_agent import make_agent
    calls = []
    async def source(self, request):
        calls.append(request)
        if len(calls) == 1:
            yield ModelStreamEvent.tool_call(id="discarded", name="host_action", arguments={"value": 1})
            yield ModelStreamEvent(type=ModelStreamEventType.ERROR, payload={
                "error": "synthetic", "provider_error_type": "RemoteProtocolError",
                "retryable": False,
            })
        else:
            yield ModelStreamEvent.text_delta("Recovered.")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
    monkeypatch.setattr(OpenAISubscriptionProvider, "stream", source)
    provider = RecoveringSubscriptionProvider()
    agent, _, _ = make_agent(provider, retry_policy=RetryPolicy(max_attempts=3))
    state = agent.get_init_state([])
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("hello"), state)
        assert reply.content == "Recovered."
        assert not reply.tool_calls
        assert len(calls) == 2
    finally:
        agent.stop(state=state)
        asyncio.run(provider.aclose())


@pytest.mark.parametrize("kind,reason", [("ExecutionDeadlineExceeded", "agent_time_budget"), ("RuntimeLimitError", "runtime_limit")])
def test_agent_budget_exhaustion_is_graded_only_when_enabled(monkeypatch, kind, reason):
    from test_runtime_agent import make_agent

    from cayu_tau3_banking.runtime_agent import RuntimeLimitError
    agent, _, _ = make_agent(ScriptedModelProvider([]), grade_budget_stops=True)
    state = agent.get_init_state([])
    def fail(message):
        if kind == "RuntimeLimitError":
            raise RuntimeLimitError()
        raise RuntimeCallbackExecutionError(kind)
    monkeypatch.setattr(state, "advance", fail)
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("continue"), state)
        assert agent.is_stop(reply)
        assert reply.raw_data["agent_termination"] == reason
        assert "cayu_runtime_usage" in reply.raw_data
    finally:
        agent.stop(state=state)


def test_real_runtime_step_budget_stops_without_publishing_rejected_draft():
    from test_runtime_agent import RejectFirst, make_agent, text_step
    agent, closed, _ = make_agent(ScriptedModelProvider([text_step("rejected draft")]),
        policies=[RejectFirst()], max_steps=1, grade_budget_stops=True)
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Question."), state)
        assert agent.is_stop(reply)
        assert reply.raw_data["agent_termination"] == "runtime_limit"
        assert "rejected draft" not in reply.content
        assert closed.is_set()
    finally:
        agent.stop(state=state)


@pytest.mark.parametrize("field,kind", [
    ("provider_error_type", "usage_limit_reached"),
    ("provider_error_code", "insufficient_quota"),
    ("provider_error_code", "billing_hard_limit_reached"),
])
def test_runtime_does_not_retry_exhausted_quota(monkeypatch, field, kind):
    from cayu import RetryPolicy
    from test_runtime_agent import make_agent
    calls = []
    async def source(self, request):
        calls.append(request)
        yield ModelStreamEvent(type=ModelStreamEventType.ERROR, payload={
            "error": "synthetic", field: kind, "status_code": 429, "retryable": True,
        })
    monkeypatch.setattr(OpenAISubscriptionProvider, "stream", source)
    provider = RecoveringSubscriptionProvider()
    agent, _, _ = make_agent(provider, retry_policy=RetryPolicy(max_attempts=5))
    state = agent.get_init_state([])
    try:
        with pytest.raises(RuntimeCallbackExecutionError):
            agent.generate_next_message(UserMessage.text("hello"), state)
        assert len(calls) == 1
    finally:
        agent.stop(state=state)
        asyncio.run(provider.aclose())


@pytest.mark.parametrize("kind,status", [("rate_limit_exceeded", 429), ("service_unavailable_error", 503)])
def test_transient_provider_errors_keep_runtime_retry(monkeypatch, kind, status):
    from cayu import RetryPolicy
    from test_runtime_agent import make_agent
    calls = []
    async def source(self, request):
        calls.append(request)
        if len(calls) == 1:
            yield ModelStreamEvent(type=ModelStreamEventType.ERROR, payload={
                "error": "synthetic", "provider_error_type": kind, "status_code": status, "retryable": True,
            })
        else:
            yield ModelStreamEvent.text_delta("Recovered.")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
    monkeypatch.setattr(OpenAISubscriptionProvider, "stream", source)
    provider = RecoveringSubscriptionProvider()
    agent, _, _ = make_agent(provider, retry_policy=RetryPolicy(max_attempts=2))
    state = agent.get_init_state([])
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("hello"), state)
        assert reply.content == "Recovered."
        assert len(calls) == 2
    finally:
        agent.stop(state=state)
        asyncio.run(provider.aclose())


def test_summary_wrapper_does_not_reenable_quota_retry(monkeypatch):
    async def source(self, request):
        yield ModelStreamEvent(type=ModelStreamEventType.ERROR, payload={
            "error": "synthetic", "provider_error_type": "service_unavailable_error",
            "provider_error_code": "insufficient_quota", "retryable": True,
        })
    monkeypatch.setattr(OpenAISubscriptionProvider, "stream", source)
    async def run():
        provider = RecoveringSubscriptionProvider()
        wrapped = ReadOnlySummaryProvider(provider)
        request = ModelRequest(model="test", messages=[Message.text("user", "Summarize.")])
        events = [event async for event in wrapped.stream(request)]
        assert events[-1].payload["retryable"] is False
        assert events[-1].payload["quota_recovery"] == "requires_capacity_or_reset"
        await provider.aclose()
    asyncio.run(run())
