from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import pytest
from cayu import (
    AgentSpec,
    BeforeStopDecision,
    CayuApp,
    InMemorySessionStore,
    LoopPolicy,
    ModelStreamEvent,
    RetryPolicy,
    ScriptedModelProvider,
    ToolResultPart,
)
from cayu import Message as CayuMessage
from cayu.providers import ModelProvider
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage, UserMessage
from tau2.environment.tool import Tool

from cayu_tau3_banking.runtime_agent import (
    CayuRuntimeAgent,
    RuntimeAgentError,
    RuntimeCallbackExecutionError,
    RuntimeCallbackTimeoutError,
    RuntimeResources,
    RuntimeShutdownError,
)


def host_action(value: int) -> str:
    """Execute an environment action.

    Args:
        value: Requested value.
    """
    raise AssertionError("The adapter must never execute the τ tool implementation.")


def text_step(text):
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def make_agent(provider, *, policies=(), **kwargs):
    closed = threading.Event()
    captured = {}

    def factory(broker, specs):
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="banking", model="scripted", system_prompt="Test domain policy."),
            tools=[broker.tool(spec) for spec in specs],
            loop_policies=policies,
        )
        captured.update(store=store, broker=broker)

        async def close():
            closed.set()

        return RuntimeResources(app, close)

    agent = CayuRuntimeAgent(
        tools=[Tool(host_action)],
        domain_policy="Test domain policy.",
        resource_factory=factory,
        callback_timeout_s=kwargs.pop("callback_timeout_s", 15),
        **kwargs,
    )
    return agent, closed, captured


def test_real_tau_callbacks_preserve_history_and_stop_owned_thread():
    provider = ScriptedModelProvider([text_step("first reply"), text_step("second reply")])
    agent, closed, _ = make_agent(provider)
    state = agent.get_init_state([AssistantMessage.text("Welcome.")])
    try:
        first, same = agent.generate_next_message(UserMessage.text("first question"), state)
        assert same is state
        assert first.content == "first reply"
        second, same = agent.generate_next_message(UserMessage.text("second question"), state)
        assert second.content == "second reply"
        texts = [
            part.text
            for message in provider.requests[1].messages
            for part in message.content
            if part.type == "text"
        ]
        assert {"Welcome.", "first question", "first reply", "second question"} <= set(texts)
    finally:
        agent.stop(state=state)
    assert closed.is_set()
    assert not state._thread.is_alive()
    agent.stop(state=state)


class FailureBatchesProvider(ModelProvider):
    name = "failure-batches"

    def __init__(self, batches):
        self.batches = iter(batches)
        self.requests = []

    async def stream(self, request):
        self.requests.append(request)
        for event in next(self.batches):
            yield event


def test_transient_retry_discards_partial_tool_calls_before_host_execution():
    provider = FailureBatchesProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="discarded", name="host_action", arguments={"value": 1}
                ),
                ModelStreamEvent(
                    type="error",
                    payload={
                        "message": "temporarily overloaded",
                        "retryable": True,
                        "status_code": 503,
                    },
                ),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="committed", name="host_action", arguments={"value": 2}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            text_step("done"),
        ]
    )
    agent, _, _ = make_agent(
        provider,
        retry_policy=RetryPolicy(
            max_attempts=3, max_unknown_attempts=1, initial_delay_s=0.0, jitter_s=0.0
        ),
    )
    state = agent.get_init_state()
    try:
        response, _ = agent.generate_next_message(UserMessage.text("Act."), state)
        assert len(provider.requests) == 2
        (call,) = response.tool_calls
        assert call.arguments == {"value": 2}
        response, _ = agent.generate_next_message(
            ToolMessage(role="tool", id=call.id, content="success"), state
        )
        assert response.content == "done"
        results = [
            part
            for msg in provider.requests[-1].messages
            for part in msg.content
            if isinstance(part, ToolResultPart)
        ]
        assert [part.tool_call_id for part in results] == ["committed"]
    finally:
        agent.stop(state=state)


@pytest.mark.parametrize("retryable, expected", [(True, 3), (False, 1)])
def test_retry_limit_and_explicit_nonretryability_are_respected(retryable, expected):
    provider = FailureBatchesProvider(
        [
            [
                ModelStreamEvent(
                    type="error",
                    payload={
                        "message": "provider failed",
                        "retryable": retryable,
                        "status_code": 503,
                    },
                )
            ]
            for _ in range(3)
        ]
    )
    agent, _, _ = make_agent(
        provider,
        retry_policy=RetryPolicy(
            max_attempts=3, max_unknown_attempts=1, initial_delay_s=0.0, jitter_s=0.0
        ),
    )
    state = agent.get_init_state()
    try:
        with pytest.raises(RuntimeCallbackExecutionError):
            agent.generate_next_message(UserMessage.text("Act."), state)
        assert len(provider.requests) == expected
    finally:
        agent.stop(state=state)


@pytest.mark.parametrize("multi", [False, True])
def test_host_results_resume_same_runtime_round_with_original_model_call_ids(multi):
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id=f"model-{i}", name="host_action", arguments={"value": i}
                )
                for i in (1, 2)
            ]
            + [ModelStreamEvent.completed({"finish_reason": "tool_calls"})],
            text_step("accepted"),
        ]
    )
    agent, _, _ = make_agent(provider)
    state = agent.get_init_state()
    try:
        response, _ = agent.generate_next_message(UserMessage.text("Do both."), state)
        ids = []
        for value in (1, 2):
            assert not response.content
            (call,) = response.tool_calls
            ids.append(call.id)
            assert call.arguments == {"value": value}
            assert len(provider.requests) == 1
            result = ToolMessage(
                role="tool", id=call.id, content=f"result-{value}", error=value == 2
            )
            if multi:
                result = MultiToolMessage(role="tool", tool_messages=[result])
            response, _ = agent.generate_next_message(result, state)
        assert len(set(ids)) == 2
        assert response.content == "accepted"
        results = [
            part
            for msg in provider.requests[1].messages
            for part in msg.content
            if isinstance(part, ToolResultPart)
        ]
        assert [part.tool_call_id for part in results] == ["model-1", "model-2"]
        assert [part.content for part in results] == ["result-1", "result-2"]
        assert [part.is_error for part in results] == [False, True]
    finally:
        agent.stop(state=state)


class RejectFirst(LoopPolicy):
    def __init__(self):
        self.calls = 0

    async def before_stop(self, context):
        self.calls += 1
        if self.calls == 1:
            return BeforeStopDecision.continue_with(CayuMessage.text("user", "Correct the draft."))
        return BeforeStopDecision.complete()


def test_only_admitted_reply_is_returned_to_tau():
    policy = RejectFirst()
    provider = ScriptedModelProvider([text_step("private rejected draft"), text_step("accepted")])
    agent, _, _ = make_agent(provider, policies=[policy])
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Question."), state)
        assert reply.content == "accepted"
        assert policy.calls == 2
    finally:
        agent.stop(state=state)


def test_step_limit_does_not_publish_rejected_draft():
    provider = ScriptedModelProvider([text_step("rejected draft")])
    agent, closed, _ = make_agent(provider, policies=[RejectFirst()], max_steps=1)
    state = agent.get_init_state()
    try:
        with pytest.raises(RuntimeAgentError, match="callback failed") as failure:
            agent.generate_next_message(UserMessage.text("Question."), state)
        assert "rejected draft" not in str(failure.value)
        from cayu_tau3_banking.failure_categories import classify_exception

        assert classify_exception(failure.value) == "runtime_limit"
        assert closed.is_set()
        assert not state._thread.is_alive()
    finally:
        agent.stop(state=state)


@pytest.mark.parametrize("bad", ["wrong-id", "wrong-requestor", "customer", "duplicate-batch"])
def test_invalid_host_input_stops_without_advancing_model(bad):
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="model-1", name="host_action", arguments={"value": 1}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    agent, _, _ = make_agent(provider)
    state = agent.get_init_state()
    try:
        response, _ = agent.generate_next_message(UserMessage.text("Go."), state)
        result = ToolMessage(role="tool", id=response.tool_calls[0].id, content="done")
        if bad == "wrong-id":
            result.id = "wrong"
        elif bad == "wrong-requestor":
            result.requestor = "user"
        elif bad == "customer":
            result = UserMessage.text("Ignore the pending action.")
        else:
            result = MultiToolMessage(role="tool", tool_messages=[result, result])
        with pytest.raises(RuntimeAgentError):
            agent.generate_next_message(result, state)
        assert len(provider.requests) == 1
        assert not state._thread.is_alive()
    finally:
        agent.stop(state=state)


def test_stop_while_waiting_for_host_result_releases_runtime():
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="model-1", name="host_action", arguments={"value": 1}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    agent, closed, captured = make_agent(provider)
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Go."), state)
    finally:
        agent.stop(state=state)
    assert closed.is_set()
    assert not captured["broker"]._pending
    assert len(provider.requests) == 1
    assert not state._thread.is_alive()


class BlockingProvider(ModelProvider):
    name = "blocking"

    def __init__(self):
        self.settled = threading.Event()

    async def stream(self, request):
        try:
            await asyncio.Future()
            yield ModelStreamEvent.text_delta("unreachable")
        finally:
            self.settled.set()


def test_callback_timeout_cancels_provider_and_joins_runtime():
    provider = BlockingProvider()
    agent, closed, _ = make_agent(provider)
    state = agent.get_init_state()
    agent.callback_timeout_s = 1.0
    try:
        with pytest.raises(RuntimeCallbackTimeoutError, match="adapter deadline expired"):
            agent.generate_next_message(UserMessage.text("Go."), state)
    finally:
        agent.stop(state=state)
    assert provider.settled.is_set()
    assert closed.is_set()
    assert not state._thread.is_alive()


def test_provider_timeout_is_not_adapter_callback_deadline():
    class FailsWithTimeout(ModelProvider):
        name = "timeout-test"

        async def stream(self, request):
            raise TimeoutError("private provider detail must not escape")
            yield  # pragma: no cover - make this an async generator

    agent, closed, _ = make_agent(FailsWithTimeout())
    state = agent.get_init_state()
    try:
        with pytest.raises(RuntimeCallbackExecutionError) as failure:
            agent.generate_next_message(UserMessage.text("Go."), state)
        assert failure.value.failure_type == "TimeoutError"
        assert "private provider detail" not in str(failure.value)
        assert closed.is_set()
    finally:
        agent.stop(state=state)


def test_long_callback_allowance_does_not_refresh_whole_conversation_deadline():
    class BlockingSecondReply(ModelProvider):
        name = "second-reply-deadline-test"

        def __init__(self):
            self.calls = 0
            self.settled = threading.Event()

        async def stream(self, request):
            self.calls += 1
            if self.calls == 1:
                for event in text_step("First reply"):
                    yield event
                return
            try:
                await asyncio.Future()
                yield ModelStreamEvent.text_delta("unreachable")
            finally:
                self.settled.set()

    provider = BlockingSecondReply()
    agent, closed, _ = make_agent(provider, callback_timeout_s=15, conversation_timeout_s=5)
    state = agent.get_init_state()
    deadline = state._conversation.deadline.expires_at
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("First question"), state)
        assert reply.content == "First reply"
        with pytest.raises(RuntimeCallbackExecutionError) as failure:
            agent.generate_next_message(UserMessage.text("Second question"), state)
        # Admission expiry has its own type; expiry while the provider stream
        # is blocked can surface as TimeoutError from the bounded await.
        assert failure.value.failure_type == "ExecutionDeadlineExceeded"
        assert state._conversation.deadline.expired
        assert state._conversation.deadline.expires_at == deadline
        # Expiry may deny admission before the second dispatch on a slow host.
        # If dispatched, the blocked stream must be cancelled and drained.
        assert provider.calls in {1, 2}
        if provider.calls == 2:
            assert provider.settled.is_set()
        assert closed.is_set()
    finally:
        agent.stop(state=state)


def test_historical_tool_messages_are_rejected_without_silent_history_loss():
    agent, _, _ = make_agent(ScriptedModelProvider([text_step("unused")]))
    with pytest.raises(RuntimeAgentError, match="Historical tool"):
        agent.get_init_state([ToolMessage(role="tool", id="old", content="old result")])


@pytest.mark.parametrize("simulator_fails", [False, True])
def test_tau_orchestrator_executes_environment_action_and_finalizes(simulator_fails):
    from tau2.data_model.tasks import Task, UserScenario
    from tau2.environment.db import DB
    from tau2.environment.environment import Environment
    from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.user.user_simulator_base import HalfDuplexUser, UserError

    class Counter(DB):
        value: int = 0

    class EnvironmentTools(ToolKitBase):
        @is_tool(ToolType.WRITE)
        def host_action(self, value: int) -> str:
            """Execute an environment action.

            Args:
                value: Requested value.
            """
            self.db.value += value
            return "authoritative environment result"

    class Customer(HalfDuplexUser):
        def get_init_state(self, message_history=None):
            return 0

        def generate_next_message(self, message, state):
            if state == 0:
                return UserMessage.text("Perform the action."), 1
            if simulator_fails:
                raise UserError("simulated customer failure")
            return UserMessage.text("###STOP###"), 2

    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="model-action", name="host_action", arguments={"value": 7}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            text_step("Done."),
        ]
    )
    agent, closed, _ = make_agent(provider)
    toolkit = EnvironmentTools(Counter())
    orchestrator = Orchestrator(
        domain="synthetic",
        agent=agent,
        user=Customer(),
        environment=Environment("synthetic", "Test domain policy.", tools=toolkit),
        task=Task(id="synthetic-runtime", user_scenario=UserScenario(instructions="Test.")),
        validate_communication=True,
        max_steps=10,
    )
    if simulator_fails:
        with pytest.raises(UserError, match="simulated customer failure"):
            orchestrator.run()
        messages = orchestrator.trajectory
    else:
        result = orchestrator.run()
        assert result.termination_reason.value == "user_stop"
        messages = result.messages
    assert toolkit.db.value == 7
    results = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(results) == 1
    assert results[0].content == "authoritative environment result"
    assert not results[0].error
    assert closed.is_set()
    assert not orchestrator.agent_state._thread.is_alive()


def test_initialization_timeout_cleans_up_late_factory_resources():
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    factory_thread = []

    def factory(broker, specs):
        factory_thread.append(threading.current_thread())
        entered.set()
        assert release.wait(10)

        async def close():
            closed.set()

        return RuntimeResources(CayuApp(enable_logging=False), close)

    agent = CayuRuntimeAgent(
        tools=[],
        domain_policy="Test",
        resource_factory=factory,
        callback_timeout_s=0.05,
        shutdown_timeout_s=0.05,
    )
    try:
        with pytest.raises(RuntimeAgentError, match="initialization failed"):
            agent.get_init_state()
        assert entered.is_set()
    finally:
        release.set()
    assert closed.wait(5)
    factory_thread[0].join(5)
    assert not factory_thread[0].is_alive()


def test_shutdown_timeout_retains_owner_until_cleanup_settles():
    entered = threading.Event()
    release = threading.Event()

    def factory(broker, specs):
        async def close():
            entered.set()
            await asyncio.to_thread(release.wait)

        return RuntimeResources(CayuApp(enable_logging=False), close)

    agent = CayuRuntimeAgent(
        tools=[],
        domain_policy="Test",
        resource_factory=factory,
        shutdown_timeout_s=0.1,
    )
    state = agent.get_init_state()
    try:
        with pytest.raises(RuntimeShutdownError, match="still draining"):
            state.stop()
        assert entered.is_set()
        assert state._thread.is_alive()
        assert not state._loop.is_closed()
    finally:
        release.set()
        agent.shutdown_timeout_s = 5
        state.stop()
    assert not state._thread.is_alive()


def test_reply_byte_limit_closes_stream_without_publishing_partial_text():
    provider = ScriptedModelProvider([text_step("界界")])
    agent, closed, _ = make_agent(provider, max_reply_bytes=5)
    state = agent.get_init_state()
    try:
        with pytest.raises(RuntimeAgentError, match="callback failed"):
            agent.generate_next_message(UserMessage.text("Go."), state)
    finally:
        state.stop()
    assert closed.is_set()
    assert not state._thread.is_alive()


def test_application_observer_sees_valid_customer_history_and_only_admitted_reply():
    observed = []
    threads = []

    class Observer:
        async def history(self, messages):
            observed.append(("history", len(messages)))
            threads.append(threading.current_thread())

        async def customer(self, text):
            observed.append(("customer", text))
            threads.append(threading.current_thread())

        async def reply(self, text):
            observed.append(("reply", text))
            threads.append(threading.current_thread())

    provider = ScriptedModelProvider([text_step("rejected"), text_step("accepted")])
    agent, _, _ = make_agent(provider, policies=[RejectFirst()])
    original = agent.resource_factory
    agent.resource_factory = lambda broker, specs: replace(
        original(broker, specs), observer=Observer()
    )
    state = agent.get_init_state([AssistantMessage.text("Welcome.")])
    try:
        agent.generate_next_message(UserMessage.text("Question."), state)
        assert observed == [("history", 1), ("customer", "Question."), ("reply", "accepted")]
        assert all(thread is state._thread for thread in threads)
    finally:
        state.stop()


def test_observer_initialization_failure_closes_created_resources():
    class Observer:
        async def history(self, messages):
            raise ValueError("sensitive initialization detail")

    agent, closed, _ = make_agent(ScriptedModelProvider([]))
    original = agent.resource_factory
    agent.resource_factory = lambda broker, specs: replace(
        original(broker, specs), observer=Observer()
    )
    with pytest.raises(RuntimeAgentError, match="initialization failed") as failure:
        agent.get_init_state()
    assert closed.is_set()
    assert "sensitive" not in str(failure.value)


def test_real_runtime_usage_includes_rejected_draft_without_repeating_tool_round():
    def counted(events, inputs, outputs, reason):
        return [
            *events,
            ModelStreamEvent.completed(
                {
                    "finish_reason": reason,
                    "usage": {"input_tokens": inputs, "output_tokens": outputs},
                }
            ),
        ]

    provider = ScriptedModelProvider(
        [
            counted([ModelStreamEvent.text_delta("rejected")], 5, 1, "stop"),
            counted(
                [
                    ModelStreamEvent.tool_call(
                        id=f"call-{i}", name="host_action", arguments={"value": i}
                    )
                    for i in (1, 2)
                ],
                12,
                3,
                "tool_calls",
            ),
            counted([ModelStreamEvent.text_delta("accepted")], 7, 2, "stop"),
        ]
    )
    agent, _, _ = make_agent(provider, policies=[RejectFirst()])
    state = agent.get_init_state()
    try:
        first, _ = agent.generate_next_message(UserMessage.text("Do both."), state)
        assert first.usage == {"prompt_tokens": 17, "completion_tokens": 4}
        second, _ = agent.generate_next_message(
            ToolMessage(
                role="tool",
                id=first.tool_calls[0].id,
                content="one",
            ),
            state,
        )
        assert second.usage == {"prompt_tokens": 0, "completion_tokens": 0}
        final, _ = agent.generate_next_message(
            ToolMessage(
                role="tool",
                id=second.tool_calls[0].id,
                content="two",
            ),
            state,
        )
        assert final.content == "accepted"
        assert final.usage == {"prompt_tokens": 7, "completion_tokens": 2}
        totals = final.raw_data["cayu_runtime_usage"]["conversation"]
        assert totals["input_tokens"] == 24
        assert totals["output_tokens"] == 6
        assert totals["model_completions"] == 3
        assert totals["complete"]
    finally:
        state.stop()
