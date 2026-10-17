"""Opt-in τ half-duplex adapter whose model loop is owned by CayuApp.

This module contains protocol and live-process lifecycle glue, not banking
policy. A resource factory registers native tools, policies and context on Cayu.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import math
import threading
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from cayu import (
    CayuApp,
    EventType,
    ExecutionDeadline,
    ResumeRequest,
    RetryPolicy,
    RunLimits,
    RunRequest,
)
from cayu import Message as CayuMessage
from cayu.failure_evidence import FailureEvidence, exception_evidence
from cayu.tools.base import ToolResult, ToolSpec
from pydantic import ValidationError
from tau2.agent.base_agent import AgentError, HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

from cayu_tau3_banking.runtime_tools import HostToolBroker, HostToolCall
from cayu_tau3_banking.runtime_usage import RuntimeUsageAccounting


class RuntimeAgentError(AgentError):
    """Content-free failure crossing the synchronous host boundary."""


class RuntimeShutdownError(RuntimeAgentError):
    """Cleanup did not settle in time; the owner is retained for another stop."""


class RuntimeCallbackTimeoutError(RuntimeAgentError):
    """The synchronous adapter's own callback deadline expired."""


class RuntimeCallbackExecutionError(RuntimeAgentError):
    """Execution failed; retain the exception type without its potentially private text."""

    def __init__(self, failure_type: str, *, evidence: FailureEvidence | None = None):
        self.failure_type = failure_type
        self.failure_evidence = evidence
        super().__init__(f"Runtime callback failed ({failure_type}); conversation stopped.")


class RuntimeLimitError(RuntimeCallbackExecutionError):
    """An authoritative runtime limit event, not an unexplained runtime crash."""

    def __init__(self):
        super().__init__("RuntimeLimitReached")


@dataclass(frozen=True)
class RuntimeResources:
    app: CayuApp
    # Factories own provider/store lifetime and supply cleanup when needed.
    aclose: Callable[[], Awaitable[None]] | None = None
    observer: RuntimeConversationObserver | None = None
    untracked_model_helpers: bool = False
    usage_accounting: RuntimeUsageAccounting | None = None
    diagnostic_snapshot: Callable[[], dict[str, Any]] | None = None


class RuntimeConversationObserver(Protocol):
    """Application projections notified only at validated conversation boundaries.

    These callbacks run on the conversation loop. Tool execution observations
    belong in Cayu runtime hooks, not in this transport-facing interface.
    """

    async def history(self, messages: tuple[CayuMessage, ...]) -> None: ...

    async def customer(self, text: str) -> None: ...

    async def reply(self, text: str) -> None: ...


ResourceFactory = Callable[
    [HostToolBroker, tuple[ToolSpec, ...]], RuntimeResources | Awaitable[RuntimeResources]
]


def _duration(value: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return float(value)


def _text(message: UserMessage | AssistantMessage) -> str:
    if message.is_audio or message.audio_content or message.tool_calls:
        raise RuntimeAgentError("Only text customer history is supported by this adapter slice.")
    if not isinstance(message.content, str) or not message.content.strip():
        raise RuntimeAgentError("A nonblank text message is required.")
    return message.content


class _Conversation:
    """All mutable protocol/runtime state is confined to the owned event loop."""

    def __init__(
        self,
        agent: CayuRuntimeAgent,
        history: tuple[CayuMessage, ...],
        broker: HostToolBroker,
        resources: RuntimeResources,
    ):
        self.agent = agent
        self.session_id = uuid4().hex
        self.broker = broker
        self.resources = resources
        if type(self.resources) is not RuntimeResources or not isinstance(
            self.resources.app, CayuApp
        ):
            raise TypeError("resource_factory must return RuntimeResources with a CayuApp.")
        self.history = history
        self.usage = self.resources.usage_accounting or RuntimeUsageAccounting(
            untracked_helpers=self.resources.untracked_model_helpers
        )
        self.started = False
        self.closed = False
        self.pending: HostToolCall | None = None
        self.run_task: asyncio.Task[str] | None = None
        self.advance_task: asyncio.Task[Any] | None = None
        self.terminal_evidence: FailureEvidence | None = None
        # One whole-conversation finish-by deadline, never refreshed per callback.
        self.deadline = ExecutionDeadline.after(
            agent.conversation_timeout_s, scope="tau_conversation"
        )

    async def advance(self, message: ValidAgentInputMessage) -> AssistantMessage:
        if self.closed:
            raise RuntimeAgentError("Runtime conversation is closed.")
        self.advance_task = asyncio.current_task()
        try:
            if isinstance(message, UserMessage):
                if self.pending is not None or (
                    self.run_task is not None and not self.run_task.done()
                ):
                    raise RuntimeAgentError(
                        "Customer input cannot replace an unresolved tool result."
                    )
                inputs = [CayuMessage.text("user", _text(message))]
                if self.resources.observer is not None:
                    await self.resources.observer.customer(_text(message))
                common = dict(
                    session_id=self.session_id,
                    max_steps=self.agent.max_steps,
                    metadata={"tau_runtime_max_steps": self.agent.max_steps},
                    limits=RunLimits(max_tool_calls=self.agent.max_tool_calls, scope="session"),
                    retry_policy=self.agent.retry_policy,
                )
                if not self.started:
                    request = RunRequest(
                        agent_name=self.agent.runtime_agent_name,
                        messages=[*self.history, *inputs],
                        execution_deadline=self.deadline,
                        **common,
                    )
                    stream = self.resources.app.run(request)
                    self.started = True
                else:
                    stream = self.resources.app.resume(ResumeRequest(messages=inputs, **common))
                self.run_task = asyncio.create_task(self._drain(stream))
            else:
                self._settle(message)
            return await self._next_output()
        except Exception as error:
            # A bounded await may raise plain TimeoutError after publishing the
            # authoritative interruption. Preserve that typed deadline rather
            # than guessing from the exception's class or elapsed wall time.
            evidence = self.terminal_evidence or exception_evidence(error)
            if evidence.classification == "deadline" and evidence.deadline is not None:
                raise RuntimeCallbackExecutionError("ExecutionDeadlineExceeded") from None
            raise
        finally:
            self.advance_task = None

    def _settle(self, message: ToolMessage | MultiToolMessage) -> None:
        if isinstance(message, MultiToolMessage):
            if message.role != "tool" or len(message.tool_messages) != 1:
                raise RuntimeAgentError("Expected exactly one host tool result.")
            message = message.tool_messages[0]
        call = self.pending
        if (
            not isinstance(message, ToolMessage)
            or message.role != "tool"
            or message.requestor != "assistant"
            or call is None
            or message.id != call.request_id
            or not isinstance(message.content, str)
        ):
            raise RuntimeAgentError("Result does not match the pending host call.")
        self.broker.resolve(
            session_id=call.session_id,
            request_id=call.request_id,
            tool_name=call.tool_name,
            result=ToolResult(content=message.content, is_error=message.error),
        )
        self.pending = None

    async def _next_output(self) -> AssistantMessage:
        if self.run_task is None:
            raise RuntimeAgentError("The conversation has not started.")
        receiver = asyncio.create_task(self.broker.next_call())
        try:
            done, _ = await asyncio.wait(
                (self.run_task, receiver), return_when=asyncio.FIRST_COMPLETED
            )
            # A terminal failure wins a race with delivery of a now-invalid call.
            if self.run_task in done:
                text = self.run_task.result()
                if receiver in done:
                    raise RuntimeAgentError("Runtime completed with an undelivered host action.")
                if self.resources.observer is not None:
                    await self.resources.observer.reply(text)
                return self._attach_diagnostics(AssistantMessage.text(text))
            call = receiver.result()
            self.pending = call
            return self._attach_diagnostics(
                AssistantMessage.text(
                    "",
                    tool_calls=[
                        ToolCall(
                            id=call.request_id,
                            name=call.tool_name,
                            arguments=call.arguments,
                        )
                    ],
                )
            )
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)

    def _attach_diagnostics(self, reply: AssistantMessage) -> AssistantMessage:
        if self.resources.diagnostic_snapshot is not None:
            reply.raw_data = self.resources.diagnostic_snapshot()
        return self.usage.attach(reply)

    async def _drain(self, stream) -> str:
        self.terminal_evidence = None
        text: list[str] = []
        text_bytes = 0
        completed = False
        terminal_failure = None
        async with aclosing(stream):
            async for event in stream:
                self.usage.observe(event)
                if (
                    self.agent.diagnostic_stop_on_tool_error
                    and event.type == EventType.TOOL_CALL_FAILED
                ):
                    raise RuntimeAgentError(
                        "Diagnostic run stopped at the first failed tool result."
                    )
                if event.type == EventType.MODEL_STARTED:
                    text = []
                    text_bytes = 0
                elif event.type == EventType.MODEL_TEXT_DELTA:
                    delta = event.payload.get("delta")
                    if isinstance(delta, str):
                        text_bytes += len(delta.encode("utf-8"))
                        if text_bytes > self.agent.max_reply_bytes:
                            raise RuntimeAgentError("Runtime reply exceeds the adapter byte limit.")
                        text.append(delta)
                elif event.type == "tool.effect.outcome_unknown":
                    self.terminal_evidence = FailureEvidence.model_validate(
                        event.payload.get("failure_evidence", {})
                    )
                elif event.type == EventType.SESSION_LIMIT_REACHED:
                    completed = False
                    terminal_failure = RuntimeLimitError()
                elif event.type == EventType.SESSION_COMPLETED:
                    completed = terminal_failure is None
                elif event.type in (EventType.SESSION_FAILED, EventType.SESSION_INTERRUPTED):
                    completed = False
                    try:
                        evidence = FailureEvidence.model_validate(
                            event.payload.get("failure_evidence")
                        )
                    except ValidationError:
                        evidence = self.terminal_evidence or FailureEvidence()
                    self.terminal_evidence = evidence
                    if evidence.exception_types or terminal_failure is None:
                        terminal_failure = RuntimeCallbackExecutionError(
                            "RecallSourceUnavailable"
                            if "RecallSourceUnavailable" in evidence.exception_types
                            else evidence.exception_types[0]
                            if evidence.exception_types
                            else "RuntimeTerminalFailure",
                            evidence=evidence,
                        )
        answer = "".join(text)
        if not completed or not answer.strip():
            if terminal_failure is not None:
                raise terminal_failure
            raise RuntimeAgentError("Runtime did not produce an accepted text reply.")
        return answer

    async def shutdown(self) -> None:
        self.closed = True
        tasks = [t for t in (self.advance_task, self.run_task) if t is not None]
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        self.broker.close()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.resources.aclose is not None:
            await self.resources.aclose()


class RuntimeAgentState:
    """Live, nonserializable handle. Stop it explicitly; never replay it after restart."""

    def __init__(self, agent: CayuRuntimeAgent, history: tuple[CayuMessage, ...]):
        self.agent = agent
        self._call_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._ready = threading.Event()
        self._closed = False
        self._shutdown_future: concurrent.futures.Future | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name=f"cayu-tau-runtime-{uuid4().hex[:8]}", daemon=True
        )
        self._thread.start()
        self._ready.wait()

        async def create():
            broker = HostToolBroker()
            resources = agent.resource_factory(broker, agent.host_specs)
            if inspect.isawaitable(resources):
                resources = await resources
            conversation = _Conversation(agent, history, broker, resources)
            try:
                if conversation.resources.observer is not None:
                    await conversation.resources.observer.history(history)
            except BaseException:
                await conversation.shutdown()
                raise
            return conversation

        creation = asyncio.run_coroutine_threadsafe(create(), self._loop)
        try:
            self._conversation = creation.result(agent.callback_timeout_s)
        except BaseException:
            # A synchronous factory cannot be forcefully interrupted. Retain
            # ownership until it returns, then close even late-created resources.
            def cleanup_late_result(future):
                async def cleanup():
                    try:
                        conversation = future.result()
                        await conversation.shutdown()
                    except BaseException:
                        pass  # Initialization has already failed at the host boundary.
                    finally:
                        self._loop.stop()

                asyncio.run_coroutine_threadsafe(cleanup(), self._loop)

            creation.add_done_callback(cleanup_late_result)
            self._thread.join(agent.shutdown_timeout_s)
            raise RuntimeAgentError("Runtime conversation initialization failed.") from None

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.run_until_complete(self._loop.shutdown_default_executor(timeout=30))
            self._loop.close()

    def advance(self, message: ValidAgentInputMessage) -> AssistantMessage:
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeAgentError("Concurrent callbacks for one conversation are not supported.")
        try:
            with self._lifecycle_lock:
                if self._closed:
                    raise RuntimeAgentError("Runtime conversation is closed.")
                future = asyncio.run_coroutine_threadsafe(
                    self._conversation.advance(message), self._loop
                )
            try:
                return future.result(self.agent.callback_timeout_s)
            except (KeyboardInterrupt, SystemExit):
                self.stop()
                raise
            except Exception as exc:
                # TimeoutError from the coroutine is not a Future.result deadline.
                callback_timeout = isinstance(exc, concurrent.futures.TimeoutError) and not (
                    future.done() and not future.cancelled() and future.exception() is exc
                )
                self.stop()
                if isinstance(exc, RuntimeCallbackExecutionError):
                    raise exc from None
                if callback_timeout:
                    raise RuntimeCallbackTimeoutError(
                        "Runtime callback failed: adapter deadline expired; conversation stopped."
                    ) from None
                raise RuntimeCallbackExecutionError(type(exc).__name__) from None
        finally:
            self._call_lock.release()

    def stop(self) -> None:
        if threading.current_thread() is self._thread:
            raise RuntimeShutdownError("Synchronous stop cannot run on the runtime event loop.")
        with self._stop_lock:
            self._stop()

    def _stop(self) -> None:
        with self._lifecycle_lock:
            if not self._thread.is_alive():
                return
            self._closed = True
            if self._shutdown_future is None:
                self._shutdown_future = asyncio.run_coroutine_threadsafe(
                    self._conversation.shutdown(), self._loop
                )
            future = self._shutdown_future
        try:
            future.result(self.agent.shutdown_timeout_s)
        except concurrent.futures.TimeoutError:
            # Do not close a loop whose owned tasks are still settling.
            raise RuntimeShutdownError(
                "Runtime cleanup is still draining; retry stop after settlement."
            ) from None
        except Exception:
            raise RuntimeShutdownError("Runtime resource cleanup failed.") from None
        finally:
            if future.done():
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(self.agent.shutdown_timeout_s)
        if self._thread.is_alive():
            raise RuntimeShutdownError("Runtime thread did not stop.")


class CayuRuntimeAgent(HalfDuplexAgent[RuntimeAgentState]):
    """τ protocol adapter; the factory configures a real Cayu agent per session.

    Not registered as the benchmark default. Text-only completed history is
    supported; historical tool trajectories require a separate import contract.
    The factory must include domain_policy in its registered agent instructions.
    """

    def __init__(
        self,
        *,
        tools: list[Tool],
        domain_policy: str,
        resource_factory: ResourceFactory,
        runtime_agent_name: str = "banking",
        callback_timeout_s: float = 600,
        conversation_timeout_s: float = 3600,
        shutdown_timeout_s: float = 30,
        grade_budget_stops: bool = False,
        max_steps: int = 64,
        max_tool_calls: int = 512,
        max_reply_bytes: int = 1024 * 1024,
        diagnostic_stop_on_tool_error: bool = False,
        retry_policy: RetryPolicy | None = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.resource_factory = resource_factory
        if retry_policy is not None and not isinstance(retry_policy, RetryPolicy):
            raise TypeError("retry_policy must be a Cayu RetryPolicy")
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=1, max_unknown_attempts=1)
        if type(diagnostic_stop_on_tool_error) is not bool:
            raise TypeError("diagnostic_stop_on_tool_error must be a boolean")
        self.diagnostic_stop_on_tool_error = diagnostic_stop_on_tool_error
        self.runtime_agent_name = runtime_agent_name
        self.grade_budget_stops = grade_budget_stops
        self.callback_timeout_s = _duration(callback_timeout_s, "callback_timeout_s")
        self.conversation_timeout_s = _duration(conversation_timeout_s, "conversation_timeout_s")
        self.shutdown_timeout_s = _duration(shutdown_timeout_s, "shutdown_timeout_s")
        for name, value in (
            ("max_steps", max_steps),
            ("max_tool_calls", max_tool_calls),
            ("max_reply_bytes", max_reply_bytes),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.max_steps, self.max_tool_calls, self.max_reply_bytes = (
            max_steps,
            max_tool_calls,
            max_reply_bytes,
        )
        self.host_specs = tuple(
            ToolSpec(
                name=tool.name,
                description=tool.openai_schema["function"].get("description", ""),
                input_schema=tool.openai_schema["function"]["parameters"],
            )
            for tool in tools
        )

    def get_init_state(self, message_history: list[Message] | None = None) -> RuntimeAgentState:
        history = []
        for message in message_history or ():
            if not isinstance(message, UserMessage | AssistantMessage):
                raise RuntimeAgentError(
                    "Historical tool trajectories are not supported by this adapter slice."
                )
            history.append(CayuMessage.text(message.role, _text(message)))
        return RuntimeAgentState(self, tuple(history))

    def generate_next_message(self, message: ValidAgentInputMessage, state: RuntimeAgentState):
        if type(state) is not RuntimeAgentState or state.agent is not self:
            raise RuntimeAgentError("State belongs to another runtime agent.")
        # Detach mutable τ envelopes before crossing onto the owned loop.
        if not isinstance(message, UserMessage | ToolMessage | MultiToolMessage):
            raise RuntimeAgentError("Unsupported runtime input message.")
        try:
            message = type(message).model_validate(message.model_dump())
        except Exception:
            state.stop()
            raise RuntimeAgentError(
                "Invalid runtime input envelope; conversation stopped."
            ) from None
        try:
            return state.advance(message), state
        except (RuntimeCallbackExecutionError, RuntimeLimitError) as error:
            reason = "planning_stall" if getattr(error, "failure_type", None) == "PlanningStallError" else None
            if self.grade_budget_stops:
                if isinstance(error, RuntimeLimitError) or getattr(error, "failure_type", None) == "RuntimeLimitError":
                    reason = "runtime_limit"
                elif error.failure_type == "ExecutionDeadlineExceeded":
                    reason = "agent_time_budget"
                elif error.failure_type == "AdmissionStallError":
                    reason = "admission_stall"
            if reason is None:
                raise
            reply = AssistantMessage.text(
                "I could not complete the remaining work within my execution budget. ###STOP###"
            )
            reply.raw_data = {"agent_termination": reason}
            return state._conversation.usage.attach(reply), state

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        return isinstance(message.raw_data, dict) and message.raw_data.get("agent_termination") in {"planning_stall", "runtime_limit", "agent_time_budget", "admission_stall"}

    def stop(self, message=None, state=None) -> None:
        if state is not None:
            if type(state) is not RuntimeAgentState or state.agent is not self:
                raise RuntimeAgentError("State belongs to another runtime agent.")
            state.stop()
