"""Live host tool transport for agents running inside ``CayuApp``.

The runtime invokes native tools normally. Their execution waits for an explicit
host result, allowing an external environment to own effects without owning the
model loop. This transport is process-local: it neither releases the runtime's
invocation nor promises restart recovery or exactly-once external effects.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from jsonschema import validators
from referencing import Registry


class HostToolProtocolError(RuntimeError):
    """A host attempted an invalid or stale live-transport operation."""


@dataclass(frozen=True)
class HostToolCall:
    """One private host request. Do not expose arguments in public telemetry."""

    session_id: str
    request_id: str
    tool_name: str
    idempotency_key: str
    _arguments_json: str = field(repr=False)

    @property
    def arguments(self) -> dict[str, Any]:
        return json.loads(self._arguments_json)


@dataclass
class _PendingCall:
    call: HostToolCall
    result: asyncio.Future[ToolResult]
    queued: bool = True
    delivered: bool = False


class HostToolBroker:
    """Bounded, single-event-loop rendezvous between native tools and their host.

    Create one broker per conversation. Register ``broker.tool(spec)`` on its
    Cayu agent, drain ``app.run`` in a task, then consume ``next_call`` and submit
    matching results. Both broker and runtime task must remain alive between
    host turns. Cancel and join the runtime task when stopping a conversation.

    Result submissions are single-use. Duplicates, expired calls and calls from
    another broker fail closed; no retry here can reexecute a host action.
    """

    def __init__(self, *, max_pending: int = 16) -> None:
        if type(max_pending) is not int or not 1 <= max_pending <= 256:
            raise ValueError("max_pending must be an integer between 1 and 256.")
        self._max_pending = max_pending
        self._pending: dict[str, _PendingCall] = {}
        self._queued: deque[str] = deque()
        self._available = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._receiving = False

    def _bind_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise HostToolProtocolError("Host tool broker used from another event loop.")

    def tool(self, spec: ToolSpec) -> Tool:
        """Build an ordinary Cayu tool with eager local argument validation."""
        if type(spec) is not ToolSpec:
            raise TypeError("Host tools require a ToolSpec.")
        if spec.workspace_mutation:
            raise ValueError("Host tools cannot claim a Cayu workspace mutation.")
        return _HostTool(self, spec)

    async def _execute(
        self, spec: ToolSpec, ctx: ToolContext, arguments: dict[str, Any]
    ) -> ToolResult:
        self._bind_loop()
        if self._closed:
            raise HostToolProtocolError("Host tool broker is closed.")
        if not ctx.idempotency_key:
            raise HostToolProtocolError("Host tool requires a runtime idempotency key.")
        if len(self._pending) >= self._max_pending:
            raise HostToolProtocolError("Host tool broker capacity exceeded.")
        call = HostToolCall(
            session_id=ctx.session_id,
            request_id=uuid4().hex,
            tool_name=spec.name,
            idempotency_key=ctx.idempotency_key,
            _arguments_json=json.dumps(arguments, allow_nan=False, ensure_ascii=True),
        )
        pending = _PendingCall(call, asyncio.get_running_loop().create_future())
        self._pending[call.request_id] = pending
        self._queued.append(call.request_id)
        self._available.set()
        try:
            return await pending.result
        finally:
            self._pending.pop(call.request_id, None)
            if pending.queued:
                self._queued.remove(call.request_id)
            if not self._queued and not self._closed:
                self._available.clear()

    async def next_call(self) -> HostToolCall:
        """Wait for one call; cancelling this wait never cancels execution."""
        self._bind_loop()
        if self._receiving:
            raise HostToolProtocolError("Only one host receiver may wait at a time.")
        self._receiving = True
        try:
            while True:
                if self._closed:
                    raise HostToolProtocolError("Host tool broker is closed.")
                if self._queued:
                    request_id = self._queued.popleft()
                    if not self._queued:
                        self._available.clear()
                    pending = self._pending[request_id]
                    pending.queued = False
                    # Task.cancel() cancels its awaited Future immediately, but
                    # the tool coroutine's finally block may not have run yet.
                    # Never hand that already-cancelled action to the host.
                    if pending.result.done():
                        continue
                    pending.delivered = True
                    return pending.call
                await self._available.wait()
        finally:
            self._receiving = False

    def resolve(
        self, *, session_id: str, request_id: str, tool_name: str, result: ToolResult
    ) -> None:
        """Submit one host result to the exact delivered, still-active call."""
        self._bind_loop()
        pending = self._pending.get(request_id)
        if (
            self._closed
            or pending is None
            or not pending.delivered
            or pending.result.done()
            or session_id != pending.call.session_id
            or tool_name != pending.call.tool_name
        ):
            raise HostToolProtocolError("Result does not match an active delivered host call.")
        if type(result) is not ToolResult:
            raise TypeError("Host results must be ToolResult instances.")
        # Revalidate at the boundary, including model_construct/model_copy inputs.
        owned_result = ToolResult.model_validate(result.model_dump(mode="python", warnings=False))
        pending.result.set_result(owned_result)

    def close(self) -> None:
        """Cancel unresolved waits; this cannot undo an already-issued host action."""
        self._bind_loop()
        self._closed = True
        for pending in self._pending.values():
            pending.result.cancel()
        self._available.set()


class _HostTool(Tool):
    def __init__(self, broker: HostToolBroker, spec: ToolSpec) -> None:
        # Sequential host calls also permit a half-duplex host to return one
        # result at a time when a model requests a mixed local/host tool round.
        super().__init__(spec.model_copy(update={"parallel_safe": False}))
        self._broker = broker
        validator_type = validators.validator_for(self.schema)
        validator_type.check_schema(self.schema)
        # No network or filesystem reference retrieval from tool schemas.
        self._validator = validator_type(self.schema, registry=Registry())

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if not self._validator.is_valid(args):
            return ToolResult(
                content="Tool arguments do not match the declared schema.",
                structured={"error": "invalid_arguments"},
                is_error=True,
            )
        return await self._broker._execute(self.spec, ctx, args)
