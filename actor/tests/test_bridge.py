from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from cayu import MessageRole, ProviderStatePart, TextPart, ToolCallPart, ToolResultPart
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamDeadlineError,
    ModelStreamEvent,
    ProviderDeadlineKind,
    ProviderStreamDeadlineEvidence,
)
from cayu.providers.openai import build_openai_payload
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.domains.banking_knowledge.environment import get_environment
from tau2.environment.tool import as_tool

from cayu_tau3_banking.agent import CayuBridgeAgent
from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.attempt_diagnostics import AttemptDiagnosticsRecorder
from cayu_tau3_banking.bridge import (
    BridgeProtocolError,
    CayuModelGateway,
    CayuTauBridge,
    CompositeExternalToolCallGate,
    ExternalToolAdmission,
    InternalRemediationResolution,
    InternalToolRemediation,
    InternalToolResult,
    TerminalOutputAdmission,
)
from cayu_tau3_banking.controller import (
    HUMAN_TRANSFER_TOOL,
    SYNC_WORK_PLAN_TOOL,
    UPDATE_WORK_PLAN_TOOL,
    BankingWorkflowController,
)
from cayu_tau3_banking.credit_card_dispute import (
    AUDIT_CREDIT_CARD_DISPUTES_TOOL,
    PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL,
)
from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL, BankingKnowledgeCorpus
from cayu_tau3_banking.ledger import ConversationLedger
from cayu_tau3_banking.product_navigation import NAVIGATION_TOOLS
from cayu_tau3_banking.reward_audit import (
    AUDIT_CREDIT_CARD_REWARDS_TOOL,
    PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL,
)
from cayu_tau3_banking.savings_interest_audit import (
    AUDIT_SAVINGS_INTEREST_TOOL,
    PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
)

_INTERNAL_TEST_TOOL = "controller_test_plan"


def lookup_account(account_id: str) -> dict[str, str]:
    """Look up one account.

    Args:
        account_id: Account identifier.

    Returns:
        The matching account.
    """

    return {"account_id": account_id}


def transfer_to_human_agents(reason: str, summary: str) -> dict[str, str]:
    """Transfer the customer to a human agent.

    Args:
        reason: Public transfer reason code.
        summary: Transfer context for the human agent.

    Returns:
        The accepted transfer request.
    """

    return {"reason": reason, "summary": summary}


def call_discoverable_agent_tool(agent_tool_name: str, arguments: str) -> dict[str, str]:
    """Call one previously unlocked agent tool.

    Args:
        agent_tool_name: Unlocked tool name.
        arguments: JSON-serialized arguments for the unlocked tool.

    Returns:
        The executed call details.
    """

    return {"agent_tool_name": agent_tool_name, "arguments": arguments}


def give_discoverable_user_tool(
    discoverable_tool_name: str, arguments: str = "{}"
) -> dict[str, str]:
    """Give the customer one discoverable tool.

    Args:
        discoverable_tool_name: Discoverable tool name.
        arguments: Optional JSON defaults for the later customer invocation.

    Returns:
        The delivered tool details.
    """

    return {"discoverable_tool_name": discoverable_tool_name, "arguments": arguments}


def forward_arguments(arguments: str) -> dict[str, str]:
    """Forward an opaque string without interpreting it."""

    return {"arguments": arguments}


class ScriptedProvider(ModelProvider):
    name = "scripted"

    def __init__(self, scripts: list[list[ModelStreamEvent]]) -> None:
        self.scripts = list(scripts)
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if not self.scripts:
            raise AssertionError("Unexpected provider request.")
        for event in self.scripts.pop(0):
            yield event


def completed(*, end_turn: bool | None = None, provider_state=None) -> ModelStreamEvent:
    return ModelStreamEvent.completed(
        {
            "model": "gpt-test",
            "status": "completed",
            "usage": {"input_tokens": 12, "output_tokens": 3},
            "provider_state": [] if provider_state is None else provider_state,
            "finish_reason": "stop",
            "end_turn": end_turn,
        }
    )


def run(coroutine):
    return asyncio.run(coroutine)


def test_text_response_and_tool_schema_are_translated() -> None:
    provider = ScriptedProvider(
        [[ModelStreamEvent.text_delta("Hello"), ModelStreamEvent.text_delta(" there"), completed()]]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")
    bridge.append_tau_input(conversation, UserMessage.text("Hi"))

    response = run(bridge.generate(conversation, [as_tool(lookup_account)]))

    assert response.content == "Hello there"
    assert response.tool_calls is None
    assert response.usage == {"prompt_tokens": 12, "completion_tokens": 3}
    diagnostics = response.raw_data["cayu_tau3_bridge"]
    assert diagnostics["internal_continuations"] == 0
    assert diagnostics["provider_call_count"] == 1
    assert diagnostics["provider_usages"] == [
        {
            "cached_input_tokens": 0,
            "input_tokens": 12,
            "output_tokens": 3,
            "reasoning_output_tokens": 0,
        }
    ]
    trace = diagnostics["provider_call_trace"]
    assert trace["schema_version"] == 1
    assert len(trace["calls"]) == 1
    assert trace["calls"][0]["phase"] == "initial"
    assert trace["calls"][0]["result_kind"] == "terminal_text"
    assert trace["calls"][0]["forced_tool"] is False
    assert trace["calls"][0]["message_count"] == 2
    assert trace["calls"][0]["system_message_count"] == 1
    assert trace["calls"][0]["tool_schema_count"] == 1
    assert trace["calls"][0]["message_bytes"] > 0
    assert trace["calls"][0]["system_message_bytes"] > 0
    assert trace["calls"][0]["tool_schema_bytes"] > 0
    assert provider.requests[0].tools == [
        {
            "name": "lookup_account",
            "description": "Look up one account.",
            "input_schema": lookup_account_tool_schema(),
        }
    ]


def lookup_account_tool_schema() -> dict:
    schema = as_tool(lookup_account).openai_schema["function"]["parameters"]
    return schema


def test_provider_usage_attribution_is_not_persisted() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.text_delta("Done."),
                ModelStreamEvent.completed(
                    {
                        "model": "gpt-test",
                        "status": "completed",
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 3,
                            "attribution": {"private-message-id": {"input_tokens": 12}},
                        },
                        "provider_state": [],
                        "finish_reason": "stop",
                        "end_turn": True,
                    }
                ),
            ]
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(bridge.generate(conversation, []))

    diagnostics = response.raw_data["cayu_tau3_bridge"]
    assert diagnostics["provider_usages"] == [
        {
            "cached_input_tokens": 0,
            "input_tokens": 12,
            "output_tokens": 3,
            "reasoning_output_tokens": 0,
        }
    ]
    assert "private-message-id" not in json.dumps(diagnostics)


def test_agent_deterministically_acknowledges_successful_human_transfer() -> None:
    provider = ScriptedProvider([])
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    gateway = CayuModelGateway(bridge)
    agent = CayuBridgeAgent(
        tools=[as_tool(transfer_to_human_agents)],
        domain_policy="Transfer an explicitly escalated conversation to a human agent.",
        gateway=gateway,
    )
    transfer_arguments = {
        "reason": "customer_requests_human_no_specific_reason",
        "summary": "The customer explicitly requested a human agent.",
    }

    try:
        state = agent.get_init_state()
        plan = state.controller.invoke(
            SYNC_WORK_PLAN_TOOL,
            {
                "items": [
                    {
                        "id": "escalate",
                        "goal": "Escalate the conversation to a human agent",
                        "completion_mode": "operation",
                        "status": "in_progress",
                        "outcome": "unresolved",
                        "depends_on": [],
                        "next_operation": {
                            "owner": "agent",
                            "kind": "tool",
                            "tool_name": HUMAN_TRANSFER_TOOL,
                            "arguments": transfer_arguments,
                            "summary": "Transfer the conversation to a human agent.",
                        },
                        "evidence": "",
                        "evidence_refs": [],
                    }
                ]
            },
        )
        assert plan.is_error is False
        transfer_call = AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="transfer-1",
                    name=HUMAN_TRANSFER_TOOL,
                    arguments=transfer_arguments,
                    requestor="assistant",
                )
            ],
        )
        bridge.append_tau_input(state.conversation, transfer_call)
        state.ledger.observe_assistant(transfer_call)

        response, returned_state = agent.generate_next_message(
            ToolMessage(
                id="transfer-1",
                role="tool",
                requestor="assistant",
                content="The transfer was successful.",
            ),
            state,
        )
    finally:
        gateway.close()

    assert returned_state is state
    assert provider.requests == []
    assert response.content == (
        "Your transfer to a human agent was successful. A human agent will assist you shortly."
    )
    diagnostics = response.raw_data["cayu_tau3_bridge"]
    assert diagnostics["deterministic_completion"] == "successful_human_transfer"
    assert diagnostics["provider_call_count"] == 0
    workflow = response.raw_data["cayu_tau3_workflow"]
    assert workflow["revision"] == 2
    assert workflow["items"][0]["status"] == "completed"
    assert workflow["items"][0]["outcome"] == "handed_off"
    assert workflow["items"][0]["evidence_refs"][-1] == {
        "kind": "tool_result",
        "reference_id": "transfer-1",
        "available": True,
    }
    assert workflow["last_response_disposition"] == {
        "items": [{"item_id": "escalate", "kind": "final"}],
        "plan_revision": 2,
    }
    assert state.conversation.messages[-1].role is MessageRole.ASSISTANT


class RecordingInternalToolHandler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def tool_schemas(self):
        return (
            {
                "name": _INTERNAL_TEST_TOOL,
                "description": "Record a private test plan.",
                "input_schema": {
                    "type": "object",
                    "properties": {"note": {"type": "string"}},
                    "required": ["note"],
                    "additionalProperties": False,
                },
            },
        )

    def handles(self, name: str) -> bool:
        return name == _INTERNAL_TEST_TOOL

    def invoke(self, name: str, arguments: dict) -> InternalToolResult:
        self.calls.append((name, dict(arguments)))
        return InternalToolResult(content="Private plan recorded.")


class AsyncRecordingInternalToolHandler(RecordingInternalToolHandler):
    async def invoke(self, name: str, arguments: dict) -> InternalToolResult:
        await asyncio.sleep(0)
        return super().invoke(name, arguments)


class AdvisingInternalToolHandler(RecordingInternalToolHandler):
    def required_action_tool_names(self) -> tuple[str, ...]:
        return ("lookup_account", _INTERNAL_TEST_TOOL)


class BatchAdvisingInternalToolHandler(AdvisingInternalToolHandler):
    def required_action_batch_size(self, name: str) -> int:
        return 2 if name == "lookup_account" else 1


class InvalidBatchAdvisingInternalToolHandler(AdvisingInternalToolHandler):
    def required_action_batch_size(self, name: str) -> int:
        return 0


class StagingInternalToolHandler(RecordingInternalToolHandler):
    def allows_staged_external_calls(self, internal_tool_names: tuple[str, ...]) -> bool:
        return internal_tool_names == (_INTERNAL_TEST_TOOL,)


class FailingStagingInternalToolHandler(StagingInternalToolHandler):
    def invoke(self, name: str, arguments: dict) -> InternalToolResult:
        self.calls.append((name, dict(arguments)))
        return InternalToolResult(content="Private plan rejected.", is_error=True)


class InvalidContextInternalToolHandler(RecordingInternalToolHandler):
    def render_internal_context(self) -> str:
        return ""


class ProgressRecordingInternalToolHandler(RecordingInternalToolHandler):
    def __init__(self) -> None:
        super().__init__()
        self.notes: set[str] = set()

    def invoke(self, name: str, arguments: dict) -> InternalToolResult:
        result = super().invoke(name, arguments)
        self.notes.add(arguments["note"])
        return result

    def progress_token(self) -> str:
        return json.dumps(sorted(self.notes))


class RetainingProgressInternalToolHandler(ProgressRecordingInternalToolHandler):
    def transcript_retention_key(self, name: str) -> str | None:
        return "test_plan" if name == _INTERNAL_TEST_TOOL else None


class SingleUseInternalToolHandler(RecordingInternalToolHandler):
    def __init__(self) -> None:
        super().__init__()
        self.available = True

    def tool_schemas(self):
        return super().tool_schemas() if self.available else ()

    def invoke(self, name: str, arguments: dict) -> InternalToolResult:
        result = super().invoke(name, arguments)
        self.available = False
        return result


class StaticErroringInternalToolHandler(RecordingInternalToolHandler):
    def invoke(self, name: str, arguments: dict) -> InternalToolResult:
        self.calls.append((name, dict(arguments)))
        return InternalToolResult(
            content="The proposed correction is invalid.",
            is_error=True,
            diagnostic_code="sync_invalid_items",
        )

    def progress_token(self) -> str:
        return "unchanged"

    def diagnostic_counts(self) -> dict[str, int]:
        return {"product_selection_transition_rejections": len(self.calls)}


class RetainingErrorInternalToolHandler(StaticErroringInternalToolHandler):
    def transcript_retention_key(self, name: str) -> str | None:
        return "test_plan" if name == _INTERNAL_TEST_TOOL else None


class TypedRemediationInternalToolHandler(RecordingInternalToolHandler):
    def __init__(self) -> None:
        super().__init__()
        self.accepted_notes: set[str] = set()
        self.remediation = InternalToolRemediation(
            code="replace_invalid_state",
            plan_revision=3,
            item_id="work-item",
            forbidden_transition="pending_to_terminal",
            allowed_repairs=("restore_pending_state",),
        )

    def invoke(self, name: str, arguments: dict) -> InternalToolResult:
        self.calls.append((name, dict(arguments)))
        if arguments["note"] == "invalid-state":
            return InternalToolResult(
                content="The proposed state transition is invalid.",
                is_error=True,
                remediation=self.remediation,
            )
        self.accepted_notes.add(arguments["note"])
        return InternalToolResult(
            content="Corrected state accepted.",
            resolutions=(
                InternalRemediationResolution.from_remediation(
                    self.remediation,
                    repair_class="restore_pending_state",
                ),
            ),
        )

    def progress_token(self) -> str:
        return json.dumps(sorted(self.accepted_notes))


class StateNeutralRepairInternalToolHandler(TypedRemediationInternalToolHandler):
    def progress_token(self) -> str:
        return "unchanged"


class ChurningTypedRemediationInternalToolHandler(RecordingInternalToolHandler):
    def __init__(self) -> None:
        super().__init__()
        self.remediation = InternalToolRemediation(
            code="selected_product_requires_advancement",
            plan_revision=3,
            item_id="work-item",
            forbidden_transition="pending_to_terminal",
            allowed_repairs=("agent_owned_operation",),
        )

    def invoke(self, name: str, arguments: dict) -> InternalToolResult:
        self.calls.append((name, dict(arguments)))
        return InternalToolResult(
            content="The proposed state transition remains invalid.",
            is_error=True,
            remediation=self.remediation,
        )

    def progress_token(self) -> str:
        return "unchanged"


class SupersedingTypedRemediationInternalToolHandler(RecordingInternalToolHandler):
    def __init__(self) -> None:
        super().__init__()
        self.resolved = False
        self.remediations = {
            revision: InternalToolRemediation(
                code="selected_product_requires_advancement",
                plan_revision=revision,
                item_id="work-item",
                forbidden_transition="pending_to_terminal",
                allowed_repairs=("agent_owned_operation",),
            )
            for revision in (3, 4)
        }

    def invoke(self, name: str, arguments: dict) -> InternalToolResult:
        self.calls.append((name, dict(arguments)))
        note = arguments["note"]
        if note in {"revision-3", "revision-4"}:
            revision = int(note.removeprefix("revision-"))
            return InternalToolResult(
                content="The proposed state transition is invalid.",
                is_error=True,
                remediation=self.remediations[revision],
            )
        self.resolved = True
        return InternalToolResult(
            content="Corrected state accepted.",
            resolutions=(
                InternalRemediationResolution.from_remediation(
                    self.remediations[4],
                    repair_class="agent_owned_operation",
                ),
            ),
        )

    def progress_token(self) -> str:
        return str(self.resolved)


class RecordingTurnContextProvider:
    def __init__(self, context: str | None) -> None:
        self.context = context
        self.calls = 0

    async def prepare_context(self) -> str | None:
        self.calls += 1
        await asyncio.sleep(0)
        return self.context


class RejectOnceExternalToolGate:
    def __init__(self) -> None:
        self.reviews = 0

    def review(self, calls):
        self.reviews += 1
        return tuple(
            ExternalToolAdmission(
                tool_call_id=call.tool_call_id,
                allowed=self.reviews > 1,
                reason=(None if self.reviews > 1 else "required sequencing is incomplete"),
            )
            for call in calls
        )


class AllowExternalToolGate:
    def review(self, calls):
        return tuple(
            ExternalToolAdmission(tool_call_id=call.tool_call_id, allowed=True) for call in calls
        )


class RejectTerminalOutputGate:
    def __init__(self, *, required_tool_names: tuple[str, ...] = ()) -> None:
        self.reviews = 0
        self.required_tool_names = required_tool_names

    def review_terminal_output(self, text: str) -> TerminalOutputAdmission:
        assert text
        self.reviews += 1
        return TerminalOutputAdmission(
            allowed=False,
            reason="work item change-email still requires change_user_email",
            correction_code=("response_actionable_items" if self.required_tool_names else None),
            required_tool_names=self.required_tool_names,
        )


class RejectTwiceThenAllowTerminalOutputGate:
    def __init__(self) -> None:
        self.reviews = 0

    def review_terminal_output(self, text: str) -> TerminalOutputAdmission:
        assert text
        self.reviews += 1
        if self.reviews > 2:
            return TerminalOutputAdmission(allowed=True)
        return TerminalOutputAdmission(
            allowed=False,
            reason="the validated plan still has an actionable item",
            correction_code="response_actionable_items",
            required_tool_names=(_INTERNAL_TEST_TOOL,),
        )


def test_tool_call_is_returned_without_mixed_text_and_result_replays_name() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.text_delta("I will check."),
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ]
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")
    bridge.append_tau_input(conversation, UserMessage.text("Find account A-1"))

    response = run(bridge.generate(conversation, [as_tool(lookup_account)]))

    assert response.content is None
    assert [call.model_dump() for call in response.tool_calls] == [
        {
            "id": "call_1",
            "name": "lookup_account",
            "arguments": {"account_id": "A-1"},
            "requestor": "assistant",
        }
    ]
    assert response.raw_data["cayu_tau3_bridge"]["suppressed_mixed_text"] is True

    bridge.append_tau_input(
        conversation,
        ToolMessage(
            id="call_1",
            role="tool",
            requestor="assistant",
            content='{"account_id":"A-1"}',
        ),
    )
    result_part = conversation.messages[-1].content[0]
    assert isinstance(result_part, ToolResultPart)
    assert result_part.tool_name == "lookup_account"


def test_end_turn_false_continues_with_provider_state() -> None:
    provider_state = [
        {
            "provider": "openai",
            "state": {"type": "reasoning", "id": "rs_1", "summary": []},
        }
    ]
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.text_delta("working"),
                completed(end_turn=False, provider_state=provider_state),
            ],
            [ModelStreamEvent.text_delta("Done."), completed(end_turn=True)],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")
    bridge.append_tau_input(conversation, UserMessage.text("Help"))

    response = run(bridge.generate(conversation, []))

    assert response.content == "Done."
    assert response.raw_data["cayu_tau3_bridge"]["internal_continuations"] == 1
    assert len(provider.requests) == 2
    continuation_message = provider.requests[1].messages[-1]
    assert continuation_message.role is MessageRole.ASSISTANT
    assert any(isinstance(part, ProviderStatePart) for part in continuation_message.content)


def test_internal_continuation_limit_has_redacted_failure_diagnostics() -> None:
    provider = ScriptedProvider(
        [
            [ModelStreamEvent.text_delta("sensitive working text"), completed(end_turn=False)],
            [ModelStreamEvent.text_delta("still sensitive"), completed(end_turn=False)],
        ]
    )
    bridge = CayuTauBridge(
        provider=provider,
        model="gpt-test",
        max_internal_continuations=1,
    )
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    with pytest.raises(BridgeProtocolError, match="internal continuation limit") as raised:
        run(bridge.generate(conversation, []))

    diagnostics = raised.value.diagnostics
    assert diagnostics is not None
    assert diagnostics.failure_code == "internal_continuation_limit"
    assert diagnostics.internal_continuations == 1
    assert diagnostics.current_output_kind == "continuation"
    assert "sensitive" not in json.dumps(diagnostics.as_record())


def test_empty_terminal_response_raises_before_constructing_tau_message() -> None:
    provider = ScriptedProvider(
        [
            [completed(end_turn=True)],
            [completed(end_turn=True)],
            [completed(end_turn=True)],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    with pytest.raises(BridgeProtocolError, match="repeatedly completed without text") as raised:
        run(bridge.generate(conversation, []))

    assert len(conversation.messages) == 1
    assert len(provider.requests) == 3
    assert raised.value.diagnostics is not None
    assert raised.value.diagnostics.failure_code == "empty_response_limit"
    assert raised.value.diagnostics.empty_response_retries == 2
    assert raised.value.diagnostics.current_output_kind == "empty_response"


def test_empty_terminal_response_is_retried_without_persisting_correction() -> None:
    provider = ScriptedProvider(
        [
            [completed(end_turn=True)],
            [ModelStreamEvent.text_delta("Recovered."), completed(end_turn=True)],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(bridge.generate(conversation, []))

    assert response.content == "Recovered."
    assert response.raw_data["cayu_tau3_bridge"]["empty_response_retries"] == 1
    assert response.usage == {"prompt_tokens": 24, "completion_tokens": 6}
    assert [
        item["phase"]
        for item in response.raw_data["cayu_tau3_bridge"]["provider_call_trace"]["calls"]
    ] == ["initial", "empty_response_retry"]
    assert provider.requests[1].messages[-1].role is MessageRole.SYSTEM
    assert len(conversation.messages) == 2


def test_transient_context_is_sent_but_never_committed() -> None:
    provider = ScriptedProvider([[ModelStreamEvent.text_delta("Done."), completed(end_turn=True)]])
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")
    bridge.append_tau_input(conversation, UserMessage.text("Help"))

    response = run(
        bridge.generate(
            conversation,
            [],
            transient_context="Controller observation ledger.",
        )
    )

    assert response.content == "Done."
    assert provider.requests[0].messages[-1].role is MessageRole.SYSTEM
    assert provider.requests[0].messages[-1].content[0].text == ("Controller observation ledger.")
    assert len(conversation.messages) == 3
    assert all(
        getattr(part, "text", None) != "Controller observation ledger."
        for message in conversation.messages
        for part in message.content
    )


def test_transient_context_precedes_internal_continuation_state() -> None:
    provider = ScriptedProvider(
        [
            [ModelStreamEvent.text_delta("working"), completed(end_turn=False)],
            [ModelStreamEvent.text_delta("Done."), completed(end_turn=True)],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(bridge.generate(conversation, [], transient_context="Ledger snapshot."))

    assert response.content == "Done."
    second_request = provider.requests[1].messages
    assert second_request[-2].role is MessageRole.SYSTEM
    assert second_request[-2].content[0].text == "Ledger snapshot."
    assert second_request[-1].role is MessageRole.ASSISTANT
    assert len(conversation.messages) == 2
    assert all(
        getattr(part, "text", None) != "working"
        for message in conversation.messages
        for part in message.content
    )


def test_gateway_prepares_async_context_on_its_event_loop() -> None:
    provider = ScriptedProvider([[ModelStreamEvent.text_delta("Done."), completed()]])
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")
    context = RecordingTurnContextProvider("Cayu memory focus.")

    with CayuModelGateway(bridge) as gateway:
        response = gateway.generate(conversation, [], context_provider=context)

    assert response.content == "Done."
    assert context.calls == 1
    assert conversation.active_failure_diagnostics is None
    prepared = provider.requests[0].messages[-1]
    assert prepared.role is MessageRole.SYSTEM
    assert prepared.content[0].text == "Cayu memory focus."


def test_agent_uses_cayu_memory_while_returning_tau_tool_calls_to_orchestrator(
    tmp_path: Path,
) -> None:
    (tmp_path / "account_lookup.json").write_text(
        json.dumps(
            {
                "id": "account_lookup",
                "title": "Looking up an account",
                "content": "Verify the customer before looking up a private account.",
            }
        ),
        encoding="utf-8",
    )
    memory_runtime = BankingMemoryRuntime(BankingKnowledgeCorpus.from_directory(tmp_path))
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="plan-account-lookup",
                    name=SYNC_WORK_PLAN_TOOL,
                    arguments={
                        "items": [
                            {
                                "id": "lookup-account",
                                "goal": "Look up the requested account",
                                "completion_mode": "operation",
                                "status": "pending",
                                "outcome": "unresolved",
                                "depends_on": [],
                                "next_operation": {
                                    "owner": "agent",
                                    "kind": "tool",
                                    "tool_name": "lookup_account",
                                    "arguments": {"account_id": "A-1"},
                                    "summary": "Look up the requested account.",
                                },
                                "evidence": "",
                                "evidence_refs": [],
                            }
                        ]
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="external-account-1",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    with CayuModelGateway(bridge) as gateway:
        agent = CayuBridgeAgent(
            tools=[as_tool(lookup_account)],
            domain_policy="Verify customers before private lookups.",
            gateway=gateway,
            memory_runtime=memory_runtime,
        )
        state = agent.get_init_state()

        response, state = agent.generate_next_message(UserMessage.text("Find account A-1"), state)

    assert [call.id for call in response.tool_calls] == ["external-account-1"]
    assert len(provider.requests) == 1
    assert response.raw_data["cayu_tau3_bridge"]["internal_tool_rounds"] == 1
    assert state.ledger.actions[0].status.value == "pending"
    assert response.raw_data["cayu_tau3_memory"]["last_recall"]["injected_count"] >= 1
    request = provider.requests[0]
    assert {tool["name"] for tool in request.tools} == {
        *NAVIGATION_TOOLS,
        "lookup_account",
        AUDIT_CREDIT_CARD_DISPUTES_TOOL,
        AUDIT_CREDIT_CARD_REWARDS_TOOL,
        AUDIT_SAVINGS_INTEREST_TOOL,
        PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL,
        PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL,
        PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
        KNOWLEDGE_SEARCH_TOOL,
        SYNC_WORK_PLAN_TOOL,
    }
    context = request.messages[-1].content[0].text
    assert "<cayu-work-context>" in context
    assert "<banking-knowledge-recall>" in context


def test_cayu_memory_does_not_enable_terminal_controller_when_workflow_is_disabled(
    tmp_path: Path,
) -> None:
    (tmp_path / "greeting.json").write_text(
        json.dumps(
            {
                "id": "greeting",
                "title": "Customer greetings",
                "content": "Ask the customer how the bank can help.",
            }
        ),
        encoding="utf-8",
    )
    memory_runtime = BankingMemoryRuntime(BankingKnowledgeCorpus.from_directory(tmp_path))
    provider = ScriptedProvider(
        [[ModelStreamEvent.text_delta("How can I help?"), completed(end_turn=True)]]
    )
    gateway = CayuModelGateway(CayuTauBridge(provider=provider, model="gpt-test"))
    agent = CayuBridgeAgent(
        tools=[],
        domain_policy="Help the customer.",
        gateway=gateway,
        memory_runtime=memory_runtime,
        workflow_controller_enabled=False,
    )

    try:
        response, _ = agent.generate_next_message(
            UserMessage.text("Hello"),
            agent.get_init_state(),
        )
    finally:
        gateway.close()

    assert response.content == "How can I help?"
    assert SYNC_WORK_PLAN_TOOL not in {tool["name"] for tool in provider.requests[0].tools}


def test_internal_controller_tool_is_executed_before_tau_tool_is_returned() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-1",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Look up the account"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="external-1",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
        ]
    )
    handler = RecordingInternalToolHandler()
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [as_tool(lookup_account)],
            internal_tool_handler=handler,
        )
    )

    assert response.tool_calls[0].id == "external-1"
    assert response.raw_data["cayu_tau3_bridge"]["internal_tool_rounds"] == 1
    assert response.usage == {"prompt_tokens": 24, "completion_tokens": 6}
    assert [
        item["phase"]
        for item in response.raw_data["cayu_tau3_bridge"]["provider_call_trace"]["calls"]
    ] == ["initial", "internal_tool_result"]
    assert handler.calls == [(_INTERNAL_TEST_TOOL, {"note": "Look up the account"})]
    assert "controller-1" not in conversation.tool_names_by_call_id
    assert conversation.tool_names_by_call_id == {"external-1": "lookup_account"}
    internal_result = provider.requests[1].messages[-1].content[0]
    assert isinstance(internal_result, ToolResultPart)
    assert internal_result.tool_name == _INTERNAL_TEST_TOOL
    assert internal_result.is_error is False
    assert len(conversation.messages) == 2
    assert all(
        not isinstance(part, ToolResultPart)
        for message in conversation.messages
        for part in message.content
    )


def test_successful_internal_round_forces_the_authorized_next_action() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-1",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Authorize the account lookup"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="external-1",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [as_tool(lookup_account)],
            internal_tool_handler=AdvisingInternalToolHandler(),
        )
    )

    assert response.tool_calls[0].name == "lookup_account"
    assert provider.requests[1].options["openai"] == {
        "parallel_tool_calls": False,
        "tool_choice": {"type": "function", "name": "lookup_account"},
    }


def test_successful_internal_round_allows_the_authorized_homogeneous_batch() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-1",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Authorize both account lookups"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="external-1",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                ModelStreamEvent.tool_call(
                    id="external-2",
                    name="lookup_account",
                    arguments={"account_id": "A-2"},
                ),
                completed(),
            ],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [as_tool(lookup_account)],
            internal_tool_handler=BatchAdvisingInternalToolHandler(),
        )
    )

    assert [call.id for call in response.tool_calls] == ["external-1", "external-2"]
    assert provider.requests[1].options["openai"] == {
        "parallel_tool_calls": True,
        "tool_choice": {"type": "function", "name": "lookup_account"},
    }
    payload = build_openai_payload(provider.requests[1])
    assert payload["parallel_tool_calls"] is True
    assert payload["tool_choice"] == {
        "type": "function",
        "name": "lookup_account",
    }


def test_required_action_batch_rejects_more_calls_than_the_controller_authorized() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-1",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Authorize two account lookups"},
                ),
                completed(),
            ],
            [
                *[
                    ModelStreamEvent.tool_call(
                        id=f"external-{index}",
                        name="lookup_account",
                        arguments={"account_id": f"A-{index}"},
                    )
                    for index in range(1, 4)
                ],
                completed(),
            ],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    with pytest.raises(BridgeProtocolError, match="did not honor") as raised:
        run(
            bridge.generate(
                conversation,
                [as_tool(lookup_account)],
                internal_tool_handler=BatchAdvisingInternalToolHandler(),
            )
        )

    assert raised.value.diagnostics is not None
    assert raised.value.diagnostics.failure_code == "required_correction_tool_not_called"
    assert len(provider.requests) == 2


def test_required_action_batch_size_fails_closed_before_the_next_provider_call() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-1",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Authorize an account lookup"},
                ),
                completed(),
            ]
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    with pytest.raises(BridgeProtocolError, match="bounded positive integer"):
        run(
            bridge.generate(
                conversation,
                [as_tool(lookup_account)],
                internal_tool_handler=InvalidBatchAdvisingInternalToolHandler(),
            )
        )

    assert len(provider.requests) == 1


def test_invalid_refreshed_internal_context_fails_closed() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-1",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Change private state"},
                ),
                completed(),
            ]
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    with pytest.raises(BridgeProtocolError, match="nonblank text or null"):
        run(
            bridge.generate(
                conversation,
                [],
                internal_tool_handler=InvalidContextInternalToolHandler(),
            )
        )

    assert len(provider.requests) == 1


def test_bridge_canonicalizes_discoverable_arguments_in_plan_and_external_call() -> None:
    nested_arguments = '{"user_id":"user-7","account_type":"savings"}'
    expected_arguments = '{"user_id": "user-7", "account_type": "savings"}'
    planned_call_arguments = {
        "agent_tool_name": "open_bank_account_4821",
        "arguments": nested_arguments,
    }
    planned_item = {
        "id": "open-account",
        "goal": "Open the requested savings account",
        "completion_mode": "operation",
        "status": "in_progress",
        "outcome": "unresolved",
        "depends_on": [],
        "next_operation": {
            "owner": "agent",
            "kind": "tool",
            "tool_name": "call_discoverable_agent_tool",
            "arguments": planned_call_arguments,
            "summary": "Call the unlocked account-opening tool.",
        },
        "evidence": "",
        "evidence_refs": [],
    }
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="plan-1",
                    name=SYNC_WORK_PLAN_TOOL,
                    arguments={"items": [planned_item]},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="external-1",
                    name="call_discoverable_agent_tool",
                    arguments=planned_call_arguments,
                ),
                completed(),
            ],
        ]
    )
    controller = BankingWorkflowController(ConversationLedger())
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [as_tool(call_discoverable_agent_tool)],
            internal_tool_handler=controller,
            external_tool_gate=controller,
        )
    )

    assert response.tool_calls[0].arguments == {
        "agent_tool_name": "open_bank_account_4821",
        "arguments": expected_arguments,
    }
    assert controller.export_state()["items"][0]["next_operation"]["arguments"] == {
        "agent_tool_name": "open_bank_account_4821",
        "arguments": expected_arguments,
    }


def test_bridge_leaves_customer_tool_arguments_for_customer_owned_operation() -> None:
    proposed_arguments = {
        "discoverable_tool_name": "get_card_last_4_digits",
        "arguments": '{"credit_card_account_id":"cc-user-7-silver"}',
    }
    planned_item = {
        "id": "deliver-card-lookup",
        "goal": "Give the customer the required card lookup tool",
        "completion_mode": "operation",
        "status": "in_progress",
        "outcome": "unresolved",
        "depends_on": [],
        "next_operation": {
            "owner": "agent",
            "kind": "tool",
            "tool_name": "give_discoverable_user_tool",
            "arguments": proposed_arguments,
            "summary": "Expose the customer card lookup capability.",
        },
        "evidence": "",
        "evidence_refs": [],
    }
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="plan-1",
                    name=SYNC_WORK_PLAN_TOOL,
                    arguments={"items": [planned_item]},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="external-1",
                    name="give_discoverable_user_tool",
                    arguments=proposed_arguments,
                ),
                completed(),
            ],
        ]
    )
    controller = BankingWorkflowController(ConversationLedger())
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [as_tool(give_discoverable_user_tool)],
            internal_tool_handler=controller,
            external_tool_gate=controller,
        )
    )

    expected = {"discoverable_tool_name": "get_card_last_4_digits"}
    assert response.tool_calls[0].arguments == expected
    assert controller.export_state()["items"][0]["next_operation"]["arguments"] == expected


def test_bridge_preserves_two_decimal_currency_spelling_for_interest_actions() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="credit-1",
                    name="call_discoverable_agent_tool",
                    arguments={
                        "agent_tool_name": "apply_savings_account_credit_6831",
                        "arguments": json.dumps(
                            {
                                "account_id": "savings-1",
                                "amount": 98,
                                "credit_type": "interest_correction",
                            },
                            separators=(",", ":"),
                        ),
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="report-1",
                    name="call_discoverable_agent_tool",
                    arguments={
                        "agent_tool_name": "submit_interest_discrepancy_report_7294",
                        "arguments": json.dumps(
                            {
                                "account_id": "savings-1",
                                "user_id": "user-7",
                                "expected_apy": 6.85,
                                "actual_apy": 5.625,
                                "amount_difference": 98,
                            },
                            separators=(",", ":"),
                        ),
                    },
                ),
                completed(),
            ]
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [as_tool(call_discoverable_agent_tool)],
        )
    )

    assert response.tool_calls[0].arguments["arguments"] == (
        '{"account_id": "savings-1", "amount": 98.00, "credit_type": "interest_correction"}'
    )
    assert response.tool_calls[1].arguments["arguments"] == (
        '{"account_id": "savings-1", "user_id": "user-7", '
        '"expected_apy": 6.85, "actual_apy": 5.625, "amount_difference": 98.00}'
    )


def test_bridge_preserves_json_looking_strings_for_non_wrapper_tools() -> None:
    opaque_arguments = '{"preserve":"this spelling"}'
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="external-1",
                    name="forward_arguments",
                    arguments={"arguments": opaque_arguments},
                ),
                completed(),
            ]
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(bridge.generate(conversation, [as_tool(forward_arguments)]))

    assert response.tool_calls[0].arguments == {"arguments": opaque_arguments}


def test_async_internal_tool_is_awaited_before_model_continuation() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-async",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Search durable memory"},
                ),
                completed(),
            ],
            [ModelStreamEvent.text_delta("Done."), completed()],
        ]
    )
    handler = AsyncRecordingInternalToolHandler()
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(bridge.generate(conversation, [], internal_tool_handler=handler))

    assert response.content == "Done."
    assert handler.calls == [(_INTERNAL_TEST_TOOL, {"note": "Search durable memory"})]
    result = provider.requests[1].messages[-1].content[0]
    assert isinstance(result, ToolResultPart)
    assert result.tool_call_id == "controller-async"
    assert result.content == "Private plan recorded."
    assert len(conversation.messages) == 2
    assert all(
        not isinstance(part, ToolResultPart)
        for message in conversation.messages
        for part in message.content
    )


def test_internal_tool_schemas_refresh_after_each_private_round() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="single-use",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Use the bounded capability"},
                ),
                completed(),
            ],
            [ModelStreamEvent.text_delta("Done."), completed()],
        ]
    )
    handler = SingleUseInternalToolHandler()
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(bridge.generate(conversation, [], internal_tool_handler=handler))

    assert response.content == "Done."
    assert _INTERNAL_TEST_TOOL in {tool["name"] for tool in provider.requests[0].tools}
    assert _INTERNAL_TEST_TOOL not in {tool["name"] for tool in provider.requests[1].tools}


def test_bridge_switches_from_plan_creation_to_incremental_updates() -> None:
    initial_item = {
        "id": "answer",
        "goal": "Answer the customer's question",
        "completion_mode": "information",
        "status": "pending",
        "outcome": "unresolved",
        "depends_on": [],
        "next_operation": {
            "owner": "agent",
            "kind": "knowledge_search",
            "tool_name": "",
            "arguments": {},
            "summary": "Retrieve the relevant policy.",
        },
        "evidence": "",
        "evidence_refs": [],
    }
    updated_item = {
        **initial_item,
        "status": "in_progress",
    }
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="create-plan",
                    name=SYNC_WORK_PLAN_TOOL,
                    arguments={"items": [initial_item]},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="update-plan",
                    name=UPDATE_WORK_PLAN_TOOL,
                    arguments={"items": [updated_item]},
                ),
                completed(),
            ],
            [ModelStreamEvent.text_delta("Done."), completed()],
        ]
    )
    controller = BankingWorkflowController(ConversationLedger())
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(bridge.generate(conversation, [], internal_tool_handler=controller))

    first_tools = {tool["name"] for tool in provider.requests[0].tools}
    second_tools = {tool["name"] for tool in provider.requests[1].tools}
    assert response.content == "Done."
    assert first_tools == {SYNC_WORK_PLAN_TOOL}
    assert second_tools == {UPDATE_WORK_PLAN_TOOL}
    assert controller.export_state()["items"][0]["status"] == "in_progress"
    final_request_call_ids = [
        part.tool_call_id
        for message in provider.requests[-1].messages
        for part in message.content
        if type(part) is ToolCallPart
    ]
    refreshed_contexts = [
        part.text
        for message in provider.requests[-1].messages
        for part in message.content
        if type(part) is TextPart and "<workflow-plan>" in part.text
    ]
    assert final_request_call_ids == ["update-plan"]
    assert len(refreshed_contexts) == 1
    assert '"revision": 2' in refreshed_contexts[0]
    assert response.raw_data["cayu_tau3_bridge"]["internal_compacted_rounds"] == 1
    assert response.raw_data["cayu_tau3_bridge"]["internal_context_refreshes"] == 2


def test_new_successful_private_snapshot_supersedes_older_round_context() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="plan-old",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "old"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="plan-current",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "current"},
                ),
                completed(),
            ],
            [ModelStreamEvent.text_delta("Finished"), completed(end_turn=True)],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")
    bridge.append_tau_input(conversation, UserMessage.text("Complete the work."))

    response = run(
        bridge.generate(
            conversation,
            [],
            internal_tool_handler=RetainingProgressInternalToolHandler(),
        )
    )

    final_request_call_ids = [
        part.tool_call_id
        for message in provider.requests[-1].messages
        for part in message.content
        if type(part) is ToolCallPart
    ]
    final_request_result_ids = [
        part.tool_call_id
        for message in provider.requests[-1].messages
        for part in message.content
        if type(part) is ToolResultPart
    ]
    assert response.content == "Finished"
    assert response.raw_data["cayu_tau3_bridge"]["internal_compacted_rounds"] == 1
    assert "plan-old" not in final_request_call_ids
    assert "plan-old" not in final_request_result_ids
    assert final_request_call_ids == ["plan-current"]
    assert final_request_result_ids == ["plan-current"]


def test_private_snapshot_compaction_preserves_failed_rounds() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="plan-invalid-one",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "invalid-one"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="plan-invalid-two",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "invalid-two"},
                ),
                completed(),
            ],
            [ModelStreamEvent.text_delta("Cannot update the plan."), completed(end_turn=True)],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [],
            internal_tool_handler=RetainingErrorInternalToolHandler(),
        )
    )

    final_request_call_ids = [
        part.tool_call_id
        for message in provider.requests[-1].messages
        for part in message.content
        if type(part) is ToolCallPart
    ]
    assert response.raw_data["cayu_tau3_bridge"]["internal_compacted_rounds"] == 0
    assert final_request_call_ids == ["plan-invalid-one", "plan-invalid-two"]


def test_compacted_private_round_preserves_external_result_replay_on_next_turn() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-private",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Prepare the account lookup"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="lookup-public",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
            [ModelStreamEvent.text_delta("The account was found."), completed()],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    tool_call = run(
        bridge.generate(
            conversation,
            [as_tool(lookup_account)],
            internal_tool_handler=RecordingInternalToolHandler(),
        )
    )
    bridge.append_tau_input(
        conversation,
        ToolMessage(
            id=tool_call.tool_calls[0].id,
            role="tool",
            requestor="assistant",
            content='{"account_id":"A-1"}',
        ),
    )
    response = run(bridge.generate(conversation, [as_tool(lookup_account)]))

    assert response.content == "The account was found."
    replayed_result = provider.requests[2].messages[-1].content[0]
    assert isinstance(replayed_result, ToolResultPart)
    assert replayed_result.tool_call_id == "lookup-public"
    assert replayed_result.tool_name == "lookup_account"
    assert all(
        getattr(part, "tool_call_id", None) != "controller-private"
        for message in provider.requests[2].messages
        for part in message.content
    )


def test_typed_internal_remediation_is_model_visible_and_repairable() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-invalid",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "invalid-state"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="controller-corrected",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "corrected-state"},
                ),
                completed(),
            ],
            [ModelStreamEvent.text_delta("Done."), completed()],
        ]
    )
    handler = TypedRemediationInternalToolHandler()
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(bridge.generate(conversation, [], internal_tool_handler=handler))

    assert response.content == "Done."
    assert handler.calls == [
        (_INTERNAL_TEST_TOOL, {"note": "invalid-state"}),
        (_INTERNAL_TEST_TOOL, {"note": "corrected-state"}),
    ]
    rejected = next(
        part
        for message in provider.requests[1].messages
        for part in message.content
        if isinstance(part, ToolResultPart) and part.tool_call_id == "controller-invalid"
    )
    assert isinstance(rejected, ToolResultPart)
    assert rejected.is_error is True
    assert rejected.structured is not None
    payload = json.loads(rejected.content)
    assert payload == {
        "message": "The proposed state transition is invalid.",
        "remediation": rejected.structured["remediation"],
    }
    remediation = payload["remediation"]
    assert remediation["schema_version"] == 1
    assert remediation["code"] == "replace_invalid_state"
    assert remediation["plan_revision"] == 3
    assert remediation["item_id"] == "work-item"
    assert remediation["forbidden_transition"] == "pending_to_terminal"
    assert remediation["allowed_repairs"] == ["restore_pending_state"]
    assert remediation["correction_id"].startswith("sha256:")
    accepted = next(
        part
        for message in provider.requests[2].messages
        for part in message.content
        if isinstance(part, ToolResultPart) and part.tool_call_id == "controller-corrected"
    )
    assert isinstance(accepted, ToolResultPart)
    assert accepted.content == "Corrected state accepted."
    assert accepted.structured is None
    assert response.raw_data["cayu_tau3_bridge"]["internal_stalled_rounds"] == 1
    assert response.raw_data["cayu_tau3_bridge"]["internal_progress_events"] == 1
    assert response.raw_data["cayu_tau3_bridge"]["internal_remediation"] == {
        "accepted_repair_classes": {"other": 1},
        "exhausted": 0,
        "issued": 1,
        "rejection_codes": {"other": 1},
        "repair_rounds_max": 1,
        "repair_rounds_total": 1,
        "repeated": 0,
        "resolved": 1,
        "schema_version": 1,
        "superseded": 0,
        "unresolved": 0,
    }


def test_state_neutral_repair_resolution_allows_completion() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-invalid",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "invalid-state"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="controller-restored",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "restored-current-state"},
                ),
                completed(),
            ],
            [ModelStreamEvent.text_delta("Done."), completed()],
        ]
    )
    handler = StateNeutralRepairInternalToolHandler()
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(bridge.generate(conversation, [], internal_tool_handler=handler))

    assert response.content == "Done."
    diagnostics = response.raw_data["cayu_tau3_bridge"]
    assert diagnostics["internal_stalled_rounds"] == 1
    assert diagnostics["internal_progress_events"] == 1
    assert diagnostics["internal_remediation"] == {
        "accepted_repair_classes": {"other": 1},
        "exhausted": 0,
        "issued": 1,
        "rejection_codes": {"other": 1},
        "repair_rounds_max": 1,
        "repair_rounds_total": 1,
        "repeated": 0,
        "resolved": 1,
        "schema_version": 1,
        "superseded": 0,
        "unresolved": 0,
    }


def test_internal_remediation_identity_is_stable_and_bounded() -> None:
    first = InternalToolRemediation(
        code="replace_invalid_state",
        plan_revision=3,
        item_id="work-item",
        forbidden_transition="pending_to_terminal",
        allowed_repairs=("restore_pending_state",),
    )
    identical = InternalToolRemediation(
        code="replace_invalid_state",
        plan_revision=3,
        item_id="work-item",
        forbidden_transition="pending_to_terminal",
        allowed_repairs=("restore_pending_state",),
    )
    changed_revision = InternalToolRemediation(
        code="replace_invalid_state",
        plan_revision=4,
        item_id="work-item",
        forbidden_transition="pending_to_terminal",
        allowed_repairs=("restore_pending_state",),
    )

    assert first.correction_id == identical.correction_id
    assert first.correction_id != changed_revision.correction_id
    with pytest.raises(ValueError, match="allowed_repairs"):
        InternalToolRemediation(
            code="replace_invalid_state",
            plan_revision=3,
            item_id="work-item",
            forbidden_transition="pending_to_terminal",
            allowed_repairs=("restore_pending_state", "restore_pending_state"),
        )
    with pytest.raises(ValueError, match="requires an error result"):
        InternalToolResult(content="Invalid combination.", remediation=first)
    with pytest.raises(ValueError, match="diagnostic_code is unsupported"):
        InternalToolResult(
            content="Invalid diagnostic.",
            is_error=True,
            diagnostic_code="private_customer_value",
        )
    with pytest.raises(ValueError, match="diagnostic_code requires an error result"):
        InternalToolResult(
            content="Invalid diagnostic.",
            diagnostic_code="sync_invalid_items",
        )
    with pytest.raises(ValueError, match="schema_version"):
        InternalToolRemediation(
            code="replace_invalid_state",
            plan_revision=3,
            item_id="work-item",
            forbidden_transition="pending_to_terminal",
            allowed_repairs=("restore_pending_state",),
            schema_version=True,
        )
    with pytest.raises(ValueError, match="item_id"):
        InternalToolRemediation(
            code="replace_invalid_state",
            plan_revision=3,
            item_id="private customer text",
            forbidden_transition="pending_to_terminal",
            allowed_repairs=("restore_pending_state",),
        )
    with pytest.raises(ValueError, match="was not allowed"):
        InternalRemediationResolution.from_remediation(
            first,
            repair_class="invented_repair",
        )
    resolution = InternalRemediationResolution.from_remediation(
        first,
        repair_class="restore_pending_state",
    )
    with pytest.raises(ValueError, match="cannot issue and resolve"):
        InternalToolResult(
            content="Invalid combination.",
            is_error=True,
            remediation=first,
            resolutions=(resolution,),
        )
    with pytest.raises(ValueError, match="correction_id"):
        InternalRemediationResolution(
            correction_id="private-target",
            code="replace_invalid_state",
            repair_class="restore_pending_state",
        )


def test_typed_remediation_identity_bounds_argument_churn() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id=f"controller-{index}",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": f"rewritten-invalid-state-{index}"},
                ),
                completed(),
            ]
            for index in range(2)
        ]
    )
    handler = ChurningTypedRemediationInternalToolHandler()
    bridge = CayuTauBridge(
        provider=provider,
        model="gpt-test",
        max_internal_tool_rounds=10,
        max_stalled_internal_tool_rounds=1,
        max_stalled_internal_tool_rounds_since_progress=8,
    )
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    with pytest.raises(BridgeProtocolError, match="consecutive equivalent no-progress") as raised:
        run(bridge.generate(conversation, [], internal_tool_handler=handler))

    diagnostics = raised.value.diagnostics
    assert diagnostics is not None
    assert diagnostics.failure_code == "internal_no_progress_limit"
    assert diagnostics.internal_tool_rounds == 2
    assert diagnostics.internal_stalled_rounds == 2
    assert diagnostics.consecutive_stalled_internal_rounds == 2
    assert len({round_.call_fingerprints for round_ in diagnostics.round_trace}) == 2
    assert [round_.error_codes for round_ in diagnostics.round_trace] == [
        ("unclassified_internal_error",),
        ("unclassified_internal_error",),
    ]
    assert diagnostics.as_record()["internal_error_codes"] == {"unclassified_internal_error": 2}
    assert diagnostics.remediation_lifecycle.as_record() == {
        "accepted_repair_classes": {},
        "exhausted": 1,
        "issued": 1,
        "rejection_codes": {"selected_product_requires_advancement": 1},
        "repair_rounds_max": 0,
        "repair_rounds_total": 0,
        "repeated": 1,
        "resolved": 0,
        "schema_version": 1,
        "superseded": 0,
        "unresolved": 1,
    }
    serialized = json.dumps(diagnostics.as_record())
    assert "work-item" not in serialized
    assert handler.remediation.correction_id not in serialized
    assert "rewritten-invalid-state" not in serialized


def test_typed_remediation_lifecycle_records_supersession_and_resolution() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-revision-3",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "revision-3"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="controller-revision-4",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "revision-4"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="controller-resolved",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "resolved"},
                ),
                completed(),
            ],
            [ModelStreamEvent.text_delta("Done."), completed()],
        ]
    )
    handler = SupersedingTypedRemediationInternalToolHandler()
    bridge = CayuTauBridge(
        provider=provider,
        model="gpt-test",
        max_stalled_internal_tool_rounds=1,
    )
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(bridge.generate(conversation, [], internal_tool_handler=handler))

    assert response.content == "Done."
    assert response.raw_data["cayu_tau3_bridge"]["internal_remediation"] == {
        "accepted_repair_classes": {"agent_owned_operation": 1},
        "exhausted": 0,
        "issued": 2,
        "rejection_codes": {"selected_product_requires_advancement": 2},
        "repair_rounds_max": 1,
        "repair_rounds_total": 1,
        "repeated": 0,
        "resolved": 1,
        "schema_version": 1,
        "superseded": 1,
        "unresolved": 0,
    }


def test_internal_resolution_without_open_remediation_fails_closed() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-resolved",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "resolved"},
                ),
                completed(),
            ]
        ]
    )
    handler = SupersedingTypedRemediationInternalToolHandler()
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    with pytest.raises(BridgeProtocolError, match="unknown or superseded remediation"):
        run(bridge.generate(conversation, [], internal_tool_handler=handler))


def test_progressing_internal_work_can_exceed_the_legacy_round_limit() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id=f"controller-{index}",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": f"distinct-state-{index}"},
                ),
                completed(),
            ]
            for index in range(13)
        ]
        + [[ModelStreamEvent.text_delta("Done."), completed()]]
    )
    handler = ProgressRecordingInternalToolHandler()
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(bridge.generate(conversation, [], internal_tool_handler=handler))

    diagnostics = response.raw_data["cayu_tau3_bridge"]
    assert response.content == "Done."
    assert diagnostics["internal_tool_rounds"] == 13
    assert diagnostics["internal_progress_events"] == 13
    assert diagnostics["internal_stalled_rounds"] == 0
    assert diagnostics["internal_calls_by_tool"] == {_INTERNAL_TEST_TOOL: 13}
    assert diagnostics["internal_tool_round_limit"] == 32
    assert diagnostics["stalled_internal_tool_round_limit"] == 4
    assert diagnostics["stalled_internal_rounds_since_progress"] == 0
    assert diagnostics["stalled_internal_tool_round_limit_since_progress"] == 8


def test_repeated_internal_work_hits_no_progress_limit_before_hard_limit() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id=f"controller-{index}",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "same-state"},
                ),
                completed(),
            ]
            for index in range(4)
        ]
    )
    handler = ProgressRecordingInternalToolHandler()
    bridge = CayuTauBridge(
        provider=provider,
        model="gpt-test",
        max_internal_tool_rounds=10,
        max_stalled_internal_tool_rounds=2,
    )
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    with pytest.raises(BridgeProtocolError, match="consecutive equivalent no-progress") as raised:
        run(bridge.generate(conversation, [], internal_tool_handler=handler))

    assert len(handler.calls) == 3
    assert len(provider.requests) == 4
    diagnostics = raised.value.diagnostics
    assert diagnostics is not None
    assert diagnostics.failure_code == "internal_no_progress_limit"
    assert diagnostics.provider_call_count == 4
    assert diagnostics.internal_tool_rounds == 3
    assert diagnostics.internal_progress_events == 1
    assert diagnostics.internal_stalled_rounds == 2
    assert diagnostics.consecutive_stalled_internal_rounds == 2
    assert [round_.progress_changed for round_ in diagnostics.round_trace] == [True, False, False]
    assert len({round_.call_fingerprints for round_ in diagnostics.round_trace}) == 1
    assert diagnostics.current_call_fingerprints == diagnostics.round_trace[-1].call_fingerprints
    assert "same-state" not in json.dumps(diagnostics.as_record())


def test_distinct_failed_corrections_fit_within_one_no_progress_window() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id=f"controller-{index}",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": f"different-correction-{index}"},
                ),
                completed(),
            ]
            for index in range(5)
        ]
        + [[ModelStreamEvent.text_delta("Done."), completed()]]
    )
    handler = StaticErroringInternalToolHandler()
    bridge = CayuTauBridge(
        provider=provider,
        model="gpt-test",
        max_internal_tool_rounds=10,
        max_stalled_internal_tool_rounds=2,
        max_stalled_internal_tool_rounds_since_progress=6,
    )
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(bridge.generate(conversation, [], internal_tool_handler=handler))

    metadata = response.raw_data["cayu_tau3_bridge"]
    assert response.content == "Done."
    assert metadata["internal_stalled_rounds"] == 5
    assert metadata["stalled_internal_rounds_since_progress"] == 5
    assert metadata["stalled_internal_tool_round_limit_since_progress"] == 6
    assert len(handler.calls) == 5


def test_verified_progress_restores_the_distinct_correction_budget() -> None:
    notes = ("first", "first", "second", "second", "third", "third")
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id=f"controller-{index}",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": note},
                ),
                completed(),
            ]
            for index, note in enumerate(notes)
        ]
        + [[ModelStreamEvent.text_delta("Done."), completed()]]
    )
    handler = ProgressRecordingInternalToolHandler()
    bridge = CayuTauBridge(
        provider=provider,
        model="gpt-test",
        max_internal_tool_rounds=10,
        max_stalled_internal_tool_rounds=2,
        max_stalled_internal_tool_rounds_since_progress=2,
    )
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(bridge.generate(conversation, [], internal_tool_handler=handler))

    metadata = response.raw_data["cayu_tau3_bridge"]
    assert response.content == "Done."
    assert metadata["internal_progress_events"] == 3
    assert metadata["internal_stalled_rounds"] == 3
    assert metadata["stalled_internal_rounds_since_progress"] == 1
    assert len(handler.calls) == 6


def test_distinct_failed_corrections_hit_no_progress_window_budget() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id=f"controller-{index}",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": f"different-correction-{index}"},
                ),
                completed(),
            ]
            for index in range(3)
        ]
    )
    handler = StaticErroringInternalToolHandler()
    bridge = CayuTauBridge(
        provider=provider,
        model="gpt-test",
        max_internal_tool_rounds=10,
        max_stalled_internal_tool_rounds=1,
        max_stalled_internal_tool_rounds_since_progress=3,
    )
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    with pytest.raises(BridgeProtocolError, match="since its last verified progress") as raised:
        run(bridge.generate(conversation, [], internal_tool_handler=handler))

    diagnostics = raised.value.diagnostics
    assert diagnostics is not None
    assert diagnostics.failure_code == "internal_no_progress_window_limit"
    assert diagnostics.internal_stalled_rounds == 3
    assert diagnostics.stalled_internal_rounds_since_progress == 3
    assert diagnostics.consecutive_stalled_internal_rounds == 1
    assert diagnostics.stalled_internal_tool_round_limit_since_progress == 3
    assert len({round_.call_fingerprints for round_ in diagnostics.round_trace}) == 3
    assert diagnostics.current_call_fingerprints == diagnostics.round_trace[-1].call_fingerprints
    assert len(handler.calls) == 3
    assert diagnostics.internal_diagnostic_counts == {"product_selection_transition_rejections": 3}
    assert [round_.error_codes for round_ in diagnostics.round_trace] == [
        ("sync_invalid_items",),
        ("sync_invalid_items",),
        ("sync_invalid_items",),
    ]
    assert diagnostics.as_record()["internal_error_codes"] == {"sync_invalid_items": 3}
    assert "different-correction" not in json.dumps(diagnostics.as_record())


def test_success_metadata_correlates_attempts_without_exposing_initial_text() -> None:
    provider = ScriptedProvider(
        [
            [ModelStreamEvent.text_delta("Done one."), completed()],
            [ModelStreamEvent.text_delta("Done two."), completed()],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    first = bridge.new_conversation(system_prompt="Follow policy.")
    second = bridge.new_conversation(system_prompt="Follow policy.")
    bridge.append_tau_input(first, UserMessage.text("private initial request"))
    bridge.append_tau_input(second, UserMessage.text("private initial request"))

    first_response = run(bridge.generate(first, []))
    second_response = run(bridge.generate(second, []))

    first_metadata = first_response.raw_data["cayu_tau3_bridge"]
    second_metadata = second_response.raw_data["cayu_tau3_bridge"]
    assert first_metadata["attempt_id"] != second_metadata["attempt_id"]
    assert first_metadata["correlation_tag"] == second_metadata["correlation_tag"]
    assert len(first_metadata["correlation_tag"]) == 64
    assert "private initial request" not in json.dumps(first_metadata)


def test_gateway_persists_structured_no_progress_failure(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id=f"controller-{index}",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "sensitive repeated arguments"},
                ),
                completed(),
            ]
            for index in range(4)
        ]
    )
    handler = ProgressRecordingInternalToolHandler()
    bridge = CayuTauBridge(
        provider=provider,
        model="gpt-test",
        max_internal_tool_rounds=10,
        max_stalled_internal_tool_rounds=2,
    )
    conversation = bridge.new_conversation(system_prompt="Follow policy.")
    bridge.append_tau_input(conversation, UserMessage.text("sensitive customer text"))
    path = tmp_path / "attempts.jsonl"

    with (
        CayuModelGateway(
            bridge,
            attempt_diagnostics_recorder=AttemptDiagnosticsRecorder(path),
        ) as gateway,
        pytest.raises(BridgeProtocolError, match="consecutive equivalent no-progress"),
    ):
        gateway.generate(conversation, [], internal_tool_handler=handler)

    persisted = path.read_text(encoding="utf-8")
    record = json.loads(persisted)
    assert record["attempt_id"] == conversation.attempt_id
    assert record["correlation_tag"] == conversation.correlation_tag
    assert record["failure_code"] == "internal_no_progress_limit"
    assert "sensitive customer text" not in persisted
    assert "sensitive repeated arguments" not in persisted


def test_tau_call_batched_with_controller_call_is_not_executed_or_returned() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-1",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Look up the account"},
                ),
                ModelStreamEvent.tool_call(
                    id="external-deferred",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="external-2",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [as_tool(lookup_account)],
            internal_tool_handler=RecordingInternalToolHandler(),
        )
    )

    assert [call.id for call in response.tool_calls] == ["external-2"]
    deferred_result = provider.requests[1].messages[-1].content[0]
    assert isinstance(deferred_result, ToolResultPart)
    assert deferred_result.tool_call_id == "external-deferred"
    assert deferred_result.is_error is True
    assert conversation.tool_names_by_call_id == {"external-2": "lookup_account"}


def test_authorized_plan_prefix_releases_a_staged_external_call_without_another_round() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-1",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Authorize the account lookup"},
                ),
                ModelStreamEvent.tool_call(
                    id="external-staged",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(
                    provider_state=[
                        {
                            "provider": "openai",
                            "state": {"type": "reasoning", "id": "reasoning-staged"},
                        }
                    ]
                ),
            ]
        ]
    )
    handler = StagingInternalToolHandler()
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [as_tool(lookup_account)],
            internal_tool_handler=handler,
        )
    )

    assert [call.id for call in response.tool_calls] == ["external-staged"]
    assert handler.calls == [(_INTERNAL_TEST_TOOL, {"note": "Authorize the account lookup"})]
    assert len(provider.requests) == 1
    assert response.raw_data["cayu_tau3_bridge"]["internal_tool_rounds"] == 1
    assert response.raw_data["cayu_tau3_bridge"]["staged_external_batches"] == 1
    assert response.raw_data["cayu_tau3_bridge"]["staged_external_calls"] == 1
    assert conversation.tool_names_by_call_id == {"external-staged": "lookup_account"}
    public_assistant = conversation.messages[-1]
    assert [
        part.tool_call_id for part in public_assistant.content if isinstance(part, ToolCallPart)
    ] == ["external-staged"]
    assert not any(isinstance(part, ProviderStatePart) for part in public_assistant.content)


def test_external_call_before_authorized_plan_call_is_withheld() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="external-too-early",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                ModelStreamEvent.tool_call(
                    id="controller-after-external",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "This private transition is not a prefix"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="external-reissued",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
        ]
    )
    handler = StagingInternalToolHandler()
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [as_tool(lookup_account)],
            internal_tool_handler=handler,
        )
    )

    assert [call.id for call in response.tool_calls] == ["external-reissued"]
    assert handler.calls == [
        (_INTERNAL_TEST_TOOL, {"note": "This private transition is not a prefix"})
    ]
    withheld = [
        part
        for message in provider.requests[1].messages
        for part in message.content
        if isinstance(part, ToolResultPart) and part.tool_call_id == "external-too-early"
    ]
    assert len(withheld) == 1
    assert withheld[0].is_error is True
    assert response.raw_data["cayu_tau3_bridge"]["staged_external_batches"] == 0
    assert response.raw_data["cayu_tau3_bridge"]["staged_external_calls"] == 0


def test_failed_plan_prefix_withholds_the_staged_external_call() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-1",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Attempt to authorize the account lookup"},
                ),
                ModelStreamEvent.tool_call(
                    id="external-withheld",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="external-reissued",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
        ]
    )
    handler = FailingStagingInternalToolHandler()
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [as_tool(lookup_account)],
            internal_tool_handler=handler,
        )
    )

    assert [call.id for call in response.tool_calls] == ["external-reissued"]
    withheld = [
        part
        for message in provider.requests[1].messages
        for part in message.content
        if isinstance(part, ToolResultPart) and part.tool_call_id == "external-withheld"
    ]
    assert len(withheld) == 1
    assert withheld[0].is_error is True
    assert "was not executed" in withheld[0].content
    assert conversation.tool_names_by_call_id == {"external-reissued": "lookup_account"}


def test_denied_staged_external_call_preserves_the_successful_private_transition() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="controller-1",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Authorize the account lookup"},
                ),
                ModelStreamEvent.tool_call(
                    id="external-denied",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="external-reissued",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
        ]
    )
    handler = StagingInternalToolHandler()
    gate = RejectOnceExternalToolGate()
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [as_tool(lookup_account)],
            internal_tool_handler=handler,
            external_tool_gate=gate,
        )
    )

    assert [call.id for call in response.tool_calls] == ["external-reissued"]
    assert handler.calls == [(_INTERNAL_TEST_TOOL, {"note": "Authorize the account lookup"})]
    assert gate.reviews == 2
    assert response.raw_data["cayu_tau3_bridge"]["external_admission_rejections"] == 1


def test_denied_external_call_is_corrected_before_tau_can_execute_it() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="denied-1",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="accepted-2",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
        ]
    )
    gate = RejectOnceExternalToolGate()
    combined_gate = CompositeExternalToolCallGate((gate, AllowExternalToolGate()))
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [as_tool(lookup_account)],
            external_tool_gate=combined_gate,
        )
    )

    assert [call.id for call in response.tool_calls] == ["accepted-2"]
    assert response.raw_data["cayu_tau3_bridge"]["external_admission_rounds"] == 1
    assert response.raw_data["cayu_tau3_bridge"]["external_admission_rejections"] == 1
    assert response.raw_data["cayu_tau3_bridge"]["external_rejection_codes"] == {"unclassified": 1}
    assert conversation.provider_call_ids == {"denied-1", "accepted-2"}
    assert conversation.tool_names_by_call_id == {"accepted-2": "lookup_account"}
    denied_result = provider.requests[1].messages[-1].content[0]
    assert isinstance(denied_result, ToolResultPart)
    assert denied_result.tool_call_id == "denied-1"
    assert denied_result.is_error is True
    assert "required sequencing is incomplete" in denied_result.content
    assert len(conversation.messages) == 2
    assert all(
        getattr(part, "tool_call_id", None) != "denied-1"
        for message in conversation.messages
        for part in message.content
    )


def test_denied_external_call_requires_the_configured_internal_correction() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="denied-1",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="plan-2",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Revise the rejected operation."},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.text_delta("The rejected operation is still pending."),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="plan-3",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "Replace the rejected operation."},
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="accepted-4",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
        ]
    )
    handler = RecordingInternalToolHandler()
    gate = CompositeExternalToolCallGate(
        (RejectOnceExternalToolGate(), AllowExternalToolGate()),
        denial_required_tool_names=(_INTERNAL_TEST_TOOL,),
    )
    terminal_gate = RejectTerminalOutputGate(
        required_tool_names=("lookup_account", _INTERNAL_TEST_TOOL)
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [as_tool(lookup_account)],
            internal_tool_handler=handler,
            external_tool_gate=gate,
            terminal_output_gate=terminal_gate,
        )
    )

    assert [call.id for call in response.tool_calls] == ["accepted-4"]
    assert handler.calls == [
        (_INTERNAL_TEST_TOOL, {"note": "Revise the rejected operation."}),
        (_INTERNAL_TEST_TOOL, {"note": "Replace the rejected operation."}),
    ]
    assert provider.requests[1].options["openai"] == {
        "parallel_tool_calls": False,
        "tool_choice": {"type": "function", "name": _INTERNAL_TEST_TOOL},
    }
    assert provider.requests[3].options["openai"] == {
        "parallel_tool_calls": False,
        "tool_choice": {"type": "function", "name": _INTERNAL_TEST_TOOL},
    }
    assert response.raw_data["cayu_tau3_bridge"]["external_admission_rejections"] == 1
    assert response.raw_data["cayu_tau3_bridge"]["terminal_gate_rejections"] == 1


def test_external_admission_limit_has_redacted_failure_diagnostics() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="denied-1",
                    name="lookup_account",
                    arguments={"account_id": "sensitive-account"},
                ),
                completed(),
            ]
        ]
    )
    bridge = CayuTauBridge(
        provider=provider,
        model="gpt-test",
        max_external_admission_rounds=0,
    )
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    with pytest.raises(BridgeProtocolError, match="tool-admission") as raised:
        run(
            bridge.generate(
                conversation,
                [as_tool(lookup_account)],
                external_tool_gate=RejectOnceExternalToolGate(),
            )
        )

    diagnostics = raised.value.diagnostics
    assert diagnostics is not None
    assert diagnostics.failure_code == "external_admission_limit"
    assert diagnostics.current_output_kind == "external_tools"
    assert diagnostics.current_tool_names == ("lookup_account",)
    assert diagnostics.external_admission_rejections == 1
    assert diagnostics.external_rejection_codes == {"unclassified": 1}
    assert "sensitive-account" not in json.dumps(diagnostics.as_record())


def test_denied_terminal_draft_is_replaced_by_required_tool_call() -> None:
    provider = ScriptedProvider(
        [
            [ModelStreamEvent.text_delta("Everything is done."), completed()],
            [
                ModelStreamEvent.tool_call(
                    id="change-1",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
        ]
    )
    gate = RejectTerminalOutputGate(required_tool_names=("lookup_account",))
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")
    bridge.append_tau_input(conversation, UserMessage.text("Look up account A-1."))

    response = run(
        bridge.generate(
            conversation,
            [as_tool(lookup_account)],
            terminal_output_gate=gate,
        )
    )

    assert response.content is None
    assert response.tool_calls[0].id == "change-1"
    assert gate.reviews == 1
    assert response.raw_data["cayu_tau3_bridge"]["terminal_gate_rounds"] == 1
    assert response.raw_data["cayu_tau3_bridge"]["terminal_gate_rejections"] == 1
    assert response.raw_data["cayu_tau3_bridge"]["terminal_rejection_codes"] == {
        "response_actionable_items": 1
    }
    trace = response.raw_data["cayu_tau3_bridge"]["provider_call_trace"]["calls"]
    assert [item["phase"] for item in trace] == ["initial", "terminal_readiness_correction"]
    assert trace[1]["forced_tool"] is True
    assert provider.requests[1].options["openai"] == {
        "parallel_tool_calls": False,
        "tool_choice": {"type": "function", "name": "lookup_account"},
    }
    forced_payload = build_openai_payload(provider.requests[1])
    assert forced_payload["parallel_tool_calls"] is False
    assert forced_payload["tool_choice"] == {
        "type": "function",
        "name": "lookup_account",
    }
    assert response.usage == {"prompt_tokens": 24, "completion_tokens": 6}
    correction = provider.requests[1].messages[-1]
    assert correction.role is MessageRole.SYSTEM
    assert "was not emitted" in correction.content[0].text
    assert "change_user_email" in correction.content[0].text
    assert all(
        "Everything is done." not in part.text
        for message in conversation.messages
        for part in message.content
        if hasattr(part, "text")
    )
    assert len(conversation.messages) == 3


def test_terminal_correction_fails_closed_when_provider_ignores_required_tool() -> None:
    provider = ScriptedProvider(
        [
            [ModelStreamEvent.text_delta("Everything is done."), completed()],
            [ModelStreamEvent.text_delta("I still think it is done."), completed()],
        ]
    )
    gate = RejectTerminalOutputGate(required_tool_names=("lookup_account",))
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    with pytest.raises(BridgeProtocolError, match="did not honor") as raised:
        run(
            bridge.generate(
                conversation,
                [as_tool(lookup_account)],
                terminal_output_gate=gate,
            )
        )

    assert gate.reviews == 1
    assert raised.value.diagnostics is not None
    assert raised.value.diagnostics.failure_code == "required_correction_tool_not_called"
    assert raised.value.diagnostics.terminal_gate_rejections == 1
    assert raised.value.diagnostics.terminal_rejection_codes == {"response_actionable_items": 1}
    assert provider.requests[1].options["openai"]["tool_choice"] == {
        "type": "function",
        "name": "lookup_account",
    }


def test_terminal_readiness_correction_is_strictly_bounded() -> None:
    provider = ScriptedProvider(
        [
            [ModelStreamEvent.text_delta("Done once."), completed()],
            [ModelStreamEvent.text_delta("Done twice."), completed()],
        ]
    )
    gate = RejectTerminalOutputGate()
    bridge = CayuTauBridge(
        provider=provider,
        model="gpt-test",
        max_terminal_gate_rounds=1,
    )
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    with pytest.raises(BridgeProtocolError, match="terminal-readiness correction limit") as raised:
        run(bridge.generate(conversation, [], terminal_output_gate=gate))

    assert gate.reviews == 2
    assert len(conversation.messages) == 2
    assert "was not emitted" in conversation.messages[-1].content[0].text
    assert raised.value.diagnostics is not None
    assert raised.value.diagnostics.failure_code == "terminal_readiness_limit"
    assert raised.value.diagnostics.terminal_gate_rounds == 1
    assert raised.value.diagnostics.terminal_gate_rejections == 1
    assert raised.value.diagnostics.terminal_rejection_codes == {"unclassified": 1}
    assert raised.value.diagnostics.current_output_kind == "terminal_text"
    assert len(raised.value.diagnostics.terminal_correction_fingerprints) == 1
    assert raised.value.diagnostics.current_output_fingerprint is not None
    assert (
        raised.value.diagnostics.current_output_fingerprint
        not in raised.value.diagnostics.terminal_correction_fingerprints
    )
    assert raised.value.diagnostics.as_record()["terminal_rejection_codes"] == {"unclassified": 1}


def test_terminal_readiness_budget_resets_after_verified_internal_progress() -> None:
    provider = ScriptedProvider(
        [
            [ModelStreamEvent.text_delta("Done before first update."), completed()],
            [
                ModelStreamEvent.tool_call(
                    id="plan-1",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "first-progress"},
                ),
                completed(),
            ],
            [ModelStreamEvent.text_delta("Done before second update."), completed()],
            [
                ModelStreamEvent.tool_call(
                    id="plan-2",
                    name=_INTERNAL_TEST_TOOL,
                    arguments={"note": "second-progress"},
                ),
                completed(),
            ],
            [ModelStreamEvent.text_delta("Everything is now done."), completed()],
        ]
    )
    handler = ProgressRecordingInternalToolHandler()
    gate = RejectTwiceThenAllowTerminalOutputGate()
    bridge = CayuTauBridge(
        provider=provider,
        model="gpt-test",
        max_terminal_gate_rounds=1,
    )
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    response = run(
        bridge.generate(
            conversation,
            [],
            internal_tool_handler=handler,
            terminal_output_gate=gate,
        )
    )

    assert response.content == "Everything is now done."
    assert gate.reviews == 3
    assert [arguments["note"] for _, arguments in handler.calls] == [
        "first-progress",
        "second-progress",
    ]
    diagnostics = response.raw_data["cayu_tau3_bridge"]
    assert diagnostics["terminal_gate_rounds"] == 2
    assert diagnostics["terminal_gate_rejections"] == 2
    assert diagnostics["terminal_gate_rounds_since_progress"] == 0
    assert diagnostics["terminal_rejection_codes"] == {"response_actionable_items": 2}


def test_provider_cannot_call_a_tool_that_was_not_offered() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="unknown-1",
                    name="invented_tool",
                    arguments={},
                ),
                completed(),
            ]
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    with pytest.raises(BridgeProtocolError, match="not offered"):
        run(bridge.generate(conversation, [as_tool(lookup_account)]))

    assert conversation.provider_call_ids == set()
    assert len(conversation.messages) == 1


def test_unknown_tool_result_is_rejected() -> None:
    provider = ScriptedProvider([])
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")

    with pytest.raises(BridgeProtocolError, match="unknown assistant call id"):
        bridge.append_tau_input(
            conversation,
            ToolMessage(
                id="missing",
                role="tool",
                requestor="assistant",
                content="result",
            ),
        )


def test_duplicate_tool_result_is_rejected_without_mutating_transcript() -> None:
    provider = ScriptedProvider([])
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(
        system_prompt="Follow policy.",
        message_history=[
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "lookup_account",
                        "arguments": {"account_id": "A-1"},
                    }
                ],
            ),
            ToolMessage(
                id="call-1",
                role="tool",
                requestor="assistant",
                content="ok",
            ),
        ],
    )
    message_count = len(conversation.messages)

    with pytest.raises(BridgeProtocolError, match="repeated resolved"):
        bridge.append_tau_input(
            conversation,
            ToolMessage(
                id="call-1",
                role="tool",
                requestor="assistant",
                content="ok again",
            ),
        )

    assert len(conversation.messages) == message_count


def test_multi_tool_result_validation_is_atomic() -> None:
    provider = ScriptedProvider([])
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(
        system_prompt="Follow policy.",
        message_history=[
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "lookup_account",
                        "arguments": {"account_id": "A-1"},
                    }
                ],
            )
        ],
    )
    message_count = len(conversation.messages)

    with pytest.raises(BridgeProtocolError, match="unknown assistant call id"):
        bridge.append_tau_input(
            conversation,
            MultiToolMessage(
                role="tool",
                tool_messages=[
                    ToolMessage(
                        id="call-1",
                        role="tool",
                        requestor="assistant",
                        content="ok",
                    ),
                    ToolMessage(
                        id="missing",
                        role="tool",
                        requestor="assistant",
                        content="bad",
                    ),
                ],
            ),
        )

    assert len(conversation.messages) == message_count
    assert conversation.resolved_call_ids == set()


def test_existing_assistant_tool_call_can_be_replayed() -> None:
    provider = ScriptedProvider([])
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    history = [
        AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "call_old",
                    "name": "lookup_account",
                    "arguments": {"account_id": "A-2"},
                }
            ],
        )
    ]

    conversation = bridge.new_conversation(
        system_prompt="Follow policy.",
        message_history=history,
    )
    bridge.append_tau_input(
        conversation,
        ToolMessage(
            id="call_old",
            role="tool",
            requestor="assistant",
            content="ok",
        ),
    )

    assert conversation.messages[-1].content[0].tool_name == "lookup_account"


class SlowProvider(ModelProvider):
    name = "slow"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        await asyncio.sleep(30)
        yield completed()


class DeadlineProvider(ModelProvider):
    name = "deadline"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        raise ModelStreamDeadlineError(
            provider=self.name,
            evidence=ProviderStreamDeadlineEvidence(
                deadline_kind=ProviderDeadlineKind.SEMANTIC_IDLE,
                configured_timeout_s=120,
                elapsed_s=120,
                last_progress_kind=None,
                last_progress_elapsed_s=None,
                last_progress_at=None,
            ),
        )
        yield completed()  # pragma: no cover - makes this an async generator


def test_gateway_persists_provider_error_event_snapshot(tmp_path: Path) -> None:
    provider = ScriptedProvider([[ModelStreamEvent.error("transient provider failure")]])
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")
    diagnostics_path = tmp_path / "attempts.jsonl"

    with (
        CayuModelGateway(
            bridge,
            attempt_diagnostics_recorder=AttemptDiagnosticsRecorder(diagnostics_path),
        ) as gateway,
        pytest.raises(BridgeProtocolError, match="provider failed"),
    ):
        gateway.generate(conversation, [])

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["failure_code"] == "provider_stream_protocol_failure"
    assert diagnostics["current_output_kind"] == "provider_call_in_progress"
    assert diagnostics["provider_call_count"] == 0


def test_gateway_persists_provider_stream_deadline_snapshot(tmp_path: Path) -> None:
    bridge = CayuTauBridge(provider=DeadlineProvider(), model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")
    diagnostics_path = tmp_path / "attempts.jsonl"

    with (
        CayuModelGateway(
            bridge,
            attempt_diagnostics_recorder=AttemptDiagnosticsRecorder(diagnostics_path),
        ) as gateway,
        pytest.raises(ModelStreamDeadlineError, match="semantic_idle"),
    ):
        gateway.generate(conversation, [])

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["failure_code"] == "provider_stream_semantic_idle_timeout"
    assert diagnostics["current_output_kind"] == "provider_call_in_progress"
    assert diagnostics["provider_call_count"] == 0
    assert diagnostics["internal_tool_rounds"] == 0


def test_gateway_timeout_cancels_without_committing_conversation(tmp_path: Path) -> None:
    bridge = CayuTauBridge(provider=SlowProvider(), model="gpt-test")
    conversation = bridge.new_conversation(system_prompt="Follow policy.")
    diagnostics_path = tmp_path / "attempts.jsonl"
    gateway = CayuModelGateway(
        bridge,
        completion_timeout_s=0.01,
        attempt_diagnostics_recorder=AttemptDiagnosticsRecorder(diagnostics_path),
    )

    with pytest.raises(TimeoutError, match="exceeded"):
        gateway.generate(conversation, [])
    gateway.close()

    assert len(conversation.messages) == 1
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["failure_code"] == "gateway_completion_timeout"
    assert diagnostics["current_output_kind"] == "provider_call_in_progress"
    assert diagnostics["provider_call_count"] == 0
    assert diagnostics["internal_tool_rounds"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        gateway.generate(conversation, [])


def test_agent_accepts_real_banking_policy_and_bm25_tool_schemas() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="plan-greeting",
                    name=SYNC_WORK_PLAN_TOOL,
                    arguments={
                        "items": [
                            {
                                "id": "greeting",
                                "goal": "Ask how the customer needs help",
                                "completion_mode": "information",
                                "status": "awaiting_customer",
                                "outcome": "unresolved",
                                "depends_on": [],
                                "next_operation": {
                                    "owner": "customer",
                                    "kind": "customer_response",
                                    "tool_name": "",
                                    "arguments": {},
                                    "summary": "Describe the banking request.",
                                },
                                "evidence": "The customer has only greeted the agent.",
                                "evidence_refs": [],
                            }
                        ]
                    },
                ),
                completed(),
            ],
            [ModelStreamEvent.text_delta("How can I help?"), completed(end_turn=True)],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    environment = get_environment(retrieval_variant="bm25")
    gateway = CayuModelGateway(bridge)
    agent = CayuBridgeAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        gateway=gateway,
    )

    try:
        state = agent.get_init_state()
        response, returned_state = agent.generate_next_message(UserMessage.text("Hello"), state)
    finally:
        gateway.close()

    assert response.content == "How can I help?"
    assert returned_state is state
    assert "<policy>" in agent.system_prompt
    normalized_prompt = " ".join(agent.system_prompt.split())
    assert "A customer-side discoverable tool" in normalized_prompt
    assert "owner customer, kind tool" in normalized_prompt
    assert "give_discoverable_user_tool with only discoverable_tool_name" in normalized_prompt
    assert "plan is an outcome contract, not a scratchpad" in normalized_prompt
    assert "do not serialize one accepted item change" in normalized_prompt
    assert "KB_search" in {tool["name"] for tool in provider.requests[0].tools}
    assert SYNC_WORK_PLAN_TOOL in {tool["name"] for tool in provider.requests[0].tools}
    assert "<observation-ledger>" in provider.requests[0].messages[-1].content[0].text
    assert response.raw_data["cayu_tau3_workflow"]["revision"] == 1
    assert response.raw_data["cayu_tau3_workflow"]["last_response_disposition"] == {
        "items": [{"item_id": "greeting", "kind": "ask_customer"}],
        "plan_revision": 1,
    }


def test_agent_derives_response_disposition_before_returning_text() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="plan-1",
                    name=SYNC_WORK_PLAN_TOOL,
                    arguments={
                        "items": [
                            {
                                "id": "email",
                                "goal": "Obtain the replacement email address",
                                "completion_mode": "operation",
                                "status": "awaiting_customer",
                                "outcome": "unresolved",
                                "depends_on": [],
                                "next_operation": {
                                    "owner": "customer",
                                    "kind": "customer_response",
                                    "tool_name": "",
                                    "arguments": {},
                                    "summary": "Provide the replacement email address.",
                                },
                                "evidence": "The replacement address is still required.",
                                "evidence_refs": [],
                            }
                        ]
                    },
                ),
                completed(),
            ],
            [
                ModelStreamEvent.text_delta("What replacement email address should I use?"),
                completed(end_turn=True),
            ],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    gateway = CayuModelGateway(bridge)
    agent = CayuBridgeAgent(
        tools=[],
        domain_policy="Ask for the replacement email before changing it.",
        gateway=gateway,
    )

    try:
        state = agent.get_init_state()
        response, _ = agent.generate_next_message(
            UserMessage.text("Please change my email address."),
            state,
        )
    finally:
        gateway.close()

    assert response.content == "What replacement email address should I use?"
    assert response.raw_data["cayu_tau3_bridge"]["internal_tool_rounds"] == 1
    assert response.raw_data["cayu_tau3_workflow"]["last_response_disposition"] == {
        "items": [{"item_id": "email", "kind": "ask_customer"}],
        "plan_revision": 1,
    }


def test_agent_composes_workflow_and_banking_admission_for_external_call() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="plan-1",
                    name=SYNC_WORK_PLAN_TOOL,
                    arguments={
                        "items": [
                            {
                                "id": "lookup",
                                "goal": "Look up account A-1",
                                "completion_mode": "information",
                                "status": "in_progress",
                                "outcome": "unresolved",
                                "depends_on": [],
                                "next_operation": {
                                    "owner": "agent",
                                    "kind": "tool",
                                    "tool_name": "lookup_account",
                                    "arguments": {"account_id": "A-1"},
                                    "summary": "Look up the requested account.",
                                },
                                "evidence": "",
                                "evidence_refs": [],
                            }
                        ]
                    },
                ),
                completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="lookup-1",
                    name="lookup_account",
                    arguments={"account_id": "A-1"},
                ),
                completed(),
            ],
        ]
    )
    bridge = CayuTauBridge(provider=provider, model="gpt-test")
    gateway = CayuModelGateway(bridge)
    agent = CayuBridgeAgent(
        tools=[as_tool(lookup_account)],
        domain_policy="Look up only the requested account.",
        gateway=gateway,
    )

    try:
        state = agent.get_init_state()
        response, _ = agent.generate_next_message(
            UserMessage.text("Please look up account A-1."),
            state,
        )
    finally:
        gateway.close()

    assert response.tool_calls[0].id == "lookup-1"
    assert response.raw_data["cayu_tau3_bridge"]["external_admission_rejections"] == 0
    assert response.raw_data["cayu_tau3_workflow"]["revision"] == 1
    assert response.raw_data["cayu_tau3_admission"]["reviewed_calls"] == 1
