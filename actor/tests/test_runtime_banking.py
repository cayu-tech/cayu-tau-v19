from __future__ import annotations

import asyncio
import json

import pytest
from cayu import ModelStreamEvent, ScriptedModelProvider, SQLiteSessionStore, ToolResultPart
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.environment.tool import Tool

from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL, BankingKnowledgeCorpus
from cayu_tau3_banking.ledger import ToolActionStatus
from cayu_tau3_banking.runtime_agent import CayuRuntimeAgent, RuntimeAgentError
from cayu_tau3_banking.runtime_banking import build_banking_runtime_resources


def text(value):
    return [
        ModelStreamEvent.text_delta(value),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def call(name, arguments, call_id="plan"):
    return [
        ModelStreamEvent.tool_call(id=call_id, name=name, arguments=arguments),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def work(*, done=False, operation=False, reference="limits", tool="host_action"):
    return {
        "id": "outcome",
        "goal": "Execute the requested action" if operation else "Explain the purchase limit",
        "completion_mode": "operation" if operation else "information",
        "status": "completed" if done else "pending",
        "outcome": ("succeeded" if operation else "answered") if done else "unresolved",
        "depends_on": [],
        "next_operation": {
            "owner": "none" if done else "agent",
            "kind": "none" if done else ("tool" if operation else "knowledge_search"),
            "tool_name": tool if operation and not done else "",
            "arguments": {"value": 7} if operation and not done else {},
            "summary": "" if done else "Perform the next operation",
        },
        "evidence": "Checked evidence" if done else "",
        "evidence_refs": (
            [{"kind": "tool_result", "reference_id": reference}]
            if operation
            else [{"kind": "knowledge_revision", "reference_id": reference, "version": "1"}]
        )
        if done
        else [],
    }


def host_action(value: int) -> str:
    """Perform a banking action.

    Args:
        value: Requested value.
    """
    raise AssertionError("Only the test host may execute environment tools")


def assembly(
    tmp_path,
    steps,
    *,
    persistent=False,
    compaction=None,
    native_recall=False,
    lexical_navigation=False,
    planning_mode="exact",
    advice_mode="controller",
    controller_mode="enforced",
    decision_packets=False,
    task_memory_mode=None,
    task_memory_directory=None,
    focused_policy_file=None,
    # Functional multi-round tests are not latency benchmarks. Keep enough room
    # for lower-clocked CI/VM CPUs; deadline tests use their own explicit limits.
    callback_timeout_s=90,
    configure_corpus=None,
    host_tools=None,
    **kwargs,
):
    (tmp_path / "limits.json").write_text(
        json.dumps(
            {
                "id": "limits",
                "title": "Purchase limit",
                "content": "The purchase limit is $4,500. host_action(value: int) performs the requested action.",
            }
        )
    )
    corpus = BankingKnowledgeCorpus.from_directory(tmp_path)
    if configure_corpus is not None:
        configure_corpus(corpus)
    provider = ScriptedModelProvider(steps)
    captured = {}

    def factory(broker, specs):
        store = SQLiteSessionStore(tmp_path / "runtime.sqlite3") if persistent else None
        resources = build_banking_runtime_resources(
            broker,
            specs,
            provider=provider,
            model="scripted",
            domain_policy="Follow public policy.",
            memory_runtime=BankingMemoryRuntime(corpus),
            session_store=store,
            aclose=store.close if store is not None else None,
            compaction=compaction,
            native_recall=native_recall,
            lexical_navigation=lexical_navigation,
            planning_mode=planning_mode,
            advice_mode=advice_mode,
            controller_mode=controller_mode,
            decision_packets=decision_packets,
            task_memory_mode=task_memory_mode,
            task_memory_directory=task_memory_directory,
            focused_policy_file=focused_policy_file,
        )
        captured["state"] = resources.observer
        captured["app"] = resources.app
        return resources

    agent = CayuRuntimeAgent(
        tools=[Tool(host_action)] if host_tools is None else host_tools,
        domain_policy="Follow public policy.",
        resource_factory=factory,
        callback_timeout_s=callback_timeout_s,
        **kwargs,
    )
    return agent, provider, captured


def offered(request):
    return {tool["name"] for tool in request.tools}


def test_public_parameter_contract_blocks_before_host_dispatch_and_allows_repair(tmp_path):
    import hashlib
    from types import SimpleNamespace

    from cayu_tau3_banking.public_argument_contracts import (
        PublicArgumentContract,
        PublicArgumentContracts,
    )

    def configure(corpus):
        source = "host_action parameter value must be an integer no greater than 7."
        contract = PublicArgumentContract(
            "host_action",
            "synthetic_api",
            1,
            hashlib.sha256(source.encode()).hexdigest(),
            source,
            json.dumps({"type": "object", "properties": {"value": {"maximum": 7}}}),
        )
        corpus.public_argument_contracts = PublicArgumentContracts(
            {"synthetic_api": SimpleNamespace(revision=1, text=source)}, [contract]
        )

    wrong = work(operation=True)
    wrong["next_operation"]["arguments"] = {"value": 9}
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [wrong]}),
            call("host_action", {"value": 9}, "invalid-representation"),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [work(operation=True)]}, "repair"),
            call("host_action", {"value": 7}, "correct-representation"),
        ],
        configure_corpus=configure,
        callback_timeout_s=60,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Perform the authorized operation."), state
        )
        assert len(provider.requests) == 4
        assert [c.arguments for c in reply.tool_calls] == [{"value": 7}]
        assert [a.call_id for a in captured["state"].ledger.actions] == ["correct-representation"]
        assert any(
            "Public parameter contract rejected" in part.text
            for request in provider.requests
            for message in request.messages
            for part in message.content
            if part.type == "text"
        )
    finally:
        state.stop()


def test_repeated_denied_rounds_stop_without_executing_host_tools(tmp_path):
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [work(operation=True)]}),
            *[call("host_action", {"value": 9}, f"denied-{i}") for i in range(12)],
        ],
        # This tests a deterministic round limit, not the adapter wall-clock limit.
        # Keep slow/loaded CI from stopping before all eight denials are observed.
        callback_timeout_s=60,
    )
    state = agent.get_init_state()
    try:
        with pytest.raises(RuntimeAgentError, match="AdmissionStallError"):
            agent.generate_next_message(UserMessage.text("Perform the requested action."), state)
        assert len(provider.requests) == 9  # Plan, then eight rejected rounds.
        assert not captured["state"].ledger.actions
        assert any(
            "Tool or response admission rejected" in part.text
            for request in provider.requests
            for message in request.messages
            for part in message.content
            if part.type == "text"
        )
    finally:
        state.stop()


def test_native_memory_plan_and_admitted_reply_do_not_escape_as_tau_tools(tmp_path):
    agent, provider, captured = assembly(
        tmp_path,
        [
            text("Premature draft."),
            call(SYNC_WORK_PLAN_TOOL, {"items": [work(done=True)]}),
            text("The purchase limit is $4,500."),
        ],
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("What is the purchase limit?"), state
        )
        assert reply.content == "The purchase limit is $4,500."
        assert not reply.tool_calls
        bank = captured["state"]
        assert bank.ledger.customer_messages == ["What is the purchase limit?"]
        assert len(bank.ledger.customer_message_ids) == 1  # Correction isn't customer evidence.
        assert not bank.ledger.actions  # Private tools aren't environment effects.
        assert offered(provider.requests[0]) == {SYNC_WORK_PLAN_TOOL}
        assert SYNC_WORK_PLAN_TOOL not in offered(provider.requests[-1])
        assert UPDATE_WORK_PLAN_TOOL in offered(provider.requests[-1])
        assert any(
            "$4,500" in part.text
            for message in provider.requests[0].messages
            for part in message.content
            if part.type == "text"
        )
        assert bank.memory.export_state()["last_recall"]["injected_count"] == 1
    finally:
        state.stop()


@pytest.mark.parametrize("error", [False, True])
@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "sqlite"])
def test_host_evidence_uses_runtime_call_identity_and_never_claims_failed_write(
    tmp_path, error, persistent
):
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [work(operation=True)]}),
            call("host_action", {"value": 7}, "model-action"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {"items": [work(done=True, operation=True, reference="model-action")]},
                "update",
            ),
            text("Done."),
        ],
        max_steps=4,
        persistent=persistent,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Execute the requested action."), state
        )
        (external,) = reply.tool_calls
        assert external.name == "host_action"
        assert external.id != "model-action"
        bank = captured["state"]
        assert bank.ledger.tool_action_status("model-action") is ToolActionStatus.PENDING
        checkpoint = asyncio.run_coroutine_threadsafe(
            captured["app"].session_store.load_checkpoint(state._conversation.session_id),
            state._loop,
        ).result(10)
        assert checkpoint["pending_tool_round"]
        result = ToolMessage(
            role="tool", id=external.id, content="failed" if error else "completed", error=error
        )
        if error:
            with pytest.raises(RuntimeAgentError):
                agent.generate_next_message(result, state)
            assert bank.ledger.tool_action_status("model-action") is ToolActionStatus.FAILED
        else:
            reply, _ = agent.generate_next_message(result, state)
            assert reply.content == "Done."
            assert bank.ledger.tool_action_status("model-action") is ToolActionStatus.SUCCEEDED
        assert bank.ledger.tool_action_status(external.id) is None
        observed = [
            part
            for message in provider.requests[2].messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        ]
        assert any(
            part.tool_call_id == "model-action" and part.is_error == error for part in observed
        )
    finally:
        state.stop()


def test_unplanned_action_does_not_reach_host_or_become_success_evidence(tmp_path):
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [work()]}),
            call("host_action", {"value": 7}, "unplanned"),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [work(done=True)]}, "update"),
            text("The purchase limit is $4,500."),
        ],
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Explain the purchase limit."), state
        )
        assert not reply.tool_calls
        assert not captured["state"].ledger.actions
        results = [
            part
            for message in provider.requests[2].messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        ]
        assert any(part.tool_call_id == "unplanned" and part.is_error for part in results)
    finally:
        state.stop()


@pytest.mark.parametrize("planning_mode", ["exact", "outcomes"])
def test_invalid_host_arguments_are_not_recorded_as_environment_evidence(tmp_path, planning_mode):
    invalid = work(operation=True)
    invalid["next_operation"]["arguments"] = {"value": "not an integer"}
    if planning_mode == "outcomes":
        invalid.pop("next_operation")
    agent, _, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [invalid]}),
            call("host_action", {"value": "not an integer"}, "invalid-host-call"),
            text("Unsupported success claim."),
        ],
        max_steps=3,
        planning_mode=planning_mode,
    )
    state = agent.get_init_state()
    try:
        with pytest.raises(RuntimeAgentError):
            agent.generate_next_message(UserMessage.text("Execute the requested action."), state)
        assert not captured["state"].ledger.actions
    finally:
        state.stop()


def test_refresh_context_does_not_reset_search_or_private_action_lifecycle(tmp_path):
    agent, _, captured = assembly(tmp_path, [text("unused")])
    state = agent.get_init_state()
    bank = captured["state"]

    async def scenario():
        await bank.customer("Explain the purchase limit.")
        await bank.memory.prepare_context()
        assert not (await bank.memory.invoke(SYNC_WORK_PLAN_TOOL, {"items": [work()]})).is_error
        assert not (
            await bank.memory.invoke(KNOWLEDGE_SEARCH_TOOL, {"query": "purchase limit", "limit": 1})
        ).is_error
        before = bank.memory.export_state()
        await bank.memory.runtime_context()
        await bank.memory.runtime_context()
        after = bank.memory.export_state()
        assert after["knowledge_searches"] == before["knowledge_searches"] == 1
        assert (
            after["knowledge_searches_for_current_customer_message"]
            == before["knowledge_searches_for_current_customer_message"]
        )

    try:
        asyncio.run_coroutine_threadsafe(scenario(), state._loop).result(10)
    finally:
        state.stop()


def test_duplicate_actions_in_one_native_round_have_one_environment_effect(tmp_path):
    double = [
        ModelStreamEvent.tool_call(id=call_id, name="host_action", arguments={"value": 7})
        for call_id in ("first-action", "duplicate-action")
    ] + [ModelStreamEvent.completed({"finish_reason": "tool_calls"})]
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [work(operation=True)]}),
            double,
            call(
                UPDATE_WORK_PLAN_TOOL,
                {"items": [work(done=True, operation=True, reference="first-action")]},
                "update",
            ),
            text("Done."),
        ],
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Execute the requested action."), state
        )
        assert len(reply.tool_calls) == 1
        reply, _ = agent.generate_next_message(
            ToolMessage(
                role="tool",
                id=reply.tool_calls[0].id,
                content="completed",
            ),
            state,
        )
        assert reply.content == "Done."
        assert [action.call_id for action in captured["state"].ledger.actions] == ["first-action"]
        results = [
            part
            for message in provider.requests[2].messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        ]
        assert any(part.tool_call_id == "duplicate-action" and part.is_error for part in results)
    finally:
        state.stop()


def test_plan_change_in_same_round_revokes_previously_authorized_host_action(tmp_path):
    revised = work(operation=True)
    revised["next_operation"]["arguments"] = {"value": 8}
    mixed = call(UPDATE_WORK_PLAN_TOOL, {"items": [revised]}, "revise")[:-1] + call(
        "host_action", {"value": 7}, "stale"
    )
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [work(operation=True)]}),
            mixed,
            text("Invalid completion."),
        ],
        max_steps=3,
    )
    state = agent.get_init_state()
    try:
        with pytest.raises(RuntimeAgentError):
            agent.generate_next_message(UserMessage.text("Execute the requested action."), state)
        assert not captured["state"].ledger.actions
        results = [
            part
            for message in provider.requests[2].messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        ]
        assert any(part.tool_call_id == "stale" and part.is_error for part in results)
    finally:
        state.stop()


def test_customer_handoff_resumes_same_runtime_and_keeps_evidence_sequence(tmp_path):
    waiting = work()
    waiting["status"] = "awaiting_customer"
    waiting["evidence"] = "The customer has not identified the account."
    waiting["evidence_refs"] = [{"kind": "customer_message", "reference_id": "customer-message:1"}]
    waiting["next_operation"] = {
        "owner": "customer",
        "kind": "customer_response",
        "tool_name": "",
        "arguments": {},
        "summary": "Ask which account the customer means",
    }
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [waiting]}),
            text("Which account do you mean?"),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [work(done=True)]}, "update"),
            text("The purchase limit is $4,500."),
        ],
    )
    state = agent.get_init_state()
    try:
        first, _ = agent.generate_next_message(
            UserMessage.text("Explain the purchase limit."), state
        )
        assert first.content == "Which account do you mean?"
        second, _ = agent.generate_next_message(UserMessage.text("The Sapphire account."), state)
        assert second.content == "The purchase limit is $4,500."
        assert captured["state"].ledger.customer_message_ids == [
            "customer-message:1",
            "customer-message:2",
        ]
        texts = [
            part.text
            for message in provider.requests[-1].messages
            for part in message.content
            if part.type == "text"
        ]
        assert "Which account do you mean?" in texts
        assert "The Sapphire account." in texts
    finally:
        state.stop()


def test_banking_assembly_runs_through_real_tau_orchestrator(tmp_path):
    from tau2.data_model.tasks import Task, UserScenario
    from tau2.environment.db import DB
    from tau2.environment.environment import Environment
    from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.user.user_simulator_base import HalfDuplexUser

    class Counter(DB):
        value: int = 0

    class EnvironmentTools(ToolKitBase):
        @is_tool(ToolType.WRITE)
        def host_action(self, value: int) -> str:
            """Perform a banking action.

            Args:
                value: Requested value.
            """
            self.db.value += value
            return "completed"

    class Customer(HalfDuplexUser):
        def get_init_state(self, message_history=None):
            return 0

        def generate_next_message(self, message, state):
            return UserMessage.text(
                "Execute the requested action." if state == 0 else "###STOP###"
            ), state + 1

    agent, _, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [work(operation=True)]}),
            call("host_action", {"value": 7}, "model-action"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {"items": [work(done=True, operation=True, reference="model-action")]},
                "update",
            ),
            text("Done."),
        ],
    )
    toolkit = EnvironmentTools(Counter())
    orchestrator = Orchestrator(
        domain="synthetic",
        agent=agent,
        user=Customer(),
        environment=Environment("synthetic", "Follow public policy.", tools=toolkit),
        task=Task(id="synthetic-banking-runtime", user_scenario=UserScenario(instructions="Test.")),
        validate_communication=True,
        max_steps=10,
    )
    result = orchestrator.run()
    assert result.termination_reason.value == "user_stop"
    assert toolkit.db.value == 7
    calls = [
        call
        for message in result.messages
        if isinstance(message, AssistantMessage)
        for call in (message.tool_calls or [])
    ]
    assert len(calls) == 1
    assert calls[0].name == "host_action"
    assert not orchestrator.agent_state._thread.is_alive()
    assert captured["state"].controller.export_state()["items"][0]["outcome"] == "succeeded"
