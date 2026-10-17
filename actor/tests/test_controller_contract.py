"""Regression contracts for the optional-workflow runtime."""

from types import SimpleNamespace

import pytest
from tau2.data_model.message import UserMessage
from test_admission import confirm_effect, proposed
from test_outcome_controller import outcome
from test_runtime_banking import assembly, call, text

from cayu_tau3_banking.admission import BankingToolAdmissionGate
from cayu_tau3_banking.controller import (
    SYNC_WORK_PLAN_TOOL,
    UPDATE_WORK_PLAN_TOOL,
    BankingWorkflowController,
)
from cayu_tau3_banking.controller_contract import ExecutionContractGate
from cayu_tau3_banking.ledger import ConversationLedger
from cayu_tau3_banking.public_argument_contracts import PublicArgumentContracts
from cayu_tau3_banking.savings_interest_actions import SAVINGS_CREDIT_TOOL


def candidate(tmp_path, steps):
    return assembly(
        tmp_path,
        steps,
        controller_mode="advisory",
        planning_mode="outcomes",
        advice_mode="lean",
    )


def test_native_host_action_does_not_require_model_authored_plan(tmp_path):
    agent, provider, captured = candidate(
        tmp_path, [call("host_action", {"value": 7}, "independent")]
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Do the work."), state)
        assert [(c.name, c.arguments) for c in reply.tool_calls] == [("host_action", {"value": 7})]
        assert len(provider.requests) == 1
        assert len(captured["state"].ledger.actions) == 1
    finally:
        state.stop()


def test_model_authored_comparison_state_cannot_veto_unrelated_call(tmp_path, monkeypatch):
    def forbidden_review(*args, **kwargs):
        raise AssertionError("Model-authored workflow cannot authorize execution in this mode")

    monkeypatch.setattr(BankingWorkflowController, "review", forbidden_review)
    agent, _provider, _captured = candidate(tmp_path, [call("host_action", {"value": 7}, "read")])
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Do the work."), state)
        assert [c.arguments for c in reply.tool_calls] == [{"value": 7}]
    finally:
        state.stop()


def test_partial_reply_leaves_unfinished_checklist_unfinished(tmp_path):
    agent, _provider, captured = candidate(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}),
            text("I still need your confirmation; the operation is not complete."),
        ],
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Please help."), state)
        assert "not complete" in reply.content
        items = captured["state"].controller.export_state()["items"]
        assert len(items) == 1
        assert items[0]["status"] != "completed"
        assert not captured["state"].ledger.actions
    finally:
        state.stop()


def test_native_reply_survives_eight_invalid_bookkeeping_updates(tmp_path):
    agent, provider, captured = candidate(
        tmp_path,
        [call(UPDATE_WORK_PLAN_TOOL, {"items": []}, f"bad-plan-{i}") for i in range(8)]
        + [text("I have not completed the operation. I need clarification.")],
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Please help."), state)
        assert "not completed" in reply.content
        assert len(provider.requests) == 9
        assert not captured["state"].ledger.actions
    finally:
        state.stop()


def test_native_host_schema_still_blocks_and_allows_repair(tmp_path):
    agent, provider, captured = candidate(
        tmp_path,
        [
            call("host_action", {"unrecognized": 7}, "invalid"),
            call("host_action", {"value": 7}, "valid"),
        ],
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Do the work."), state)
        assert [(c.name, c.arguments) for c in reply.tool_calls] == [("host_action", {"value": 7})]
        assert len(provider.requests) == 2
        assert [a.call_id for a in captured["state"].ledger.actions] == ["valid"]
    finally:
        state.stop()


def test_authoritative_gate_preserves_verification_and_unlock():
    gate = BankingToolAdmissionGate(ConversationLedger(), authoritative_only=True)
    for tool, arguments in (
        ("get_credit_card_accounts_by_user", {"user_id": "u"}),
        ("call_discoverable_agent_tool", {"agent_tool_name": "private_1234", "arguments": "{}"}),
    ):
        assert not gate.review((proposed("c", tool, **arguments),))[0].allowed


@pytest.mark.parametrize("reason", ["made_up", [], None])
def test_transfer_reason_schema_remains_required(reason):
    gate = BankingToolAdmissionGate(ConversationLedger(), authoritative_only=True)
    assert not gate.review(
        (proposed("c", "transfer_to_human_agents", reason=reason, summary="Please escalate."),)
    )[0].allowed


def test_safe_reread_is_not_a_policy_violation():
    ledger = ConversationLedger()
    confirm_effect(ledger, "first", "get_current_time", result="2026-09-17")
    gate = BankingToolAdmissionGate(ledger, authoritative_only=True)
    assert gate.review((proposed("second", "get_current_time"),))[0].allowed


def test_transfer_does_not_require_matching_customer_keywords():
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("A real employee should take over this conversation."))
    gate = BankingToolAdmissionGate(ledger, authoritative_only=True)
    assert gate.review(
        (
            proposed(
                "transfer",
                "transfer_to_human_agents",
                reason="customer_requests_human_no_specific_reason",
                summary="Customer requested a human.",
            ),
        )
    )[0].allowed


def test_duplicate_effect_is_still_rejected():
    ledger = ConversationLedger()
    confirm_effect(
        ledger, "first", "unlock_discoverable_agent_tool", agent_tool_name="specialized_action_1234"
    )
    gate = BankingToolAdmissionGate(
        ledger, authoritative_only=True, require_knowledge_evidence=False
    )
    assert not gate.review(
        (
            proposed(
                "second",
                "unlock_discoverable_agent_tool",
                agent_tool_name="specialized_action_1234",
            ),
        )
    )[0].allowed


def test_financial_duplicate_compares_parsed_arguments():
    ledger = ConversationLedger()
    confirm_effect(
        ledger,
        "first",
        "call_discoverable_agent_tool",
        agent_tool_name=SAVINGS_CREDIT_TOOL,
        arguments='{"account_id":"a","amount":12,"credit_type":"interest_correction"}',
    )
    corpus = SimpleNamespace(public_argument_contracts=PublicArgumentContracts({}, []), document_directory=lambda: ())
    gate = ExecutionContractGate(corpus, ledger)
    decision = gate.review(
        (
            proposed(
                "second",
                "call_discoverable_agent_tool",
                agent_tool_name=SAVINGS_CREDIT_TOOL,
                arguments='{ "credit_type": "interest_correction", "amount": 12.0, "account_id": "a" }',
            ),
        )
    )[0]
    assert not decision.allowed


@pytest.mark.parametrize("arguments", ['{"x":1,"x":2}', '{"x":NaN}', "[]"])
def test_nested_argument_contract_rejects_ambiguous_json(arguments):
    corpus = SimpleNamespace(public_argument_contracts=PublicArgumentContracts({}, []), document_directory=lambda: ())
    decision = ExecutionContractGate(corpus, ConversationLedger()).review(
        (
            proposed(
                "c",
                "call_discoverable_agent_tool",
                agent_tool_name="synthetic_1234",
                arguments=arguments,
            ),
        )
    )[0]
    assert not decision.allowed


def test_conflicting_record_does_not_discard_other_accounts():
    ledger = ConversationLedger()
    result = (
        "Found 2 record(s) in 'credit_card_accounts':\n\n"
        "1. Record ID: old\n   account_id: old\n   user_id: user-7\n"
        "   card_type: Gold\n   account_status: ACTIVE\n   status: CLOSED\n\n"
        "2. Record ID: new\n   account_id: new\n   user_id: user-7\n"
        "   card_type: EcoCard\n   account_status: ACTIVE"
    )
    confirm_effect(
        ledger, "read", "get_credit_card_accounts_by_user", user_id="user-7", result=result
    )
    accounts = ledger.credit_card_accounts("user-7")
    assert accounts is not None
    assert [(a.account_id, a.status) for a in accounts] == [("old", "UNKNOWN"), ("new", "ACTIVE")]
    assert accounts[0].account_status == "ACTIVE"
    assert accounts[0].record_status == "CLOSED"
    confirm_effect(
        ledger,
        "refresh",
        "get_credit_card_accounts_by_user",
        user_id="user-7",
        result="Unrecognized response",
    )
    assert ledger.credit_card_accounts("user-7") is None
