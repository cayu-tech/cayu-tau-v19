"""Synthetic debit-resource conditions, independent of benchmark tasks."""

import json

import pytest
from cayu import ToolCallPart
from tau2.data_model.message import AssistantMessage
from test_case_conditions import POLICY, condition, execute, obligation, patch, read, setup

from cayu_tau3_banking.controller import BankingWorkflowController
from cayu_tau3_banking.ledger import ConversationLedger


def card_read(
    ledger,
    *,
    call_id="card-read",
    card_id="debit-a",
    account_id="account-a",
    user="person-a",
    status="ACTIVE",
    error=False,
    records=None,
):
    value = (
        [
            {
                "card_id": card_id,
                "account_id": account_id,
                "user_id": user,
                "status": status,
                "card_number_last_4": "1234",
            }
        ]
        if records is None
        else records
    )
    execute(
        ledger,
        call_id,
        "call_discoverable_agent_tool",
        {
            "agent_tool_name": "get_debit_cards_by_account_id_7823",
            "arguments": json.dumps({"account_id": account_id}),
        },
        json.dumps(value),
        error=error,
    )


def card_call(card_id="debit-a", **extra):
    return ToolCallPart(
        tool_call_id="effect",
        tool_name="call_discoverable_agent_tool",
        arguments={
            "agent_tool_name": "apply_adjustment",
            "arguments": json.dumps({"card_id": card_id, **extra}),
        },
    )


def test_parent_prerequisite_can_target_its_observed_child_card():
    controller, ledger, _ = setup(obligations=[obligation(status="ready")])
    card_read(ledger)
    assert controller.review([card_call()])[0].allowed
    assert not controller.review([card_call("other-card")])[0].allowed
    assert not controller.review([card_call(user_id="other-person")])[0].allowed


@pytest.mark.parametrize(
    "invalid", ["wrong_owner", "wrong_parent", "failed", "stale", "removed", "malformed"]
)
def test_parent_join_requires_current_authoritative_relationship(invalid):
    controller, ledger, _ = setup(obligations=[obligation(status="ready")])
    if invalid == "wrong_owner":
        card_read(ledger, user="other-person")
    elif invalid == "wrong_parent":
        card_read(ledger, account_id="other-account")
    elif invalid == "failed":
        card_read(ledger, error=True)
    elif invalid == "malformed":
        card_read(ledger, records=[{"card_id": "debit-a", "account_id": "account-a"}])
    else:
        card_read(ledger)
        if invalid == "stale":
            execute(
                ledger,
                "write-card",
                "call_discoverable_agent_tool",
                {"agent_tool_name": "change_card", "arguments": json.dumps({"card_id": "debit-a"})},
            )
        else:
            card_read(ledger, call_id="new-read", records=[])
    assert not controller.review([card_call()])[0].allowed


def test_debit_card_is_not_misrepresented_as_a_credit_card():
    controller, ledger, _ = setup(obligations=[])
    card_read(ledger)
    debit = condition(
        resource_kind="debit_card",
        resource_id="debit-a",
        observed_in="card-read",
        checks=[
            {"field": "status", "operator": "equal", "value": "ACTIVE"},
            {"field": "user_id", "operator": "equal", "value": "person-a"},
        ],
    )
    result = patch(
        controller,
        obligations=[
            obligation(
                status="ready",
                condition=debit,
                evidence_refs=[POLICY, {"kind": "tool_result", "reference_id": "card-read"}],
            )
        ],
    )
    assert not result.is_error, result.content
    assert controller.review([card_call(user_id="person-a")])[0].allowed
    assert ledger.resource_evidence("card-read", "credit_card", "debit-a") is None
    assert ledger.resource_evidence("card-read", "debit_card", "debit-a") is not None


def test_parent_mutation_invalidates_debit_observation():
    ledger = ConversationLedger()
    read(ledger)
    card_read(ledger)
    assert ledger.resource_evidence("card-read", "debit_card", "debit-a") is not None
    execute(
        ledger,
        "close-account",
        "call_discoverable_agent_tool",
        {"agent_tool_name": "change_account", "arguments": json.dumps({"account_id": "account-a"})},
    )
    assert ledger.resource_evidence("card-read", "debit_card", "debit-a") is None


def test_child_mutation_conservatively_invalidates_linked_parent_observation():
    ledger = ConversationLedger()
    read(ledger)
    card_read(ledger)
    execute(
        ledger,
        "effect",
        "call_discoverable_agent_tool",
        {"agent_tool_name": "change_card", "arguments": json.dumps({"card_id": "debit-a"})},
    )
    assert ledger.resource_evidence("read-1", "bank_account", "account-a") is None


@pytest.mark.parametrize("later_unknown_refresh", [False, True])
def test_child_execution_receipt_survives_mutation_and_controller_restore(later_unknown_refresh):
    controller, ledger, _ = setup(obligations=[obligation(status="ready")])
    card_read(ledger)
    call = card_call()
    assert controller.review([call])[0].allowed
    controller.before_condition_execution(call)
    execute(ledger, call.tool_call_id, call.tool_name, call.arguments)
    controller.after_condition_execution(call.tool_call_id)
    if later_unknown_refresh:
        card_read(ledger, call_id="later-read", records=[{"card_id": "debit-a"}])
    saved = controller.export_state()
    assert saved["items"][0]["obligations"][0]["applied_to"] == "effect"
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Continuing", raw_data={"cayu_tau3_workflow": saved})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == saved
    assert not restored.review([card_call()])[0].allowed


@pytest.mark.parametrize("change", ["removed", "malformed", "reparented", "only_later_read"])
def test_historical_join_cannot_use_missing_or_later_relationship(change):
    ledger = ConversationLedger()
    read(ledger)
    if change != "only_later_read":
        card_read(ledger)
        if change == "removed":
            card_read(ledger, call_id="new-read", records=[])
        elif change == "malformed":
            card_read(ledger, call_id="new-read", records=[{"card_id": "debit-a"}])
        else:
            card_read(ledger, call_id="new-read", account_id="other-account")
    execute(
        ledger,
        "effect",
        "call_discoverable_agent_tool",
        {"agent_tool_name": "change_card", "arguments": json.dumps({"card_id": "debit-a"})},
    )
    if change == "only_later_read":
        card_read(ledger)
    assert not ledger.debit_card_belongs_to(
        "debit-a", "account-a", "person-a", before_call="effect"
    )


def test_identity_requires_same_owner_and_rejects_conflicting_or_duplicate_records():
    ledger = ConversationLedger()
    read(ledger)
    record = {
        "card_id": "debit-a",
        "account_id": "account-a",
        "user_id": "person-a",
        "status": "ACTIVE",
    }
    card_read(ledger, records=[record, record])
    assert ledger.resource_evidence("card-read", "debit_card", "debit-a") is None
    execute(
        ledger,
        "ambiguous-json",
        "call_discoverable_agent_tool",
        {
            "agent_tool_name": "get_debit_cards_by_account_id_7823",
            "arguments": json.dumps({"account_id": "account-a"}),
        },
        '[{"card_id":"debit-a","account_id":"other","account_id":"account-a","user_id":"person-a","status":"ACTIVE"}]',
    )
    assert ledger.resource_evidence("ambiguous-json", "debit_card", "debit-a") is None


def test_debit_resource_refs_are_copy_ready_and_cleared_on_mutation():
    ledger = ConversationLedger()
    read(ledger)
    card_read(ledger)
    assert {
        "resource_kind": "debit_card",
        "resource_id": "debit-a",
        "observed_in": "card-read",
    } in ledger.resource_evidence_refs()
    execute(
        ledger,
        "change",
        "call_discoverable_agent_tool",
        {"agent_tool_name": "change_card", "arguments": json.dumps({"card_id": "debit-a"})},
    )
    assert not any(r["resource_kind"] == "debit_card" for r in ledger.resource_evidence_refs())
