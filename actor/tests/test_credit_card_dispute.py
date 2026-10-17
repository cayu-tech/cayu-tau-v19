from __future__ import annotations

import json
from collections.abc import Mapping

import pytest
from cayu import ToolCallPart
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage

from cayu_tau3_banking.credit_card_dispute import (
    FILE_CREDIT_CARD_DISPUTE_TOOL,
    CreditCardDisputeActionGate,
    compile_dispute_policy_dossier,
    dispute_audit_tool_schema,
    invoke_dispute_audit,
)
from cayu_tau3_banking.ledger import ConversationLedger

_DOSSIER_ID = "a" * 64
_ELIGIBILITY_POLICY = """## Eligibility Criteria — Customer is ELIGIBLE for provisional credit if ALL of the following are true:
1. Account Standing: The customer's credit card account has been open for at least 60 days
2. Dispute Reason Category: The dispute reason is one of the following:
   - 'unauthorized_fraudulent_charge'
   - 'duplicate_charge'
   - 'goods_services_not_received' (only if purchase was made more than 30 days ago)
3. Dispute Amount: The transaction amount must be at least $25.00 and must not exceed the card's maximum provisional credit limit based on card tier (see below)
4. Previous Disputes: The customer has not filed more than 2 disputes in the past 12 months
5. Contacted Merchant: For non-fraud disputes, the customer must contact the merchant

## Maximum Provisional Credit Limits by Card Tier
- Entry Tier (Bronze Rewards Card, EcoCard): Maximum $2500.00
- Premium Tier (Gold Rewards Card, Business Gold Rewards Card): Maximum $10000.00

## NOT Eligible Scenarios
"""


def _policy_content(*, documents: list[dict[str, str]] | None = None) -> str:
    return json.dumps(
        {
            "policy_dossier_id": _DOSSIER_ID,
            "documents": documents
            or [
                {
                    "title": "Checking User Dispute History (Internal)",
                    "text": "Retrieve the complete dispute history for the user.",
                },
                {
                    "title": "Filing a Credit Card Transaction Dispute (Internal)",
                    "text": "Gather every required filing field.",
                },
                {
                    "title": "How to Find Your Rho-Bank Credit Card Numbers Online",
                    "text": (
                        "Give the customer get_card_last_4_digits with their "
                        "credit_card_account_id."
                    ),
                },
                {
                    "title": "Provisional Credit Eligibility Guidelines (Internal)",
                    "text": _ELIGIBILITY_POLICY,
                },
            ],
        }
    )


def _assistant_call(call_id: str, name: str, **arguments: object) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
                requestor="assistant",
            )
        ],
    )


def _wrapper_call(call_id: str, name: str, arguments: Mapping[str, object]) -> AssistantMessage:
    return _assistant_call(
        call_id,
        "call_discoverable_agent_tool",
        agent_tool_name=name,
        arguments=json.dumps(arguments),
    )


def _observe_result(ledger: ConversationLedger, call: AssistantMessage, content: str) -> None:
    ledger.observe_assistant(call)
    assert call.tool_calls
    ledger.observe_input(
        ToolMessage(
            id=call.tool_calls[0].id,
            role="tool",
            requestor="assistant",
            content=content,
        )
    )


def _ledger(*, include_history: bool = True) -> ConversationLedger:
    ledger = ConversationLedger()
    _observe_result(
        ledger,
        _assistant_call("time-1", "get_current_time"),
        "The current time is 2025-11-14 03:40:00 EST.",
    )
    _observe_result(
        ledger,
        _assistant_call("transactions-1", "get_credit_card_transactions_by_user", user_id="user-7"),
        (
            "Found 6 record(s) in 'credit_card_transaction_history':\n\n"
            "1. Record ID: txn-fraud\n"
            "   transaction_id: txn-fraud\n"
            "   user_id: user-7\n"
            "   credit_card_type: Gold Rewards Card\n"
            "   merchant_name: Airline\n"
            "   transaction_amount: $600.00\n"
            "   transaction_date: 10/01/2025\n"
            "   category: Travel\n"
            "   status: COMPLETED\n"
            "   rewards_earned: 0 points\n\n"
            "2. Record ID: txn-duplicate\n"
            "   transaction_id: txn-duplicate\n"
            "   user_id: user-7\n"
            "   credit_card_type: Gold Rewards Card\n"
            "   merchant_name: Store\n"
            "   transaction_amount: $500.00\n"
            "   transaction_date: 10/02/2025\n"
            "   category: Shopping\n"
            "   status: COMPLETED\n"
            "   rewards_earned: 0 points\n\n"
            "3. Record ID: txn-goods\n"
            "   transaction_id: txn-goods\n"
            "   user_id: user-7\n"
            "   credit_card_type: Gold Rewards Card\n"
            "   merchant_name: Delivery\n"
            "   transaction_amount: $400.00\n"
            "   transaction_date: 10/03/2025\n"
            "   category: Shopping\n"
            "   status: COMPLETED\n"
            "   rewards_earned: 0 points\n\n"
            "4. Record ID: txn-no-contact\n"
            "   transaction_id: txn-no-contact\n"
            "   user_id: user-7\n"
            "   credit_card_type: Gold Rewards Card\n"
            "   merchant_name: Duplicate Shop\n"
            "   transaction_amount: $700.00\n"
            "   transaction_date: 10/04/2025\n"
            "   category: Shopping\n"
            "   status: COMPLETED\n"
            "   rewards_earned: 0 points\n\n"
            "5. Record ID: txn-ineligible-reason\n"
            "   transaction_id: txn-ineligible-reason\n"
            "   user_id: user-7\n"
            "   credit_card_type: Gold Rewards Card\n"
            "   merchant_name: Utility\n"
            "   transaction_amount: $900.00\n"
            "   transaction_date: 10/05/2025\n"
            "   category: Utilities\n"
            "   status: COMPLETED\n"
            "   rewards_earned: 0 points\n\n"
            "6. Record ID: txn-small\n"
            "   transaction_id: txn-small\n"
            "   user_id: user-7\n"
            "   credit_card_type: Gold Rewards Card\n"
            "   merchant_name: Cafe\n"
            "   transaction_amount: $20.00\n"
            "   transaction_date: 10/06/2025\n"
            "   category: Dining\n"
            "   status: COMPLETED\n"
            "   rewards_earned: 0 points"
        ),
    )
    _observe_result(
        ledger,
        _assistant_call("accounts-1", "get_credit_card_accounts_by_user", user_id="user-7"),
        (
            "Found 1 record(s) in 'credit_card_accounts':\n\n"
            "1. Record ID: card-1\n"
            "   account_id: card-1\n"
            "   user_id: user-7\n"
            "   card_type: Gold Rewards Card\n"
            "   date_of_account_open: 03/15/2024\n"
            "   current_balance: $100.00"
        ),
    )
    if include_history:
        _observe_result(
            ledger,
            _wrapper_call("history-1", "get_user_dispute_history_7291", {"user_id": "user-7"}),
            (
                "User transaction dispute history retrieved successfully.\n\n"
                "Executed: get_user_dispute_history_7291\n"
                "Transaction dispute history for user user-7:\n"
                "Found 1 record(s) in 'transaction_disputes':\n\n"
                "1. Record ID: dispute-old\n"
                "   dispute_id: dispute-old\n"
                "   transaction_id: txn-old\n"
                "   user_id: user-7\n"
                "   provisional_credit_given: True\n"
                "   submitted_at: 09/20/2025"
            ),
        )
    return ledger


def _disputes() -> list[dict[str, object]]:
    return [
        {
            "transaction_id": "txn-fraud",
            "dispute_reason": "unauthorized_fraudulent_charge",
            "contacted_merchant": False,
            "issue_noticed_date": "11/14/2025",
        },
        {
            "transaction_id": "txn-duplicate",
            "dispute_reason": "duplicate_charge",
            "contacted_merchant": True,
            "issue_noticed_date": "11/14/2025",
        },
        {
            "transaction_id": "txn-goods",
            "dispute_reason": "goods_services_not_received",
            "contacted_merchant": True,
            "issue_noticed_date": "11/14/2025",
        },
        {
            "transaction_id": "txn-no-contact",
            "dispute_reason": "duplicate_charge",
            "contacted_merchant": False,
            "issue_noticed_date": "11/14/2025",
        },
        {
            "transaction_id": "txn-ineligible-reason",
            "dispute_reason": "incorrect_amount",
            "contacted_merchant": True,
            "issue_noticed_date": "11/14/2025",
        },
        {
            "transaction_id": "txn-small",
            "dispute_reason": "duplicate_charge",
            "contacted_merchant": True,
            "issue_noticed_date": "11/14/2025",
        },
    ]


def _audit(*, ledger: ConversationLedger | None = None):
    policy = compile_dispute_policy_dossier(_policy_content())
    return invoke_dispute_audit(
        ledger or _ledger(),
        {
            "user_id": "user-7",
            "policy_dossier_id": _DOSSIER_ID,
            "priority_transaction_ids": ["txn-duplicate"],
            "disputes": _disputes(),
        },
        policy=policy,
    )


def _payload(result) -> dict[str, object]:
    return json.loads(result.content.rsplit("\n", 1)[-1])


def _filing_call(call_id: str, arguments: Mapping[str, object]) -> ToolCallPart:
    return ToolCallPart(
        tool_call_id=call_id,
        tool_name="call_discoverable_agent_tool",
        arguments={
            "agent_tool_name": FILE_CREDIT_CARD_DISPUTE_TOOL,
            "arguments": json.dumps(arguments),
        },
    )


def test_policy_compiler_requires_complete_family_and_compiles_exact_limits() -> None:
    policy = compile_dispute_policy_dossier(_policy_content())

    assert policy.minimum_account_age_days == 60
    assert policy.goods_not_received_minimum_age_days == 30
    assert str(policy.minimum_amount) == "25.00"
    assert policy.maximum_provisional_credits == 3
    assert policy.eligible_reasons == {
        "unauthorized_fraudulent_charge",
        "duplicate_charge",
        "goods_services_not_received",
    }
    assert policy.card_limits["gold rewards"] == 10_000

    incomplete = json.loads(_policy_content())
    incomplete["documents"][0]["title"] = "Unrelated policy"
    with pytest.raises(ValueError, match="family is incomplete"):
        compile_dispute_policy_dossier(json.dumps(incomplete))


def test_complete_batch_audit_allocates_remaining_slots_by_priority_then_amount() -> None:
    result = _audit()

    assert result.is_error is False
    payload = _payload(result)
    audited = {item["transaction_id"]: item for item in payload["audited_disputes"]}
    assert payload["prior_provisional_credits_in_period"] == 1
    assert payload["provisional_credit_slots_before_batch"] == 2
    assert {
        transaction_id
        for transaction_id, item in audited.items()
        if item["eligible_for_provisional_credit"]
    } == {"txn-duplicate", "txn-fraud"}
    assert audited["txn-no-contact"]["eligible_for_provisional_credit"] is False
    assert audited["txn-ineligible-reason"]["eligible_for_provisional_credit"] is False
    assert audited["txn-small"]["eligible_for_provisional_credit"] is False


def test_refreshed_history_does_not_double_count_an_in_session_filing() -> None:
    ledger = _ledger()
    first = _audit(ledger=ledger)
    authorized = {item["transaction_id"]: item for item in _payload(first)["audited_disputes"]}
    _observe_result(
        ledger,
        _wrapper_call(
            "filing-1",
            FILE_CREDIT_CARD_DISPUTE_TOOL,
            {
                **authorized["txn-fraud"],
                "card_action": "keep_active",
                "card_last_4_digits": "1234",
                "full_name": "Example User",
                "phone": "555-0100",
                "email": "user@example.com",
                "address": "1 Main St",
                "resolution_requested": "full_refund",
            },
        ),
        "Credit card transaction dispute filed successfully.",
    )
    _observe_result(
        ledger,
        _wrapper_call("history-2", "get_user_dispute_history_7291", {"user_id": "user-7"}),
        (
            "User transaction dispute history retrieved successfully.\n\n"
            "Found 2 record(s) in 'transaction_disputes':\n\n"
            "1. Record ID: dispute-old\n"
            "   dispute_id: dispute-old\n"
            "   transaction_id: txn-old\n"
            "   user_id: user-7\n"
            "   provisional_credit_given: True\n"
            "   submitted_at: 09/20/2025\n\n"
            "2. Record ID: dispute-new\n"
            "   dispute_id: dispute-new\n"
            "   transaction_id: txn-fraud\n"
            "   user_id: user-7\n"
            "   provisional_credit_given: True\n"
            "   submitted_at: 11/14/2025"
        ),
    )
    remaining = [
        item for item in _disputes() if item["transaction_id"] in {"txn-duplicate", "txn-goods"}
    ]

    result = invoke_dispute_audit(
        ledger,
        {
            "user_id": "user-7",
            "policy_dossier_id": _DOSSIER_ID,
            "priority_transaction_ids": [],
            "disputes": remaining,
        },
        policy=compile_dispute_policy_dossier(_policy_content()),
    )

    assert result.is_error is False
    payload = _payload(result)
    assert payload["prior_provisional_credits_in_period"] == 2
    assert payload["provisional_credit_slots_before_batch"] == 1
    assert sum(item["eligible_for_provisional_credit"] for item in payload["audited_disputes"]) == 1


def test_audit_rejects_impossible_issue_dates() -> None:
    disputes = _disputes()
    disputes[0] = {**disputes[0], "issue_noticed_date": "09/01/2025"}

    result = invoke_dispute_audit(
        _ledger(),
        {
            "user_id": "user-7",
            "policy_dossier_id": _DOSSIER_ID,
            "priority_transaction_ids": [],
            "disputes": disputes,
        },
        policy=compile_dispute_policy_dossier(_policy_content()),
    )

    assert result.is_error is True
    assert "on or after its observed purchase date" in result.content


def test_audit_rejects_missing_observations_duplicate_candidates_and_stale_policy() -> None:
    policy = compile_dispute_policy_dossier(_policy_content())
    arguments = {
        "user_id": "user-7",
        "policy_dossier_id": _DOSSIER_ID,
        "priority_transaction_ids": [],
        "disputes": _disputes(),
    }

    missing_history = invoke_dispute_audit(_ledger(include_history=False), arguments, policy=policy)
    duplicate_arguments = {**arguments, "disputes": [*_disputes(), _disputes()[0]]}
    duplicate = invoke_dispute_audit(_ledger(), duplicate_arguments, policy=policy)
    stale = invoke_dispute_audit(
        _ledger(),
        {**arguments, "policy_dossier_id": "b" * 64},
        policy=policy,
    )

    assert missing_history.is_error is True
    assert "dispute-history lookup" in missing_history.content
    assert duplicate.is_error is True
    assert "duplicates a transaction" in duplicate.content
    assert stale.is_error is True
    assert "Prepare the current" in stale.content


def test_action_gate_requires_exact_audit_fields_and_rejects_batch_duplicates() -> None:
    ledger = _ledger()
    result = _audit(ledger=ledger)
    gate = CreditCardDisputeActionGate(ledger)
    gate.observe_audit(result, expected_user_id="user-7")
    audited = {item["transaction_id"]: item for item in _payload(result)["audited_disputes"]}
    exact_arguments = {
        **audited["txn-fraud"],
        "card_action": "keep_active",
        "card_last_4_digits": "1234",
        "full_name": "Example User",
        "phone": "555-0100",
        "email": "user@example.com",
        "address": "1 Main St",
        "resolution_requested": "full_refund",
    }
    exact = _filing_call("file-1", exact_arguments)
    altered = _filing_call(
        "file-2",
        {**exact_arguments, "eligible_for_provisional_credit": False},
    )
    unaudited = _filing_call(
        "file-3",
        {**exact_arguments, "transaction_id": "txn-unobserved"},
    )

    assert gate.review([exact])[0].allowed is True
    altered_decision = gate.review([altered])[0]
    assert altered_decision.allowed is False
    assert altered_decision.diagnostic_code == "credit_card_dispute_audit"
    assert gate.review([unaudited])[0].allowed is False
    duplicate = gate.review([exact, _filing_call("file-4", exact_arguments)])
    assert [decision.allowed for decision in duplicate] == [True, False]


def test_action_gate_rejects_a_filing_after_its_exact_transaction_succeeds() -> None:
    ledger = _ledger()
    result = _audit(ledger=ledger)
    gate = CreditCardDisputeActionGate(ledger)
    gate.observe_audit(result, expected_user_id="user-7")
    audited = {item["transaction_id"]: item for item in _payload(result)["audited_disputes"]}
    arguments = {
        **audited["txn-fraud"],
        "card_action": "keep_active",
        "card_last_4_digits": "1234",
        "full_name": "Example User",
        "phone": "555-0100",
        "email": "user@example.com",
        "address": "1 Main St",
        "resolution_requested": "full_refund",
    }
    _observe_result(
        ledger,
        _wrapper_call("file-success", FILE_CREDIT_CARD_DISPUTE_TOOL, arguments),
        "Credit card transaction dispute filed successfully.",
    )

    decision = gate.review([_filing_call("file-again", arguments)])[0]

    assert gate.pending_filing_count() == 5
    assert decision.allowed is False
    assert "already succeeded" in (decision.reason or "")


def test_audited_filings_cannot_be_rewritten_as_generic_customer_handoffs() -> None:
    ledger = _ledger()
    result = _audit(ledger=ledger)
    gate = CreditCardDisputeActionGate(ledger)
    gate.observe_audit(result, expected_user_id="user-7")
    self_service_state = {
        "items": [
            {
                "id": "dispute-charge",
                "goal": "File the requested credit-card transaction dispute",
                "status": "awaiting_customer",
                "next_operation": {
                    "owner": "customer",
                    "kind": "customer_response",
                    "tool_name": "",
                    "arguments": {},
                    "summary": "Submit the dispute through another support channel.",
                },
            }
        ]
    }
    missing_fact_state = {
        "items": [
            {
                **self_service_state["items"][0],
                "next_operation": {
                    "owner": "customer",
                    "kind": "customer_response",
                    "tool_name": "",
                    "arguments": {},
                    "summary": "Provide the resolution requested for the dispute.",
                },
            }
        ]
    }
    disguised_self_service_state = {
        "items": [
            {
                **self_service_state["items"][0],
                "next_operation": {
                    "owner": "customer",
                    "kind": "customer_response",
                    "tool_name": "",
                    "arguments": {},
                    "summary": ("Ask the customer to contact the merchant, then file by phone."),
                },
            }
        ]
    }
    merchant_fact_state = {
        "items": [
            {
                **self_service_state["items"][0],
                "next_operation": {
                    "owner": "customer",
                    "kind": "customer_response",
                    "tool_name": "",
                    "arguments": {},
                    "summary": "Ask whether the customer already contacted the merchant.",
                },
            }
        ]
    }

    denial = gate.terminal_handoff_denial(self_service_state)

    assert gate.pending_filing_count() == 6
    assert denial is not None
    assert "agent-executable" in denial
    assert gate.terminal_handoff_denial(missing_fact_state) is None
    assert gate.terminal_handoff_denial(disguised_self_service_state) is not None
    assert gate.terminal_handoff_denial(merchant_fact_state) is None
    assert gate.terminal_handoff_denial({"items": []}) is None


def test_audit_schema_requires_one_bounded_complete_candidate_batch() -> None:
    schema = dispute_audit_tool_schema()["input_schema"]

    assert schema["additionalProperties"] is False
    assert schema["properties"]["disputes"]["minItems"] == 1
    assert schema["properties"]["disputes"]["maxItems"] == 24
    assert schema["properties"]["priority_transaction_ids"]["uniqueItems"] is True
