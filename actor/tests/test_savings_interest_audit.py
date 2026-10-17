from __future__ import annotations

import json

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage

from cayu_tau3_banking.ledger import ConversationLedger
from cayu_tau3_banking.savings_interest_audit import (
    invoke_savings_interest_audit,
    savings_interest_audit_tool_schema,
    savings_interest_policy_dossier_tool_schema,
)

_POLICY_DOSSIER_ID = "a" * 64
_COMPONENT_SOURCE = {"entry_id": "fictional-independent-policy", "revision": 1,
                     "quote": "Independent relationship component applies."}


def component(name, rate):
    return {"name": name, "apy_percent": rate, "component_kind": "independent_relationship",
            "benefit_id": "independent-relationship", "source": dict(_COMPONENT_SOURCE)}



def _wrapper_call(call_id: str, tool_name: str, arguments: dict[str, str]) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="call_discoverable_agent_tool",
                arguments={
                    "agent_tool_name": tool_name,
                    "arguments": json.dumps(arguments),
                },
                requestor="assistant",
            )
        ],
    )


def _gold_interest_ledger(
    *,
    holdings: str = "96000.00",
    interest: str = "450.0",
    multiple_savings: bool = False,
    savings_level: str = "Gold Account",
) -> ConversationLedger:
    extra_savings = (
        "4. Record ID: savings-2\n"
        "   account_id: savings-2\n"
        "   user_id: user-7\n"
        "   class: saving\n"
        "   level: Silver Plus Account\n"
        "   status: OPEN\n"
        "   current_holdings: 18000.00\n\n"
        if multiple_savings
        else ""
    )
    ledger = ConversationLedger()
    ledger.observe_assistant(
        _wrapper_call(
            "accounts-1",
            "get_all_user_accounts_by_user_id_3847",
            {"user_id": "user-7"},
        )
    )
    ledger.observe_input(
        ToolMessage(
            id="accounts-1",
            role="tool",
            requestor="assistant",
            content=(
                "Bank Accounts:\n"
                f"Found {4 if multiple_savings else 3} record(s) in 'accounts':\n\n"
                "1. Record ID: savings-1\n"
                "   account_id: savings-1\n"
                "   user_id: user-7\n"
                "   class: saving\n"
                f"   level: {savings_level}\n"
                "   status: OPEN\n"
                f"   current_holdings: {holdings}\n\n"
                "2. Record ID: checking-green\n"
                "   account_id: checking-green\n"
                "   user_id: user-7\n"
                "   class: checking\n"
                "   level: Green Account\n"
                "   status: OPEN\n"
                "   current_holdings: 8500.00\n\n"
                "3. Record ID: checking-purple\n"
                "   account_id: checking-purple\n"
                "   user_id: user-7\n"
                "   class: checking\n"
                "   level: Purple Account\n"
                "   status: OPEN\n"
                "   current_holdings: 2100.00\n\n"
                f"{extra_savings}"
                "Credit Card Accounts:\n"
                "Found 3 record(s) in 'credit_card_accounts':\n\n"
                "1. Record ID: card-platinum\n"
                "   account_id: card-platinum\n"
                "   user_id: user-7\n"
                "   card_type: Platinum Rewards Card\n"
                "   account_status: ACTIVE\n\n"
                "2. Record ID: card-gold\n"
                "   account_id: card-gold\n"
                "   user_id: user-7\n"
                "   card_type: Gold Rewards Card\n"
                "   account_status: ACTIVE\n\n"
                "3. Record ID: card-eco\n"
                "   account_id: card-eco\n"
                "   user_id: user-7\n"
                "   card_type: EcoCard\n"
                "   account_status: ACTIVE"
            ),
        )
    )
    ledger.observe_assistant(
        _wrapper_call(
            "transactions-1",
            "get_bank_account_transactions_9173",
            {"account_id": "savings-1"},
        )
    )
    ledger.observe_input(
        ToolMessage(
            id="transactions-1",
            role="tool",
            requestor="assistant",
            content=(
                "Transactions for account savings-1:\n"
                "Found 2 record(s) in 'bank_account_transaction_history':\n\n"
                "1. Record ID: interest-1\n"
                "   transaction_id: interest-1\n"
                "   account_id: savings-1\n"
                "   date: 10/31/2025\n"
                "   description: MONTHLY INTEREST CREDIT\n"
                f"   amount: {interest}\n"
                "   type: interest_credit\n"
                "   status: posted\n\n"
                "2. Record ID: deposit-1\n"
                "   transaction_id: deposit-1\n"
                "   account_id: savings-1\n"
                "   date: 09/15/2025\n"
                "   description: ACH TRANSFER IN\n"
                "   amount: 15000.0\n"
                "   type: ach_transfer_in\n"
                "   status: posted"
            ),
        )
    )
    return ledger


def _task_095_arguments() -> dict[str, object]:
    return {
        "user_id": "user-7",
        "account_id": "savings-1",
        "policy_dossier_id": _POLICY_DOSSIER_ID,
        "base_or_tier_apy_percent": 5.5,
        "checking_boosts": [
            {"account_id": "checking-green", "apy_percent": 0.75},
            {"account_id": "checking-purple", "apy_percent": 0.1},
        ],
        "credit_card_bonuses": [
            {"account_id": "card-platinum", "apy_percent": 0.15},
            {"account_id": "card-gold", "apy_percent": 0.025},
            {"account_id": "card-eco", "apy_percent": 0.6},
        ],
        "other_stackable_components": [],
    }


def _payload(content: str) -> dict[str, object]:
    return json.loads(content.splitlines()[-1])


def test_audit_selects_highest_nonstacking_boosts_and_returns_exact_correction() -> None:
    result = invoke_savings_interest_audit(
        _gold_interest_ledger(),
        _task_095_arguments(),
        required_policy_dossier_id=_POLICY_DOSSIER_ID,
        component_source_validator=lambda source: source == _COMPONENT_SOURCE,
    )

    assert result.is_error is False
    payload = _payload(result.content)
    assert payload["expected_apy_percent"] == "6.85"
    assert payload["actual_apy_percent"] == "5.625"
    assert payload["expected_interest"] == "548.00"
    assert payload["actual_interest"] == "450.00"
    assert payload["difference"] == "98.00"
    assert payload["selected_checking_boost"] == {
        "account_id": "checking-green",
        "apy_percent": "0.75",
        "product": "Green Account",
    }
    assert payload["selected_credit_card_bonus"] == {
        "account_id": "card-eco",
        "apy_percent": "0.6",
        "product": "EcoCard",
    }
    assert payload["credit_arguments"] == {
        "account_id": "savings-1",
        "amount": "98.00",
        "credit_type": "interest_correction",
    }
    assert payload["report_arguments"] == {
        "account_id": "savings-1",
        "actual_apy": "5.625",
        "amount_difference": "98.00",
        "expected_apy": "6.85",
        "user_id": "user-7",
    }


def test_audit_adds_independently_stackable_components() -> None:
    arguments = _task_095_arguments()
    arguments["credit_card_bonuses"] = [
        {"account_id": "card-platinum", "apy_percent": 0},
        {"account_id": "card-gold", "apy_percent": 0},
        {"account_id": "card-eco", "apy_percent": 0},
    ]
    arguments["base_or_tier_apy_percent"] = 4
    arguments["checking_boosts"] = [
        {"account_id": "checking-green", "apy_percent": 0.25},
        {"account_id": "checking-purple", "apy_percent": 0},
    ]
    arguments["other_stackable_components"] = [
        component("Multi-product relationship", 0.025)
    ]

    result = invoke_savings_interest_audit(
        _gold_interest_ledger(),
        arguments,
        required_policy_dossier_id=_POLICY_DOSSIER_ID,
        component_source_validator=lambda source: source == _COMPONENT_SOURCE,
    )

    assert result.is_error is False
    payload = _payload(result.content)
    assert payload["expected_apy_percent"] == "4.275"
    assert payload["other_stackable_components"] == {"multi-product relationship": "0.025"}


def test_audit_supplies_documented_single_silver_relationship_component() -> None:
    arguments = _task_095_arguments()
    arguments["base_or_tier_apy_percent"] = 4
    arguments["checking_boosts"] = [
        {"account_id": "checking-green", "apy_percent": 0.25},
        {"account_id": "checking-purple", "apy_percent": 0},
    ]
    arguments["credit_card_bonuses"] = [
        {"account_id": "card-platinum", "apy_percent": 0},
        {"account_id": "card-gold", "apy_percent": 0},
        {"account_id": "card-eco", "apy_percent": 0},
    ]
    arguments["other_stackable_components"] = []

    result = invoke_savings_interest_audit(
        _gold_interest_ledger(savings_level="Silver Account"),
        arguments,
        required_policy_dossier_id=_POLICY_DOSSIER_ID,
        component_source_validator=lambda source: source == _COMPONENT_SOURCE,
    )

    assert result.is_error is False
    payload = _payload(result.content)
    assert payload["expected_apy_percent"] == "4.275"
    assert payload["other_stackable_components"] == {"multi-product relationship": "0.025"}


def test_audit_rejects_conflicting_single_silver_relationship_component() -> None:
    arguments = _task_095_arguments()
    arguments["other_stackable_components"] = [
        component("multi-product relationship", 0.25)
    ]

    result = invoke_savings_interest_audit(
        _gold_interest_ledger(savings_level="Silver Account"),
        arguments,
        required_policy_dossier_id=_POLICY_DOSSIER_ID,
        component_source_validator=lambda source: source == _COMPONENT_SOURCE,
    )

    assert result.is_error is True
    assert "exactly 0.025%" in result.content


def test_audit_does_not_supply_silver_relationship_to_another_savings_product() -> None:
    arguments = _task_095_arguments()
    arguments["other_stackable_components"] = []

    result = invoke_savings_interest_audit(
        _gold_interest_ledger(savings_level="Silver Plus Account"),
        arguments,
        required_policy_dossier_id=_POLICY_DOSSIER_ID,
        component_source_validator=lambda source: source == _COMPONENT_SOURCE,
    )

    assert result.is_error is False
    assert _payload(result.content)["other_stackable_components"] == {}


def test_audit_does_not_supply_relationship_when_silver_assignment_is_ambiguous() -> None:
    arguments = _task_095_arguments()
    arguments["other_stackable_components"] = []

    result = invoke_savings_interest_audit(
        _gold_interest_ledger(multiple_savings=True, savings_level="Silver Account"),
        arguments,
        required_policy_dossier_id=_POLICY_DOSSIER_ID,
        component_source_validator=lambda source: source == _COMPONENT_SOURCE,
    )

    assert result.is_error is False
    assert _payload(result.content)["other_stackable_components"] == {}


def test_audit_rejects_unassigned_relationship_bonus_across_multiple_savings() -> None:
    arguments = _task_095_arguments()
    arguments["other_stackable_components"] = [component("Relationship bonus", 0.025)]

    result = invoke_savings_interest_audit(
        _gold_interest_ledger(multiple_savings=True),
        arguments,
        required_policy_dossier_id=_POLICY_DOSSIER_ID,
        component_source_validator=lambda source: source == _COMPONENT_SOURCE,
    )

    assert result.is_error is True
    assert "multiple OPEN savings accounts" in result.content
    assert "account-specific relationship-eligibility" in result.content


def test_audit_rejects_incomplete_account_coverage_and_invalid_rates() -> None:
    incomplete = _task_095_arguments()
    incomplete["checking_boosts"] = [{"account_id": "checking-green", "apy_percent": 0.75}]
    invalid = _task_095_arguments()
    invalid["base_or_tier_apy_percent"] = float("nan")

    incomplete_result = invoke_savings_interest_audit(
        _gold_interest_ledger(),
        incomplete,
        required_policy_dossier_id=_POLICY_DOSSIER_ID,
        component_source_validator=lambda source: source == _COMPONENT_SOURCE,
    )
    invalid_result = invoke_savings_interest_audit(
        _gold_interest_ledger(),
        invalid,
        required_policy_dossier_id=_POLICY_DOSSIER_ID,
        component_source_validator=lambda source: source == _COMPONENT_SOURCE,
    )

    assert incomplete_result.is_error is True
    assert "missing account ids: checking-purple" in incomplete_result.content
    assert invalid_result.is_error is True
    assert "finite number" in invalid_result.content


def test_audit_rejects_missing_or_stale_policy_dossier() -> None:
    missing = invoke_savings_interest_audit(
        _gold_interest_ledger(),
        _task_095_arguments(),
        required_policy_dossier_id=None,
    )
    stale = invoke_savings_interest_audit(
        _gold_interest_ledger(),
        _task_095_arguments(),
        required_policy_dossier_id="b" * 64,
    )

    assert missing.is_error is True
    assert "Prepare this account's Cayu" in missing.content
    assert stale.is_error is True
    assert "does not match" in stale.content


def test_audit_emits_no_credit_for_a_non_undercredit() -> None:
    arguments = _task_095_arguments()
    arguments["base_or_tier_apy_percent"] = 4.5
    arguments["checking_boosts"] = [
        {"account_id": "checking-green", "apy_percent": 0},
        {"account_id": "checking-purple", "apy_percent": 0},
    ]
    arguments["credit_card_bonuses"] = [
        {"account_id": "card-platinum", "apy_percent": 0},
        {"account_id": "card-gold", "apy_percent": 0},
        {"account_id": "card-eco", "apy_percent": 0},
    ]

    result = invoke_savings_interest_audit(
        _gold_interest_ledger(),
        arguments,
        required_policy_dossier_id=_POLICY_DOSSIER_ID,
        component_source_validator=lambda source: source == _COMPONENT_SOURCE,
    )

    assert result.is_error is False
    payload = _payload(result.content)
    assert payload["direction"] == "overcredited"
    assert payload["credit_arguments"] is None
    assert "Do not apply a credit" in result.content


def test_audit_rejects_zero_holdings_instead_of_dividing_by_zero() -> None:
    result = invoke_savings_interest_audit(
        _gold_interest_ledger(holdings="0.00"),
        _task_095_arguments(),
        required_policy_dossier_id=_POLICY_DOSSIER_ID,
        component_source_validator=lambda source: source == _COMPONENT_SOURCE,
    )

    assert result.is_error is True
    assert "positive current holdings" in result.content


def test_savings_interest_schema_requires_complete_component_groups() -> None:
    schema = savings_interest_audit_tool_schema()
    prepare_schema = savings_interest_policy_dossier_tool_schema()

    assert schema["name"] == "controller_audit_savings_interest"
    assert prepare_schema["name"] == "controller_prepare_savings_interest_policy"
    assert set(prepare_schema["input_schema"]["required"]) == {"user_id", "account_id"}
    assert set(schema["input_schema"]["required"]) == {
        "user_id",
        "account_id",
        "policy_dossier_id",
        "base_or_tier_apy_percent",
        "checking_boosts",
        "credit_card_bonuses",
        "other_stackable_components",
    }
