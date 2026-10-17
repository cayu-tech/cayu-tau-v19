from __future__ import annotations

import json

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage

from cayu_tau3_banking.ledger import ConversationLedger
from cayu_tau3_banking.reward_audit import invoke_reward_audit, reward_audit_tool_schema

POLICY_DOSSIER_ID = "a" * 64


def ledger_with_transactions() -> ConversationLedger:
    ledger = ConversationLedger()
    ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="transactions-1",
                    name="get_credit_card_transactions_by_user",
                    arguments={"user_id": "user-7"},
                    requestor="assistant",
                )
            ],
        )
    )
    ledger.observe_input(
        ToolMessage(
            id="transactions-1",
            role="tool",
            requestor="assistant",
            content=(
                "Found 3 record(s) in 'credit_card_transaction_history':\n\n"
                "1. Record ID: txn-1\n"
                "   transaction_id: txn-1\n"
                "   user_id: user-7\n"
                "   credit_card_type: Silver Rewards Card\n"
                "   merchant_name: Example Air\n"
                "   transaction_amount: $126.77\n"
                "   transaction_date: 09/10/2025\n"
                "   category: Travel\n"
                "   status: COMPLETED\n"
                "   rewards_earned: 507 points\n\n"
                "2. Record ID: txn-2\n"
                "   transaction_id: txn-2\n"
                "   user_id: user-7\n"
                "   credit_card_type: Silver Rewards Card\n"
                "   merchant_name: Example Shop\n"
                "   transaction_amount: $99.99\n"
                "   transaction_date: 09/11/2025\n"
                "   category: Shopping\n"
                "   status: COMPLETED\n"
                "   rewards_earned: 50 points\n\n"
                "3. Record ID: txn-3\n"
                "   transaction_id: txn-3\n"
                "   user_id: user-7\n"
                "   credit_card_type: Silver Rewards Card\n"
                "   merchant_name: Pending Shop\n"
                "   transaction_amount: $20.00\n"
                "   transaction_date: 09/12/2025\n"
                "   category: Shopping\n"
                "   status: PENDING\n"
                "   rewards_earned: 0 points"
            ),
        )
    )
    return ledger


def test_reward_audit_covers_all_completed_records_and_floors_points() -> None:
    result = invoke_reward_audit(
        ledger_with_transactions(),
        {
            "user_id": "user-7",
            "policy_dossier_id": POLICY_DOSSIER_ID,
            "rates": [
                {"transaction_id": "txn-1", "cash_back_percent": 4},
                {"transaction_id": "txn-2", "cash_back_percent": 1},
            ],
        },
        required_policy_dossier_id=POLICY_DOSSIER_ID,
    )

    assert result.is_error is False
    payload = json.loads(result.content.splitlines()[-1])
    assert payload["audited_transaction_count"] == 2
    assert payload["matching_transaction_count"] == 1
    assert payload["overcredited_transaction_count"] == 0
    assert payload["multi_point_mismatch_count"] == 1
    assert payload["single_point_mismatch_count"] == 0
    assert payload["undercredited_transaction_count"] == 1
    assert payload["mismatches"] == [
        {
            "absolute_difference_points": 49,
            "card_type": "Silver Rewards Card",
            "cash_back_percent": "1",
            "category": "Shopping",
            "difference_points": 49,
            "direction": "undercredited",
            "expected_points": 99,
            "merchant_name": "Example Shop",
            "magnitude_class": "multi_point",
            "recorded_points": 50,
            "transaction_amount": "99.99",
            "transaction_date": "09/11/2025",
            "transaction_id": "txn-2",
        }
    ]


def test_reward_audit_labels_overcredits_explicitly() -> None:
    result = invoke_reward_audit(
        ledger_with_transactions(),
        {
            "user_id": "user-7",
            "policy_dossier_id": POLICY_DOSSIER_ID,
            "rates": [
                {"transaction_id": "txn-1", "cash_back_percent": 4},
                {"transaction_id": "txn-2", "cash_back_percent": 0.49},
            ],
        },
        required_policy_dossier_id=POLICY_DOSSIER_ID,
    )

    assert result.is_error is False
    payload = json.loads(result.content.splitlines()[-1])
    assert payload["undercredited_transaction_count"] == 0
    assert payload["overcredited_transaction_count"] == 1
    assert payload["mismatches"][0]["direction"] == "overcredited"
    assert payload["mismatches"][0]["magnitude_class"] == "multi_point"
    assert payload["mismatches"][0]["difference_points"] == -2
    assert payload["mismatches"][0]["absolute_difference_points"] == 2
    assert result.structured is not None
    assert result.structured["finding_kind"] == "rewards_discrepancy"
    assert result.structured["findings"] == payload["mismatches"]


def test_reward_audit_classifies_single_point_rounding_mismatch() -> None:
    result = invoke_reward_audit(
        ledger_with_transactions(),
        {
            "user_id": "user-7",
            "policy_dossier_id": POLICY_DOSSIER_ID,
            "rates": [
                {"transaction_id": "txn-1", "cash_back_percent": 4},
                {"transaction_id": "txn-2", "cash_back_percent": 0.5},
            ],
        },
        required_policy_dossier_id=POLICY_DOSSIER_ID,
    )

    payload = json.loads(result.content.splitlines()[-1])
    assert payload["multi_point_mismatch_count"] == 0
    assert payload["single_point_mismatch_count"] == 1
    assert payload["mismatches"][0]["magnitude_class"] == "single_point"
    assert payload["mismatches"][0]["difference_points"] == -1


def test_reward_audit_rejects_missing_or_stale_policy_dossier() -> None:
    arguments = {
        "user_id": "user-7",
        "policy_dossier_id": POLICY_DOSSIER_ID,
        "rates": [
            {"transaction_id": "txn-1", "cash_back_percent": 4},
            {"transaction_id": "txn-2", "cash_back_percent": 1},
        ],
    }

    missing = invoke_reward_audit(
        ledger_with_transactions(), arguments, required_policy_dossier_id=None
    )
    stale = invoke_reward_audit(
        ledger_with_transactions(),
        arguments,
        required_policy_dossier_id="b" * 64,
    )

    assert missing.is_error is True
    assert "Prepare this user's Cayu reward-policy dossier" in missing.content
    assert stale.is_error is True
    assert "does not match" in stale.content


def test_reward_audit_rejects_partial_duplicate_and_unobserved_coverage() -> None:
    ledger = ledger_with_transactions()

    partial = invoke_reward_audit(
        ledger,
        {
            "user_id": "user-7",
            "policy_dossier_id": POLICY_DOSSIER_ID,
            "rates": [{"transaction_id": "txn-1", "cash_back_percent": 4}],
        },
        required_policy_dossier_id=POLICY_DOSSIER_ID,
    )
    duplicate = invoke_reward_audit(
        ledger,
        {
            "user_id": "user-7",
            "policy_dossier_id": POLICY_DOSSIER_ID,
            "rates": [
                {"transaction_id": "txn-1", "cash_back_percent": 4},
                {"transaction_id": "txn-1", "cash_back_percent": 4},
            ],
        },
        required_policy_dossier_id=POLICY_DOSSIER_ID,
    )
    unobserved = invoke_reward_audit(
        ConversationLedger(),
        {"user_id": "user-7", "policy_dossier_id": POLICY_DOSSIER_ID, "rates": []},
        required_policy_dossier_id=POLICY_DOSSIER_ID,
    )

    assert partial.is_error is True
    assert "missing transaction ids: txn-2" in partial.content
    assert duplicate.is_error is True
    assert "duplicates transaction_id" in duplicate.content
    assert unobserved.is_error is True
    assert "No parsed successful" in unobserved.content


def test_reward_audit_rejects_nonfinite_boolean_and_out_of_range_rates() -> None:
    ledger = ledger_with_transactions()

    for invalid_rate in (True, float("nan"), -1, 101):
        result = invoke_reward_audit(
            ledger,
            {
                "user_id": "user-7",
                "policy_dossier_id": POLICY_DOSSIER_ID,
                "rates": [
                    {
                        "transaction_id": "txn-1",
                        "cash_back_percent": invalid_rate,
                    },
                    {"transaction_id": "txn-2", "cash_back_percent": 1},
                ],
            },
            required_policy_dossier_id=POLICY_DOSSIER_ID,
        )

        assert result.is_error is True


def test_reward_audit_schema_requires_bounded_numeric_rates() -> None:
    schema = reward_audit_tool_schema()

    rate_schema = schema["input_schema"]["properties"]["rates"]
    percent_schema = rate_schema["items"]["properties"]["cash_back_percent"]

    assert "policy_dossier_id" in schema["input_schema"]["required"]
    assert rate_schema["maxItems"] == 256
    assert percent_schema == {"type": "number", "minimum": 0, "maximum": 100}
