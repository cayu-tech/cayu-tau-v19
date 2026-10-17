"""Deterministic arithmetic for complete public credit-card reward audits."""

from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from typing import Any

from cayu_tau3_banking.bridge import InternalToolResult
from cayu_tau3_banking.ledger import ConversationLedger, CreditCardTransactionObservation

AUDIT_CREDIT_CARD_REWARDS_TOOL = "controller_audit_credit_card_rewards"
PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL = "controller_prepare_credit_card_reward_policy"

_MAX_RATE_ITEMS = 256
_MAX_IDENTIFIER_CHARS = 128
_MAX_RATE_PERCENT = Decimal("100")


def reward_policy_dossier_tool_schema() -> Mapping[str, Any]:
    """Return the private Cayu policy-dossier tool schema."""

    return {
        "name": PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL,
        "description": (
            "Prepare a complete exact-revision Cayu policy dossier for every credit-card type "
            "in the latest successful transaction lookup. The dossier includes applicable "
            "earning, category, promotion, exclusion, points-representation, and rounding "
            "documents selected only from public card names and public knowledge titles. Call "
            "this before controller_audit_credit_card_rewards and use its policy_dossier_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _MAX_IDENTIFIER_CHARS,
                }
            },
            "required": ["user_id"],
            "additionalProperties": False,
        },
    }


def reward_audit_tool_schema() -> Mapping[str, Any]:
    """Return the private controller-tool schema exposed to the model."""

    return {
        "name": AUDIT_CREDIT_CARD_REWARDS_TOOL,
        "description": (
            "Deterministically audit every COMPLETED credit-card transaction from the latest "
            "successful get_credit_card_transactions_by_user result. First prepare and read the "
            "complete Cayu policy dossier, then provide its policy_dossier_id and exactly one "
            "expected cash-back percentage for every completed transaction id. The controller "
            "verifies complete coverage, floors point arithmetic, and reports every mismatch "
            "with direction and single-point versus multi-point magnitude. It does not choose "
            "rates, retrieve policy, submit disputes, or mutate bank state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _MAX_IDENTIFIER_CHARS,
                },
                "policy_dossier_id": {
                    "type": "string",
                    "minLength": 64,
                    "maxLength": 64,
                    "pattern": "^[0-9a-f]{64}$",
                },
                "rates": {
                    "type": "array",
                    "maxItems": _MAX_RATE_ITEMS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "transaction_id": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": _MAX_IDENTIFIER_CHARS,
                            },
                            "cash_back_percent": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 100,
                            },
                        },
                        "required": ["transaction_id", "cash_back_percent"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["user_id", "policy_dossier_id", "rates"],
            "additionalProperties": False,
        },
    }


def invoke_reward_audit(
    ledger: ConversationLedger,
    arguments: Mapping[str, Any],
    *,
    required_policy_dossier_id: str | None,
) -> InternalToolResult:
    """Audit cached public transactions with explicit model-supplied policy rates."""

    if not isinstance(ledger, ConversationLedger):
        raise TypeError("ledger must be a ConversationLedger.")
    if not isinstance(arguments, Mapping) or set(arguments) != {
        "user_id",
        "policy_dossier_id",
        "rates",
    }:
        return _error("Expected exactly 'user_id', 'policy_dossier_id', and 'rates'.")
    user_id = _bounded_nonblank(arguments["user_id"])
    if user_id is None:
        return _error("user_id must be nonblank text within its size bound.")
    raw_rates = arguments["rates"]
    if type(raw_rates) is not list or len(raw_rates) > _MAX_RATE_ITEMS:
        return _error(f"rates must be an array with at most {_MAX_RATE_ITEMS} items.")

    observations = ledger.credit_card_transactions(user_id)
    if observations is None:
        return _error("No parsed successful credit-card transaction lookup exists for this user.")
    policy_dossier_id = _bounded_nonblank(arguments["policy_dossier_id"])
    if required_policy_dossier_id is None:
        return _error(
            "Prepare this user's Cayu reward-policy dossier before submitting audit rates."
        )
    if policy_dossier_id != required_policy_dossier_id:
        return _error("policy_dossier_id does not match this user's prepared Cayu dossier.")
    completed = tuple(
        observation for observation in observations if observation.status.casefold() == "completed"
    )
    rates_or_error = _parse_rates(raw_rates)
    if isinstance(rates_or_error, str):
        return _error(rates_or_error)
    rates = rates_or_error

    expected_ids = {observation.transaction_id for observation in completed}
    provided_ids = set(rates)
    missing = sorted(expected_ids - provided_ids)
    unexpected = sorted(provided_ids - expected_ids)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing transaction ids: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected transaction ids: {', '.join(unexpected)}")
        return _error(
            "Reward audit requires exactly one rate for every completed transaction; "
            + "; ".join(details)
            + "."
        )

    mismatches = [
        mismatch
        for observation in completed
        if (mismatch := _audit_observation(observation, rates[observation.transaction_id]))
        is not None
    ]
    undercredited = sum(mismatch["direction"] == "undercredited" for mismatch in mismatches)
    overcredited = len(mismatches) - undercredited
    single_point = sum(mismatch["magnitude_class"] == "single_point" for mismatch in mismatches)
    payload = {
        "audited_transaction_count": len(completed),
        "matching_transaction_count": len(completed) - len(mismatches),
        "mismatches": mismatches,
        "mismatching_transaction_count": len(mismatches),
        "multi_point_mismatch_count": len(mismatches) - single_point,
        "overcredited_transaction_count": overcredited,
        "policy_dossier_id": policy_dossier_id,
        "single_point_mismatch_count": single_point,
        "undercredited_transaction_count": undercredited,
        "user_id": user_id,
    }
    return InternalToolResult(
        content=(
            "Arithmetic audit completed using model-supplied policy rates. Reconcile every "
            "reported mismatch with the customer's request and public policy before acting.\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        ),
        structured={
            "finding_kind": "rewards_discrepancy",
            "findings": mismatches,
            "source": AUDIT_CREDIT_CARD_REWARDS_TOOL,
        },
    )


def _parse_rates(raw_rates: list[object]) -> dict[str, Decimal] | str:
    rates: dict[str, Decimal] = {}
    for index, raw in enumerate(raw_rates):
        if type(raw) is not dict or set(raw) != {"transaction_id", "cash_back_percent"}:
            return f"rate item {index} requires exactly transaction_id and cash_back_percent"
        transaction_id = _bounded_nonblank(raw["transaction_id"])
        if transaction_id is None:
            return f"rate item {index} has an invalid transaction_id"
        if transaction_id in rates:
            return f"rate item {index} duplicates transaction_id {transaction_id!r}"
        raw_percent = raw["cash_back_percent"]
        if isinstance(raw_percent, bool) or not isinstance(raw_percent, int | float):
            return f"rate item {index} cash_back_percent must be a number"
        try:
            percent = Decimal(str(raw_percent))
        except InvalidOperation:
            return f"rate item {index} cash_back_percent is invalid"
        if not percent.is_finite() or not Decimal("0") <= percent <= _MAX_RATE_PERCENT:
            return f"rate item {index} cash_back_percent must be between 0 and 100"
        rates[transaction_id] = percent
    return rates


def _audit_observation(
    observation: CreditCardTransactionObservation,
    percent: Decimal,
) -> dict[str, Any] | None:
    expected = int((observation.amount * percent).to_integral_value(rounding=ROUND_FLOOR))
    if expected == observation.rewards_earned:
        return None
    difference = expected - observation.rewards_earned
    return {
        "absolute_difference_points": abs(difference),
        "card_type": observation.card_type,
        "cash_back_percent": _decimal_text(percent),
        "category": observation.category,
        "difference_points": difference,
        "direction": "undercredited" if difference > 0 else "overcredited",
        "expected_points": expected,
        "merchant_name": observation.merchant_name,
        "magnitude_class": "single_point" if abs(difference) == 1 else "multi_point",
        "recorded_points": observation.rewards_earned,
        "transaction_amount": _decimal_text(observation.amount),
        "transaction_date": observation.transaction_date,
        "transaction_id": observation.transaction_id,
    }


def _bounded_nonblank(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > _MAX_IDENTIFIER_CHARS:
        return None
    return normalized


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _error(message: str) -> InternalToolResult:
    return InternalToolResult(content=f"Reward audit rejected: {message}", is_error=True)


__all__ = [
    "AUDIT_CREDIT_CARD_REWARDS_TOOL",
    "PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL",
    "invoke_reward_audit",
    "reward_audit_tool_schema",
    "reward_policy_dossier_tool_schema",
]
