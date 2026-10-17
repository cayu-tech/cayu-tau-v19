"""Deterministic reconciliation of observed monthly savings interest."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from cayu_tau3_banking.bridge import InternalToolResult
from cayu_tau3_banking.ledger import ConversationLedger

AUDIT_SAVINGS_INTEREST_TOOL = "controller_audit_savings_interest"
PREPARE_SAVINGS_INTEREST_POLICY_TOOL = "controller_prepare_savings_interest_policy"

_MAX_IDENTIFIER_CHARS = 128
_MAX_COMPONENT_NAME_CHARS = 160
_MAX_RATE_ITEMS = 128
_MAX_RATE_PERCENT = Decimal("100")
_MONEY_QUANTUM = Decimal("0.01")
_APY_QUANTUM = Decimal("0.001")
_SILVER_RELATIONSHIP_APY_PERCENT = Decimal("0.025")
_SILVER_ACCOUNT_LEVEL = "silver account"


def savings_interest_policy_dossier_tool_schema() -> Mapping[str, Any]:
    """Return the private Cayu policy-dossier tool schema."""

    return {
        "name": PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
        "description": (
            "Prepare complete exact-revision Cayu APY policy families for one savings account "
            "from the latest successful all-accounts lookup. The dossier covers that savings "
            "product, every observed OPEN checking product, card bonuses, non-stacking rules, "
            "interest reconciliation, credits, and discrepancy reports. Call this after the "
            "all-accounts lookup and before controller_audit_savings_interest, then read every "
            "document and copy its policy_dossier_id into the audit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _MAX_IDENTIFIER_CHARS,
                },
                "account_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _MAX_IDENTIFIER_CHARS,
                },
            },
            "required": ["user_id", "account_id"],
            "additionalProperties": False,
        },
    }


def savings_interest_audit_tool_schema() -> Mapping[str, Any]:
    """Return the private controller tool used for savings-interest arithmetic."""

    rate_item = {
        "type": "object",
        "properties": {
            "account_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": _MAX_IDENTIFIER_CHARS,
            },
            "apy_percent": {"type": "number", "minimum": 0, "maximum": 100},
        },
        "required": ["account_id", "apy_percent"],
        "additionalProperties": False,
    }
    return {
        "name": AUDIT_SAVINGS_INTEREST_TOOL,
        "description": (
            "Deterministically reconcile the latest posted monthly interest credit for one "
            "observed savings account. Supply the documented base/tier APY, exactly one rate "
            "for every OPEN checking account and ACTIVE credit card returned by the latest "
            "all-accounts lookup, and any independently stackable APY components. The "
            "controller validates complete account coverage and typed additional benefit sources, selects only the highest checking "
            "boost and highest card bonus, computes the monthly expected interest and implied "
            "actual APY, and returns copy-ready credit/report arguments. Retrieve the public "
            "account transaction history first, prepare the Cayu savings-interest policy "
            "dossier, read every document, and copy its policy_dossier_id here. This tool does "
            "not retrieve policy or mutate bank state. Source availability and arithmetic checks do not prove your classification: a card-granted relationship bonus is still a card bonus, not an independent extra."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _MAX_IDENTIFIER_CHARS,
                },
                "account_id": {
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
                "base_or_tier_apy_percent": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                },
                "checking_boosts": {
                    "type": "array",
                    "maxItems": _MAX_RATE_ITEMS,
                    "items": rate_item,
                },
                "credit_card_bonuses": {
                    "type": "array",
                    "maxItems": _MAX_RATE_ITEMS,
                    "items": rate_item,
                },
                "other_stackable_components": {
                    "type": "array",
                    "maxItems": _MAX_RATE_ITEMS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "component_kind": {"enum": ["independent_relationship", "independent_promotion", "card_bonus", "checking_boost"], "description": "Classify by the benefit's granting condition, not its name. A card-owned relationship bonus belongs in credit_card_bonuses, never here."},
                            "benefit_id": {"type": "string", "minLength": 1, "maxLength": 128, "description": "Stable entitlement identity; reuse across descriptions of the same benefit."},
                            "source": {
                                "type": "object", "additionalProperties": False,
                                "required": ["entry_id", "revision", "quote"],
                                "properties": {
                                    "entry_id": {"type": "string", "minLength": 1},
                                    "revision": {"type": "integer", "minimum": 1},
                                    "quote": {"type": "string", "minLength": 1, "maxLength": 1600, "description": "Exact delivered passage establishing a distinct additive benefit and its condition; not merely a second description of the same card/checking bonus."}
                                }
                            },
                            "name": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": _MAX_COMPONENT_NAME_CHARS,
                            },
                            "apy_percent": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 100,
                            },
                        },
                        "required": ["name", "apy_percent", "component_kind", "benefit_id", "source"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "user_id",
                "account_id",
                "policy_dossier_id",
                "base_or_tier_apy_percent",
                "checking_boosts",
                "credit_card_bonuses",
                "other_stackable_components",
            ],
            "additionalProperties": False,
        },
    }


def invoke_savings_interest_audit(
    ledger: ConversationLedger,
    arguments: Mapping[str, Any],
    *,
    required_policy_dossier_id: str | None,
    component_source_validator: Callable[[Mapping[str, Any]], bool] | None = None,
) -> InternalToolResult:
    """Audit a public account observation using explicit documented APY components."""

    if not isinstance(ledger, ConversationLedger):
        raise TypeError("ledger must be a ConversationLedger.")
    required = {
        "user_id",
        "account_id",
        "policy_dossier_id",
        "base_or_tier_apy_percent",
        "checking_boosts",
        "credit_card_bonuses",
        "other_stackable_components",
    }
    if not isinstance(arguments, Mapping) or set(arguments) != required:
        return _error("Expected exactly the fields declared by the audit schema.")
    user_id = _bounded_nonblank(arguments["user_id"], _MAX_IDENTIFIER_CHARS)
    account_id = _bounded_nonblank(arguments["account_id"], _MAX_IDENTIFIER_CHARS)
    if user_id is None or account_id is None:
        return _error("user_id and account_id must be bounded nonblank text.")
    policy_dossier_id = _bounded_nonblank(
        arguments["policy_dossier_id"],
        _MAX_IDENTIFIER_CHARS,
    )
    if required_policy_dossier_id is None:
        return _error("Prepare this account's Cayu savings-interest policy dossier first.")
    if policy_dossier_id != required_policy_dossier_id:
        return _error("policy_dossier_id does not match this account's prepared Cayu dossier.")

    bank_accounts = ledger.bank_accounts(user_id)
    credit_cards = ledger.credit_card_accounts(user_id)
    if bank_accounts is None or credit_cards is None:
        return _error("No complete parsed successful all-accounts lookup exists for this user.")
    account = next((item for item in bank_accounts if item.account_id == account_id), None)
    if account is None:
        return _error("account_id was not present in this user's latest account lookup.")
    if account.account_class.casefold() not in {"saving", "savings"}:
        return _error("account_id is not an observed savings account.")
    if account.status.casefold() != "open":
        return _error("The observed savings account is not OPEN.")
    if account.current_holdings <= 0:
        return _error("Savings-interest reconciliation requires positive current holdings.")
    open_savings_count = sum(
        item.account_class.casefold() in {"saving", "savings"} and item.status.casefold() == "open"
        for item in bank_accounts
    )
    transactions = ledger.bank_account_transactions(account_id)
    if transactions is None:
        return _error("No parsed successful transaction lookup exists for this account.")
    interest = next(
        (
            item
            for item in transactions
            if item.transaction_type.casefold() == "interest_credit"
            and item.status.casefold() == "posted"
            and item.amount >= 0
        ),
        None,
    )
    if interest is None:
        return _error("No nonnegative posted interest credit exists in the latest lookup.")
    base_apy = _parse_rate(arguments["base_or_tier_apy_percent"])
    if base_apy is None:
        return _error("base_or_tier_apy_percent must be a finite number from 0 through 100.")

    eligible_checking = {
        item.account_id: item.level
        for item in bank_accounts
        if item.account_class.casefold() == "checking" and item.status.casefold() == "open"
    }
    eligible_cards = {
        item.account_id: item.card_type
        for item in credit_cards
        if item.status.casefold() == "active"
    }
    checking_rates = _parse_account_rates(
        arguments["checking_boosts"],
        expected_ids=set(eligible_checking),
        label="checking_boosts",
    )
    if isinstance(checking_rates, str):
        return _error(checking_rates)
    card_rates = _parse_account_rates(
        arguments["credit_card_bonuses"],
        expected_ids=set(eligible_cards),
        label="credit_card_bonuses",
    )
    if isinstance(card_rates, str):
        return _error(card_rates)
    other_components = _parse_other_components(arguments["other_stackable_components"], component_source_validator)
    if isinstance(other_components, str):
        return _error(other_components)
    relationship_component_names = {
        raw["name"].casefold() for raw in arguments["other_stackable_components"]
        if raw["component_kind"] == "independent_relationship"
    }
    if open_savings_count > 1 and (
        relationship_component_names
        or _SILVER_RELATIONSHIP_APY_PERCENT in other_components.values()
    ):
        return _error(
            "The account observation contains multiple OPEN savings accounts but no "
            "account-specific relationship-eligibility field. Omit the unproven 0.025% "
            "relationship component; do not infer its assignment from product ownership alone."
        )

    other_open_products = sum(
        item.account_id != account_id and item.status.casefold() == "open" for item in bank_accounts
    ) + sum(item.status.casefold() == "active" for item in credit_cards)
    silver_relationship_eligible = (
        account.level.strip().casefold() == _SILVER_ACCOUNT_LEVEL
        and open_savings_count == 1
        and other_open_products > 0
    )
    if silver_relationship_eligible:
        conflicting_relationship_rates = {
            other_components[name]
            for name in relationship_component_names
            if other_components[name] != _SILVER_RELATIONSHIP_APY_PERCENT
        }
        if conflicting_relationship_rates:
            return _error(
                "The documented Silver multi-product relationship APY component is exactly "
                "0.025%; remove the conflicting relationship component."
            )
        if _SILVER_RELATIONSHIP_APY_PERCENT not in other_components.values():
            other_components["multi-product relationship"] = _SILVER_RELATIONSHIP_APY_PERCENT

    selected_checking = _select_highest(checking_rates)
    selected_card = _select_highest(card_rates)
    expected_apy = (
        base_apy
        + (selected_checking[1] if selected_checking is not None else Decimal("0"))
        + (selected_card[1] if selected_card is not None else Decimal("0"))
        + sum(other_components.values(), start=Decimal("0"))
    )
    if expected_apy > _MAX_RATE_PERCENT:
        return _error("The combined expected APY exceeds 100 percent.")

    expected_interest = (
        account.current_holdings * expected_apy / Decimal("100") / Decimal("12")
    ).quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)
    actual_interest = interest.amount.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)
    difference = (expected_interest - actual_interest).quantize(
        _MONEY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )
    actual_apy = (
        interest.amount * Decimal("12") / account.current_holdings * Decimal("100")
    ).quantize(_APY_QUANTUM, rounding=ROUND_HALF_UP)
    direction = "undercredited" if difference > 0 else "overcredited" if difference < 0 else "match"
    credit_arguments = (
        None
        if difference <= 0
        else {
            "account_id": account_id,
            "amount": _decimal_text(difference),
            "credit_type": "interest_correction",
        }
    )
    report_arguments = {
        "account_id": account_id,
        "actual_apy": _decimal_text(actual_apy),
        "amount_difference": _decimal_text(abs(difference)),
        "expected_apy": _decimal_text(expected_apy),
        "user_id": user_id,
    }
    payload = {
        "account": {
            "account_id": account_id,
            "account_level": account.level,
            "current_holdings": _decimal_text(account.current_holdings),
            "user_id": user_id,
        },
        "actual_apy_percent": _decimal_text(actual_apy),
        "actual_interest": _decimal_text(actual_interest),
        "calculation_basis": "current_holdings_times_annual_apy_divided_by_12",
        "credit_arguments": credit_arguments,
        "difference": _decimal_text(difference),
        "direction": direction,
        "expected_apy_percent": _decimal_text(expected_apy),
        "expected_interest": _decimal_text(expected_interest),
        "interest_transaction": {
            "amount": _decimal_text(interest.amount),
            "date": interest.date,
            "transaction_id": interest.transaction_id,
        },
        "other_stackable_components": {
            name: _decimal_text(rate) for name, rate in sorted(other_components.items())
        },
        "policy_dossier_id": policy_dossier_id,
        "additional_component_sources": arguments["other_stackable_components"],
        "verification_scope": "Observed balances, coverage, component structure, available quoted additional sources and arithmetic checked. Actor owns applicability and classification; a second description of one entitlement is not additive.",
        "report_arguments": report_arguments,
        "selected_credit_card_bonus": _selected_payload(selected_card, eligible_cards),
        "selected_checking_boost": _selected_payload(selected_checking, eligible_checking),
    }
    instruction = (
        "Do not apply a credit for a match or overcredit."
        if difference <= 0
        else (
            "After obtaining customer authorization, copy credit_arguments exactly to "
            "apply_savings_account_credit_6831. Only after that succeeds, copy "
            "report_arguments exactly to submit_interest_discrepancy_report_7294."
        )
    )
    return InternalToolResult(
        content=(
            "Savings-interest audit completed from observed account data and explicit "
            "actor-supplied APY components; arithmetic success is not proof of their policy interpretation. "
            + instruction
            + " Do not subtract the posted interest from current_holdings or invent a "
            "statement-cycle day count.\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        )
    )


def _parse_account_rates(
    raw_items: object,
    *,
    expected_ids: set[str],
    label: str,
) -> dict[str, Decimal] | str:
    if type(raw_items) is not list or len(raw_items) > _MAX_RATE_ITEMS:
        return f"{label} must be an array with at most {_MAX_RATE_ITEMS} items"
    rates: dict[str, Decimal] = {}
    for index, raw in enumerate(raw_items):
        if type(raw) is not dict or set(raw) != {"account_id", "apy_percent"}:
            return f"{label} item {index} requires exactly account_id and apy_percent"
        account_id = _bounded_nonblank(raw["account_id"], _MAX_IDENTIFIER_CHARS)
        rate = _parse_rate(raw["apy_percent"])
        if account_id is None or rate is None:
            return f"{label} item {index} has an invalid account_id or apy_percent"
        if account_id in rates:
            return f"{label} item {index} duplicates account_id {account_id!r}"
        rates[account_id] = rate
    missing = sorted(expected_ids - set(rates))
    unexpected = sorted(set(rates) - expected_ids)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing account ids: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected account ids: {', '.join(unexpected)}")
        return f"{label} requires exact eligible-account coverage; {'; '.join(details)}"
    return rates


def _parse_other_components(raw_items: object, source_validator=None) -> dict[str, Decimal] | str:
    if type(raw_items) is not list or len(raw_items) > _MAX_RATE_ITEMS:
        return f"other_stackable_components must be an array with at most {_MAX_RATE_ITEMS} items"
    components: dict[str, Decimal] = {}
    identities = set()
    source_clauses = set()
    for index, raw in enumerate(raw_items):
        if type(raw) is not dict or set(raw) != {"name", "apy_percent", "component_kind", "benefit_id", "source"}:
            return f"other_stackable_components item {index} requires typed benefit identity and an exact delivered source; classify a card or checking bonus in its existing category, not as an additional component"
        if raw["component_kind"] not in {"independent_relationship", "independent_promotion"}:
            return "Card/checking-granted bonuses belong only in credit_card_bonuses/checking_boosts, even when their source calls them relationship bonuses. Do not add the same entitlement again."
        identity = _bounded_nonblank(raw["benefit_id"], 128)
        if identity is None or identity in identities:
            return "Each additional benefit requires one distinct stable benefit_id."
        source = raw["source"]
        if (not isinstance(source, dict) or set(source) != {"entry_id", "revision", "quote"}
            or source_validator is None or not source_validator(source)):
            return "Additional components require an exact passage from a delivered source revision."
        clause = (source["entry_id"], source["revision"], source["quote"])
        if clause in source_clauses:
            return "The same source clause cannot establish two additional benefits."
        identities.add(identity)
        source_clauses.add(clause)
        name = _bounded_nonblank(raw["name"], _MAX_COMPONENT_NAME_CHARS)
        rate = _parse_rate(raw["apy_percent"])
        if name is None or rate is None:
            return f"other_stackable_components item {index} has an invalid name or apy_percent"
        normalized_name = name.casefold()
        if normalized_name in components:
            return f"other_stackable_components item {index} duplicates name {name!r}"
        components[normalized_name] = rate
    return components


def _parse_rate(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        rate = Decimal(str(value))
    except InvalidOperation:
        return None
    if not rate.is_finite() or not Decimal("0") <= rate <= _MAX_RATE_PERCENT:
        return None
    return rate


def _select_highest(rates: Mapping[str, Decimal]) -> tuple[str, Decimal] | None:
    if not rates:
        return None
    return min(rates.items(), key=lambda item: (-item[1], item[0]))


def _selected_payload(
    selected: tuple[str, Decimal] | None,
    labels: Mapping[str, str],
) -> Mapping[str, str] | None:
    if selected is None:
        return None
    account_id, rate = selected
    return {
        "account_id": account_id,
        "apy_percent": _decimal_text(rate),
        "product": labels[account_id],
    }


def _bounded_nonblank(value: object, maximum_chars: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > maximum_chars:
        return None
    return normalized


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _error(message: str) -> InternalToolResult:
    return InternalToolResult(
        content=f"Savings-interest audit rejected: {message}",
        is_error=True,
    )


__all__ = [
    "AUDIT_SAVINGS_INTEREST_TOOL",
    "PREPARE_SAVINGS_INTEREST_POLICY_TOOL",
    "invoke_savings_interest_audit",
    "savings_interest_audit_tool_schema",
    "savings_interest_policy_dossier_tool_schema",
]
