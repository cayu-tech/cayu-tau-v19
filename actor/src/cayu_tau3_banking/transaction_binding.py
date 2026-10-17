"""Explicit binding of transaction data, never interpretation of customer intent."""

import json

TRANSACTION_DATE_REFERENCE = "@transaction_date"
DISPUTE_TOOL = "file_credit_card_transaction_dispute_4829"


def bind_transaction_fields(ledger, tool_name, arguments):
    if (
        tool_name != "call_discoverable_agent_tool"
        or arguments.get("agent_tool_name") != DISPUTE_TOOL
    ):
        return arguments
    raw = arguments.get("arguments")
    if not isinstance(raw, str):
        return arguments  # Ordinary public-schema validation reports this.
    try:
        values = json.loads(raw)
    except (ValueError, TypeError):
        return arguments
    if not isinstance(values, dict):
        return arguments
    user = values.get("user_id")
    transaction = values.get("transaction_id")
    if not isinstance(user, str) or not user.strip():
        if values.get("purchase_date") == TRANSACTION_DATE_REFERENCE:
            raise ValueError(
                "Transaction-date binding requires an explicit user_id and transaction_id."
            )
        return arguments
    matches = [
        row
        for row in (ledger.credit_card_transactions(user) or ())
        if row.transaction_id == transaction and row.user_id == user
    ]
    if len(matches) != 1:
        if values.get("purchase_date") == TRANSACTION_DATE_REFERENCE:
            raise ValueError(
                "Read this user's transaction record before binding its purchase date."
            )
        return arguments
    date = matches[0].transaction_date
    supplied = values.get("purchase_date")
    if supplied == TRANSACTION_DATE_REFERENCE:
        return {**arguments, "arguments": json.dumps({**values, "purchase_date": date})}
    if supplied is not None and supplied != date:
        raise ValueError(
            f"purchase_date disagrees with the observed transaction {transaction}: {date}. "
            "Use @transaction_date to bind the observed value, or resolve the source discrepancy. "
            "No remedy, amount, consent or eligibility was inferred."
        )
    return arguments
