"""Resource roles in public banking tool arguments, shared by checks and receipts.

These are schema fields, not classifications of customer language. Identical IDs
in different resource namespaces must not satisfy each other's prerequisites.
"""

from collections.abc import Mapping

RESOURCE_ARGUMENT_KINDS = {
    "account_id": frozenset({"bank_account", "credit_card"}),
    "bank_account_id": frozenset({"bank_account"}),
    "checking_account_id": frozenset({"bank_account"}),
    "savings_account_id": frozenset({"bank_account"}),
    "source_account_id": frozenset({"bank_account"}),
    "destination_account_id": frozenset({"bank_account"}),
    "credit_card_account_id": frozenset({"credit_card"}),
    "card_id": frozenset({"debit_card"}),
    "resource_id": frozenset({"bank_account", "credit_card", "debit_card"}),
}


def resource_targets(arguments: Mapping[str, object], kind: str | None = None) -> set[str]:
    return {
        value
        for field, value in arguments.items()
        if field in RESOURCE_ARGUMENT_KINDS
        and (kind is None or kind in RESOURCE_ARGUMENT_KINDS[field])
        and isinstance(value, str)
        and value
    }
