"""Exact public product-title bindings, never customer-intent classification.

Only unambiguous names explicitly appearing before ':' in public document titles
are bound. Unknown/ambiguous names remain the actor's responsibility. No product
is selected or argument silently rewritten.
"""

CATEGORY_TYPES = {
    "business_savings": "business_savings",
    "business_checking": "business_checking",
    "personal_savings": "savings",
    "personal_checking": "checking",
}


class ProductTypeBindings:
    def __init__(self, records):
        self.names = {}
        for row in records:
            expected = CATEGORY_TYPES.get(row["category"])
            if expected is None:
                continue
            title, separator, _ = row["title"].partition(":")
            if not separator:
                continue
            self.names.setdefault(title.strip().casefold(), {}).setdefault(expected, []).append(
                row["entry_id"]
            )

    def error(self, name, arguments):
        if name != "open_bank_account_4821":
            return None
        label = arguments.get("account_class")
        if not isinstance(label, str):
            return None
        matches = self.names.get(label.strip().casefold(), {})
        if len(matches) != 1:
            return None
        expected, sources = next(iter(matches.items()))
        if arguments.get("account_type") == expected:
            return None
        return (
            f"Product/type mismatch: public product title {label!r} belongs to "
            f"{expected}; account_type={arguments.get('account_type')!r} represents a different "
            f"resource. Source: {sources[0]}. Confirm the selected product's name and tool "
            "type together. Nothing executed; this binding does not establish eligibility or consent."
        )
