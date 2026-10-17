"""Audited public-source navigation, never hidden task/tool implementation access."""

import hashlib
import json

from cayu_tau3_banking.bridge import ExternalToolAdmission

CARDS = [
    {
        "source": "doc_bank_accounts_bank_accounts_(general)_017",
        "sha256": "153d07314e3d5b5d47acf2a05282e15d740c8191b8d51ac85d8e3731bfaa568f",
        "tools": ["apply_checking_account_credit_5829"],
        "owner": "agent",
        "guidance": "Read the full procedure. Fee mischarges require the net correction across ALL discrepancies, not only positive overcharges. Reconcile charged minus required fee per transaction, including missing fees and undercharges with their negative sign; add eligible missing rebates once. Recompute an earlier estimate after reading this procedure. A policy-required net correction is distinct from an unrelated new charge. Explain the net amount and resolve any scope uncertainty before acting; do not invent missing fee rules. Only one credit per account is allowed, so reconcile before applying it.",
    },
    {
        "source": "doc_credit_cards_credit_cards_(general)_013",
        "sha256": "70a14db843c218f58d0312c4cf55d05c268c1b3fecb0299d11264c8be1917e1f",
        "tools": ["get_card_last_4_digits"],
        "owner": "customer",
        "guidance": "Provide via give_discoverable_user_tool, then tell the customer to call get_card_last_4_digits with the exact credit_card_account_id. Do not unlock it as an agent tool. An unavailable card number is not proof that this documented retrieval path is unavailable.",
    },
    {
        "source": "doc_credit_cards_credit_cards_(general)_003",
        "sha256": "75227139324ccabb6ac51ed6f571be4fb4451cbf3f23320599080ebd584f467b",
        "tools": ["submit_cash_back_dispute_0589"],
        "owner": "customer",
        "guidance": "Provide via give_discoverable_user_tool. The customer executes it with user_id and transaction_id. Submission and approved resolution are distinct facts.",
    },
    {
        "source": "doc_credit_cards_credit_cards_(general)_004",
        "sha256": "2fcdea51ed24ee01db0219cc40583839a33a4557bd90bca0dfc379c1e3824e68",
        "tools": ["update_transaction_rewards_3847"],
        "owner": "agent",
        "guidance": "The procedure requires resolved approvals from cash_back_disputes. An empty lookup of a different dispute family cannot establish absence here. Do not invent a lookup capability or treat a customer assertion as independent bank verification. Read the full procedure.",
    },
    {
        "source": "doc_credit_cards_credit_cards_(general)_016",
        "sha256": "c5d5c7845dbff80fd531d59020618d3479990322af0c9d9d9929ef99590053e8",
        "tools": ["get_user_dispute_history_7291"],
        "owner": "agent",
        "guidance": "Documented for credit-card dispute history. Check that the record family matches the actual question; do not substitute it for another database named by a different procedure.",
    },
    {
        "source": "doc_bank_accounts_bank_accounts_(general)_042",
        "sha256": "d0781ff56c31fc00ebc9fb5c5d1a902bee8a22cabddbe893e6381250d5dff59c",
        "tools": ["transfer_to_human_agents"],
        "owner": "agent",
        "guidance": "Read the public reason-code priority guide before choosing a transfer reason: highest applicable tier wins. customer_demands_after_unavailable_offer_refusal is Tier 1; kb_search_unsuccessful_customer_requests_transfer and unconfirmed_external_communication are Tier 2. Interpret applicability from the actual conversation, not keywords.",
    },
]


class PublicOperationGuidance:
    def __init__(self, entries):
        self.cards = []
        for card in CARDS:
            entry = entries.get(card["source"])
            if entry is None:
                continue
            if (
                entry.revision != 1
                or hashlib.sha256(entry.text.encode()).hexdigest() != card["sha256"]
            ):
                raise ValueError("Public operation source changed: " + card["source"])
            self.cards.append(card)

    def context(self):
        return (
            '<public_operation_directory full_documents_delivered="false">'
            + json.dumps([{k: v for k, v in card.items() if k != "sha256"} for card in self.cards])
            + "</public_operation_directory>"
        )

    def review(self, calls):
        result = []
        for call in calls:
            field = {
                "unlock_discoverable_agent_tool": "agent_tool_name",
                "call_discoverable_agent_tool": "agent_tool_name",
                "give_discoverable_user_tool": "discoverable_tool_name",
            }.get(call.tool_name)
            owner = "customer" if call.tool_name == "give_discoverable_user_tool" else "agent"
            card = next(
                (c for c in self.cards if field and call.arguments.get(field) in c["tools"]), None
            )
            wrong = card is not None and card["owner"] != owner
            reason = (
                ("Public source " + card["source"] + ": " + card["guidance"]) if wrong else None
            )
            result.append(
                ExternalToolAdmission(
                    tool_call_id=call.tool_call_id,
                    allowed=not wrong,
                    reason=reason,
                    diagnostic_code="public_tool_owner" if wrong else None,
                )
            )
        return tuple(result)
