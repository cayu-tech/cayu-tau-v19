"""Opt-in mutation binding and declared cross-outcome interference checks.

The actor interprets policy. This gate checks concrete calls and recorded effects;
it cannot establish that the actor found every requirement or interpreted it well.
"""

import json
from collections import Counter

from cayu_tau3_banking.bridge import ExternalToolAdmission
from cayu_tau3_banking.controller import UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.ledger import ToolActionStatus
from cayu_tau3_banking.public_argument_contracts import (
    _finite_float,
    _reject_constant,
    _unique_object,
)

# Explicit supported operations, not name heuristics. Schema, identity, consent,
# and duplicate-effect checks remain the existing admission gates' responsibility.
PACKET_MUTATIONS = frozenset(
    {
        "file_credit_card_transaction_dispute_4829",
        "order_replacement_credit_card_7291",
        "submit_credit_limit_increase_request_7392",
        "approve_credit_limit_increase_5847",
        "deny_credit_limit_increase_5848",
        "file_debit_card_transaction_dispute_6281",
        "freeze_debit_card_3892",
        "unfreeze_debit_card_3893",
        "close_debit_card_4721",
        "order_debit_card_5739",
        "activate_debit_card_8292",
        "open_bank_account_4821",
        "close_bank_account_7392",
    }
)


def normalized_operation(name, arguments):
    if not isinstance(name, str) or not name.strip() or type(arguments) is not dict:
        raise ValueError("Packet requires a tool name and object arguments")
    if name == "call_discoverable_agent_tool":
        if set(arguments) != {"agent_tool_name", "arguments"}:
            raise ValueError("Wrapper packet requires agent_tool_name and JSON arguments")
        name = arguments["agent_tool_name"]
        arguments = json.loads(
            arguments["arguments"],
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
        if not isinstance(name, str) or not name.strip() or type(arguments) is not dict:
            raise ValueError("Wrapper arguments must decode to an object")
    return name, json.dumps(arguments, sort_keys=True, allow_nan=False, separators=(",", ":"))


class DecisionPacketGate:
    def __init__(self, controller, ledger, *, mutations=PACKET_MUTATIONS):
        self.controller = controller
        self.ledger = ledger
        self.mutations = mutations
        self.counts = Counter()
        self._bindings = {}
        self.receipts = []

    def observe_result(self, call_id):
        """Settle an already authorized exact packet from an observed bank result.

        Never infer completion from a draft, a private calculation or a denial.
        The existing controller validates the nested patch and its evidence.
        """
        binding = self._bindings.pop(call_id, None)
        if binding is None:
            return
        parent, child, expected = binding
        status = self.ledger.tool_action_status(call_id)
        if status is not ToolActionStatus.SUCCEEDED:
            self.counts["execution_not_successful"] += 1
            return
        actual = normalized_operation(
            self.ledger.tool_action_name(call_id), self.ledger.tool_action_arguments(call_id)
        )
        if actual != expected:
            raise ValueError("Observed execution differs from the authorized packet")
        result = self.controller.invoke(
            UPDATE_WORK_PLAN_TOOL,
            {
                "items": [
                    {
                        "id": parent,
                        "obligations": [
                            {
                                "id": child,
                                "status": "resolved",
                                "evidence": "The exact planned operation returned a successful banking result.",
                                "evidence_refs": [{"kind": "tool_result", "reference_id": call_id}],
                            }
                        ],
                    }
                ]
            },
        )
        self.counts["settlement_rejected" if result.is_error else "settled_from_execution"] += 1
        self.receipts = [
            *self.receipts[-15:],
            {
                "outcome": parent,
                "obligation": child,
                "tool_result": call_id,
                "executed": True,
                "plan_settled": not result.is_error,
                "repair": result.content[:1000] if result.is_error else None,
            },
        ]

    def review(self, calls):
        return tuple(self._review(call) for call in calls)

    def _review(self, call):
        name = (
            call.arguments.get("agent_tool_name")
            if call.tool_name == "call_discoverable_agent_tool"
            else call.tool_name
        )
        if not isinstance(name, str) or name not in self.mutations:
            return ExternalToolAdmission(tool_call_id=call.tool_call_id, allowed=True)
        self.counts["reviewed"] += 1
        try:
            actual = normalized_operation(call.tool_name, call.arguments)
        except (TypeError, ValueError, RecursionError):
            return self._deny(call, "packet_arguments_invalid", "Use a concrete JSON object.")
        snapshot = self.controller.export_state()
        items = snapshot["items"]
        candidates = []
        for item in items:
            if item["status"] == "completed":
                continue
            for obligation in item.get("obligations", []):
                packet = obligation.get("planned_tool")
                if obligation.get("kind") != "action" or not packet:
                    continue
                try:
                    expected = normalized_operation(packet["tool_name"], packet["arguments"])
                except (TypeError, ValueError, RecursionError):
                    continue
                if expected == actual:
                    candidates.append((item, obligation, packet))
        if not candidates:
            return self._deny(
                call,
                "packet_missing_or_mismatch",
                "Record one ready action obligation with the exact tool and arguments.",
            )
        # Multiple matching packets must not provide an escape around a conflicting plan.
        if len(candidates) != 1:
            return self._deny(call, "packet_ambiguous", "Bind this mutation to one obligation.")
        item, obligation, packet = candidates[0]
        by_id = {x["id"]: x for x in items}
        statuses = {o["id"]: o["status"] for o in item.get("obligations", [])}
        if (
            obligation["status"] != "ready"
            or any(by_id[d]["status"] != "completed" for d in item["depends_on"])
            or any(
                statuses[d] not in {"resolved", "not_applicable"}
                for d in obligation.get("depends_on", [])
            )
        ):
            return self._deny(
                call,
                "packet_prerequisite_pending",
                "Complete the recorded prerequisites before this mutation.",
            )
        pending = {
            x["id"]
            for x in items
            if x["id"] != item["id"]
            and x["status"] != "completed"
            and x["completion_mode"] == "operation"
        }
        effects = packet.get("outcome_effects", {})
        if pending - effects.keys():
            return self._deny(
                call,
                "packet_interference_unreviewed",
                "Before mutating, read relevant policy and record outcome_effects "
                "for pending outcomes: " + ", ".join(sorted(pending - effects.keys())),
            )
        if pending:
            refs = obligation.get("evidence_refs", [])
            if not any(
                r["kind"] == "knowledge_revision" and r.get("available") is True for r in refs
            ):
                return self._deny(
                    call,
                    "packet_policy_evidence_missing",
                    "Attach available policy evidence to this action obligation.",
                )
        conflicts = {i for i in pending if effects[i] in {"blocks", "unknown"}}
        if conflicts:
            return self._deny(
                call,
                "packet_outcome_conflict",
                "This mutation blocks or has unknown effects on: "
                + ", ".join(sorted(conflicts))
                + ". Resolve dependencies first, or explain the consequence and "
                "obtain an informed customer choice before updating the plan.",
            )
        # Customer acceptance is a provenance check, not a semantic consent verifier.
        if any(effects[i] == "customer_accepted" for i in pending):
            ref = packet.get("customer_choice")
            sequence = self.ledger.customer_message_sequence(ref) if ref else None
            floors = snapshot.get("customer_evidence_floors", {})
            acceptance_floor = max(
                floors.get(i, 0) for i in pending if effects[i] == "customer_accepted"
            )
            if (
                sequence is None
                or sequence != self.ledger.last_customer_message_sequence
                or sequence <= acceptance_floor
            ):
                return self._deny(
                    call,
                    "packet_customer_choice_missing",
                    "Cite a subsequent, current customer message accepting this consequence; "
                    "the original request is not an informed sequencing choice.",
                )
        self.counts["allowed"] += 1
        self._bindings[call.tool_call_id] = (item["id"], obligation["id"], actual)
        return ExternalToolAdmission(tool_call_id=call.tool_call_id, allowed=True)

    def _deny(self, call, code, reason):
        self.counts[code] += 1
        return ExternalToolAdmission(
            tool_call_id=call.tool_call_id, allowed=False, diagnostic_code=code, reason=reason
        )


PACKET_INSTRUCTIONS = """
Decision-packet treatment is enabled for supported banking mutations. Before each
such call, record one ready action obligation with planned_tool: tool_name and
concrete arguments (inner discovered tool plus object arguments is preferred).
Never use prose as arguments or combine several calls into one packet. Schema,
identity, consent and successful execution remain separate checks.
Before an irreversible action, inspect its effects on EVERY other pending operation
outcome, including work awaiting customer input. In planned_tool.outcome_effects,
map each other outcome ID to preserves, blocks, or unknown. Cite the applicable
read policy in the action obligation's evidence_refs and explain the consequence
in its evidence. A future active dispute or pending replacement may change another
operation's eligibility: discover such interactions before committing the change.
For blocks/unknown, resolve the condition, execute the prerequisite outcome first,
or explain the actual tradeoff and ask the customer. Do not silently reverse their
priority or abandon work. After informed acceptance, mark that effect customer_accepted
and set planned_tool.customer_choice to the accepting customer-message ID. Mere
urgency or the initial request is not informed acceptance. Keep deferred outcomes
truthfully pending/blocked, not completed. After new customer input reassess acceptance.
Use updates to existing obligations; do not create another worksheet or reviewer.
After successful execution the runtime records the receipt and resolves the exact
matching action obligation. Do not duplicate that action or copy its arguments into
another plan. This does not complete the parent outcome or waive remaining work.
This gate validates declared effects and reference availability, not interpretation
of policy. Preserve unknowns and do not invent dependencies to satisfy bookkeeping.
"""
PACKET_INSTRUCTIONS += "\nSupported packet-gated operations: " + ", ".join(sorted(PACKET_MUTATIONS))
