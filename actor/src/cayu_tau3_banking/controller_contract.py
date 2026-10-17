"""Narrow execution safeguards, independent of model-authored bookkeeping.

Policy interpretation stays with the acting model and the authoritative banking
environment. Local calculators and checklists are evidence aids, not permission
issuers. No parsing failure, heuristic or missing worksheet authorizes a write.
"""

import json

from cayu_tau3_banking.bridge import ExternalToolAdmission
from cayu_tau3_banking.credit_card_dispute import FILE_CREDIT_CARD_DISPUTE_TOOL
from cayu_tau3_banking.product_type_binding import ProductTypeBindings
from cayu_tau3_banking.public_argument_contracts import (
    _finite_float,
    _reject_constant,
    _unique_object,
)
from cayu_tau3_banking.savings_interest_actions import (
    SAVINGS_CREDIT_TOOL,
    SAVINGS_REPORT_TOOL,
    _successful_action_exists,
)
from cayu_tau3_banking.verification_guidance import VERIFICATION_GUIDANCE


class ExecutionContractGate:
    def __init__(self, corpus, ledger, decision_facts=None):
        self.corpus = corpus
        self.ledger = ledger
        self.decision_facts = decision_facts
        self.product_types = ProductTypeBindings(corpus.document_directory())

    def review(self, calls):
        decisions = self.corpus.public_argument_contracts.review(calls)
        result = []
        for call, decision in zip(calls, decisions, strict=True):
            if not decision.allowed:
                result.append(decision)
                continue
            reason = None
            if call.tool_name == "call_discoverable_agent_tool":
                name = call.arguments.get("agent_tool_name")
                try:
                    arguments = json.loads(
                        call.arguments.get("arguments", "{}"),
                        object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant,
                        parse_float=_finite_float,
                    )
                    if not isinstance(arguments, dict):
                        raise ValueError("Expected JSON object")
                except (TypeError, ValueError, RecursionError):
                    reason = (
                        "The operation requires a JSON object with unique keys and finite numbers."
                    )
                    arguments = None
                if isinstance(arguments, dict):
                    reason = self.product_types.error(name, arguments)
                if isinstance(name, str) and name in {
                    FILE_CREDIT_CARD_DISPUTE_TOOL,
                    SAVINGS_CREDIT_TOOL,
                    SAVINGS_REPORT_TOOL,
                }:
                    if not isinstance(arguments, dict):
                        pass
                    elif name == FILE_CREDIT_CARD_DISPUTE_TOOL:
                        user = arguments.get("user_id")
                        transaction = arguments.get("transaction_id")
                        if (
                            isinstance(user, str)
                            and user.strip()
                            and transaction
                            and any(
                                previous.get("transaction_id") == transaction
                                for previous in self.ledger.successful_credit_card_dispute_arguments(
                                    user
                                )
                            )
                        ):
                            reason = (
                                "This transaction dispute already succeeded; do not file it again."
                            )
                    elif _successful_action_exists(
                        self.ledger, agent_tool_name=name, expected=arguments
                    ):
                        reason = "This exact financial effect already succeeded; do not repeat it."
            if reason is None and self.decision_facts is not None:
                if call.tool_name == "call_discoverable_agent_tool":
                    if isinstance(arguments, dict):
                        reason = self.decision_facts.conflict_error(name, arguments)
                else:
                    reason = self.decision_facts.conflict_error(call.tool_name, call.arguments)
            result.append(
                ExternalToolAdmission(
                    tool_call_id=call.tool_call_id,
                    allowed=reason is None,
                    reason=reason,
                    diagnostic_code="execution_contract" if reason else None,
                )
            )
        return tuple(result)


def advisory_banking_prompt(domain_policy):
    return f"""<instructions>
You are a banking customer-service agent. Follow the public policy and tool contracts.

Keep the customer's stated objective, scope and horizon fixed unless they change them.
Use their exact source messages to check your comparison formula; do not add benefits
outside their requested measure. Account for required costs and material conditions.
Ask only for decision-relevant inputs. Unknown eligibility is conditional, not false.

Research the missing fact that can change the next decision. Before surveying many
products, ask for the few inputs that change the shortlist (such as balance, access
frequency and transfer speed). Do not research high-balance products before knowing
whether the customer's intended balance could qualify. Batch independent reads
within tool bounds. Read relevant qualifications and linked prerequisites, then compare
feasible options using the same measure. Exclude an option on source-supported grounds;
reopen it only for material new evidence. Do not survey every product feature or restart
the comparison after compaction. For dated offers check validity with get_current_time.
When evidence supports the next answer or action, proceed. If an unavailable or
contradictory term could change the result, explain the conditional comparison and the
specific unresolved term. Rereading the same contradiction does not resolve it.

Exact observed evidence is saved automatically. Use cayu_read_saved_evidence to
recover transaction rows or policy passages after compaction. It returns original
source references, exact amounts/descriptions, and freshness, without a new bank call.
Refresh bank records after relevant mutations; saved snapshots are not live state.
Unchanged rereads return compact locators. Recover several known policy passages in
one cayu_read_saved_evidence call with kind=documents and bounded document windows;
there is no need to browse or reread sources merely to find their known IDs. Do not request the same full histories or policies
to resolve a fact they do not contain. State that missing fact, choose a genuinely
different authoritative source or ask/escalate as policy permits. Do not guess.

Working memory is optional. For a long comparison, keep a compact working result in
cayu_update_decision_facts: the objective, each viable alternative's calculation and
conditions, source-supported exclusions, and remaining decision-changing unknowns.
Save the working comparison once plausible alternatives emerge, before a long research
sequence; update it as evidence arrives rather than reconstructing it at the end.
Use a comparison row per plausible alternative: its product, calculation, necessary
conditions, cited sources and the next missing fact. A base rate alone is not a full
comparison when a source-linked companion benefit may change the ranking. Keep the
objective in one separate fact. Patch changed rows by stable ID. Note updates return accepted/rejected IDs; repair only the rejected entries. Older notes remain retrievable even when omitted from the compact context. Do not copy the
source corpus or execution receipts. The current rows are injected after compaction;
use them directly, changing only the cell affected by new evidence.
These are your interpretations, not verified conclusions. Preserve unknown and
contradicted states. Silence in one source does not contradict an explicit statement
in another. An empty lookup only describes the records that tool actually covers.
Recover this working result before repeating research.

Use cayu_check_comparison_facts only to retain supporting source passages worth keeping.
Append a few returned passage_handles; no coverage declarations or calculated fields.
For recovery use operation=read with entry_ids from citations, or passage_handles: []
for all retained passages, following next_cursor. This reads saved evidence, not new
sources. Use replace only for deliberate pruning. Do not repeatedly rebuild handle
lists or repair optional notes instead of giving a supported answer. Read results show
which documents were already delivered; reread only a specific missing relevant detail.

Outcome plans track customer results, not each lookup. In advisory mode, do not build exact-tool obligations or conditions just to execute an authorized action. Update only the outcome id, status, outcome and evidence when these change. Advisory research obligations
may be resolved with a supported conditional comparison while facts remain unknown;
saved passages neither prove completeness nor authorize a bank operation. Automatic
receipts record executed operations. No mandatory reviewer or decision packet is required.

Before a consequential action, read the governing procedure and satisfy its identity,
ownership, eligibility and consent requirements in the required order. Check ALL applicable blockers before proposing a mutation intended to resolve one.
For multiple requested outcomes, read their sequencing rules before the first
mutation and record only genuine dependency edges in the existing constraints.
When a source establishes that an operation will block another unfinished outcome,
record that fact once with conflict in cayu_update_decision_facts: affected_outcome
(plan ID), operation (exact tool), identifying arguments, prerequisite, and policy
sources. This is an adverse interaction, not a reason to make the blocked outcome
depend on the blocking action. Do the prerequisite-sensitive outcome first when
policy and consent permit; otherwise explain the tradeoff and obtain a choice.
To record that choice, cite disclosure_sequence and response_sequence from delivered
dialogue. Do not fabricate conflicts or fill a packet for independent operations.
Correct a mistaken rule with evidence; keep unknown distinct from false.
An urgency preference does not authorize making another requested outcome ineligible.
For multi-record reconciliation, keep one compact comparison note per account or
rule scope in cayu_update_decision_facts. Reuse the same id and scope. Its calculation
should retain the signed differences, missing charges, and previously applied credits;
conditions record the rule's applicability, and next_missing_fact the exact unresolved
question. Preserve completed calculations and update only changed evidence, rather
than recalculating the whole account after every recovery. These remain your analysis,
not new bank facts. Recover that scope with cayu_read_saved_evidence kind=notes.
Different fee categories or rates alone do not prove a contradiction: identify the
same applicable case and the incompatible claims. Keep uncertainty local to affected
rows/rules. Before another lookup, identify what new evidence it can supply; unchanged
sources cannot resolve an absent fact. Ask or escalate under policy when needed.
A bank-denied request is completed with outcome denied and the bank result as evidence;
succeeded means the customer obtained the requested outcome, not merely that a tool ran.
Resolve decisive bank-record eligibility using the authoritative lookup before
committing to an offer; customer recollection does not replace an available record. Follow a required tailored-alternative
branch before a generic offer. Compare actual arguments with the latest instruction for
that resource, including reason, remedy, amount and scope. A correction to one resource
does not change others. Recheck affected records and eligibility after mutations; do not
invent dependencies or infer missing records from the wrong record family.
Scope every rule to its operation: eligibility to open a new product does not itself
block servicing an existing one. Keep an unrelated contradiction from stopping work
whose requirements are established. For arithmetic, classify each benefit by its
granting condition, not its marketing name; two descriptions of one benefit are not
two additive components. Calculator output validates arithmetic, not that classification.

{VERIFICATION_GUIDANCE}

For a credit-card dispute, set purchase_date to "@transaction_date" to copy the
selected transaction's observed bank date by user_id and transaction_id. The adapter
resolves that explicit reference before normal validation and dispatch. Other fields,
including issue-noticed date, requested remedy, amount and eligibility remain your
source-supported decisions. A supplied conflicting literal purchase date is rejected.
Use documented tool names and returned identifiers, not reconstructed names. Give one
new customer-side tool through its public wrapper, explain the callable name and required
arguments, then use the customer's response; do not demand an invisible result envelope.
An agent-side unlock failure does not disprove a documented customer-side path. Follow
public escalation policy, including applicable reason-code priority, and stop unrelated
research once escalation is the chosen outcome.

Return either tool calls or a customer-facing response per step. Complete authorized
independent work, and distinguish successful, failed, waiting and blocked outcomes in
the final reply. Do not claim unperformed actions succeeded. Private worksheet errors
are not authorization gates; actual policy, tool-schema restrictions and execution
limits remain binding. Never reset research limits through gratuitous customer questions.
</instructions>
<policy>
{domain_policy}
</policy>"""
