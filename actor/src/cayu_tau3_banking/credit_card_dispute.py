"""Deterministic provisional-credit audit and filing admission.

The audit consumes only public Cayu policy revisions and observations already
returned through the normal banking tools. It never reads tau task metadata or
evaluation criteria.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from cayu import ToolCallPart

from cayu_tau3_banking.bridge import ExternalToolAdmission, InternalToolResult
from cayu_tau3_banking.ledger import (
    ConversationLedger,
    CreditCardAccountObservation,
    CreditCardTransactionObservation,
)

AUDIT_CREDIT_CARD_DISPUTES_TOOL = "controller_audit_credit_card_disputes"
PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL = "controller_prepare_credit_card_dispute_policy"
DISCOVERABLE_AGENT_CALL_TOOL = "call_discoverable_agent_tool"
FILE_CREDIT_CARD_DISPUTE_TOOL = "file_credit_card_transaction_dispute_4829"

_MAX_DISPUTES = 24
_MAX_ID_CHARS = 128
_MAX_POLICY_DOCUMENTS = 4
_DATE_FORMAT = "%m/%d/%Y"
_ELIGIBILITY_TITLE = "provisional credit eligibility guidelines (internal)"
_REQUIRED_POLICY_TITLES = frozenset(
    {
        "checking user dispute history (internal)",
        "filing a credit card transaction dispute (internal)",
        "how to find your rho-bank credit card numbers online",
        _ELIGIBILITY_TITLE,
    }
)
_VALID_REASONS = frozenset(
    {
        "unauthorized_fraudulent_charge",
        "duplicate_charge",
        "incorrect_amount",
        "goods_services_not_received",
        "goods_services_not_as_described",
        "canceled_subscription_still_charging",
        "refund_never_processed",
    }
)
_ACCOUNT_AGE_PATTERN = re.compile(r"open for at least (?P<days>\d+) days", re.IGNORECASE)
_GOODS_AGE_PATTERN = re.compile(
    r"goods_services_not_received[^\n]*more than (?P<days>\d+) days",
    re.IGNORECASE,
)
_MINIMUM_AMOUNT_PATTERN = re.compile(
    r"amount must be at least \$(?P<amount>[0-9,]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_MAXIMUM_PRIOR_PATTERN = re.compile(
    r"has not filed more than (?P<count>\d+) disputes",
    re.IGNORECASE,
)
_TIER_LIMIT_PATTERN = re.compile(
    r"^- [^(]+ \((?P<cards>[^)]+)\): Maximum \$(?P<amount>[0-9,]+(?:\.[0-9]+)?)$",
    re.MULTILINE,
)
_DOSSIER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DISPUTE_WORK_PATTERN = re.compile(r"\bdisput(?:e|ed|es|ing)\b", re.IGNORECASE)
_CUSTOMER_FACT_REQUEST_PATTERN = re.compile(
    r"\b(?:ask|clarify|confirm|identify|provide|select|share|state|supply|tell)\b"
    r"[^\n]{0,96}"
    r"(?:\b(?:last|final)\s+(?:four|4)\s+digits?\b|"
    r"\bfull\s+(?:legal\s+)?name\b|\bphone(?:\s+number)?\b|"
    r"\bemail(?:\s+address)?\b|\b(?:home|mailing)?\s*address\b|"
    r"\b(?:whether|if)\b[^\n]{0,48}\b(?:contacted|tried[^\n]{0,24}contact)\b"
    r"[^\n]{0,24}\bmerchant\b|\bpurchase\s+date\b|"
    r"\b(?:issue[- ]noticed\s+date|noticed\s+date|when[^\n]{0,32}noticed)\b|"
    r"\bdispute\s+reason\b|\b(?:requested\s+resolution|resolution\s+requested)\b|"
    r"\b(?:full|partial)\s+refund\b|\bpartial\s+(?:refund\s+)?amount\b|"
    r"\b(?:whether|if)\b[^\n]{0,48}\b(?:keep[^\n]{0,16}card[^\n]{0,16}active|"
    r"cancel|reissue|replace)\b)",
    re.IGNORECASE,
)
_GENERIC_FILING_HANDOFF_PATTERN = re.compile(
    r"\b(?:complete|file|open|submit)\b[^\n]{0,80}"
    r"\b(?:app|branch|channel|chat|online|phone|self[- ]service|support|web(?:site)?)\b|"
    r"\b(?:ask|have|instruct|tell)\b[^\n]{0,32}\b(?:customer|them|you)\b"
    r"[^\n]{0,24}\bto\s+contact\b[^\n]{0,24}\bmerchant\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CreditCardDisputePolicy:
    """Structured policy compiled from one exact Cayu dossier."""

    dossier_id: str
    minimum_account_age_days: int
    goods_not_received_minimum_age_days: int
    minimum_amount: Decimal
    maximum_provisional_credits: int
    eligible_reasons: frozenset[str]
    card_limits: Mapping[str, Decimal]


@dataclass(frozen=True)
class _DisputeCandidate:
    transaction: CreditCardTransactionObservation
    dispute_reason: str
    contacted_merchant: bool
    issue_noticed_on: date


@dataclass(frozen=True)
class CreditCardDisputeAuthorization:
    """Audit-derived fields that every matching filing must preserve."""

    transaction_id: str
    user_id: str
    purchase_date: str
    issue_noticed_date: str
    dispute_reason: str
    contacted_merchant: bool
    eligible_for_provisional_credit: bool


def dispute_policy_dossier_tool_schema() -> Mapping[str, Any]:
    """Return the private policy-family preparation schema."""

    return {
        "name": PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL,
        "description": (
            "Prepare the complete exact-revision Cayu policy family for credit-card dispute "
            "filing, history, and provisional-credit eligibility. Call this only after current "
            "time, card accounts, transactions, and dispute history have been retrieved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "minLength": 1, "maxLength": _MAX_ID_CHARS}
            },
            "required": ["user_id"],
            "additionalProperties": False,
        },
    }


def dispute_audit_tool_schema() -> Mapping[str, Any]:
    """Return the private complete-batch eligibility audit schema."""

    return {
        "name": AUDIT_CREDIT_CARD_DISPUTES_TOOL,
        "description": (
            "Deterministically allocate provisional credit across one complete intended "
            "credit-card dispute batch. Include every transaction the customer currently "
            "asked to dispute exactly once. Put explicitly customer-prioritized transaction "
            "IDs first; all other qualifying transactions are ranked by observed amount."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "minLength": 1, "maxLength": _MAX_ID_CHARS},
                "policy_dossier_id": {
                    "type": "string",
                    "minLength": 64,
                    "maxLength": 64,
                    "pattern": "^[0-9a-f]{64}$",
                },
                "priority_transaction_ids": {
                    "type": "array",
                    "maxItems": _MAX_DISPUTES,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1, "maxLength": _MAX_ID_CHARS},
                },
                "disputes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": _MAX_DISPUTES,
                    "items": {
                        "type": "object",
                        "properties": {
                            "transaction_id": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": _MAX_ID_CHARS,
                            },
                            "dispute_reason": {
                                "type": "string",
                                "enum": sorted(_VALID_REASONS),
                            },
                            "contacted_merchant": {"type": "boolean"},
                            "issue_noticed_date": {
                                "type": "string",
                                "pattern": "^[0-9]{2}/[0-9]{2}/[0-9]{4}$",
                            },
                        },
                        "required": [
                            "transaction_id",
                            "dispute_reason",
                            "contacted_merchant",
                            "issue_noticed_date",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "user_id",
                "policy_dossier_id",
                "priority_transaction_ids",
                "disputes",
            ],
            "additionalProperties": False,
        },
    }


def compile_dispute_policy_dossier(content: str) -> CreditCardDisputePolicy:
    """Compile bounded policy facts from a complete exact-revision dossier."""

    try:
        payload = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise ValueError("Credit-card dispute policy dossier must be valid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Credit-card dispute policy dossier must be an object.")
    dossier_id = payload.get("policy_dossier_id")
    documents = payload.get("documents")
    if (
        not isinstance(dossier_id, str)
        or _DOSSIER_ID_PATTERN.fullmatch(dossier_id) is None
        or type(documents) is not list
        or len(documents) != _MAX_POLICY_DOCUMENTS
    ):
        raise ValueError("Credit-card dispute policy dossier is incomplete.")
    eligibility_text: str | None = None
    seen_titles: set[str] = set()
    for document in documents:
        if not isinstance(document, Mapping):
            raise ValueError("Credit-card dispute policy document must be an object.")
        title = document.get("title")
        text = document.get("text")
        if not isinstance(title, str) or not isinstance(text, str) or not text.strip():
            raise ValueError("Credit-card dispute policy document is malformed.")
        normalized_title = " ".join(title.casefold().split())
        if normalized_title in seen_titles:
            raise ValueError("Credit-card dispute policy titles must be unique.")
        seen_titles.add(normalized_title)
        if normalized_title == _ELIGIBILITY_TITLE:
            eligibility_text = text
    if seen_titles != _REQUIRED_POLICY_TITLES or eligibility_text is None:
        raise ValueError("Credit-card dispute policy family is incomplete.")

    account_days = _single_integer(_ACCOUNT_AGE_PATTERN, eligibility_text, "account age")
    goods_days = _single_integer(_GOODS_AGE_PATTERN, eligibility_text, "purchase age")
    minimum_amount = _single_decimal(
        _MINIMUM_AMOUNT_PATTERN,
        eligibility_text,
        "minimum amount",
    )
    maximum_prior = _single_integer(
        _MAXIMUM_PRIOR_PATTERN,
        eligibility_text,
        "prior-dispute limit",
    )
    reason_section = eligibility_text.partition("Dispute Reason Category:")[2].partition(
        "3. Dispute Amount:"
    )[0]
    eligible_reasons = frozenset(re.findall(r"'([a-z_]+)'", reason_section))
    if not eligible_reasons or not eligible_reasons.issubset(_VALID_REASONS):
        raise ValueError("Provisional-credit eligible reasons could not be compiled.")
    card_limits: dict[str, Decimal] = {}
    for match in _TIER_LIMIT_PATTERN.finditer(eligibility_text):
        limit = _decimal(match.group("amount"))
        for card in match.group("cards").split(","):
            normalized = _normalize_card_type(card)
            if not normalized or normalized in card_limits:
                raise ValueError("Provisional-credit card limits are ambiguous.")
            card_limits[normalized] = limit
    if not card_limits:
        raise ValueError("Provisional-credit card limits could not be compiled.")
    return CreditCardDisputePolicy(
        dossier_id=dossier_id,
        minimum_account_age_days=account_days,
        goods_not_received_minimum_age_days=goods_days,
        minimum_amount=minimum_amount,
        maximum_provisional_credits=maximum_prior + 1,
        eligible_reasons=eligible_reasons,
        card_limits=card_limits,
    )


def invoke_dispute_audit(
    ledger: ConversationLedger,
    arguments: Mapping[str, Any],
    *,
    policy: CreditCardDisputePolicy | None,
) -> InternalToolResult:
    """Audit a complete candidate batch against public observations and policy."""

    if not isinstance(ledger, ConversationLedger):
        raise TypeError("ledger must be a ConversationLedger.")
    if not isinstance(arguments, Mapping) or set(arguments) != {
        "user_id",
        "policy_dossier_id",
        "priority_transaction_ids",
        "disputes",
    }:
        return _error("Expected exactly user_id, policy_dossier_id, priorities, and disputes.")
    user_id = _bounded_id(arguments.get("user_id"))
    if user_id is None:
        return _error("user_id must be bounded nonblank text.")
    if policy is None or arguments.get("policy_dossier_id") != policy.dossier_id:
        return _error("Prepare the current Cayu credit-card dispute policy dossier first.")
    current_date = ledger.current_date
    transactions = ledger.credit_card_transactions(user_id)
    accounts = ledger.credit_card_accounts(user_id)
    history = ledger.credit_card_disputes(user_id)
    if current_date is None:
        return _error("No parsed successful current-time result exists.")
    if transactions is None:
        return _error("No parsed successful credit-card transaction lookup exists for this user.")
    if accounts is None:
        return _error("No parsed successful credit-card account lookup exists for this user.")
    if history is None:
        return _error("No parsed successful credit-card dispute-history lookup exists.")
    candidates_or_error = _parse_candidates(arguments.get("disputes"), transactions, user_id)
    if isinstance(candidates_or_error, str):
        return _error(candidates_or_error)
    candidates = candidates_or_error
    priority_ids_or_error = _priority_ids(arguments.get("priority_transaction_ids"), candidates)
    if isinstance(priority_ids_or_error, str):
        return _error(priority_ids_or_error)
    priority_ids = priority_ids_or_error
    account_by_card = _accounts_by_card_type(accounts)
    if isinstance(account_by_card, str):
        return _error(account_by_card)

    already_filed = {
        item.get("transaction_id")
        for item in ledger.successful_credit_card_dispute_arguments(user_id)
        if isinstance(item.get("transaction_id"), str)
    }
    repeated = sorted(set(candidates) & already_filed)
    if repeated:
        return _error("The candidate batch contains a dispute that already succeeded.")

    base_eligible: dict[str, bool] = {}
    for transaction_id, candidate in candidates.items():
        account = account_by_card.get(_normalize_card_type(candidate.transaction.card_type))
        if account is None or account.opened_on is None:
            return _error("A candidate card lacks one unambiguous observed opening date.")
        card_limit = policy.card_limits.get(_normalize_card_type(candidate.transaction.card_type))
        if card_limit is None:
            return _error("The compiled policy has no limit for an observed card type.")
        purchase_on = _parse_date(candidate.transaction.transaction_date)
        if purchase_on is None:
            return _error("An observed transaction has an invalid purchase date.")
        if not purchase_on <= candidate.issue_noticed_on <= current_date:
            return _error(
                "Every issue-noticed date must be on or after its observed purchase date "
                "and no later than the observed current date."
            )
        purchase_age = current_date - purchase_on
        account_age = current_date - account.opened_on
        base_eligible[transaction_id] = (
            candidate.issue_noticed_on <= current_date
            and account_age.days >= policy.minimum_account_age_days
            and candidate.dispute_reason in policy.eligible_reasons
            and (
                candidate.dispute_reason != "goods_services_not_received"
                or purchase_age.days > policy.goods_not_received_minimum_age_days
            )
            and policy.minimum_amount <= candidate.transaction.amount <= card_limit
            and (
                candidate.dispute_reason == "unauthorized_fraudulent_charge"
                or candidate.contacted_merchant
            )
        )

    period_start = _one_year_before(current_date)
    prior_credits = sum(
        item.provisional_credit_given and period_start <= item.submitted_on <= current_date
        for item in history
    )
    history_transaction_ids = {item.transaction_id for item in history}
    session_credits = sum(
        item.get("eligible_for_provisional_credit") is True
        for item in ledger.successful_credit_card_dispute_arguments(user_id)
        if item.get("transaction_id") not in history_transaction_ids
    )
    slots = max(0, policy.maximum_provisional_credits - prior_credits - session_credits)
    remaining_ids = [item_id for item_id in candidates if item_id not in set(priority_ids)]
    remaining_ids.sort(key=lambda item_id: (-candidates[item_id].transaction.amount, item_id))
    ordered_ids = [*priority_ids, *remaining_ids]
    selected: set[str] = set()
    for transaction_id in ordered_ids:
        if base_eligible[transaction_id] and len(selected) < slots:
            selected.add(transaction_id)

    audited = [
        {
            "contacted_merchant": candidate.contacted_merchant,
            "dispute_reason": candidate.dispute_reason,
            "eligible_for_provisional_credit": transaction_id in selected,
            "issue_noticed_date": candidate.issue_noticed_on.strftime(_DATE_FORMAT),
            "purchase_date": candidate.transaction.transaction_date,
            "transaction_id": transaction_id,
            "user_id": user_id,
        }
        for transaction_id, candidate in candidates.items()
    ]
    audit_body = {
        "audited_disputes": audited,
        "policy_dossier_id": policy.dossier_id,
        "prior_provisional_credits_in_period": prior_credits + session_credits,
        "provisional_credit_slots_before_batch": slots,
        "user_id": user_id,
    }
    audit_id = hashlib.sha256(
        json.dumps(
            audit_body,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return InternalToolResult(
        content=(
            "Credit-card dispute batch audited against exact Cayu policy revisions and "
            "current public observations. Copy every audited eligibility field into the "
            "matching filing; do not file an unaudited transaction.\n"
            + json.dumps(
                {**audit_body, "audit_id": audit_id},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    )


class CreditCardDisputeActionGate:
    """Admit filings only when their policy-derived fields match a current audit."""

    def __init__(self, ledger: ConversationLedger) -> None:
        if not isinstance(ledger, ConversationLedger):
            raise TypeError("ledger must be a ConversationLedger.")
        self._ledger = ledger
        self._authorizations: dict[str, dict[str, CreditCardDisputeAuthorization]] = {}

    def observe_audit(self, result: InternalToolResult, *, expected_user_id: str) -> None:
        if not isinstance(expected_user_id, str) or not expected_user_id.strip():
            raise ValueError("expected_user_id must be nonblank text.")
        user_id, authorizations = _authorizations_from_audit(result)
        normalized_user_id = expected_user_id.strip()
        if user_id != normalized_user_id:
            raise RuntimeError("Credit-card dispute audit authorized a different user.")
        self._authorizations[normalized_user_id] = {
            item.transaction_id: item for item in authorizations
        }

    def render_context(self) -> str | None:
        if not self._authorizations:
            return None
        batches = []
        for user_id, authorizations in sorted(self._authorizations.items()):
            batches.append(
                {
                    "audited_disputes": [
                        {
                            "contacted_merchant": item.contacted_merchant,
                            "dispute_reason": item.dispute_reason,
                            "eligible_for_provisional_credit": (
                                item.eligible_for_provisional_credit
                            ),
                            "issue_noticed_date": item.issue_noticed_date,
                            "purchase_date": item.purchase_date,
                            "transaction_id": item.transaction_id,
                        }
                        for item in authorizations.values()
                    ],
                    "user_id": user_id,
                }
            )
        return (
            "<credit-card-dispute-authorizations>\n"
            "Application-owned results from complete deterministic Cayu policy audits. "
            "Every credit-card filing must preserve the audited transaction, dates, reason, "
            "merchant-contact fact, and provisional-credit flag. The admission boundary "
            "rejects missing, altered, duplicate, or unaudited filings.\n"
            + json.dumps(
                {"schema_version": 1, "batches": batches},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n</credit-card-dispute-authorizations>"
        )

    def pending_filing_count(self) -> int:
        """Return audited transactions without a successful session filing."""

        pending = 0
        for user_id, authorizations in self._authorizations.items():
            successful_ids = {
                item.get("transaction_id")
                for item in self._ledger.successful_credit_card_dispute_arguments(user_id)
            }
            pending += len(set(authorizations).difference(successful_ids))
        return pending

    def terminal_handoff_denial(self, response_state: Mapping[str, Any]) -> str | None:
        """Reject offloading an audited agent filing as generic customer work."""

        pending = self.pending_filing_count()
        if pending == 0:
            return None
        raw_items = response_state.get("items")
        if not isinstance(raw_items, list):
            return None
        for item in raw_items:
            if not isinstance(item, Mapping) or item.get("status") != "awaiting_customer":
                continue
            identity = " ".join(
                value for value in (item.get("id"), item.get("goal")) if isinstance(value, str)
            )
            operation = item.get("next_operation")
            if (
                _DISPUTE_WORK_PATTERN.search(identity) is None
                or not isinstance(operation, Mapping)
                or operation.get("owner") != "customer"
                or operation.get("kind") != "customer_response"
            ):
                continue
            summary = operation.get("summary")
            if (
                isinstance(summary, str)
                and _GENERIC_FILING_HANDOFF_PATTERN.search(summary) is None
                and _CUSTOMER_FACT_REQUEST_PATTERN.search(summary) is not None
            ):
                continue
            return (
                f"{pending} transaction dispute filing(s) already have a complete current "
                "Cayu policy audit. Do not replace those agent-executable filings with a "
                "generic customer self-service, support-channel, or completion-report step. "
                "Update each active dispute item to the exact agent-owned "
                "call_discoverable_agent_tool filing and issue all ready filings together."
            )
        return None

    def review(self, calls: Sequence[ToolCallPart]) -> tuple[ExternalToolAdmission, ...]:
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
            raise TypeError("calls must be a sequence.")
        decisions = []
        proposed_transaction_ids: set[str] = set()
        for call in calls:
            if not isinstance(call, ToolCallPart):
                raise TypeError("calls must contain ToolCallPart instances.")
            reason, transaction_id = self._denial_reason(call)
            if reason is None and transaction_id in proposed_transaction_ids:
                reason = "the same audited transaction appears more than once in this batch"
            if transaction_id is not None:
                proposed_transaction_ids.add(transaction_id)
            decisions.append(
                ExternalToolAdmission(
                    tool_call_id=call.tool_call_id,
                    allowed=reason is None,
                    reason=reason,
                    diagnostic_code=(None if reason is None else "credit_card_dispute_audit"),
                )
            )
        return tuple(decisions)

    def _denial_reason(self, call: ToolCallPart) -> tuple[str | None, str | None]:
        if call.tool_name != DISCOVERABLE_AGENT_CALL_TOOL:
            return None, None
        if call.arguments.get("agent_tool_name") != FILE_CREDIT_CARD_DISPUTE_TOOL:
            return None, None
        nested = _nested_arguments(call.arguments.get("arguments"))
        if nested is None:
            return "credit-card dispute filing requires one valid JSON argument object", None
        user_id = _bounded_id(nested.get("user_id"))
        transaction_id = _bounded_id(nested.get("transaction_id"))
        if user_id is None or transaction_id is None:
            return "credit-card dispute filing requires bounded user and transaction IDs", None
        authorization = self._authorizations.get(user_id, {}).get(transaction_id)
        if authorization is None:
            return (
                "no complete current deterministic Cayu dispute audit authorizes this "
                "transaction; prepare the policy family and audit the complete batch first",
                transaction_id,
            )
        expected = {
            "contacted_merchant": authorization.contacted_merchant,
            "dispute_reason": authorization.dispute_reason,
            "eligible_for_provisional_credit": authorization.eligible_for_provisional_credit,
            "issue_noticed_date": authorization.issue_noticed_date,
            "purchase_date": authorization.purchase_date,
            "transaction_id": authorization.transaction_id,
            "user_id": authorization.user_id,
        }
        if any(
            type(nested.get(key)) is not type(value) or nested.get(key) != value
            for key, value in expected.items()
        ):
            return (
                "the filing differs from deterministic public-policy audit fields; copy the "
                "audited transaction, dates, reason, merchant-contact fact, and eligibility",
                transaction_id,
            )
        if any(
            item.get("transaction_id") == transaction_id
            for item in self._ledger.successful_credit_card_dispute_arguments(user_id)
        ):
            return (
                "this transaction dispute already succeeded; do not file it again",
                transaction_id,
            )
        return None, transaction_id


def _parse_candidates(
    value: object,
    transactions: Sequence[CreditCardTransactionObservation],
    user_id: str,
) -> dict[str, _DisputeCandidate] | str:
    if type(value) is not list or not 1 <= len(value) <= _MAX_DISPUTES:
        return f"disputes must contain 1 through {_MAX_DISPUTES} items."
    observed = {item.transaction_id: item for item in transactions if item.user_id == user_id}
    candidates: dict[str, _DisputeCandidate] = {}
    expected_fields = {
        "transaction_id",
        "dispute_reason",
        "contacted_merchant",
        "issue_noticed_date",
    }
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            return f"dispute item {index} has an invalid shape."
        transaction_id = _bounded_id(raw.get("transaction_id"))
        reason = raw.get("dispute_reason")
        contacted = raw.get("contacted_merchant")
        noticed = _parse_date(raw.get("issue_noticed_date"))
        if transaction_id is None or transaction_id not in observed:
            return f"dispute item {index} does not identify an observed transaction."
        if transaction_id in candidates:
            return f"dispute item {index} duplicates a transaction."
        if reason not in _VALID_REASONS or type(contacted) is not bool or noticed is None:
            return f"dispute item {index} has invalid public dispute facts."
        candidates[transaction_id] = _DisputeCandidate(
            transaction=observed[transaction_id],
            dispute_reason=reason,
            contacted_merchant=contacted,
            issue_noticed_on=noticed,
        )
    return candidates


def _priority_ids(
    value: object,
    candidates: Mapping[str, _DisputeCandidate],
) -> tuple[str, ...] | str:
    if type(value) is not list or len(value) > _MAX_DISPUTES:
        return "priority_transaction_ids must be a bounded array."
    if any(_bounded_id(item) is None for item in value):
        return "priority_transaction_ids must contain bounded nonblank IDs."
    normalized = tuple(item.strip() for item in value)
    if len(set(normalized)) != len(normalized) or not set(normalized).issubset(candidates):
        return "priority_transaction_ids must be unique IDs from disputes."
    return normalized


def _accounts_by_card_type(
    accounts: Sequence[CreditCardAccountObservation],
) -> dict[str, CreditCardAccountObservation] | str:
    found: dict[str, CreditCardAccountObservation] = {}
    for account in accounts:
        normalized = _normalize_card_type(account.card_type)
        if normalized in found:
            return "The latest account lookup has ambiguous duplicate card types."
        found[normalized] = account
    return found


def _authorizations_from_audit(
    result: InternalToolResult,
) -> tuple[str, tuple[CreditCardDisputeAuthorization, ...]]:
    if result.is_error:
        raise ValueError("An error result cannot authorize credit-card dispute filings.")
    try:
        payload = json.loads(result.content.rsplit("\n", 1)[-1])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Successful credit-card dispute audit omitted valid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("Successful credit-card dispute audit payload must be an object.")
    user_id = _bounded_id(payload.get("user_id"))
    raw_items = payload.get("audited_disputes")
    if user_id is None or type(raw_items) is not list or not 1 <= len(raw_items) <= _MAX_DISPUTES:
        raise RuntimeError("Successful credit-card dispute audit has invalid identity or items.")
    items: list[CreditCardDisputeAuthorization] = []
    seen: set[str] = set()
    expected_fields = {
        "contacted_merchant",
        "dispute_reason",
        "eligible_for_provisional_credit",
        "issue_noticed_date",
        "purchase_date",
        "transaction_id",
        "user_id",
    }
    for raw in raw_items:
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise RuntimeError("Successful credit-card dispute audit item is malformed.")
        transaction_id = _bounded_id(raw.get("transaction_id"))
        if (
            transaction_id is None
            or transaction_id in seen
            or raw.get("user_id") != user_id
            or raw.get("dispute_reason") not in _VALID_REASONS
            or type(raw.get("contacted_merchant")) is not bool
            or type(raw.get("eligible_for_provisional_credit")) is not bool
            or _parse_date(raw.get("purchase_date")) is None
            or _parse_date(raw.get("issue_noticed_date")) is None
        ):
            raise RuntimeError("Successful credit-card dispute audit item is inconsistent.")
        seen.add(transaction_id)
        items.append(
            CreditCardDisputeAuthorization(
                transaction_id=transaction_id,
                user_id=user_id,
                purchase_date=raw["purchase_date"],
                issue_noticed_date=raw["issue_noticed_date"],
                dispute_reason=raw["dispute_reason"],
                contacted_merchant=raw["contacted_merchant"],
                eligible_for_provisional_credit=raw["eligible_for_provisional_credit"],
            )
        )
    return user_id, tuple(items)


def _single_integer(pattern: re.Pattern[str], text: str, field: str) -> int:
    matches = tuple(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Provisional-credit {field} is missing or ambiguous.")
    values = tuple(value for value in matches[0].groupdict().values() if value is not None)
    if len(values) != 1:
        raise ValueError(f"Provisional-credit {field} is malformed.")
    value = int(values[0])
    if value <= 0:
        raise ValueError(f"Provisional-credit {field} must be positive.")
    return value


def _single_decimal(pattern: re.Pattern[str], text: str, field: str) -> Decimal:
    matches = tuple(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Provisional-credit {field} is missing or ambiguous.")
    value = _decimal(matches[0].group("amount"))
    if value <= 0:
        raise ValueError(f"Provisional-credit {field} must be positive.")
    return value


def _decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value.replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError("Provisional-credit policy contains an invalid amount.") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError("Provisional-credit policy amount must be finite and nonnegative.")
    return parsed


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, _DATE_FORMAT).date()
    except ValueError:
        return None


def _one_year_before(value: date) -> date:
    try:
        return value.replace(year=value.year - 1)
    except ValueError:
        return value.replace(year=value.year - 1, day=28)


def _bounded_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized and len(normalized) <= _MAX_ID_CHARS else None


def _normalize_card_type(value: str) -> str:
    normalized = " ".join(value.casefold().split())
    return normalized.removesuffix(" card")


def _nested_arguments(value: object) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if type(parsed) is dict else None


def _error(message: str) -> InternalToolResult:
    return InternalToolResult(
        content=f"Credit-card dispute audit rejected: {message}", is_error=True
    )


__all__ = [
    "AUDIT_CREDIT_CARD_DISPUTES_TOOL",
    "PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL",
    "CreditCardDisputeActionGate",
    "CreditCardDisputePolicy",
    "compile_dispute_policy_dossier",
    "dispute_audit_tool_schema",
    "dispute_policy_dossier_tool_schema",
    "invoke_dispute_audit",
]
