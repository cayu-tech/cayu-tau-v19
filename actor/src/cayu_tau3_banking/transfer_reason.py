"""Evidence-bound validation for public human-transfer reason codes.

The benchmark exposes a tiered reason taxonomy in public knowledge. This module does
not decide whether a transfer should happen. It only prevents an external transfer
call from carrying a reason that is invalid or contradicted by the bounded dialogue
the customer and agent actually observed.
"""

from __future__ import annotations

import re
from collections.abc import Collection

from cayu_tau3_banking.customer_intent import (
    customer_declines_human_assistance,
    customer_requests_human_assistance,
)
from cayu_tau3_banking.ledger import ConversationLedger

_TRANSFER_REASONS = frozenset(
    {
        "abusive_customer_behavior",
        "accessibility_or_special_needs",
        "account_closure_request",
        "account_ownership_dispute",
        "complex_billing_dispute",
        "customer_demands_after_unavailable_offer_refusal",
        "customer_frustrated_demands_human",
        "customer_requests_human_no_specific_reason",
        "deceased_account_holder",
        "fraud_or_security_concern",
        "kb_search_unsuccessful_customer_requests_transfer",
        "legal_or_regulatory_matter",
        "other",
        "request_completed_customer_wants_human_followup",
        "specialized_department_required",
        "supervisor_request_service_complaint",
        "technical_system_error",
        "third_party_inquiry",
        "unconfirmed_external_communication",
    }
)
_UNAVAILABLE_OFFER_REFUSAL_REASON = "customer_demands_after_unavailable_offer_refusal"
_UNCONFIRMED_EXTERNAL_REASON = "unconfirmed_external_communication"
_OFFER_TERM = r"(?:offer|promotion|promo|program|bonus|deal|waiver|statement credit)"
_EXTERNAL_MEDIUM = (
    r"(?:e-?mail|letter|flyer|mail(?:er)?|postcard|advertisement|advert|message|"
    r"notification|text(?: message)?)"
)
_EXTERNAL_OFFER_CLAIM_PATTERNS = (
    re.compile(
        rf"\b(?:received|got|have|was sent|sent me|came in|saw)\b.{{0,80}}"
        rf"\b{_EXTERNAL_MEDIUM}"
        rf"\b.{{0,120}}\b{_OFFER_TERM}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_EXTERNAL_MEDIUM}\b.{{0,100}}\b(?:says?|mentions?|promises?|advertises?)"
        rf"\b.{{0,100}}\b{_OFFER_TERM}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_OFFER_TERM}\b.{{0,100}}\b(?:in|from|on)\b.{{0,40}}\b{_EXTERNAL_MEDIUM}\b",
        re.IGNORECASE,
    ),
)
_UNCERTAIN_OFFER_RESPONSE_PATTERN = re.compile(
    r"\b(?:cannot|can't|could not|couldn't|unable to|did not|didn't)\b.{0,40}"
    r"\b(?:confirm|find|locate|verify|be confirmed|be found|be located|be verified)\b"
    rf"|\bno\s+documented\b.{{0,80}}\b{_OFFER_TERM}\b",
    re.IGNORECASE,
)
_OFFER_REFERENCE_PATTERN = re.compile(rf"\b{_OFFER_TERM}\b", re.IGNORECASE)
_DEFINITE_UNAVAILABLE_OFFER_PATTERNS = (
    re.compile(
        rf"\b{_OFFER_TERM}\b.{{0,80}}\b(?:does not exist|doesn't exist|is not available|"
        r"isn't available|is unavailable|not offered|no longer available|not in (?:our|the) "
        r"system)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:do not|don't|does not|doesn't)\b.{{0,30}}\b(?:have|offer)\b.{{0,80}}"
        rf"\b{_OFFER_TERM}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bno such\b.{{0,40}}\b{_OFFER_TERM}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_OFFER_TERM}\b.{{0,160}}\b(?:expired|ended|closed|cannot be (?:added|applied|"
        r"honou?red|redeemed))\b",
        re.IGNORECASE,
    ),
)
_DEFINITE_OFFER_OPERATION_REFUSAL_PATTERN = re.compile(
    rf"\b(?:cannot|can't|unable to)\b.{{0,40}}\b(?:apply|honou?r|redeem|add|attach)\b"
    rf".{{0,100}}\b{_OFFER_TERM}\b",
    re.IGNORECASE,
)
_DEFINITE_CONTRADICTED_OFFER_PATTERN = re.compile(
    rf"\b(?:closest|only)\s+(?:documented|published|available)\s+{_OFFER_TERM}\b"
    r".{0,320}\b(?:not|rather than|instead of)\b",
    re.IGNORECASE,
)
_CONDITIONAL_OFFER_OPERATION_PATTERN = re.compile(
    r"\b(?:until|unless|once|after you|if you|provided that)\b",
    re.IGNORECASE,
)
_PERSISTENCE_PATTERN = re.compile(
    rf"\b{_OFFER_TERM}\b|\b(?:again|still|insist|but|must|should)\b",
    re.IGNORECASE,
)

_INVALID_TRANSFER_REASON = "transfer reason must be one exact public transfer reason code"
_UNEXPOSED_TRANSFER_REASON = (
    "the exact transfer reason code has not appeared in Cayu-retrieved public knowledge; "
    "retrieve the human-transfer reason policy before retrying"
)
_UNPROVEN_UNAVAILABLE_OFFER_REFUSAL = (
    "customer_demands_after_unavailable_offer_refusal requires observed proof that the bank "
    "definitively said the offer does not exist or is unavailable before the final human "
    "demand, after multiple customer persistence turns during the offer-resolution exchange"
)
_UNPROVEN_UNCONFIRMED_EXTERNAL = (
    "unconfirmed_external_communication requires an observed claim about an offer from a "
    "letter, email, flyer, or similar external communication, a successful Cayu knowledge "
    "search after that claim, and then an agent response that the offer could not be "
    "confirmed, found, located, or verified"
)
_HIGHER_PRIORITY_UNAVAILABLE_OFFER_REFUSAL = (
    "customer_demands_after_unavailable_offer_refusal is fully established and has higher "
    "priority than unconfirmed_external_communication"
)
_CUSTOMER_DECLINED_TRANSFER = (
    "the latest customer turn explicitly declines or revokes a human transfer"
)
_CURRENT_HUMAN_REQUEST_REQUIRED = (
    "the selected transfer reason requires a human request in the latest customer turn"
)
_CUSTOMER_DISPOSITION_TRANSFER_REASONS = frozenset(
    {
        "customer_frustrated_demands_human",
        "customer_requests_human_no_specific_reason",
        "request_completed_customer_wants_human_followup",
        "supervisor_request_service_complaint",
    }
)
_CURRENT_REQUEST_TRANSFER_REASONS = frozenset(
    {
        *_CUSTOMER_DISPOSITION_TRANSFER_REASONS,
        "kb_search_unsuccessful_customer_requests_transfer",
    }
)
_LOWER_PRIORITY_OFFER_REASONS = frozenset({*_CUSTOMER_DISPOSITION_TRANSFER_REASONS, "other"})
_HIGHER_PRIORITY_UNCONFIRMED_EXTERNAL = (
    "unconfirmed_external_communication is fully established and is more accurate than the "
    "selected lower-tier customer-disposition or catch-all reason"
)
_TYPOGRAPHIC_PUNCTUATION = str.maketrans({"\u2018": "'", "\u2019": "'"})


def transfer_reason_denial(
    reason: object,
    *,
    ledger: ConversationLedger,
    require_knowledge_evidence: bool,
    knowledge_identifiers: Collection[str],
    knowledge_search_dialogue_sequences: Collection[int],
) -> str | None:
    """Return a correction for an unsupported reason, otherwise ``None``."""

    normalized_reason = reason.strip() if isinstance(reason, str) and reason.strip() else None
    if normalized_reason not in _TRANSFER_REASONS:
        return _INVALID_TRANSFER_REASON
    if require_knowledge_evidence and normalized_reason not in knowledge_identifiers:
        return _UNEXPOSED_TRANSFER_REASON
    latest_customer_text = _latest_customer_text(ledger)
    if latest_customer_text is not None and customer_declines_human_assistance(
        latest_customer_text
    ):
        return _CUSTOMER_DECLINED_TRANSFER
    if normalized_reason in _CURRENT_REQUEST_TRANSFER_REASONS and (
        latest_customer_text is None or not customer_requests_human_assistance(latest_customer_text)
    ):
        return _CURRENT_HUMAN_REQUEST_REQUIRED
    unavailable_offer_refusal = _supports_unavailable_offer_refusal(ledger)
    unconfirmed_external_offer = _supports_unconfirmed_external_offer(
        ledger,
        require_search_evidence=require_knowledge_evidence,
        knowledge_search_dialogue_sequences=knowledge_search_dialogue_sequences,
    )
    if normalized_reason == _UNAVAILABLE_OFFER_REFUSAL_REASON:
        if not unavailable_offer_refusal:
            return _UNPROVEN_UNAVAILABLE_OFFER_REFUSAL
    elif normalized_reason == _UNCONFIRMED_EXTERNAL_REASON:
        if unavailable_offer_refusal:
            return _HIGHER_PRIORITY_UNAVAILABLE_OFFER_REFUSAL
        if not unconfirmed_external_offer:
            return _UNPROVEN_UNCONFIRMED_EXTERNAL
    elif normalized_reason in _LOWER_PRIORITY_OFFER_REASONS:
        if unavailable_offer_refusal:
            return _HIGHER_PRIORITY_UNAVAILABLE_OFFER_REFUSAL
        if unconfirmed_external_offer:
            return _HIGHER_PRIORITY_UNCONFIRMED_EXTERNAL
    return None


def current_offer_capability_gap(
    *,
    ledger: ConversationLedger,
    knowledge_identifiers: Collection[str],
    knowledge_search_dialogue_sequences: Collection[int],
) -> bool:
    """An actual current offer gap may override the within-capability request count.

    Do not let an old offer discussion waive policy for an unrelated later request.
    This is evidence for controller timing, not permission to execute a transfer.
    """
    latest = _latest_customer_text(ledger) or ""
    if not customer_requests_human_assistance(latest) or customer_declines_human_assistance(latest):
        return False
    assistant = next(
        (item for item in reversed(ledger.recent_dialogue_messages()) if item.role == "assistant"),
        None,
    )
    if assistant is None or not (
        _is_uncertain_offer_response(assistant.content)
        or _is_definite_unavailable_offer_response(assistant.content)
    ):
        return False
    if not any(sequence < assistant.sequence for sequence in knowledge_search_dialogue_sequences):
        return False
    return any(
        transfer_reason_denial(
            reason,
            ledger=ledger,
            require_knowledge_evidence=True,
            knowledge_identifiers=knowledge_identifiers,
            knowledge_search_dialogue_sequences=knowledge_search_dialogue_sequences,
        )
        is None
        for reason in (_UNCONFIRMED_EXTERNAL_REASON, _UNAVAILABLE_OFFER_REFUSAL_REASON)
    )


def _latest_customer_text(ledger: ConversationLedger) -> str | None:
    customer_messages = tuple(
        item for item in ledger.recent_dialogue_messages() if item.role == "customer"
    )
    return customer_messages[-1].content if customer_messages else None


def _supports_unavailable_offer_refusal(ledger: ConversationLedger) -> bool:
    dialogue = ledger.recent_dialogue_messages()
    customer_messages = tuple(item for item in dialogue if item.role == "customer")
    if not customer_messages or not customer_requests_human_assistance(
        customer_messages[-1].content
    ):
        return False
    definite_refusals = tuple(
        item
        for item in dialogue
        if item.role == "assistant" and _is_definite_unavailable_offer_response(item.content)
    )
    if not definite_refusals:
        return False

    # A customer may persist through several attempts to identify or verify an offer
    # before the bank can make the definitive refusal. Count that whole observed
    # exchange, but require the definite refusal itself to precede the final demand.
    # This preserves the distinction from a merely unsuccessful knowledge search.
    final_demand = customer_messages[-1]
    if not any(item.sequence < final_demand.sequence for item in definite_refusals):
        return False
    offer_response_sequences = tuple(
        item.sequence
        for item in dialogue
        if item.role == "assistant"
        and (
            _is_definite_unavailable_offer_response(item.content)
            or _is_uncertain_offer_response(item.content)
        )
    )
    if not offer_response_sequences:
        return False
    first_offer_response = min(offer_response_sequences)
    persistence = sum(
        item.role == "customer"
        and first_offer_response < item.sequence <= final_demand.sequence
        and (
            _PERSISTENCE_PATTERN.search(item.content) is not None
            or customer_requests_human_assistance(item.content)
        )
        for item in dialogue
    )
    return persistence >= 2


def _supports_unconfirmed_external_offer(
    ledger: ConversationLedger,
    *,
    require_search_evidence: bool,
    knowledge_search_dialogue_sequences: Collection[int],
) -> bool:
    dialogue = ledger.recent_dialogue_messages()
    for item in dialogue:
        if item.role != "customer" or not _contains_external_offer_claim(item.content):
            continue
        for next_item in dialogue:
            if (
                next_item.sequence <= item.sequence
                or next_item.role != "assistant"
                or not _is_uncertain_offer_response(next_item.content)
            ):
                continue
            if not require_search_evidence or any(
                item.sequence <= sequence < next_item.sequence
                for sequence in knowledge_search_dialogue_sequences
            ):
                return True
    return False


def _is_definite_unavailable_offer_response(content: str) -> bool:
    content = _normalize_typographic_punctuation(content)
    # Cautious nonconfirmation can precede a definitive material contradiction in
    # the same public response. The stronger evidence must win regardless of order.
    if _DEFINITE_CONTRADICTED_OFFER_PATTERN.search(content):
        return True
    if _UNCERTAIN_OFFER_RESPONSE_PATTERN.search(content):
        return False
    if any(pattern.search(content) for pattern in _DEFINITE_UNAVAILABLE_OFFER_PATTERNS):
        return True
    return bool(
        _DEFINITE_OFFER_OPERATION_REFUSAL_PATTERN.search(content)
        and not _CONDITIONAL_OFFER_OPERATION_PATTERN.search(content)
    )


def _contains_external_offer_claim(content: str) -> bool:
    normalized = _normalize_typographic_punctuation(content)
    return any(pattern.search(normalized) for pattern in _EXTERNAL_OFFER_CLAIM_PATTERNS)


def _is_uncertain_offer_response(content: str) -> bool:
    normalized = _normalize_typographic_punctuation(content)
    return bool(
        _UNCERTAIN_OFFER_RESPONSE_PATTERN.search(normalized)
        and _OFFER_REFERENCE_PATTERN.search(normalized)
    )


def _normalize_typographic_punctuation(content: str) -> str:
    return content.translate(_TYPOGRAPHIC_PUNCTUATION)


__all__ = [
    "customer_declines_human_assistance",
    "customer_requests_human_assistance",
    "transfer_reason_denial",
]
