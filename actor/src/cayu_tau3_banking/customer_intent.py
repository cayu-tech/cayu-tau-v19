"""Deterministic classifiers for customer directives in public dialogue text."""

from __future__ import annotations

import re

_HUMAN_DEMAND_PATTERN = re.compile(
    r"\b(?:transfer|connect|escalate)\b.{0,50}\b(?:human|person|representative|agent|"
    r"supervisor|someone)\b|\b(?:speak|talk)\b.{0,30}\b(?:human|person|representative|agent|"
    r"supervisor|someone)\b|\b(?:human|live agent|representative|supervisor)\b.{0,40}"
    r"\b(?:now|please|instead|requested|want|need)\b|\b(?:get|put)\b.{0,30}"
    r"\b(?:human|person|representative|agent|supervisor|someone)\b|"
    r"\bask(?:ing)?\b.{0,40}\b(?:human|live agent|person|representative|supervisor)\b",
    re.IGNORECASE,
)
_ESCALATION_REQUEST_PATTERN = re.compile(
    r"\b(?:want|need|ask)(?:\s+[a-z']+){0,3}\s+to\s+escalate\b|"
    r"\bplease\s+escalate\b|\b(?:can|could|would|will)\s+you\s+escalate\b|"
    r"\bescalate\s+(?:this|it|my|the)\b",
    re.IGNORECASE,
)
_HUMAN_TRANSFER_REFUSAL_PATTERN = re.compile(
    r"\b(?:do\s+not|don['\u2019]t|stop|not\s+yet|wait)\b[^.!?;\n]{0,24}\b(?:"
    r"transfer(?:ring)?\s+(?:(?:me|us)(?:\s+to\s+(?:a\s+)?(?:human|person|"
    r"representative|agent|supervisor|someone))?|this|it|(?:my|our|the)\s+(?:case|request)|"
    r"to\s+(?:a\s+)?(?:human|person|representative|agent|supervisor|someone))|"
    r"connect(?:ing)?\s+(?:me|us|(?:to|with)\s+(?:a\s+)?"
    r"(?:human|person|representative|agent|supervisor|someone))|"
    r"escalat(?:e|ing)\s+(?:me|us|this|it|(?:my|our|the)\s+(?:case|request))"
    r")\b",
    re.IGNORECASE,
)
_TYPOGRAPHIC_PUNCTUATION = str.maketrans({"\u2018": "'", "\u2019": "'"})


def customer_requests_human_assistance(content: str) -> bool:
    """Return whether the latest directive in public text asks for a human."""

    if not isinstance(content, str):
        return False
    normalized = content.translate(_TYPOGRAPHIC_PUNCTUATION)
    demand_end = max(
        (
            match.end()
            for pattern in (_HUMAN_DEMAND_PATTERN, _ESCALATION_REQUEST_PATTERN)
            for match in pattern.finditer(normalized)
        ),
        default=-1,
    )
    refusal_end = _latest_match_end(_HUMAN_TRANSFER_REFUSAL_PATTERN, normalized)
    return demand_end >= 0 and demand_end > refusal_end


def customer_declines_human_assistance(content: str) -> bool:
    """Return whether the latest transfer directive explicitly refuses a handoff."""

    if not isinstance(content, str):
        return False
    normalized = content.translate(_TYPOGRAPHIC_PUNCTUATION)
    refusal_end = _latest_match_end(_HUMAN_TRANSFER_REFUSAL_PATTERN, normalized)
    demand_end = max(
        (
            match.end()
            for pattern in (_HUMAN_DEMAND_PATTERN, _ESCALATION_REQUEST_PATTERN)
            for match in pattern.finditer(normalized)
        ),
        default=-1,
    )
    return refusal_end >= 0 and refusal_end >= demand_end


def _latest_match_end(pattern: re.Pattern[str], content: str) -> int:
    return max((match.end() for match in pattern.finditer(content)), default=-1)


__all__ = [
    "customer_declines_human_assistance",
    "customer_requests_human_assistance",
]
