"""Explicit resource predicates for the existing plan, not a policy interpreter.

The actor chooses the applicable procedure, resource and predicates. Infrastructure
checks those predicates against public observations. It does not infer requirements
from customer language or claim that the actor extracted every policy condition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, cast

from cayu import WorkEvidenceReference

from cayu_tau3_banking.ledger import ConversationLedger

_FIELDS = {
    "bank_account": {
        "account_id",
        "user_id",
        "account_class",
        "level",
        "status",
        "current_holdings",
        "opened_on",
    },
    "credit_card": {"account_id", "user_id", "card_type", "status", "opened_on", "card_last_four"},
    "debit_card": {"card_id", "user_id", "account_id", "status", "card_last_four"},
}
_OPERATORS = {"equal", "not_equal", "at_least", "at_most", "on_or_before", "on_or_after"}


class ConditionEvaluation(StrEnum):
    """Three-valued prerequisite result; missing data is not a negative fact."""

    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ResourceCheck:
    field: str
    operator: str
    value: str

    def evaluate(self, observed: object) -> ConditionEvaluation:
        if observed is None or (isinstance(observed, str) and observed == "UNKNOWN"):
            return ConditionEvaluation.UNKNOWN
        if self.field == "opened_on":
            if not isinstance(observed, date):
                return ConditionEvaluation.UNKNOWN
            expected = date.fromisoformat(self.value)
            if self.operator in {"on_or_before", "on_or_after"}:
                matched = (
                    observed <= expected
                    if self.operator == "on_or_before"
                    else observed >= expected
                )
            else:
                matched = observed == expected if self.operator == "equal" else observed != expected
            return ConditionEvaluation.SATISFIED if matched else ConditionEvaluation.UNSATISFIED
        if self.field == "current_holdings":
            if not isinstance(observed, Decimal) or not observed.is_finite():
                return ConditionEvaluation.UNKNOWN
            expected = Decimal(self.value)
            if self.operator in {"at_least", "at_most"}:
                matched = observed >= expected if self.operator == "at_least" else observed <= expected
            else:
                matched = observed == expected if self.operator == "equal" else observed != expected
            return ConditionEvaluation.SATISFIED if matched else ConditionEvaluation.UNSATISFIED
        actual = str(observed)
        matched = actual == self.value if self.operator == "equal" else actual != self.value
        return ConditionEvaluation.SATISFIED if matched else ConditionEvaluation.UNSATISFIED

    def matches(self, observed: object) -> bool:
        """Compatibility boolean: only an established true result matches."""

        return self.evaluate(observed) is ConditionEvaluation.SATISFIED


@dataclass(frozen=True)
class ResourceCondition:
    resource_kind: str
    resource_id: str
    observed_in: str
    procedure: WorkEvidenceReference
    checks: tuple[ResourceCheck, ...]
    for_tools: tuple[str, ...]
    customer_message_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "resource_kind": self.resource_kind,
            "resource_id": self.resource_id,
            "observed_in": self.observed_in,
            "procedure": self.procedure.model_dump(
                mode="json", exclude_none=True, exclude={"available"}
            ),
            "checks": [asdict(check) for check in self.checks],
            "for_tools": list(self.for_tools),
            **(
                {"customer_message_id": self.customer_message_id}
                if self.customer_message_id
                else {}
            ),
        }

    def evaluate(self, ledger: ConversationLedger) -> ConditionEvaluation:
        """Evaluate predicates while retaining unknown versus unsatisfied."""

        observation = ledger.resource_evidence(
            self.observed_in, self.resource_kind, self.resource_id
        )
        if observation is None:
            return ConditionEvaluation.UNKNOWN
        results = tuple(check.evaluate(getattr(observation, check.field)) for check in self.checks)
        if ConditionEvaluation.UNSATISFIED in results:
            return ConditionEvaluation.UNSATISFIED
        if ConditionEvaluation.UNKNOWN in results:
            return ConditionEvaluation.UNKNOWN
        return ConditionEvaluation.SATISFIED

    def gap(self, ledger: ConversationLedger) -> str | None:
        observation = ledger.resource_evidence(
            self.observed_in, self.resource_kind, self.resource_id
        )
        if observation is None:
            return "resource_observation_stale_or_unavailable"
        if self.customer_message_id is not None and (
            ledger.customer_message_sequence(self.customer_message_id)
            != ledger.last_customer_message_sequence
        ):
            return "customer_context_changed"
        evaluation = self.evaluate(ledger)
        if evaluation is ConditionEvaluation.UNKNOWN:
            return "recorded_condition_unknown"
        if evaluation is ConditionEvaluation.UNSATISFIED:
            return "recorded_condition_not_satisfied"
        return None


def parse_condition(raw: object) -> ResourceCondition | None:
    if raw is None:
        return None
    required = {"resource_kind", "resource_id", "observed_in", "procedure", "checks", "for_tools"}
    if (
        type(raw) is not dict
        or not required <= raw.keys()
        or raw.keys() - required - {"customer_message_id"}
    ):
        raise ValueError(
            "condition requires resource_kind, resource_id, observed_in, procedure and checks"
        )
    raw = cast(dict[str, Any], raw)
    kind = raw["resource_kind"]
    if not isinstance(kind, str) or kind not in _FIELDS:
        raise ValueError("unsupported condition resource_kind")
    for field in ("resource_id", "observed_in", "customer_message_id"):
        if field in raw and (
            not isinstance(raw[field], str) or not raw[field].strip() or len(raw[field]) > 200
        ):
            raise ValueError(f"invalid condition {field}")
    procedure = WorkEvidenceReference.model_validate(raw["procedure"])
    if procedure.kind != "knowledge_revision" or not procedure.version or not procedure.available:
        raise ValueError("condition procedure must be an available knowledge revision")
    entries = raw["checks"]
    if type(entries) is not list or not 1 <= len(entries) <= 6:
        raise ValueError("condition checks must have one to six predicates")
    checks = []
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"field", "operator", "value"}:
            raise ValueError("condition predicate requires field, operator and value")
        entry = cast(dict[str, Any], entry)
        field, operator, value = entry["field"], entry["operator"], entry["value"]
        if (
            not isinstance(field, str)
            or field not in _FIELDS[kind]
            or not isinstance(operator, str)
            or operator not in _OPERATORS
        ):
            raise ValueError("invalid condition field or operator")
        if not isinstance(value, str) or not value or len(value) > 200:
            raise ValueError("condition values must be bounded strings")
        if operator.startswith("on_or_") and field != "opened_on":
            raise ValueError("date comparison requires opened_on and an ISO calendar date")
        if field == "opened_on":
            try:
                canonical = date.fromisoformat(value).isoformat()
            except ValueError as exc:
                raise ValueError(
                    "opened_on requires a valid calendar date in YYYY-MM-DD format"
                ) from exc
            if canonical != value:
                raise ValueError("opened_on requires a valid calendar date in YYYY-MM-DD format")
        if operator in {"at_least", "at_most"} and field != "current_holdings":
            raise ValueError("numeric comparison requires current_holdings")
        if field == "current_holdings":
            try:
                if not Decimal(value).is_finite():
                    raise ValueError("current_holdings requires a finite decimal")
            except InvalidOperation as exc:
                raise ValueError("condition value is not a decimal") from exc
        checks.append(ResourceCheck(field, operator, value))
    if len(set(checks)) != len(checks):
        raise ValueError("duplicate condition predicates")
    tools = raw["for_tools"]
    if (
        type(tools) is not list
        or not 1 <= len(tools) <= 8
        or any(not isinstance(tool, str) or not tool.strip() or len(tool) > 200 for tool in tools)
        or len(set(tools)) != len(tools)
    ):
        raise ValueError("for_tools requires one to eight distinct exact tool names")
    return ResourceCondition(
        kind,
        raw["resource_id"],
        raw["observed_in"],
        procedure,
        tuple(checks),
        tuple(tools),
        raw.get("customer_message_id"),
    )


def condition_schema(evidence_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "description": (
            "A resource-specific policy prerequisite. Copy the exact resource id and successful account-read call id. "
            "Cite its procedure and express each applicable field condition separately (active is not old enough). "
            "Use customer_message_id only when applicability depends on current customer preferences/consent; "
            "new customer input then requires reassessment. Predicates check observations, not your policy interpretation. "
            "Resolved requires every predicate to hold on a still-current observation. No model-generated digest."
            " for_tools lists only the exact operations requiring this check, not preparatory reads or repairs."
            " A pending executable action may stay ready with its conditions attached; do not mark the action"
            " resolved merely to permit execution. Admission checks its current predicates and separate"
            " dependencies. Only actual results can establish that the action was performed."
            " Debit cards use resource_kind=debit_card, not credit_card. Their parent account"
            " prerequisites can bind to card_id operations only through a current successful"
            " get_debit_cards_by_account_id_7823 read proving the same account and owner."
        ),
        "properties": {
            "resource_kind": {"type": "string", "enum": sorted(_FIELDS)},
            "resource_id": {"type": "string", "maxLength": 200},
            "observed_in": {"type": "string", "maxLength": 200},
            "procedure": evidence_schema,
            "customer_message_id": {"type": "string", "maxLength": 200},
            "for_tools": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1, "maxLength": 200},
            },
            "checks": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                            "enum": sorted(set().union(*_FIELDS.values())),
                            "description": "Fields by resource_kind: "
                            + "; ".join(
                                f"{kind}: {', '.join(sorted(fields))}"
                                for kind, fields in _FIELDS.items()
                            )
                            + ". account_id identifies a bank/credit account; on a debit card it "
                            "identifies the parent account, while card_id identifies the card.",
                        },
                        "operator": {"type": "string", "enum": sorted(_OPERATORS)},
                        "value": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 200,
                            "description": (
                                "For opened_on use an unambiguous calendar date in YYYY-MM-DD "
                                "format, even when the bank displays another format. For "
                                "current_holdings use a finite decimal string without currency "
                                "symbols or grouping separators. Other fields match exact observed values."
                            ),
                        },
                    },
                    "required": ["field", "operator", "value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "resource_kind",
            "resource_id",
            "observed_in",
            "procedure",
            "checks",
            "for_tools",
        ],
        "additionalProperties": False,
    }
