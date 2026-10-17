"""Capabilities and admission for deterministic savings-interest actions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from cayu import ToolCallPart

from cayu_tau3_banking.bridge import ExternalToolAdmission, InternalToolResult
from cayu_tau3_banking.ledger import ConversationLedger

DISCOVERABLE_AGENT_CALL_TOOL = "call_discoverable_agent_tool"
SAVINGS_CREDIT_TOOL = "apply_savings_account_credit_6831"
SAVINGS_REPORT_TOOL = "submit_interest_discrepancy_report_7294"


@dataclass(frozen=True)
class SavingsInterestActionAuthorization:
    """Exact environment actions authorized by one deterministic interest audit."""

    account_id: str
    user_id: str
    credit_amount: Decimal | None
    actual_apy: Decimal
    amount_difference: Decimal
    expected_apy: Decimal

    def credit_arguments(self) -> dict[str, object] | None:
        if self.credit_amount is None:
            return None
        return {
            "account_id": self.account_id,
            "amount": self.credit_amount,
            "credit_type": "interest_correction",
        }

    def report_arguments(self) -> dict[str, object]:
        return {
            "account_id": self.account_id,
            "user_id": self.user_id,
            "expected_apy": self.expected_apy,
            "actual_apy": self.actual_apy,
            "amount_difference": self.amount_difference,
        }


class SavingsInterestActionGate:
    """Persist and enforce exact write capabilities produced by savings audits."""

    def __init__(self, ledger: ConversationLedger) -> None:
        if not isinstance(ledger, ConversationLedger):
            raise TypeError("ledger must be a ConversationLedger.")
        self._ledger = ledger
        self._authorizations: dict[str, SavingsInterestActionAuthorization] = {}

    def observe_audit(self, result: InternalToolResult, *, expected_account_id: str) -> None:
        """Replace one account's capability from a successful current audit."""

        if not isinstance(expected_account_id, str) or not expected_account_id.strip():
            raise ValueError("expected_account_id must be nonblank text.")
        authorization = _authorization_from_audit(result)
        normalized_account_id = expected_account_id.strip()
        if authorization.account_id != normalized_account_id:
            raise RuntimeError("Savings audit authorized a different account.")
        self._authorizations[normalized_account_id] = authorization

    def render_context(self) -> str | None:
        """Render bounded copy-ready actions for a subsequent customer turn."""

        if not self._authorizations:
            return None
        actions = []
        for account_id, authorization in sorted(self._authorizations.items()):
            credit = authorization.credit_arguments()
            actions.append(
                {
                    "account_id": account_id,
                    "credit_call": (
                        None
                        if credit is None
                        else _discoverable_action_payload(SAVINGS_CREDIT_TOOL, credit)
                    ),
                    "report_call": _discoverable_action_payload(
                        SAVINGS_REPORT_TOOL,
                        authorization.report_arguments(),
                    ),
                }
            )
        return (
            "<savings-interest-action-authorizations>\n"
            "Application-owned actions from successful deterministic audits. Copy each outer "
            "agent_tool_name and nested arguments string exactly. Apply an authorized credit "
            "only after customer approval, wait for its successful result, and then submit "
            "that account's report. The admission boundary rejects altered calculations, "
            "unaudited accounts, duplicates, and reports sent before their required credit.\n"
            + json.dumps(
                {"schema_version": 1, "actions": actions},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n</savings-interest-action-authorizations>"
        )

    def review(self, calls: Sequence[ToolCallPart]) -> tuple[ExternalToolAdmission, ...]:
        """Admit only exact, ordered, nonduplicate audit-derived writes."""

        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
            raise TypeError("calls must be a sequence.")
        decisions: list[ExternalToolAdmission] = []
        proposed_savings_actions: set[tuple[str, str]] = set()
        for call in calls:
            if not isinstance(call, ToolCallPart):
                raise TypeError("calls must contain ToolCallPart instances.")
            reason = self._denial_reason(call)
            action_key = _savings_action_key(call)
            if reason is None and action_key is not None and action_key in proposed_savings_actions:
                reason = "this exact audited savings action appears more than once in the batch"
            if action_key is not None:
                proposed_savings_actions.add(action_key)
            decisions.append(
                ExternalToolAdmission(
                    tool_call_id=call.tool_call_id,
                    allowed=reason is None,
                    reason=reason,
                    diagnostic_code=None if reason is None else "savings_interest_audit",
                )
            )
        return tuple(decisions)

    def _denial_reason(self, call: ToolCallPart) -> str | None:
        if call.tool_name != DISCOVERABLE_AGENT_CALL_TOOL:
            return None
        agent_tool_name = call.arguments.get("agent_tool_name")
        if agent_tool_name not in {SAVINGS_CREDIT_TOOL, SAVINGS_REPORT_TOOL}:
            return None
        nested = _nested_discoverable_arguments(call.arguments.get("arguments"))
        if nested is None:
            return "the audited savings action requires one valid JSON argument object"
        account_id = nested.get("account_id")
        if not isinstance(account_id, str) or not account_id.strip():
            return "the audited savings action requires a nonblank account_id"
        authorization = self._authorizations.get(account_id.strip())
        if authorization is None:
            return (
                "no successful current deterministic savings-interest audit authorizes this "
                "account; retrieve current account data and complete the Cayu audit first"
            )
        expected = (
            authorization.credit_arguments()
            if agent_tool_name == SAVINGS_CREDIT_TOOL
            else authorization.report_arguments()
        )
        if expected is None:
            return "the current deterministic audit does not authorize a positive credit"
        if not _arguments_match(nested, expected):
            return (
                "the proposed savings action differs from the deterministic audit; copy this "
                "exact wrapper payload: "
                + json.dumps(
                    _discoverable_action_payload(agent_tool_name, expected),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        if _successful_action_exists(
            self._ledger,
            agent_tool_name=agent_tool_name,
            expected=expected,
        ):
            return "this exact audited savings action already succeeded; do not repeat it"
        credit = authorization.credit_arguments()
        if (
            agent_tool_name == SAVINGS_REPORT_TOOL
            and credit is not None
            and not _successful_action_exists(
                self._ledger,
                agent_tool_name=SAVINGS_CREDIT_TOOL,
                expected=credit,
            )
        ):
            return (
                "submit the discrepancy report only after this account's exact audited credit "
                "has succeeded"
            )
        return None


def _authorization_from_audit(
    result: InternalToolResult,
) -> SavingsInterestActionAuthorization:
    if result.is_error:
        raise ValueError("An error result cannot authorize savings-interest actions.")
    try:
        payload = json.loads(result.content.rsplit("\n", 1)[-1])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Successful savings audit did not contain valid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("Successful savings audit payload must be an object.")
    account = payload.get("account")
    report = payload.get("report_arguments")
    credit = payload.get("credit_arguments")
    if not isinstance(account, Mapping) or not isinstance(report, Mapping):
        raise RuntimeError("Successful savings audit omitted action authorization fields.")
    account_id = account.get("account_id")
    user_id = account.get("user_id")
    if not isinstance(account_id, str) or not account_id.strip():
        raise RuntimeError("Successful savings audit account_id is invalid.")
    if not isinstance(user_id, str) or not user_id.strip():
        raise RuntimeError("Successful savings audit user_id is invalid.")
    if set(report) != {
        "account_id",
        "actual_apy",
        "amount_difference",
        "expected_apy",
        "user_id",
    } or (report.get("account_id"), report.get("user_id")) != (account_id, user_id):
        raise RuntimeError("Successful savings audit report identity is inconsistent.")
    actual_apy = _finite_decimal(report.get("actual_apy"))
    amount_difference = _finite_decimal(report.get("amount_difference"))
    expected_apy = _finite_decimal(report.get("expected_apy"))
    if (
        actual_apy is None
        or amount_difference is None
        or expected_apy is None
        or min(actual_apy, amount_difference, expected_apy) < 0
    ):
        raise RuntimeError("Successful savings audit report numbers are invalid.")
    credit_amount: Decimal | None = None
    if credit is not None:
        if not isinstance(credit, Mapping) or set(credit) != {
            "account_id",
            "amount",
            "credit_type",
        }:
            raise RuntimeError("Successful savings audit credit capability is invalid.")
        credit_amount = _finite_decimal(credit.get("amount"))
        if (
            credit.get("account_id") != account_id
            or credit.get("credit_type") != "interest_correction"
            or credit_amount is None
            or credit_amount <= 0
            or credit_amount != amount_difference
        ):
            raise RuntimeError("Successful savings audit credit details are inconsistent.")
    return SavingsInterestActionAuthorization(
        account_id=account_id,
        user_id=user_id,
        credit_amount=credit_amount,
        actual_apy=actual_apy,
        amount_difference=amount_difference,
        expected_apy=expected_apy,
    )


def _discoverable_action_payload(
    agent_tool_name: str,
    arguments: Mapping[str, object],
) -> dict[str, str]:
    return {
        "agent_tool_name": agent_tool_name,
        "arguments": _render_action_arguments(agent_tool_name, arguments),
    }


def _render_action_arguments(
    agent_tool_name: str,
    arguments: Mapping[str, object],
) -> str:
    rendered: list[str] = []
    for key, value in arguments.items():
        if isinstance(value, Decimal):
            if key in {"amount", "amount_difference"}:
                encoded = f"{value:.2f}"
            elif agent_tool_name == SAVINGS_REPORT_TOOL:
                encoded = _decimal_with_fraction(value)
            else:
                encoded = format(value, "f")
        else:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        rendered.append(f"{json.dumps(key)}: {encoded}")
    return "{" + ", ".join(rendered) + "}"


def _decimal_with_fraction(value: Decimal) -> str:
    normalized = format(value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized if "." in normalized else normalized + ".0"


def _nested_discoverable_arguments(value: object) -> dict[str, object] | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if type(parsed) is dict else None


def _savings_action_key(call: ToolCallPart) -> tuple[str, str] | None:
    if call.tool_name != DISCOVERABLE_AGENT_CALL_TOOL:
        return None
    agent_tool_name = call.arguments.get("agent_tool_name")
    if agent_tool_name not in {SAVINGS_CREDIT_TOOL, SAVINGS_REPORT_TOOL}:
        return None
    nested = _nested_discoverable_arguments(call.arguments.get("arguments"))
    if nested is None:
        return None
    try:
        encoded = json.dumps(
            nested,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return None
    return agent_tool_name, encoded


def _arguments_match(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    if set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, Decimal):
            parsed = _finite_decimal(actual_value)
            if parsed is None or parsed != expected_value:
                return False
        elif actual_value != expected_value:
            return False
    return True


def _successful_action_exists(
    ledger: ConversationLedger,
    *,
    agent_tool_name: str,
    expected: Mapping[str, object],
) -> bool:
    for wrapper_arguments in ledger.successful_call_arguments(DISCOVERABLE_AGENT_CALL_TOOL):
        if wrapper_arguments.get("agent_tool_name") != agent_tool_name:
            continue
        nested = _nested_discoverable_arguments(wrapper_arguments.get("arguments"))
        if nested is not None and _arguments_match(nested, expected):
            return True
    return False


def _finite_decimal(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, str | int | float | Decimal):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


__all__ = [
    "SavingsInterestActionAuthorization",
    "SavingsInterestActionGate",
]
