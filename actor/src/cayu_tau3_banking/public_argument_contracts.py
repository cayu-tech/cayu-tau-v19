"""Audited public parameter constraints, not inference about customer intent.

Constraints are application configuration translated from exact public revisions.
They reject invalid representations; they never select a product, rewrite a call,
authorize an effect, or claim semantic correctness for a schema-valid argument.
No task initialization, reference actions, evaluator state or model is consulted.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cayu import KnowledgeEntry, ToolCallPart
from jsonschema import Draft202012Validator
from referencing import Registry

from cayu_tau3_banking.argument_contract import SOURCE_BINDING_RECOVERY
from cayu_tau3_banking.bridge import ExternalToolAdmission


@dataclass(frozen=True)
class PublicArgumentContract:
    tool_name: str
    entry_id: str
    revision: int
    text_sha256: str
    quote: str
    schema_json: str


# This is a format rule for ALL personal savings products, not a product alias
# table. The condition deliberately excludes checking and business accounts.
SAVINGS_NAME_CONTRACT = PublicArgumentContract(
    tool_name="open_bank_account_4821",
    entry_id="doc_bank_accounts_bank_accounts_(general)_002",
    revision=1,
    text_sha256="4d717088104f04313b990693460392b5c52c7fadac1a3c0cf903d351f4bdd7c5",
    quote="account_class: the full official account name ending with 'Account' exactly as confirmed with the customer.",
    schema_json=json.dumps(
        {
            "type": "object",
            "required": ["account_type", "account_class"],
            "if": {
                "properties": {"account_type": {"const": "savings"}},
                "required": ["account_type"],
            },
            "then": {
                "properties": {
                    "account_class": {"type": "string", "pattern": r"^\S[^\r\n]* Account(?![\s\S])"}
                }
            },
        }
    ),
)


class PublicArgumentContracts:
    """Compile once against pinned public corpus entries, review without I/O.

    Missing sources are allowed for other/synthetic corpora. An existing source
    whose revision/body changed is a configuration error, never a silent waiver.
    The source quote is shown on rejection; loading a validator does not fabricate
    a full-document recall receipt or imply the actor already read the source.
    """

    def __init__(
        self,
        entries: Mapping[str, KnowledgeEntry],
        contracts: Sequence[PublicArgumentContract] = (SAVINGS_NAME_CONTRACT,),
    ) -> None:
        compiled: dict[str, list[tuple[PublicArgumentContract, Draft202012Validator]]] = {}
        for contract in contracts:
            entry = entries.get(contract.entry_id)
            if entry is None:
                continue
            if (
                entry.revision != contract.revision
                or hashlib.sha256(entry.text.encode()).hexdigest() != contract.text_sha256
                or not contract.quote
                or contract.quote not in entry.text
            ):
                raise ValueError(f"Public argument contract source changed: {contract.entry_id}")
            schema = json.loads(contract.schema_json)
            Draft202012Validator.check_schema(schema)
            compiled.setdefault(contract.tool_name, []).append(
                (contract, Draft202012Validator(schema, registry=Registry()))
            )
        self._compiled = compiled

    def review(self, calls: Sequence[ToolCallPart]) -> tuple[ExternalToolAdmission, ...]:
        return tuple(self._review(call) for call in calls)

    def _review(self, call: ToolCallPart) -> ExternalToolAdmission:
        name = call.tool_name
        arguments: Any = call.arguments
        if name == "call_discoverable_agent_tool":
            name = arguments.get("agent_tool_name")
            if isinstance(name, str) and name in self._compiled:
                try:
                    arguments = json.loads(
                        arguments.get("arguments", "{}"),
                        object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant,
                        parse_float=_finite_float,
                    )
                    if not isinstance(arguments, dict):
                        raise ValueError("Expected an object")
                except (ValueError, TypeError, RecursionError):
                    return self._denied(
                        call,
                        "The discoverable tool arguments must be a JSON object with unique keys and finite numbers. No call was sent. Repair the JSON; do not change the authorized operation.",
                    )
        if isinstance(name, str):
            for contract, validator in self._compiled.get(name, ()):
                error = next(validator.iter_errors(arguments), None)
                if error is not None:
                    # Do not echo arbitrary arguments/customer data in diagnostics.
                    detail = {
                        "tool": name,
                        "field_path": list(error.absolute_path),
                        "rule": error.validator,
                        "constraint": error.validator_value,
                        "source": {
                            "entry_id": contract.entry_id,
                            "revision": contract.revision,
                            "sha256": contract.text_sha256,
                            "quote": contract.quote,
                        },
                    }
                    return self._denied(
                        call,
                        "Public parameter contract rejected this call before dispatch. "
                        "Nothing was executed. Resolve the selected product's API representation "
                        "from its public sources; a customer's confirmation of a display label "
                        "does not redefine the parameter. Do not blindly strip or replace text. "
                        "If the mapping is ambiguous, investigate or clarify without creating a resource. "
                        "Format validity alone does not establish product identity, eligibility or consent. "
                        + json.dumps(detail, ensure_ascii=False)
                        + "\n"
                        + SOURCE_BINDING_RECOVERY,
                    )
        return ExternalToolAdmission(tool_call_id=call.tool_call_id, allowed=True)

    @staticmethod
    def _denied(call: ToolCallPart, reason: str) -> ExternalToolAdmission:
        return ExternalToolAdmission(
            tool_call_id=call.tool_call_id,
            allowed=False,
            reason=reason,
            diagnostic_code="public_argument_contract",
        )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate argument key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite argument")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite argument")
    return number
