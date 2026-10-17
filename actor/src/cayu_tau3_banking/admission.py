"""Deterministic admission for externally executed tau banking tools.

This gate does not execute tools or replace the benchmark policy. It rejects only
protocol mistakes that are provable from successful public observations, then lets
the Cayu-backed model correct the call before tau sees it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from cayu import ToolCallPart
from tau2.data_model.message import AssistantMessage, Message

from cayu_tau3_banking.bridge import ExternalToolAdmission
from cayu_tau3_banking.ledger import ConversationLedger
from cayu_tau3_banking.transfer_reason import current_offer_capability_gap, transfer_reason_denial

_NONREPEATABLE_EFFECTS = frozenset(
    {
        "change_user_email",
        "give_discoverable_user_tool",
        "log_verification",
        "transfer_to_human_agents",
        "unlock_discoverable_agent_tool",
    }
)
_IDENTITY_LOOKUP_TOOLS = (
    "get_user_information_by_email",
    "get_user_information_by_id",
    "get_user_information_by_name",
)
_PROTECTED_USER_TOOLS = frozenset(
    {
        "change_user_email",
        "get_credit_card_accounts_by_user",
        "get_credit_card_transactions_by_user",
        "get_referrals_by_user",
    }
)
_VERIFICATION_FIELDS = ("address", "date_of_birth", "email", "phone_number")
_VERIFICATION_IDENTITY_ARGUMENTS = ("name", "user_id", *_VERIFICATION_FIELDS)
_KNOWLEDGE_IDENTIFIER_PATTERN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_PUBLIC_TOOL_IDENTIFIER_PATTERN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){2,}_[0-9]{4}\b")
_RECOVERY_BYPASS_CODE_PATTERN = re.compile(
    r"\b(?=[A-Z0-9]{12,64}\b)(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]+\b"
)
_LOOKUP_RECORD_BOUNDARY_PATTERN = re.compile(r"(?m)^\d+\. Record ID:")
_MAX_KNOWLEDGE_SEARCH_RECEIPTS = 32
_MAX_PUBLIC_TOOL_IDENTIFIERS = 96

_DUPLICATE_EFFECT_REASON = "the same side effect was already confirmed successful; do not repeat it"
_SAME_BATCH_DUPLICATE_REASON = (
    "the same non-repeatable side effect appears more than once in this batch"
)
_MULTIPLE_USER_TOOL_DELIVERY_REASON = (
    "give only one customer-side action at a time, explain its exact callable name and "
    "arguments, and wait for its result before giving the next action"
)
_REPEATED_USER_TOOL_DELIVERY_REASON = (
    "this customer-side tool was already given successfully; reuse the available tool name "
    "with the next exact arguments instead of giving it again"
)
_LOCKED_DISCOVERABLE_TOOL_REASON = (
    "the requested agent-discoverable tool has not been successfully unlocked"
)
_MISSING_VERIFICATION_REASON = (
    "customer-specific data access requires a successful verification log for this user"
)
_REPEATED_VERIFICATION_REASON = (
    "verification was already logged successfully in this conversation; do not create another "
    "record"
)
_UNPROVEN_VERIFICATION_REASON = (
    "verification arguments need a matching identity lookup from authoritative records; customer "
    "confirmation must be assessed by the acting agent under the public policy"
)
_UNPROVEN_VERIFICATION_TIME_REASON = (
    "time_verified must match a successful get_current_time result from this conversation"
)
_UNPROVEN_RECOVERY_BYPASS_REASON = (
    "an account-recovery bypass must exactly match a Cayu-retrieved recovery code, be "
    "supplied by the customer, and populate every documented verification identity field"
)
_UNEXPOSED_DISCOVERABLE_TOOL_REASON = (
    "the exact discoverable tool name has not appeared in Cayu-retrieved public knowledge"
)
_REDUNDANT_READ_REASON = (
    "this exact read already succeeded in the current customer turn and no later mutation "
    "could have changed its result; reuse the observation ledger"
)
_SAME_BATCH_REDUNDANT_READ_REASON = "the same exact read appears more than once in this batch"


class BankingToolAdmissionGate:
    """Fail closed on duplicate effects and invalid discoverable-tool sequencing."""

    def __init__(
        self,
        ledger: ConversationLedger,
        *,
        require_knowledge_evidence: bool = True,
        knowledge_search_dialogue_sequences: Sequence[int] = (),
        public_tool_identifiers: Sequence[str] = (),
        authoritative_only: bool = False,
    ) -> None:
        if not isinstance(ledger, ConversationLedger):
            raise TypeError("ledger must be a ConversationLedger.")
        if type(require_knowledge_evidence) is not bool:
            raise TypeError("require_knowledge_evidence must be a boolean.")
        if not isinstance(knowledge_search_dialogue_sequences, Sequence) or isinstance(
            knowledge_search_dialogue_sequences, (str, bytes)
        ):
            raise TypeError("knowledge search receipts must be a sequence of integers.")
        if not isinstance(public_tool_identifiers, Sequence) or isinstance(
            public_tool_identifiers, (str, bytes)
        ):
            raise TypeError("public tool identifiers must be a sequence of strings.")
        normalized_search_sequences = tuple(knowledge_search_dialogue_sequences)
        if (
            len(normalized_search_sequences) > _MAX_KNOWLEDGE_SEARCH_RECEIPTS
            or any(
                type(sequence) is not int
                or not 1 <= sequence <= ledger.last_dialogue_message_sequence
                for sequence in normalized_search_sequences
            )
            or tuple(sorted(set(normalized_search_sequences))) != normalized_search_sequences
        ):
            raise ValueError(
                "knowledge search receipts must be bounded, ordered, unique dialogue sequences."
            )
        normalized_public_tool_identifiers = tuple(public_tool_identifiers)
        if (
            len(normalized_public_tool_identifiers) > _MAX_PUBLIC_TOOL_IDENTIFIERS
            or any(
                not isinstance(identifier, str)
                or _PUBLIC_TOOL_IDENTIFIER_PATTERN.fullmatch(identifier) is None
                for identifier in normalized_public_tool_identifiers
            )
            or tuple(sorted(set(normalized_public_tool_identifiers)))
            != normalized_public_tool_identifiers
        ):
            raise ValueError(
                "public tool identifiers must be bounded, sorted, unique tool-shaped names."
            )
        self.ledger = ledger
        if type(authoritative_only) is not bool:
            raise TypeError("authoritative_only must be a boolean.")
        self.authoritative_only = authoritative_only
        self.require_knowledge_evidence = require_knowledge_evidence
        self.reviewed_calls = 0
        self.denied_calls = 0
        self._denials_by_reason: Counter[str] = Counter()
        self._knowledge_identifiers: set[str] = set(normalized_public_tool_identifiers)
        self._public_tool_identifiers: set[str] = set(normalized_public_tool_identifiers)
        self._recovery_bypass_codes: set[str] = set()
        self._knowledge_search_dialogue_sequences = list(normalized_search_sequences)

    @classmethod
    def from_history(
        cls,
        messages: list[Message] | None,
        *,
        ledger: ConversationLedger,
        require_knowledge_evidence: bool = True,
    ) -> BankingToolAdmissionGate:
        """Restore content-free Cayu search receipts from trusted assistant state."""

        return cls(
            ledger,
            require_knowledge_evidence=require_knowledge_evidence,
            knowledge_search_dialogue_sequences=_restored_knowledge_search_sequences(
                messages,
                last_dialogue_sequence=ledger.last_dialogue_message_sequence,
            ),
            public_tool_identifiers=_restored_public_tool_identifiers(messages),
        )

    def observe_knowledge_text(self, value: str) -> None:
        """Record identifiers actually delivered from public Cayu knowledge."""

        if not isinstance(value, str) or not value.strip():
            raise ValueError("Knowledge evidence must be nonblank text.")
        self._knowledge_identifiers.update(_KNOWLEDGE_IDENTIFIER_PATTERN.findall(value))
        self._public_tool_identifiers.update(_PUBLIC_TOOL_IDENTIFIER_PATTERN.findall(value))
        normalized = value.casefold()
        if "account recovery bypass code" in normalized and "log_verification" in normalized:
            self._recovery_bypass_codes.update(_RECOVERY_BYPASS_CODE_PATTERN.findall(value))

    def observe_knowledge_search(self) -> None:
        """Record one successful Cayu retrieval at the current dialogue boundary."""

        sequence = self.ledger.last_dialogue_message_sequence
        if sequence < 1:
            return
        if self._knowledge_search_dialogue_sequences[-1:] == [sequence]:
            return
        self._knowledge_search_dialogue_sequences.append(sequence)
        del self._knowledge_search_dialogue_sequences[:-_MAX_KNOWLEDGE_SEARCH_RECEIPTS]

    def review(self, calls: Sequence[ToolCallPart]) -> tuple[ExternalToolAdmission, ...]:
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
            raise TypeError("calls must be a sequence.")
        proposed_effects: set[tuple[str, str]] = set()
        proposed_reads: set[tuple[str, str]] = set()
        user_tool_delivery_seen = False
        decisions: list[ExternalToolAdmission] = []
        for call in calls:
            if not isinstance(call, ToolCallPart):
                raise TypeError("calls must contain ToolCallPart instances.")
            self.reviewed_calls += 1
            reason = self._denial_reason(call, proposed_effects, proposed_reads)
            if call.tool_name == "give_discoverable_user_tool":
                if user_tool_delivery_seen and reason is None:
                    reason = _MULTIPLE_USER_TOOL_DELIVERY_REASON
                user_tool_delivery_seen = True
            allowed = reason is None
            if reason is not None:
                self.denied_calls += 1
                self._denials_by_reason[reason] += 1
            decisions.append(
                ExternalToolAdmission(
                    tool_call_id=call.tool_call_id,
                    allowed=allowed,
                    reason=reason,
                    diagnostic_code=(
                        None
                        if reason is None
                        else (
                            "banking_discoverable_tool_locked"
                            if reason == _LOCKED_DISCOVERABLE_TOOL_REASON
                            else "banking_admission"
                        )
                    ),
                )
            )
        return tuple(decisions)

    def has_verified_offer_capability_gap(self) -> bool:
        """Read-only policy exception, backed by observed dialogue and retrieval.

        A model-proposed reason alone cannot waive the within-capability transfer
        delay. The ordinary transfer reason/consent gate still reviews the actual call.
        """
        return current_offer_capability_gap(
            ledger=self.ledger,
            knowledge_identifiers=self._knowledge_identifiers,
            knowledge_search_dialogue_sequences=self._knowledge_search_dialogue_sequences,
        )

    def render_offer_transfer_review_context(self) -> str | None:
        """Expose the same observed policy facts to the isolated action reviewer.

        This proves an offer-review escalation basis, not that an offer objectively
        does not exist, nor that separate banking operations may be abandoned.
        Only accepted dialogue and delivered public knowledge can establish it.
        """
        if not self.has_verified_offer_capability_gap():
            return None
        reasons = [
            reason
            for reason in (
                "unconfirmed_external_communication",
                "customer_demands_after_unavailable_offer_refusal",
            )
            if transfer_reason_denial(
                reason,
                ledger=self.ledger,
                require_knowledge_evidence=True,
                knowledge_identifiers=self._knowledge_identifiers,
                knowledge_search_dialogue_sequences=self._knowledge_search_dialogue_sequences,
            )
            is None
        ]
        evidence = {
            "schema_version": 1,
            "scope": "current_offer_human_review",
            "current_customer_requests_human": True,
            "knowledge_search_precedes_observed_offer_nonconfirmation_or_refusal": True,
            "supported_delivered_policy_reason_codes": reasons,
            "separate_banking_outcomes_remain_subject_to_review": True,
            "does_not_prove_offer_nonexistence": True,
        }
        return (
            "<offer-transfer-policy-evidence>\n"
            + json.dumps(evidence, sort_keys=True)
            + "\n</offer-transfer-policy-evidence>"
        )

    def render_knowledge_search_review_context(self) -> str:
        """Expose observed search timing, without asserting exhaustive coverage.

        Private Cayu searches do not appear in the external banking-tool ledger.
        Their host-recorded dialogue sequence lets a reviewer distinguish a real
        preceding search from a model-authored claim that it searched.
        """
        return (
            "<knowledge-search-observations>\n"
            + json.dumps(
                {
                    "dialogue_sequences": sorted(self._knowledge_search_dialogue_sequences),
                    "does_not_prove_exhaustive_search_or_absence": True,
                },
                sort_keys=True,
            )
            + "\n</knowledge-search-observations>"
        )

    def export_state(self) -> dict[str, Any]:
        return {
            "denied_calls": self.denied_calls,
            "denials_by_reason": dict(sorted(self._denials_by_reason.items())),
            "knowledge_identifier_count": len(self._knowledge_identifiers),
            "public_tool_identifier_count": len(self._public_tool_identifiers),
            "public_tool_identifiers": list(self.public_tool_identifiers()),
            "knowledge_search_dialogue_sequences": list(self._knowledge_search_dialogue_sequences),
            "recovery_bypass_code_count": len(self._recovery_bypass_codes),
            "requires_knowledge_evidence": self.require_knowledge_evidence,
            "reviewed_calls": self.reviewed_calls,
            "schema_version": 3,
        }

    def public_tool_identifiers(self) -> tuple[str, ...]:
        """Return bounded public capability names already exposed through Cayu."""

        return tuple(sorted(self._public_tool_identifiers))[-_MAX_PUBLIC_TOOL_IDENTIFIERS:]

    def knowledge_identifier_is_exposed(self, identifier: str) -> bool:
        """Return whether an exact identifier appeared in delivered Cayu knowledge."""

        if (
            not isinstance(identifier, str)
            or _KNOWLEDGE_IDENTIFIER_PATTERN.fullmatch(identifier) is None
        ):
            raise ValueError("identifier must be one normalized underscore-delimited name.")
        return identifier in self._knowledge_identifiers

    def render_public_capability_context(self) -> str | None:
        """Keep exposed public capability names visible across transient turns."""

        identifiers = self.public_tool_identifiers()
        if not identifiers:
            return None
        payload = {
            "identifiers": list(identifiers),
            "omitted": max(0, len(self._public_tool_identifiers) - len(identifiers)),
        }
        return (
            "These exact identifiers appeared in public banking documents already delivered "
            "by Cayu. This catalog preserves names, not procedures or authorization: use the "
            "relevant public document and current workflow before calling one. "
            "list_discoverable_agent_tools reports tools already called; it does not discover "
            "new tools.\n<public-capability-memory>\n"
            f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n"
            "</public-capability-memory>"
        )

    def _denial_reason(
        self,
        call: ToolCallPart,
        proposed_effects: set[tuple[str, str]],
        proposed_reads: set[tuple[str, str]],
    ) -> str | None:
        if call.tool_name == "transfer_to_human_agents":
            from cayu_tau3_banking.transfer_reason import _TRANSFER_REASONS

            transfer_denial = (
                (
                    None
                    if isinstance(call.arguments.get("reason"), str)
                    and call.arguments["reason"] in _TRANSFER_REASONS
                    else "transfer reason must be one exact public transfer reason code"
                )
                if self.authoritative_only
                else transfer_reason_denial(
                    call.arguments.get("reason"),
                    ledger=self.ledger,
                    require_knowledge_evidence=self.require_knowledge_evidence,
                    knowledge_identifiers=self._knowledge_identifiers,
                    knowledge_search_dialogue_sequences=(self._knowledge_search_dialogue_sequences),
                )
            )
            if transfer_denial is not None:
                return transfer_denial

        if call.tool_name in {
            "give_discoverable_user_tool",
            "unlock_discoverable_agent_tool",
        }:
            argument_name = (
                "discoverable_tool_name"
                if call.tool_name == "give_discoverable_user_tool"
                else "agent_tool_name"
            )
            requested_tool = _nonblank_string(call.arguments.get(argument_name))
            if (
                self.require_knowledge_evidence
                and requested_tool not in self._knowledge_identifiers
            ):
                return _UNEXPOSED_DISCOVERABLE_TOOL_REASON
            if (
                call.tool_name == "give_discoverable_user_tool"
                and requested_tool is not None
                and any(
                    _nonblank_string(arguments.get("discoverable_tool_name")) == requested_tool
                    for arguments in self.ledger.successful_call_arguments(
                        "give_discoverable_user_tool"
                    )
                )
            ):
                return _REPEATED_USER_TOOL_DELIVERY_REASON

        if call.tool_name in _PROTECTED_USER_TOOLS:
            user_id = _nonblank_string(call.arguments.get("user_id"))
            if user_id is None or not self._user_is_verified(user_id):
                return _MISSING_VERIFICATION_REASON

        if call.tool_name == "log_verification":
            verification_reason = self._verification_denial_reason(call)
            if verification_reason is not None:
                return verification_reason

        if call.tool_name == "call_discoverable_agent_tool":
            tool_name = call.arguments.get("agent_tool_name")
            if (
                not isinstance(tool_name, str)
                or not tool_name.strip()
                or tool_name.strip() not in self.ledger.unlocked_agent_tools
            ):
                return _LOCKED_DISCOVERABLE_TOOL_REASON

        read_key = _public_read_key(call)
        if read_key is not None and not self.authoritative_only:
            if read_key in proposed_reads:
                return _SAME_BATCH_REDUNDANT_READ_REASON
            proposed_reads.add(read_key)
            if self.ledger.redundant_read_succeeded_in_current_customer_turn(
                call.tool_name,
                call.arguments,
            ):
                return _REDUNDANT_READ_REASON

        if call.tool_name not in _NONREPEATABLE_EFFECTS:
            return None
        effect_key = _proposed_effect_key(call)
        if call.tool_name == "change_user_email":
            # A reversible assignment is duplicate only against its latest value,
            # not every value ever assigned. Preserve same-customer batch ordering.
            user_id = call.arguments.get("user_id")
            prior_in_batch = [
                key
                for key in proposed_effects
                if key[0] == call.tool_name and json.loads(key[1]).get("user_id") == user_id
            ]
            previous = next(
                (
                    args
                    for args in reversed(self.ledger.successful_call_arguments(call.tool_name))
                    if args.get("user_id") == user_id
                ),
                None,
            )
            if prior_in_batch:
                if effect_key in prior_in_batch:
                    return _SAME_BATCH_DUPLICATE_REASON
                proposed_effects.difference_update(prior_in_batch)
            elif previous == call.arguments:
                return _DUPLICATE_EFFECT_REASON
            proposed_effects.add(effect_key)
            return None
        if effect_key in proposed_effects:
            return _SAME_BATCH_DUPLICATE_REASON
        proposed_effects.add(effect_key)
        if self.ledger.effect_succeeded(call.tool_name, call.arguments):
            return _DUPLICATE_EFFECT_REASON
        return None

    def _verification_denial_reason(self, call: ToolCallPart) -> str | None:
        if self.ledger.successful_call_arguments("log_verification"):
            return _REPEATED_VERIFICATION_REASON

        user_id = _nonblank_string(call.arguments.get("user_id"))
        time_verified = _nonblank_string(call.arguments.get("time_verified"))
        time_evidence = self.ledger.successful_result_texts("get_current_time")
        if time_verified is None or not _value_appears_in_any(time_verified, time_evidence):
            return _UNPROVEN_VERIFICATION_TIME_REASON

        bypass_code = _uniform_identity_value(call)
        if bypass_code is not None:
            if bypass_code not in self._recovery_bypass_codes or not _exact_value_appears_in_any(
                bypass_code,
                self.ledger.customer_evidence_messages,
            ):
                return _UNPROVEN_RECOVERY_BYPASS_REASON
            return None

        lookup_evidence = tuple(
            result
            for tool_name in _IDENTITY_LOOKUP_TOOLS
            for result in self.ledger.successful_result_texts(tool_name)
        )
        required_values = [
            _nonblank_string(call.arguments.get(field)) for field in _VERIFICATION_FIELDS
        ]
        if user_id is None or any(value is None for value in required_values):
            return _UNPROVEN_VERIFICATION_REASON
        asserted_values = [user_id, *required_values]
        if not any(
            _lookup_contains_one_identity_record(result, asserted_values)
            for result in lookup_evidence
        ):
            return _UNPROVEN_VERIFICATION_REASON

        # Mentions, denials and confirmations can contain identical field values.
        # The acting model owns their interpretation when proposing this call.
        # Passing record checks does not establish customer confirmation.
        return None

    def _user_is_verified(self, user_id: str) -> bool:
        if user_id in self.ledger.verification_logs_by_user:
            return True
        successful_bypass_codes = {
            code
            for arguments in self.ledger.successful_call_arguments("log_verification")
            if (code := _successful_bypass_code(arguments)) is not None
            and code in self._recovery_bypass_codes
        }
        if not successful_bypass_codes:
            return False
        lookup_evidence = tuple(
            result
            for tool_name in _IDENTITY_LOOKUP_TOOLS
            for result in self.ledger.successful_result_texts(tool_name)
        )
        return any(
            _lookup_contains_one_identity_record(result, (user_id,)) for result in lookup_evidence
        )


def _restored_knowledge_search_sequences(
    messages: list[Message] | None,
    *,
    last_dialogue_sequence: int,
) -> tuple[int, ...]:
    for message in reversed(messages or ()):
        if not isinstance(message, AssistantMessage) or not isinstance(message.raw_data, Mapping):
            continue
        state = message.raw_data.get("cayu_tau3_admission")
        if not isinstance(state, Mapping) or state.get("schema_version") not in {2, 3}:
            continue
        raw_sequences = state.get("knowledge_search_dialogue_sequences")
        if (
            not isinstance(raw_sequences, list)
            or len(raw_sequences) > _MAX_KNOWLEDGE_SEARCH_RECEIPTS
        ):
            continue
        sequences = tuple(raw_sequences)
        if (
            any(
                type(sequence) is not int or not 1 <= sequence <= last_dialogue_sequence
                for sequence in sequences
            )
            or tuple(sorted(set(sequences))) != sequences
        ):
            continue
        return sequences
    return ()


def _restored_public_tool_identifiers(messages: list[Message] | None) -> tuple[str, ...]:
    for message in reversed(messages or ()):
        if not isinstance(message, AssistantMessage) or not isinstance(message.raw_data, Mapping):
            continue
        state = message.raw_data.get("cayu_tau3_admission")
        if not isinstance(state, Mapping) or state.get("schema_version") != 3:
            continue
        raw_identifiers = state.get("public_tool_identifiers")
        if not isinstance(raw_identifiers, list):
            continue
        identifiers = tuple(raw_identifiers)
        if (
            len(identifiers) > _MAX_PUBLIC_TOOL_IDENTIFIERS
            or any(
                not isinstance(identifier, str)
                or _PUBLIC_TOOL_IDENTIFIER_PATTERN.fullmatch(identifier) is None
                for identifier in identifiers
            )
            or tuple(sorted(set(identifiers))) != identifiers
        ):
            continue
        return identifiers
    return ()


def _proposed_effect_key(call: ToolCallPart) -> tuple[str, str]:
    return call.tool_name, json.dumps(
        call.arguments,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _public_read_key(call: ToolCallPart) -> tuple[str, str] | None:
    if call.tool_name == "list_discoverable_agent_tools":
        return _proposed_effect_key(call)
    if call.tool_name != "call_discoverable_agent_tool":
        return None
    tool_name = call.arguments.get("agent_tool_name")
    raw_arguments = call.arguments.get("arguments")
    if (
        not isinstance(tool_name, str)
        or not tool_name.strip().startswith(("get_", "list_"))
        or not isinstance(raw_arguments, str)
    ):
        return None
    try:
        inner_arguments = json.loads(raw_arguments)
        if type(inner_arguments) is not dict:
            return None
        encoded_arguments = json.dumps(
            inner_arguments,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return None
    return f"{call.tool_name}:{tool_name.strip()}", encoded_arguments


def _nonblank_string(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _value_appears_in_any(value: str, texts: Sequence[str]) -> bool:
    return any(_value_appears_in_text(value, text) for text in texts)


def _exact_value_appears_in_any(value: str, texts: Sequence[str]) -> bool:
    return any(value in text for text in texts)


def _value_appears_in_text(value: str, text: str) -> bool:
    normalized_value = _identity_normal_form(value)
    return len(normalized_value) >= 4 and normalized_value in _identity_normal_form(text)


def _lookup_contains_one_identity_record(result: str, values: Sequence[str]) -> bool:
    """Match fields in one returned record, never substrings across arbitrary prose."""
    if not values or len(values) not in {1, 1 + len(_VERIFICATION_FIELDS)}:
        return False
    try:
        from cayu_tau3_banking.public_argument_contracts import _unique_object

        parsed = json.loads(result, object_pairs_hook=_unique_object)
        records = parsed if isinstance(parsed, list) else [parsed]
    except (ValueError, TypeError, RecursionError):
        sections = _LOOKUP_RECORD_BOUNDARY_PATTERN.split(result)
        records = []
        for section in sections[1:] if len(sections) > 1 else sections:
            record = {}
            duplicated = False
            for line in section.splitlines():
                key, separator, value = line.strip().partition(":")
                if separator and key in {"user_id", *_VERIFICATION_FIELDS}:
                    if key in record:
                        duplicated = True
                    record[key] = value.strip()
            if not duplicated:
                records.append(record)
    fields = ("user_id", *_VERIFICATION_FIELDS)[: len(values)]
    for record in records:
        if not isinstance(record, dict) or record.get("user_id") != values[0]:
            continue
        if all(
            isinstance(record.get(field), str)
            and (
                record[field].strip().casefold() == value.strip().casefold()
                if field in {"user_id", "email"}
                else _identity_normal_form(record[field]) == _identity_normal_form(value)
            )
            for field, value in zip(fields, values, strict=True)
        ):
            return True
    return False


def _identity_normal_form(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _uniform_identity_value(call: ToolCallPart) -> str | None:
    return _uniform_identity_arguments(call.arguments)


def _uniform_identity_arguments(arguments: dict[str, Any]) -> str | None:
    values = tuple(
        _nonblank_string(arguments.get(argument)) for argument in _VERIFICATION_IDENTITY_ARGUMENTS
    )
    if any(value is None for value in values):
        return None
    first = values[0]
    if first is None or any(value != first for value in values[1:]):
        return None
    return first


def _successful_bypass_code(arguments: dict[str, Any]) -> str | None:
    return _uniform_identity_arguments(arguments)


__all__ = ["BankingToolAdmissionGate"]
