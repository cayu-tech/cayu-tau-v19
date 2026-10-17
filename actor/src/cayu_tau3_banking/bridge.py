"""Lossless-enough message bridge between tau2 and Cayu model providers.

The benchmark owns tool execution. This module invokes only Cayu's model-provider
contract and translates the resulting text or function calls back to tau2. It never
executes a benchmark tool itself.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hmac
import inspect
import json
import math
import re
import secrets
import threading
from collections import Counter
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Protocol

from cayu import Message as CayuMessage
from cayu import (
    MessageRole,
    ProviderStatePart,
    TextPart,
    ToolCallPart,
    UsageMetrics,
    normalize_usage_metrics,
)
from cayu.providers import (
    ModelCompletion,
    ModelProvider,
    ModelRequest,
    ModelStreamDeadlineError,
    ModelStreamEventType,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

from cayu_tau3_banking.attempt_diagnostics import (
    SAFE_INTERNAL_DIAGNOSTIC_COUNT_KEYS,
    SAFE_INTERNAL_ERROR_CODES,
    SAFE_INTERNAL_REMEDIATION_CODES,
    SAFE_INTERNAL_REPAIR_CLASSES,
    AttemptDiagnosticsRecorder,
    BridgeFailureDiagnostics,
    InternalRoundDiagnostic,
    RemediationLifecycleDiagnostic,
)

DEFAULT_COMPLETION_TIMEOUT_SECONDS = 600.0
_MAX_INTERNAL_CONTINUATIONS = 8
_MAX_INTERNAL_TOOL_ROUNDS = 32
_MAX_STALLED_INTERNAL_TOOL_ROUNDS = 4
_MAX_STALLED_INTERNAL_TOOL_ROUNDS_SINCE_PROGRESS = 8
_MAX_EXTERNAL_ADMISSION_ROUNDS = 8
_MAX_TERMINAL_GATE_ROUNDS = 4
_MAX_EMPTY_RESPONSE_RETRIES = 2
_MAX_REQUIRED_ACTION_BATCH_SIZE = 24
_EMPTY_RESPONSE_CORRECTION = (
    "The provider returned an unusable empty response for this same turn. "
    "Now emit exactly one nonblank customer-facing text response or one or more valid tool calls."
)
_INTERNAL_REMEDIATION_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_INTERNAL_REMEDIATION_ITEM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_INTERNAL_REMEDIATION_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_CORRECTION_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_INTERNAL_REMEDIATION_REPAIRS = 8
_DISCOVERABLE_AGENT_CALL_TOOL = "call_discoverable_agent_tool"
_DISCOVERABLE_USER_DELIVERY_TOOL = "give_discoverable_user_tool"
_FIXED_CURRENCY_ARGUMENT_FIELDS = {
    "apply_savings_account_credit_6831": frozenset({"amount"}),
    "submit_interest_discrepancy_report_7294": frozenset({"amount_difference"}),
}
_PROVIDER_CALL_PHASES = frozenset(
    {
        "empty_response_retry",
        "external_admission_correction",
        "initial",
        "internal_tool_result",
        "provider_continuation",
        "terminal_readiness_correction",
    }
)
_PROVIDER_RESULT_KINDS = frozenset(
    {
        "continuation_text",
        "empty",
        "mixed_tool_calls",
        "terminal_text",
        "tool_calls",
    }
)


class BridgeProtocolError(RuntimeError):
    """A provider response cannot be represented safely in tau2."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: BridgeFailureDiagnostics | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class InternalToolRemediation:
    """Bounded machine-actionable repair guidance for one rejected internal call."""

    code: str
    plan_revision: int
    item_id: str
    forbidden_transition: str
    allowed_repairs: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Internal remediation schema_version must be 1.")
        for field_name in ("code", "forbidden_transition"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not _INTERNAL_REMEDIATION_TOKEN.fullmatch(value):
                raise ValueError(f"Internal remediation {field_name} is invalid.")
        if type(self.plan_revision) is not int or self.plan_revision < 0:
            raise ValueError("Internal remediation plan_revision must be nonnegative.")
        if not isinstance(self.item_id, str) or not _INTERNAL_REMEDIATION_ITEM_ID.fullmatch(
            self.item_id
        ):
            raise ValueError("Internal remediation item_id is invalid.")
        if (
            type(self.allowed_repairs) is not tuple
            or not self.allowed_repairs
            or len(self.allowed_repairs) > _MAX_INTERNAL_REMEDIATION_REPAIRS
            or len(set(self.allowed_repairs)) != len(self.allowed_repairs)
            or any(
                not isinstance(repair, str) or not _INTERNAL_REMEDIATION_TOKEN.fullmatch(repair)
                for repair in self.allowed_repairs
            )
        ):
            raise ValueError("Internal remediation allowed_repairs are invalid.")

    @property
    def correction_id(self) -> str:
        """Return a stable identity for comparing equivalent corrections."""

        return "sha256:" + sha256(_canonical_json_bytes(self._identity_payload())).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical public repair object shown to the model."""

        return {**self._identity_payload(), "correction_id": self.correction_id}

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "allowed_repairs": list(self.allowed_repairs),
            "code": self.code,
            "forbidden_transition": self.forbidden_transition,
            "item_id": self.item_id,
            "plan_revision": self.plan_revision,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class InternalRemediationResolution:
    """One controller-authoritative resolution of a prior typed correction."""

    correction_id: str
    code: str
    repair_class: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Internal remediation resolution schema_version must be 1.")
        if not isinstance(self.correction_id, str) or not _INTERNAL_REMEDIATION_ID.fullmatch(
            self.correction_id
        ):
            raise ValueError("Internal remediation resolution correction_id is invalid.")
        for field_name in ("code", "repair_class"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not _INTERNAL_REMEDIATION_TOKEN.fullmatch(value):
                raise ValueError(f"Internal remediation resolution {field_name} is invalid.")

    @classmethod
    def from_remediation(
        cls,
        remediation: InternalToolRemediation,
        *,
        repair_class: str,
    ) -> InternalRemediationResolution:
        """Bind an accepted repair class to the exact correction it resolves."""

        if not isinstance(remediation, InternalToolRemediation):
            raise TypeError("remediation must be an InternalToolRemediation.")
        if repair_class not in remediation.allowed_repairs:
            raise ValueError("Resolved repair_class was not allowed by the remediation.")
        return cls(
            correction_id=remediation.correction_id,
            code=remediation.code,
            repair_class=repair_class,
        )


@dataclass(frozen=True)
class InternalToolResult:
    """Result of a controller-only tool that is never exposed to tau."""

    content: str
    is_error: bool = False
    diagnostic_code: str | None = None
    remediation: InternalToolRemediation | None = None
    resolutions: tuple[InternalRemediationResolution, ...] = ()
    structured: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("Internal tool result content must be text.")
        if type(self.is_error) is not bool:
            raise TypeError("Internal tool result is_error must be a boolean.")
        if self.structured is not None:
            if not isinstance(self.structured, dict):
                raise TypeError("Internal tool structured result must be an object.")
            try:
                encoded = _canonical_json_bytes(self.structured)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("Internal tool structured result must be finite JSON.") from exc
            if len(encoded) > 128_000:
                raise ValueError("Internal tool structured result is too large.")
        if self.diagnostic_code is not None and (
            not isinstance(self.diagnostic_code, str)
            or self.diagnostic_code not in SAFE_INTERNAL_ERROR_CODES
        ):
            raise ValueError("Internal tool result diagnostic_code is unsupported.")
        if self.diagnostic_code is not None and not self.is_error:
            raise ValueError("Internal tool result diagnostic_code requires an error result.")
        if self.remediation is not None and not isinstance(
            self.remediation, InternalToolRemediation
        ):
            raise TypeError("Internal tool result remediation is invalid.")
        if self.remediation is not None and not self.is_error:
            raise ValueError("Internal remediation requires an error result.")
        if (
            type(self.resolutions) is not tuple
            or len(self.resolutions) > _MAX_INTERNAL_REMEDIATION_REPAIRS
        ):
            raise ValueError("Internal remediation resolutions are invalid.")
        if any(
            not isinstance(resolution, InternalRemediationResolution)
            for resolution in self.resolutions
        ):
            raise ValueError("Internal remediation resolutions are invalid.")
        if len({resolution.correction_id for resolution in self.resolutions}) != len(
            self.resolutions
        ):
            raise ValueError("Internal remediation resolutions are invalid.")
        if self.remediation is not None and self.resolutions:
            raise ValueError("An internal result cannot issue and resolve remediation together.")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class _OpenInternalRemediation:
    remediation: InternalToolRemediation
    first_round: int


class InternalRemediationLifecycle:
    """Track typed corrections without exporting their identities or targets."""

    def __init__(self) -> None:
        self._open_by_id: dict[str, _OpenInternalRemediation] = {}
        self._open_id_by_target: dict[tuple[str, str], str] = {}
        self._issued = 0
        self._repeated = 0
        self._resolved = 0
        self._superseded = 0
        self._repair_rounds_total = 0
        self._repair_rounds_max = 0
        self._rejection_codes: Counter[str] = Counter()
        self._accepted_repair_classes: Counter[str] = Counter()

    @property
    def unresolved_count(self) -> int:
        return len(self._open_by_id)

    def observe(
        self,
        results: Sequence[InternalToolResult],
        *,
        round_number: int,
    ) -> tuple[str, ...]:
        """Apply one ordered internal round and return its private correction signature."""

        if type(round_number) is not int or round_number < 1:
            raise ValueError("round_number must be a positive integer.")
        correction_ids: set[str] = set()
        for result in results:
            remediation = result.remediation
            if remediation is not None:
                correction_ids.add(remediation.correction_id)
                self._observe_remediation(remediation, round_number=round_number)
            for resolution in result.resolutions:
                self._observe_resolution(resolution, round_number=round_number)
        return tuple(sorted(correction_ids))

    def snapshot(self, *, exhausted: bool) -> RemediationLifecycleDiagnostic:
        """Return bounded aggregate telemetry with application allowlisting."""

        if type(exhausted) is not bool:
            raise TypeError("exhausted must be a boolean.")
        return RemediationLifecycleDiagnostic(
            issued=self._issued,
            repeated=self._repeated,
            resolved=self._resolved,
            superseded=self._superseded,
            unresolved=self.unresolved_count,
            exhausted=self.unresolved_count if exhausted else 0,
            repair_rounds_total=self._repair_rounds_total,
            repair_rounds_max=self._repair_rounds_max,
            rejection_codes=_safe_remediation_counts(
                self._rejection_codes,
                allowed=SAFE_INTERNAL_REMEDIATION_CODES,
            ),
            accepted_repair_classes=_safe_remediation_counts(
                self._accepted_repair_classes,
                allowed=SAFE_INTERNAL_REPAIR_CLASSES,
            ),
        )

    def _observe_remediation(
        self,
        remediation: InternalToolRemediation,
        *,
        round_number: int,
    ) -> None:
        correction_id = remediation.correction_id
        if correction_id in self._open_by_id:
            self._repeated += 1
            return
        target = (remediation.code, remediation.item_id)
        superseded_id = self._open_id_by_target.get(target)
        if superseded_id is not None:
            self._open_by_id.pop(superseded_id)
            self._superseded += 1
        self._open_by_id[correction_id] = _OpenInternalRemediation(
            remediation=remediation,
            first_round=round_number,
        )
        self._open_id_by_target[target] = correction_id
        self._issued += 1
        self._rejection_codes[remediation.code] += 1

    def _observe_resolution(
        self,
        resolution: InternalRemediationResolution,
        *,
        round_number: int,
    ) -> None:
        pending = self._open_by_id.pop(resolution.correction_id, None)
        if pending is None:
            raise BridgeProtocolError(
                "Internal controller resolved an unknown or superseded remediation."
            )
        remediation = pending.remediation
        if (
            resolution.code != remediation.code
            or resolution.repair_class not in remediation.allowed_repairs
        ):
            raise BridgeProtocolError(
                "Internal controller resolution conflicts with its remediation."
            )
        target = (remediation.code, remediation.item_id)
        if self._open_id_by_target.get(target) != resolution.correction_id:
            raise BridgeProtocolError("Internal remediation target identity is inconsistent.")
        self._open_id_by_target.pop(target)
        repair_rounds = round_number - pending.first_round
        self._resolved += 1
        self._repair_rounds_total += repair_rounds
        self._repair_rounds_max = max(self._repair_rounds_max, repair_rounds)
        self._accepted_repair_classes[resolution.repair_class] += 1


def _safe_remediation_counts(
    counts: Mapping[str, int],
    *,
    allowed: frozenset[str],
) -> dict[str, int]:
    safe: Counter[str] = Counter()
    for key, count in counts.items():
        safe[key if key in allowed else "other"] += count
    return dict(sorted(safe.items()))


class InternalToolHandler(Protocol):
    """Minimal contract for model-visible, controller-local tools."""

    def tool_schemas(self) -> Sequence[Mapping[str, Any]]:
        """Return Cayu function-tool schemas."""

    def handles(self, name: str) -> bool:
        """Return whether this handler owns a tool name."""

    def invoke(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> InternalToolResult | Awaitable[InternalToolResult]:
        """Execute a controller-local operation without benchmark side effects.

        Handlers may be synchronous for pure state transitions or asynchronous for
        Cayu-backed retrieval and persistence operations.
        """


class InternalProgressSource(Protocol):
    """Optional deterministic progress signal for controller-local work."""

    def progress_token(self) -> str:
        """Return a stable token that changes only when useful state changes."""


class InternalActionToolSource(Protocol):
    """Optional deterministic continuation advice after a successful private round."""

    def required_action_tool_names(self) -> tuple[str, ...]:
        """Return ordered tools that can advance current authoritative work."""

    def required_action_batch_size(self, name: str) -> int:
        """Return the maximum simultaneous calls for one required tool name."""


class InternalExternalStagingSource(Protocol):
    """Optional authority to stage external calls after one private state change."""

    def allows_staged_external_calls(self, internal_tool_names: tuple[str, ...]) -> bool:
        """Return whether this exact ordered private prefix may precede external calls."""


class InternalContextSource(Protocol):
    """Optional source of authoritative state refreshed inside a private tool loop."""

    def render_internal_context(self) -> str | None:
        """Return current bounded controller context, or null when none is needed."""


class InternalTranscriptRetentionSource(Protocol):
    """Optional policy for superseding private state-publication rounds."""

    def transcript_retention_key(self, name: str) -> str | None:
        """Return a stable key when the newest successful call supersedes older ones."""


@dataclass(frozen=True)
class ExternalToolAdmission:
    """One host-side decision made before a tau tool call is returned."""

    tool_call_id: str
    allowed: bool
    reason: str | None = None
    required_tool_names: tuple[str, ...] = ()
    diagnostic_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip():
            raise ValueError("tool_call_id must be nonblank text.")
        if type(self.allowed) is not bool:
            raise TypeError("allowed must be a boolean.")
        if self.allowed and self.reason is not None:
            raise ValueError("Allowed external tool calls cannot include a denial reason.")
        if self.allowed and self.diagnostic_code is not None:
            raise ValueError("Allowed external tool calls cannot include a diagnostic code.")
        if not self.allowed and (not isinstance(self.reason, str) or not self.reason.strip()):
            raise ValueError("Denied external tool calls require a nonblank reason.")
        if self.diagnostic_code is not None and (
            not isinstance(self.diagnostic_code, str)
            or _INTERNAL_REMEDIATION_TOKEN.fullmatch(self.diagnostic_code) is None
        ):
            raise ValueError("External admission diagnostic_code is invalid.")
        if self.allowed and self.required_tool_names:
            raise ValueError("Allowed external tool calls cannot include correction guidance.")
        if (
            type(self.required_tool_names) is not tuple
            or len(set(self.required_tool_names)) != len(self.required_tool_names)
            or any(
                not isinstance(name, str) or not _CORRECTION_TOOL_NAME.fullmatch(name)
                for name in self.required_tool_names
            )
        ):
            raise ValueError("External admission required_tool_names are invalid.")


class ExternalToolCallGate(Protocol):
    """Host-owned admission boundary for tau environment tool calls."""

    def review(
        self,
        calls: Sequence[ToolCallPart],
    ) -> Sequence[ExternalToolAdmission] | Awaitable[Sequence[ExternalToolAdmission]]:
        """Return exactly one ordered decision for every proposed call."""


class CompositeExternalToolCallGate:
    """Fail-closed conjunction of independent external tool-call gates."""

    def __init__(
        self,
        gates: Sequence[ExternalToolCallGate],
        *,
        denial_required_tool_names: tuple[str, ...] = (),
    ) -> None:
        if isinstance(gates, str | bytes) or not gates:
            raise ValueError("gates must contain at least one external tool-call gate.")
        if (
            type(denial_required_tool_names) is not tuple
            or len(set(denial_required_tool_names)) != len(denial_required_tool_names)
            or any(
                not isinstance(name, str) or not _CORRECTION_TOOL_NAME.fullmatch(name)
                for name in denial_required_tool_names
            )
        ):
            raise ValueError("denial_required_tool_names are invalid.")
        self._gates = tuple(gates)
        self._denial_required_tool_names = denial_required_tool_names

    async def review(
        self,
        calls: Sequence[ToolCallPart],
    ) -> tuple[ExternalToolAdmission, ...]:
        combined = [
            ExternalToolAdmission(tool_call_id=call.tool_call_id, allowed=True) for call in calls
        ]
        for gate in self._gates:
            admissions = await _review_external_tool_calls(calls, gate)
            combined = [
                _combine_external_admissions(current, additional)
                for current, additional in zip(combined, admissions, strict=True)
            ]
        return tuple(
            admission
            if admission.allowed or not self._denial_required_tool_names
            else ExternalToolAdmission(
                tool_call_id=admission.tool_call_id,
                allowed=False,
                reason=admission.reason,
                required_tool_names=_ordered_unique(
                    (
                        *admission.required_tool_names,
                        *self._denial_required_tool_names,
                    )
                ),
                diagnostic_code=admission.diagnostic_code,
            )
            for admission in combined
        )


@dataclass(frozen=True)
class TerminalOutputAdmission:
    """Host-side decision made before customer-facing text leaves the bridge."""

    allowed: bool
    reason: str | None = None
    correction_code: str | None = None
    required_tool_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise TypeError("allowed must be a boolean.")
        if self.allowed and self.reason is not None:
            raise ValueError("Allowed terminal output cannot include a denial reason.")
        if not self.allowed and (not isinstance(self.reason, str) or not self.reason.strip()):
            raise ValueError("Denied terminal output requires a nonblank reason.")
        if self.allowed and (self.correction_code is not None or self.required_tool_names):
            raise ValueError("Allowed terminal output cannot include correction guidance.")
        if self.correction_code is not None and (
            not isinstance(self.correction_code, str)
            or not _INTERNAL_REMEDIATION_TOKEN.fullmatch(self.correction_code)
        ):
            raise ValueError("Terminal correction_code is invalid.")
        if (
            type(self.required_tool_names) is not tuple
            or len(set(self.required_tool_names)) != len(self.required_tool_names)
            or any(
                not isinstance(name, str) or not _CORRECTION_TOOL_NAME.fullmatch(name)
                for name in self.required_tool_names
            )
        ):
            raise ValueError("Terminal required_tool_names are invalid.")


class TerminalOutputGate(Protocol):
    """Host-owned readiness boundary for customer-facing text outputs."""

    def review_terminal_output(
        self,
        text: str,
    ) -> TerminalOutputAdmission | Awaitable[TerminalOutputAdmission]:
        """Decide whether this customer-facing text may leave the bridge."""


class TurnContextProvider(Protocol):
    """Asynchronous provider of one transient, provider-facing turn context."""

    async def prepare_context(self) -> str | None:
        """Return bounded context for the next model call, or ``None``."""


@dataclass
class CayuTauConversation:
    """Provider-facing transcript and tool-call identity ledger for one simulation."""

    messages: list[CayuMessage]
    tool_names_by_call_id: dict[str, str] = field(default_factory=dict)
    resolved_call_ids: set[str] = field(default_factory=set)
    provider_call_ids: set[str] = field(default_factory=set)
    attempt_id: str = field(default_factory=lambda: secrets.token_hex(16))
    correlation_tag: str | None = None
    active_failure_diagnostics: BridgeFailureDiagnostics | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class _ModelOutput:
    text: str
    tool_calls: tuple[ToolCallPart, ...]
    provider_state: tuple[ProviderStatePart, ...]
    completion: ModelCompletion
    usage: Mapping[str, Any] | None
    model: str | None


class CayuTauBridge:
    """Translate tau2 turns to one Cayu provider without executing tools."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        options: Mapping[str, Any] | None = None,
        max_internal_continuations: int = _MAX_INTERNAL_CONTINUATIONS,
        max_internal_tool_rounds: int = _MAX_INTERNAL_TOOL_ROUNDS,
        max_stalled_internal_tool_rounds: int = _MAX_STALLED_INTERNAL_TOOL_ROUNDS,
        max_stalled_internal_tool_rounds_since_progress: int = (
            _MAX_STALLED_INTERNAL_TOOL_ROUNDS_SINCE_PROGRESS
        ),
        max_external_admission_rounds: int = _MAX_EXTERNAL_ADMISSION_ROUNDS,
        max_terminal_gate_rounds: int = _MAX_TERMINAL_GATE_ROUNDS,
        max_empty_response_retries: int = _MAX_EMPTY_RESPONSE_RETRIES,
    ) -> None:
        if not isinstance(provider, ModelProvider):
            raise TypeError("provider must implement ModelProvider.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonblank string.")
        if type(max_internal_continuations) is not int or max_internal_continuations < 0:
            raise ValueError("max_internal_continuations must be a nonnegative integer.")
        if type(max_internal_tool_rounds) is not int or max_internal_tool_rounds < 0:
            raise ValueError("max_internal_tool_rounds must be a nonnegative integer.")
        if (
            type(max_stalled_internal_tool_rounds) is not int
            or max_stalled_internal_tool_rounds < 0
        ):
            raise ValueError("max_stalled_internal_tool_rounds must be a nonnegative integer.")
        if (
            type(max_stalled_internal_tool_rounds_since_progress) is not int
            or max_stalled_internal_tool_rounds_since_progress < 0
        ):
            raise ValueError(
                "max_stalled_internal_tool_rounds_since_progress must be a nonnegative integer."
            )
        if type(max_external_admission_rounds) is not int or max_external_admission_rounds < 0:
            raise ValueError("max_external_admission_rounds must be a nonnegative integer.")
        if type(max_terminal_gate_rounds) is not int or max_terminal_gate_rounds < 0:
            raise ValueError("max_terminal_gate_rounds must be a nonnegative integer.")
        if type(max_empty_response_retries) is not int or max_empty_response_retries < 0:
            raise ValueError("max_empty_response_retries must be a nonnegative integer.")
        copied_options = _copy_json_object(options or {}, "options")
        self.provider = provider
        self.model = model.strip()
        self.options = MappingProxyType(copied_options)
        self.max_internal_continuations = max_internal_continuations
        self.max_internal_tool_rounds = max_internal_tool_rounds
        self.max_stalled_internal_tool_rounds = max_stalled_internal_tool_rounds
        self.max_stalled_internal_tool_rounds_since_progress = (
            max_stalled_internal_tool_rounds_since_progress
        )
        self.max_external_admission_rounds = max_external_admission_rounds
        self.max_terminal_gate_rounds = max_terminal_gate_rounds
        self.max_empty_response_retries = max_empty_response_retries
        self._diagnostic_hmac_key = secrets.token_bytes(32)

    def new_conversation(
        self,
        *,
        system_prompt: str,
        message_history: Sequence[APICompatibleMessage] | None = None,
    ) -> CayuTauConversation:
        """Create one provider transcript, optionally replaying public tau history."""

        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a nonblank string.")
        conversation = CayuTauConversation(
            messages=[CayuMessage.text(MessageRole.SYSTEM, system_prompt.strip())]
        )
        for message in message_history or ():
            self.append_tau_input(conversation, message)
        return conversation

    def append_tau_input(
        self,
        conversation: CayuTauConversation,
        message: APICompatibleMessage | MultiToolMessage,
    ) -> None:
        """Append a tau message while retaining function-call name identity."""

        _require_conversation(conversation)
        if isinstance(message, MultiToolMessage):
            if not message.tool_messages:
                raise BridgeProtocolError("A tau multi-tool message cannot be empty.")
            _append_tau_tool_results(conversation, message.tool_messages)
            return
        if isinstance(message, SystemMessage):
            conversation.messages.append(
                CayuMessage.text(
                    MessageRole.SYSTEM, _require_text(message.content, "system content")
                )
            )
            return
        if isinstance(message, UserMessage):
            if message.is_tool_call():
                raise BridgeProtocolError("User-authored tool calls are not valid agent inputs.")
            content = _require_text(message.content, "user content")
            if conversation.correlation_tag is None:
                conversation.correlation_tag = _opaque_fingerprint(
                    self._diagnostic_hmac_key,
                    {"initial_user_message": content},
                )
            conversation.messages.append(CayuMessage.text(MessageRole.USER, content))
            return
        if isinstance(message, AssistantMessage):
            converted = _tau_assistant_to_cayu(message)
            calls = tuple(part for part in converted.content if type(part) is ToolCallPart)
            _record_call_names(conversation, calls)
            conversation.messages.append(converted)
            return
        if isinstance(message, ToolMessage):
            _append_tau_tool_results(conversation, (message,))
            return
        raise TypeError(f"Unsupported tau message type: {type(message).__name__}.")

    async def generate(
        self,
        conversation: CayuTauConversation,
        tools: Sequence[Tool],
        *,
        transient_context: str | None = None,
        internal_tool_handler: InternalToolHandler | None = None,
        external_tool_gate: ExternalToolCallGate | None = None,
        terminal_output_gate: TerminalOutputGate | None = None,
    ) -> AssistantMessage:
        """Generate one valid tau assistant turn and commit provider state atomically."""

        _require_conversation(conversation)
        if transient_context is not None and (
            not isinstance(transient_context, str) or not transient_context.strip()
        ):
            raise ValueError("transient_context must be nonblank text or null.")
        external_tool_schemas = _cayu_tool_schemas(tools)
        external_tool_names = {schema["name"] for schema in external_tool_schemas}
        base_messages = list(conversation.messages)
        transient_message = (
            None
            if transient_context is None
            else CayuMessage.text(MessageRole.SYSTEM, transient_context.strip())
        )
        generated_messages: list[CayuMessage] = []
        refreshed_internal_context: CayuMessage | None = None
        retained_internal_messages: dict[str, tuple[CayuMessage, ...]] = {}
        continuation_count = 0
        internal_tool_rounds = 0
        internal_progress_events = 0
        internal_stalled_rounds = 0
        stalled_internal_rounds_since_progress = 0
        internal_compacted_rounds = 0
        internal_context_refreshes = 0
        staged_external_batches = 0
        staged_external_calls = 0
        consecutive_stalled_internal_rounds = 0
        last_stalled_internal_signature: tuple[str, ...] | None = None
        external_admission_rounds = 0
        external_admission_rejections = 0
        external_rejection_codes: Counter[str] = Counter()
        terminal_gate_rounds = 0
        terminal_gate_rejections = 0
        terminal_gate_rounds_since_progress = 0
        total_empty_response_retries = 0
        outputs: list[_ModelOutput] = []
        provider_call_trace: list[dict[str, Any]] = []
        next_provider_call_phase = "initial"
        internal_round_trace: list[InternalRoundDiagnostic] = []
        internal_calls_by_tool: Counter[str] = Counter()
        terminal_rejection_codes: Counter[str] = Counter()
        terminal_correction_fingerprints: list[str] = []
        pending_required_tool_names: tuple[str, ...] = ()
        denied_external_tool_names: set[str] = set()
        remediation_lifecycle = InternalRemediationLifecycle()

        def failure_diagnostics(
            failure_code: str,
            current_output_kind: str,
            *,
            current_calls: Sequence[ToolCallPart] = (),
            recorded_tool_names: set[str] | None = None,
            current_output_fingerprint: str | None = None,
            exhausted: bool = True,
        ) -> BridgeFailureDiagnostics:
            return self._failure_diagnostics(
                conversation=conversation,
                failure_code=failure_code,
                current_output_kind=current_output_kind,
                current_output_fingerprint=current_output_fingerprint,
                outputs=outputs,
                round_trace=internal_round_trace,
                terminal_correction_fingerprints=terminal_correction_fingerprints,
                current_calls=current_calls,
                recorded_tool_names=recorded_tool_names or set(),
                internal_tool_rounds=internal_tool_rounds,
                internal_progress_events=internal_progress_events,
                internal_stalled_rounds=internal_stalled_rounds,
                stalled_internal_rounds_since_progress=(stalled_internal_rounds_since_progress),
                internal_diagnostic_counts=_internal_diagnostic_counts(internal_tool_handler),
                remediation_lifecycle=remediation_lifecycle.snapshot(exhausted=exhausted),
                consecutive_stalled_internal_rounds=consecutive_stalled_internal_rounds,
                internal_continuations=continuation_count,
                external_admission_rounds=external_admission_rounds,
                external_admission_rejections=external_admission_rejections,
                external_rejection_codes=external_rejection_codes,
                terminal_gate_rounds=terminal_gate_rounds,
                terminal_gate_rejections=terminal_gate_rejections,
                terminal_gate_rounds_since_progress=(terminal_gate_rounds_since_progress),
                terminal_rejection_codes=terminal_rejection_codes,
                empty_response_retries=total_empty_response_retries,
            )

        while True:
            # Internal capabilities can change after an accepted private round. Refresh
            # their schemas before every provider call so exhausted bounded tools vanish
            # immediately and state-specific controller operations can replace one another.
            internal_tool_schemas = _internal_tool_schemas(internal_tool_handler)
            internal_tool_names = {schema["name"] for schema in internal_tool_schemas}
            collisions = external_tool_names & internal_tool_names
            if collisions:
                names = ", ".join(sorted(collisions))
                raise BridgeProtocolError(f"Internal and tau tool names collide: {names}.")
            tool_schemas = [*external_tool_schemas, *internal_tool_schemas]
            offered_tool_names = external_tool_names | internal_tool_names
            forced_tool_name = next(
                (name for name in pending_required_tool_names if name in offered_tool_names),
                None,
            )
            forced_tool_call_limit = _internal_required_action_batch_size(
                internal_tool_handler,
                forced_tool_name,
            )
            if pending_required_tool_names and forced_tool_name is None:
                raise BridgeProtocolError(
                    "Controller correction requires a tool that is no longer available.",
                    diagnostics=failure_diagnostics(
                        "required_correction_tool_unavailable",
                        "controller_correction",
                    ),
                )
            empty_retry_count = 0
            while True:
                request_messages = list(base_messages)
                if transient_message is not None:
                    request_messages.append(transient_message)
                request_messages.extend(generated_messages)
                if refreshed_internal_context is not None:
                    request_messages.append(refreshed_internal_context)
                if empty_retry_count:
                    request_messages.append(
                        CayuMessage.text(MessageRole.SYSTEM, _EMPTY_RESPONSE_CORRECTION)
                    )
                request = ModelRequest(
                    model=self.model,
                    messages=request_messages,
                    tools=tool_schemas,
                    options=_required_tool_options(
                        self.options,
                        forced_tool_name,
                        max_calls=forced_tool_call_limit,
                    ),
                )
                provider_call_phase = (
                    "empty_response_retry" if empty_retry_count else next_provider_call_phase
                )
                conversation.active_failure_diagnostics = failure_diagnostics(
                    "gateway_completion_timeout",
                    "provider_call_in_progress",
                    exhausted=False,
                )
                output = await self._collect_output(request)
                outputs.append(output)
                provider_call_trace.append(
                    _provider_call_diagnostic(
                        request,
                        output,
                        phase=provider_call_phase,
                        forced_tool=forced_tool_name is not None,
                    )
                )
                terminal_empty = (
                    not output.tool_calls
                    and output.completion.end_turn is not False
                    and not output.text.strip()
                )
                if not terminal_empty:
                    if forced_tool_name is not None and (
                        not 1 <= len(output.tool_calls) <= forced_tool_call_limit
                        or any(call.tool_name != forced_tool_name for call in output.tool_calls)
                    ):
                        raise BridgeProtocolError(
                            "Provider did not honor the required controller-correction tool.",
                            diagnostics=failure_diagnostics(
                                "required_correction_tool_not_called",
                                "controller_correction",
                                current_calls=output.tool_calls,
                                recorded_tool_names=offered_tool_names,
                            ),
                        )
                    pending_required_tool_names = ()
                    break
                if empty_retry_count >= self.max_empty_response_retries:
                    raise BridgeProtocolError(
                        "Provider repeatedly completed without text or tool calls; "
                        "no invalid tau message was emitted.",
                        diagnostics=failure_diagnostics(
                            "empty_response_limit",
                            "empty_response",
                        ),
                    )
                empty_retry_count += 1
                total_empty_response_retries += 1

            if output.tool_calls:
                unknown_names = {
                    call.tool_name
                    for call in output.tool_calls
                    if call.tool_name not in external_tool_names
                    and call.tool_name not in internal_tool_names
                }
                if unknown_names:
                    names = ", ".join(sorted(unknown_names))
                    raise BridgeProtocolError(f"Provider called tools not offered: {names}.")
                internal_calls = tuple(
                    call for call in output.tool_calls if call.tool_name in internal_tool_names
                )
                external_calls = tuple(
                    call for call in output.tool_calls if call.tool_name in external_tool_names
                )
                is_staged_external_batch = (
                    bool(internal_calls)
                    and bool(external_calls)
                    and output.tool_calls == (*internal_calls, *external_calls)
                    and _internal_allows_staged_external_calls(
                        internal_tool_handler,
                        internal_calls,
                    )
                )
                has_internal_calls = bool(internal_calls)
                if has_internal_calls:
                    internal_round_calls = (
                        internal_calls if is_staged_external_batch else output.tool_calls
                    )
                    if internal_tool_rounds >= self.max_internal_tool_rounds:
                        raise BridgeProtocolError(
                            "Provider exceeded the controller-internal hard round limit.",
                            diagnostics=failure_diagnostics(
                                "internal_hard_round_limit",
                                "internal_tools",
                                current_calls=output.tool_calls,
                                recorded_tool_names=offered_tool_names,
                            ),
                        )
                    progress_before = _internal_progress_token(internal_tool_handler)
                    round_call_fingerprints = tuple(
                        _opaque_tool_call_fingerprint(self._diagnostic_hmac_key, call)
                        for call in internal_round_calls
                    )
                    round_call_stall_signature = tuple(
                        f"call:{fingerprint}" for fingerprint in round_call_fingerprints
                    )
                    if (
                        progress_before is not None
                        and consecutive_stalled_internal_rounds
                        >= self.max_stalled_internal_tool_rounds
                        and round_call_stall_signature == last_stalled_internal_signature
                    ):
                        raise BridgeProtocolError(
                            "Provider exceeded the consecutive equivalent no-progress "
                            "internal round limit.",
                            diagnostics=failure_diagnostics(
                                "internal_no_progress_limit",
                                "internal_tools",
                                current_calls=output.tool_calls,
                                recorded_tool_names=offered_tool_names,
                            ),
                        )
                    _record_provider_call_ids(conversation, internal_round_calls)
                    internal_calls_by_tool.update(
                        call.tool_name
                        for call in internal_round_calls
                        if call.tool_name in internal_tool_names
                    )
                    internal_assistant_message = _model_output_to_cayu_message(output)
                    generated_messages.append(internal_assistant_message)
                    result_messages, internal_results = await _internal_tool_result_messages(
                        internal_round_calls,
                        external_tool_names=external_tool_names,
                        handler=internal_tool_handler,
                    )
                    correction_ids = remediation_lifecycle.observe(
                        internal_results,
                        round_number=internal_tool_rounds + 1,
                    )
                    retention_key = (
                        None
                        if is_staged_external_batch
                        else _internal_transcript_retention_key(
                            internal_tool_handler,
                            internal_round_calls,
                            internal_results,
                        )
                    )
                    current_round_messages = (
                        internal_assistant_message,
                        *result_messages,
                    )
                    if retention_key is not None:
                        superseded = retained_internal_messages.get(retention_key)
                        if superseded is not None:
                            superseded_ids = {id(message) for message in superseded}
                            generated_messages[:] = [
                                message
                                for message in generated_messages
                                if id(message) not in superseded_ids
                            ]
                            internal_compacted_rounds += 1
                        retained_internal_messages[retention_key] = current_round_messages
                    generated_messages.extend(result_messages)
                    conversation.messages = [*base_messages, *generated_messages]
                    internal_tool_rounds += 1
                    current_internal_context = _internal_refreshed_context(internal_tool_handler)
                    if current_internal_context is not None:
                        refreshed_internal_context = CayuMessage.text(
                            MessageRole.SYSTEM,
                            current_internal_context,
                        )
                        internal_context_refreshes += 1
                    progress_after = _internal_progress_token(internal_tool_handler)
                    resolved_remediation = any(result.resolutions for result in internal_results)
                    progress_changed = (
                        True
                        if resolved_remediation
                        else (
                            None
                            if progress_before is None or progress_after is None
                            else progress_after != progress_before
                        )
                    )
                    round_stall_signature = (
                        tuple(f"remediation:{correction_id}" for correction_id in correction_ids)
                        if correction_ids
                        else round_call_stall_signature
                    )
                    if progress_changed is not None:
                        if not progress_changed:
                            internal_stalled_rounds += 1
                            stalled_internal_rounds_since_progress += 1
                            if round_stall_signature == last_stalled_internal_signature:
                                consecutive_stalled_internal_rounds += 1
                            else:
                                consecutive_stalled_internal_rounds = 1
                            last_stalled_internal_signature = round_stall_signature
                        else:
                            internal_progress_events += 1
                            stalled_internal_rounds_since_progress = 0
                            terminal_gate_rounds_since_progress = 0
                            consecutive_stalled_internal_rounds = 0
                            last_stalled_internal_signature = None
                    internal_round_trace.append(
                        InternalRoundDiagnostic(
                            tool_names=tuple(call.tool_name for call in internal_round_calls),
                            call_fingerprints=round_call_fingerprints,
                            progress_changed=progress_changed,
                            error_result_count=sum(result.is_error for result in internal_results),
                            error_codes=tuple(
                                result.diagnostic_code or "unclassified_internal_error"
                                for result in internal_results
                                if result.is_error
                            ),
                        )
                    )
                    if (
                        progress_changed is False
                        and consecutive_stalled_internal_rounds
                        > self.max_stalled_internal_tool_rounds
                    ):
                        raise BridgeProtocolError(
                            "Provider exceeded the consecutive equivalent no-progress "
                            "internal round limit.",
                            diagnostics=failure_diagnostics(
                                "internal_no_progress_limit",
                                "internal_tools",
                                current_calls=output.tool_calls,
                                recorded_tool_names=offered_tool_names,
                            ),
                        )
                    if (
                        progress_changed is False
                        and stalled_internal_rounds_since_progress
                        >= self.max_stalled_internal_tool_rounds_since_progress
                    ):
                        raise BridgeProtocolError(
                            "Provider exhausted the no-progress internal round budget since "
                            "its last verified progress event.",
                            diagnostics=failure_diagnostics(
                                "internal_no_progress_window_limit",
                                "internal_tools",
                                current_calls=output.tool_calls,
                                recorded_tool_names=offered_tool_names,
                            ),
                        )
                    internal_round_succeeded = bool(internal_results) and not any(
                        result.is_error for result in internal_results
                    )
                    if internal_round_succeeded and not is_staged_external_batch:
                        called_tool_names = {call.tool_name for call in internal_round_calls}
                        next_internal_tool_names = {
                            schema["name"]
                            for schema in _internal_tool_schemas(internal_tool_handler)
                        }
                        next_offered_tool_names = external_tool_names | next_internal_tool_names
                        pending_required_tool_names = tuple(
                            name
                            for name in _internal_required_action_tool_names(internal_tool_handler)
                            if name not in called_tool_names and name in next_offered_tool_names
                        )
                    if not is_staged_external_batch:
                        next_provider_call_phase = "internal_tool_result"
                        continue
                    if not internal_round_succeeded:
                        _record_provider_call_ids(conversation, external_calls)
                        generated_messages.extend(
                            _staged_external_failure_result_messages(external_calls)
                        )
                        conversation.messages = [*base_messages, *generated_messages]
                        next_provider_call_phase = "internal_tool_result"
                        continue
                    output = _external_only_model_output(output, external_calls)
                admissions = await _review_external_tool_calls(
                    output.tool_calls,
                    external_tool_gate,
                )
                denied = tuple(admission for admission in admissions if not admission.allowed)
                if denied:
                    external_admission_rejections += len(denied)
                    external_rejection_codes.update(
                        admission.diagnostic_code or "unclassified" for admission in denied
                    )
                    if external_admission_rounds >= self.max_external_admission_rounds:
                        raise BridgeProtocolError(
                            "Provider exceeded the external tool-admission correction limit.",
                            diagnostics=failure_diagnostics(
                                "external_admission_limit",
                                "external_tools",
                                current_calls=output.tool_calls,
                                recorded_tool_names=external_tool_names,
                            ),
                        )
                    _record_provider_call_ids(conversation, output.tool_calls)
                    if not is_staged_external_batch:
                        generated_messages.append(_model_output_to_cayu_message(output))
                    generated_messages.extend(
                        _external_admission_result_messages(output.tool_calls, admissions)
                    )
                    conversation.messages = [*base_messages, *generated_messages]
                    denied_call_ids = {admission.tool_call_id for admission in denied}
                    denied_external_tool_names.update(
                        call.tool_name
                        for call in output.tool_calls
                        if call.tool_call_id in denied_call_ids
                    )
                    pending_required_tool_names = _ordered_unique(
                        name for admission in denied for name in admission.required_tool_names
                    )
                    external_admission_rounds += 1
                    next_provider_call_phase = "external_admission_correction"
                    continue
                if remediation_lifecycle.unresolved_count:
                    raise BridgeProtocolError(
                        "Provider attempted an environment tool while an internal remediation "
                        "remained unresolved.",
                        diagnostics=failure_diagnostics(
                            "unresolved_internal_remediation",
                            "external_tools",
                            current_calls=output.tool_calls,
                            recorded_tool_names=external_tool_names,
                        ),
                    )
                if is_staged_external_batch:
                    staged_external_batches += 1
                    staged_external_calls += len(output.tool_calls)
                _record_call_names(conversation, output.tool_calls)
                assistant_cayu = _model_output_to_cayu_message(output)
                generated_messages.append(assistant_cayu)
                # Private controller/search rounds are needed only while producing this
                # outward turn. Durable Cayu state and the public tau transcript carry
                # their authoritative effects across turns; retaining the private trace
                # here causes quadratic prompt growth and stale-reasoning interference.
                conversation.messages = [*base_messages, assistant_cayu]
                result = _model_output_to_tau_message(
                    output,
                    continuation_count,
                    total_empty_response_retries,
                    internal_tool_rounds,
                    external_admission_rounds,
                    external_admission_rejections,
                    terminal_gate_rounds,
                    terminal_gate_rejections,
                    terminal_gate_rounds_since_progress,
                    internal_progress_events,
                    internal_stalled_rounds,
                    stalled_internal_rounds_since_progress,
                    internal_compacted_rounds,
                    internal_context_refreshes,
                    staged_external_batches,
                    staged_external_calls,
                    self.max_internal_tool_rounds,
                    self.max_stalled_internal_tool_rounds,
                    self.max_stalled_internal_tool_rounds_since_progress,
                    external_rejection_codes=external_rejection_codes,
                    internal_calls_by_tool=internal_calls_by_tool,
                    terminal_rejection_codes=terminal_rejection_codes,
                    remediation_lifecycle=remediation_lifecycle.snapshot(exhausted=False),
                    usage=_aggregate_tau_usage(outputs, self.provider),
                    provider_usages=_provider_usages(outputs, self.provider),
                    provider_call_trace=provider_call_trace,
                    attempt_id=conversation.attempt_id,
                    correlation_tag=conversation.correlation_tag,
                )
                conversation.active_failure_diagnostics = None
                return result

            if output.completion.end_turn is False:
                if continuation_count >= self.max_internal_continuations:
                    raise BridgeProtocolError(
                        "Provider exceeded the internal continuation limit without ending its turn.",
                        diagnostics=failure_diagnostics(
                            "internal_continuation_limit",
                            "continuation",
                        ),
                    )
                assistant_cayu = _model_output_to_cayu_message(output)
                generated_messages.append(assistant_cayu)
                continuation_count += 1
                next_provider_call_phase = "provider_continuation"
                continue

            terminal_admission = await _review_terminal_output(
                output.text,
                terminal_output_gate,
            )
            if not terminal_admission.allowed:
                if terminal_gate_rounds_since_progress >= self.max_terminal_gate_rounds:
                    raise BridgeProtocolError(
                        "Provider exceeded the terminal-readiness correction limit.",
                        diagnostics=failure_diagnostics(
                            "terminal_readiness_limit",
                            "terminal_text",
                            current_output_fingerprint=_opaque_fingerprint(
                                self._diagnostic_hmac_key,
                                {"terminal_output": output.text},
                            ),
                        ),
                    )
                terminal_rejection_codes[terminal_admission.correction_code or "unclassified"] += 1
                terminal_correction_fingerprints.append(
                    _opaque_fingerprint(
                        self._diagnostic_hmac_key,
                        {"terminal_output": output.text},
                    )
                )
                generated_messages.append(
                    CayuMessage.text(
                        MessageRole.SYSTEM,
                        "The preceding customer-facing draft was not emitted because the "
                        "application-owned readiness check rejected it. Continue the same turn "
                        "by synchronizing the plan or issuing its next required tool call. "
                        f"Reason: {terminal_admission.reason}",
                    )
                )
                conversation.messages = [*base_messages, *generated_messages]
                nonrejected_corrections = tuple(
                    name
                    for name in terminal_admission.required_tool_names
                    if name not in denied_external_tool_names
                )
                pending_required_tool_names = (
                    nonrejected_corrections
                    if nonrejected_corrections
                    else terminal_admission.required_tool_names
                )
                terminal_gate_rounds += 1
                terminal_gate_rejections += 1
                terminal_gate_rounds_since_progress += 1
                next_provider_call_phase = "terminal_readiness_correction"
                continue

            assistant_cayu = _model_output_to_cayu_message(output)
            if remediation_lifecycle.unresolved_count:
                raise BridgeProtocolError(
                    "Provider attempted terminal output while an internal remediation remained "
                    "unresolved.",
                    diagnostics=failure_diagnostics(
                        "unresolved_internal_remediation",
                        "terminal_text",
                        current_output_fingerprint=_opaque_fingerprint(
                            self._diagnostic_hmac_key,
                            {"terminal_output": output.text},
                        ),
                    ),
                )
            generated_messages.append(assistant_cayu)
            conversation.messages = [*base_messages, assistant_cayu]
            result = _model_output_to_tau_message(
                output,
                continuation_count,
                total_empty_response_retries,
                internal_tool_rounds,
                external_admission_rounds,
                external_admission_rejections,
                terminal_gate_rounds,
                terminal_gate_rejections,
                terminal_gate_rounds_since_progress,
                internal_progress_events,
                internal_stalled_rounds,
                stalled_internal_rounds_since_progress,
                internal_compacted_rounds,
                internal_context_refreshes,
                staged_external_batches,
                staged_external_calls,
                self.max_internal_tool_rounds,
                self.max_stalled_internal_tool_rounds,
                self.max_stalled_internal_tool_rounds_since_progress,
                external_rejection_codes=external_rejection_codes,
                internal_calls_by_tool=internal_calls_by_tool,
                terminal_rejection_codes=terminal_rejection_codes,
                remediation_lifecycle=remediation_lifecycle.snapshot(exhausted=False),
                usage=_aggregate_tau_usage(outputs, self.provider),
                provider_usages=_provider_usages(outputs, self.provider),
                provider_call_trace=provider_call_trace,
                attempt_id=conversation.attempt_id,
                correlation_tag=conversation.correlation_tag,
            )
            conversation.active_failure_diagnostics = None
            return result

    def _failure_diagnostics(
        self,
        *,
        conversation: CayuTauConversation,
        failure_code: str,
        current_output_kind: str,
        current_output_fingerprint: str | None,
        outputs: Sequence[_ModelOutput],
        round_trace: Sequence[InternalRoundDiagnostic],
        terminal_correction_fingerprints: Sequence[str],
        current_calls: Sequence[ToolCallPart],
        recorded_tool_names: set[str],
        internal_tool_rounds: int,
        internal_progress_events: int,
        internal_stalled_rounds: int,
        stalled_internal_rounds_since_progress: int,
        internal_diagnostic_counts: Mapping[str, int],
        remediation_lifecycle: RemediationLifecycleDiagnostic,
        consecutive_stalled_internal_rounds: int,
        internal_continuations: int,
        external_admission_rounds: int,
        external_admission_rejections: int,
        external_rejection_codes: Mapping[str, int],
        terminal_gate_rounds: int,
        terminal_gate_rejections: int,
        terminal_gate_rounds_since_progress: int,
        terminal_rejection_codes: Mapping[str, int],
        empty_response_retries: int,
    ) -> BridgeFailureDiagnostics:
        recorded_calls = tuple(
            call for call in current_calls if call.tool_name in recorded_tool_names
        )
        return BridgeFailureDiagnostics(
            attempt_id=conversation.attempt_id,
            correlation_tag=conversation.correlation_tag,
            failure_code=failure_code,
            provider_call_count=len(outputs),
            internal_continuations=internal_continuations,
            internal_tool_rounds=internal_tool_rounds,
            internal_progress_events=internal_progress_events,
            internal_stalled_rounds=internal_stalled_rounds,
            stalled_internal_rounds_since_progress=(stalled_internal_rounds_since_progress),
            internal_diagnostic_counts=dict(internal_diagnostic_counts),
            remediation_lifecycle=remediation_lifecycle,
            consecutive_stalled_internal_rounds=consecutive_stalled_internal_rounds,
            internal_tool_round_limit=self.max_internal_tool_rounds,
            stalled_internal_tool_round_limit=self.max_stalled_internal_tool_rounds,
            stalled_internal_tool_round_limit_since_progress=(
                self.max_stalled_internal_tool_rounds_since_progress
            ),
            external_admission_rounds=external_admission_rounds,
            external_admission_rejections=external_admission_rejections,
            external_rejection_codes=dict(sorted(external_rejection_codes.items())),
            terminal_gate_rounds=terminal_gate_rounds,
            terminal_gate_rejections=terminal_gate_rejections,
            terminal_gate_rounds_since_progress=terminal_gate_rounds_since_progress,
            terminal_rejection_codes=dict(sorted(terminal_rejection_codes.items())),
            empty_response_retries=empty_response_retries,
            round_trace=tuple(round_trace),
            terminal_correction_fingerprints=tuple(terminal_correction_fingerprints),
            current_output_kind=current_output_kind,
            current_output_fingerprint=current_output_fingerprint,
            current_tool_names=tuple(call.tool_name for call in recorded_calls),
            current_call_fingerprints=tuple(
                _opaque_tool_call_fingerprint(self._diagnostic_hmac_key, call)
                for call in recorded_calls
            ),
        )

    async def _collect_output(self, request: ModelRequest) -> _ModelOutput:
        text_parts: list[str] = []
        tool_calls: list[ToolCallPart] = []
        completed_event = None

        async for event in self.provider.runtime_stream(request):
            if event.type is ModelStreamEventType.TEXT_DELTA:
                text_parts.append(event.delta)
            elif event.type is ModelStreamEventType.TOOL_CALL:
                call_id = event.payload.get("id")
                name = event.payload.get("name")
                arguments = event.payload.get("arguments")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise BridgeProtocolError("Provider tool calls require a nonblank id.")
                if not isinstance(name, str) or not name.strip():
                    raise BridgeProtocolError("Provider tool calls require a nonblank name.")
                if type(arguments) is not dict:
                    raise BridgeProtocolError("Provider tool-call arguments must be an object.")
                if any(existing.tool_call_id == call_id for existing in tool_calls):
                    raise BridgeProtocolError(f"Provider repeated tool-call id {call_id!r}.")
                normalized_arguments = _normalize_tool_arguments(name, arguments)
                tool_calls.append(
                    ToolCallPart(
                        tool_call_id=call_id,
                        tool_name=name,
                        arguments=normalized_arguments,
                    )
                )
            elif event.type is ModelStreamEventType.ERROR:
                detail = event.payload.get("error")
                message = detail if isinstance(detail, str) and detail.strip() else "unknown error"
                raise BridgeProtocolError(f"Cayu provider failed: {message}")
            elif event.type is ModelStreamEventType.COMPLETED:
                if completed_event is not None:
                    raise BridgeProtocolError("Provider emitted more than one completion event.")
                completed_event = event

        if completed_event is None or completed_event.completion is None:
            raise BridgeProtocolError("Provider stream ended without completion metadata.")
        raw_state = completed_event.payload.get("provider_state", [])
        provider_state = _provider_state_parts(raw_state, suppress_messages=bool(tool_calls))
        usage = completed_event.payload.get("usage")
        if usage is not None and not isinstance(usage, Mapping):
            raise BridgeProtocolError("Provider completion usage must be an object or null.")
        model = completed_event.payload.get("model")
        if model is not None and not isinstance(model, str):
            raise BridgeProtocolError("Provider completion model must be a string or null.")
        return _ModelOutput(
            text="".join(text_parts),
            tool_calls=tuple(tool_calls),
            provider_state=provider_state,
            completion=completed_event.completion,
            usage=(
                None
                if usage is None
                else MappingProxyType(_copy_json_object(usage, "provider completion usage"))
            ),
            model=model,
        )

    async def aclose(self) -> None:
        close = getattr(self.provider, "aclose", None)
        if close is not None:
            await close()


class CayuModelGateway:
    """Thread-safe synchronous facade for tau2's synchronous agent contract."""

    def __init__(
        self,
        bridge: CayuTauBridge,
        *,
        completion_timeout_s: float = DEFAULT_COMPLETION_TIMEOUT_SECONDS,
        attempt_diagnostics_recorder: AttemptDiagnosticsRecorder | None = None,
    ) -> None:
        if not isinstance(bridge, CayuTauBridge):
            raise TypeError("bridge must be a CayuTauBridge.")
        if (
            type(completion_timeout_s) not in {int, float}
            or isinstance(completion_timeout_s, bool)
            or not math.isfinite(completion_timeout_s)
            or completion_timeout_s <= 0
        ):
            raise ValueError("completion_timeout_s must be a positive number.")
        self.bridge = bridge
        self.completion_timeout_s = float(completion_timeout_s)
        if attempt_diagnostics_recorder is not None and not isinstance(
            attempt_diagnostics_recorder, AttemptDiagnosticsRecorder
        ):
            raise TypeError(
                "attempt_diagnostics_recorder must be AttemptDiagnosticsRecorder or null."
            )
        self.attempt_diagnostics_recorder = attempt_diagnostics_recorder
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._state_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run_loop,
            name="cayu-tau3-model-gateway",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def generate(
        self,
        conversation: CayuTauConversation,
        tools: Sequence[Tool],
        *,
        transient_context: str | None = None,
        context_provider: TurnContextProvider | None = None,
        internal_tool_handler: InternalToolHandler | None = None,
        external_tool_gate: ExternalToolCallGate | None = None,
        terminal_output_gate: TerminalOutputGate | None = None,
    ) -> AssistantMessage:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Cayu model gateway is closed.")
        future = asyncio.run_coroutine_threadsafe(
            self._generate(
                conversation,
                tools,
                transient_context=transient_context,
                context_provider=context_provider,
                internal_tool_handler=internal_tool_handler,
                external_tool_gate=external_tool_gate,
                terminal_output_gate=terminal_output_gate,
            ),
            self._loop,
        )
        try:
            return future.result(timeout=self.completion_timeout_s)
        except BridgeProtocolError as exc:
            diagnostics = exc.diagnostics
            if diagnostics is None and conversation.active_failure_diagnostics is not None:
                diagnostics = replace(
                    conversation.active_failure_diagnostics,
                    failure_code="provider_stream_protocol_failure",
                )
            if self.attempt_diagnostics_recorder is not None and diagnostics is not None:
                try:
                    self.attempt_diagnostics_recorder.record(diagnostics)
                except OSError as diagnostic_error:
                    exc.add_note(
                        "Attempt diagnostics could not be persisted: "
                        f"{type(diagnostic_error).__name__}."
                    )
            raise
        except ModelStreamDeadlineError as exc:
            diagnostics = conversation.active_failure_diagnostics
            if self.attempt_diagnostics_recorder is not None and diagnostics is not None:
                try:
                    self.attempt_diagnostics_recorder.record(
                        replace(
                            diagnostics,
                            failure_code=(exc.error_code or "provider_stream_deadline_timeout"),
                        )
                    )
                except OSError as diagnostic_error:
                    exc.add_note(
                        "Attempt diagnostics could not be persisted: "
                        f"{type(diagnostic_error).__name__}."
                    )
            raise
        except concurrent.futures.TimeoutError:
            if not future.cancel():
                # The coroutine completed between the timeout and cancellation
                # attempt. Return that authoritative result instead of claiming a
                # timeout after the conversation may already have been committed.
                return future.result()
            try:
                future.result(timeout=5.0)
            except concurrent.futures.CancelledError:
                pass
            except concurrent.futures.TimeoutError:
                raise RuntimeError(
                    "Timed-out Cayu model generation did not acknowledge cancellation."
                ) from None
            timeout_error = TimeoutError(
                f"Cayu model generation exceeded {self.completion_timeout_s:g} seconds."
            )
            diagnostics = conversation.active_failure_diagnostics
            if self.attempt_diagnostics_recorder is not None and diagnostics is not None:
                try:
                    self.attempt_diagnostics_recorder.record(diagnostics)
                except OSError as diagnostic_error:
                    timeout_error.add_note(
                        "Attempt diagnostics could not be persisted: "
                        f"{type(diagnostic_error).__name__}."
                    )
            raise timeout_error from None

    async def _generate(
        self,
        conversation: CayuTauConversation,
        tools: Sequence[Tool],
        *,
        transient_context: str | None,
        context_provider: TurnContextProvider | None,
        internal_tool_handler: InternalToolHandler | None,
        external_tool_gate: ExternalToolCallGate | None,
        terminal_output_gate: TerminalOutputGate | None,
    ) -> AssistantMessage:
        contexts: list[str] = []
        if transient_context is not None:
            if not isinstance(transient_context, str) or not transient_context.strip():
                raise ValueError("Transient turn context must be nonblank text or null.")
            contexts.append(transient_context.strip())
        if context_provider is not None:
            prepared = await context_provider.prepare_context()
            if prepared is not None:
                if not isinstance(prepared, str) or not prepared.strip():
                    raise ValueError("Prepared turn context must be nonblank text or null.")
                contexts.append(prepared.strip())
        return await self.bridge.generate(
            conversation,
            tools,
            transient_context="\n\n".join(contexts) or None,
            internal_tool_handler=internal_tool_handler,
            external_tool_gate=external_tool_gate,
            terminal_output_gate=terminal_output_gate,
        )

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        future = asyncio.run_coroutine_threadsafe(self.bridge.aclose(), self._loop)
        close_failure: BaseException | None = None
        try:
            future.result(timeout=min(self.completion_timeout_s, 30.0))
        except BaseException as exc:
            close_failure = exc
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            raise RuntimeError("Cayu model gateway event-loop thread did not stop.")
        self._loop.close()
        if close_failure is not None:
            raise close_failure

    def __enter__(self) -> CayuModelGateway:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _require_conversation(value: object) -> CayuTauConversation:
    if not isinstance(value, CayuTauConversation):
        raise TypeError("conversation must be a CayuTauConversation.")
    return value


def _require_text(value: str | None, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BridgeProtocolError(f"{field_name} must be nonblank text.")
    return value


def _copy_json_object(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    try:
        copied = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain JSON-compatible values.") from exc
    if type(copied) is not dict:
        raise ValueError(f"{field_name} must be an object.")
    return copied


def _required_tool_options(
    base_options: Mapping[str, Any],
    tool_name: str | None,
    *,
    max_calls: int = 1,
) -> dict[str, Any]:
    """Require one host-selected tool, allowing a bounded homogeneous batch."""

    options = _copy_json_object(base_options, "options")
    if type(max_calls) is not int or not 1 <= max_calls <= _MAX_REQUIRED_ACTION_BATCH_SIZE:
        raise BridgeProtocolError("Required tool-call batch size is invalid.")
    if tool_name is None:
        return options
    if not _CORRECTION_TOOL_NAME.fullmatch(tool_name):
        raise BridgeProtocolError("Terminal correction selected an invalid tool name.")
    raw_openai = options.get("openai")
    if raw_openai is None:
        openai_options: dict[str, Any] = {}
    elif type(raw_openai) is dict:
        openai_options = raw_openai
    else:
        raise BridgeProtocolError("options.openai must be an object.")
    openai_options.update(
        {
            "parallel_tool_calls": max_calls > 1,
            "tool_choice": {"type": "function", "name": tool_name},
        }
    )
    options["openai"] = openai_options
    return options


def _opaque_fingerprint(key: bytes, value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hmac.digest(key, encoded, "sha256").hex()


def _opaque_tool_call_fingerprint(key: bytes, call: ToolCallPart) -> str:
    return _opaque_fingerprint(
        key,
        {"tool_name": call.tool_name, "arguments": call.arguments},
    )


def _record_call_names(
    conversation: CayuTauConversation,
    calls: Sequence[ToolCallPart],
) -> None:
    _record_provider_call_ids(conversation, calls)
    additions = {call.tool_call_id: call.tool_name for call in calls}
    conversation.tool_names_by_call_id.update(additions)


def _record_provider_call_ids(
    conversation: CayuTauConversation,
    calls: Sequence[ToolCallPart],
) -> None:
    call_ids = [call.tool_call_id for call in calls]
    if len(set(call_ids)) != len(call_ids):
        raise BridgeProtocolError("Assistant repeated a tool-call id within one message.")
    conflict = next(
        (call_id for call_id in call_ids if call_id in conversation.provider_call_ids),
        None,
    )
    if conflict is not None:
        raise BridgeProtocolError(f"Assistant repeated tool-call id {conflict!r}.")
    conversation.provider_call_ids.update(call_ids)


def _append_tau_tool_results(
    conversation: CayuTauConversation,
    messages: Sequence[ToolMessage],
) -> None:
    """Validate a result batch completely before mutating the transcript."""

    ids = [message.id for message in messages]
    if len(set(ids)) != len(ids):
        raise BridgeProtocolError("A tau tool-result batch repeated a call id.")
    converted: list[CayuMessage] = []
    for message in messages:
        if message.requestor != "assistant":
            raise BridgeProtocolError(
                "Only assistant-requested tool results belong in agent history."
            )
        tool_name = conversation.tool_names_by_call_id.get(message.id)
        if tool_name is None:
            raise BridgeProtocolError(
                f"Tool result references unknown assistant call id {message.id!r}."
            )
        if message.id in conversation.resolved_call_ids:
            raise BridgeProtocolError(
                f"Tool result repeated resolved assistant call id {message.id!r}."
            )
        converted.append(
            CayuMessage.tool_result(
                tool_call_id=message.id,
                tool_name=tool_name,
                content="" if message.content is None else message.content,
                is_error=message.error,
            )
        )
    conversation.messages.extend(converted)
    conversation.resolved_call_ids.update(ids)


def _tau_assistant_to_cayu(message: AssistantMessage) -> CayuMessage:
    has_text = message.has_text_content()
    has_calls = message.is_tool_call()
    if has_text == has_calls:
        raise BridgeProtocolError(
            "A tau assistant message must contain text or tool calls, not both."
        )
    if has_text:
        return CayuMessage.text(
            MessageRole.ASSISTANT, _require_text(message.content, "assistant content")
        )
    calls = message.tool_calls or []
    if not calls:
        raise BridgeProtocolError("A tau assistant tool-call message cannot be empty.")
    converted: list[ToolCallPart] = []
    for call in calls:
        if call.requestor != "assistant":
            raise BridgeProtocolError("Agent history cannot contain a user-requested tool call.")
        if not call.id.strip():
            raise BridgeProtocolError("Assistant tool calls require a nonblank id.")
        converted.append(
            ToolCallPart(
                tool_call_id=call.id,
                tool_name=call.name,
                arguments=call.arguments,
            )
        )
    return CayuMessage.tool_call(calls=converted)


def _cayu_tool_schemas(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, Tool):
            raise TypeError("tools must contain tau2 Tool instances.")
        raw = tool.openai_schema
        function = raw.get("function") if isinstance(raw, Mapping) else None
        if not isinstance(function, Mapping):
            raise BridgeProtocolError(f"Tool {tool.name!r} has no OpenAI function schema.")
        name = function.get("name")
        description = function.get("description", "")
        parameters = function.get("parameters", {})
        if not isinstance(name, str) or not name.strip():
            raise BridgeProtocolError("Tool schema name must be nonblank.")
        if name in names:
            raise BridgeProtocolError(f"Duplicate tool schema name {name!r}.")
        if not isinstance(description, str):
            raise BridgeProtocolError(f"Tool {name!r} description must be text.")
        if type(parameters) is not dict:
            raise BridgeProtocolError(f"Tool {name!r} parameters must be an object schema.")
        names.add(name)
        schemas.append(
            {
                "name": name,
                "description": description,
                "input_schema": parameters,
            }
        )
    return schemas


def _internal_tool_schemas(
    handler: InternalToolHandler | None,
) -> list[dict[str, Any]]:
    if handler is None:
        return []
    raw_schemas = handler.tool_schemas()
    if not isinstance(raw_schemas, Sequence) or isinstance(raw_schemas, (str, bytes)):
        raise TypeError("Internal tool schemas must be a sequence.")
    schemas: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in raw_schemas:
        if not isinstance(raw, Mapping):
            raise TypeError("Each internal tool schema must be a mapping.")
        copied = _copy_json_object(raw, "internal tool schema")
        if set(copied) != {"name", "description", "input_schema"}:
            raise BridgeProtocolError(
                "Internal tool schemas require exactly name, description, and input_schema."
            )
        name = copied["name"]
        description = copied["description"]
        input_schema = copied["input_schema"]
        if not isinstance(name, str) or not name.strip():
            raise BridgeProtocolError("Internal tool schema name must be nonblank.")
        if name in names:
            raise BridgeProtocolError(f"Duplicate internal tool schema name {name!r}.")
        if not isinstance(description, str) or not description.strip():
            raise BridgeProtocolError(f"Internal tool {name!r} description must be nonblank text.")
        if type(input_schema) is not dict:
            raise BridgeProtocolError(
                f"Internal tool {name!r} input_schema must be an object schema."
            )
        names.add(name)
        schemas.append(copied)
    return schemas


async def _internal_tool_result_messages(
    calls: Sequence[ToolCallPart],
    *,
    external_tool_names: set[str],
    handler: InternalToolHandler | None,
) -> tuple[list[CayuMessage], tuple[InternalToolResult, ...]]:
    if handler is None:
        raise AssertionError("Internal calls require an internal tool handler.")
    messages: list[CayuMessage] = []
    results: list[InternalToolResult] = []
    for call in calls:
        if call.tool_name in external_tool_names:
            result = InternalToolResult(
                content=(
                    "This tau environment tool was not executed because it was batched "
                    "with a controller-only tool. Reissue the environment tool in a new "
                    "call after applying the controller result."
                ),
                is_error=True,
                diagnostic_code="mixed_external_internal_batch",
            )
        elif not handler.handles(call.tool_name):
            result = InternalToolResult(
                content="The controller does not handle this advertised tool name.",
                is_error=True,
                diagnostic_code="unhandled_internal_tool",
            )
        else:
            try:
                invoked = handler.invoke(call.tool_name, call.arguments)
                result = await invoked if inspect.isawaitable(invoked) else invoked
            except Exception as exc:
                result = InternalToolResult(
                    content=f"Controller tool failed safely ({type(exc).__name__}).",
                    is_error=True,
                    diagnostic_code="internal_tool_exception",
                )
            if not isinstance(result, InternalToolResult):
                result = InternalToolResult(
                    content="Controller returned an invalid internal result.",
                    is_error=True,
                    diagnostic_code="invalid_internal_result",
                )
            elif not isinstance(result.content, str) or not result.content.strip():
                result = InternalToolResult(
                    content="Controller returned an empty internal result.",
                    is_error=True,
                    diagnostic_code="empty_internal_result",
                )
        model_content, structured = internal_tool_result_projection(result)
        results.append(result)
        messages.append(
            CayuMessage.tool_result(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                content=model_content,
                structured=structured,
                is_error=result.is_error,
            )
        )
    return messages, tuple(results)


def internal_tool_result_projection(
    result: InternalToolResult,
) -> tuple[str, dict[str, Any] | None]:
    """Project one typed correction into both stored and model-visible forms."""

    structured = dict(result.structured or {})
    if result.remediation is not None:
        structured["remediation"] = result.remediation.as_dict()
    if not structured:
        return result.content, None
    model_payload = {"message": result.content, **structured}
    return _canonical_json_bytes(model_payload).decode("utf-8"), structured


async def _review_external_tool_calls(
    calls: Sequence[ToolCallPart],
    gate: ExternalToolCallGate | None,
) -> tuple[ExternalToolAdmission, ...]:
    if gate is None:
        return tuple(
            ExternalToolAdmission(tool_call_id=call.tool_call_id, allowed=True) for call in calls
        )
    try:
        reviewed = gate.review(calls)
        raw_admissions = await reviewed if inspect.isawaitable(reviewed) else reviewed
    except Exception as exc:
        raise BridgeProtocolError(
            f"External tool admission failed closed ({type(exc).__name__})."
        ) from exc
    if not isinstance(raw_admissions, Sequence) or isinstance(raw_admissions, (str, bytes)):
        raise BridgeProtocolError("External tool admission must return a decision sequence.")
    admissions = tuple(raw_admissions)
    if len(admissions) != len(calls):
        raise BridgeProtocolError(
            "External tool admission must return exactly one decision per call."
        )
    for call, admission in zip(calls, admissions, strict=True):
        if not isinstance(admission, ExternalToolAdmission):
            raise BridgeProtocolError(
                "External tool admission returned an invalid decision object."
            )
        if admission.tool_call_id != call.tool_call_id:
            raise BridgeProtocolError(
                "External tool admission decisions must preserve call order and identity."
            )
    return admissions


async def _review_terminal_output(
    text: str,
    gate: TerminalOutputGate | None,
) -> TerminalOutputAdmission:
    if gate is None:
        return TerminalOutputAdmission(allowed=True)
    try:
        reviewed = gate.review_terminal_output(text)
        admission = await reviewed if inspect.isawaitable(reviewed) else reviewed
    except Exception as exc:
        raise BridgeProtocolError(
            f"Terminal output admission failed closed ({type(exc).__name__})."
        ) from exc
    if not isinstance(admission, TerminalOutputAdmission):
        raise BridgeProtocolError("Terminal output admission returned an invalid decision object.")
    return admission


def _combine_external_admissions(
    current: ExternalToolAdmission,
    additional: ExternalToolAdmission,
) -> ExternalToolAdmission:
    if current.tool_call_id != additional.tool_call_id:
        raise BridgeProtocolError("Cannot combine admissions for different tool calls.")
    if current.allowed and additional.allowed:
        return current
    reasons = [
        admission.reason
        for admission in (current, additional)
        if not admission.allowed and admission.reason is not None
    ]
    return ExternalToolAdmission(
        tool_call_id=current.tool_call_id,
        allowed=False,
        reason="; ".join(reasons),
        required_tool_names=_ordered_unique(
            (*current.required_tool_names, *additional.required_tool_names)
        ),
        diagnostic_code=(
            current.diagnostic_code
            if not current.allowed and current.diagnostic_code is not None
            else additional.diagnostic_code
        ),
    )


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    """Return stable first-seen values for bounded correction guidance."""

    return tuple(dict.fromkeys(values))


def _normalize_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize benchmark wrapper arguments without changing their meaning."""

    normalized = _normalize_nested_tool_arguments(arguments)
    if tool_name == _DISCOVERABLE_AGENT_CALL_TOOL:
        return _canonicalize_discoverable_arguments(normalized)
    if tool_name == _DISCOVERABLE_USER_DELIVERY_TOOL:
        # Delivery exposes a capability; its optional JSON only pre-populates a
        # later customer call. Public tau references compare this call exactly
        # and leave that later input to the customer-owned workflow operation.
        normalized.pop("arguments", None)
    return normalized


def _normalize_nested_tool_arguments(value: Any) -> Any:
    """Copy provider arguments and normalize explicitly named nested tool calls."""

    if type(value) is list:
        return [_normalize_nested_tool_arguments(item) for item in value]
    if type(value) is not dict:
        return value
    normalized = {key: _normalize_nested_tool_arguments(item) for key, item in value.items()}
    nested_name = normalized.get("tool_name")
    nested_arguments = normalized.get("arguments")
    if isinstance(nested_name, str) and type(nested_arguments) is dict:
        normalized["arguments"] = _normalize_tool_arguments(nested_name, nested_arguments)
    return normalized


def _canonicalize_discoverable_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Use the benchmark's standard JSON spelling for wrapper arguments."""

    raw_arguments = arguments.get("arguments")
    if not isinstance(raw_arguments, str):
        return arguments
    try:
        parsed = json.loads(raw_arguments)
        if type(parsed) is dict:
            agent_tool_name = arguments.get("agent_tool_name")
            fixed_currency_fields = (
                _FIXED_CURRENCY_ARGUMENT_FIELDS.get(agent_tool_name, frozenset())
                if isinstance(agent_tool_name, str)
                else frozenset()
            )
            arguments["arguments"] = _render_discoverable_argument_object(
                parsed,
                fixed_currency_fields=fixed_currency_fields,
            )
    except (TypeError, ValueError):
        pass
    return arguments


def _render_discoverable_argument_object(
    arguments: Mapping[str, Any],
    *,
    fixed_currency_fields: frozenset[str],
) -> str:
    rendered: list[str] = []
    for key, value in arguments.items():
        encoded_value = (
            _fixed_currency_number(value)
            if key in fixed_currency_fields
            else json.dumps(value, allow_nan=False)
        )
        rendered.append(f"{json.dumps(key)}: {encoded_value}")
    return "{" + ", ".join(rendered) + "}"


def _fixed_currency_number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return json.dumps(value, allow_nan=False)
    rendered = format(Decimal(str(value)), "f")
    if "." not in rendered:
        return f"{rendered}.00"
    integer, fraction = rendered.split(".", 1)
    return f"{integer}.{fraction.ljust(2, '0')}"


def _internal_progress_token(handler: InternalToolHandler | None) -> str | None:
    if handler is None:
        return None
    progress_token = getattr(handler, "progress_token", None)
    if progress_token is None:
        return None
    if not callable(progress_token):
        raise BridgeProtocolError("Internal progress_token must be callable.")
    try:
        value = progress_token()
    except Exception as exc:
        raise BridgeProtocolError(
            f"Internal progress inspection failed closed ({type(exc).__name__})."
        ) from exc
    if not isinstance(value, str) or not value.strip():
        raise BridgeProtocolError("Internal progress_token must return nonblank text.")
    return value


def _internal_required_action_tool_names(
    handler: InternalToolHandler | None,
) -> tuple[str, ...]:
    """Read validated, state-derived continuation advice from an internal handler."""

    if handler is None:
        return ()
    resolver = getattr(handler, "required_action_tool_names", None)
    if resolver is None:
        return ()
    if not callable(resolver):
        raise BridgeProtocolError("Internal required_action_tool_names must be callable.")
    try:
        value = resolver()
    except Exception as exc:
        raise BridgeProtocolError(
            f"Internal action-tool inspection failed closed ({type(exc).__name__})."
        ) from exc
    if (
        type(value) is not tuple
        or len(set(value)) != len(value)
        or any(
            not isinstance(name, str) or not _CORRECTION_TOOL_NAME.fullmatch(name) for name in value
        )
    ):
        raise BridgeProtocolError(
            "Internal required_action_tool_names must return unique valid names."
        )
    return value


def _internal_required_action_batch_size(
    handler: InternalToolHandler | None,
    name: str | None,
) -> int:
    """Read a bounded homogeneous batch size for one required action."""

    if handler is None or name is None:
        return 1
    resolver = getattr(handler, "required_action_batch_size", None)
    if resolver is None:
        return 1
    if not callable(resolver):
        raise BridgeProtocolError("Internal required_action_batch_size must be callable.")
    try:
        value = resolver(name)
    except Exception as exc:
        raise BridgeProtocolError(
            f"Internal action batch-size inspection failed closed ({type(exc).__name__})."
        ) from exc
    if type(value) is not int or not 1 <= value <= _MAX_REQUIRED_ACTION_BATCH_SIZE:
        raise BridgeProtocolError(
            "Internal required_action_batch_size must return a bounded positive integer."
        )
    return value


def _internal_refreshed_context(handler: InternalToolHandler | None) -> str | None:
    """Read current private controller context after an internal state transition."""

    if handler is None:
        return None
    renderer = getattr(handler, "render_internal_context", None)
    if renderer is None:
        return None
    if not callable(renderer):
        raise BridgeProtocolError("Internal render_internal_context must be callable.")
    try:
        context = renderer()
    except Exception as exc:
        raise BridgeProtocolError(
            f"Internal context refresh failed closed ({type(exc).__name__})."
        ) from exc
    if context is None:
        return None
    if not isinstance(context, str) or not context.strip():
        raise BridgeProtocolError(
            "Internal render_internal_context must return nonblank text or null."
        )
    return context.strip()


def _internal_allows_staged_external_calls(
    handler: InternalToolHandler | None,
    calls: Sequence[ToolCallPart],
) -> bool:
    """Ask the private controller whether one ordered prefix can be committed first."""

    if handler is None or not calls:
        return False
    resolver = getattr(handler, "allows_staged_external_calls", None)
    if resolver is None:
        return False
    if not callable(resolver):
        raise BridgeProtocolError("Internal allows_staged_external_calls must be callable.")
    names = tuple(call.tool_name for call in calls)
    try:
        allowed = resolver(names)
    except Exception as exc:
        raise BridgeProtocolError(
            f"Internal external-staging inspection failed closed ({type(exc).__name__})."
        ) from exc
    if type(allowed) is not bool:
        raise BridgeProtocolError("Internal allows_staged_external_calls must return a boolean.")
    return allowed


def _internal_transcript_retention_key(
    handler: InternalToolHandler | None,
    calls: Sequence[ToolCallPart],
    results: Sequence[InternalToolResult],
) -> str | None:
    """Resolve a common supersession key for one entirely successful private round."""

    if handler is None or len(calls) != len(results) or any(result.is_error for result in results):
        return None
    resolver = getattr(handler, "transcript_retention_key", None)
    if resolver is None:
        return None
    if not callable(resolver):
        raise BridgeProtocolError("Internal transcript_retention_key must be callable.")
    try:
        keys = tuple(resolver(call.tool_name) for call in calls)
    except Exception as exc:
        raise BridgeProtocolError(
            f"Internal transcript retention inspection failed closed ({type(exc).__name__})."
        ) from exc
    if not keys or any(key is None for key in keys):
        return None
    if any(
        not isinstance(key, str) or not _INTERNAL_REMEDIATION_TOKEN.fullmatch(key) for key in keys
    ):
        raise BridgeProtocolError("Internal transcript retention key is invalid.")
    return keys[0] if len(set(keys)) == 1 else None


def _internal_diagnostic_counts(handler: InternalToolHandler | None) -> dict[str, int]:
    if handler is None:
        return {}
    diagnostic_counts = getattr(handler, "diagnostic_counts", None)
    if diagnostic_counts is None:
        return {}
    if not callable(diagnostic_counts):
        raise BridgeProtocolError("Internal diagnostic_counts must be callable.")
    try:
        value = diagnostic_counts()
    except Exception as exc:
        raise BridgeProtocolError(
            f"Internal diagnostics inspection failed closed ({type(exc).__name__})."
        ) from exc
    if not isinstance(value, Mapping) or set(value) - SAFE_INTERNAL_DIAGNOSTIC_COUNT_KEYS:
        raise BridgeProtocolError("Internal diagnostic_counts returned unsupported fields.")
    if any(type(count) is not int or count < 0 for count in value.values()):
        raise BridgeProtocolError("Internal diagnostic counts must be nonnegative integers.")
    return dict(value)


def _external_admission_result_messages(
    calls: Sequence[ToolCallPart],
    admissions: Sequence[ExternalToolAdmission],
) -> list[CayuMessage]:
    denied_by_id = {
        admission.tool_call_id: admission.reason
        for admission in admissions
        if not admission.allowed
    }
    messages: list[CayuMessage] = []
    for call in calls:
        reason = denied_by_id.get(call.tool_call_id)
        if reason is None:
            content = (
                "This tau environment tool was not executed because another call in "
                "the same batch was denied by the admission gate. Reissue it in a new "
                "call after correcting the denied call."
            )
        else:
            content = (
                "This tau environment tool was not executed. The admission gate denied "
                f"it: {reason} Correct the sequencing or arguments before trying again."
            )
        messages.append(
            CayuMessage.tool_result(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                content=content,
                is_error=True,
            )
        )
    return messages


def _staged_external_failure_result_messages(
    calls: Sequence[ToolCallPart],
) -> list[CayuMessage]:
    """Report that an atomic external suffix was withheld after private failure."""

    return [
        CayuMessage.tool_result(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            content=(
                "This tau environment tool was not executed because the preceding private "
                "plan transition in the same staged batch failed. Correct the plan and reissue "
                "the environment call."
            ),
            is_error=True,
        )
        for call in calls
    ]


def _provider_state_parts(
    value: object,
    *,
    suppress_messages: bool,
) -> tuple[ProviderStatePart, ...]:
    if type(value) is not list:
        raise BridgeProtocolError("Provider state must be a list.")
    parts: list[ProviderStatePart] = []
    for item in value:
        if type(item) is not dict or set(item) != {"provider", "state"}:
            raise BridgeProtocolError("Provider state items require provider and state fields.")
        provider = item["provider"]
        state = item["state"]
        if not isinstance(provider, str) or not provider.strip() or type(state) is not dict:
            raise BridgeProtocolError("Provider state item has an invalid shape.")
        if suppress_messages and state.get("type") == "message":
            continue
        parts.append(ProviderStatePart(provider=provider, state=state))
    return tuple(parts)


def _model_output_to_cayu_message(output: _ModelOutput) -> CayuMessage:
    parts: list[ProviderStatePart | TextPart | ToolCallPart] = list(output.provider_state)
    if output.tool_calls:
        parts.extend(output.tool_calls)
    elif output.text.strip():
        parts.append(TextPart(text=output.text.strip()))
    if not parts:
        raise BridgeProtocolError("Provider output has no replayable assistant state.")
    return CayuMessage(role=MessageRole.ASSISTANT, content=tuple(parts))


def _external_only_model_output(
    output: _ModelOutput,
    calls: tuple[ToolCallPart, ...],
) -> _ModelOutput:
    """Project a staged mixed result onto the public tau-owned external suffix."""

    if not calls:
        raise BridgeProtocolError("A staged external projection requires tool calls.")
    return _ModelOutput(
        text=output.text,
        tool_calls=calls,
        # The provider state describes the original mixed response. Reusing it after
        # hiding private calls would create an incomplete server-side tool-result chain;
        # force the next public turn to rebuild from the authoritative tau transcript.
        provider_state=(),
        completion=output.completion,
        usage=output.usage,
        model=output.model,
    )


def _model_output_to_tau_message(
    output: _ModelOutput,
    internal_continuations: int,
    empty_response_retries: int,
    internal_tool_rounds: int,
    external_admission_rounds: int,
    external_admission_rejections: int,
    terminal_gate_rounds: int,
    terminal_gate_rejections: int,
    terminal_gate_rounds_since_progress: int,
    internal_progress_events: int,
    internal_stalled_rounds: int,
    stalled_internal_rounds_since_progress: int,
    internal_compacted_rounds: int,
    internal_context_refreshes: int,
    staged_external_batches: int,
    staged_external_calls: int,
    internal_tool_round_limit: int,
    stalled_internal_tool_round_limit: int,
    stalled_internal_tool_round_limit_since_progress: int,
    *,
    external_rejection_codes: Mapping[str, int],
    internal_calls_by_tool: Mapping[str, int],
    terminal_rejection_codes: Mapping[str, int],
    remediation_lifecycle: RemediationLifecycleDiagnostic,
    usage: dict[str, int] | None,
    provider_usages: list[dict[str, int] | None],
    provider_call_trace: Sequence[Mapping[str, Any]],
    attempt_id: str,
    correlation_tag: str | None,
) -> AssistantMessage:
    raw_data = {
        "cayu_tau3_bridge": {
            "model": output.model,
            "finish_reason": output.completion.finish_reason.value,
            "raw_finish_reason": output.completion.raw_finish_reason,
            "status": output.completion.status,
            "end_turn": output.completion.end_turn,
            "internal_continuations": internal_continuations,
            "internal_tool_rounds": internal_tool_rounds,
            "internal_progress_events": internal_progress_events,
            "internal_stalled_rounds": internal_stalled_rounds,
            "stalled_internal_rounds_since_progress": (stalled_internal_rounds_since_progress),
            "internal_compacted_rounds": internal_compacted_rounds,
            "internal_context_refreshes": internal_context_refreshes,
            "staged_external_batches": staged_external_batches,
            "staged_external_calls": staged_external_calls,
            "internal_calls_by_tool": dict(sorted(internal_calls_by_tool.items())),
            "internal_remediation": remediation_lifecycle.as_record(),
            "internal_tool_round_limit": internal_tool_round_limit,
            "stalled_internal_tool_round_limit": stalled_internal_tool_round_limit,
            "stalled_internal_tool_round_limit_since_progress": (
                stalled_internal_tool_round_limit_since_progress
            ),
            "external_admission_rounds": external_admission_rounds,
            "external_admission_rejections": external_admission_rejections,
            "external_rejection_codes": dict(sorted(external_rejection_codes.items())),
            "terminal_gate_rounds": terminal_gate_rounds,
            "terminal_gate_rejections": terminal_gate_rejections,
            "terminal_gate_rounds_since_progress": terminal_gate_rounds_since_progress,
            "terminal_rejection_codes": dict(sorted(terminal_rejection_codes.items())),
            "provider_call_count": len(provider_usages),
            "provider_usages": provider_usages,
            "provider_call_trace": {
                "schema_version": 1,
                "calls": [dict(item) for item in provider_call_trace],
            },
            "empty_response_retries": empty_response_retries,
            "suppressed_mixed_text": bool(output.tool_calls and output.text.strip()),
            "attempt_id": attempt_id,
            "correlation_tag": correlation_tag,
        }
    }
    if output.tool_calls:
        return AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id=call.tool_call_id,
                    name=call.tool_name,
                    arguments=call.arguments,
                    requestor="assistant",
                )
                for call in output.tool_calls
            ],
            usage=usage,
            raw_data=raw_data,
        )
    return AssistantMessage.text(
        output.text.strip(),
        usage=usage,
        raw_data=raw_data,
    )


def _aggregate_tau_usage(
    outputs: Sequence[_ModelOutput],
    provider: ModelProvider,
) -> dict[str, int] | None:
    if not outputs or any(output.usage is None for output in outputs):
        return None
    provider_name = getattr(provider, "name", None)
    raw_dialect = getattr(provider, "usage_dialect", None)
    usage_dialect = raw_dialect.value if hasattr(raw_dialect, "value") else raw_dialect
    prompt_tokens = 0
    completion_tokens = 0
    for output in outputs:
        metrics = _normalized_output_usage(
            output,
            provider_name=provider_name,
            usage_dialect=usage_dialect,
        )
        if metrics is None:
            return None
        prompt_tokens += metrics.input_tokens
        completion_tokens += metrics.output_tokens
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
    }


def _provider_usages(
    outputs: Sequence[_ModelOutput],
    provider: ModelProvider,
) -> list[dict[str, int] | None]:
    """Return bounded normalized usage, never provider attribution payloads."""

    provider_name = getattr(provider, "name", None)
    raw_dialect = getattr(provider, "usage_dialect", None)
    usage_dialect = raw_dialect.value if hasattr(raw_dialect, "value") else raw_dialect
    normalized: list[dict[str, int] | None] = []
    for output in outputs:
        metrics = _normalized_output_usage(
            output,
            provider_name=provider_name,
            usage_dialect=usage_dialect,
        )
        normalized.append(
            None
            if metrics is None
            else {
                "cached_input_tokens": metrics.cache.cached_input_tokens,
                "input_tokens": metrics.input_tokens,
                "output_tokens": metrics.output_tokens,
                "reasoning_output_tokens": metrics.reasoning_output_tokens,
            }
        )
    return normalized


def _normalized_output_usage(
    output: _ModelOutput,
    *,
    provider_name: object,
    usage_dialect: object,
) -> UsageMetrics | None:
    raw_usage = None if output.usage is None else dict(output.usage)
    return normalize_usage_metrics(
        provider_name=provider_name if isinstance(provider_name, str) else None,
        model=output.model,
        raw_usage=raw_usage,
        usage_dialect=usage_dialect if isinstance(usage_dialect, str) else None,
        requested_model=None,
    )


def _provider_call_diagnostic(
    request: ModelRequest,
    output: _ModelOutput,
    *,
    phase: str,
    forced_tool: bool,
) -> dict[str, Any]:
    """Return content-free request-shape telemetry for one acting-model call."""

    if phase not in _PROVIDER_CALL_PHASES:
        raise ValueError(f"Unsupported provider call phase: {phase!r}.")
    if type(forced_tool) is not bool:
        raise TypeError("forced_tool must be a boolean.")
    message_records = [
        message.model_dump(mode="json", exclude_none=True) for message in request.messages
    ]
    system_records = [
        record
        for message, record in zip(request.messages, message_records, strict=True)
        if message.role is MessageRole.SYSTEM
    ]
    result_kind = _provider_result_kind(output)
    return {
        "forced_tool": forced_tool,
        "message_bytes": _json_size(message_records),
        "message_count": len(message_records),
        "phase": phase,
        "result_kind": result_kind,
        "system_message_bytes": _json_size(system_records),
        "system_message_count": len(system_records),
        "tool_schema_bytes": _json_size(request.tools),
        "tool_schema_count": len(request.tools),
    }


def _provider_result_kind(output: _ModelOutput) -> str:
    if output.tool_calls:
        kind = "mixed_tool_calls" if output.text.strip() else "tool_calls"
    elif not output.text.strip():
        kind = "empty"
    elif output.completion.end_turn is False:
        kind = "continuation_text"
    else:
        kind = "terminal_text"
    if kind not in _PROVIDER_RESULT_KINDS:
        raise AssertionError(f"Unsupported provider result kind: {kind!r}.")
    return kind


def _json_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
