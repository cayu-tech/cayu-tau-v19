"""Controller-local structured work planning for compound banking requests."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
from typing import Any

from cayu import ToolCallPart, WorkEvidenceReference
from tau2.data_model.message import AssistantMessage, Message

from cayu_tau3_banking.bridge import (
    ExternalToolAdmission,
    InternalRemediationResolution,
    InternalToolRemediation,
    InternalToolResult,
    TerminalOutputAdmission,
)
from cayu_tau3_banking.case_conditions import ResourceCondition, condition_schema, parse_condition
from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL
from cayu_tau3_banking.ledger import ConversationLedger, ToolActionStatus
from cayu_tau3_banking.resource_scope import RESOURCE_ARGUMENT_KINDS, resource_targets

SYNC_WORK_PLAN_TOOL = "controller_sync_work_plan"
UPDATE_WORK_PLAN_TOOL = "controller_update_work_plan"
PREPARE_RESPONSE_TOOL = "controller_prepare_response"
HUMAN_TRANSFER_TOOL = "transfer_to_human_agents"

_CONTROL_PLANE_TOOL_NAMES = frozenset(
    {
        PREPARE_RESPONSE_TOOL,
        SYNC_WORK_PLAN_TOOL,
        UPDATE_WORK_PLAN_TOOL,
    }
)
_SERIAL_ENVIRONMENT_TOOL_NAMES = frozenset(
    {
        "give_discoverable_user_tool",
        "log_verification",
        "transfer_to_human_agents",
    }
)

_MAX_WORK_ITEMS = 24
_MAX_EVIDENCE_REFERENCES = 16
_MAX_ITEM_ID_CHARS = 48
_MAX_GOAL_CHARS = 500
_MAX_EVIDENCE_CHARS = 800
_MAX_OPERATION_SUMMARY_CHARS = 500
_MAX_OPERATION_ARGUMENTS_CHARS = 1_600
_MAX_TOOL_NAME_CHARS = 200
_MAX_OPERATION_CUSTOMER_HANDOFFS = 8
_MAX_SAME_PROMPT_OPERATION_CUSTOMER_HANDOFFS = 2
_HUMAN_TRANSFER_REQUEST_THRESHOLD = 4
_REVISION_REPAIR_CLASSES = {
    "revision_completed_item_changed": "restore_completed_item",
    "revision_completion_mode_changed": "preserve_prior_completion_mode",
    "revision_goal_changed": "preserve_prior_goal",
    "revision_prior_item_removed": "restore_prior_item",
    "revision_status_regressed": "use_non_regressive_status",
}
_ITEM_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_HANDOFF_SIGNATURE_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_WORKFLOW_CONTEXT_PREAMBLE = """
This is the current authoritative controller-maintained planning state and supersedes
any earlier workflow-plan snapshot in this request. It is not bank policy or evidence.
Keep it in sync as work evolves. Evidence notes are descriptive only; evidence_refs are
checked against customer messages, successful environment-tool results, and exact Cayu
knowledge revisions that were actually delivered. Controller tools never perform bank
actions and never substitute for required customer or environment tools.
""".strip()


class EvidenceKind(StrEnum):
    """Evidence channels visible at the Cayu/τ integration boundary."""

    CUSTOMER_MESSAGE = "customer_message"
    KNOWLEDGE_REVISION = "knowledge_revision"
    TOOL_ERROR = "tool_error"
    TOOL_RESULT = "tool_result"


class WorkStatus(StrEnum):
    """State of one customer-visible outcome in the work plan."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_CUSTOMER = "awaiting_customer"
    BLOCKED = "blocked"
    COMPLETED = "completed"


class WorkOutcome(StrEnum):
    """Typed disposition for one work item."""

    ANSWERED = "answered"
    BLOCKED = "blocked"
    DECLINED = "declined"
    DENIED = "denied"
    HANDED_OFF = "handed_off"
    SUCCEEDED = "succeeded"
    UNRESOLVED = "unresolved"


class WorkCompletionMode(StrEnum):
    """Immutable kind of proof needed to satisfy a requested outcome."""

    INFORMATION = "information"
    OPERATION = "operation"


class WorkOwner(StrEnum):
    """Party responsible for the next operation."""

    AGENT = "agent"
    CUSTOMER = "customer"
    NONE = "none"


class WorkOperationKind(StrEnum):
    """Kind of operation that must happen next."""

    CUSTOMER_RESPONSE = "customer_response"
    CUSTOMER_PRODUCT_SELECTION = "customer_product_selection"
    KNOWLEDGE_SEARCH = "knowledge_search"
    NONE = "none"
    TOOL = "tool"


class ResponseDispositionKind(StrEnum):
    """Declared purpose of one customer-facing response."""

    ASK_CUSTOMER = "ask_customer"
    ASK_PRODUCT_SELECTION = "ask_product_selection"
    FINAL = "final"
    HANDOFF_CUSTOMER_TOOL = "handoff_customer_tool"


class WorkflowTransitionRejection(ValueError):
    """A plan transition failed with a closed machine-actionable correction."""

    def __init__(self, message: str, *, remediation: InternalToolRemediation) -> None:
        super().__init__(message)
        if not isinstance(remediation, InternalToolRemediation):
            raise TypeError("Workflow transition remediation is invalid.")
        self.remediation = remediation


class _ControllerValidationRejection(ValueError):
    """A content-bearing rejection paired with one persistence-safe code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WorkOperation:
    """One explicit future operation, or a canonical terminal sentinel."""

    owner: WorkOwner
    kind: WorkOperationKind
    tool_name: str
    arguments: dict[str, Any]
    summary: str

    def as_dict(self) -> dict[str, str]:
        return {
            "arguments": _json_object_copy(self.arguments),
            "kind": self.kind.value,
            "owner": self.owner.value,
            "summary": self.summary,
            "tool_name": self.tool_name,
        }


@dataclass(frozen=True)
class PlannedToolCall:
    """The mutation grounding record attached to an action obligation."""

    tool_name: str
    arguments: dict[str, Any]
    outcome_effects: dict[str, str] | None = None
    customer_choice: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": _json_object_copy(self.arguments),
            **(
                {"outcome_effects": dict(self.outcome_effects)}
                if self.outcome_effects is not None
                else {}
            ),
            **({"customer_choice": self.customer_choice} if self.customer_choice else {}),
        }


@dataclass(frozen=True)
class WorkObligation:
    """Model-identified preparation, independent of the parent effect's dependencies."""

    obligation_id: str
    goal: str
    status: str
    evidence: str
    evidence_refs: tuple[WorkEvidenceReference, ...]
    kind: str = "action"
    comparison_worksheet_id: str | None = None
    condition: ResourceCondition | None = None
    depends_on: tuple[str, ...] = ()
    applied_to: str | None = None
    planned_tool: PlannedToolCall | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.obligation_id,
            "goal": self.goal,
            "kind": self.kind,
            **(
                {"comparison_worksheet_id": self.comparison_worksheet_id}
                if self.comparison_worksheet_id is not None
                else {}
            ),
            "status": self.status,
            "evidence": self.evidence,
            "evidence_refs": [
                ref.model_dump(mode="json", exclude_none=True) for ref in self.evidence_refs
            ],
            **({"condition": self.condition.as_dict()} if self.condition else {}),
            **({"depends_on": list(self.depends_on)} if self.depends_on else {}),
            **({"applied_to": self.applied_to} if self.applied_to else {}),
            **({"planned_tool": self.planned_tool.as_dict()} if self.planned_tool else {}),
        }


@dataclass(frozen=True)
class WorkItem:
    """One stable unit of customer-requested work."""

    item_id: str
    goal: str
    completion_mode: WorkCompletionMode
    status: WorkStatus
    outcome: WorkOutcome
    depends_on: tuple[str, ...]
    next_operation: WorkOperation
    evidence: str
    evidence_refs: tuple[WorkEvidenceReference, ...]
    obligations: tuple[WorkObligation, ...] = ()
    prior_customer_report: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "completion_mode": self.completion_mode.value,
            "depends_on": list(self.depends_on),
            "evidence": self.evidence,
            "evidence_refs": [
                reference.model_dump(mode="json", exclude_none=True)
                for reference in self.evidence_refs
            ],
            "goal": self.goal,
            "id": self.item_id,
            "next_operation": self.next_operation.as_dict(),
            "outcome": self.outcome.value,
            "status": self.status.value,
            **(
                {"obligations": [o.as_dict() for o in self.obligations]} if self.obligations else {}
            ),
            **({"prior_customer_report": True} if self.prior_customer_report else {}),
        }


@dataclass(frozen=True)
class ResponseItemDisposition:
    """Response purpose declared for one work item."""

    item_id: str
    kind: ResponseDispositionKind

    def as_dict(self) -> dict[str, str]:
        return {"item_id": self.item_id, "kind": self.kind.value}


@dataclass(frozen=True)
class ResponseDisposition:
    """One response declaration bound to an exact work-plan revision."""

    plan_revision: int
    items: tuple[ResponseItemDisposition, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": [item.as_dict() for item in self.items],
            "plan_revision": self.plan_revision,
        }


class BankingWorkflowController:
    """Validated model-facing plan store with no benchmark-environment access."""

    def __init__(
        self,
        ledger: ConversationLedger,
        *,
        available_environment_tool_names: Iterable[str] | None = None,
        available_private_tool_names: Iterable[str] | None = None,
        knowledge_search_available: bool = True,
        verified_capability_gap: Callable[[], bool] | None = None,
        planning_mode: str = "exact",
        enforce_comparison_coverage: bool = True,
    ) -> None:
        if planning_mode not in {"exact", "outcomes"}:
            raise ValueError("planning_mode must be exact or outcomes")
        self.planning_mode = planning_mode
        self.enforce_comparison_coverage = enforce_comparison_coverage
        if not isinstance(ledger, ConversationLedger):
            raise TypeError("ledger must be a ConversationLedger.")
        if available_environment_tool_names is None:
            normalized_tool_names = None
        else:
            normalized_tool_names = tuple(available_environment_tool_names)
            if any(
                not isinstance(name, str) or not name.strip() or name != name.strip()
                for name in normalized_tool_names
            ):
                raise ValueError(
                    "available environment tool names must be nonblank normalized strings."
                )
        if available_private_tool_names is None:
            normalized_private_tool_names = None
        else:
            normalized_private_tool_names = tuple(available_private_tool_names)
            if any(
                not isinstance(name, str) or not name.strip() or name != name.strip()
                for name in normalized_private_tool_names
            ):
                raise ValueError(
                    "available private tool names must be nonblank normalized strings."
                )
        if type(knowledge_search_available) is not bool:
            raise TypeError("knowledge_search_available must be a boolean.")
        self._ledger = ledger
        if verified_capability_gap is not None and not callable(verified_capability_gap):
            raise TypeError("verified_capability_gap must be callable or None.")
        self._verified_capability_gap = verified_capability_gap
        self._available_agent_tool_names = (
            None
            if normalized_tool_names is None and normalized_private_tool_names is None
            else frozenset(normalized_tool_names or ())
            | frozenset(normalized_private_tool_names or ())
        )
        self._knowledge_search_available = knowledge_search_available
        self._comparison_readiness: dict[str, bool] = {}
        self._items: dict[str, WorkItem] = {}
        self._previous_information_completions: dict[str, list[dict[str, Any]]] = {}
        self._knowledge_evidence: dict[tuple[str, str], WorkEvidenceReference] = {}
        self._customer_evidence_floors: dict[str, int] = {}
        self._customer_response_handoffs: dict[str, int] = {}
        self._customer_response_handoff_signatures: dict[str, list[str]] = {}
        self._customer_response_handoff_sequences: dict[str, int] = {}
        self._tool_evidence_floors: dict[str, int] = {}
        self._pending_response_disposition: ResponseDisposition | None = None
        self._last_response_disposition: ResponseDisposition | None = None
        self._semantic_rejection_revision: int | None = None
        self._semantic_reclassifiable_item_ids: set[str] = set()
        self._semantic_required_new_item_mode: WorkCompletionMode | None = None
        self._semantic_required_new_item_count = 0
        self._semantic_requires_operation_increase = False
        self._product_selection_transition_rejections = 0
        self._pending_remediations: dict[str, InternalToolRemediation] = {}
        self._pending_condition_checks: dict[
            str, tuple[tuple[str, str, ResourceCondition], ...]
        ] = {}
        self.revision = 0
        self._plan_advisories: list[str] = []

    @classmethod
    def from_history(
        cls,
        messages: list[Message] | None,
        *,
        ledger: ConversationLedger,
        available_environment_tool_names: Iterable[str] | None = None,
        available_private_tool_names: Iterable[str] | None = None,
        knowledge_search_available: bool = True,
        verified_capability_gap: Callable[[], bool] | None = None,
        planning_mode: str = "exact",
        enforce_comparison_coverage: bool = True,
    ) -> BankingWorkflowController:
        """Restore the latest controller snapshot embedded in public agent history."""

        controller = cls(
            ledger,
            available_environment_tool_names=available_environment_tool_names,
            available_private_tool_names=available_private_tool_names,
            knowledge_search_available=knowledge_search_available,
            verified_capability_gap=verified_capability_gap,
            planning_mode=planning_mode,
            enforce_comparison_coverage=enforce_comparison_coverage,
        )
        for message in messages or ():
            if not isinstance(message, AssistantMessage) or not isinstance(
                message.raw_data, Mapping
            ):
                continue
            raw = message.raw_data.get("cayu_tau3_workflow")
            if not isinstance(raw, Mapping):
                continue
            if raw.get("planning_mode", "exact") != planning_mode:
                raise ValueError("Cannot restore a different planning mode into this controller")
            schema_version = raw.get("schema_version")
            if schema_version not in {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
                continue
            revision = raw.get("revision")
            advisories = raw.get("plan_advisories", [])
            if (
                not isinstance(advisories, list)
                or len(advisories) > 64
                or any(
                    not isinstance(a, str) or not _HANDOFF_SIGNATURE_PATTERN.fullmatch(a)
                    for a in advisories
                )
            ):
                continue
            items = raw.get("items")
            floors = raw.get("customer_evidence_floors")
            handoffs = raw.get("customer_response_handoffs")
            handoff_signatures = raw.get("customer_response_handoff_signatures")
            handoff_sequences = raw.get("customer_response_handoff_sequences")
            selection_rejections = (
                raw.get("product_selection_transition_rejections", 0)
                if schema_version in {11, 12, 13, 14, 15, 16, 17, 18, 19}
                else 0
            )
            tool_floors = raw.get("tool_evidence_floors")
            if (
                type(revision) is not int
                or revision < 0
                or type(items) is not list
                or type(floors) is not dict
                or (schema_version == 10 and type(handoffs) is not dict)
                or (
                    schema_version in {11, 12, 13, 14, 15, 16, 17, 18, 19}
                    and (
                        type(handoffs) is not dict
                        or type(selection_rejections) is not int
                        or selection_rejections < 0
                    )
                )
                or (
                    schema_version in {14, 15, 16, 17, 18, 19}
                    and (
                        type(handoff_signatures) is not dict or type(handoff_sequences) is not dict
                    )
                )
                or type(tool_floors) is not dict
            ):
                continue
            try:
                parsed = controller._parse_items(
                    items,
                    validate_references=False,
                    allow_legacy_terminal_blocked=schema_version in {6, 7},
                    allow_legacy_completion_mode=schema_version in {6, 7, 8},
                )
                previous_completions = controller._parse_previous_information_completions(
                    raw.get("previous_information_completions", {}), parsed, revision
                )
                if previous_completions and schema_version != 19:
                    raise ValueError("Previous information completions require schema 19")
                for item in parsed:
                    previous = controller._items.get(item.item_id)
                    if (
                        previous is not None
                        and controller._is_information_reopening(previous, item)
                        and not any(
                            record["item"] == previous.as_dict()
                            for record in previous_completions.get(item.item_id, [])
                        )
                    ):
                        raise ValueError("Reopening snapshot must retain the previous completion")
                if schema_version < 17 and any(
                    o.comparison_worksheet_id is not None
                    for item in parsed
                    for o in item.obligations
                ):
                    raise ValueError("Comparison bindings require workflow snapshot version 17")
                if schema_version < 18 and any(
                    o.condition is not None or o.depends_on
                    for item in parsed
                    for o in item.obligations
                ):
                    raise ValueError("Resource conditions require workflow snapshot version 18")
                controller._validate_revision(
                    parsed,
                    validate_handoff_progress=False,
                    validate_comparisons=False,
                    historical_reopen_ids=frozenset(
                        item_id
                        for item_id, records in previous_completions.items()
                        if item_id in controller._items
                        and any(
                            record["item"] == controller._items[item_id].as_dict()
                            for record in records
                        )
                    ),
                )
                parsed_floors = _parse_customer_evidence_floors(
                    floors,
                    item_ids={item.item_id for item in parsed},
                    latest_sequence=ledger.last_customer_message_sequence,
                )
                parsed_tool_floors = _parse_tool_evidence_floors(
                    tool_floors,
                    item_ids={item.item_id for item in parsed},
                    latest_sequence=ledger.last_tool_action_sequence,
                )
                last_response_disposition = _parse_restored_response_disposition(
                    raw.get("last_response_disposition"),
                    item_ids={item.item_id for item in parsed},
                    maximum_revision=revision,
                )
                if schema_version in {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}:
                    parsed_handoffs = _parse_customer_response_handoffs(
                        handoffs,
                        item_ids={item.item_id for item in parsed},
                    )
                else:
                    parsed_handoffs = {
                        item.item_id: controller._customer_response_handoffs.get(item.item_id, 0)
                        for item in parsed
                    }
                    if last_response_disposition is not None and message.has_text_content():
                        for disposition in last_response_disposition.items:
                            restored_item = next(
                                item for item in parsed if item.item_id == disposition.item_id
                            )
                            if (
                                disposition.kind
                                in {
                                    ResponseDispositionKind.ASK_CUSTOMER,
                                    ResponseDispositionKind.ASK_PRODUCT_SELECTION,
                                }
                                and restored_item.completion_mode is WorkCompletionMode.OPERATION
                            ):
                                parsed_handoffs[disposition.item_id] = min(
                                    parsed_handoffs[disposition.item_id] + 1,
                                    _MAX_OPERATION_CUSTOMER_HANDOFFS,
                                )
                if schema_version in {14, 15, 16, 17, 18, 19}:
                    parsed_handoff_signatures = _parse_customer_response_handoff_signatures(
                        handoff_signatures,
                        handoffs=parsed_handoffs,
                    )
                    parsed_handoff_sequences = _parse_customer_response_handoff_sequences(
                        handoff_sequences,
                        handoffs=parsed_handoffs,
                        latest_sequence=ledger.last_customer_message_sequence,
                        counted_handoffs=planning_mode != "outcomes",
                    )
                else:
                    parsed_handoff_signatures = {item.item_id: [] for item in parsed}
                    parsed_handoff_sequences = {item.item_id: 0 for item in parsed}
            except ValueError:
                continue
            if revision < controller.revision:
                continue
            controller._items = {item.item_id: item for item in parsed}
            controller._previous_information_completions = previous_completions
            controller._customer_evidence_floors = parsed_floors
            controller._customer_response_handoffs = parsed_handoffs
            controller._customer_response_handoff_signatures = parsed_handoff_signatures
            controller._customer_response_handoff_sequences = parsed_handoff_sequences
            controller._tool_evidence_floors = parsed_tool_floors
            controller._pending_response_disposition = None
            controller._last_response_disposition = last_response_disposition
            controller._product_selection_transition_rejections = max(
                controller._product_selection_transition_rejections,
                selection_rejections,
            )
            archived_items = [
                controller._parse_items(
                    [
                        record["item"] if item.item_id == item_id else item.as_dict()
                        for item in parsed
                    ],
                    validate_references=False,
                )
                for item_id, records in previous_completions.items()
                for record in records
            ]
            for item in [*parsed, *(item for group in archived_items for item in group)]:
                retained_refs = (
                    *item.evidence_refs,
                    *(r for o in item.obligations for r in o.evidence_refs),
                    *(o.condition.procedure for o in item.obligations if o.condition is not None),
                )
                for reference in retained_refs:
                    if reference.kind == EvidenceKind.KNOWLEDGE_REVISION.value:
                        controller._remember_knowledge_evidence(reference)
            controller.revision = revision
            controller._plan_advisories = list(dict.fromkeys(advisories))
        return controller

    def tool_schemas(self) -> Sequence[Mapping[str, Any]]:
        return self._plan_tool_schemas(updating=bool(self._items))

    def registration_tool_schemas(self) -> Sequence[Mapping[str, Any]]:
        """Static native-tool catalogue; current exposure still uses tool_schemas."""

        return (*self._plan_tool_schemas(updating=False), *self._plan_tool_schemas(updating=True))

    def _plan_tool_schemas(self, *, updating: bool) -> Sequence[Mapping[str, Any]]:
        tool_name = UPDATE_WORK_PLAN_TOOL if updating else SYNC_WORK_PLAN_TOOL
        scope_description = (
            "Update only new or changed outcomes in the existing private plan. Omitted "
            "accepted items remain unchanged automatically; do not copy unchanged items. "
            "This is an item-level upsert, not a field-level patch: every included item "
            "must contain its complete required shape."
            if updating
            else (
                "Create private controller memory for every distinct outcome in the "
                "customer's request."
            )
        )
        schemas = (
            {
                "name": tool_name,
                "description": (
                    f"{scope_description} Call this before any other tool or customer-facing "
                    "text when the plan must change. Preserve stable ids, goals, and completion "
                    "modes. This does not call the bank, retrieve policy, contact the customer, "
                    "or complete an action. evidence is an explanatory note only. evidence_refs are "
                    "structurally checked against real ledger observations or Cayu-delivered "
                    "knowledge. Mark completed only with checked customer_message, tool_result, "
                    "or knowledge_revision references; tool_error can support only blocked work."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": _MAX_WORK_ITEMS,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "goal": {"type": "string"},
                                    "completion_mode": {
                                        "type": "string",
                                        "enum": [mode.value for mode in WorkCompletionMode],
                                        "description": (
                                            "Immutable proof class derived from the customer's "
                                            "requested outcome: information for an answer, or "
                                            "operation for an action, application, change, "
                                            "customer-side tool, or escalation."
                                        ),
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": [status.value for status in WorkStatus],
                                    },
                                    "outcome": {
                                        "type": "string",
                                        "enum": [outcome.value for outcome in WorkOutcome],
                                    },
                                    "depends_on": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "evidence": {
                                        "type": "string",
                                        "maxLength": _MAX_EVIDENCE_CHARS,
                                        "description": (
                                            "Concise explanation of what the checked references "
                                            "prove; this text is not proof by itself."
                                        ),
                                    },
                                    "evidence_refs": {
                                        "type": "array",
                                        "maxItems": _MAX_EVIDENCE_REFERENCES,
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "kind": {
                                                    "type": "string",
                                                    "enum": [kind.value for kind in EvidenceKind],
                                                },
                                                "reference_id": {
                                                    "type": "string",
                                                    "description": (
                                                        "Exact customer-message, tool-call, or "
                                                        "knowledge-entry id shown in context. For "
                                                        "knowledge, copy a document's evidence_ref "
                                                        "object unchanged."
                                                    ),
                                                },
                                                "version": {
                                                    "type": "string",
                                                    "description": (
                                                        "Required only for knowledge_revision; "
                                                        "use its delivered entry_revision."
                                                    ),
                                                },
                                                "digest": {
                                                    "type": "string",
                                                    "description": (
                                                        "Optional knowledge content_sha256; when "
                                                        "provided it must match exactly."
                                                    ),
                                                },
                                            },
                                            "required": ["kind", "reference_id"],
                                            "additionalProperties": False,
                                        },
                                    },
                                    "next_operation": {
                                        "type": "object",
                                        "description": (
                                            "The single next required operation. Use the canonical "
                                            "none/none sentinel with blank tool_name and summary "
                                            "and empty arguments: {owner: none, kind: none, "
                                            "tool_name: '', arguments: {}, summary: ''}. Use it "
                                            "only for completed work, or pending work "
                                            "whose declared dependency is incomplete. For "
                                            "knowledge_search use owner agent with blank tool_name "
                                            "and empty arguments; the bridge selects the Cayu search "
                                            "tool after accepting the plan."
                                        ),
                                        "properties": {
                                            "owner": {
                                                "type": "string",
                                                "enum": [owner.value for owner in WorkOwner],
                                            },
                                            "kind": {
                                                "type": "string",
                                                "enum": [kind.value for kind in WorkOperationKind],
                                                "description": (
                                                    "Use customer_product_selection only for one "
                                                    "direct choice among already-compared exact "
                                                    "product names; use customer_response for "
                                                    "other missing customer facts. A customer-side "
                                                    "discoverable tool is owner customer with kind "
                                                    "tool, its exact tool_name, and exact arguments; "
                                                    "represent other required facts as separate "
                                                    "customer_response items."
                                                ),
                                            },
                                            "tool_name": {"type": "string"},
                                            "arguments": {
                                                "type": "object",
                                                "description": (
                                                    "Exact arguments for kind tool; otherwise an "
                                                    "empty object."
                                                ),
                                            },
                                            "summary": {"type": "string"},
                                        },
                                        "required": [
                                            "owner",
                                            "kind",
                                            "tool_name",
                                            "arguments",
                                            "summary",
                                        ],
                                        "additionalProperties": False,
                                    },
                                },
                                "required": [
                                    "id",
                                    "goal",
                                    "completion_mode",
                                    "status",
                                    "outcome",
                                    "depends_on",
                                    "next_operation",
                                    "evidence",
                                    "evidence_refs",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["items"],
                    "additionalProperties": False,
                },
            },
        )
        if self.planning_mode == "outcomes":
            schema = schemas[0]
            schema["description"] = (
                "Track customer-visible outcomes, not individual reads or tool calls. "
                "Update only when a request, outcome, blocker or customer handoff changes. "
                "Updates are field patches by stable id; omitted fields and items are retained. "
                "Create a new item with id, goal and completion_mode; omitted status defaults to "
                "pending, unresolved outcome and no dependencies. For example create "
                "{id: compare, goal: Compare suitable cards, completion_mode: information}; "
                "later send {id: compare, status: in_progress}. Keep independent customer "
                "outcomes separate; add a new item when the customer requests additional work. "
                "New pending/in_progress items may omit empty evidence and evidence_refs. "
                "Other new items require an explicit outcome and supporting evidence. "
                "When research discovers a material customer-affecting issue outside the "
                "explicit current action scope, record it as a separate information item "
                "with awaiting_customer status, unresolved outcome, customer-owned next "
                "operation and real evidence references; this records a clarification need, "
                "not authorization. Do not create such items for trivial rounding differences "
                "or issues the customer explicitly declined. "
                "If a consequential action could reduce a benefit, create a fee or reverse "
                "a credit, and the customer's authorization is ambiguous, ask one direct "
                "action-specific yes/no question naming the risk; do not infer authorization "
                "from a general request to proceed or silently choose a narrower scope. "
                "Separate beneficial and adverse scopes: completion or authorization of a "
                "beneficial operation never authorizes other adverse operations. "
                "Evidence references must identify real observed "
                "results or delivered knowledge; explain their relevance in evidence. "
                "Use denied for a bank-denied operation, citing its decision result; declined is customer withdrawal. "
                "Succeeded means the requested customer result was achieved, not merely a successful tool call. "
                "Completed means supported by evidence, not merely planned. Blocked and "
                "awaiting_customer require an explanation; they do not mean success."
                " An information outcome may be completed with outcome declined only when "
                "a later customer message explicitly withdraws that research; cite that "
                "message, not acceptance as proof that an answer was correct. Patch changed "
                "fields only. Information parents may inherit checked child citations when "
                "parent evidence_refs are omitted and empty. Awaiting-customer or blocked "
                "operations may similarly reuse explicitly evidenced waiting obligations. "
                "Supplied parent evidence is never replaced; operation success proof is never inherited. "
                "To reconsider completed INFORMATION, update status to pending or in_progress and "
                "outcome unresolved (default when omitted). Prior completion is archived; omitted "
                "active evidence is cleared and research obligations reopen for reassessment. "
                "Completed operations remain immutable; track new requested operations separately."
            )
            item_schema = schema["input_schema"]["properties"]["items"]["items"]
            # Integrity belongs to the delivered-revision registry, not model
            # transcription. Legacy persisted/caller-supplied hashes still
            # validate normally; this changes only the advertised input shape.
            del item_schema["properties"]["evidence_refs"]["items"]["properties"]["digest"]
            del item_schema["properties"]["next_operation"]
            item_schema["required"].remove("next_operation")
            item_schema["required"].remove("evidence")
            item_schema["required"].remove("evidence_refs")
            for optional in ("status", "outcome", "depends_on"):
                item_schema["required"].remove(optional)
            item_schema["properties"]["prior_customer_report"] = {
                "type": "boolean",
                "description": (
                    "Only for a NEW completed operation: true when the latest customer message "
                    "explicitly reports prior success or refusal, not a request, intention, or "
                    "consent. Cite that exact customer message and explain what it reports. "
                    "This records customer testimony, not independently verified execution. "
                    "Cannot bypass an existing item's customer handoff. Defaults to false."
                ),
            }
            item_schema["properties"]["depends_on"]["description"] = (
                "Only ids of OTHER TOP-LEVEL outcome items, never ids from obligations. "
                "Use [] when prerequisites are within this item's own obligations. "
                "These constrain the final effect, not independent preparation."
            )
            item_schema["properties"]["obligations"] = {
                "type": "array",
                "maxItems": 8,
                "description": (
                    "Bounded policy obligations within this outcome, not a log of tool calls. "
                    "ready means preparation can proceed even if the final effect is blocked. "
                    "For a consequential banking action, include planned_tool with the exact "
                    "environment tool name and arguments; this binds the intended mutation at "
                    "dispatch without claiming that it already succeeded. "
                    "waiting, resolved and not_applicable ALL require a nonempty explanation "
                    "and at least one real evidence_ref. Only ready may omit references. "
                    "Ready means executable now, not simply unfinished. An inspection is ready "
                    "only if its own prerequisites are satisfied; otherwise record waiting. "
                    "For a known procedural prerequisite, cite its delivered policy revision. "
                    "Do not copy the parent final effect or every tool call into obligations. "
                    "Use kind communication for an explanation or offer. Use kind customer_input "
                    "for a necessary question requesting missing customer facts, consent or selection: "
                    "its pending/in_progress parent becomes awaiting_customer while ready, "
                    "without completing work; do not duplicate that lifecycle update. Even ready questions "
                    "require an explanation and real evidence_refs for the missing prerequisite. "
                    "Such a question may pause unrelated work, never complete it. Finish the "
                    "question's own ready preparatory actions before asking. "
                    "Use kind knowledge_search for a document lookup, not a banking action. "
                    "An unresolved ready knowledge_search is resource-blocked while search is "
                    "unavailable; it is not completed or waived. You may explain that limit or "
                    "ask a genuine customer-choice question without claiming an unverified effect. "
                    "A ready communication is performed by the next response, not completed in advance. "
                    "Use waiting after delivery when customer input is required. "
                    "Updates patch obligations by stable id; omitted obligations and fields stay "
                    "unchanged. For existing obligations send id and changed status/evidence only, "
                    "not the immutable goal. New obligations require id, goal and status; "
                    "ready defaults missing evidence to '' and evidence_refs to []. "
                    "Other states require explicit nonempty evidence and evidence_refs."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "goal": {"type": "string"},
                        "kind": {
                            "type": "string",
                            "enum": [
                                "action",
                                "communication",
                                "customer_input",
                                "knowledge_search",
                            ],
                        },
                        "status": {
                            "type": "string",
                            "enum": ["ready", "waiting", "resolved", "not_applicable"],
                        },
                        "evidence": {"type": "string"},
                        "comparison_worksheet_id": {
                            "type": "string",
                            "pattern": "^[a-zA-Z0-9_-]{1,64}$",
                            "description": (
                                "Optional stable evidence-collection link for a knowledge_search obligation. "
                                "In advisory mode this does not require exhaustive coverage or removal of uncertainty. "
                                "Research may be resolved with a supported conditional comparison; keep unknown or contradictory terms explicit. "
                                "Resolution is not evidence that a customer reply was delivered or a banking action executed."
                                if not self.enforce_comparison_coverage else
                                "For a knowledge_search obligation comparing products, bind its stable worksheet_id. It cannot be resolved until that worksheet's declared source cells are complete, coverage is recorded and no material question remains open. Unknown is not excluded; lack of acquisition consent does not prohibit advice. The link cannot later be removed or changed."
                            ),
                        },
                        "evidence_refs": item_schema["properties"]["evidence_refs"],
                        "condition": condition_schema(
                            item_schema["properties"]["evidence_refs"]["items"]
                        ),
                        "depends_on": {
                            "type": "array",
                            "maxItems": 7,
                            "uniqueItems": True,
                            "items": {"type": "string"},
                            "description": "Ids of prerequisite obligations in THIS outcome. The system reopens dependent checks after prerequisite state or evidence changes. Not top-level outcome ids. Omission preserves existing links.",
                        },
                        "planned_tool": {
                            "type": "object",
                            "description": (
                                "Optional mutation grounding packet for an action obligation. "
                                "When supplied, the named tool and exact arguments are checked "
                                "at the external-tool boundary. This is not proof of success."
                            ),
                            "properties": {
                                "tool_name": {"type": "string", "minLength": 1},
                                "arguments": {"type": "object"},
                                "outcome_effects": {
                                    "type": "object",
                                    "additionalProperties": {
                                        "type": "string",
                                        "enum": [
                                            "preserves",
                                            "blocks",
                                            "unknown",
                                            "customer_accepted",
                                        ],
                                    },
                                    "description": "Effect on each OTHER unfinished operation outcome, keyed by outcome ID. Cite policy in the obligation evidence_refs.",
                                },
                                "customer_choice": {
                                    "type": "string",
                                    "description": "Latest customer-message ID accepting a previously explained consequence.",
                                },
                            },
                            "required": ["tool_name", "arguments"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["id", "goal", "status"],
                    "additionalProperties": False,
                },
            }
            if updating:
                item_schema["required"] = ["id"]
                item_schema["properties"]["obligations"]["items"]["required"] = ["id"]
            item_schema["properties"]["obligations"]["items"]["properties"]["evidence"][
                "maxLength"
            ] = _MAX_EVIDENCE_CHARS
        return schemas

    def handles(self, name: str) -> bool:
        return name in {
            PREPARE_RESPONSE_TOOL,
            SYNC_WORK_PLAN_TOOL,
            UPDATE_WORK_PLAN_TOOL,
        }

    def transcript_retention_key(self, name: str) -> str | None:
        """Compact plan calls once refreshed controller context makes them supersedable."""

        return "workflow_plan" if name in {SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL} else None

    def render_internal_context(self) -> str:
        """Return the authoritative controller state after an in-turn transition."""

        return self.render_context()

    def has_work_items(self) -> bool:
        """Return whether the controller has a populated work plan."""

        return bool(self._items)

    def set_comparison_readiness(self, readiness: Mapping[str, bool]) -> None:
        if any(type(k) is not str or type(v) is not bool for k, v in readiness.items()):
            raise TypeError("Comparison readiness requires exact boolean values.")
        self._comparison_readiness = dict(readiness)

    def invalid_comparison_bindings(self) -> tuple[str, ...]:
        if not self.enforce_comparison_coverage:
            return ()
        return tuple(
            f"{item.item_id}/{o.obligation_id}"
            for item in self._items.values()
            if item.status is not WorkStatus.COMPLETED
            for o in item.obligations
            if o.status == "resolved"
            and o.comparison_worksheet_id is not None
            and not self._comparison_readiness.get(o.comparison_worksheet_id, False)
        )

    def required_action_tool_names(self) -> tuple[str, ...]:
        """Return ordered executable continuations for dependency-ready work."""

        if self.planning_mode == "outcomes":
            return ()
        actionable = self._dependency_ready_actionable_items()
        if not actionable:
            return ()
        unreconciled = tuple(
            item for item in actionable if not self._planned_agent_tool_result_is_resolved(item)
        )
        return _required_action_tool_names(
            unreconciled,
            knowledge_search_available=self._knowledge_search_available,
        )

    def set_knowledge_search_available(self, available: bool) -> None:
        """Update the transient search capability used for plan validation."""

        if type(available) is not bool:
            raise TypeError("knowledge search availability must be a boolean")
        self._knowledge_search_available = available

    def allows_staged_external_calls(self, internal_tool_names: tuple[str, ...]) -> bool:
        """Authorize one plan publication before a gated external suffix."""

        return internal_tool_names in {
            (SYNC_WORK_PLAN_TOOL,),
            (UPDATE_WORK_PLAN_TOOL,),
        }

    def required_action_batch_size(self, name: str) -> int:
        """Return a safe homogeneous batch size for one required action name."""

        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("required action name must be normalized nonblank text")
        if name == KNOWLEDGE_SEARCH_TOOL:
            if not self._knowledge_search_available:
                return 1
            return max(
                1,
                sum(
                    item.next_operation.owner is WorkOwner.AGENT
                    and item.next_operation.kind is WorkOperationKind.KNOWLEDGE_SEARCH
                    for item in self._dependency_ready_actionable_items()
                ),
            )
        matching_batches = [
            len(batch["item_ids"])
            for batch in self._ready_agent_tool_batches()
            if batch["tool_name"] == name
        ]
        return max(matching_batches, default=1)

    def reconcile_successful_human_transfer(self, call_id: str) -> None:
        """Hand all unfinished work to a human using authoritative tool evidence.

        A successful transfer terminates the automated agent's responsibility for the
        remaining conversation, but it does not prove that any requested banking
        mutation itself succeeded. The distinct ``handed_off`` outcome preserves that
        distinction in durable workflow state.
        """

        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("Human-transfer call_id must be nonblank text.")
        normalized_call_id = call_id.strip()
        if self._ledger.tool_action_name(normalized_call_id) != HUMAN_TRANSFER_TOOL:
            raise ValueError("Human-transfer reconciliation requires the transfer tool.")
        if self._ledger.tool_action_status(normalized_call_id) is not ToolActionStatus.SUCCEEDED:
            raise ValueError("Human-transfer reconciliation requires a successful result.")
        transfer_arguments = self._ledger.tool_action_arguments(normalized_call_id)
        planned_transfer = any(
            item.status is not WorkStatus.COMPLETED
            and item.next_operation.owner is WorkOwner.AGENT
            and item.next_operation.kind is WorkOperationKind.TOOL
            and item.next_operation.tool_name == HUMAN_TRANSFER_TOOL
            and _json_objects_equal(item.next_operation.arguments, transfer_arguments)
            for item in self._items.values()
        )
        if not planned_transfer and self.planning_mode != "outcomes":
            raise ValueError(
                "Human-transfer reconciliation requires a matching unfinished planned transfer."
            )

        transfer_reference = WorkEvidenceReference(
            kind=EvidenceKind.TOOL_RESULT.value,
            reference_id=normalized_call_id,
        )
        reconciled: list[WorkItem] = []
        for item in self._items.values():
            if item.status is WorkStatus.COMPLETED:
                reconciled.append(item)
                continue
            retained_references = tuple(
                reference
                for reference in item.evidence_refs
                if (
                    reference.kind,
                    reference.reference_id,
                    reference.version,
                )
                != (
                    transfer_reference.kind,
                    transfer_reference.reference_id,
                    transfer_reference.version,
                )
            )[: _MAX_EVIDENCE_REFERENCES - 1]
            reconciled.append(
                replace(
                    item,
                    status=WorkStatus.COMPLETED,
                    outcome=WorkOutcome.HANDED_OFF,
                    next_operation=_terminal_operation(),
                    evidence=(
                        "A successful human-agent transfer handed off this unresolved outcome; "
                        "it does not prove the requested banking operation itself succeeded."
                    ),
                    evidence_refs=(*retained_references, transfer_reference),
                    obligations=tuple(
                        replace(
                            obligation,
                            status="not_applicable",
                            evidence="Responsibility handed to a human; not completed by the agent.",
                            evidence_refs=(transfer_reference,),
                        )
                        if obligation.status in {"ready", "waiting"}
                        else obligation
                        for obligation in item.obligations
                    ),
                )
            )

        semantic_rejection = (
            self._semantic_rejection_revision,
            set(self._semantic_reclassifiable_item_ids),
            self._semantic_required_new_item_mode,
            self._semantic_required_new_item_count,
            self._semantic_requires_operation_increase,
        )
        # A successful terminal transfer is authoritative environment evidence that
        # supersedes a model-authored plan-coverage correction. Requiring that stale
        # correction while installing the handoff state can turn a completed transfer
        # into an infrastructure error after the irreversible call already succeeded.
        self._semantic_rejection_revision = None
        self._semantic_reclassifiable_item_ids.clear()
        self._semantic_required_new_item_mode = None
        self._semantic_required_new_item_count = 0
        self._semantic_requires_operation_increase = False
        inputs = [_work_item_sync_input(item) for item in reconciled]
        if self.planning_mode == "outcomes":
            for item in inputs:
                item.pop("next_operation")
        result = self.invoke(
            SYNC_WORK_PLAN_TOOL,
            {"items": inputs},
        )
        if result.is_error:
            (
                self._semantic_rejection_revision,
                self._semantic_reclassifiable_item_ids,
                self._semantic_required_new_item_mode,
                self._semantic_required_new_item_count,
                self._semantic_requires_operation_increase,
            ) = semantic_rejection
            raise RuntimeError(f"Human-transfer reconciliation failed: {result.content}")
        final_disposition = ResponseDisposition(
            plan_revision=self.revision,
            items=tuple(
                ResponseItemDisposition(
                    item_id=item.item_id,
                    kind=ResponseDispositionKind.FINAL,
                )
                for item in self._items.values()
            ),
        )
        self._last_response_disposition = final_disposition
        self._pending_response_disposition = None

    def active_knowledge_references(self) -> tuple[WorkEvidenceReference, ...]:
        """Known revisions supporting unfinished work, not every historical citation.

        Prefer unfinished obligations to parent evidence. Completed research is
        still a source for its unfinished parent; finishing a lookup does not
        finish the requested operation. Other resolved obligations are released.
        """
        selected: dict[tuple[str, str], WorkEvidenceReference] = {}
        active = [item for item in self._items.values() if item.status != WorkStatus.COMPLETED]
        groups_conditions = [
            (o.condition.procedure,)
            for item in active
            for o in item.obligations
            if o.condition is not None
        ]
        groups = [
            obligation.evidence_refs
            for item in active
            for obligation in item.obligations
            if obligation.status in {"ready", "waiting"}
        ]
        groups.extend(item.evidence_refs for item in active)
        groups.extend(groups_conditions)
        groups.extend(
            obligation.evidence_refs
            for item in active
            for obligation in item.obligations
            if obligation.kind == "knowledge_search" and obligation.status == "resolved"
        )
        for references in groups:
            for reference in references:
                key = (reference.reference_id, reference.version)
                if reference.kind == "knowledge_revision" and key in self._knowledge_evidence:
                    selected.setdefault(key, self._knowledge_evidence[key])
        return tuple(selected.values())

    def diagnostic_counts(self) -> dict[str, int]:
        """Return content-free controller counters safe for persisted diagnostics."""

        return {
            "product_selection_transition_rejections": (
                self._product_selection_transition_rejections
            )
        }

    def progress_token(self) -> str:
        """Return a stable digest of useful controller and delivered-knowledge state."""

        workflow_state = self.export_state()
        workflow_state.pop("product_selection_transition_rejections")
        payload = {
            "knowledge_evidence": [
                reference.model_dump(mode="json", exclude_none=True)
                for _, reference in sorted(self._knowledge_evidence.items())
            ],
            "pending_response_disposition": (
                None
                if self._pending_response_disposition is None
                else self._pending_response_disposition.as_dict()
            ),
            "semantic_rejection_revision": self._semantic_rejection_revision,
            "semantic_reclassifiable_item_ids": sorted(self._semantic_reclassifiable_item_ids),
            "semantic_required_new_item_mode": (
                None
                if self._semantic_required_new_item_mode is None
                else self._semantic_required_new_item_mode.value
            ),
            "semantic_required_new_item_count": self._semantic_required_new_item_count,
            "semantic_requires_operation_increase": (self._semantic_requires_operation_increase),
            "workflow": workflow_state,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def review_terminal_output(self, text: str) -> TerminalOutputAdmission:
        """Authorize a response and record its controller-derived disposition."""

        admission = self.preview_terminal_output(text)
        if not admission.allowed:
            return admission
        if self.planning_mode == "outcomes":
            # Record the customer-message epoch, not a guessed question signature.
            for item in self._response_handoff_items():
                self._customer_response_handoff_sequences[item.item_id] = (
                    self._ledger.last_customer_message_sequence
                )
            return admission
        disposition = self._current_response_disposition()
        self._last_response_disposition = disposition
        self._pending_response_disposition = None
        for item in disposition.items:
            if (
                item.kind
                in {
                    ResponseDispositionKind.ASK_CUSTOMER,
                    ResponseDispositionKind.ASK_PRODUCT_SELECTION,
                }
                and self._items[item.item_id].completion_mode is WorkCompletionMode.OPERATION
            ):
                self._customer_response_handoffs[item.item_id] = (
                    self._customer_response_handoffs.get(item.item_id, 0) + 1
                )
                self._customer_response_handoff_signatures.setdefault(item.item_id, []).append(
                    _customer_response_handoff_signature(self._items[item.item_id])
                )
                self._customer_response_handoff_sequences[item.item_id] = (
                    self._ledger.last_customer_message_sequence
                )
        return admission

    def preview_terminal_output(self, text: str) -> TerminalOutputAdmission:
        """Check structural response readiness without consuming its disposition."""

        self.reconcile_conditions()
        if self.planning_mode == "outcomes":
            return self._preview_outcome_response()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("terminal output must be nonblank text")
        actionable = list(self._dependency_ready_actionable_items())
        if actionable:
            details = "; ".join(
                f"{item.item_id}: "
                f"{item.next_operation.tool_name or item.next_operation.kind.value} "
                f"({item.next_operation.summary})"
                for item in actionable[:6]
            )
            omitted = len(actionable) - min(len(actionable), 6)
            if omitted:
                details = f"{details}; plus {omitted} more item(s)"
            unavailable_search_note = (
                " The bounded knowledge-search capability is now exhausted or unavailable; "
                "revise the plan to use delivered evidence or another available operation."
                if not self._knowledge_search_available
                and any(
                    item.next_operation.kind is WorkOperationKind.KNOWLEDGE_SEARCH
                    for item in actionable
                )
                else ""
            )
            return TerminalOutputAdmission(
                allowed=False,
                reason=(
                    "Unresolved dependency-ready workflow items remain pending, in progress, "
                    f"or blocked. Do not send a customer-facing turn yet: {details}"
                    f"{unavailable_search_note}"
                ),
                correction_code="response_actionable_items",
                required_tool_names=_required_action_tool_names(
                    actionable,
                    knowledge_search_available=self._knowledge_search_available,
                ),
            )
        if not self._items:
            return TerminalOutputAdmission(
                allowed=False,
                reason=(
                    "Customer-facing text requires a populated workflow plan. Synchronize "
                    "every requested outcome before responding."
                ),
                correction_code="response_without_plan",
                required_tool_names=(SYNC_WORK_PLAN_TOOL,),
            )
        if self._semantic_rejection_revision == self.revision:
            return TerminalOutputAdmission(
                allowed=False,
                reason=(
                    "The current plan revision failed semantic coverage review. Revise the "
                    "work plan before preparing or emitting another response."
                ),
                correction_code="response_semantic_coverage_rejected",
                required_tool_names=(UPDATE_WORK_PLAN_TOOL,),
            )
        disposition = self._current_response_disposition()
        try:
            self._validate_response_disposition(disposition)
        except (WorkflowTransitionRejection, _ControllerValidationRejection, ValueError) as exc:
            return TerminalOutputAdmission(
                allowed=False,
                reason=f"The controller-derived response disposition is not ready: {exc}",
                correction_code=(
                    exc.remediation.code
                    if isinstance(exc, WorkflowTransitionRejection)
                    else (
                        exc.code
                        if isinstance(exc, _ControllerValidationRejection)
                        else "response_disposition_not_ready"
                    )
                ),
                required_tool_names=(UPDATE_WORK_PLAN_TOOL,),
            )
        return TerminalOutputAdmission(allowed=True)

    def _preview_outcome_response(self) -> TerminalOutputAdmission:
        if not self._items:
            return TerminalOutputAdmission(
                allowed=False,
                reason="Create a checklist of the customer's requested outcomes.",
                correction_code="response_without_plan",
                required_tool_names=(SYNC_WORK_PLAN_TOOL,),
            )
        actionable = [
            item.item_id
            for item in self._items.values()
            if item.status in {WorkStatus.PENDING, WorkStatus.IN_PROGRESS}
            and all(self._items[d].status is WorkStatus.COMPLETED for d in item.depends_on)
            and not self._has_ready_communication(item)
        ]
        actionable.extend(
            f"{item.item_id}/{obligation.obligation_id}"
            for item in self._items.values()
            for obligation in item.obligations
            if obligation.status == "ready"
            and (
                obligation.kind == "action"
                or (obligation.kind == "knowledge_search" and self._knowledge_search_available)
            )
        )
        if actionable and not self._ready_customer_input_items():
            return TerminalOutputAdmission(
                allowed=False,
                reason=(
                    f"Unfinished actionable outcomes: {', '.join(actionable)}. Continue "
                    "working if you can execute now. If customer input or authorization prevents "
                    "execution, call controller_update_work_plan to mark ALL affected obligations "
                    "waiting with policy/customer evidence; leave independent work ready. A necessary "
                    "question can use kind customer_input with real evidence and an awaiting_customer "
                    "parent; that handoff pauses unrelated work without completing it. Do not "
                    "claim success for unresolved work. Classify document lookups as kind "
                    "knowledge_search so the runtime can distinguish exhausted search capacity "
                    "from executable actions. Do not relabel a banking action as a lookup."
                ),
                correction_code="response_actionable_items",
                required_tool_names=(),
            )
        for item in self._response_handoff_items():
            if (
                self._customer_response_handoff_sequences.get(item.item_id)
                == self._ledger.last_customer_message_sequence
            ):
                return TerminalOutputAdmission(
                    allowed=False,
                    reason="This customer handoff was already delivered; reconcile new evidence.",
                    correction_code="response_duplicate_customer_handoff",
                    required_tool_names=(UPDATE_WORK_PLAN_TOOL,),
                )
        return TerminalOutputAdmission(allowed=True)

    def _response_handoff_items(self) -> tuple[WorkItem, ...]:
        # Input-only mode authorizes those questions, not delivery of unrelated
        # messages that merely happen to be ready in the same plan.
        questions = self._ready_customer_input_items()
        if questions:
            return questions
        return tuple(
            item
            for item in self._items.values()
            if item.status is WorkStatus.AWAITING_CUSTOMER or self._has_ready_communication(item)
        )

    @staticmethod
    def _has_ready_communication(item: WorkItem) -> bool:
        # A runtime turn ending is not the customer's overall task ending. A
        # response can perform a recorded communication without falsely marking
        # it delivered in advance. Independently ready actions still block above.
        return any(
            o.kind in {"communication", "customer_input"} and o.status == "ready"
            for o in item.obligations
        )

    def _ready_customer_input_items(self) -> tuple[WorkItem, ...]:
        """Explicit input handoffs, not permission to declare the task complete.

        The model supplies the semantic need; normal validation binds its evidence
        to observations. A question cannot skip its own executable preparation.
        Other outcomes remain untouched and resume when new customer input arrives.
        """
        questions = tuple(
            item
            for item in self._items.values()
            if item.status is WorkStatus.AWAITING_CUSTOMER
            and any(o.kind == "customer_input" and o.status == "ready" for o in item.obligations)
        )
        if any(
            o.status == "ready"
            and (
                o.kind == "action"
                or (o.kind == "knowledge_search" and self._knowledge_search_available)
            )
            for item in questions
            for o in item.obligations
        ):
            return ()
        return questions

    def _current_response_disposition(self) -> ResponseDisposition:
        """Return an explicit compatible declaration or derive one from accepted state."""

        pending = self._pending_response_disposition
        if pending is not None and pending.plan_revision == self.revision:
            return pending
        awaiting = tuple(
            item for item in self._items.values() if item.status is WorkStatus.AWAITING_CUSTOMER
        )
        if awaiting:
            dispositions = tuple(
                ResponseItemDisposition(
                    item_id=item.item_id,
                    kind=(
                        ResponseDispositionKind.ASK_PRODUCT_SELECTION
                        if item.next_operation.kind is WorkOperationKind.CUSTOMER_PRODUCT_SELECTION
                        else (
                            ResponseDispositionKind.ASK_CUSTOMER
                            if item.next_operation.kind is WorkOperationKind.CUSTOMER_RESPONSE
                            else ResponseDispositionKind.HANDOFF_CUSTOMER_TOOL
                        )
                    ),
                )
                for item in awaiting
            )
        else:
            dispositions = tuple(
                ResponseItemDisposition(
                    item_id=item.item_id,
                    kind=ResponseDispositionKind.FINAL,
                )
                for item in self._items.values()
            )
        return ResponseDisposition(plan_revision=self.revision, items=dispositions)

    def _dependency_ready_actionable_items(self) -> tuple[WorkItem, ...]:
        """Return incomplete agent-owned work whose prerequisites are complete."""

        return tuple(
            item
            for item in self._items.values()
            if item.status in {WorkStatus.PENDING, WorkStatus.IN_PROGRESS, WorkStatus.BLOCKED}
            and all(
                self._items[dependency].status is WorkStatus.COMPLETED
                for dependency in item.depends_on
            )
        )

    def reject_semantic_plan(
        self,
        *,
        reclassifiable_item_ids: Iterable[str] = (),
        required_new_item_mode: WorkCompletionMode | None = None,
        required_new_item_count: int = 0,
        require_operation_increase: bool = False,
    ) -> None:
        """Invalidate response authority and bind the next revision to its repair class."""

        if not self._items:
            raise RuntimeError("Cannot reject semantic coverage for an empty plan.")
        if (
            required_new_item_mode is not None
            and type(required_new_item_mode) is not WorkCompletionMode
        ):
            raise TypeError("required_new_item_mode must be a WorkCompletionMode or null.")
        if type(require_operation_increase) is not bool:
            raise TypeError("require_operation_increase must be a boolean.")
        if (
            type(required_new_item_count) is not int
            or not 0 <= required_new_item_count <= _MAX_WORK_ITEMS
        ):
            raise ValueError("required_new_item_count is outside the workflow item bound.")
        if (required_new_item_mode is None) != (required_new_item_count == 0):
            raise ValueError(
                "A required new-item mode and positive count must be supplied together."
            )
        if required_new_item_mode is not None and require_operation_increase:
            raise ValueError(
                "A semantic repair must require either a new item or reclassification."
            )
        candidate_ids = set(reclassifiable_item_ids)
        if not candidate_ids.issubset(self._items):
            raise ValueError("reclassifiable semantic item ids must exist in the current plan.")
        if candidate_ids and not require_operation_increase:
            raise ValueError("reclassifiable item ids require an operation-mode increase.")
        self._semantic_rejection_revision = self.revision
        self._semantic_reclassifiable_item_ids = candidate_ids
        self._semantic_required_new_item_mode = required_new_item_mode
        self._semantic_required_new_item_count = required_new_item_count
        self._semantic_requires_operation_increase = require_operation_increase
        self._pending_response_disposition = None

    def semantic_review_state(self) -> dict[str, Any]:
        """Return immutable plan semantics without operational or evidence content."""

        return {
            "items": [
                {
                    "completion_mode": item.completion_mode.value,
                    "goal": item.goal,
                    "id": item.item_id,
                }
                for item in sorted(self._items.values(), key=lambda candidate: candidate.item_id)
            ],
        }

    def response_review_state(self) -> dict[str, Any]:
        """Return the complete controller-derived contract for one public response."""

        disposition = self._current_response_disposition()
        visible_human_requests = self._ledger.visible_human_transfer_request_count
        transfer_threshold_applies = self._human_transfer_request_threshold_applies()
        return {
            "human_transfer_request_policy": {
                "minimum_visible_request_count": (
                    _HUMAN_TRANSFER_REQUEST_THRESHOLD if transfer_threshold_applies else 1
                ),
                "threshold_satisfied": (
                    visible_human_requests
                    >= (_HUMAN_TRANSFER_REQUEST_THRESHOLD if transfer_threshold_applies else 1)
                ),
                "visible_request_count": visible_human_requests,
            },
            "items": [
                {
                    "completion_mode": self._items[item.item_id].completion_mode.value,
                    "evidence_refs": [
                        reference.model_dump(mode="json", exclude_none=True)
                        for reference in self._items[item.item_id].evidence_refs
                    ],
                    "goal": self._items[item.item_id].goal,
                    "id": item.item_id,
                    "next_operation": self._items[item.item_id].next_operation.as_dict(),
                    "outcome": self._items[item.item_id].outcome.value,
                    "response_kind": item.kind.value,
                    "status": self._items[item.item_id].status.value,
                }
                for item in disposition.items
            ],
            "plan_revision": disposition.plan_revision,
        }

    def review(self, calls: Sequence[ToolCallPart]) -> tuple[ExternalToolAdmission, ...]:
        """Require populated workflow plans to authorize exact next tool calls."""

        self.reconcile_conditions()
        if gaps := self.invalid_comparison_bindings():
            return tuple(
                ExternalToolAdmission(
                    tool_call_id=call.tool_call_id,
                    allowed=False,
                    reason="A resolved comparison has incomplete or unavailable declared coverage. "
                    "Reopen its obligation before continuing external work: " + ", ".join(gaps),
                    diagnostic_code="comparison_coverage_invalidated",
                )
                for call in calls
            )
        if self.planning_mode == "outcomes" and self._items:
            # Tool schema, authorization, evidence and duplicate-effect checks remain
            # the admission gate's responsibility. A supplied action packet additionally
            # binds a mutation to the exact tool and arguments recorded in the outcome.
            decisions = []
            for call in calls:
                packet = self._review_planned_tool_call(call)
                decisions.append(
                    packet if not packet.allowed else self._review_condition_call(call)
                )
            return tuple(decisions)
        if not self._items:
            return tuple(
                ExternalToolAdmission(
                    tool_call_id=call.tool_call_id,
                    allowed=False,
                    reason=(
                        "External tools require a populated workflow plan. Synchronize every "
                        "requested outcome and its exact next operation before issuing a tool "
                        "call."
                    ),
                    diagnostic_code="workflow_plan_missing",
                )
                for call in calls
            )
        if self._semantic_rejection_revision == self.revision:
            return tuple(
                ExternalToolAdmission(
                    tool_call_id=call.tool_call_id,
                    allowed=False,
                    reason=(
                        "The current workflow revision failed semantic coverage review. "
                        "Revise the plan before issuing any environment call."
                    ),
                    diagnostic_code="workflow_semantic_rejected",
                )
                for call in calls
            )
        available_by_tool: dict[str, list[WorkItem]] = {}
        for item in self._items.values():
            if item.status not in {
                WorkStatus.PENDING,
                WorkStatus.IN_PROGRESS,
                WorkStatus.BLOCKED,
            }:
                continue
            if any(
                self._items[dependency].status is not WorkStatus.COMPLETED
                for dependency in item.depends_on
            ):
                continue
            operation = item.next_operation
            if (
                operation.owner is WorkOwner.AGENT
                and operation.kind is WorkOperationKind.TOOL
                and not self._planned_agent_tool_result_is_resolved(item)
            ):
                available_by_tool.setdefault(operation.tool_name, []).append(item)

        admissions: list[ExternalToolAdmission] = []
        for call in calls:
            candidates = available_by_tool.get(call.tool_name)
            matched_index = next(
                (
                    index
                    for index, item in enumerate(candidates or ())
                    if _json_objects_equal(item.next_operation.arguments, call.arguments)
                ),
                None,
            )
            if candidates is not None and matched_index is not None:
                if (
                    call.tool_name == HUMAN_TRANSFER_TOOL
                    and self._human_transfer_request_threshold_applies()
                    and self._ledger.visible_human_transfer_request_count
                    < _HUMAN_TRANSFER_REQUEST_THRESHOLD
                ):
                    admissions.append(
                        ExternalToolAdmission(
                            tool_call_id=call.tool_call_id,
                            allowed=False,
                            reason=(
                                "This multi-outcome or repeated-request conversation has not "
                                "reached the customer's fourth visible human-assistance "
                                "request. Continue the automated work, or reconcile completed "
                                "and explicitly declined operations; do not transfer yet."
                            ),
                            diagnostic_code="workflow_human_transfer_request_threshold",
                        )
                    )
                    continue
                candidates.pop(matched_index)
                admissions.append(
                    ExternalToolAdmission(tool_call_id=call.tool_call_id, allowed=True)
                )
                continue
            admissions.append(
                ExternalToolAdmission(
                    tool_call_id=call.tool_call_id,
                    allowed=False,
                    reason=(
                        f"No dependency-ready workflow item has {call.tool_name!r} as its next "
                        "agent-owned tool operation with these exact arguments. Synchronize the "
                        "plan before issuing it."
                    ),
                    diagnostic_code="workflow_operation_mismatch",
                )
            )
        return tuple(admissions)

    def _review_planned_tool_call(self, call: ToolCallPart) -> ExternalToolAdmission:
        """Check optional outcome-mode mutation packets without routing all tools."""

        declared = []
        for item in self._items.values():
            if item.status is WorkStatus.COMPLETED:
                continue
            if any(
                self._items[dependency].status is not WorkStatus.COMPLETED
                for dependency in item.depends_on
            ):
                continue
            for obligation in item.obligations:
                if (
                    obligation.kind == "action"
                    and obligation.status == "ready"
                    and obligation.planned_tool is not None
                    and all(
                        item.obligations[index].status in {"resolved", "not_applicable"}
                        for index, candidate in enumerate(item.obligations)
                        if candidate.obligation_id in obligation.depends_on
                    )
                ):
                    declared.append(
                        (item.item_id, obligation.obligation_id, obligation.planned_tool)
                    )
        matching_name = [entry for entry in declared if entry[2].tool_name == call.tool_name]
        if not matching_name:
            return ExternalToolAdmission(tool_call_id=call.tool_call_id, allowed=True)
        if any(
            _json_objects_equal(packet.arguments, call.arguments) for _, _, packet in matching_name
        ):
            return ExternalToolAdmission(tool_call_id=call.tool_call_id, allowed=True)
        labels = ", ".join(
            f"{item_id}/{obligation_id}" for item_id, obligation_id, _ in matching_name
        )
        return ExternalToolAdmission(
            tool_call_id=call.tool_call_id,
            allowed=False,
            reason=(
                f"The action packet for {labels} declares different arguments for "
                f"{call.tool_name!r}. Reconcile the target and arguments before executing it."
            ),
            diagnostic_code="outcome_action_packet_mismatch",
        )

    def _has_unfinished_nontransfer_operation(self) -> bool:
        """Return whether handoff would replace a separate requested operation."""

        return any(
            item.status is not WorkStatus.COMPLETED
            and item.completion_mode is WorkCompletionMode.OPERATION
            and not (
                item.next_operation.owner is WorkOwner.AGENT
                and item.next_operation.kind is WorkOperationKind.TOOL
                and item.next_operation.tool_name == HUMAN_TRANSFER_TOOL
            )
            for item in self._items.values()
        )

    def _human_transfer_request_threshold_applies(self) -> bool:
        """Return whether this dialogue requires four visible transfer requests."""

        if self._verified_capability_gap is not None and self._verified_capability_gap() is True:
            # Public policy distinguishes an unsupported request from one the
            # automated agent can resolve. Still do not abandon a concrete bank
            # action (including a pending customer tool) for the capability gap.
            concrete_work = any(
                item.status is not WorkStatus.COMPLETED
                and item.next_operation.kind is WorkOperationKind.TOOL
                and item.next_operation.tool_name != HUMAN_TRANSFER_TOOL
                for item in self._items.values()
            )
            if not concrete_work:
                return False
        visible_requests = self._ledger.visible_human_transfer_request_count
        return visible_requests > 0 and (
            visible_requests > 1 or self._has_unfinished_nontransfer_operation()
        )

    def observe_knowledge_evidence(
        self,
        references: Sequence[WorkEvidenceReference],
    ) -> None:
        """Register exact Cayu revisions delivered to the model this session."""

        if isinstance(references, str | bytes):
            raise TypeError("references must be a sequence of WorkEvidenceReference values.")
        for reference in references:
            if type(reference) is not WorkEvidenceReference:
                raise TypeError("Every knowledge reference must be a WorkEvidenceReference.")
            if reference.kind != EvidenceKind.KNOWLEDGE_REVISION.value:
                raise ValueError("Only knowledge_revision evidence can be registered here.")
            if not reference.available or reference.version is None:
                raise ValueError("Knowledge evidence must be available and revision-pinned.")
            self._remember_knowledge_evidence(reference)

    def has_delivered_knowledge_revision(self, entry_id: str, version: str) -> bool:
        """Exact evidence availability, not semantic relevance or entailment."""
        return (entry_id, version) in self._knowledge_evidence

    def delivered_knowledge_reference(
        self, entry_id: str, version: str
    ) -> WorkEvidenceReference | None:
        """Return already accepted immutable evidence, never infer delivery from a locator."""
        return self._knowledge_evidence.get((entry_id, version))

    def invoke_model_update(self, name: str, arguments: Mapping[str, Any]) -> InternalToolResult:
        """Keep an invalid optional checklist patch from becoming a repair lock.

        The strict transaction still validates everything and never installs the
        invalid state. Actual operation admission and internal reconciliation use
        their existing strict paths. A warning is not permission to execute.
        """
        if name == UPDATE_WORK_PLAN_TOOL and self.planning_mode == "outcomes":
            try:
                arguments = self._route_obligation_patches(arguments)
            except ValueError as error:
                return InternalToolResult(content=str(error), is_error=True)
        pending = dict(self._pending_remediations)
        result = self.invoke(name, arguments)
        advisory_codes = {
            "sync_invalid_arguments",
            "sync_invalid_items",
            *_REVISION_REPAIR_CLASSES,
        }
        if (
            self.planning_mode != "outcomes"
            or name not in {SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL}
            or not result.is_error
            or result.diagnostic_code not in advisory_codes
        ):
            return result
        self._pending_remediations = pending
        fingerprint = sha256(
            json.dumps([self.revision, result.diagnostic_code, result.content]).encode()
        ).hexdigest()
        repeated = fingerprint in self._plan_advisories
        if not repeated:
            self._plan_advisories = [*self._plan_advisories[-63:], fingerprint]
        return InternalToolResult(
            content=json.dumps(
                {
                    "status": "not_applied",
                    "advisory_code": "plan_update_advisory_repeated"
                    if repeated
                    else "plan_update_advisory",
                    "plan_revision": self.revision,
                    "reason": "Already reported for this unchanged plan."
                    if repeated
                    else result.content,
                    "instruction": "The last valid plan is unchanged. Do not repeat the unchanged update. "
                    "Continue independent permitted work; revise the checklist only when needed to record "
                    "new information. This grants no execution permission and marks no work complete.",
                }
            ),
        )

    def _route_obligation_patches(self, arguments):
        """Resolve existing unique obligation IDs; never invent a new outcome.

        This is an input-shape repair only. Ordinary atomic evidence, dependency
        and completion validation still runs on the reconstructed nested patch.
        """
        if type(arguments) is not dict or set(arguments) != {"items"} or type(arguments["items"]) is not list:
            return arguments
        items = self.export_state()["items"]
        parents = {item["id"] for item in items}
        owners = {}
        for item in items:
            for obligation in item.get("obligations", []):
                owners.setdefault(obligation["id"], []).append(item["id"])
        result, nested = [], {}
        for raw in arguments["items"]:
            if type(raw) is not dict or raw.get("id") in parents or raw.get("id") not in owners:
                result.append(raw)
                continue
            matches = owners[raw["id"]]
            if len(matches) != 1:
                raise ValueError("Ambiguous obligation id; nest the patch under its explicit parent outcome.")
            child = dict(raw)
            if "completion_mode" in child or "outcome" in child:
                if child.get("outcome") != "succeeded" or child.get("status") != "completed":
                    raise ValueError("An obligation is not a top-level outcome; patch its status under its parent.")
                child.pop("completion_mode", None)
                child.pop("outcome", None)
            if child.get("status") == "completed":
                child["status"] = "resolved"
            nested.setdefault(matches[0], []).append(child)
        explicit = {item.get("id") for item in result if type(item) is dict}
        if explicit & nested.keys():
            raise ValueError("Combine parent and child updates in one explicit nested outcome patch.")
        return {"items": [*result, *({"id": parent, "obligations": children} for parent, children in nested.items())]}

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> InternalToolResult:
        """Validate and atomically install a complete or incrementally merged revision."""

        # applied_to is a hook-owned receipt, never a model-writable plan field.
        supplied = arguments.get("items")
        if isinstance(supplied, list) and any(
            isinstance(item, dict)
            and isinstance(item.get("obligations"), list)
            and any(isinstance(o, dict) and "applied_to" in o for o in item["obligations"])
            for item in supplied
        ):
            return InternalToolResult(
                content="applied_to is runtime-owned; omit it from plan patches.", is_error=True
            )
        if name == PREPARE_RESPONSE_TOOL:
            return self._prepare_response(arguments)
        if name not in {SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL}:
            return InternalToolResult(
                content=f"Unknown controller tool {name!r}.",
                is_error=True,
                diagnostic_code="controller_unknown_tool",
            )
        if name == UPDATE_WORK_PLAN_TOOL and not self._items:
            return InternalToolResult(
                content="Create the initial complete plan with controller_sync_work_plan first.",
                is_error=True,
                diagnostic_code="update_requires_existing_plan",
            )
        if not isinstance(arguments, Mapping) or set(arguments) != {"items"}:
            return InternalToolResult(
                content="Expected exactly one 'items' array.",
                is_error=True,
                diagnostic_code="sync_invalid_arguments",
            )
        raw_items = arguments["items"]
        if type(raw_items) is not list:
            return InternalToolResult(
                content="'items' must be an array.",
                is_error=True,
                diagnostic_code="sync_invalid_items",
            )
        if not raw_items:
            return InternalToolResult(
                content="'items' must contain at least one work item.",
                is_error=True,
                diagnostic_code="sync_invalid_items",
            )
        validation_code = "sync_invalid_items"
        try:
            if self.planning_mode == "outcomes":
                raw_items = self._expand_outcome_items(
                    raw_items, updating=name == UPDATE_WORK_PLAN_TOOL
                )
            candidate_inputs = (
                self._merge_plan_update_inputs(raw_items)
                if name == UPDATE_WORK_PLAN_TOOL
                else raw_items
            )
            items = self._parse_items(candidate_inputs, validate_references=True)
            validation_code = "sync_revision_rejected"
            self._validate_revision(items)
            validation_code = "sync_semantic_repair_rejected"
            self._validate_semantic_repair(items)
            validation_code = "sync_tool_evidence_rejected"
            tool_evidence_floors = self._candidate_tool_evidence_floors(items)
            validation_code = "sync_completed_operations_rejected"
            self._validate_completed_operations(items, tool_evidence_floors)
            validation_code = "sync_customer_evidence_rejected"
            customer_evidence_floors = self._candidate_customer_evidence_floors(items)
            self._validate_customer_evidence_timing(items, customer_evidence_floors)
        except WorkflowTransitionRejection as exc:
            self._register_remediation(exc.remediation)
            return InternalToolResult(
                content=f"Plan rejected: {exc}",
                is_error=True,
                diagnostic_code=exc.remediation.code,
                remediation=exc.remediation,
            )
        except _ControllerValidationRejection as exc:
            return InternalToolResult(
                content=f"Plan rejected: {exc}",
                is_error=True,
                diagnostic_code=exc.code,
            )
        except ValueError as exc:
            return InternalToolResult(
                content=f"Plan rejected: {exc}",
                is_error=True,
                diagnostic_code=validation_code,
            )

        try:
            resolutions = self._candidate_remediation_resolutions(items)
        except WorkflowTransitionRejection as exc:
            self._register_remediation(exc.remediation)
            return InternalToolResult(
                content=f"Plan rejected: {exc}",
                is_error=True,
                diagnostic_code=exc.remediation.code,
                remediation=exc.remediation,
            )

        if self._semantic_rejection_revision == self.revision and _semantic_item_identity(
            items
        ) == _semantic_item_identity(self._items.values()):
            return InternalToolResult(
                content=(
                    f"Plan revision {self.revision} still has the semantic coverage rejection. "
                    "Add the missing or correctly classified outcome; changing status, evidence, "
                    "dependencies, or operation details cannot address this rejection."
                ),
                is_error=True,
                diagnostic_code="sync_semantic_coverage_rejected",
            )
        proposed_items = {item.item_id: item for item in items}
        if (
            proposed_items == self._items
            and customer_evidence_floors == self._customer_evidence_floors
            and tool_evidence_floors == self._tool_evidence_floors
        ):
            for resolution in resolutions:
                self._pending_remediations.pop(resolution.correction_id)
            return InternalToolResult(
                content=(
                    f"Plan revision {self.revision} is already current; no planning state changed."
                ),
                resolutions=resolutions,
            )
        for item in items:
            previous = self._items.get(item.item_id)
            if previous is not None and self._is_information_reopening(previous, item):
                self._previous_information_completions.setdefault(item.item_id, []).append(
                    {
                        "revision": self.revision,
                        "item": previous.as_dict(),
                    }
                )
        self._items = proposed_items
        self._customer_evidence_floors = customer_evidence_floors
        self._customer_response_handoffs = {
            item.item_id: self._customer_response_handoffs.get(item.item_id, 0) for item in items
        }
        self._customer_response_handoff_signatures = {
            item.item_id: list(self._customer_response_handoff_signatures.get(item.item_id, ()))
            for item in items
        }
        self._customer_response_handoff_sequences = {
            item.item_id: self._customer_response_handoff_sequences.get(item.item_id, 0)
            for item in items
        }
        self._tool_evidence_floors = tool_evidence_floors
        self._pending_response_disposition = None
        self.revision += 1
        self._semantic_rejection_revision = None
        self._semantic_reclassifiable_item_ids.clear()
        self._semantic_required_new_item_mode = None
        self._semantic_required_new_item_count = 0
        self._semantic_requires_operation_increase = False
        for resolution in resolutions:
            self._pending_remediations.pop(resolution.correction_id)
        return InternalToolResult(
            content=(
                f"Plan revision {self.revision} accepted for {len(self._items)} work item(s). "
                "This changed only private planning memory."
            ),
            resolutions=resolutions,
        )

    def _prepare_response(self, arguments: Mapping[str, Any]) -> InternalToolResult:
        """Validate a one-use response declaration against the current plan revision."""

        if self._semantic_rejection_revision == self.revision:
            return InternalToolResult(
                content=(
                    f"Response disposition rejected: plan revision {self.revision} failed "
                    "semantic coverage review and must be changed first."
                ),
                is_error=True,
                diagnostic_code="response_semantic_coverage_rejected",
            )
        try:
            disposition = _parse_response_disposition(arguments)
            self._validate_response_disposition(disposition)
        except WorkflowTransitionRejection as exc:
            self._register_remediation(exc.remediation)
            return InternalToolResult(
                content=f"Response disposition rejected: {exc}",
                is_error=True,
                diagnostic_code=exc.remediation.code,
                remediation=exc.remediation,
            )
        except _ControllerValidationRejection as exc:
            return InternalToolResult(
                content=f"Response disposition rejected: {exc}",
                is_error=True,
                diagnostic_code=exc.code,
            )
        except ValueError as exc:
            return InternalToolResult(
                content=f"Response disposition rejected: {exc}",
                is_error=True,
                diagnostic_code="response_invalid_arguments",
            )
        if disposition == self._pending_response_disposition:
            return InternalToolResult(
                content=(
                    f"Response disposition for plan revision {self.revision} is already "
                    "prepared; no controller state changed."
                )
            )
        self._pending_response_disposition = disposition
        return InternalToolResult(
            content=(
                f"Response disposition accepted for plan revision {self.revision}. Emit the "
                "declared customer-facing response now; this authorization is consumed once."
            )
        )

    def _validate_response_disposition(self, disposition: ResponseDisposition) -> None:
        if not self._items:
            raise _ControllerValidationRejection(
                "response_without_plan",
                "a response disposition is unnecessary without a work plan",
            )
        if disposition.plan_revision != self.revision:
            raise _ControllerValidationRejection(
                "response_stale_revision",
                f"plan_revision must equal the current revision {self.revision}",
            )
        declared = {item.item_id: item.kind for item in disposition.items}
        unknown = set(declared) - set(self._items)
        if unknown:
            raise _ControllerValidationRejection(
                "response_unknown_items",
                f"unknown work item ids: {', '.join(sorted(unknown))}",
            )
        actionable = {item.item_id for item in self._dependency_ready_actionable_items()}
        if actionable:
            raise _ControllerValidationRejection(
                "response_actionable_items",
                "dependency-ready agent-owned workflow items remain actionable: "
                f"{', '.join(sorted(actionable))}",
            )
        awaiting = {
            item.item_id: item
            for item in self._items.values()
            if item.status is WorkStatus.AWAITING_CUSTOMER
        }
        if awaiting:
            if set(declared) != set(awaiting):
                raise _ControllerValidationRejection(
                    "response_awaiting_coverage",
                    "response disposition must cover every awaiting-customer item exactly once",
                )
            for item_id, item in awaiting.items():
                expected = (
                    ResponseDispositionKind.ASK_PRODUCT_SELECTION
                    if item.next_operation.kind is WorkOperationKind.CUSTOMER_PRODUCT_SELECTION
                    else (
                        ResponseDispositionKind.ASK_CUSTOMER
                        if item.next_operation.kind is WorkOperationKind.CUSTOMER_RESPONSE
                        else ResponseDispositionKind.HANDOFF_CUSTOMER_TOOL
                    )
                )
                if declared[item_id] is not expected:
                    raise _ControllerValidationRejection(
                        "response_disposition_mismatch",
                        f"item {item_id!r} requires disposition {expected.value!r}",
                    )
                if (
                    expected
                    in {
                        ResponseDispositionKind.ASK_CUSTOMER,
                        ResponseDispositionKind.ASK_PRODUCT_SELECTION,
                    }
                    and item.completion_mode is WorkCompletionMode.OPERATION
                ):
                    self._validate_operation_customer_handoff(item)
            return
        expected_ids = set(self._items)
        if set(declared) != expected_ids or any(
            kind is not ResponseDispositionKind.FINAL for kind in declared.values()
        ):
            raise _ControllerValidationRejection(
                "response_final_coverage",
                "a final response must declare every completed item as final",
            )

    def _validate_operation_customer_handoff(self, item: WorkItem) -> None:
        """Permit advancing questions while rejecting stalled clarification loops."""

        count = self._customer_response_handoffs.get(item.item_id, 0)
        last_sequence = self._customer_response_handoff_sequences.get(item.item_id, 0)
        if count and self._ledger.last_customer_message_sequence <= last_sequence:
            raise _ControllerValidationRejection(
                "response_without_new_customer_input",
                f"item {item.item_id!r} cannot emit another question before a new customer "
                "message; use the response already sent",
            )
        signature = _customer_response_handoff_signature(item)
        prior_signatures = self._customer_response_handoff_signatures.get(item.item_id, ())
        if prior_signatures.count(signature) >= _MAX_SAME_PROMPT_OPERATION_CUSTOMER_HANDOFFS:
            raise _ControllerValidationRejection(
                "response_repeated_clarification",
                f"item {item.item_id!r} already emitted this normalized planned operation "
                "question twice; advance to a different required fact, an executable agent "
                "tool, a customer-side tool handoff, evidenced completion or refusal, or an "
                "agent-owned recovery or escalation",
            )
        if count >= _MAX_OPERATION_CUSTOMER_HANDOFFS:
            raise _ControllerValidationRejection(
                "response_clarification_budget",
                f"item {item.item_id!r} exhausted its {_MAX_OPERATION_CUSTOMER_HANDOFFS}-turn "
                "operation clarification safety ceiling; advance to an executable agent tool, "
                "a customer-side tool handoff, evidenced completion or refusal, or an "
                "agent-owned recovery or escalation",
            )

    def render_action_review_context(self) -> str:
        """Stable decision inputs: work semantics and evidence, not mutable wording.

        The verifier and its cache must see the same projection. Excluding free
        evidence prose, operation summaries and revision counters prevents cosmetic
        edits from obtaining another semantic verdict while legitimate state changes
        invalidate decisions about readiness and remaining work.
        """
        items = []
        for item in sorted(self._items.values(), key=lambda item: item.item_id):
            projected = _work_item_sync_input(item)
            projected.pop("evidence", None)
            projected["next_operation"].pop("summary", None)
            items.append(projected)
        state = {
            "schema_version": 1,
            "items": items,
            "human_transfer_request_policy": self.response_review_state()[
                "human_transfer_request_policy"
            ],
        }
        return (
            "<workflow-plan>\n"
            + json.dumps(state, ensure_ascii=False, sort_keys=True)
            + "\n</workflow-plan>"
        )

    def render_context(self, *, advisory: bool = False) -> str:
        self.reconcile_conditions()
        if getattr(self, "lightweight_advisory", False):
            from cayu_tau3_banking.workflow_profile import outcome_context
            return outcome_context(self)
        if self.planning_mode == "outcomes":
            items = [item.as_dict() for item in self._items.values()]
            for item in items:
                item.pop("next_operation")
                # The snapshot remains lossless in export_state. Model context
                # supplies references it can actually copy into the tool schema.
                references = list(item["evidence_refs"])
                for obligation in item.get("obligations", []):
                    references.extend(obligation["evidence_refs"])
                    if obligation.get("condition"):
                        references.append(obligation["condition"]["procedure"])
                for reference in references:
                    reference.pop("digest", None)
                    reference.pop("available", None)
            return (
                "Recorded, model-maintained outcome checklist. Not bank policy or execution proof. "
                "Choose tools directly; update only material outcome/status changes. "
                "Dependencies constrain final effects, not independent preparatory obligations. "
                "Check policy for omitted obligations; this checklist is not proof of coverage. "
                "Blocked/waiting items are unresolved, not successful.\n<workflow-plan>\n"
                + json.dumps(
                    {
                        "revision": self.revision,
                        "items": items,
                        "knowledge_search_available": self._knowledge_search_available,
                        "response_contract": {
                            "mode": "advisory"
                            if advisory
                            else "customer_input_only"
                            if self._ready_customer_input_items()
                            else "standard_readiness_check",
                            "input_outcome_ids": [
                                i.item_id for i in self._ready_customer_input_items()
                            ],
                            "plan_initialized": bool(self._items),
                            # Tracked coverage is model-authored, not proof that
                            # every customer request has been represented.
                            "task_complete": None,
                            "tracked_items_complete": bool(self._items)
                            and all(i.status is WorkStatus.COMPLETED for i in self._items.values()),
                        },
                        "resource_blocked_obligations": [
                            f"{item.item_id}/{obligation.obligation_id}"
                            for item in self._items.values()
                            for obligation in item.obligations
                            if obligation.kind == "knowledge_search"
                            and obligation.status == "ready"
                            and not self._knowledge_search_available
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n</workflow-plan>"
            )
        context_state = self.export_state()
        context_state.pop("previous_information_completions", None)
        context_state["items"] = [_work_item_sync_input(item) for item in self._items.values()]
        ready_batches = self._ready_agent_tool_batches()
        if ready_batches:
            context_state["ready_agent_tool_batches"] = ready_batches
            context_state["ready_agent_tool_batches_instruction"] = (
                "Issue every call in each batch together in the next assistant tool-call "
                "response; then consolidate their results in one plan update."
            )
        return (
            f"{_WORKFLOW_CONTEXT_PREAMBLE}\n<workflow-plan>\n"
            f"{json.dumps(context_state, ensure_ascii=False, sort_keys=True)}\n"
            "</workflow-plan>"
        )

    def _ready_agent_tool_batches(self) -> list[dict[str, Any]]:
        """Render repeated dependency-ready operations as execution batches."""

        grouped: dict[tuple[str, str | None], list[WorkItem]] = {}
        for item in self._dependency_ready_actionable_items():
            operation = item.next_operation
            if (
                operation.owner is not WorkOwner.AGENT
                or operation.kind is not WorkOperationKind.TOOL
                or operation.tool_name in _SERIAL_ENVIRONMENT_TOOL_NAMES
                or self._planned_agent_tool_result_is_resolved(item)
            ):
                continue
            discoverable_name = None
            if operation.tool_name == "call_discoverable_agent_tool":
                candidate = operation.arguments.get("agent_tool_name")
                if not isinstance(candidate, str) or not candidate.strip():
                    continue
                discoverable_name = candidate.strip()
            grouped.setdefault((operation.tool_name, discoverable_name), []).append(item)
        batches: list[dict[str, Any]] = []
        for (tool_name, discoverable_name), items in grouped.items():
            if len(items) < 2:
                continue
            batch: dict[str, Any] = {
                "item_ids": [item.item_id for item in items],
                "tool_name": tool_name,
            }
            if discoverable_name is not None:
                batch["discoverable_agent_tool_name"] = discoverable_name
            batches.append(batch)
        return batches

    def _planned_agent_tool_result_is_resolved(self, item: WorkItem) -> bool:
        """Return whether the exact currently planned call already returned."""

        operation = item.next_operation
        if operation.owner is not WorkOwner.AGENT or operation.kind is not WorkOperationKind.TOOL:
            return False
        return (
            self._ledger.latest_resolved_tool_action(
                operation.tool_name,
                operation.arguments,
                after_sequence=self._tool_evidence_floors.get(item.item_id, 0),
            )
            is not None
        )

    def reconcile_conditions(self) -> None:
        """Reopen stale assessments, never completed effects or customer intent.

        No model call, inference from prose, or external action occurs here. The
        saved old evidence remains visible but no longer supports a resolved check.
        """
        changed = False
        for item_id, item in self._items.items():
            if item.status is WorkStatus.COMPLETED:
                continue
            obligations = {o.obligation_id: o for o in item.obligations}
            for identifier, obligation in obligations.items():
                condition = obligation.condition
                if (
                    condition is not None
                    and obligation.status == "resolved"
                    and obligation.applied_to is None
                    and (gap := condition.gap(self._ledger))
                ):
                    obligations[identifier] = replace(
                        obligation,
                        status="ready",
                        evidence=f"Recheck required: {gap}. Prior evidence retained, not current eligibility.",
                    )
            for _ in range(len(obligations)):
                reopened = False
                for identifier, obligation in obligations.items():
                    if obligation.status == "resolved" and any(
                        obligations[d].status not in {"resolved", "not_applicable"}
                        for d in obligation.depends_on
                    ):
                        obligations[identifier] = replace(
                            obligation,
                            status="ready",
                            evidence="Recheck required: a prerequisite changed. Prior conclusion is not current.",
                        )
                        reopened = True
                if not reopened:
                    break
            updated = tuple(obligations.values())
            if updated != item.obligations:
                self._items[item_id] = replace(item, obligations=updated)
                changed = True
        if changed:
            self.revision += 1
            self._pending_response_disposition = None

    def _review_condition_call(self, call: ToolCallPart) -> ExternalToolAdmission:
        name, arguments = call.tool_name, call.arguments
        if name in {"call_discoverable_agent_tool", "give_discoverable_user_tool"}:
            key = (
                "agent_tool_name"
                if name == "call_discoverable_agent_tool"
                else "discoverable_tool_name"
            )
            name = arguments.get(key, "")
            if "arguments" in arguments:
                try:
                    arguments = json.loads(arguments["arguments"])
                except (TypeError, ValueError):
                    arguments = {}
        scoped = [
            (item, obligation)
            for item in self._items.values()
            if item.status is not WorkStatus.COMPLETED
            for obligation in item.obligations
            if obligation.condition is not None and name in obligation.condition.for_tools
        ]
        # Explicit operation names scope a condition; it must not block unrelated
        # preparation, source reads, consent questions, or alternative operations.
        if not scoped:
            return ExternalToolAdmission(tool_call_id=call.tool_call_id, allowed=True)
        targets = resource_targets(arguments) if isinstance(arguments, dict) else set()
        has_scope = bool(targets) or (
            isinstance(arguments, dict) and isinstance(arguments.get("user_id"), str)
        )
        matching = [
            (item, o)
            for item, o in scoped
            if not has_scope or _condition_matches_call(o.condition, call, self._ledger)
        ]
        reason = None
        if has_scope and not matching:
            reason = "This operation targets a different resource from its recorded prerequisites. Bind and verify conditions for the actual resource before proceeding."
        else:
            gaps = [
                f"{item.item_id}/{o.obligation_id}: {gap}"
                for item, o in matching
                if (gap := self._condition_execution_gap(item, o)) is not None
            ]
            if gaps:
                reason = (
                    "This operation has unresolved recorded prerequisites: "
                    + ", ".join(gaps)
                    + ". Read/reassess the conditions, perform independent preparation, or ask for needed customer input."
                )
        return ExternalToolAdmission(
            tool_call_id=call.tool_call_id,
            allowed=reason is None,
            reason=reason,
            diagnostic_code="case_prerequisite_unresolved" if reason else None,
        )

    def _condition_execution_gap(self, item: WorkItem, obligation: WorkObligation) -> str | None:
        """Check permission independently of whether the described action is done.

        A ready action can carry its own prerequisites. Requiring it to be
        resolved before dispatch creates a circular completion requirement.
        Readiness still cannot waive dependencies, current evidence or consent;
        successful execution is recorded separately by the existing result hook.
        """
        if obligation.status == "not_applicable":
            return None  # Existing evidenced-waiver semantics are unchanged.
        if obligation.status not in {"ready", "resolved"}:
            return (
                "obligation_waiting; resolve the recorded reason before marking this action ready"
            )
        by_id = {o.obligation_id: o for o in item.obligations}
        pending = [
            dependency
            for dependency in obligation.depends_on
            if by_id[dependency].status not in {"resolved", "not_applicable"}
        ]
        if pending:
            return "unresolved_dependency: " + ", ".join(pending)
        assert obligation.condition is not None
        return obligation.condition.gap(self._ledger)

    def before_condition_execution(self, call: ToolCallPart) -> None:
        """Called only after final native admission, immediately before dispatch."""
        checks = tuple(
            (item.item_id, o.obligation_id, o.condition)
            for item in self._items.values()
            if item.status is not WorkStatus.COMPLETED
            for o in item.obligations
            if o.condition is not None
            and o.status in {"ready", "resolved"}
            and self._condition_execution_gap(item, o) is None
            and _condition_matches_call(o.condition, call, self._ledger)
        )
        if checks:
            if len(self._pending_condition_checks) >= 64:
                raise ValueError("too many pending condition execution receipts")
            self._pending_condition_checks[call.tool_call_id] = checks

    def after_condition_execution(self, call_id: str) -> None:
        checks = self._pending_condition_checks.pop(call_id, ())
        if self._ledger.tool_action_status(call_id) is not ToolActionStatus.SUCCEEDED:
            return
        changed = False
        for item_id, obligation_id, condition in checks:
            item = self._items[item_id]
            updated = tuple(
                replace(o, applied_to=call_id)
                if o.obligation_id == obligation_id
                and o.condition == condition
                and o.status in {"ready", "resolved"}
                else o
                for o in item.obligations
            )
            if updated != item.obligations:
                self._items[item_id] = replace(item, obligations=updated)
                changed = True
        if changed:
            self.revision += 1
            self._pending_response_disposition = None

    def export_state(self) -> dict[str, Any]:
        self.reconcile_conditions()
        return {
            **(
                {
                    "previous_information_completions": deepcopy(
                        self._previous_information_completions
                    )
                }
                if self._previous_information_completions
                else {}
            ),
            **({"plan_advisories": list(self._plan_advisories)} if self._plan_advisories else {}),
            **({"planning_mode": "outcomes"} if self.planning_mode == "outcomes" else {}),
            "customer_evidence_floors": dict(sorted(self._customer_evidence_floors.items())),
            "customer_response_handoffs": dict(sorted(self._customer_response_handoffs.items())),
            "customer_response_handoff_sequences": dict(
                sorted(self._customer_response_handoff_sequences.items())
            ),
            "customer_response_handoff_signatures": {
                item_id: list(signatures)
                for item_id, signatures in sorted(
                    self._customer_response_handoff_signatures.items()
                )
            },
            "items": [item.as_dict() for item in self._items.values()],
            "last_response_disposition": (
                None
                if self._last_response_disposition is None
                else self._last_response_disposition.as_dict()
            ),
            "product_selection_transition_rejections": (
                self._product_selection_transition_rejections
            ),
            "revision": self.revision,
            "schema_version": 19
            if self._previous_information_completions
            else 18
            if any(
                o.condition is not None or o.depends_on
                for item in self._items.values()
                for o in item.obligations
            )
            else 17
            if any(
                o.comparison_worksheet_id is not None
                for item in self._items.values()
                for o in item.obligations
            )
            else 16
            if self.planning_mode == "outcomes"
            else 15,
            "tool_evidence_floors": dict(sorted(self._tool_evidence_floors.items())),
        }

    def _register_remediation(self, remediation: InternalToolRemediation) -> None:
        """Retain one state-neutral correction until an admitted plan resolves it."""

        self._pending_remediations[remediation.correction_id] = remediation

    def _candidate_remediation_resolutions(
        self,
        items: Iterable[WorkItem],
    ) -> tuple[InternalRemediationResolution, ...]:
        """Bind every open correction to an exact validator-admissible plan repair."""

        if not self._pending_remediations:
            return ()
        proposed = {item.item_id: item for item in items}
        resolutions: list[InternalRemediationResolution] = []
        for remediation in self._pending_remediations.values():
            if remediation.plan_revision != self.revision:
                raise RuntimeError("Pending remediation revision is inconsistent.")
            item = proposed.get(remediation.item_id)
            if remediation.code not in _REVISION_REPAIR_CLASSES:
                raise RuntimeError("Pending remediation code is unsupported.")
            repair_class = _revision_preservation_repair_class(
                remediation.code, item, self._items.get(remediation.item_id)
            )
            if repair_class not in remediation.allowed_repairs:
                raise WorkflowTransitionRejection(
                    f"item {remediation.item_id!r} must apply the accepted-plan "
                    "preservation repair from the attached remediation object",
                    remediation=remediation,
                )
            resolutions.append(
                InternalRemediationResolution.from_remediation(
                    remediation,
                    repair_class=repair_class,
                )
            )
        return tuple(resolutions)

    def _items_json(self) -> str:
        return json.dumps(
            [item.as_dict() for item in self._items.values()],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _expand_outcome_items(self, raw_items: list[object], *, updating: bool) -> list[object]:
        expanded = []
        for raw in raw_items:
            if type(raw) is not dict or "next_operation" in raw:
                raise ValueError("Outcome checklist items must not contain next_operation")
            item_id = raw.get("id")
            if not isinstance(item_id, str):
                raise ValueError("Outcome checklist item requires a string id")
            previous = self._items.get(item_id) if updating else None
            # Retained state must re-enter the model-input schema, not the
            # persistence schema (whose evidence includes internal availability).
            merged = _work_item_sync_input(previous) if previous is not None else {}
            retained_obligations = merged.get("obligations", [])
            merged.update(raw)
            reopening = (
                previous is not None
                and previous.status is WorkStatus.COMPLETED
                and previous.completion_mode is WorkCompletionMode.INFORMATION
                and merged.get("status") in {"pending", "in_progress"}
            )
            if reopening:
                # Archive the old completion only after atomic validation. Do not
                # silently treat its conclusion or resolved checks as current.
                if "outcome" not in raw:
                    merged["outcome"] = "unresolved"
                merged["evidence"] = raw.get("evidence", "")
                merged["evidence_refs"] = raw.get("evidence_refs", [])
                for obligation in retained_obligations:
                    if obligation.get("kind") == "knowledge_search":
                        obligation["status"] = "ready"
                        obligation["evidence"] = ""
                        obligation["evidence_refs"] = []
            if previous is None:
                # Defaults describe unstarted work, never consent or completion.
                # Existing patches preserve state; explicit invalid values still
                # pass through to the normal atomic validation below.
                merged.setdefault("status", "pending")
                merged.setdefault("depends_on", [])
                if merged.get("status") in ("pending", "in_progress"):
                    merged.setdefault("outcome", "unresolved")
            if merged.get("status") in ("pending", "in_progress"):
                merged.setdefault("evidence", "")
                merged.setdefault("evidence_refs", [])
            if updating and "obligations" in raw:
                merged["obligations"] = _merge_obligation_updates(
                    retained_obligations, raw["obligations"]
                )
            # The actor explicitly declares the semantic need for customer
            # input. Derive only its redundant nonterminal parent phase. This
            # neither guesses intent nor resolves prerequisites; ordinary
            # evidence, dependency and response checks still run atomically.
            obligations = merged.get("obligations", [])
            derived_customer_wait = False
            if (
                merged.get("status") in ("pending", "in_progress")
                and merged.get("outcome") == "unresolved"
                and type(obligations) is list
                and any(
                    type(o) is dict
                    and o.get("kind") == "customer_input"
                    and o.get("status") == "ready"
                    for o in obligations
                )
            ):
                merged["status"] = "awaiting_customer"
                derived_customer_wait = True
                if merged.get("evidence", "") == "":
                    merged["evidence"] = (
                        "Awaiting the customer input identified by the ready obligation."
                    )
            # Reuse an explicitly evidenced blocker without asking the actor to
            # duplicate it on its parent. This cannot prove operation success or
            # change any child's readiness, and all references validate below.
            waiting_operation = merged.get("completion_mode") == "operation" and merged.get(
                "status"
            ) in {"awaiting_customer", "blocked"}
            if (
                (
                    waiting_operation
                    or (
                        merged.get("completion_mode") == "information"
                        and merged.get("status") in {"completed", "awaiting_customer", "blocked"}
                    )
                )
                and ("evidence_refs" not in raw or derived_customer_wait)
                and merged.get("evidence_refs", []) == []
            ):
                obligations = merged.get("obligations", [])
                if type(obligations) is not list or any(type(o) is not dict for o in obligations):
                    raise ValueError("obligations must be an array of objects")
                if any(type(o.get("evidence_refs", [])) is not list for o in obligations):
                    raise ValueError("obligation evidence_refs must be an array")
                refs = [
                    ref
                    for obligation in obligations
                    if not waiting_operation
                    or obligation.get("status") == "waiting"
                    or (
                        obligation.get("kind") == "customer_input"
                        and obligation.get("status") == "ready"
                    )
                    for ref in obligation.get("evidence_refs", [])
                ]
                merged["evidence_refs"] = list(
                    {json.dumps(r, sort_keys=True): r for r in refs}.values()
                )
                if refs and "evidence" not in raw and not merged.get("evidence"):
                    merged["evidence"] = (
                        "See the explicitly cited waiting obligations or customer-input requirements."
                        if waiting_operation
                        else "See the explicitly cited child obligations."
                    )
            # Validate the public outcome contract before adding the internal
            # compatibility sentinel. Otherwise a missing evidence field produces
            # an exact-mode error asking the actor for a forbidden next_operation.
            public_required = {
                "id",
                "goal",
                "completion_mode",
                "status",
                "outcome",
                "depends_on",
                "evidence",
                "evidence_refs",
            }
            missing = public_required - merged.keys()
            unexpected = (
                merged.keys()
                - public_required
                - {"obligations", "next_operation", "prior_customer_report"}
            )
            if missing:
                raise ValueError(
                    f"Outcome checklist item is missing required fields: {sorted(missing)}"
                )
            if unexpected:
                raise ValueError("Outcome checklist item contains unsupported fields")
            # Compatibility with shared evidence/state projections only. This is
            # never exposed as a planned future call or used to authorize execution.
            merged["next_operation"] = {
                "owner": "none",
                "kind": "none",
                "tool_name": "",
                "arguments": {},
                "summary": "",
            }
            expanded.append(merged)
        return expanded

    def _merge_plan_update_inputs(self, raw_updates: list[object]) -> list[object]:
        """Merge model-supplied upserts with accepted items before full validation.

        This grants no transition authority. It only reconstructs a complete candidate
        document so every existing revision, evidence, dependency, semantic, and
        remediation invariant runs unchanged and atomically in ``invoke``.
        """

        candidate: list[object] = [_work_item_sync_input(item) for item in self._items.values()]
        positions = {item.item_id: index for index, item in enumerate(self._items.values())}
        supplied_ids: set[str] = set()
        for raw_update in raw_updates:
            raw_id = raw_update.get("id") if type(raw_update) is dict else None
            item_id = raw_id.strip() if isinstance(raw_id, str) else None
            if item_id and item_id not in supplied_ids and item_id in positions:
                candidate[positions[item_id]] = raw_update
            else:
                if item_id and item_id not in positions:
                    positions[item_id] = len(candidate)
                candidate.append(raw_update)
            if item_id:
                supplied_ids.add(item_id)
        return candidate

    def _parse_items(
        self,
        raw_items: list[object],
        *,
        validate_references: bool,
        allow_legacy_terminal_blocked: bool = False,
        allow_legacy_completion_mode: bool = False,
    ) -> list[WorkItem]:
        if len(raw_items) > _MAX_WORK_ITEMS:
            raise ValueError(f"at most {_MAX_WORK_ITEMS} work items are allowed")
        items: list[WorkItem] = []
        seen_ids: set[str] = set()
        for index, raw in enumerate(raw_items):
            if type(raw) is not dict:
                raise ValueError(f"item {index} must be an object")
            required = {
                "id",
                "goal",
                "completion_mode",
                "status",
                "outcome",
                "depends_on",
                "next_operation",
                "evidence",
                "evidence_refs",
            }
            raw_obligations = raw.get("obligations", [])
            prior_customer_report = raw.get("prior_customer_report", False)
            if type(prior_customer_report) is not bool:
                raise ValueError("prior_customer_report must be boolean")
            if "prior_customer_report" in raw:
                if self.planning_mode != "outcomes":
                    raise ValueError("prior_customer_report requires outcomes planning mode")
                raw = {key: value for key, value in raw.items() if key != "prior_customer_report"}
            if "obligations" in raw:
                if self.planning_mode != "outcomes":
                    raise ValueError("obligations require outcomes planning mode")
                raw = {key: value for key, value in raw.items() if key != "obligations"}
            legacy_without_mode = required - {"completion_mode"}
            if set(raw) == legacy_without_mode and allow_legacy_completion_mode:
                raw_completion_mode: object = _infer_legacy_completion_mode(raw).value
            elif set(raw) == required:
                raw_completion_mode = raw["completion_mode"]
            else:
                raise ValueError(f"item {index} requires exactly {sorted(required)}")
            item_id = _bounded_required_string(
                raw["id"],
                field=f"item {index} id",
                limit=_MAX_ITEM_ID_CHARS,
            )
            if not _ITEM_ID_PATTERN.fullmatch(item_id):
                raise ValueError(f"item {index} id has unsupported characters")
            if item_id in seen_ids:
                raise ValueError(f"duplicate item id {item_id!r}")
            seen_ids.add(item_id)
            goal = _bounded_required_string(
                raw["goal"],
                field=f"item {item_id} goal",
                limit=_MAX_GOAL_CHARS,
            )
            try:
                completion_mode = WorkCompletionMode(raw_completion_mode)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"item {item_id} has an invalid completion_mode") from exc
            try:
                status = WorkStatus(raw["status"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"item {item_id} has an invalid status") from exc
            try:
                outcome = WorkOutcome(raw["outcome"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"item {item_id} has an invalid outcome") from exc
            raw_dependencies = raw["depends_on"]
            if type(raw_dependencies) is not list or not all(
                isinstance(value, str) for value in raw_dependencies
            ):
                raise ValueError(f"item {item_id} depends_on must contain strings")
            dependencies = tuple(value.strip() for value in raw_dependencies)
            if any(not value for value in dependencies) or len(set(dependencies)) != len(
                dependencies
            ):
                raise ValueError(f"item {item_id} has invalid or duplicate dependencies")
            try:
                next_operation = _parse_work_operation(
                    raw["next_operation"],
                    item_id=item_id,
                )
            except ValueError as exc:
                raise _ControllerValidationRejection(
                    "sync_operation_shape_rejected", str(exc)
                ) from exc
            evidence = _bounded_optional_string(
                raw["evidence"],
                field=f"item {item_id} evidence",
                limit=_MAX_EVIDENCE_CHARS,
            )
            raw_evidence_refs = raw["evidence_refs"]
            if type(raw_evidence_refs) is not list:
                raise ValueError(f"item {item_id} evidence_refs must be an array")
            if len(raw_evidence_refs) > _MAX_EVIDENCE_REFERENCES:
                raise ValueError(
                    f"item {item_id} may cite at most {_MAX_EVIDENCE_REFERENCES} references"
                )
            try:
                evidence_refs = tuple(
                    _parse_evidence_reference(
                        value, item_id=item_id, persisted=not validate_references
                    )
                    for value in raw_evidence_refs
                )
            except ValueError as exc:
                raise _ControllerValidationRejection(
                    "sync_evidence_reference_shape_rejected", str(exc)
                ) from exc
            identities = [
                (reference.kind, reference.reference_id, reference.version)
                for reference in evidence_refs
            ]
            if len(set(identities)) != len(identities):
                raise ValueError(f"item {item_id} has duplicate evidence references")
            if (
                status
                in {
                    WorkStatus.AWAITING_CUSTOMER,
                    WorkStatus.BLOCKED,
                    WorkStatus.COMPLETED,
                }
                and not evidence
            ):
                raise ValueError(f"item {item_id} status {status.value} requires evidence")
            if status in {WorkStatus.BLOCKED, WorkStatus.COMPLETED} and not evidence_refs:
                raise ValueError(f"item {item_id} status {status.value} requires evidence_refs")
            if validate_references:
                try:
                    self._validate_evidence_references(item_id, status, evidence_refs)
                except _ControllerValidationRejection:
                    raise
                except ValueError as exc:
                    raise _ControllerValidationRejection(
                        "sync_item_evidence_rejected", str(exc)
                    ) from exc
            try:
                if prior_customer_report and (
                    status is not WorkStatus.COMPLETED
                    or completion_mode is not WorkCompletionMode.OPERATION
                    or outcome not in {WorkOutcome.DECLINED, WorkOutcome.SUCCEEDED}
                    or not evidence_refs
                    or any(r.kind != EvidenceKind.CUSTOMER_MESSAGE.value for r in evidence_refs)
                ):
                    raise ValueError(
                        "prior_customer_report requires a completed operation supported only "
                        "by customer testimony"
                    )
                _validate_outcome_evidence(
                    item_id,
                    completion_mode,
                    status,
                    outcome,
                    evidence_refs,
                )
            except ValueError as exc:
                raise _ControllerValidationRejection(
                    "sync_item_outcome_rejected", str(exc)
                ) from exc
            obligations = self._parse_obligations(
                raw_obligations,
                item_id=item_id,
                validate_references=validate_references,
                validate_current_conditions=not (
                    item_id in self._items and self._items[item_id].status is WorkStatus.COMPLETED
                ),
            )
            if status is WorkStatus.COMPLETED and any(
                obligation.status in {"ready", "waiting"} for obligation in obligations
            ):
                raise ValueError(f"item {item_id} has unresolved preparatory obligations")
            items.append(
                WorkItem(
                    item_id=item_id,
                    goal=goal,
                    completion_mode=completion_mode,
                    status=status,
                    outcome=outcome,
                    depends_on=dependencies,
                    next_operation=next_operation,
                    evidence=evidence,
                    evidence_refs=evidence_refs,
                    obligations=obligations,
                    prior_customer_report=prior_customer_report,
                )
            )
        try:
            self._validate_dependency_graph(items)
        except ValueError as exc:
            raise _ControllerValidationRejection(
                "sync_dependency_graph_rejected", str(exc)
            ) from exc
        try:
            self._validate_operations(
                items,
                allow_legacy_terminal_blocked=allow_legacy_terminal_blocked,
            )
        except ValueError as exc:
            raise _ControllerValidationRejection("sync_operation_shape_rejected", str(exc)) from exc
        return items

    def _parse_obligations(
        self,
        raw: object,
        *,
        item_id: str,
        validate_references: bool,
        validate_current_conditions: bool = True,
    ) -> tuple[WorkObligation, ...]:
        if type(raw) is not list or len(raw) > 8:
            raise ValueError("obligations must be an array of at most eight entries")
        result: list[WorkObligation] = []
        seen: set[str] = set()
        for entry in raw:
            if type(entry) is dict:
                entry = {
                    "kind": "action",
                    "comparison_worksheet_id": None,
                    "condition": None,
                    "depends_on": [],
                    "applied_to": None,
                    "planned_tool": None,
                    **entry,
                }
            if type(entry) is dict and entry.get("status") == "ready":
                # Empty defaults only for executable work, never waiting or terminal states.
                entry = {"evidence": "", "evidence_refs": [], **entry}
            if type(entry) is not dict or set(entry) != {
                "id",
                "goal",
                "kind",
                "status",
                "evidence",
                "evidence_refs",
                "comparison_worksheet_id",
                "condition",
                "depends_on",
                "applied_to",
                "planned_tool",
            }:
                raise ValueError("obligation requires id, goal, status, evidence and evidence_refs")
            identifier = _bounded_required_string(
                entry["id"], field="obligation id", limit=_MAX_ITEM_ID_CHARS
            )
            if not _ITEM_ID_PATTERN.fullmatch(identifier) or identifier in seen:
                raise ValueError("invalid or duplicate obligation id")
            seen.add(identifier)
            goal = _bounded_required_string(
                entry["goal"], field="obligation goal", limit=_MAX_GOAL_CHARS
            )
            status = entry["status"]
            kind = entry["kind"]
            if not isinstance(kind, str) or kind not in {
                "action",
                "communication",
                "customer_input",
                "knowledge_search",
            }:
                raise ValueError("invalid obligation kind")
            if not isinstance(status, str) or status not in {
                "ready",
                "waiting",
                "resolved",
                "not_applicable",
            }:
                raise ValueError("invalid obligation status")
            worksheet_id = entry["comparison_worksheet_id"]
            condition = parse_condition(entry["condition"])
            applied_to = entry["applied_to"]
            planned_tool_raw = entry["planned_tool"]
            planned_tool = None
            if planned_tool_raw is not None:
                if (
                    kind != "action"
                    or type(planned_tool_raw) is not dict
                    or not {"tool_name", "arguments"} <= set(planned_tool_raw)
                    or set(planned_tool_raw)
                    - {"tool_name", "arguments", "outcome_effects", "customer_choice"}
                    or not isinstance(planned_tool_raw["tool_name"], str)
                    or not planned_tool_raw["tool_name"].strip()
                    or type(planned_tool_raw["arguments"]) is not dict
                ):
                    raise ValueError(
                        "planned_tool requires an action obligation with tool_name and object arguments"
                    )
                effects = planned_tool_raw.get("outcome_effects")
                choice = planned_tool_raw.get("customer_choice")
                if effects is not None and (
                    type(effects) is not dict
                    or len(effects) > 32
                    or any(
                        not isinstance(k, str)
                        or not k
                        or not isinstance(v, str)
                        or v not in {"preserves", "blocks", "unknown", "customer_accepted"}
                        for k, v in effects.items()
                    )
                ):
                    raise ValueError("outcome_effects requires outcome IDs and typed effects")
                if choice is not None and (not isinstance(choice, str) or not choice.strip()):
                    raise ValueError("customer_choice requires a customer-message ID")
                planned_tool = PlannedToolCall(
                    tool_name=planned_tool_raw["tool_name"].strip(),
                    arguments=_json_object_copy(planned_tool_raw["arguments"]),
                    outcome_effects=dict(effects) if effects is not None else None,
                    customer_choice=choice,
                )
            if applied_to is not None:
                if (
                    not isinstance(applied_to, str)
                    or not applied_to
                    or len(applied_to) > 200
                    or condition is None
                ):
                    raise ValueError("invalid condition execution receipt")
                action_name = self._ledger.tool_action_name(applied_to)
                action_args = self._ledger.tool_action_arguments(applied_to)
                if (
                    self._ledger.tool_action_status(applied_to) is not ToolActionStatus.SUCCEEDED
                    or action_name is None
                    or action_args is None
                    or not _condition_matches_call(
                        condition,
                        ToolCallPart(
                            tool_call_id=applied_to, tool_name=action_name, arguments=action_args
                        ),
                        self._ledger,
                        historical_call_id=applied_to,
                    )
                ):
                    raise ValueError("condition receipt must refer to the successful scoped action")
            dependencies = entry["depends_on"]
            if (
                type(dependencies) is not list
                or len(dependencies) > 7
                or any(
                    not isinstance(d, str) or not _ITEM_ID_PATTERN.fullmatch(d)
                    for d in dependencies
                )
                or len(set(dependencies)) != len(dependencies)
            ):
                raise ValueError("invalid obligation dependencies")
            if condition is not None and kind != "action":
                raise ValueError("a resource condition requires an action obligation")
            if worksheet_id is not None and (
                not isinstance(worksheet_id, str)
                or re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", worksheet_id) is None
                or kind != "knowledge_search"
            ):
                raise ValueError(
                    "comparison_worksheet_id requires a valid worksheet id and knowledge_search kind"
                )
            evidence = _bounded_optional_string(
                entry["evidence"], field="obligation evidence", limit=_MAX_EVIDENCE_CHARS
            )
            raw_refs = entry["evidence_refs"]
            if type(raw_refs) is not list or len(raw_refs) > _MAX_EVIDENCE_REFERENCES:
                raise ValueError("invalid obligation evidence_refs array")
            refs = tuple(
                _parse_evidence_reference(ref, item_id=item_id, persisted=not validate_references)
                for ref in raw_refs
            )
            identities = [(ref.kind, ref.reference_id, ref.version) for ref in refs]
            if len(set(identities)) != len(identities):
                raise ValueError("duplicate obligation evidence references")
            if status != "ready" and (not evidence or not refs):
                raise ValueError("non-ready obligations require an explanation and evidence_refs")
            if kind == "customer_input" and (not evidence or not refs):
                raise ValueError(
                    "customer_input requires an explanation and evidence_refs even when ready"
                )
            if status in {"resolved", "not_applicable"} and any(
                ref.kind == EvidenceKind.TOOL_ERROR.value for ref in refs
            ):
                raise ValueError("a failed tool cannot resolve or waive an obligation")
            if validate_references:
                self._validate_evidence_references(
                    f"{item_id}/{identifier}", WorkStatus.PENDING, refs
                )
                if condition is not None:
                    self._validate_evidence_references(
                        f"{item_id}/{identifier}", WorkStatus.PENDING, (condition.procedure,)
                    )
                    if (
                        self._ledger.tool_action_status(condition.observed_in)
                        is not ToolActionStatus.SUCCEEDED
                    ):
                        raise ValueError(
                            "condition observed_in must identify a successful public tool read"
                        )
                    if (
                        condition.customer_message_id is not None
                        and not self._ledger.has_customer_message(condition.customer_message_id)
                    ):
                        raise ValueError("unknown condition customer_message_id")
                    if (
                        validate_current_conditions
                        and applied_to is None
                        and status == "resolved"
                        and (gap := condition.gap(self._ledger))
                    ):
                        raise ValueError(
                            f"condition {identifier}: {gap}; inspect the resource and update the prerequisite, not just its status"
                        )
            result.append(
                WorkObligation(
                    identifier,
                    goal,
                    status,
                    evidence,
                    refs,
                    kind,
                    worksheet_id,
                    condition,
                    tuple(dependencies),
                    applied_to,
                    planned_tool,
                )
            )
        by_id = {o.obligation_id: o for o in result}
        visited, visiting = set(), set()

        def visit(identifier):
            if identifier in visiting:
                raise ValueError("obligation dependencies contain a cycle")
            if identifier in visited:
                return
            visiting.add(identifier)
            for dependency in by_id[identifier].depends_on:
                if dependency not in by_id:
                    raise ValueError("unknown obligation dependency")
                visit(dependency)
                if by_id[identifier].status == "resolved" and by_id[dependency].status not in {
                    "resolved",
                    "not_applicable",
                }:
                    raise ValueError("cannot resolve an obligation before its prerequisites")
            visiting.remove(identifier)
            visited.add(identifier)

        for identifier in by_id:
            visit(identifier)
        return tuple(result)

    def _is_information_reopening(self, previous: WorkItem, current: WorkItem) -> bool:
        return (
            self.planning_mode == "outcomes"
            and previous.completion_mode is WorkCompletionMode.INFORMATION
            and current.completion_mode is WorkCompletionMode.INFORMATION
            and previous.status is WorkStatus.COMPLETED
            and current.status
            in {WorkStatus.PENDING, WorkStatus.IN_PROGRESS, WorkStatus.AWAITING_CUSTOMER}
            and current.outcome is WorkOutcome.UNRESOLVED
        )

    def _parse_previous_information_completions(self, raw, items, revision):
        if type(raw) is not dict or len(raw) > _MAX_WORK_ITEMS:
            raise ValueError("Invalid previous completion inventory")
        if raw and self.planning_mode != "outcomes":
            raise ValueError("Previous information completions require outcome planning")
        active = {item.item_id: item for item in items}
        parsed = {}
        for identifier, records in raw.items():
            current = active.get(identifier)
            if current is None or type(records) is not list or not records:
                raise ValueError("Invalid previous completion records")
            last_revision = -1
            for record in records:
                if type(record) is not dict or set(record) != {"revision", "item"}:
                    raise ValueError("Invalid previous completion record")
                if (
                    type(record["revision"]) is not int
                    or not last_revision < record["revision"] < revision
                ):
                    raise ValueError("Invalid previous completion revision")
                last_revision = record["revision"]
                candidates = [
                    record["item"] if item.item_id == identifier else item.as_dict()
                    for item in items
                ]
                old_items = self._parse_items(candidates, validate_references=False)
                old = next((item for item in old_items if item.item_id == identifier), None)
                if (
                    old is None
                    or old.goal != current.goal
                    or old.status is not WorkStatus.COMPLETED
                    or old.completion_mode is not WorkCompletionMode.INFORMATION
                    or current.completion_mode is not WorkCompletionMode.INFORMATION
                ):
                    raise ValueError("Previous completion does not match informational item")
            parsed[identifier] = deepcopy(records)
        for identifier, records in self._previous_information_completions.items():
            if parsed.get(identifier, [])[: len(records)] != records:
                raise ValueError("Previous completion history cannot be removed or rewritten")
        return parsed

    def _validate_revision(
        self,
        items: list[WorkItem],
        *,
        validate_handoff_progress: bool = True,
        validate_comparisons: bool = True,
        historical_reopen_ids: frozenset[str] = frozenset(),
    ) -> None:
        proposed = {item.item_id: item for item in items}
        if validate_comparisons and self.enforce_comparison_coverage:
            for item in items:
                previous = self._items.get(item.item_id)
                if (
                    item.status is WorkStatus.COMPLETED
                    and previous is not None
                    and previous.status is WorkStatus.COMPLETED
                ):
                    continue
                for obligation in item.obligations:
                    worksheet_id = obligation.comparison_worksheet_id
                    if (
                        worksheet_id is not None
                        and obligation.status == "resolved"
                        and not self._comparison_readiness.get(worksheet_id, False)
                    ):
                        raise ValueError(
                            f"comparison worksheet {worksheet_id!r} has no complete revalidated declared coverage; keep the comparison unresolved and resolve its material questions"
                        )
        missing = set(self._items) - set(proposed)
        if missing:
            item_id = sorted(missing)[0]
            raise WorkflowTransitionRejection(
                f"prior items cannot be removed: {', '.join(sorted(missing))}",
                remediation=self._revision_remediation(
                    code="revision_prior_item_removed",
                    item_id=item_id,
                ),
            )
        for item_id, previous in self._items.items():
            current = proposed[item_id]
            current_obligations = {o.obligation_id: o for o in current.obligations}
            for obligation in previous.obligations:
                replacement = current_obligations.get(obligation.obligation_id)
                if replacement is None or replacement.goal != obligation.goal:
                    raise ValueError(
                        "prior obligations cannot be removed or have their goals changed"
                    )
                if obligation.condition is not None and replacement.condition is None:
                    raise ValueError(
                        "a resource condition cannot be removed; record an evidenced waiver or retarget it"
                    )
                if (
                    obligation.comparison_worksheet_id is not None
                    and replacement.comparison_worksheet_id != obligation.comparison_worksheet_id
                ):
                    raise ValueError("a comparison worksheet binding cannot be removed or changed")
            if current.goal != previous.goal:
                raise WorkflowTransitionRejection(
                    f"prior item {item_id!r} goal is immutable",
                    remediation=self._revision_remediation(
                        code="revision_goal_changed",
                        item_id=item_id,
                    ),
                )
            if current.completion_mode is not previous.completion_mode and not (
                self._semantic_rejection_revision == self.revision
                and item_id in self._semantic_reclassifiable_item_ids
            ):
                raise WorkflowTransitionRejection(
                    f"prior item {item_id!r} completion_mode is immutable",
                    remediation=self._revision_remediation(
                        code="revision_completion_mode_changed",
                        item_id=item_id,
                    ),
                )
            if (
                not _status_transition_allowed(previous.status, current.status)
                and not (
                    self.planning_mode == "outcomes" and previous.status is not WorkStatus.COMPLETED
                )
                and not self._is_information_reopening(previous, current)
            ):
                raise WorkflowTransitionRejection(
                    f"item {item_id!r} cannot regress from {previous.status.value} "
                    f"to {current.status.value}",
                    remediation=self._revision_remediation(
                        code="revision_status_regressed",
                        item_id=item_id,
                    ),
                )
            if (
                previous.status is WorkStatus.COMPLETED
                and current.depends_on != previous.depends_on
                and not self._is_information_reopening(previous, current)
                and item_id not in historical_reopen_ids
            ):
                raise WorkflowTransitionRejection(
                    f"completed item {item_id!r} dependencies are immutable",
                    remediation=self._revision_remediation(
                        code="revision_completed_item_changed",
                        item_id=item_id,
                    ),
                )
            if (
                previous.status is WorkStatus.COMPLETED
                and not self._is_information_reopening(previous, current)
                and item_id not in historical_reopen_ids
                and (
                    current.outcome is not previous.outcome
                    or current.next_operation != previous.next_operation
                    or current.evidence != previous.evidence
                    or current.evidence_refs != previous.evidence_refs
                    or current.prior_customer_report != previous.prior_customer_report
                    or current.obligations != previous.obligations
                )
            ):
                raise WorkflowTransitionRejection(
                    f"completed item {item_id!r} evidence is immutable",
                    remediation=self._revision_remediation(
                        code="revision_completed_item_changed",
                        item_id=item_id,
                    ),
                )
            if (
                validate_handoff_progress
                and self.planning_mode != "outcomes"
                and current.completion_mode is WorkCompletionMode.OPERATION
                and current.status is WorkStatus.AWAITING_CUSTOMER
                and current.next_operation.kind
                in {
                    WorkOperationKind.CUSTOMER_RESPONSE,
                    WorkOperationKind.CUSTOMER_PRODUCT_SELECTION,
                }
            ):
                self._validate_operation_customer_handoff(current)

    def _revision_remediation(
        self,
        *,
        code: str,
        item_id: str,
    ) -> InternalToolRemediation:
        """Build a bounded repair contract for an accepted-item invariant."""

        repair_class = _REVISION_REPAIR_CLASSES[code]
        return InternalToolRemediation(
            code=code,
            plan_revision=self.revision,
            item_id=item_id,
            forbidden_transition=code.removeprefix("revision_"),
            allowed_repairs=(repair_class,),
        )

    def _validate_semantic_repair(self, items: list[WorkItem]) -> None:
        """Require a rejected plan's next revision to make the directed semantic repair."""

        if self._semantic_rejection_revision != self.revision:
            return
        proposed = {item.item_id: item for item in items}
        required_new_mode = self._semantic_required_new_item_mode
        if required_new_mode is not None:
            new_items = [proposed[item_id] for item_id in set(proposed) - set(self._items)]
            matching_new_items = sum(
                item.completion_mode is required_new_mode for item in new_items
            )
            if matching_new_items < self._semantic_required_new_item_count:
                raise ValueError(
                    "semantic review requires at least "
                    f"{self._semantic_required_new_item_count} new "
                    f"{required_new_mode.value}-mode outcome item(s) in one revision"
                )
        if self._semantic_requires_operation_increase:
            current_operations = sum(
                item.completion_mode is WorkCompletionMode.OPERATION
                for item in self._items.values()
            )
            proposed_operations = sum(
                item.completion_mode is WorkCompletionMode.OPERATION for item in items
            )
            if proposed_operations <= current_operations:
                raise ValueError(
                    "semantic review requires the corrected plan to contain an additional "
                    "operation-mode outcome"
                )

    def _validate_evidence_references(
        self,
        item_id: str,
        status: WorkStatus,
        references: tuple[WorkEvidenceReference, ...],
    ) -> None:
        for reference in references:
            if not reference.available:
                raise _ControllerValidationRejection(
                    "sync_evidence_unavailable_rejected",
                    f"item {item_id} evidence must be available",
                )
            if reference.kind == EvidenceKind.CUSTOMER_MESSAGE.value:
                if reference.version is not None or reference.digest is not None:
                    raise _ControllerValidationRejection(
                        "sync_customer_reference_metadata_rejected",
                        f"item {item_id} customer_message evidence cannot specify version or digest",
                    )
                if not self._ledger.has_customer_message(reference.reference_id):
                    raise _ControllerValidationRejection(
                        "sync_customer_reference_unknown",
                        f"item {item_id} cites an unknown customer message",
                    )
            elif reference.kind in {
                EvidenceKind.TOOL_ERROR.value,
                EvidenceKind.TOOL_RESULT.value,
            }:
                if reference.version is not None or reference.digest is not None:
                    raise _ControllerValidationRejection(
                        "sync_tool_reference_metadata_rejected",
                        f"item {item_id} tool evidence cannot specify version or digest",
                    )
                expected = (
                    ToolActionStatus.FAILED
                    if reference.kind == EvidenceKind.TOOL_ERROR.value
                    else ToolActionStatus.SUCCEEDED
                )
                if self._ledger.tool_action_status(reference.reference_id) is not expected:
                    raise _ControllerValidationRejection(
                        "sync_tool_reference_unavailable",
                        f"item {item_id} cites an unavailable {reference.kind}. "
                        "tool_result requires a successful executed environment call; "
                        "tool_error requires a failed executed environment call. "
                        "For private knowledge/search results, copy the delivered document's "
                        "knowledge_revision evidence_ref instead of the private call id. "
                        "A pre-dispatch policy denial is not an executed tool failure: "
                        "explain that blocker and cite its delivered policy/customer sources. "
                        "Do not fabricate a call or claim the operation completed.",
                    )
                if status is WorkStatus.COMPLETED and expected is ToolActionStatus.FAILED:
                    raise _ControllerValidationRejection(
                        "sync_tool_error_completion_rejected",
                        f"item {item_id} cannot complete from a failed tool result",
                    )
            elif reference.kind == EvidenceKind.KNOWLEDGE_REVISION.value:
                if reference.version is None:
                    raise _ControllerValidationRejection(
                        "sync_knowledge_reference_version_rejected",
                        f"item {item_id} knowledge evidence requires a revision version",
                    )
                known = self._knowledge_evidence.get((reference.reference_id, reference.version))
                if known is None:
                    raise _ControllerValidationRejection(
                        "sync_knowledge_reference_undelivered",
                        f"item {item_id} cites a knowledge revision not delivered by Cayu",
                    )
                if reference.digest is not None and reference.digest != known.digest:
                    raise _ControllerValidationRejection(
                        "sync_knowledge_reference_digest_mismatch",
                        f"item {item_id} knowledge evidence digest does not match",
                    )
            else:  # WorkEvidenceReference rejects this before this branch.
                raise _ControllerValidationRejection(
                    "sync_evidence_kind_rejected",
                    f"item {item_id} has an unsupported evidence kind",
                )

    def _remember_knowledge_evidence(self, reference: WorkEvidenceReference) -> None:
        assert reference.version is not None
        key = (reference.reference_id, reference.version)
        previous = self._knowledge_evidence.get(key)
        if previous is not None and previous.digest is not None:
            if reference.digest is not None and reference.digest != previous.digest:
                raise ValueError("Conflicting canonical digest for an immutable knowledge revision")
            return  # A digest-less observation must not erase established proof.
        self._knowledge_evidence[key] = reference

    def _candidate_customer_evidence_floors(
        self,
        items: list[WorkItem],
    ) -> dict[str, int]:
        latest = self._ledger.last_customer_message_sequence
        floors: dict[str, int] = {}
        for item in items:
            previous = self._items.get(item.item_id)
            floor = self._customer_evidence_floors.get(item.item_id, latest)
            if previous is None and item.prior_customer_report:
                floor = latest - 1
            if item.status is WorkStatus.AWAITING_CUSTOMER and (
                previous is None or previous.status is not WorkStatus.AWAITING_CUSTOMER
            ):
                floor = latest
            floors[item.item_id] = floor
        return floors

    def _candidate_tool_evidence_floors(self, items: list[WorkItem]) -> dict[str, int]:
        latest = self._ledger.last_tool_action_sequence
        floors: dict[str, int] = {}
        for item in items:
            previous = self._items.get(item.item_id)
            floor = self._tool_evidence_floors.get(item.item_id, latest)
            if item.next_operation.kind is WorkOperationKind.TOOL and (
                previous is None or item.next_operation != previous.next_operation
            ):
                floor = latest
            floors[item.item_id] = floor
        return floors

    def _validate_completed_operations(
        self,
        items: list[WorkItem],
        tool_evidence_floors: Mapping[str, int],
    ) -> None:
        if self.planning_mode == "outcomes":
            for item in items:
                if item.status is not WorkStatus.COMPLETED:
                    continue
                previous = self._items.get(item.item_id)
                if previous is not None and previous.status is WorkStatus.COMPLETED:
                    continue
                if item.outcome is WorkOutcome.HANDED_OFF and not any(
                    r.kind == EvidenceKind.TOOL_RESULT.value
                    and self._ledger.tool_action_name(r.reference_id) == HUMAN_TRANSFER_TOOL
                    for r in item.evidence_refs
                ):
                    raise ValueError(
                        f"item {item.item_id} handoff requires a successful transfer result"
                    )
                if item.outcome in {WorkOutcome.SUCCEEDED, WorkOutcome.HANDED_OFF, WorkOutcome.DENIED}:
                    tool_refs = [
                        r for r in item.evidence_refs if r.kind == EvidenceKind.TOOL_RESULT.value
                    ]
                    if tool_refs and not (previous is None and getattr(self, "allow_retrospective_outcomes", False)) and not any(
                        (self._ledger.tool_action_sequence(r.reference_id) or 0)
                        > tool_evidence_floors[item.item_id]
                        for r in tool_refs
                    ):
                        raise ValueError(
                            f"item {item.item_id} requires a result after plan creation"
                        )
            return
        for item in items:
            if item.status is not WorkStatus.COMPLETED:
                continue
            previous = self._items.get(item.item_id)
            if previous is None:
                if item.outcome in {WorkOutcome.HANDED_OFF, WorkOutcome.SUCCEEDED} or any(
                    reference.kind == EvidenceKind.TOOL_RESULT.value
                    for reference in item.evidence_refs
                ):
                    raise ValueError(
                        f"item {item.item_id} tool-backed work must be planned before completion"
                    )
                continue
            if previous.status is WorkStatus.COMPLETED:
                # Revision validation already proved this accepted terminal item is
                # immutable. Replaying it must not reinterpret the canonical none
                # sentinel as the operation that originally established completion.
                continue
            if item.outcome is WorkOutcome.HANDED_OFF:
                matching_transfer_results = [
                    reference
                    for reference in item.evidence_refs
                    if reference.kind == EvidenceKind.TOOL_RESULT.value
                    and self._ledger.tool_action_name(reference.reference_id) == HUMAN_TRANSFER_TOOL
                    and (self._ledger.tool_action_sequence(reference.reference_id) or 0)
                    > tool_evidence_floors[item.item_id]
                ]
                if not matching_transfer_results:
                    raise ValueError(
                        f"item {item.item_id} handoff requires a subsequent successful "
                        f"{HUMAN_TRANSFER_TOOL!r} result"
                    )
                continue
            operation = previous.next_operation
            if item.outcome is WorkOutcome.DECLINED:
                if operation.owner is not WorkOwner.CUSTOMER:
                    raise ValueError(
                        f"item {item.item_id} cannot be declined without a customer-owned "
                        "prior operation"
                    )
                continue
            if (
                item.outcome in {WorkOutcome.SUCCEEDED, WorkOutcome.DENIED}
                and operation.kind is not WorkOperationKind.TOOL
            ):
                raise ValueError(
                    f"item {item.item_id} succeeded outcome requires a planned tool operation"
                )
            if operation.kind is not WorkOperationKind.TOOL:
                continue
            if operation.owner is WorkOwner.CUSTOMER:
                if not any(
                    reference.kind == EvidenceKind.CUSTOMER_MESSAGE.value
                    for reference in item.evidence_refs
                ):
                    raise ValueError(
                        f"item {item.item_id} customer tool completion requires a subsequent "
                        "customer message"
                    )
                continue
            matching_results = [
                reference
                for reference in item.evidence_refs
                if reference.kind == EvidenceKind.TOOL_RESULT.value
                and self._ledger.tool_action_name(reference.reference_id) == operation.tool_name
                and _json_objects_equal(
                    self._ledger.tool_action_arguments(reference.reference_id),
                    operation.arguments,
                )
                and (self._ledger.tool_action_sequence(reference.reference_id) or 0)
                > tool_evidence_floors[item.item_id]
            ]
            if not matching_results:
                raise ValueError(
                    f"item {item.item_id} completion requires a successful result from planned "
                    f"tool {operation.tool_name!r}"
                )

    def _validate_customer_evidence_timing(
        self,
        items: list[WorkItem],
        floors: Mapping[str, int],
    ) -> None:
        for item in items:
            if item.prior_customer_report:
                previous = self._items.get(item.item_id)
                if (
                    item.status is not WorkStatus.COMPLETED
                    or item.completion_mode is not WorkCompletionMode.OPERATION
                    or item.outcome not in {WorkOutcome.DECLINED, WorkOutcome.SUCCEEDED}
                    or not item.evidence_refs
                    or any(
                        r.kind != EvidenceKind.CUSTOMER_MESSAGE.value for r in item.evidence_refs
                    )
                    or (previous is not None and not previous.prior_customer_report)
                ):
                    raise ValueError(
                        f"item {item.item_id} prior_customer_report requires a new completed "
                        "operation supported only by customer testimony; it cannot bypass a handoff"
                    )
            if item.status is not WorkStatus.COMPLETED or item.outcome not in {
                WorkOutcome.DECLINED,
                WorkOutcome.SUCCEEDED,
            }:
                continue
            customer_references = [
                reference
                for reference in item.evidence_refs
                if reference.kind == EvidenceKind.CUSTOMER_MESSAGE.value
            ]
            has_successful_tool_result = any(
                reference.kind == EvidenceKind.TOOL_RESULT.value for reference in item.evidence_refs
            )
            if not customer_references or has_successful_tool_result:
                continue
            floor = floors[item.item_id]
            if not any(
                (self._ledger.customer_message_sequence(reference.reference_id) or 0) > floor
                for reference in customer_references
            ):
                raise ValueError(
                    f"item {item.item_id} customer evidence must follow plan creation or its "
                    "latest customer handoff"
                )

    def _validate_operations(
        self,
        items: list[WorkItem],
        *,
        allow_legacy_terminal_blocked: bool = False,
    ) -> None:
        if self.planning_mode == "outcomes":
            for item in items:
                if item.status is not WorkStatus.AWAITING_CUSTOMER and any(
                    o.kind == "customer_input" and o.status == "ready" for o in item.obligations
                ):
                    raise ValueError("a ready customer_input requires an awaiting_customer parent")
            return
        by_id = {item.item_id: item for item in items}
        for item in items:
            operation = item.next_operation
            terminal = item.status is WorkStatus.COMPLETED
            is_none = (
                operation.owner is WorkOwner.NONE
                and operation.kind is WorkOperationKind.NONE
                and not operation.tool_name
                and not operation.arguments
                and not operation.summary
            )
            if terminal:
                if not is_none:
                    raise ValueError(
                        f"item {item.item_id} terminal status requires a none next_operation"
                    )
                continue
            if item.status is WorkStatus.BLOCKED:
                if is_none and allow_legacy_terminal_blocked:
                    continue
                if (
                    operation.owner is not WorkOwner.AGENT
                    or operation.kind is not WorkOperationKind.TOOL
                ):
                    raise ValueError(
                        f"item {item.item_id} blocked status requires an agent-owned recovery "
                        "or escalation tool operation"
                    )
            incomplete_dependencies = [
                dependency
                for dependency in item.depends_on
                if by_id[dependency].status is not WorkStatus.COMPLETED
            ]
            if is_none:
                if item.status is not WorkStatus.PENDING or not incomplete_dependencies:
                    raise ValueError(
                        f"item {item.item_id} actionable status requires a next_operation"
                    )
                continue
            if operation.owner is WorkOwner.NONE or operation.kind is WorkOperationKind.NONE:
                raise ValueError(
                    f"item {item.item_id} next_operation cannot partially use none; "
                    "the canonical sentinel requires owner=none, kind=none, tool_name='', "
                    "arguments={}, summary=''. Put dependency explanations in evidence, "
                    "not the sentinel summary. Use it only for terminal work or pending "
                    "work with an incomplete declared dependency."
                )
            if not operation.summary:
                raise ValueError(f"item {item.item_id} next_operation requires a summary")
            if operation.kind is WorkOperationKind.TOOL:
                if not operation.tool_name:
                    raise ValueError(f"item {item.item_id} tool operation requires tool_name")
                if operation.tool_name in _CONTROL_PLANE_TOOL_NAMES:
                    raise ValueError(
                        f"item {item.item_id} cannot plan controller bookkeeping tool "
                        f"{operation.tool_name!r} as customer work"
                    )
                if (
                    self._available_agent_tool_names is not None
                    and operation.owner is WorkOwner.AGENT
                    and operation.tool_name not in self._available_agent_tool_names
                ):
                    raise ValueError(
                        f"item {item.item_id} tool {operation.tool_name!r} is not an offered "
                        "top-level environment tool or registered private operation; use the "
                        "documented discoverable-tool wrapper when applicable"
                    )
            elif operation.tool_name or operation.arguments:
                raise ValueError(
                    f"item {item.item_id} non-tool operation cannot specify tool details"
                )
            if (
                operation.kind is WorkOperationKind.KNOWLEDGE_SEARCH
                and not self._knowledge_search_available
            ):
                raise ValueError(
                    f"item {item.item_id} cannot plan knowledge_search because its bounded "
                    "capability is exhausted or unavailable; use delivered evidence or "
                    "another offered operation"
                )
            if item.status is WorkStatus.AWAITING_CUSTOMER:
                if operation.owner is not WorkOwner.CUSTOMER or operation.kind not in {
                    WorkOperationKind.CUSTOMER_RESPONSE,
                    WorkOperationKind.CUSTOMER_PRODUCT_SELECTION,
                    WorkOperationKind.TOOL,
                }:
                    raise ValueError(
                        f"item {item.item_id} awaiting_customer requires a customer-owned "
                        "response or tool operation"
                    )
            elif operation.owner is not WorkOwner.AGENT:
                raise ValueError(
                    f"item {item.item_id} {item.status.value} work must remain agent-owned"
                )
            if (
                operation.kind
                in {
                    WorkOperationKind.CUSTOMER_RESPONSE,
                    WorkOperationKind.CUSTOMER_PRODUCT_SELECTION,
                }
                and operation.owner is not WorkOwner.CUSTOMER
            ):
                raise ValueError(
                    f"item {item.item_id} {operation.kind.value} must be customer-owned"
                )
            if (
                operation.kind is WorkOperationKind.KNOWLEDGE_SEARCH
                and operation.owner is not WorkOwner.AGENT
            ):
                raise ValueError(f"item {item.item_id} {operation.kind.value} must be agent-owned")

    def _validate_dependency_graph(self, items: list[WorkItem]) -> None:
        by_id = {item.item_id: item for item in items}
        for item in items:
            unknown = set(item.depends_on) - set(by_id)
            if unknown:
                raise ValueError(
                    f"item {item.item_id!r} has unknown dependencies: {', '.join(sorted(unknown))}"
                )
            if item.item_id in item.depends_on:
                raise ValueError(f"item {item.item_id!r} cannot depend on itself")
            # Outcome dependencies order completion, not independent preparation.
            # Merely recording that preparation is in progress neither resolves a
            # prerequisite nor authorizes an external effect. Legacy exact-operation
            # plans retain their stricter start-before-dependencies restriction.
            requires_completed_dependencies = item.status is WorkStatus.COMPLETED or (
                self.planning_mode != "outcomes" and item.status is WorkStatus.IN_PROGRESS
            )
            if requires_completed_dependencies:
                incomplete = [
                    dependency
                    for dependency in item.depends_on
                    if by_id[dependency].status is not WorkStatus.COMPLETED
                ]
                if incomplete:
                    raise ValueError(
                        f"item {item.item_id!r} cannot be {item.status.value} before "
                        f"dependencies complete: {', '.join(incomplete)}"
                    )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(item_id: str) -> None:
            if item_id in visiting:
                raise ValueError("work-item dependencies contain a cycle")
            if item_id in visited:
                return
            visiting.add(item_id)
            for dependency in by_id[item_id].depends_on:
                visit(dependency)
            visiting.remove(item_id)
            visited.add(item_id)

        for item_id in by_id:
            visit(item_id)


def _required_action_tool_names(
    items: Sequence[WorkItem],
    *,
    knowledge_search_available: bool = True,
) -> tuple[str, ...]:
    """Map dependency-ready work to exact tools for one forced correction round."""

    selected: list[str] = []
    for item in items:
        operation = item.next_operation
        if operation.owner is WorkOwner.AGENT and operation.kind is WorkOperationKind.TOOL:
            name = operation.tool_name
        elif (
            operation.owner is WorkOwner.AGENT
            and operation.kind is WorkOperationKind.KNOWLEDGE_SEARCH
            and knowledge_search_available
        ):
            name = KNOWLEDGE_SEARCH_TOOL
        else:
            name = UPDATE_WORK_PLAN_TOOL
        if name not in selected:
            selected.append(name)
    if UPDATE_WORK_PLAN_TOOL not in selected:
        selected.append(UPDATE_WORK_PLAN_TOOL)
    return tuple(selected)


def _revision_preservation_repair_class(
    code: str,
    item: WorkItem | None,
    previous: WorkItem | None,
) -> str | None:
    """Classify a validator-admissible repair of one accepted-plan invariant."""

    if item is None or previous is None:
        return None
    repair_class = _REVISION_REPAIR_CLASSES[code]
    if code == "revision_prior_item_removed":
        return repair_class
    if code == "revision_goal_changed" and item.goal == previous.goal:
        return repair_class
    if (
        code == "revision_completion_mode_changed"
        and item.completion_mode is previous.completion_mode
    ):
        return repair_class
    if code == "revision_status_regressed" and _status_transition_allowed(
        previous.status,
        item.status,
    ):
        return repair_class
    if code == "revision_completed_item_changed" and item == previous:
        return repair_class
    return None


def _status_transition_allowed(previous: WorkStatus, current: WorkStatus) -> bool:
    """Return whether a work-item status transition preserves forward progress."""

    if previous is WorkStatus.PENDING:
        return True
    if previous is WorkStatus.COMPLETED:
        return current is WorkStatus.COMPLETED
    return current is not WorkStatus.PENDING


def _bounded_required_string(value: object, *, field: str, limit: int) -> str:
    normalized = _bounded_optional_string(value, field=field, limit=limit)
    if not normalized:
        raise ValueError(f"{field} must be nonblank")
    return normalized


def _parse_evidence_reference(
    value: object,
    *,
    item_id: str,
    persisted: bool = False,
) -> WorkEvidenceReference:
    if type(value) is not dict:
        raise ValueError(f"item {item_id} evidence references must be objects")
    allowed = {"kind", "reference_id", "version", "digest"}
    if persisted:
        allowed.add("available")
    if not {"kind", "reference_id"}.issubset(value) or not set(value).issubset(allowed):
        raise ValueError(
            f"item {item_id} evidence references require kind/reference_id and only "
            "allow version/digest"
        )
    try:
        return WorkEvidenceReference.model_validate(value)
    except ValueError as exc:
        raise ValueError(f"item {item_id} has an invalid evidence reference") from exc


def _parse_work_operation(value: object, *, item_id: str) -> WorkOperation:
    required = {"owner", "kind", "tool_name", "arguments", "summary"}
    if type(value) is not dict or set(value) != required:
        raise ValueError(f"item {item_id} next_operation requires exactly {sorted(required)}")
    try:
        owner = WorkOwner(value["owner"])
        kind = WorkOperationKind(value["kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"item {item_id} has an invalid next_operation") from exc
    return WorkOperation(
        owner=owner,
        kind=kind,
        tool_name=_bounded_optional_string(
            value["tool_name"],
            field=f"item {item_id} next_operation tool_name",
            limit=_MAX_TOOL_NAME_CHARS,
        ),
        arguments=_bounded_json_object(
            value["arguments"],
            field=f"item {item_id} next_operation arguments",
            limit=_MAX_OPERATION_ARGUMENTS_CHARS,
        ),
        summary=_bounded_optional_string(
            value["summary"],
            field=f"item {item_id} next_operation summary",
            limit=_MAX_OPERATION_SUMMARY_CHARS,
        ),
    )


def _terminal_operation() -> WorkOperation:
    """Return the canonical operation sentinel for terminal work."""

    return WorkOperation(
        owner=WorkOwner.NONE,
        kind=WorkOperationKind.NONE,
        tool_name="",
        arguments={},
        summary="",
    )


def _work_item_sync_input(item: WorkItem) -> dict[str, Any]:
    """Serialize one item to the narrower model-facing plan-sync schema."""

    serialized = item.as_dict()
    for reference in serialized["evidence_refs"]:
        reference.pop("available", None)
    for obligation in serialized.get("obligations", []):
        for reference in obligation["evidence_refs"]:
            reference.pop("available", None)
    return serialized


def _merge_obligation_updates(previous: list[dict], updates: object) -> list[dict]:
    """Reconstruct a full candidate without weakening identity or evidence checks."""
    if type(updates) is not list or len(updates) > 8:
        raise ValueError("obligation updates must be an array of at most eight entries")
    candidate = {entry["id"]: dict(entry) for entry in previous}
    seen: set[str] = set()
    reassessed: set[str] = set()
    for entry in updates:
        if type(entry) is not dict or not isinstance(entry.get("id"), str):
            raise ValueError("obligation update requires an id")
        identifier = entry["id"].strip()
        if identifier in seen:
            raise ValueError("duplicate obligation update id")
        seen.add(identifier)
        prior = candidate.get(identifier, {})
        candidate[identifier] = {**prior, **entry, "id": identifier}
        if any(
            key in entry and entry[key] != prior.get(key) for key in ("condition", "depends_on")
        ):
            candidate[identifier].pop("applied_to", None)
        if entry.get("status") in ("resolved", "not_applicable"):
            reassessed.add(identifier)
        if "depends_on" in entry and (
            type(entry["depends_on"]) is not list
            or any(not isinstance(d, str) for d in entry["depends_on"])
        ):
            raise ValueError("obligation dependencies must be an array of ids")

    def inputs(entry):
        return {key: value for key, value in entry.items() if key not in {"goal", "evidence"}}

    changed = {entry["id"] for entry in previous if inputs(candidate[entry["id"]]) != inputs(entry)}
    for _ in range(len(candidate)):
        reopened = set()
        for identifier, entry in candidate.items():
            if (
                entry.get("status") == "resolved"
                and identifier not in reassessed
                and changed.intersection(entry.get("depends_on", []))
            ):
                candidate[identifier] = {
                    **entry,
                    "status": "ready",
                    "evidence": "Recheck required: prerequisite inputs changed.",
                }
                reopened.add(identifier)
        if not reopened - changed:
            break
        changed.update(reopened)
    return list(candidate.values())


def _condition_matches_call(
    condition: ResourceCondition,
    call: ToolCallPart,
    ledger: ConversationLedger,
    *,
    historical_call_id: str | None = None,
) -> bool:
    name, arguments = call.tool_name, call.arguments
    if name in {"call_discoverable_agent_tool", "give_discoverable_user_tool"}:
        key = (
            "agent_tool_name"
            if name == "call_discoverable_agent_tool"
            else "discoverable_tool_name"
        )
        name = arguments.get(key)
        try:
            arguments = json.loads(arguments.get("arguments", "{}"))
        except (TypeError, ValueError):
            return False
    if name not in condition.for_tools or not isinstance(arguments, dict):
        return False
    target_fields = RESOURCE_ARGUMENT_KINDS.keys()
    # Reject malformed known scope rather than falling back to customer scope.
    if any(
        not isinstance(value, str) or not value
        for key, value in arguments.items()
        if key in target_fields
    ):
        return False
    targets = resource_targets(arguments)
    if targets and condition.resource_id not in resource_targets(
        arguments, condition.resource_kind
    ):
        parent = ledger.recorded_resource(
            condition.observed_in, condition.resource_kind, condition.resource_id
        )
        card_id = arguments.get("card_id")
        # Only the explicitly observed bank-account -> debit-card relationship
        # may bridge distinct IDs. Never join arbitrary resource names or users.
        if not (
            condition.resource_kind == "bank_account"
            and parent is not None
            and isinstance(card_id, str)
            and all(key == "card_id" for key in arguments if key in target_fields)
            and ledger.debit_card_belongs_to(
                card_id, condition.resource_id, parent.user_id, before_call=historical_call_id
            )
        ):
            return False
    if targets and "user_id" not in arguments:
        return True
    # Creation can consume an existing account's eligibility without accepting
    # that account's ID. Bind its customer scope to the successful source read.
    # The pre-dispatch caller separately verifies every current field predicate.
    # Historical receipt validation uses the durable call identity, not a stale
    # account projection or a model-provided user ID in narrative evidence.
    user_id = arguments.get("user_id")
    if condition.resource_kind == "debit_card":
        observed_card = ledger.recorded_resource(
            condition.observed_in, "debit_card", condition.resource_id
        )
        return (
            isinstance(user_id, str)
            and observed_card is not None
            and observed_card.user_id == user_id
        )
    observed = ledger.tool_action_arguments(condition.observed_in)
    observed_name = ledger.tool_action_name(condition.observed_in)
    if observed_name == "call_discoverable_agent_tool" and isinstance(observed, dict):
        observed_name = observed.get("agent_tool_name")
        try:
            observed = json.loads(observed.get("arguments", "{}"))
        except (TypeError, ValueError):
            return False
    return (
        isinstance(user_id, str)
        and bool(user_id)
        and observed_name
        in {"get_all_user_accounts_by_user_id_3847", "get_credit_card_accounts_by_user"}
        and ledger.tool_action_status(condition.observed_in) is ToolActionStatus.SUCCEEDED
        and isinstance(observed, dict)
        and observed.get("user_id") == user_id
    )


def _semantic_item_identity(
    items: Iterable[WorkItem],
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (item.item_id, item.goal, item.completion_mode.value)
        for item in sorted(items, key=lambda candidate: candidate.item_id)
    )


def _parse_response_disposition(value: object) -> ResponseDisposition:
    if not isinstance(value, Mapping) or set(value) != {"items", "plan_revision"}:
        raise ValueError("expected exactly plan_revision and items")
    plan_revision = value["plan_revision"]
    if type(plan_revision) is not int or plan_revision < 0:
        raise ValueError("plan_revision must be a nonnegative integer")
    raw_items = value["items"]
    if type(raw_items) is not list or not raw_items:
        raise ValueError("items must be a nonempty array")
    if len(raw_items) > _MAX_WORK_ITEMS:
        raise ValueError(f"at most {_MAX_WORK_ITEMS} response items are allowed")
    items: list[ResponseItemDisposition] = []
    seen_ids: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        if type(raw_item) is not dict or set(raw_item) != {"item_id", "kind"}:
            raise ValueError(f"response item {index} requires exactly item_id and kind")
        item_id = _bounded_required_string(
            raw_item["item_id"],
            field=f"response item {index} item_id",
            limit=_MAX_ITEM_ID_CHARS,
        )
        if item_id in seen_ids:
            raise ValueError(f"duplicate response item id {item_id!r}")
        seen_ids.add(item_id)
        try:
            kind = ResponseDispositionKind(raw_item["kind"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"response item {item_id!r} has an invalid kind") from exc
        items.append(ResponseItemDisposition(item_id=item_id, kind=kind))
    return ResponseDisposition(plan_revision=plan_revision, items=tuple(items))


def _parse_restored_response_disposition(
    value: object,
    *,
    item_ids: set[str],
    maximum_revision: int,
) -> ResponseDisposition | None:
    if value is None:
        return None
    disposition = _parse_response_disposition(value)
    if disposition.plan_revision > maximum_revision:
        raise ValueError("response disposition revision exceeds its work-plan revision")
    unknown = {item.item_id for item in disposition.items} - item_ids
    if unknown:
        raise ValueError("response disposition contains unknown work-item ids")
    return disposition


def _parse_customer_evidence_floors(
    value: Mapping[object, object],
    *,
    item_ids: set[str],
    latest_sequence: int,
) -> dict[str, int]:
    if set(value) != item_ids:
        raise ValueError("customer evidence floors must match the work-item ids")
    floors: dict[str, int] = {}
    for item_id, sequence in value.items():
        if not isinstance(item_id, str) or type(sequence) is not int:
            raise ValueError("customer evidence floors must map item ids to integers")
        if sequence < 0 or sequence > latest_sequence:
            raise ValueError("customer evidence floor is outside the observed conversation")
        floors[item_id] = sequence
    return floors


def _parse_customer_response_handoffs(
    value: Mapping[object, object],
    *,
    item_ids: set[str],
) -> dict[str, int]:
    if set(value) != item_ids:
        raise ValueError("customer response handoffs must match the work-item ids")
    handoffs: dict[str, int] = {}
    for item_id, count in value.items():
        if not isinstance(item_id, str) or type(count) is not int:
            raise ValueError("customer response handoffs must map item ids to integers")
        if not 0 <= count <= _MAX_OPERATION_CUSTOMER_HANDOFFS:
            raise ValueError("customer response handoff count is outside its allowed range")
        handoffs[item_id] = count
    return handoffs


def _parse_customer_response_handoff_signatures(
    value: Mapping[object, object],
    *,
    handoffs: Mapping[str, int],
) -> dict[str, list[str]]:
    if set(value) != set(handoffs):
        raise ValueError("customer response signatures must match the work-item ids")
    parsed: dict[str, list[str]] = {}
    for item_id, raw_signatures in value.items():
        if not isinstance(item_id, str) or type(raw_signatures) is not list:
            raise ValueError("customer response signatures must map item ids to arrays")
        if len(raw_signatures) != handoffs[item_id] or any(
            not isinstance(signature, str)
            or _HANDOFF_SIGNATURE_PATTERN.fullmatch(signature) is None
            for signature in raw_signatures
        ):
            raise ValueError("customer response signature history is invalid")
        parsed[item_id] = list(raw_signatures)
    return parsed


def _parse_customer_response_handoff_sequences(
    value: Mapping[object, object],
    *,
    handoffs: Mapping[str, int],
    latest_sequence: int,
    counted_handoffs: bool = True,
) -> dict[str, int]:
    if set(value) != set(handoffs):
        raise ValueError("customer response sequences must match the work-item ids")
    parsed: dict[str, int] = {}
    for item_id, sequence in value.items():
        if not isinstance(item_id, str) or type(sequence) is not int:
            raise ValueError("customer response sequences must map item ids to integers")
        # Outcome mode records a customer-message epoch, not the legacy counted
        # question/signature history. Its zero legacy counter may coexist with a
        # real delivered epoch. Both modes still validate the epoch's range/type.
        if not 0 <= sequence <= latest_sequence or (
            counted_handoffs and (handoffs[item_id] == 0) != (sequence == 0)
        ):
            raise ValueError("customer response sequence history is invalid")
        parsed[item_id] = sequence
    return parsed


def _customer_response_handoff_signature(item: WorkItem) -> str:
    operation = item.next_operation
    payload = {
        "kind": operation.kind.value,
        "summary": " ".join(operation.summary.casefold().split()),
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _parse_tool_evidence_floors(
    value: Mapping[object, object],
    *,
    item_ids: set[str],
    latest_sequence: int,
) -> dict[str, int]:
    if set(value) != item_ids:
        raise ValueError("tool evidence floors must match the work-item ids")
    floors: dict[str, int] = {}
    for item_id, sequence in value.items():
        if not isinstance(item_id, str) or type(sequence) is not int:
            raise ValueError("tool evidence floors must map item ids to integers")
        if sequence < 0 or sequence > latest_sequence:
            raise ValueError("tool evidence floor is outside the observed conversation")
        floors[item_id] = sequence
    return floors


def _validate_outcome_evidence(
    item_id: str,
    completion_mode: WorkCompletionMode,
    status: WorkStatus,
    outcome: WorkOutcome,
    references: tuple[WorkEvidenceReference, ...],
) -> None:
    reference_kinds = {reference.kind for reference in references}
    if (
        completion_mode is WorkCompletionMode.INFORMATION
        and outcome is WorkOutcome.DECLINED
        and reference_kinds != {EvidenceKind.CUSTOMER_MESSAGE.value}
    ):
        raise ValueError(f"item {item_id} withdrawn research requires only customer evidence")
    if status in {WorkStatus.PENDING, WorkStatus.IN_PROGRESS, WorkStatus.AWAITING_CUSTOMER}:
        if outcome is not WorkOutcome.UNRESOLVED:
            raise ValueError(f"item {item_id} status {status.value} requires unresolved outcome")
        return
    if status is WorkStatus.BLOCKED:
        if outcome is not WorkOutcome.BLOCKED:
            raise ValueError(f"item {item_id} blocked status requires blocked outcome")
        return
    allowed_outcomes = (
        {WorkOutcome.ANSWERED, WorkOutcome.HANDED_OFF, WorkOutcome.DECLINED}
        if completion_mode is WorkCompletionMode.INFORMATION
        else {WorkOutcome.DECLINED, WorkOutcome.DENIED, WorkOutcome.HANDED_OFF, WorkOutcome.SUCCEEDED}
    )
    if outcome not in allowed_outcomes:
        expected = ", ".join(sorted(value.value for value in allowed_outcomes))
        raise ValueError(
            f"item {item_id} completion_mode {completion_mode.value} requires completed "
            f"outcome {expected}"
        )
    allowed_kinds = {
        WorkOutcome.ANSWERED: {
            EvidenceKind.KNOWLEDGE_REVISION.value,
            EvidenceKind.TOOL_RESULT.value,
        },
        WorkOutcome.DECLINED: {EvidenceKind.CUSTOMER_MESSAGE.value},
        WorkOutcome.DENIED: {EvidenceKind.TOOL_RESULT.value},
        WorkOutcome.HANDED_OFF: {EvidenceKind.TOOL_RESULT.value},
        WorkOutcome.SUCCEEDED: {
            EvidenceKind.CUSTOMER_MESSAGE.value,
            EvidenceKind.TOOL_RESULT.value,
        },
    }[outcome]
    if not reference_kinds & allowed_kinds:
        raise ValueError(
            f"item {item_id} outcome {outcome.value} lacks a compatible evidence reference"
        )


def _infer_legacy_completion_mode(raw: Mapping[str, object]) -> WorkCompletionMode:
    """Conservatively classify pre-schema-9 public controller snapshots."""

    if raw.get("outcome") == WorkOutcome.ANSWERED.value:
        return WorkCompletionMode.INFORMATION
    if raw.get("outcome") in {
        WorkOutcome.BLOCKED.value,
        WorkOutcome.DECLINED.value,
        WorkOutcome.HANDED_OFF.value,
        WorkOutcome.SUCCEEDED.value,
    }:
        return WorkCompletionMode.OPERATION
    operation = raw.get("next_operation")
    if isinstance(operation, Mapping) and operation.get("kind") == WorkOperationKind.TOOL.value:
        return WorkCompletionMode.OPERATION
    # Active legacy work has no trustworthy semantic type. Treat it as an operation
    # so an old snapshot cannot gain authority to close from policy text alone.
    return WorkCompletionMode.OPERATION


def _bounded_json_object(value: object, *, field: str, limit: int) -> dict[str, Any]:
    copied = _json_object_copy(value, field=field)
    encoded = json.dumps(
        copied,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > limit:
        raise ValueError(f"{field} exceeds {limit} UTF-8 bytes")
    return copied


def _json_object_copy(value: object, *, field: str = "arguments") -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an object")
    try:
        copied = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain JSON-compatible values") from exc
    if type(copied) is not dict:
        raise AssertionError("JSON object copy changed type unexpectedly")
    return copied


def _json_objects_equal(left: object, right: object) -> bool:
    if type(left) is not dict or type(right) is not dict:
        return False
    return json.dumps(
        left,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) == json.dumps(
        right,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _bounded_optional_string(value: object, *, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    normalized = " ".join(value.split())
    if len(normalized) > limit:
        raise ValueError(f"{field} exceeds {limit} characters")
    return normalized
