"""Oracle-free diagnostics for discarded Cayu bridge attempts."""

from __future__ import annotations

import json
import os
import stat
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

SAFE_INTERNAL_DIAGNOSTIC_COUNT_KEYS = frozenset({"product_selection_transition_rejections"})
SAFE_EXTERNAL_REJECTION_CODES = frozenset(
    {
        "banking_admission",
        "banking_discoverable_tool_locked",
        "credit_card_dispute_audit",
        "savings_interest_audit",
        "semantic_action_argument_semantic_mismatch",
        "semantic_action_customer_intent_mismatch",
        "semantic_action_insufficient_grounding",
        "semantic_action_premature_handoff",
        "semantic_action_unnecessary_or_duplicate",
        "unclassified",
        "workflow_human_transfer_request_threshold",
        "workflow_operation_mismatch",
        "workflow_plan_missing",
        "workflow_semantic_rejected",
    }
)
SAFE_INTERNAL_REMEDIATION_CODES = frozenset(
    {
        "other",
        "product_choice_question_required",
        "revision_completed_item_changed",
        "revision_completion_mode_changed",
        "revision_goal_changed",
        "revision_prior_item_removed",
        "revision_status_regressed",
        "selected_product_requires_advancement",
    }
)
SAFE_INTERNAL_REPAIR_CLASSES = frozenset(
    {
        "agent_owned_operation",
        "customer_tool_handoff",
        "other",
        "pending_on_incomplete_dependency",
        "preserve_prior_completion_mode",
        "preserve_prior_goal",
        "restore_completed_item",
        "restore_prior_item",
        "typed_product_selection",
        "use_non_regressive_status",
        "verified_completion",
    }
)
SAFE_INTERNAL_ERROR_CODES = frozenset(
    {
        "controller_unknown_tool",
        "empty_internal_result",
        "embedding_search_unavailable",
        "internal_tool_exception",
        "invalid_internal_result",
        "mixed_external_internal_batch",
        "product_choice_question_required",
        "response_actionable_items",
        "response_awaiting_coverage",
        "response_clarification_budget",
        "response_disposition_mismatch",
        "response_final_coverage",
        "response_invalid_arguments",
        "response_repeated_clarification",
        "response_semantic_coverage_rejected",
        "response_stale_revision",
        "response_unknown_items",
        "response_without_new_customer_input",
        "response_without_plan",
        "revision_completed_item_changed",
        "revision_completion_mode_changed",
        "revision_goal_changed",
        "revision_prior_item_removed",
        "revision_status_regressed",
        "selected_product_requires_advancement",
        "sync_completed_operations_rejected",
        "sync_customer_evidence_rejected",
        "sync_customer_reference_metadata_rejected",
        "sync_customer_reference_unknown",
        "sync_dependency_graph_rejected",
        "sync_evidence_kind_rejected",
        "sync_evidence_reference_shape_rejected",
        "sync_evidence_unavailable_rejected",
        "sync_invalid_arguments",
        "sync_invalid_items",
        "sync_item_evidence_rejected",
        "sync_item_outcome_rejected",
        "sync_knowledge_reference_digest_mismatch",
        "sync_knowledge_reference_undelivered",
        "sync_knowledge_reference_version_rejected",
        "sync_operation_shape_rejected",
        "sync_revision_rejected",
        "sync_semantic_coverage_rejected",
        "sync_semantic_plan_missing_information_outcome",
        "sync_semantic_plan_missing_operation_outcome",
        "sync_semantic_plan_review_rejected",
        "sync_semantic_plan_wrong_completion_mode_to_operation",
        "sync_semantic_repair_rejected",
        "sync_tool_evidence_rejected",
        "sync_tool_error_completion_rejected",
        "sync_tool_reference_metadata_rejected",
        "sync_tool_reference_unavailable",
        "update_requires_existing_plan",
        "unclassified_internal_error",
        "unhandled_internal_tool",
    }
)
SAFE_TERMINAL_REJECTION_CODES = frozenset(
    {
        "product_choice_question_required",
        "response_audited_dispute_filing_required",
        "response_actionable_items",
        "response_clarification_budget",
        "response_dispute_last_four_already_observed",
        "response_dispute_last_four_policy_required",
        "response_dispute_last_four_tool_delivery_required",
        "response_dispute_last_four_tool_handoff_required",
        "response_referral_eligibility_evidence_required",
        "response_disposition_not_ready",
        "response_repeated_clarification",
        "response_semantic_coverage_rejected",
        "semantic_plan_missing_information_outcome",
        "semantic_plan_missing_operation_outcome",
        "semantic_plan_wrong_completion_mode_to_operation",
        "semantic_response_customer_instruction_mismatch",
        "semantic_response_missing_required_item",
        "semantic_response_repeated_or_stalled_request",
        "semantic_response_unsupported_completion_claim",
        "response_without_new_customer_input",
        "response_without_plan",
        "selected_product_requires_advancement",
        "unclassified",
    }
)


@dataclass(frozen=True)
class RemediationLifecycleDiagnostic:
    """Content-free aggregate state for typed controller corrections."""

    issued: int
    repeated: int
    resolved: int
    superseded: int
    unresolved: int
    exhausted: int
    repair_rounds_total: int
    repair_rounds_max: int
    rejection_codes: dict[str, int]
    accepted_repair_classes: dict[str, int]

    def as_record(self) -> dict[str, object]:
        """Return a validated, versioned representation safe for persistence."""

        counts = (
            self.issued,
            self.repeated,
            self.resolved,
            self.superseded,
            self.unresolved,
            self.exhausted,
            self.repair_rounds_total,
            self.repair_rounds_max,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("remediation lifecycle counts must be nonnegative integers")
        if self.exhausted > self.unresolved:
            raise ValueError("exhausted remediation count cannot exceed unresolved count")
        if self.resolved > self.issued or self.superseded > self.issued:
            raise ValueError("remediation lifecycle terminal counts cannot exceed issued count")
        if self.resolved + self.superseded + self.unresolved != self.issued:
            raise ValueError("remediation lifecycle state counts must account for every issue")
        if self.resolved == 0 and (self.repair_rounds_total or self.repair_rounds_max):
            raise ValueError("repair round metrics require a resolved remediation")
        if self.resolved and self.repair_rounds_max > self.repair_rounds_total:
            raise ValueError("maximum repair rounds cannot exceed total repair rounds")
        _validate_safe_counts(
            self.rejection_codes,
            allowed=SAFE_INTERNAL_REMEDIATION_CODES,
            field="remediation rejection codes",
        )
        _validate_safe_counts(
            self.accepted_repair_classes,
            allowed=SAFE_INTERNAL_REPAIR_CLASSES,
            field="accepted remediation repair classes",
        )
        if sum(self.rejection_codes.values()) != self.issued:
            raise ValueError("remediation rejection code counts must equal issued count")
        if sum(self.accepted_repair_classes.values()) != self.resolved:
            raise ValueError("accepted remediation repair counts must equal resolved count")
        if self.exhausted not in {0, self.unresolved}:
            raise ValueError("exhausted remediation count must be zero or all unresolved")
        return {
            "schema_version": 1,
            **asdict(self),
        }


def _validate_safe_counts(value: dict[str, int], *, allowed: frozenset[str], field: str) -> None:
    if (
        type(value) is not dict
        or set(value) - allowed
        or any(type(count) is not int or count < 0 for count in value.values())
    ):
        raise ValueError(f"{field} contain unsupported data")


@dataclass(frozen=True)
class InternalRoundDiagnostic:
    """One controller-local round without arguments, results, or model text."""

    tool_names: tuple[str, ...]
    call_fingerprints: tuple[str, ...]
    progress_changed: bool | None
    error_result_count: int
    error_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BridgeFailureDiagnostics:
    """Bounded metadata attached to a bridge failure before tau discards the attempt."""

    attempt_id: str
    correlation_tag: str | None
    failure_code: str
    provider_call_count: int
    internal_continuations: int
    internal_tool_rounds: int
    internal_progress_events: int
    internal_stalled_rounds: int
    stalled_internal_rounds_since_progress: int
    internal_diagnostic_counts: dict[str, int]
    remediation_lifecycle: RemediationLifecycleDiagnostic
    consecutive_stalled_internal_rounds: int
    internal_tool_round_limit: int
    stalled_internal_tool_round_limit: int
    stalled_internal_tool_round_limit_since_progress: int
    external_admission_rounds: int
    external_admission_rejections: int
    external_rejection_codes: dict[str, int]
    terminal_gate_rounds: int
    terminal_gate_rejections: int
    terminal_gate_rounds_since_progress: int
    terminal_rejection_codes: dict[str, int]
    empty_response_retries: int
    round_trace: tuple[InternalRoundDiagnostic, ...]
    terminal_correction_fingerprints: tuple[str, ...]
    current_output_kind: str
    current_output_fingerprint: str | None
    current_tool_names: tuple[str, ...]
    current_call_fingerprints: tuple[str, ...]

    def as_record(self) -> dict[str, object]:
        """Return the stable JSON record written by :class:`AttemptDiagnosticsRecorder`."""

        if (
            type(self.internal_diagnostic_counts) is not dict
            or set(self.internal_diagnostic_counts) - SAFE_INTERNAL_DIAGNOSTIC_COUNT_KEYS
            or any(
                type(count) is not int or count < 0
                for count in self.internal_diagnostic_counts.values()
            )
        ):
            raise ValueError("internal diagnostic counts contain unsupported data")
        _validate_safe_counts(
            self.terminal_rejection_codes,
            allowed=SAFE_TERMINAL_REJECTION_CODES,
            field="terminal rejection codes",
        )
        if sum(self.terminal_rejection_codes.values()) != self.terminal_gate_rejections:
            raise ValueError("terminal rejection code counts must equal terminal rejections")
        _validate_safe_counts(
            self.external_rejection_codes,
            allowed=SAFE_EXTERNAL_REJECTION_CODES,
            field="external rejection codes",
        )
        if sum(self.external_rejection_codes.values()) != self.external_admission_rejections:
            raise ValueError("external rejection code counts must equal external rejections")
        internal_error_counts: dict[str, int] = {}
        for round_ in self.round_trace:
            if not isinstance(round_, InternalRoundDiagnostic):
                raise ValueError("internal round trace contains unsupported data")
            if (
                type(round_.error_codes) is not tuple
                or any(type(code) is not str for code in round_.error_codes)
                or any(code not in SAFE_INTERNAL_ERROR_CODES for code in round_.error_codes)
                or type(round_.error_result_count) is not int
                or round_.error_result_count < 0
                or len(round_.error_codes) != round_.error_result_count
            ):
                raise ValueError("internal round error codes contain unsupported data")
            for code in round_.error_codes:
                internal_error_counts[code] = internal_error_counts.get(code, 0) + 1
        record = asdict(self)
        record["remediation_lifecycle"] = self.remediation_lifecycle.as_record()
        record["internal_error_codes"] = dict(sorted(internal_error_counts.items()))
        return {
            "schema_version": 10,
            "event_type": "bridge_attempt_failure",
            **record,
        }


class AttemptDiagnosticsRecorder:
    """Append private, redacted bridge-failure records to a mode-0600 JSONL file."""

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a pathlib.Path.")
        if path.name in {"", ".", ".."}:
            raise ValueError("path must name a diagnostics file.")
        self.path = path
        self._lock = threading.Lock()

    def record(self, diagnostics: BridgeFailureDiagnostics) -> None:
        """Durably append one bounded record without following a destination symlink."""

        if not isinstance(diagnostics, BridgeFailureDiagnostics):
            raise TypeError("diagnostics must be BridgeFailureDiagnostics.")
        encoded = (
            json.dumps(
                diagnostics.as_record(),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        with self._lock:
            if self.path.is_symlink():
                raise OSError("Refusing to write attempt diagnostics through a symbolic link.")
            descriptor = os.open(self.path, flags, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise OSError("Attempt diagnostics destination must be a regular file.")
                os.fchmod(descriptor, 0o600)
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("Attempt diagnostics write made no progress.")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
