from __future__ import annotations

import json
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from cayu_tau3_banking.attempt_diagnostics import (
    AttemptDiagnosticsRecorder,
    BridgeFailureDiagnostics,
    InternalRoundDiagnostic,
    RemediationLifecycleDiagnostic,
)


def diagnostics(attempt: int) -> BridgeFailureDiagnostics:
    return BridgeFailureDiagnostics(
        attempt_id=f"attempt-{attempt}",
        correlation_tag="opaque-correlation",
        failure_code="internal_no_progress_limit",
        provider_call_count=5,
        internal_continuations=0,
        internal_tool_rounds=4,
        internal_progress_events=0,
        internal_stalled_rounds=4,
        stalled_internal_rounds_since_progress=4,
        internal_diagnostic_counts={"product_selection_transition_rejections": 2},
        remediation_lifecycle=RemediationLifecycleDiagnostic(
            issued=1,
            repeated=1,
            resolved=0,
            superseded=0,
            unresolved=1,
            exhausted=1,
            repair_rounds_total=0,
            repair_rounds_max=0,
            rejection_codes={"selected_product_requires_advancement": 1},
            accepted_repair_classes={},
        ),
        consecutive_stalled_internal_rounds=4,
        internal_tool_round_limit=32,
        stalled_internal_tool_round_limit=4,
        stalled_internal_tool_round_limit_since_progress=8,
        external_admission_rounds=0,
        external_admission_rejections=0,
        external_rejection_codes={},
        terminal_gate_rounds=0,
        terminal_gate_rejections=0,
        terminal_gate_rounds_since_progress=0,
        terminal_rejection_codes={},
        empty_response_retries=0,
        round_trace=(
            InternalRoundDiagnostic(
                tool_names=("controller_sync_work_plan",),
                call_fingerprints=(f"opaque-{attempt}",),
                progress_changed=False,
                error_result_count=1,
                error_codes=("response_actionable_items",),
            ),
        ),
        terminal_correction_fingerprints=(),
        current_output_kind="internal_tools",
        current_output_fingerprint=None,
        current_tool_names=("controller_sync_work_plan",),
        current_call_fingerprints=(f"opaque-{attempt}",),
    )


def test_recorder_appends_concurrently_with_private_permissions(tmp_path) -> None:
    path = tmp_path / "private" / "attempts.jsonl"
    recorder = AttemptDiagnosticsRecorder(path)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda attempt: recorder.record(diagnostics(attempt)), range(12)))

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {record["attempt_id"] for record in records} == {
        f"attempt-{attempt}" for attempt in range(12)
    }
    assert all(record["schema_version"] == 10 for record in records)
    assert all(record["event_type"] == "bridge_attempt_failure" for record in records)
    assert all(
        record["internal_diagnostic_counts"] == {"product_selection_transition_rejections": 2}
        for record in records
    )
    assert all(
        record["internal_error_codes"] == {"response_actionable_items": 1} for record in records
    )
    assert all(
        record["remediation_lifecycle"]
        == {
            "accepted_repair_classes": {},
            "exhausted": 1,
            "issued": 1,
            "rejection_codes": {"selected_product_requires_advancement": 1},
            "repair_rounds_max": 0,
            "repair_rounds_total": 0,
            "repeated": 1,
            "resolved": 0,
            "schema_version": 1,
            "superseded": 0,
            "unresolved": 1,
        }
        for record in records
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_recorder_rejects_symlink_destination(tmp_path) -> None:
    destination = tmp_path / "destination.jsonl"
    destination.touch()
    link = tmp_path / "attempts.jsonl"
    link.symlink_to(destination)

    with pytest.raises(OSError, match="symbolic link"):
        AttemptDiagnosticsRecorder(link).record(diagnostics(1))

    assert destination.read_bytes() == b""


def test_recorder_rejects_unsupported_internal_diagnostics(tmp_path) -> None:
    unsafe = diagnostics(1)
    unsafe.internal_diagnostic_counts["customer_text"] = 1

    with pytest.raises(ValueError, match="unsupported data"):
        AttemptDiagnosticsRecorder(tmp_path / "attempts.jsonl").record(unsafe)

    assert not (tmp_path / "attempts.jsonl").exists()


def test_recorder_rejects_unsupported_remediation_labels(tmp_path) -> None:
    unsafe = diagnostics(1)
    unsafe.remediation_lifecycle.rejection_codes["private_customer_value"] = 1

    with pytest.raises(ValueError, match="remediation rejection codes"):
        AttemptDiagnosticsRecorder(tmp_path / "attempts.jsonl").record(unsafe)

    assert not (tmp_path / "attempts.jsonl").exists()


def test_recorder_rejects_unsupported_internal_error_codes(tmp_path) -> None:
    unsafe = replace(
        diagnostics(1),
        round_trace=(
            InternalRoundDiagnostic(
                tool_names=("controller_prepare_response",),
                call_fingerprints=("opaque",),
                progress_changed=False,
                error_result_count=1,
                error_codes=("private_customer_value",),
            ),
        ),
    )

    with pytest.raises(ValueError, match="internal round error codes"):
        AttemptDiagnosticsRecorder(tmp_path / "attempts.jsonl").record(unsafe)

    assert not (tmp_path / "attempts.jsonl").exists()


@pytest.mark.parametrize(
    "error_code",
    [
        "sync_semantic_plan_missing_information_outcome",
        "sync_semantic_plan_missing_operation_outcome",
        "sync_semantic_plan_wrong_completion_mode_to_operation",
    ],
)
def test_recorder_accepts_typed_semantic_plan_error_codes(tmp_path, error_code: str) -> None:
    path = tmp_path / "attempts.jsonl"
    rejected = replace(
        diagnostics(1),
        round_trace=(
            InternalRoundDiagnostic(
                tool_names=("controller_update_work_plan",),
                call_fingerprints=("opaque",),
                progress_changed=True,
                error_result_count=1,
                error_codes=(error_code,),
            ),
        ),
    )

    AttemptDiagnosticsRecorder(path).record(rejected)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["internal_error_codes"] == {error_code: 1}


def test_recorder_rejects_unsupported_terminal_rejection_codes(tmp_path) -> None:
    unsafe = replace(
        diagnostics(1),
        terminal_gate_rounds=1,
        terminal_gate_rejections=1,
        terminal_rejection_codes={"private_customer_value": 1},
    )

    with pytest.raises(ValueError, match="terminal rejection codes"):
        AttemptDiagnosticsRecorder(tmp_path / "attempts.jsonl").record(unsafe)

    assert not (tmp_path / "attempts.jsonl").exists()


def test_recorder_rejects_unsupported_external_rejection_codes(tmp_path) -> None:
    unsafe = replace(
        diagnostics(1),
        external_admission_rounds=1,
        external_admission_rejections=1,
        external_rejection_codes={"private_customer_value": 1},
    )

    with pytest.raises(ValueError, match="external rejection codes"):
        AttemptDiagnosticsRecorder(tmp_path / "attempts.jsonl").record(unsafe)

    assert not (tmp_path / "attempts.jsonl").exists()


@pytest.mark.parametrize(
    "rejection_code",
    [
        "banking_discoverable_tool_locked",
        "credit_card_dispute_audit",
        "semantic_action_insufficient_grounding",
        "workflow_operation_mismatch",
    ],
)
def test_recorder_accepts_typed_external_rejection_codes(tmp_path, rejection_code: str) -> None:
    path = tmp_path / "attempts.jsonl"
    rejected = replace(
        diagnostics(1),
        external_admission_rounds=1,
        external_admission_rejections=1,
        external_rejection_codes={rejection_code: 1},
    )

    AttemptDiagnosticsRecorder(path).record(rejected)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["external_rejection_codes"] == {rejection_code: 1}


@pytest.mark.parametrize(
    "rejection_code",
    [
        "response_audited_dispute_filing_required",
        "response_dispute_last_four_already_observed",
        "response_dispute_last_four_policy_required",
        "response_dispute_last_four_tool_delivery_required",
        "response_dispute_last_four_tool_handoff_required",
        "semantic_plan_missing_information_outcome",
        "semantic_plan_missing_operation_outcome",
        "semantic_plan_wrong_completion_mode_to_operation",
        "semantic_response_customer_instruction_mismatch",
        "semantic_response_missing_required_item",
        "semantic_response_repeated_or_stalled_request",
        "semantic_response_unsupported_completion_claim",
    ],
)
def test_recorder_accepts_typed_terminal_rejection_codes(tmp_path, rejection_code: str) -> None:
    path = tmp_path / "attempts.jsonl"
    rejected = replace(
        diagnostics(1),
        terminal_gate_rounds=1,
        terminal_gate_rejections=1,
        terminal_rejection_codes={rejection_code: 1},
    )

    AttemptDiagnosticsRecorder(path).record(rejected)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["terminal_rejection_codes"] == {rejection_code: 1}


def test_recorder_accepts_typed_product_choice_remediation_labels(tmp_path) -> None:
    path = tmp_path / "attempts.jsonl"
    product_choice = replace(
        diagnostics(1),
        remediation_lifecycle=RemediationLifecycleDiagnostic(
            issued=1,
            repeated=0,
            resolved=1,
            superseded=0,
            unresolved=0,
            exhausted=0,
            repair_rounds_total=1,
            repair_rounds_max=1,
            rejection_codes={"product_choice_question_required": 1},
            accepted_repair_classes={"typed_product_selection": 1},
        ),
        round_trace=(
            InternalRoundDiagnostic(
                tool_names=("controller_sync_work_plan",),
                call_fingerprints=("opaque",),
                progress_changed=False,
                error_result_count=1,
                error_codes=("product_choice_question_required",),
            ),
        ),
    )

    AttemptDiagnosticsRecorder(path).record(product_choice)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["internal_error_codes"] == {"product_choice_question_required": 1}
    assert record["remediation_lifecycle"]["rejection_codes"] == {
        "product_choice_question_required": 1
    }
    assert record["remediation_lifecycle"]["accepted_repair_classes"] == {
        "typed_product_selection": 1
    }


def test_recorder_requires_one_code_per_error_result(tmp_path) -> None:
    unsafe = replace(
        diagnostics(1),
        round_trace=(
            InternalRoundDiagnostic(
                tool_names=("controller_prepare_response",),
                call_fingerprints=("opaque",),
                progress_changed=False,
                error_result_count=0,
                error_codes=("response_actionable_items",),
            ),
        ),
    )

    with pytest.raises(ValueError, match="internal round error codes"):
        AttemptDiagnosticsRecorder(tmp_path / "attempts.jsonl").record(unsafe)

    assert not (tmp_path / "attempts.jsonl").exists()


def test_remediation_lifecycle_requires_accounted_terminal_state() -> None:
    lifecycle = RemediationLifecycleDiagnostic(
        issued=2,
        repeated=0,
        resolved=1,
        superseded=0,
        unresolved=0,
        exhausted=0,
        repair_rounds_total=1,
        repair_rounds_max=1,
        rejection_codes={"selected_product_requires_advancement": 2},
        accepted_repair_classes={"customer_tool_handoff": 1},
    )

    with pytest.raises(ValueError, match="account for every issue"):
        lifecycle.as_record()
