from __future__ import annotations

import json

import pytest
from cayu import ToolCallPart, WorkEvidenceReference
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from cayu_tau3_banking.bridge import TerminalOutputAdmission
from cayu_tau3_banking.controller import (
    HUMAN_TRANSFER_TOOL,
    PREPARE_RESPONSE_TOOL,
    SYNC_WORK_PLAN_TOOL,
    UPDATE_WORK_PLAN_TOOL,
    BankingWorkflowController,
    WorkCompletionMode,
)
from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL
from cayu_tau3_banking.ledger import ConversationLedger
from cayu_tau3_banking.reward_audit import AUDIT_CREDIT_CARD_REWARDS_TOOL


def item(
    item_id: str,
    *,
    goal: str = "Resolve the customer's request",
    completion_mode: str | None = None,
    status: str = "pending",
    outcome: str | None = None,
    depends_on: list[str] | None = None,
    next_operation: dict | None = None,
    evidence: str = "",
    evidence_refs: list[dict] | None = None,
) -> dict:
    if outcome is None:
        outcome = "succeeded" if status == "completed" else status
        if outcome not in {"succeeded", "blocked"}:
            outcome = "unresolved"
    if next_operation is None:
        if status == "completed":
            next_operation = {
                "owner": "none",
                "kind": "none",
                "tool_name": "",
                "arguments": {},
                "summary": "",
            }
        elif status == "blocked":
            next_operation = {
                "owner": "agent",
                "kind": "tool",
                "tool_name": "transfer_to_human_agents",
                "arguments": {},
                "summary": "Escalate unresolved work to a human agent.",
            }
        elif status == "awaiting_customer":
            next_operation = {
                "owner": "customer",
                "kind": "customer_response",
                "tool_name": "",
                "arguments": {},
                "summary": "Wait for the customer's response",
            }
        else:
            next_operation = {
                "owner": "agent",
                "kind": "knowledge_search",
                "tool_name": "",
                "arguments": {},
                "summary": "Retrieve the information required for this item",
            }
    elif "arguments" not in next_operation:
        next_operation = {**next_operation, "arguments": {}}
    if completion_mode is None:
        completion_mode = "information" if outcome == "answered" else "operation"
    return {
        "id": item_id,
        "goal": goal,
        "completion_mode": completion_mode,
        "status": status,
        "outcome": outcome,
        "depends_on": depends_on or [],
        "next_operation": next_operation,
        "evidence": evidence,
        "evidence_refs": evidence_refs or [],
    }


def invoke(controller: BankingWorkflowController, items: list[dict]):
    return controller.invoke(SYNC_WORK_PLAN_TOOL, {"items": items})


def update(controller: BankingWorkflowController, items: list[dict]):
    return controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": items})


def prepare_response(
    controller: BankingWorkflowController,
    *,
    plan_revision: int,
    dispositions: list[tuple[str, str]],
):
    return controller.invoke(
        PREPARE_RESPONSE_TOOL,
        {
            "plan_revision": plan_revision,
            "items": [{"item_id": item_id, "kind": kind} for item_id, kind in dispositions],
        },
    )


def observe_successful_tool_result(
    ledger: ConversationLedger,
    *,
    call_id: str = "verify-1",
    tool_name: str = "log_verification",
) -> None:
    ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id=call_id,
                    name=tool_name,
                    arguments={"user_id": "user-7"},
                    requestor="assistant",
                )
            ],
        )
    )
    ledger.observe_input(
        ToolMessage(
            id=call_id,
            role="tool",
            requestor="assistant",
            content="Verification logged successfully.",
        )
    )


def test_successful_human_transfer_hands_off_every_unfinished_item() -> None:
    ledger = ConversationLedger()
    controller = BankingWorkflowController(ledger)
    transfer_arguments = {
        "reason": "customer_requests_human_no_specific_reason",
        "summary": "The customer explicitly requested a human agent.",
    }
    initial = invoke(
        controller,
        [
            item(
                "escalate",
                status="in_progress",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": HUMAN_TRANSFER_TOOL,
                    "arguments": transfer_arguments,
                    "summary": "Transfer the conversation to a human agent.",
                },
            ),
            item(
                "application",
                status="awaiting_customer",
                next_operation={
                    "owner": "customer",
                    "kind": "customer_response",
                    "tool_name": "",
                    "arguments": {},
                    "summary": "Choose whether to continue the application.",
                },
                evidence="The application still requires a customer decision.",
            ),
        ],
    )
    assert initial.is_error is False
    ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="transfer-1",
                    name=HUMAN_TRANSFER_TOOL,
                    arguments=transfer_arguments,
                    requestor="assistant",
                )
            ],
        )
    )
    ledger.observe_input(
        ToolMessage(
            id="transfer-1",
            role="tool",
            requestor="assistant",
            content="The transfer was successful.",
        )
    )

    controller.reconcile_successful_human_transfer("transfer-1")

    snapshot = controller.export_state()
    assert snapshot["revision"] == 2
    assert snapshot["schema_version"] == 15
    assert {entry["status"] for entry in snapshot["items"]} == {"completed"}
    assert {entry["outcome"] for entry in snapshot["items"]} == {"handed_off"}
    assert all(
        entry["next_operation"]
        == {
            "arguments": {},
            "kind": "none",
            "owner": "none",
            "summary": "",
            "tool_name": "",
        }
        for entry in snapshot["items"]
    )
    assert all(
        any(
            reference["kind"] == "tool_result" and reference["reference_id"] == "transfer-1"
            for reference in entry["evidence_refs"]
        )
        for entry in snapshot["items"]
    )
    assert snapshot["last_response_disposition"] == {
        "items": [
            {"item_id": "escalate", "kind": "final"},
            {"item_id": "application", "kind": "final"},
        ],
        "plan_revision": 2,
    }


def test_successful_human_transfer_supersedes_stale_semantic_plan_rejection() -> None:
    ledger = ConversationLedger()
    controller = BankingWorkflowController(ledger)
    transfer_arguments = {
        "reason": "customer_requests_human_no_specific_reason",
        "summary": "The customer explicitly requested a human agent.",
    }
    assert not invoke(
        controller,
        [
            item(
                "escalate",
                status="in_progress",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": HUMAN_TRANSFER_TOOL,
                    "arguments": transfer_arguments,
                    "summary": "Transfer the conversation to a human agent.",
                },
            )
        ],
    ).is_error
    ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="transfer-1",
                    name=HUMAN_TRANSFER_TOOL,
                    arguments=transfer_arguments,
                    requestor="assistant",
                )
            ],
        )
    )
    ledger.observe_input(
        ToolMessage(
            id="transfer-1",
            role="tool",
            requestor="assistant",
            content="The transfer was successful.",
        )
    )
    controller.reject_semantic_plan(
        required_new_item_mode=WorkCompletionMode.OPERATION,
        required_new_item_count=1,
    )

    controller.reconcile_successful_human_transfer("transfer-1")

    snapshot = controller.export_state()
    assert snapshot["items"][0]["status"] == "completed"
    assert snapshot["items"][0]["outcome"] == "handed_off"
    assert controller.preview_terminal_output("The transfer succeeded.").allowed is True


def test_human_transfer_reconciliation_rejects_failed_or_unplanned_results() -> None:
    ledger = ConversationLedger()
    controller = BankingWorkflowController(ledger)
    transfer_arguments = {
        "reason": "customer_requests_human_no_specific_reason",
        "summary": "The customer explicitly requested a human agent.",
    }
    assert not invoke(
        controller,
        [
            item(
                "escalate",
                status="in_progress",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": HUMAN_TRANSFER_TOOL,
                    "arguments": transfer_arguments,
                    "summary": "Transfer the conversation to a human agent.",
                },
            )
        ],
    ).is_error
    ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="transfer-failed",
                    name=HUMAN_TRANSFER_TOOL,
                    arguments=transfer_arguments,
                    requestor="assistant",
                )
            ],
        )
    )
    ledger.observe_input(
        ToolMessage(
            id="transfer-failed",
            role="tool",
            requestor="assistant",
            content="Transfer failed.",
            error=True,
        )
    )

    with pytest.raises(ValueError, match="successful result"):
        controller.reconcile_successful_human_transfer("transfer-failed")
    with pytest.raises(ValueError, match="transfer tool"):
        controller.reconcile_successful_human_transfer("unknown-call")


def test_model_cannot_claim_handoff_from_a_nontransfer_tool_result() -> None:
    ledger = ConversationLedger()
    controller = BankingWorkflowController(ledger)
    initial_item = item(
        "verify",
        status="in_progress",
        next_operation={
            "owner": "agent",
            "kind": "tool",
            "tool_name": "log_verification",
            "arguments": {"user_id": "user-7"},
            "summary": "Log identity verification.",
        },
    )
    assert not invoke(controller, [initial_item]).is_error
    observe_successful_tool_result(ledger)
    completed_item = item(
        "verify",
        status="completed",
        outcome="handed_off",
        evidence="The work was handed off.",
        evidence_refs=[{"kind": "tool_result", "reference_id": "verify-1"}],
    )

    result = invoke(controller, [completed_item])

    assert result.is_error is True
    assert result.diagnostic_code == "sync_completed_operations_rejected"
    assert "handoff requires" in result.content


def test_controller_accepts_valid_dependency_plan_and_updates_status() -> None:
    ledger = ConversationLedger()
    controller = BankingWorkflowController(ledger)

    initial = invoke(
        controller,
        [
            item(
                "verify",
                goal="Verify customer identity",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": "log_verification",
                    "arguments": {"user_id": "user-7"},
                    "summary": "Log the verified customer identity.",
                },
            ),
            item(
                "change-email",
                goal="Change the email address",
                depends_on=["verify"],
            ),
        ],
    )
    observe_successful_tool_result(ledger)
    updated = invoke(
        controller,
        [
            item(
                "verify",
                goal="Verify customer identity",
                status="completed",
                evidence="log_verification returned success",
                evidence_refs=[{"kind": "tool_result", "reference_id": "verify-1"}],
            ),
            item(
                "change-email",
                goal="Change the email address",
                status="in_progress",
                depends_on=["verify"],
            ),
        ],
    )

    assert initial.is_error is False
    assert updated.is_error is False
    assert controller.revision == 2
    assert controller.export_state()["items"][0]["status"] == "completed"


def test_rejected_revision_is_atomic() -> None:
    controller = BankingWorkflowController(ConversationLedger())
    original = [item("one"), item("two", depends_on=["one"])]
    assert invoke(controller, original).is_error is False
    before = controller.export_state()

    empty = invoke(controller, [])
    removed = invoke(controller, [item("one")])
    restored = invoke(controller, original)
    cycle = invoke(
        controller,
        [
            item("one", depends_on=["two"]),
            item("two", depends_on=["one"]),
        ],
    )

    assert empty.is_error is True
    assert removed.is_error is True
    assert cycle.is_error is True
    assert empty.diagnostic_code == "sync_invalid_items"
    assert removed.diagnostic_code == "revision_prior_item_removed"
    assert removed.remediation is not None
    assert removed.remediation.allowed_repairs == ("restore_prior_item",)
    assert restored.is_error is False
    assert restored.resolutions[0].correction_id == removed.remediation.correction_id
    assert restored.resolutions[0].repair_class == "restore_prior_item"
    assert cycle.diagnostic_code == "sync_dependency_graph_rejected"
    assert controller.export_state() == before


def test_identical_plan_sync_is_a_noop_for_progress_accounting() -> None:
    controller = BankingWorkflowController(ConversationLedger())
    plan = [item("one")]
    initial = invoke(controller, plan)
    token = controller.progress_token()

    repeated = invoke(controller, plan)

    assert initial.is_error is False
    assert repeated.is_error is False
    assert "already current" in repeated.content
    assert controller.revision == 1
    assert controller.progress_token() == token


def test_incremental_update_retains_omitted_items_and_runs_full_validation() -> None:
    controller = BankingWorkflowController(ConversationLedger())
    initial = [item("one"), item("two", depends_on=["one"])]
    assert invoke(controller, initial).is_error is False

    changed = update(controller, [item("one", status="in_progress")])

    assert changed.is_error is False
    assert controller.revision == 2
    exported = {entry["id"]: entry for entry in controller.export_state()["items"]}
    assert exported["one"]["status"] == "in_progress"
    assert exported["two"] == item("two", depends_on=["one"])


def test_incremental_update_rejects_duplicate_upserts_atomically() -> None:
    controller = BankingWorkflowController(ConversationLedger())
    assert invoke(controller, [item("one"), item("two")]).is_error is False
    before = controller.export_state()

    result = update(
        controller,
        [
            item("one", status="in_progress"),
            item("one", status="in_progress"),
        ],
    )

    assert result.is_error is True
    assert result.diagnostic_code == "sync_invalid_items"
    assert controller.export_state() == before


def test_controller_advertises_create_then_incremental_update() -> None:
    controller = BankingWorkflowController(ConversationLedger())

    initial_names = {schema["name"] for schema in controller.tool_schemas()}
    before_initial = update(controller, [item("one")])
    assert invoke(controller, [item("one")]).is_error is False
    update_schemas = controller.tool_schemas()
    update_names = {schema["name"] for schema in update_schemas}

    assert initial_names == {SYNC_WORK_PLAN_TOOL}
    assert before_initial.is_error is True
    assert before_initial.diagnostic_code == "update_requires_existing_plan"
    assert update_names == {UPDATE_WORK_PLAN_TOOL}
    assert "Omitted accepted items remain unchanged" in update_schemas[0]["description"]
    assert "item-level upsert, not a field-level patch" in update_schemas[0]["description"]
    next_operation = update_schemas[0]["input_schema"]["properties"]["items"]["items"][
        "properties"
    ]["next_operation"]
    assert "knowledge_search use owner agent with blank tool_name" in next_operation["description"]
    assert (
        "customer-side discoverable tool is owner customer"
        in next_operation["properties"]["kind"]["description"]
    )
    reference_id = update_schemas[0]["input_schema"]["properties"]["items"]["items"]["properties"][
        "evidence_refs"
    ]["items"]["properties"]["reference_id"]
    assert "copy a document's evidence_ref object unchanged" in reference_id["description"]
    assert controller.transcript_retention_key(SYNC_WORK_PLAN_TOOL) == "workflow_plan"
    assert controller.transcript_retention_key(UPDATE_WORK_PLAN_TOOL) == "workflow_plan"


def test_preparing_a_new_response_counts_as_progress_but_repeating_it_does_not() -> None:
    controller = BankingWorkflowController(ConversationLedger())
    waiting = item(
        "one",
        status="awaiting_customer",
        evidence="A customer response is required.",
    )
    assert invoke(controller, [waiting]).is_error is False
    before = controller.progress_token()

    prepared = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[("one", "ask_customer")],
    )
    after = controller.progress_token()
    repeated = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[("one", "ask_customer")],
    )

    assert prepared.is_error is False
    assert after != before
    assert repeated.is_error is False
    assert "already prepared" in repeated.content
    assert controller.progress_token() == after


def test_completed_work_cannot_regress() -> None:
    ledger = ConversationLedger()
    controller = BankingWorkflowController(ledger)
    assert (
        invoke(
            controller,
            [
                item(
                    "one",
                    next_operation={
                        "owner": "agent",
                        "kind": "tool",
                        "tool_name": "log_verification",
                        "arguments": {"user_id": "user-7"},
                        "summary": "Log the verified customer identity.",
                    },
                )
            ],
        ).is_error
        is False
    )
    observe_successful_tool_result(ledger)
    completed = item(
        "one",
        status="completed",
        evidence="environment tool succeeded",
        evidence_refs=[{"kind": "tool_result", "reference_id": "verify-1"}],
    )
    assert invoke(controller, [completed]).is_error is False

    result = invoke(controller, [item("one")])
    restored = invoke(controller, [completed])

    assert result.is_error is True
    assert result.diagnostic_code == "revision_status_regressed"
    assert result.remediation is not None
    assert result.remediation.allowed_repairs == ("use_non_regressive_status",)
    assert "cannot regress" in result.content
    assert restored.is_error is False
    assert restored.resolutions[0].correction_id == result.remediation.correction_id
    assert restored.resolutions[0].repair_class == "use_non_regressive_status"
    assert controller.export_state()["items"][0]["status"] == "completed"

    mutated_completed = {**completed, "evidence": "Mutated terminal evidence."}
    mutation = invoke(controller, [mutated_completed])
    restored_again = invoke(controller, [completed])

    assert mutation.is_error is True
    assert mutation.diagnostic_code == "revision_completed_item_changed"
    assert mutation.remediation is not None
    assert mutation.remediation.allowed_repairs == ("restore_completed_item",)
    assert restored_again.is_error is False
    assert restored_again.resolutions[0].correction_id == mutation.remediation.correction_id
    assert restored_again.resolutions[0].repair_class == "restore_completed_item"


def test_noncompleted_dependencies_can_be_refined_without_changing_goal() -> None:
    controller = BankingWorkflowController(ConversationLedger())
    original = [item("verify"), item("act")]
    assert invoke(controller, original).is_error is False

    changed_goal = invoke(
        controller,
        [item("verify", goal="Replace the accepted goal"), item("act")],
    )

    result = invoke(controller, [item("verify"), item("act", depends_on=["verify"])])

    assert changed_goal.is_error is True
    assert changed_goal.diagnostic_code == "revision_goal_changed"
    assert changed_goal.remediation is not None
    assert changed_goal.remediation.allowed_repairs == ("preserve_prior_goal",)
    assert result.is_error is False
    assert result.resolutions[0].correction_id == changed_goal.remediation.correction_id
    assert result.resolutions[0].repair_class == "preserve_prior_goal"
    assert controller.export_state()["items"][1]["depends_on"] == ["verify"]


def test_work_cannot_start_before_declared_dependencies_complete() -> None:
    controller = BankingWorkflowController(ConversationLedger())

    result = invoke(
        controller,
        [
            item("verify"),
            item("act", status="in_progress", depends_on=["verify"]),
        ],
    )

    assert result.is_error is True
    assert "before dependencies complete" in result.content
    assert controller.export_state() == {
        "customer_evidence_floors": {},
        "customer_response_handoffs": {},
        "customer_response_handoff_sequences": {},
        "customer_response_handoff_signatures": {},
        "items": [],
        "last_response_disposition": None,
        "product_selection_transition_rejections": 0,
        "revision": 0,
        "schema_version": 15,
        "tool_evidence_floors": {},
    }


def test_controller_restores_latest_valid_public_snapshot() -> None:
    ledger = ConversationLedger()
    controller = BankingWorkflowController(ledger)
    assert invoke(controller, [item("one")]).is_error is False
    history = [
        AssistantMessage.text(
            "Still working.",
            raw_data={"cayu_tau3_workflow": controller.export_state()},
        ),
        AssistantMessage.text(
            "Ignore corrupt snapshot.",
            raw_data={
                "cayu_tau3_workflow": {
                    "revision": "bad",
                    "items": [],
                    "customer_evidence_floors": {},
                    "schema_version": 6,
                    "tool_evidence_floors": {},
                }
            },
        ),
        AssistantMessage.text(
            "Ignore older snapshot.",
            raw_data={
                "cayu_tau3_workflow": {
                    "revision": 0,
                    "items": [],
                    "customer_evidence_floors": {},
                    "schema_version": 6,
                    "tool_evidence_floors": {},
                }
            },
        ),
    ]

    restored = BankingWorkflowController.from_history(history, ledger=ledger)

    assert restored.export_state() == controller.export_state()
    context_payload = (
        restored.render_context().split("<workflow-plan>\n", 1)[1].split("\n</workflow-plan>", 1)[0]
    )
    assert json.loads(context_payload) == controller.export_state()


def test_controller_context_uses_only_model_accepted_evidence_fields() -> None:
    reference = WorkEvidenceReference(
        kind="knowledge_revision",
        reference_id="public-policy",
        version="1",
        digest="a" * 64,
    )
    controller = BankingWorkflowController(ConversationLedger())
    controller.observe_knowledge_evidence((reference,))
    assert (
        invoke(
            controller,
            [
                item(
                    "answer",
                    completion_mode="information",
                    evidence="A delivered public revision is available for the answer.",
                    evidence_refs=[
                        {
                            "kind": "knowledge_revision",
                            "reference_id": "public-policy",
                            "version": "1",
                            "digest": "a" * 64,
                        }
                    ],
                )
            ],
        ).is_error
        is False
    )

    context_payload = json.loads(
        controller.render_context()
        .split("<workflow-plan>\n", 1)[1]
        .split("\n</workflow-plan>", 1)[0]
    )
    model_reference = context_payload["items"][0]["evidence_refs"][0]
    exported_reference = controller.export_state()["items"][0]["evidence_refs"][0]

    assert model_reference == {
        "digest": "a" * 64,
        "kind": "knowledge_revision",
        "reference_id": "public-policy",
        "version": "1",
    }
    assert exported_reference["available"] is True


def test_controller_restores_legacy_terminal_block_as_unresolved() -> None:
    ledger = ConversationLedger()
    legacy_item = item(
        "legacy-block",
        status="blocked",
        next_operation={
            "owner": "none",
            "kind": "none",
            "tool_name": "",
            "arguments": {},
            "summary": "",
        },
        evidence="Legacy terminal block.",
        evidence_refs=[
            {
                "kind": "customer_message",
                "reference_id": "legacy-message",
            }
        ],
    )
    legacy_item.pop("completion_mode")
    history = [
        AssistantMessage.text(
            "Legacy response.",
            raw_data={
                "cayu_tau3_workflow": {
                    "customer_evidence_floors": {"legacy-block": 0},
                    "items": [legacy_item],
                    "last_response_disposition": None,
                    "revision": 1,
                    "schema_version": 7,
                    "tool_evidence_floors": {"legacy-block": 0},
                }
            },
        )
    ]

    restored = BankingWorkflowController.from_history(history, ledger=ledger)

    assert restored.revision == 1
    assert restored.export_state()["schema_version"] == 15
    assert restored.export_state()["items"][0]["completion_mode"] == "operation"
    admission = restored.review_terminal_output("Still blocked.")
    assert admission.allowed is False
    assert "blocked" in admission.reason


def test_controller_reconstructs_legacy_operation_clarification_handoffs() -> None:
    controller = BankingWorkflowController(ConversationLedger())
    waiting = item(
        "legacy-operation",
        completion_mode="operation",
        status="awaiting_customer",
        evidence="The operation requires customer input.",
    )
    assert invoke(controller, [waiting]).is_error is False
    assert (
        prepare_response(
            controller,
            plan_revision=controller.revision,
            dispositions=[("legacy-operation", "ask_customer")],
        ).is_error
        is False
    )
    assert controller.review_terminal_output("Please provide the required input.").allowed is True
    legacy_snapshot = controller.export_state()
    legacy_snapshot["schema_version"] = 9
    legacy_snapshot.pop("customer_response_handoffs")

    restored = BankingWorkflowController.from_history(
        [
            AssistantMessage.text(
                "Please provide the required input.",
                raw_data={"cayu_tau3_workflow": legacy_snapshot},
            )
            for _ in range(3)
        ],
        ledger=ConversationLedger(),
    )
    exhausted = prepare_response(
        restored,
        plan_revision=restored.revision,
        dispositions=[("legacy-operation", "ask_customer")],
    )

    assert restored.export_state()["customer_response_handoffs"] == {"legacy-operation": 3}
    assert exhausted.is_error is True
    assert exhausted.diagnostic_code == "response_without_new_customer_input"
    assert "before a new customer message" in exhausted.content


def test_controller_rejects_fabricated_and_failed_completion_evidence() -> None:
    ledger = ConversationLedger()
    controller = BankingWorkflowController(ledger)

    fabricated = invoke(
        controller,
        [
            item(
                "one",
                status="completed",
                evidence="claimed success",
                evidence_refs=[{"kind": "tool_result", "reference_id": "missing-call"}],
            )
        ],
    )

    ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="failed-1",
                    name="change_user_email",
                    arguments={"user_id": "user-7", "new_email": "new@example.com"},
                    requestor="assistant",
                )
            ],
        )
    )
    ledger.observe_input(
        ToolMessage(
            id="failed-1",
            role="tool",
            requestor="assistant",
            content="Error: rejected",
        )
    )
    failed = invoke(
        controller,
        [
            item(
                "one",
                status="completed",
                evidence="the mutation failed",
                evidence_refs=[{"kind": "tool_error", "reference_id": "failed-1"}],
            )
        ],
    )

    assert fabricated.is_error is True
    assert fabricated.diagnostic_code == "sync_tool_reference_unavailable"
    assert "unavailable tool_result" in fabricated.content
    assert failed.is_error is True
    assert failed.diagnostic_code == "sync_tool_error_completion_rejected"
    assert "cannot complete from a failed tool result" in failed.content
    assert controller.revision == 0


@pytest.mark.parametrize("kind", ["tool_result", "tool_error"])
def test_unavailable_tool_reference_feedback_offers_valid_source_repair(kind) -> None:
    ledger = ConversationLedger()
    controller = BankingWorkflowController(ledger)
    reference = WorkEvidenceReference(
        kind="knowledge_revision", reference_id="public-policy", version="1"
    )
    controller.observe_knowledge_evidence((reference,))
    rejected = invoke(
        controller,
        [
            item(
                "one",
                status="blocked",
                evidence="A public policy prerequisite is unresolved",
                evidence_refs=[{"kind": kind, "reference_id": "non-environment-call"}],
            )
        ],
    )
    assert rejected.is_error
    assert rejected.diagnostic_code == "sync_tool_reference_unavailable"
    assert "knowledge_revision evidence_ref" in rejected.content
    assert "policy denial is not an executed tool failure" in rejected.content
    assert ledger.tool_action_status("non-environment-call") is None
    assert controller.revision == 0
    repaired = invoke(
        controller,
        [
            item(
                "one",
                status="blocked",
                evidence="A public policy prerequisite is unresolved",
                evidence_refs=[
                    reference.model_dump(mode="json", include={"kind", "reference_id", "version"})
                ],
            )
        ],
    )
    assert not repaired.is_error
    assert controller.export_state()["items"][0]["outcome"] == "blocked"
    assert ledger.tool_action_status("non-environment-call") is None


def test_completion_mode_is_immutable_and_prevents_operation_relabeling() -> None:
    reference = WorkEvidenceReference(
        kind="knowledge_revision",
        reference_id="public-policy",
        version="1",
        digest="c" * 64,
    )
    evidence_refs = [
        {
            "kind": "knowledge_revision",
            "reference_id": "public-policy",
            "version": "1",
            "digest": "c" * 64,
        }
    ]
    controller = BankingWorkflowController(ConversationLedger())
    controller.observe_knowledge_evidence((reference,))
    assert invoke(controller, [item("perform-action")]).is_error is False

    relabeled = invoke(
        controller,
        [
            item(
                "perform-action",
                completion_mode="operation",
                status="completed",
                outcome="answered",
                evidence="Policy information was retrieved but no action ran.",
                evidence_refs=evidence_refs,
            )
        ],
    )
    changed_mode = invoke(
        controller,
        [
            item(
                "perform-action",
                completion_mode="information",
                status="completed",
                outcome="answered",
                evidence="Attempt to downgrade the requested operation.",
                evidence_refs=evidence_refs,
            )
        ],
    )

    informational = BankingWorkflowController(ConversationLedger())
    informational.observe_knowledge_evidence((reference,))
    assert (
        invoke(
            informational,
            [item("answer-question", completion_mode="information")],
        ).is_error
        is False
    )
    answered = invoke(
        informational,
        [
            item(
                "answer-question",
                completion_mode="information",
                status="completed",
                outcome="answered",
                evidence="The delivered policy answers the question.",
                evidence_refs=evidence_refs,
            )
        ],
    )

    assert relabeled.is_error is True
    assert relabeled.diagnostic_code == "sync_item_outcome_rejected"
    assert "completion_mode operation" in relabeled.content
    assert changed_mode.is_error is True
    assert changed_mode.diagnostic_code == "revision_completion_mode_changed"
    assert changed_mode.remediation is not None
    assert changed_mode.remediation.allowed_repairs == ("preserve_prior_completion_mode",)
    assert "completion_mode is immutable" in changed_mode.content
    assert answered.is_error is False


def test_controller_accepts_only_observed_customer_and_knowledge_references() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("The customer-side update succeeded."))
    controller = BankingWorkflowController(ledger)
    controller.observe_knowledge_evidence(
        (
            WorkEvidenceReference(
                kind="knowledge_revision",
                reference_id="policy-document",
                version="1",
                digest="a" * 64,
            ),
        )
    )

    initial = invoke(
        controller,
        [
            item(
                "customer-action",
                status="awaiting_customer",
                evidence="Waiting for the customer-side update result.",
                next_operation={
                    "owner": "customer",
                    "kind": "tool",
                    "tool_name": "customer_update_1234",
                    "summary": "Run the provided customer-side update.",
                },
            ),
            item(
                "answer-policy",
                status="completed",
                outcome="answered",
                evidence="answer grounded in a delivered policy revision",
                evidence_refs=[
                    {
                        "kind": "knowledge_revision",
                        "reference_id": "policy-document",
                        "version": "1",
                        "digest": "a" * 64,
                    }
                ],
            ),
        ],
    )
    ledger.observe_input(UserMessage.text("The customer-side update is now confirmed."))
    accepted = invoke(
        controller,
        [
            item(
                "customer-action",
                status="completed",
                evidence="customer reported the result",
                evidence_refs=[
                    {
                        "kind": "customer_message",
                        "reference_id": "customer-message:2",
                    }
                ],
            ),
            item(
                "answer-policy",
                status="completed",
                outcome="answered",
                evidence="answer grounded in a delivered policy revision",
                evidence_refs=[
                    {
                        "kind": "knowledge_revision",
                        "reference_id": "policy-document",
                        "version": "1",
                        "digest": "a" * 64,
                    }
                ],
            ),
        ],
    )

    assert initial.is_error is False
    assert accepted.is_error is False

    wrong_revision = invoke(
        BankingWorkflowController(ledger),
        [
            item(
                "answer-policy",
                status="completed",
                outcome="answered",
                evidence="fabricated revision",
                evidence_refs=[
                    {
                        "kind": "knowledge_revision",
                        "reference_id": "policy-document",
                        "version": "2",
                    }
                ],
            )
        ],
    )
    assert wrong_revision.is_error is True
    assert wrong_revision.diagnostic_code == "sync_knowledge_reference_undelivered"
    assert "not delivered by Cayu" in wrong_revision.content


def test_controller_enforces_typed_outcomes_and_next_operations() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please do this, then wait for me."))
    controller = BankingWorkflowController(ledger)

    missing_operation = invoke(
        controller,
        [
            item(
                "act",
                next_operation={
                    "owner": "none",
                    "kind": "none",
                    "tool_name": "",
                    "summary": "",
                },
            )
        ],
    )
    wrong_wait_owner = invoke(
        controller,
        [
            item(
                "act",
                status="awaiting_customer",
                evidence="Waiting for the requested customer response.",
                next_operation={
                    "owner": "agent",
                    "kind": "customer_response",
                    "tool_name": "",
                    "summary": "Wait for the customer",
                },
            )
        ],
    )
    wrong_active_response = invoke(
        controller,
        [
            item(
                "act",
                status="in_progress",
                next_operation={
                    "owner": "agent",
                    "kind": "customer_response",
                    "tool_name": "",
                    "summary": "Wait for the customer",
                },
            )
        ],
    )
    answered_from_customer_claim = invoke(
        controller,
        [
            item(
                "act",
                status="completed",
                outcome="answered",
                evidence="The customer message supposedly proves a policy answer.",
                evidence_refs=[
                    {
                        "kind": "customer_message",
                        "reference_id": "customer-message:1",
                    }
                ],
            )
        ],
    )
    control_plane_operations = [
        invoke(
            BankingWorkflowController(ledger),
            [
                item(
                    "act",
                    next_operation={
                        "owner": "agent",
                        "kind": "tool",
                        "tool_name": tool_name,
                        "summary": "Revise controller state.",
                    },
                )
            ],
        )
        for tool_name in (SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL, PREPARE_RESPONSE_TOOL)
    ]

    assert missing_operation.is_error is True
    assert "actionable status requires a next_operation" in missing_operation.content
    assert wrong_wait_owner.is_error is True
    assert "awaiting_customer requires a customer-owned" in wrong_wait_owner.content
    assert wrong_active_response.is_error is True
    assert "customer_response must be customer-owned" in wrong_active_response.content
    assert answered_from_customer_claim.is_error is True
    assert "lacks a compatible evidence reference" in answered_from_customer_claim.content
    assert all(result.is_error for result in control_plane_operations)
    assert all(
        "cannot plan controller bookkeeping tool" in result.content
        for result in control_plane_operations
    )


def test_controller_rejects_unoffered_environment_operation_names() -> None:
    ledger = ConversationLedger()
    controller = BankingWorkflowController(
        ledger,
        available_environment_tool_names={"call_discoverable_agent_tool"},
    )

    hidden_tool = invoke(
        controller,
        [
            item(
                "open-account",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": "open_bank_account_4821",
                    "summary": "Open the selected account.",
                },
            )
        ],
    )
    wrapped_tool = invoke(
        controller,
        [
            item(
                "open-account",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": "call_discoverable_agent_tool",
                    "arguments": {
                        "agent_tool_name": "open_bank_account_4821",
                        "arguments": '{"user_id":"user-7"}',
                    },
                    "summary": "Call the unlocked account-opening tool.",
                },
            )
        ],
    )

    assert hidden_tool.is_error is True
    assert "not an offered top-level environment tool" in hidden_tool.content
    assert wrapped_tool.is_error is False
    assert controller.required_action_tool_names() == (
        "call_discoverable_agent_tool",
        UPDATE_WORK_PLAN_TOOL,
    )


def test_returned_planned_tool_result_requires_reconciliation_before_reuse() -> None:
    ledger = ConversationLedger()
    controller = BankingWorkflowController(
        ledger,
        available_environment_tool_names={"change_user_email"},
    )
    arguments = {"user_id": "user-7", "new_email": "new@example.com"}
    planned = invoke(
        controller,
        [
            item(
                "change-email",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": "change_user_email",
                    "arguments": arguments,
                    "summary": "Apply the authorized email change.",
                },
            )
        ],
    )
    call = ToolCallPart(
        tool_call_id="change-1",
        tool_name="change_user_email",
        arguments=arguments,
    )

    assert planned.is_error is False
    assert controller.review((call,))[0].allowed is True
    ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="change-1",
                    name="change_user_email",
                    arguments=arguments,
                    requestor="assistant",
                )
            ],
        )
    )
    ledger.observe_input(
        ToolMessage(
            id="change-1",
            role="tool",
            requestor="assistant",
            content="Email updated successfully.",
        )
    )

    rejected_replay = controller.review((call,))[0]
    assert rejected_replay.allowed is False
    assert controller.required_action_tool_names() == (UPDATE_WORK_PLAN_TOOL,)
    assert "ready_agent_tool_batches" not in controller.render_context()


def test_controller_renders_dependency_ready_calls_as_one_exact_batch() -> None:
    controller = BankingWorkflowController(
        ConversationLedger(),
        available_environment_tool_names={
            "call_discoverable_agent_tool",
            "give_discoverable_user_tool",
        },
    )
    dispute_items = [
        item(
            f"dispute-{index}",
            next_operation={
                "owner": "agent",
                "kind": "tool",
                "tool_name": "call_discoverable_agent_tool",
                "arguments": {
                    "agent_tool_name": "file_credit_card_transaction_dispute_4829",
                    "arguments": json.dumps({"transaction_id": f"txn-{index}"}),
                },
                "summary": f"File dispute {index}.",
            },
        )
        for index in range(3)
    ]
    result = invoke(
        controller,
        [
            *dispute_items,
            item(
                "different-wrapper-operation",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": "call_discoverable_agent_tool",
                    "arguments": {
                        "agent_tool_name": "order_replacement_credit_card_7291",
                        "arguments": json.dumps({"account_id": "card-1"}),
                    },
                    "summary": "Replace the card after disputes are filed.",
                },
            ),
            *[
                item(
                    f"customer-tool-{index}",
                    next_operation={
                        "owner": "agent",
                        "kind": "tool",
                        "tool_name": "give_discoverable_user_tool",
                        "arguments": {"discoverable_tool_name": f"customer_action_{index}"},
                        "summary": f"Give customer action {index}.",
                    },
                )
                for index in range(2)
            ],
        ],
    )

    assert result.is_error is False
    payload = json.loads(
        controller.render_context()
        .split("<workflow-plan>\n", 1)[1]
        .split("\n</workflow-plan>", 1)[0]
    )
    batches = payload["ready_agent_tool_batches"]
    assert len(batches) == 1
    assert batches[0]["tool_name"] == "call_discoverable_agent_tool"
    assert batches[0]["discoverable_agent_tool_name"] == "file_credit_card_transaction_dispute_4829"
    assert batches[0]["item_ids"] == [
        "dispute-0",
        "dispute-1",
        "dispute-2",
    ]
    assert "Issue every call" in payload["ready_agent_tool_batches_instruction"]
    assert controller.required_action_batch_size("call_discoverable_agent_tool") == 3
    assert controller.required_action_batch_size("give_discoverable_user_tool") == 1


def test_controller_batches_dependency_ready_knowledge_searches() -> None:
    controller = BankingWorkflowController(ConversationLedger())
    result = invoke(
        controller,
        [
            item(
                "search-card-policy",
                completion_mode="information",
                next_operation={
                    "owner": "agent",
                    "kind": "knowledge_search",
                    "tool_name": "",
                    "summary": "Retrieve the relevant card policy.",
                },
            ),
            item(
                "search-transfer-policy",
                completion_mode="information",
                next_operation={
                    "owner": "agent",
                    "kind": "knowledge_search",
                    "tool_name": "",
                    "summary": "Retrieve the relevant transfer policy.",
                },
            ),
        ],
    )

    assert result.is_error is False
    assert controller.required_action_batch_size(KNOWLEDGE_SEARCH_TOOL) == 2
    assert controller.required_action_batch_size(UPDATE_WORK_PLAN_TOOL) == 1


def test_controller_only_stages_one_plan_publication_before_external_calls() -> None:
    controller = BankingWorkflowController(ConversationLedger())

    assert controller.allows_staged_external_calls((SYNC_WORK_PLAN_TOOL,)) is True
    assert controller.allows_staged_external_calls((UPDATE_WORK_PLAN_TOOL,)) is True
    assert controller.allows_staged_external_calls((KNOWLEDGE_SEARCH_TOOL,)) is False
    assert (
        controller.allows_staged_external_calls((SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL))
        is False
    )


@pytest.mark.parametrize("name", ["", " call_discoverable_agent_tool", "bad name "])
def test_controller_rejects_invalid_required_action_batch_names(name: str) -> None:
    controller = BankingWorkflowController(ConversationLedger())

    with pytest.raises(ValueError, match="normalized nonblank"):
        controller.required_action_batch_size(name)


def test_controller_allows_named_internal_computation_operations() -> None:
    controller = BankingWorkflowController(
        ConversationLedger(),
        available_environment_tool_names=set(),
        available_private_tool_names={AUDIT_CREDIT_CARD_REWARDS_TOOL},
    )

    result = invoke(
        controller,
        [
            item(
                "audit-rewards",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": AUDIT_CREDIT_CARD_REWARDS_TOOL,
                    "summary": "Run the deterministic reward audit.",
                },
            )
        ],
    )

    assert result.is_error is False


def test_controller_rejects_an_unregistered_private_operation() -> None:
    controller = BankingWorkflowController(
        ConversationLedger(),
        available_environment_tool_names=set(),
        available_private_tool_names={AUDIT_CREDIT_CARD_REWARDS_TOOL},
    )

    result = invoke(
        controller,
        [
            item(
                "invented-computation",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": "controller_invent_policy_decision",
                    "summary": "Run a nonexistent private computation.",
                },
            )
        ],
    )

    assert result.is_error is True
    assert result.diagnostic_code == "sync_operation_shape_rejected"
    assert "registered private operation" in result.content


def test_controller_rejects_new_search_work_after_the_capability_is_withdrawn() -> None:
    controller = BankingWorkflowController(ConversationLedger())
    planned_item = item(
        "search-policy",
        completion_mode="information",
        next_operation={
            "owner": "agent",
            "kind": "knowledge_search",
            "tool_name": "",
            "summary": "Retrieve the relevant public policy.",
        },
    )
    planned = invoke(controller, [planned_item])

    controller.set_knowledge_search_available(False)
    revised = update(
        controller,
        [
            {
                **planned_item,
                "next_operation": {
                    **planned_item["next_operation"],
                    "summary": "Search again for the relevant public policy.",
                },
            }
        ],
    )

    assert planned.is_error is False
    assert controller.required_action_tool_names() == (UPDATE_WORK_PLAN_TOOL,)
    assert controller.required_action_batch_size(KNOWLEDGE_SEARCH_TOOL) == 1
    assert revised.is_error is True
    assert revised.diagnostic_code == "sync_operation_shape_rejected"
    assert "capability is exhausted or unavailable" in revised.content


def test_controller_validates_dynamic_knowledge_search_availability() -> None:
    controller = BankingWorkflowController(ConversationLedger())

    with pytest.raises(TypeError, match="availability must be a boolean"):
        controller.set_knowledge_search_available(1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "malformed",
    [{"summary": "Waiting for prerequisite"}, {"tool_name": "unused"}, {"arguments": {"x": 1}}],
)
def test_pending_dependency_sentinel_rejection_explains_exact_repair(malformed) -> None:
    controller = BankingWorkflowController(ConversationLedger())
    sentinel = {"owner": "none", "kind": "none", "tool_name": "", "arguments": {}, "summary": ""}
    rejected = invoke(
        controller,
        [
            item("first"),
            item("second", depends_on=["first"], next_operation={**sentinel, **malformed}),
        ],
    )
    assert rejected.is_error
    assert "arguments={}, summary=''" in rejected.content
    assert "dependency explanations in evidence" in rejected.content
    assert controller.revision == 0
    repaired = invoke(
        controller,
        [item("first"), item("second", depends_on=["first"], next_operation=sentinel)],
    )
    assert not repaired.is_error
    assert controller.revision == 1


def test_pending_dependent_work_can_omit_an_operation_only_until_dependency_completes() -> None:
    ledger = ConversationLedger()
    controller = BankingWorkflowController(ledger)
    none_operation = {
        "owner": "none",
        "kind": "none",
        "tool_name": "",
        "summary": "",
    }
    initial = invoke(
        controller,
        [
            item(
                "verify",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": "log_verification",
                    "arguments": {"user_id": "user-7"},
                    "summary": "Log the verified customer identity.",
                },
            ),
            item("act", depends_on=["verify"], next_operation=none_operation),
        ],
    )
    observe_successful_tool_result(ledger)
    stale = invoke(
        controller,
        [
            item(
                "verify",
                status="completed",
                evidence="Verification succeeded.",
                evidence_refs=[{"kind": "tool_result", "reference_id": "verify-1"}],
            ),
            item("act", depends_on=["verify"], next_operation=none_operation),
        ],
    )

    assert initial.is_error is False
    assert stale.is_error is True
    assert "actionable status requires a next_operation" in stale.content
    assert controller.revision == 1


def test_pending_dependent_work_does_not_block_prerequisite_customer_handoff() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(
        UserMessage.text("Assess which card fits me, then help me apply for the selected card.")
    )
    controller = BankingWorkflowController(ledger)
    controller.observe_knowledge_evidence(
        (
            WorkEvidenceReference(
                kind="knowledge_revision",
                reference_id="credit-card-fit-guide",
                version="1",
                digest="a" * 64,
            ),
        )
    )
    fit = item(
        "assess-fit",
        goal="Assess which credit card fits the customer's requirements.",
        completion_mode="information",
        status="awaiting_customer",
        evidence="The assessment requires the customer's missing preferences.",
        next_operation={
            "owner": "customer",
            "kind": "customer_response",
            "tool_name": "",
            "summary": "Provide the missing card preferences.",
        },
    )
    application = item(
        "apply-card",
        goal="Apply for the selected credit card.",
        completion_mode="operation",
        status="pending",
        depends_on=["assess-fit"],
        next_operation={
            "owner": "none",
            "kind": "none",
            "tool_name": "",
            "summary": "",
        },
    )

    installed = invoke(controller, [fit, application])
    prepared = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[("assess-fit", "ask_customer")],
    )
    admitted = controller.review_terminal_output("What matters most to you in a card?")

    assert installed.is_error is False
    assert prepared.is_error is False
    assert admitted.allowed is True

    ledger.observe_input(UserMessage.text("I care most about travel rewards."))
    fit["status"] = "completed"
    fit["outcome"] = "answered"
    fit["evidence"] = "The customer supplied the requested preference."
    fit["evidence_refs"] = [
        {
            "kind": "knowledge_revision",
            "reference_id": "credit-card-fit-guide",
            "version": "1",
            "digest": "a" * 64,
        }
    ]
    fit["next_operation"] = {
        "owner": "none",
        "kind": "none",
        "tool_name": "",
        "arguments": {},
        "summary": "",
    }
    stale_downstream = invoke(controller, [fit, application])

    assert stale_downstream.is_error is True
    assert stale_downstream.diagnostic_code == "sync_operation_shape_rejected"
    assert "actionable status requires a next_operation" in stale_downstream.content


def test_customer_completion_must_follow_plan_creation_or_latest_handoff() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please submit the customer-side application."))
    controller = BankingWorkflowController(ledger)
    waiting = invoke(
        controller,
        [
            item(
                "apply",
                status="awaiting_customer",
                evidence="The customer must run the provided application tool.",
                next_operation={
                    "owner": "customer",
                    "kind": "tool",
                    "tool_name": "submit_application_1234",
                    "summary": "Run the application tool with the selected product.",
                },
            )
        ],
    )
    claimed_immediately = invoke(
        controller,
        [
            item(
                "apply",
                status="completed",
                evidence="The request itself is not a completion result.",
                evidence_refs=[
                    {
                        "kind": "customer_message",
                        "reference_id": "customer-message:1",
                    }
                ],
            )
        ],
    )
    ledger.observe_input(UserMessage.text("The application tool completed successfully."))
    completed = invoke(
        controller,
        [
            item(
                "apply",
                status="completed",
                evidence="The post-handoff customer message reports success.",
                evidence_refs=[
                    {
                        "kind": "customer_message",
                        "reference_id": "customer-message:2",
                    }
                ],
            )
        ],
    )

    assert waiting.is_error is False
    assert claimed_immediately.is_error is True
    assert "must follow plan creation" in claimed_immediately.content
    assert completed.is_error is False


def test_agent_tool_completion_must_match_the_planned_tool_and_follow_the_plan() -> None:
    ledger = ConversationLedger()
    observe_successful_tool_result(ledger, call_id="old-change", tool_name="change_user_email")
    controller = BankingWorkflowController(ledger)
    planned = invoke(
        controller,
        [
            item(
                "change-email",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": "change_user_email",
                    "arguments": {"user_id": "user-7"},
                    "summary": "Apply the verified email change.",
                },
            )
        ],
    )
    stale = invoke(
        controller,
        [
            item(
                "change-email",
                status="completed",
                evidence="This result predates the plan.",
                evidence_refs=[{"kind": "tool_result", "reference_id": "old-change"}],
            )
        ],
    )
    observe_successful_tool_result(ledger, call_id="wrong-tool", tool_name="log_verification")
    wrong_tool = invoke(
        controller,
        [
            item(
                "change-email",
                status="completed",
                evidence="A different tool succeeded.",
                evidence_refs=[{"kind": "tool_result", "reference_id": "wrong-tool"}],
            )
        ],
    )
    observe_successful_tool_result(ledger, call_id="new-change", tool_name="change_user_email")
    completed = invoke(
        controller,
        [
            item(
                "change-email",
                status="completed",
                evidence="The planned email-change tool succeeded after planning.",
                evidence_refs=[{"kind": "tool_result", "reference_id": "new-change"}],
            )
        ],
    )

    assert planned.is_error is False
    assert stale.is_error is True
    assert wrong_tool.is_error is True
    assert completed.is_error is False


def test_terminal_readiness_derives_and_limits_customer_handoffs() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please change my email."))
    controller = BankingWorkflowController(ledger)
    planned = invoke(controller, [item("change-email")])

    rejected = controller.review_terminal_output("Your request is complete.")
    waiting = invoke(
        controller,
        [
            item(
                "change-email",
                status="awaiting_customer",
                evidence="The customer must provide the replacement email address.",
                next_operation={
                    "owner": "customer",
                    "kind": "customer_response",
                    "tool_name": "",
                    "summary": "Provide the replacement email address.",
                },
            )
        ],
    )
    derived = controller.review_terminal_output("What email address should I use?")
    repeated_without_customer = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[("change-email", "ask_customer")],
    )
    ledger.observe_input(UserMessage.text("I still need help finding the right address."))
    unchanged_wait = update(
        controller,
        [
            item(
                "change-email",
                status="awaiting_customer",
                evidence="The customer must provide the replacement email address.",
                next_operation={
                    "owner": "customer",
                    "kind": "customer_response",
                    "tool_name": "",
                    "summary": "Provide the replacement email address.",
                },
            )
        ],
    )
    prepared = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[("change-email", "ask_customer")],
    )
    preview = controller.preview_terminal_output("What email address should I use?")
    allowed = controller.review_terminal_output("What email address should I use?")
    ledger.observe_input(UserMessage.text("I do not know which address to provide."))
    repeated = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[("change-email", "ask_customer")],
    )
    unrelated_revision = update(
        controller,
        [
            item(
                "change-email",
                status="awaiting_customer",
                evidence="The customer must provide the replacement email address.",
                next_operation={
                    "owner": "customer",
                    "kind": "customer_response",
                    "tool_name": "",
                    "summary": "Ask for the replacement email address again.",
                },
            )
        ],
    )
    advanced = update(
        controller,
        [
            item(
                "change-email",
                status="in_progress",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": "change_user_email",
                    "arguments": {"email": "replacement@example.test"},
                    "summary": "Change the email using the supplied replacement address.",
                },
            )
        ],
    )

    assert planned.is_error is False
    assert rejected.allowed is False
    assert "change-email" in rejected.reason
    assert rejected.correction_code == "response_actionable_items"
    assert rejected.required_tool_names == (KNOWLEDGE_SEARCH_TOOL, UPDATE_WORK_PLAN_TOOL)
    assert waiting.is_error is False
    assert derived == TerminalOutputAdmission(allowed=True)
    assert repeated_without_customer.is_error is True
    assert repeated_without_customer.diagnostic_code == "response_without_new_customer_input"
    assert unchanged_wait.is_error is False
    assert prepared.is_error is False
    assert preview == TerminalOutputAdmission(allowed=True)
    assert allowed == TerminalOutputAdmission(allowed=True)
    assert repeated.is_error is True
    assert repeated.diagnostic_code == "response_repeated_clarification"
    assert unrelated_revision.is_error is False
    assert advanced.is_error is False


def test_terminal_readiness_requires_a_populated_plan() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please help with my banking request."))
    controller = BankingWorkflowController(ledger)

    admission = controller.review_terminal_output("Here is a response without a plan.")

    assert admission.allowed is False
    assert "populated workflow plan" in admission.reason
    assert admission.correction_code == "response_without_plan"
    assert admission.required_tool_names == (SYNC_WORK_PLAN_TOOL,)


def test_semantic_rejection_requires_a_changed_outcome_identity() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please perform an account operation."))
    controller = BankingWorkflowController(ledger)
    waiting = item(
        "account-operation",
        completion_mode="operation",
        status="awaiting_customer",
        evidence="The operation requires customer input.",
    )
    assert invoke(controller, [waiting]).is_error is False
    assert (
        prepare_response(
            controller,
            plan_revision=controller.revision,
            dispositions=[("account-operation", "ask_customer")],
        ).is_error
        is False
    )

    controller.reject_semantic_plan()
    terminal = controller.review_terminal_output("Please provide the required input.")
    repeated_plan = invoke(controller, [waiting])
    repeated_disposition = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[("account-operation", "ask_customer")],
    )
    waiting["completion_mode"] = "information"
    forbidden_reclassification = invoke(controller, [waiting])
    waiting["completion_mode"] = "operation"
    waiting["next_operation"]["summary"] = "Provide all required input."
    irrelevant_revision = invoke(controller, [waiting])
    additional = item(
        "additional-operation",
        completion_mode="operation",
        status="awaiting_customer",
        evidence="The additional operation requires customer input.",
    )
    revised = invoke(controller, [waiting, additional])
    prepared = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[
            ("account-operation", "ask_customer"),
            ("additional-operation", "ask_customer"),
        ],
    )

    assert terminal.allowed is False
    assert "failed semantic coverage" in terminal.reason
    assert terminal.correction_code == "response_semantic_coverage_rejected"
    assert terminal.required_tool_names == (UPDATE_WORK_PLAN_TOOL,)
    assert repeated_plan.is_error is True
    assert repeated_disposition.is_error is True
    assert repeated_plan.diagnostic_code == "sync_semantic_coverage_rejected"
    assert repeated_disposition.diagnostic_code == "response_semantic_coverage_rejected"
    assert forbidden_reclassification.is_error is True
    assert irrelevant_revision.is_error is True
    assert revised.is_error is False
    assert prepared.is_error is False
    assert controller.review_terminal_output("Please provide all required input.").allowed is True


def test_semantically_rejected_plan_cannot_authorize_external_calls() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please perform an account operation."))
    controller = BankingWorkflowController(ledger)
    planned = item(
        "account-operation",
        completion_mode="operation",
        status="in_progress",
        next_operation={
            "owner": "agent",
            "kind": "tool",
            "tool_name": "change_account",
            "arguments": {"account_id": "account-1"},
            "summary": "Perform the requested account change.",
        },
    )
    assert invoke(controller, [planned]).is_error is False
    controller.reject_semantic_plan(
        required_new_item_mode=WorkCompletionMode.OPERATION,
        required_new_item_count=1,
    )

    admission = controller.review(
        [
            ToolCallPart(
                tool_call_id="change-1",
                tool_name="change_account",
                arguments={"account_id": "account-1"},
            )
        ]
    )[0]

    assert admission.allowed is False
    assert "failed semantic coverage review" in admission.reason


def test_missing_outcome_semantic_repair_requires_a_new_item_in_the_directed_mode() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please resolve both requests."))
    controller = BankingWorkflowController(ledger)
    existing = item("existing", completion_mode="information")
    assert invoke(controller, [existing]).is_error is False
    controller.reject_semantic_plan(
        required_new_item_mode=WorkCompletionMode.OPERATION,
        required_new_item_count=2,
    )

    incomplete_repair = invoke(
        controller,
        [existing, item("first-operation", completion_mode="operation")],
    )
    corrected = invoke(
        controller,
        [
            existing,
            item("first-operation", completion_mode="operation"),
            item("second-operation", completion_mode="operation"),
        ],
    )

    assert incomplete_repair.is_error is True
    assert "at least 2 new operation-mode outcome item(s)" in incomplete_repair.content
    assert corrected.is_error is False


def test_wrong_mode_semantic_repair_requires_more_operation_coverage() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please perform the requested action."))
    controller = BankingWorkflowController(ledger)
    proposed = item("requested-action", completion_mode="information")
    assert invoke(controller, [proposed]).is_error is False
    controller.reject_semantic_plan(
        reclassifiable_item_ids=("requested-action",),
        require_operation_increase=True,
    )

    unchanged_direction = invoke(
        controller,
        [proposed, item("extra-information", completion_mode="information")],
    )
    proposed["completion_mode"] = "operation"
    corrected = invoke(controller, [proposed])

    assert unchanged_direction.is_error is True
    assert "additional operation-mode outcome" in unchanged_direction.content
    assert corrected.is_error is False


def test_operation_clarification_budget_is_durable_and_requires_a_new_path() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please perform an account operation."))
    controller = BankingWorkflowController(ledger)
    waiting = item(
        "account-operation",
        completion_mode="operation",
        status="awaiting_customer",
        evidence="The operation requires customer input.",
    )
    assert invoke(controller, [waiting]).is_error is False

    for index in range(8):
        if index:
            ledger.observe_input(UserMessage.text(f"Here is partial answer {index}."))
            waiting = item(
                "account-operation",
                completion_mode="operation",
                status="awaiting_customer",
                evidence="The operation requires another customer fact.",
                next_operation={
                    "owner": "customer",
                    "kind": "customer_response",
                    "tool_name": "",
                    "summary": f"Ask for required fact {index}.",
                },
            )
            assert update(controller, [waiting]).is_error is False
        prepared = prepare_response(
            controller,
            plan_revision=controller.revision,
            dispositions=[("account-operation", "ask_customer")],
        )
        assert prepared.is_error is False
        assert (
            controller.review_terminal_output(f"Please provide required fact {index}.").allowed
            is True
        )

    snapshot = controller.export_state()
    ledger.observe_input(UserMessage.text("Here is another partial answer."))
    exhausted = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[("account-operation", "ask_customer")],
    )
    restored = BankingWorkflowController.from_history(
        [
            AssistantMessage.text(
                "Please provide the missing input.", raw_data={"cayu_tau3_workflow": snapshot}
            )
        ],
        ledger=ledger,
    )
    restored_exhausted = prepare_response(
        restored,
        plan_revision=restored.revision,
        dispositions=[("account-operation", "ask_customer")],
    )

    assert exhausted.is_error is True
    assert "8-turn operation clarification safety ceiling" in exhausted.content
    assert snapshot["customer_response_handoffs"] == {"account-operation": 8}
    assert len(snapshot["customer_response_handoff_signatures"]["account-operation"]) == 8
    assert snapshot["customer_response_handoff_sequences"] == {"account-operation": 8}
    assert restored_exhausted.is_error is True
    assert "8-turn operation clarification safety ceiling" in restored_exhausted.content


def test_information_clarifications_are_not_subject_to_operation_budget() -> None:
    controller = BankingWorkflowController(ConversationLedger())
    waiting = item(
        "information-request",
        completion_mode="information",
        status="awaiting_customer",
        evidence="The question is ambiguous and needs clarification.",
    )
    assert invoke(controller, [waiting]).is_error is False

    for _ in range(4):
        prepared = prepare_response(
            controller,
            plan_revision=controller.revision,
            dispositions=[("information-request", "ask_customer")],
        )
        assert prepared.is_error is False
        assert controller.review_terminal_output("Please clarify the question.").allowed is True
    assert controller.export_state()["customer_response_handoffs"] == {"information-request": 0}


def test_blocked_work_requires_executable_recovery_and_cannot_close() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please resolve the request or get me help."))
    message_id = ledger.customer_message_ids[-1]
    controller = BankingWorkflowController(ledger)
    evidence = [
        {
            "kind": "customer_message",
            "reference_id": message_id,
        }
    ]

    rejected = invoke(
        controller,
        [
            item(
                "needs-help",
                status="blocked",
                next_operation={
                    "owner": "none",
                    "kind": "none",
                    "tool_name": "",
                    "arguments": {},
                    "summary": "",
                },
                evidence="Ordinary completion is blocked.",
                evidence_refs=evidence,
            )
        ],
    )
    accepted = invoke(
        controller,
        [
            item(
                "needs-help",
                status="blocked",
                evidence="Ordinary completion is blocked and needs escalation.",
                evidence_refs=evidence,
            )
        ],
    )
    terminal = controller.review_terminal_output("I cannot help with that.")
    final_disposition = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[("needs-help", "final")],
    )
    escalation = controller.review(
        (
            ToolCallPart(
                tool_call_id="handoff-1",
                tool_name="transfer_to_human_agents",
                arguments={},
            ),
        )
    )

    assert rejected.is_error is True
    assert "recovery or escalation tool" in rejected.content
    assert accepted.is_error is False
    assert terminal.allowed is False
    assert "transfer_to_human_agents" in terminal.reason
    assert terminal.correction_code == "response_actionable_items"
    assert terminal.required_tool_names == (HUMAN_TRANSFER_TOOL, UPDATE_WORK_PLAN_TOOL)
    assert final_disposition.is_error is True
    assert final_disposition.diagnostic_code == "response_actionable_items"
    assert "remain actionable" in final_disposition.content
    assert escalation[0].allowed is True


def test_response_disposition_rejects_stale_incomplete_or_wrong_purpose() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please help with these two customer actions."))
    controller = BankingWorkflowController(ledger)
    assert (
        invoke(
            controller,
            [
                item(
                    "answer",
                    status="awaiting_customer",
                    evidence="The customer must answer a required question.",
                ),
                item(
                    "run-tool",
                    status="awaiting_customer",
                    evidence="The customer must run the provided tool.",
                    next_operation={
                        "owner": "customer",
                        "kind": "tool",
                        "tool_name": "customer_action_1234",
                        "summary": "Run the provided customer-side action.",
                    },
                ),
            ],
        ).is_error
        is False
    )

    stale = prepare_response(
        controller,
        plan_revision=controller.revision - 1,
        dispositions=[
            ("answer", "ask_customer"),
            ("run-tool", "handoff_customer_tool"),
        ],
    )
    incomplete = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[("answer", "ask_customer")],
    )
    wrong = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[
            ("answer", "handoff_customer_tool"),
            ("run-tool", "ask_customer"),
        ],
    )
    accepted = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[
            ("answer", "ask_customer"),
            ("run-tool", "handoff_customer_tool"),
        ],
    )

    assert stale.is_error is True
    assert stale.diagnostic_code == "response_stale_revision"
    assert "current revision" in stale.content
    assert incomplete.is_error is True
    assert incomplete.diagnostic_code == "response_awaiting_coverage"
    assert "cover every" in incomplete.content
    assert wrong.is_error is True
    assert wrong.diagnostic_code == "response_disposition_mismatch"
    assert "requires disposition" in wrong.content
    assert accepted.is_error is False


@pytest.mark.parametrize(
    "customer",
    [
        "I want the single card in your lineup that offers the highest cash back. Recommend one.",
        "I want a credit card for everyday purchases.",
        "Which savings account would fit my needs?",
        "Je cherche une carte adaptée à mes voyages.",
        "I selected the Jade Rewards Card, but need to clarify its eligibility first.",
    ],
)
def test_controller_leaves_language_interpretation_to_semantic_review(customer):
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text(customer))
    controller = BankingWorkflowController(ledger)
    waiting = item(
        "selection",
        goal="Select and obtain the appropriate product.",
        status="awaiting_customer",
        evidence="The current customer request requires interpretation and a reply.",
        evidence_refs=[{"kind": "customer_message", "reference_id": "customer-message:1"}],
        next_operation={
            "owner": "customer",
            "kind": "customer_product_selection",
            "tool_name": "",
            "summary": "Ask the necessary selection question.",
        },
    )
    result = invoke(controller, [waiting])
    assert not result.is_error
    assert result.remediation is None
    assert controller.export_state()["product_selection_transition_rejections"] == 0
    # Structural acceptance is neither semantic approval nor evidence of completion.
    invalid_completion = {
        **waiting,
        "status": "completed",
        "outcome": "succeeded",
        "next_operation": {"owner": "none", "kind": "none", "tool_name": "", "summary": ""},
    }
    assert invoke(controller, [invalid_completion]).is_error


def test_service_request_for_named_product_is_not_product_acquisition() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(
        UserMessage.text(
            "I received a letter about my credit card. How can I get its referral link?"
        )
    )
    controller = BankingWorkflowController(ledger)
    waiting = item(
        "referral-link",
        goal="Get a referral link for the named credit card.",
        status="awaiting_customer",
        evidence="A customer response is needed before the service request can continue.",
    )

    result = invoke(controller, [waiting])

    assert result.is_error is False
    assert controller.revision == 1


def test_existing_card_transaction_request_is_not_product_acquisition() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(
        UserMessage.text(
            "Can you show me my recent transactions? I want to see what's been "
            "charged to my Silver Rewards Card."
        )
    )
    controller = BankingWorkflowController(ledger)
    waiting = item(
        "review-transactions",
        goal="Obtain recent transactions for the Silver Rewards Card.",
        status="awaiting_customer",
        evidence="The customer requested recent transactions for an existing card.",
    )

    result = invoke(controller, [waiting])

    assert result.is_error is False
    assert controller.revision == 1
    assert controller.export_state()["product_selection_transition_rejections"] == 0


def test_existing_card_replacement_is_not_product_acquisition() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Show me the transactions on my Silver Rewards Card."))
    ledger.observe_input(
        UserMessage.text("I need to cancel this card and get a new one right away.")
    )
    controller = BankingWorkflowController(ledger)
    waiting = item(
        "replace-card",
        goal="Obtain a replacement Silver Rewards Card.",
        status="awaiting_customer",
        evidence="The customer requested replacement of an existing card.",
    )

    result = invoke(controller, [waiting])

    assert result.is_error is False
    assert controller.revision == 1
    assert controller.export_state()["product_selection_transition_rejections"] == 0


def test_contextual_product_choice_requires_matching_selection_handoff() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(
        UserMessage.text("I received an offer concerning the Platinum Rewards Card.")
    )
    ledger.observe_input(UserMessage.text("Go ahead with a human transfer."))
    controller = BankingWorkflowController(ledger)
    waiting = item(
        "promotional-offer",
        goal="Obtain the advertised offer for the Platinum Rewards Card.",
        status="awaiting_customer",
        evidence="The customer requested help with an offer and a human transfer.",
    )

    result = invoke(controller, [waiting])

    assert result.is_error is False
    assert controller.revision == 1
    assert controller.export_state()["product_selection_transition_rejections"] == 0


def test_typed_product_selection_restores_from_public_history() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Compare your credit cards and help me apply for one."))
    controller = BankingWorkflowController(ledger)
    waiting = item(
        "apply-card",
        goal="Apply for the selected credit card product.",
        status="awaiting_customer",
        evidence="The customer requested help selecting and applying for a card.",
        next_operation={
            "owner": "customer",
            "kind": "customer_product_selection",
            "tool_name": "",
            "summary": "Ask the customer to select one of the compared credit cards.",
        },
    )
    assert invoke(controller, [waiting]).is_error is False
    assert (
        prepare_response(
            controller,
            plan_revision=controller.revision,
            dispositions=[("apply-card", "ask_product_selection")],
        ).is_error
        is False
    )
    assert controller.review_terminal_output("Which compared credit card do you select?").allowed
    snapshot = controller.export_state()

    restored = BankingWorkflowController.from_history(
        [
            AssistantMessage.text(
                "Which compared credit card do you select?",
                raw_data={"cayu_tau3_workflow": snapshot},
            )
        ],
        ledger=ledger,
    )

    assert restored.export_state() == snapshot
    assert restored.export_state()["last_response_disposition"] == {
        "items": [{"item_id": "apply-card", "kind": "ask_product_selection"}],
        "plan_revision": 1,
    }


def test_final_response_disposition_covers_every_terminal_item() -> None:
    ledger = ConversationLedger()
    controller = BankingWorkflowController(ledger)
    controller.observe_knowledge_evidence(
        (
            WorkEvidenceReference(
                kind="knowledge_revision",
                reference_id="policy-document",
                version="1",
                digest="a" * 64,
            ),
        )
    )
    assert (
        invoke(
            controller,
            [
                item(
                    "answer-one",
                    status="completed",
                    outcome="answered",
                    evidence="The answer is grounded in policy.",
                    evidence_refs=[
                        {
                            "kind": "knowledge_revision",
                            "reference_id": "policy-document",
                            "version": "1",
                            "digest": "a" * 64,
                        }
                    ],
                ),
                item(
                    "answer-two",
                    status="completed",
                    outcome="answered",
                    evidence="The second answer is grounded in policy.",
                    evidence_refs=[
                        {
                            "kind": "knowledge_revision",
                            "reference_id": "policy-document",
                            "version": "1",
                            "digest": "a" * 64,
                        }
                    ],
                ),
            ],
        ).is_error
        is False
    )

    incomplete = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[("answer-one", "final")],
    )
    wrong = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[
            ("answer-one", "final"),
            ("answer-two", "ask_customer"),
        ],
    )
    accepted = prepare_response(
        controller,
        plan_revision=controller.revision,
        dispositions=[("answer-one", "final"), ("answer-two", "final")],
    )

    assert incomplete.is_error is True
    assert wrong.is_error is True
    assert accepted.is_error is False
    assert controller.review_terminal_output("Both questions are answered.").allowed is True
    assert controller.export_state()["last_response_disposition"] == {
        "items": [
            {"item_id": "answer-one", "kind": "final"},
            {"item_id": "answer-two", "kind": "final"},
        ],
        "plan_revision": controller.revision,
    }


def test_workflow_admission_matches_external_calls_to_next_operations_one_to_one() -> None:
    controller = BankingWorkflowController(ConversationLedger())
    unplanned = controller.review(
        (
            ToolCallPart(
                tool_call_id="simple-1",
                tool_name="lookup_account",
                arguments={"account_id": "A-1"},
            ),
        )
    )
    planned = invoke(
        controller,
        [
            item(
                "change-email",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": "change_user_email",
                    "arguments": {
                        "user_id": "user-7",
                        "new_email": "new@example.com",
                    },
                    "summary": "Apply the verified email change.",
                },
            )
        ],
    )
    reviewed = controller.review(
        (
            ToolCallPart(
                tool_call_id="change-1",
                tool_name="change_user_email",
                arguments={"user_id": "user-7", "new_email": "new@example.com"},
            ),
            ToolCallPart(
                tool_call_id="change-duplicate",
                tool_name="change_user_email",
                arguments={"user_id": "user-7", "new_email": "other@example.com"},
            ),
            ToolCallPart(
                tool_call_id="lookup-unplanned",
                tool_name="lookup_account",
                arguments={"account_id": "A-1"},
            ),
        )
    )

    assert unplanned[0].allowed is False
    assert "populated workflow plan" in unplanned[0].reason
    assert planned.is_error is False
    assert [admission.allowed for admission in reviewed] == [True, False, False]
    assert "Synchronize the plan" in reviewed[1].reason


def test_human_transfer_requires_four_visible_requests_while_operations_remain() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please transfer me to a human agent."))
    controller = BankingWorkflowController(ledger)
    transfer_arguments = {
        "reason": "fraud_or_security_concern",
        "summary": "The customer requested human help after reporting stolen cards.",
    }
    planned = invoke(
        controller,
        [
            item(
                "freeze-card",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": "call_discoverable_agent_tool",
                    "arguments": {
                        "agent_tool_name": "freeze_debit_card_3892",
                        "arguments": '{"card_id":"card-1"}',
                    },
                    "summary": "Freeze the stolen debit card.",
                },
            ),
            item(
                "human-handoff",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": HUMAN_TRANSFER_TOOL,
                    "arguments": transfer_arguments,
                    "summary": "Transfer the customer to a human agent.",
                },
            ),
        ],
    )
    call = ToolCallPart(
        tool_call_id="transfer-1",
        tool_name=HUMAN_TRANSFER_TOOL,
        arguments=transfer_arguments,
    )

    assert planned.is_error is False
    for request_number in range(1, 4):
        policy = controller.response_review_state()["human_transfer_request_policy"]
        assert policy == {
            "minimum_visible_request_count": 4,
            "threshold_satisfied": False,
            "visible_request_count": request_number,
        }
        admission = controller.review((call,))[0]
        assert admission.allowed is False
        assert admission.diagnostic_code == "workflow_human_transfer_request_threshold"
        assert "fourth visible" in admission.reason
        if request_number < 3:
            ledger.observe_input(UserMessage.text("I am asking again for a human agent."))
    ledger.observe_input(UserMessage.text("This is my fourth request. Transfer me to a human."))

    assert ledger.visible_human_transfer_request_count == 4
    assert controller.response_review_state()["human_transfer_request_policy"] == {
        "minimum_visible_request_count": 4,
        "threshold_satisfied": True,
        "visible_request_count": 4,
    }
    assert controller.review((call,))[0].allowed is True


def test_human_transfer_remains_immediate_when_no_separate_operation_exists() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please transfer me to a human agent."))
    controller = BankingWorkflowController(ledger)
    transfer_arguments = {
        "reason": "customer_requests_human_no_specific_reason",
        "summary": "The customer explicitly requested a human agent.",
    }
    planned = invoke(
        controller,
        [
            item(
                "human-handoff",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": HUMAN_TRANSFER_TOOL,
                    "arguments": transfer_arguments,
                    "summary": "Transfer the customer to a human agent.",
                },
            )
        ],
    )

    admission = controller.review(
        (
            ToolCallPart(
                tool_call_id="transfer-1",
                tool_name=HUMAN_TRANSFER_TOOL,
                arguments=transfer_arguments,
            ),
        )
    )[0]

    assert planned.is_error is False
    assert controller.response_review_state()["human_transfer_request_policy"] == {
        "minimum_visible_request_count": 1,
        "threshold_satisfied": True,
        "visible_request_count": 1,
    }
    assert admission.allowed is True


def test_repeated_human_request_threshold_remains_after_other_work_finishes() -> None:
    ledger = ConversationLedger()
    for request in ("first", "second", "third"):
        ledger.observe_input(
            UserMessage.text(f"This is my {request} request. Transfer me to a human agent.")
        )
    controller = BankingWorkflowController(ledger)
    transfer_arguments = {
        "reason": "request_completed_customer_wants_human_followup",
        "summary": "The customer requested human follow-up after completed work.",
    }
    assert (
        invoke(
            controller,
            [
                item(
                    "human-handoff",
                    next_operation={
                        "owner": "agent",
                        "kind": "tool",
                        "tool_name": HUMAN_TRANSFER_TOOL,
                        "arguments": transfer_arguments,
                        "summary": "Transfer the customer to a human agent.",
                    },
                )
            ],
        ).is_error
        is False
    )
    call = ToolCallPart(
        tool_call_id="transfer-1",
        tool_name=HUMAN_TRANSFER_TOOL,
        arguments=transfer_arguments,
    )

    before_fourth = controller.review((call,))[0]
    ledger.observe_input(UserMessage.text("This is my fourth request. Transfer me to a human."))
    at_fourth = controller.review((call,))[0]

    assert before_fourth.allowed is False
    assert before_fourth.diagnostic_code == "workflow_human_transfer_request_threshold"
    assert at_fourth.allowed is True


@pytest.mark.parametrize("concrete_work", [False, True])
def test_verified_capability_gap_does_not_delay_handoff_unless_bank_work_can_advance(concrete_work):
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please transfer me to a human agent."))
    verified = False
    controller = BankingWorkflowController(ledger, verified_capability_gap=lambda: verified)
    transfer_arguments = {
        "reason": "unconfirmed_external_communication",
        "summary": "Human review requested.",
    }
    work_items = [
        item(
            "unresolved-offer",
            next_operation={
                "owner": "agent",
                "kind": "tool" if concrete_work else "knowledge_search",
                "tool_name": "protect_account" if concrete_work else "",
                "arguments": {},
                "summary": "Perform protective action"
                if concrete_work
                else "Resolve unavailable offer",
            },
        ),
        item(
            "human-handoff",
            next_operation={
                "owner": "agent",
                "kind": "tool",
                "tool_name": HUMAN_TRANSFER_TOOL,
                "arguments": transfer_arguments,
                "summary": "Transfer to human review",
            },
        ),
    ]
    assert not invoke(controller, work_items).is_error
    call = ToolCallPart(
        tool_call_id="handoff", tool_name=HUMAN_TRANSFER_TOOL, arguments=transfer_arguments
    )
    assert not controller.review((call,))[0].allowed
    verified = True
    assert controller.review((call,))[0].allowed is not concrete_work
    assert controller.response_review_state()["human_transfer_request_policy"][
        "minimum_visible_request_count"
    ] == (4 if concrete_work else 1)


def test_action_review_projection_omits_cosmetic_prose_without_mutating_the_plan():
    controller = BankingWorkflowController(ConversationLedger())
    planned = item("operation", evidence="cosmetic evidence claim")
    planned["next_operation"]["summary"] = "cosmetic operation wording"
    assert not invoke(controller, [planned]).is_error
    before = controller.export_state()
    projection = controller.render_action_review_context()
    assert "cosmetic evidence claim" not in projection
    assert "cosmetic operation wording" not in projection
    assert controller.export_state() == before
    assert '"status": "pending"' in projection


def test_semantic_review_projection_omits_arguments_and_evidence_content() -> None:
    controller = BankingWorkflowController(ConversationLedger())
    planned = invoke(
        controller,
        [
            item(
                "change-email",
                evidence="private evidence explanation",
                next_operation={
                    "owner": "agent",
                    "kind": "tool",
                    "tool_name": "change_user_email",
                    "arguments": {"private": "argument-value"},
                    "summary": "Apply the requested change.",
                },
            )
        ],
    )

    review_state = controller.semantic_review_state()
    serialized = json.dumps(review_state)

    assert planned.is_error is False
    assert review_state == {
        "items": [
            {
                "completion_mode": "operation",
                "goal": "Resolve the customer's request",
                "id": "change-email",
            }
        ]
    }
    assert "change_user_email" not in serialized
    assert "argument-value" not in serialized
    assert "private evidence explanation" not in serialized
