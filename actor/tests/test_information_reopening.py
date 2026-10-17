from copy import deepcopy

from cayu import WorkEvidenceReference
from tau2.data_model.message import AssistantMessage
from test_outcome_controller import obligation, outcome, setup_plan, update

from cayu_tau3_banking.controller import BankingWorkflowController


def completed():
    controller, ledger = setup_plan(outcome(completion_mode="information"))
    controller.observe_knowledge_evidence(
        [WorkEvidenceReference(kind="knowledge_revision", reference_id="policy", version="1")]
    )
    result = update(
        controller,
        status="completed",
        outcome="answered",
        evidence="Original answer",
        evidence_refs=[{"kind": "knowledge_revision", "reference_id": "policy", "version": "1"}],
    )
    assert not result.is_error, result.content
    return controller, ledger


def snapshot_message(state):
    return AssistantMessage.text("Answer", raw_data={"cayu_tau3_workflow": state})


def test_reopening_preserves_completion_and_restores_without_context_bloat():
    controller, ledger = completed()
    before = controller.export_state()
    result = update(controller, status="in_progress")
    assert not result.is_error, result.content
    reopened = controller.export_state()
    assert reopened["items"][0]["outcome"] == "unresolved"
    assert reopened["items"][0]["evidence_refs"] == []
    assert reopened["previous_information_completions"]["request"][0]["item"] == before["items"][0]
    assert "previous_information_completions" not in controller.render_context()
    restored = BankingWorkflowController.from_history(
        [snapshot_message(before), snapshot_message(reopened)],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == reopened
    assert not update(
        restored,
        status="completed",
        outcome="answered",
        evidence="Revised answer",
        evidence_refs=[{"kind": "knowledge_revision", "reference_id": "policy", "version": "1"}],
    ).is_error
    revised = restored.export_state()
    again = BankingWorkflowController.from_history(
        [snapshot_message(before), snapshot_message(revised)],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert again.export_state() == revised
    assert not update(again, status="pending").is_error
    assert len(again.export_state()["previous_information_completions"]["request"]) == 2


def test_failed_reopening_is_atomic_and_history_cannot_be_rewritten():
    controller, ledger = completed()
    before = controller.export_state()
    assert update(controller, status="in_progress", goal="Different goal").is_error
    assert controller.export_state() == before
    # A rejected goal edit has an existing remediation contract; use a clean controller.
    controller, ledger = completed()
    assert not update(controller, status="in_progress").is_error
    state = controller.export_state()
    corrupted = deepcopy(state)
    corrupted["revision"] += 1
    corrupted["previous_information_completions"]["request"][0]["item"]["evidence"] = "forged"
    restored = BankingWorkflowController.from_history(
        [snapshot_message(state), snapshot_message(corrupted)],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == state


def test_reopening_resets_research_not_completed_actions():
    # Add obligations through a fresh plan before completion.
    controller, _ = setup_plan(
        {
            **outcome(completion_mode="information"),
            "obligations": [
                obligation("ready", kind="knowledge_search"),
            ],
        }
    )
    ref = {"kind": "knowledge_revision", "reference_id": "policy", "version": "1"}
    controller.observe_knowledge_evidence([WorkEvidenceReference(**ref)])
    assert not update(
        controller,
        status="completed",
        outcome="answered",
        evidence="Old answer",
        evidence_refs=[ref],
        obligations=[obligation("resolved", kind="knowledge_search", evidence_refs=[ref])],
    ).is_error
    assert not update(controller, status="in_progress").is_error
    item = controller.export_state()["items"][0]
    assert item["obligations"][0]["status"] == "ready"
    assert item["obligations"][0]["evidence_refs"] == []


def test_completed_operation_cannot_be_reopened():
    controller, _ = setup_plan(
        outcome(
            status="completed",
            outcome="succeeded",
            prior_customer_report=True,
            evidence="Customer reported completion",
            evidence_refs=[{"kind": "customer_message", "reference_id": "customer-message:1"}],
        )
    )
    before = controller.export_state()
    assert update(controller, status="in_progress", outcome="unresolved").is_error
    assert controller.export_state() == before


def test_restore_reopening_without_archived_completion_keeps_prior_snapshot():
    controller, ledger = completed()
    before = controller.export_state()
    assert not update(controller, status="in_progress").is_error
    corrupt = controller.export_state()
    corrupt.pop("previous_information_completions")
    restored = BankingWorkflowController.from_history(
        [snapshot_message(before), snapshot_message(corrupt)],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == before


def test_completed_answer_cannot_be_silently_overwritten():
    controller, _ = completed()
    before = controller.export_state()
    assert update(controller, evidence="Changed without reopening").is_error
    assert controller.export_state() == before
