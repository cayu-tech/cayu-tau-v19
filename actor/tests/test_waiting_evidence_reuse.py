import pytest
from tau2.data_model.message import AssistantMessage
from test_outcome_controller import obligation, outcome, setup_plan, update

from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, BankingWorkflowController


@pytest.mark.parametrize(
    "status,outcome_value", [("awaiting_customer", "unresolved"), ("blocked", "blocked")]
)
def test_operation_waiting_parent_reuses_verified_child_blocker(status, outcome_value):
    controller, ledger = setup_plan()
    result = update(
        controller, status=status, outcome=outcome_value, obligations=[obligation("waiting")]
    )
    assert not result.is_error, result.content
    saved = controller.export_state()
    parent = saved["items"][0]
    assert parent["status"] == status and parent["outcome"] == outcome_value
    assert parent["evidence_refs"][0]["reference_id"] == "customer-message:1"
    assert parent["obligations"][0]["status"] == "waiting"
    assert parent["evidence"]
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Waiting", raw_data={"cayu_tau3_workflow": saved})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == saved


def test_new_operation_can_reuse_blocker_without_parent_evidence_fields():
    _, ledger = setup_plan()
    fresh = BankingWorkflowController(ledger, planning_mode="outcomes")
    raw = {**outcome(status="awaiting_customer"), "obligations": [obligation("waiting")]}
    raw.pop("evidence")
    raw.pop("evidence_refs")
    result = fresh.invoke(SYNC_WORK_PLAN_TOOL, {"items": [raw]})
    assert not result.is_error, result.content
    assert fresh.export_state()["items"][0]["outcome"] == "unresolved"


@pytest.mark.parametrize("child_status", ["ready", "resolved", "not_applicable"])
def test_nonwaiting_child_cannot_supply_an_operation_blocker(child_status):
    controller, _ = setup_plan()
    before = controller.export_state()
    result = update(
        controller, status="blocked", outcome="blocked", obligations=[obligation(child_status)]
    )
    assert result.is_error
    assert controller.export_state() == before


def test_fabricated_child_reference_rejects_without_mutating_parent():
    controller, _ = setup_plan()
    before = controller.export_state()
    result = update(
        controller,
        status="awaiting_customer",
        obligations=[
            obligation(
                "waiting",
                evidence_refs=[
                    {"kind": "customer_message", "reference_id": "customer-message:999"}
                ],
            )
        ],
    )
    assert result.is_error
    assert controller.export_state() == before


def test_explicit_invalid_parent_evidence_is_not_replaced_by_child():
    controller, _ = setup_plan()
    for fields in ({"evidence": ""}, {"evidence_refs": []}):
        before = controller.export_state()
        result = update(
            controller,
            status="blocked",
            outcome="blocked",
            obligations=[obligation("waiting")],
            **fields,
        )
        assert result.is_error
        assert controller.export_state() == before


def test_reusing_waiting_evidence_cannot_hide_independent_ready_work():
    controller, _ = setup_plan()
    result = update(
        controller,
        status="awaiting_customer",
        obligations=[
            obligation("waiting"),
            obligation("ready", id="independent"),
            obligation("ready", id="ask", kind="communication"),
        ],
    )
    assert not result.is_error, result.content
    assert not controller.preview_terminal_output("Question").allowed


def test_child_evidence_never_becomes_operation_success():
    controller, _ = setup_plan()
    before = controller.export_state()
    result = update(
        controller, status="completed", outcome="succeeded", obligations=[obligation("resolved")]
    )
    assert result.is_error and controller.export_state() == before


@pytest.mark.parametrize(
    "children", [None, {}, ["invalid"], [{"status": "waiting", "evidence_refs": None}]]
)
def test_new_waiting_item_rejects_malformed_children_without_raising(children):
    _, ledger = setup_plan()
    fresh = BankingWorkflowController(ledger, planning_mode="outcomes")
    raw = {**outcome(status="awaiting_customer"), "obligations": children}
    raw.pop("evidence")
    raw.pop("evidence_refs")
    result = fresh.invoke(SYNC_WORK_PLAN_TOOL, {"items": [raw]})
    assert result.is_error and not fresh.has_work_items()
