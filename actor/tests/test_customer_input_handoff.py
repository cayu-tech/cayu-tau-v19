import pytest
from tau2.data_model.message import AssistantMessage, UserMessage
from test_outcome_controller import obligation, outcome, setup_plan, update
from test_runtime_banking import assembly, call, text

from cayu_tau3_banking.controller import (
    SYNC_WORK_PLAN_TOOL,
    UPDATE_WORK_PLAN_TOOL,
    BankingWorkflowController,
)


def input_plan(*, kind="customer_input", own_action=False):
    return [
        {
            **outcome(
                "request",
                status="awaiting_customer",
                evidence="The customer has not supplied a delivery region.",
            ),
            "obligations": [
                obligation(
                    "ready",
                    id="region",
                    kind=kind,
                    goal="Ask which delivery region is required",
                    evidence="A delivery region is required and missing from the customer's request.",
                    evidence_refs=[
                        {"kind": "customer_message", "reference_id": "customer-message:1"}
                    ],
                ),
                *([obligation(id="inspect")] if own_action else []),
            ],
        },
        {**outcome("comparison"), "obligations": [obligation(kind="knowledge_search")]},
    ]


def test_evidenced_customer_question_can_pause_independent_work_without_completing_it():
    controller, _ = setup_plan(*input_plan())
    before = controller.export_state()
    assert controller.preview_terminal_output("Which delivery region?").allowed
    assert controller.export_state() == before
    assert '"mode": "customer_input_only"' in controller.render_context()
    assert update(controller, status="completed", outcome="succeeded").is_error


def test_new_question_reuses_its_evidence_without_duplicate_parent_fields():
    items = input_plan()
    items[0].pop("evidence")
    items[0].pop("evidence_refs")
    controller, _ = setup_plan(*items)
    assert controller.preview_terminal_output("Which delivery region?").allowed
    saved = controller.export_state()["items"][0]
    assert saved["evidence_refs"][0]["reference_id"] == "customer-message:1"
    assert saved["outcome"] == "unresolved"


@pytest.mark.parametrize("other_kind", ["action", "knowledge_search"])
def test_input_handoff_keeps_other_outcome_ready(other_kind):
    items = input_plan()
    items[1]["obligations"][0]["kind"] = other_kind
    controller, _ = setup_plan(*items)
    assert controller.review_terminal_output("Which delivery region?").allowed
    saved = controller.export_state()
    assert saved["items"][1]["status"] == "pending"
    assert saved["items"][1]["obligations"][0]["status"] == "ready"
    assert not controller.review_terminal_output("Which delivery region again?").allowed


def test_ordinary_communication_still_cannot_hide_independent_work():
    controller, _ = setup_plan(*input_plan(kind="communication"))
    assert not controller.preview_terminal_output("Everything is done.").allowed


def test_question_only_handoff_does_not_claim_unrelated_communication_was_delivered():
    items = input_plan()
    items[1]["obligations"][0]["kind"] = "communication"
    controller, _ = setup_plan(*items)
    assert controller.review_terminal_output("Which delivery region?").allowed
    epochs = controller.export_state()["customer_response_handoff_sequences"]
    assert epochs["request"] == 1
    assert epochs["comparison"] == 0


def test_question_does_not_skip_its_own_required_preparation():
    controller, _ = setup_plan(*input_plan(own_action=True))
    assert not controller.preview_terminal_output("Which delivery region?").allowed


@pytest.mark.parametrize(
    "patch",
    [
        {"evidence": ""},
        {"evidence_refs": []},
        {"evidence_refs": [{"kind": "customer_message", "reference_id": "customer-message:999"}]},
    ],
)
def test_customer_input_requires_real_explained_evidence(patch):
    controller, _ = setup_plan()
    before = controller.export_state()
    items = input_plan()
    items[0]["obligations"][0].update(patch)
    result = controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": items})
    assert result.is_error
    assert controller.export_state() == before


@pytest.mark.parametrize("status", ["pending", "in_progress", "blocked"])
def test_ready_customer_input_derives_only_nonterminal_parent_phase(status):
    controller, _ = setup_plan()
    items = input_plan()
    items[0]["status"] = status
    items[0]["outcome"] = "blocked" if status == "blocked" else "unresolved"
    result = controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": items})
    assert result.is_error is (status == "blocked")
    if not result.is_error:
        assert controller.export_state()["items"][0]["status"] == "awaiting_customer"
        assert controller.export_state()["items"][0]["outcome"] == "unresolved"


def test_handoff_restores_and_new_input_reenables_completion_gate():
    controller, ledger = setup_plan(*input_plan())
    assert controller.review_terminal_output("Which delivery region?").allowed
    state = controller.export_state()
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Which delivery region?", raw_data={"cayu_tau3_workflow": state})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == state
    assert not restored.preview_terminal_output("Which region again?").allowed
    ledger.observe_input(UserMessage.text("EU."))
    assert not update(
        restored,
        status="in_progress",
        obligations=[
            {
                "id": "region",
                "status": "resolved",
                "evidence": "Customer supplied EU.",
                "evidence_refs": [
                    {"kind": "customer_message", "reference_id": "customer-message:2"}
                ],
            }
        ],
    ).is_error
    assert not restored.preview_terminal_output("Everything is done.").allowed
    assert '"mode": "standard_readiness_check"' in restored.render_context()


@pytest.mark.parametrize("version", [15, 16])
def test_existing_communication_handoff_epoch_restores_without_legacy_counter(version):
    controller, ledger = setup_plan(input_plan(kind="communication")[0])
    assert controller.review_terminal_output("Which delivery region?").allowed
    state = controller.export_state()
    state["schema_version"] = version
    assert state["customer_response_handoffs"]["request"] == 0
    assert state["customer_response_handoff_sequences"]["request"] == 1
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Which delivery region?", raw_data={"cayu_tau3_workflow": state})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == {**state, "schema_version": 16}
    assert not restored.preview_terminal_output("Which region again?").allowed


@pytest.mark.parametrize("epoch", [True, -1, 2])
def test_outcome_restore_rejects_invalid_epoch(epoch):
    controller, ledger = setup_plan(*input_plan())
    state = controller.export_state()
    state["customer_response_handoff_sequences"]["request"] = epoch
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Invalid snapshot", raw_data={"cayu_tau3_workflow": state})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert not restored.export_state()["items"]


@pytest.mark.parametrize("initial_status", ["pending", "in_progress", "awaiting_customer"])
def test_native_runtime_delivers_question_then_resumes_unfinished_work(tmp_path, initial_status):
    initial = input_plan()
    initial[0]["status"] = initial_status
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": initial}),
            text("Which delivery region is required?"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "request",
                            "status": "in_progress",
                            "obligations": [
                                {
                                    "id": "region",
                                    "status": "resolved",
                                    "evidence": "Customer supplied EU.",
                                    "evidence_refs": [
                                        {
                                            "kind": "customer_message",
                                            "reference_id": "customer-message:2",
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                },
            ),
            text("Everything is done."),
            call("host_action", {"value": 7}, "continue-work"),
        ],
        planning_mode="outcomes",
        native_recall=True,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Arrange delivery and compare options."), state
        )
        assert reply.content == "Which delivery region is required?"
        assert len(provider.requests) == 2
        reply, _ = agent.generate_next_message(UserMessage.text("EU."), state)
        assert reply.tool_calls[0].name == "host_action"
        assert captured["state"].controller.export_state()["items"][1]["status"] == "pending"
    finally:
        state.stop()
