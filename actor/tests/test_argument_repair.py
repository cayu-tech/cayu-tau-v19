import copy

import pytest
from tau2.data_model.message import UserMessage
from test_canonical_constraints import prepared
from test_runtime_banking import assembly, call, text
from test_task_memory import constraint_args

from cayu_tau3_banking.task_memory import UPDATE_CONSTRAINTS


@pytest.mark.parametrize(
    "choice",
    [
        None,
        {},
        {"disclosure_sequence": 0, "response_sequence": 0},
        {"disclosure_sequence": 2, "response_sequence": 0},
    ],
)
def test_empty_choice_preserves_constraint_without_granting_consent(tmp_path, choice):
    memory, row = prepared(tmp_path)
    row["conflict"]["customer_choice"] = choice
    original = copy.deepcopy(row)
    args = constraint_args(memory, constraints=[row])
    args.pop("customer_message_id")
    result = memory.invoke(UPDATE_CONSTRAINTS, args)
    memory.context("next")
    assert result["normalizations"]
    assert row == original
    assert "customer_choice" not in memory.data["constraints"][row["id"]]["conflict"]
    assert memory.pre_action_error("lock_resource", {"resource_id": "A"})


def test_supplied_stale_turn_remains_rejected_with_exact_current_id(tmp_path):
    memory, row = prepared(tmp_path)
    memory.ledger.observe_input(UserMessage.text("I changed my request."))
    args = constraint_args(memory, constraints=[row])
    args["customer_message_id"] = "customer-message:1"
    with pytest.raises(ValueError, match="customer-message:2"):
        memory.invoke(UPDATE_CONSTRAINTS, args)
    assert memory.revision == 0
    assert "latest_customer_message_id=customer-message:2" in memory.context("step")


def test_plan_revision_parameter_is_checked_and_stale_patch_does_not_apply(tmp_path):
    item = {
        "id": "result",
        "goal": "Explain the result",
        "completion_mode": "information",
        "status": "pending",
        "outcome": "unresolved",
    }
    agent, _provider, captured = assembly(
        tmp_path,
        [
            call("controller_sync_work_plan", {"expected_revision": 0, "items": [item]}, "create"),
            call(
                "controller_update_work_plan",
                {
                    "expected_revision": 0,
                    "items": [{"id": "result", "status": "awaiting_customer"}],
                },
                "stale",
            ),
            text("Ready."),
        ],
        controller_mode="advisory",
        planning_mode="outcomes",
        advice_mode="lean",
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Hello."), state)
        assert reply.content == "Ready."
        controller = captured["state"].controller
        assert controller.revision == 1
        assert controller.export_state()["items"][0]["status"] == "pending"
    finally:
        state.stop()
