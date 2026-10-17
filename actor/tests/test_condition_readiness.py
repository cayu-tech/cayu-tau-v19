"""Admission is not completion: fictional resource conditions, no live APIs."""

import json

import pytest
from cayu import ToolCallPart
from tau2.data_model.message import AssistantMessage, UserMessage
from test_case_conditions import (
    POLICY,
    admitted,
    condition,
    execute,
    obligation,
    patch,
    read,
    setup,
)

from cayu_tau3_banking.controller import BankingWorkflowController


def effect_call():
    return ToolCallPart(
        tool_call_id="effect",
        tool_name="call_discoverable_agent_tool",
        arguments={
            "agent_tool_name": "apply_adjustment",
            "arguments": json.dumps({"account_id": "account-a"}),
        },
    )


def test_ready_action_with_current_conditions_can_execute_without_claiming_completion():
    controller, _, created = setup(
        obligations=[obligation(goal="Perform and verify the adjustment", status="ready")]
    )
    assert not created.is_error
    assert admitted(controller)
    saved = controller.export_state()["items"][0]
    assert saved["obligations"][0]["status"] == "ready"
    assert "applied_to" not in saved["obligations"][0]
    assert not controller.preview_terminal_output("All done.").allowed


def test_ready_action_cannot_skip_its_separate_dependency():
    controller, _, created = setup(
        obligations=[
            {"id": "prepare", "goal": "Perform separate preparation", "status": "ready"},
            obligation(status="ready", depends_on=["prepare"]),
        ]
    )
    assert not created.is_error
    blocked = controller.review([effect_call()])[0]
    assert not blocked.allowed
    assert "prepare" in blocked.reason
    assert not patch(
        controller,
        obligations=[
            {
                "id": "prepare",
                "status": "resolved",
                "evidence": "Preparation verified",
                "evidence_refs": [POLICY],
            }
        ],
    ).is_error
    assert admitted(controller)


def test_waiting_action_is_not_implicitly_made_ready_by_matching_predicates():
    controller, _, created = setup(obligations=[obligation(status="waiting")])
    assert not created.is_error
    assert not admitted(controller)
    assert not patch(controller, obligations=[{"id": "eligibility", "status": "ready"}]).is_error
    assert admitted(controller)


@pytest.mark.parametrize("change", ["predicate", "stale", "customer"])
def test_ready_still_requires_current_conditions_and_explains_the_gap(change):
    controller, ledger, created = setup(
        obligations=[
            obligation(
                status="ready", condition=condition(customer_message_id="customer-message:1")
            )
        ]
    )
    assert not created.is_error
    if change == "predicate":
        update = condition(
            checks=[{"field": "status", "operator": "equal", "value": "CLOSED"}],
            customer_message_id="customer-message:1",
        )
        assert not patch(
            controller, obligations=[{"id": "eligibility", "condition": update}]
        ).is_error
        expected = "recorded_condition_not_satisfied"
    elif change == "stale":
        read(ledger, "new-observation", status="CLOSED")
        expected = "resource_observation_stale_or_unavailable"
    else:
        ledger.observe_input(UserMessage.text("Please change the requested adjustment."))
        expected = "customer_context_changed"
    blocked = controller.review([effect_call()])[0]
    assert not blocked.allowed
    assert expected in blocked.reason
    assert admitted(controller, name="get_all_user_accounts_by_user_id_3847")


@pytest.mark.parametrize("error", [False, True])
def test_only_real_success_retains_a_ready_actions_pre_execution_receipt(error):
    controller, ledger, _ = setup(obligations=[obligation(status="ready")])
    call = effect_call()
    assert controller.review([call])[0].allowed
    controller.before_condition_execution(call)
    execute(ledger, call.tool_call_id, call.tool_name, call.arguments, error=error)
    controller.after_condition_execution(call.tool_call_id)
    saved = controller.export_state()
    item = saved["items"][0]["obligations"][0]
    # A successful API call is not proof of the model's entire free-text goal.
    assert item["status"] == "ready"
    assert item.get("applied_to") == (None if error else "effect")
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Continuing", raw_data={"cayu_tau3_workflow": saved})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == saved
    if not error:
        assert not admitted(restored)  # The write invalidated the earlier read.
        assert not patch(
            restored,
            obligations=[
                {
                    "id": "eligibility",
                    "status": "resolved",
                    "evidence": "Actual effect verified",
                    "evidence_refs": [{"kind": "tool_result", "reference_id": "effect"}],
                }
            ],
        ).is_error


@pytest.mark.parametrize("status", ["ready", "resolved"])
def test_readiness_does_not_authorize_a_different_resource(status):
    controller, _, _ = setup(obligations=[obligation(status=status)])
    assert not admitted(controller, account_id="other-account")


@pytest.mark.parametrize("controller_mode", ["enforced", "advisory"])
def test_native_ready_action_after_compaction_waits_for_external_result(tmp_path, controller_mode):
    from cayu import ScriptedModelProvider
    from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
    from tau2.data_model.message import ToolMessage
    from tau2.environment.tool import Tool
    from test_outcome_controller import outcome
    from test_runtime_banking import assembly, call, text

    from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
    from cayu_tau3_banking.ledger import ConversationLedger, ToolActionStatus
    from cayu_tau3_banking.product_navigation import READ_TOOL

    def get_all_user_accounts_by_user_id_3847(user_id: str) -> str:
        """Read account records.

        Args:
            user_id: Account owner.
        """
        raise AssertionError("Only the external host performs account operations")

    def apply_adjustment(account_id: str) -> str:
        """Perform the authorized adjustment.

        Args:
            account_id: Account receiving the adjustment.
        """
        raise AssertionError("Only the external host performs account operations")

    (tmp_path / f"{POLICY['reference_id']}.json").write_text(
        json.dumps(
            {
                "id": POLICY["reference_id"],
                "title": "Fictional adjustment prerequisites",
                "content": "The account must be ACTIVE and opened on or before January 1, 2025.",
            }
        )
    )
    compactor = ScriptedModelProvider([text("Earlier details omitted.") for _ in range(30)])
    agent, _, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}, "plan"),
            call(
                READ_TOOL,
                {"documents": [{"entry_id": POLICY["reference_id"], "revision": 1}]},
                "policy",
            ),
            call("get_all_user_accounts_by_user_id_3847", {"user_id": "person-a"}, "read-1"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "request",
                            "status": "awaiting_customer",
                            "evidence": "Waiting for customer confirmation",
                            "obligations": [
                                obligation(
                                    status="waiting",
                                    goal="Perform and verify the adjustment",
                                    evidence="Awaiting customer confirmation of the requested operation",
                                )
                            ],
                        }
                    ]
                },
                "await-confirmation",
            ),
            text("Please confirm the adjustment."),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "request",
                            "status": "pending",
                            "obligations": [
                                obligation(status="ready", goal="Perform and verify the adjustment")
                            ],
                        }
                    ]
                },
                "ready",
            ),
            call("apply_adjustment", {"account_id": "account-a"}, "effect"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "request",
                            "status": "completed",
                            "outcome": "succeeded",
                            "evidence": "The actual external result confirms success",
                            "evidence_refs": [{"kind": "tool_result", "reference_id": "effect"}],
                            "obligations": [
                                {
                                    "id": "eligibility",
                                    "status": "resolved",
                                    "evidence": "Effect observed",
                                    "evidence_refs": [
                                        {"kind": "tool_result", "reference_id": "effect"}
                                    ],
                                }
                            ],
                        }
                    ]
                },
                "done",
            ),
            text("The adjustment succeeded."),
        ],
        planning_mode="outcomes",
        controller_mode=controller_mode,
        advice_mode="lean",
        persistent=True,
        compaction=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=compactor, model="scripted-summary"),
            compact_after_messages=2,
            max_user_turns=1,
        ),
        host_tools=[Tool(get_all_user_accounts_by_user_id_3847), Tool(apply_adjustment)],
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Please make the adjustment."), state
        )
        assert reply.tool_calls[0].name == "get_all_user_accounts_by_user_id_3847"
        reply, _ = agent.generate_next_message(
            ToolMessage(
                role="tool",
                id=reply.tool_calls[0].id,
                content=read(ConversationLedger()),
            ),
            state,
        )
        assert reply.content == "Please confirm the adjustment."
        reply, _ = agent.generate_next_message(UserMessage.text("Yes, make the adjustment."), state)
        assert [c.name for c in reply.tool_calls] == ["apply_adjustment"]
        external_call_id = reply.tool_calls[0].id
        bank = captured["state"]
        before = bank.controller.export_state()["items"][0]
        assert before["status"] != "completed"
        assert before["obligations"][0]["status"] == "ready"
        assert "applied_to" not in before["obligations"][0]
        assert bank.ledger.tool_action_status("effect") is not ToolActionStatus.SUCCEEDED
        reply, _ = agent.generate_next_message(
            ToolMessage(role="tool", id=external_call_id, content="Adjustment completed."), state
        )
        assert reply.content == "The adjustment succeeded."
        finished = bank.controller.export_state()["items"][0]
        assert finished["status"] == "completed"
        assert finished["obligations"][0]["applied_to"] == "effect"
        assert bank.ledger.tool_action_status("effect") is ToolActionStatus.SUCCEEDED
        assert compactor.requests
    finally:
        agent.stop(state=state)
