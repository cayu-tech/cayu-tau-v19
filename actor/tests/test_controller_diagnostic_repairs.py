"""Fictional regressions for receipt scope and non-blocking bookkeeping."""

import json

import pytest
from cayu import ToolCallPart
from tau2.data_model.message import AssistantMessage
from test_case_conditions import condition, execute, obligation, patch, read, setup
from test_outcome_controller import setup_plan

from cayu_tau3_banking.case_conditions import parse_condition
from cayu_tau3_banking.controller import (
    UPDATE_WORK_PLAN_TOOL,
    BankingWorkflowController,
    _condition_matches_call,
)
from cayu_tau3_banking.ledger import ConversationLedger

PAYMENT = "pay_credit_card_from_checking_9182"


def payment(**changes):
    args = {
        "user_id": "person-a",
        "checking_account_id": "account-a",
        "credit_card_account_id": "credit-a",
        "amount": 25,
        **changes,
    }
    return ToolCallPart(
        tool_call_id="payment",
        tool_name="call_discoverable_agent_tool",
        arguments={"agent_tool_name": PAYMENT, "arguments": json.dumps(args)},
    )


@pytest.mark.parametrize("failed", [False, True])
def test_multi_resource_payment_receipts_and_restore(failed):
    bank = condition(for_tools=[PAYMENT])
    controller, ledger, _ = setup(obligations=[obligation(condition=bank)])
    execute(
        ledger,
        "credit-read",
        "call_discoverable_agent_tool",
        {
            "agent_tool_name": "get_all_user_accounts_by_user_id_3847",
            "arguments": json.dumps({"user_id": "person-a"}),
        },
        "Bank Accounts:\nFound 1 record(s) in 'accounts':\n\n"
        "1. Record ID: account-a\n   account_id: account-a\n   user_id: person-a\n"
        "   class: Checking\n   level: Standard\n   status: ACTIVE\n"
        "   current_holdings: $100\n   date_opened: 01/01/2020\n\nCredit Card Accounts:\n"
        "Found 1 record(s) in 'credit_card_accounts':\n\n"
        "1. Record ID: credit-a\n   account_id: credit-a\n   user_id: person-a\n"
        "   card_type: Fictional Card\n   account_status: ACTIVE\n",
    )
    credit = condition(
        resource_kind="credit_card",
        resource_id="credit-a",
        observed_in="credit-read",
        for_tools=[PAYMENT],
        checks=[{"field": "status", "operator": "equal", "value": "ACTIVE"}],
    )
    assert not patch(
        controller,
        obligations=[
            {
                "id": "eligibility",
                "condition": condition(for_tools=[PAYMENT], observed_in="credit-read"),
            },
            obligation(id="credit", condition=credit),
        ],
    ).is_error
    call = payment()
    assert controller.review([call])[0].allowed
    controller.before_condition_execution(call)
    execute(ledger, call.tool_call_id, call.tool_name, call.arguments, error=failed)
    controller.after_condition_execution(call.tool_call_id)
    saved = controller.export_state()
    for item in saved["items"][0]["obligations"]:
        assert item.get("applied_to") == (None if failed else "payment")
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Observed", raw_data={"cayu_tau3_workflow": saved})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == saved
    if not failed:
        assert not restored.review([payment()])[0].allowed  # Receipt is not new permission.
        assert not patch(
            restored,
            obligations=[
                {"id": "eligibility", "status": "resolved"},
                {"id": "credit", "status": "resolved"},
            ],
        ).is_error


@pytest.mark.parametrize(
    "changes",
    [
        {"checking_account_id": "other", "credit_card_account_id": "account-a"},
        {"checking_account_id": None},
        {"checking_account_id": []},
        {"user_id": "other-person"},
    ],
)
def test_payment_cannot_bind_wrong_role_malformed_scope_or_owner(changes):
    bank = condition(for_tools=[PAYMENT])
    controller, ledger, _ = setup(obligations=[obligation(condition=bank)])
    call = payment(**changes)
    assert not _condition_matches_call(parse_condition(bank), call, ledger)
    assert not controller.review([call])[0].allowed
    controller.before_condition_execution(call)
    assert not controller._pending_condition_checks


def test_explicit_checking_mutation_invalidates_read_without_user_id():
    bank = condition(for_tools=[PAYMENT])
    _, ledger, _ = setup(obligations=[obligation(condition=bank)])
    call = payment()
    args = json.loads(call.arguments["arguments"])
    del args["user_id"]
    execute(
        ledger,
        "changed",
        call.tool_name,
        {"agent_tool_name": PAYMENT, "arguments": json.dumps(args)},
    )
    assert parse_condition(bank).gap(ledger) == "resource_observation_stale_or_unavailable"


def test_model_update_warning_preserves_plan_deduplicates_and_restores():
    controller, ledger = setup_plan()
    before = controller.export_state()
    update = {"items": [{"id": "request", "goal": "Rewrite the original goal"}]}
    result = controller.invoke_model_update(UPDATE_WORK_PLAN_TOOL, update)
    assert not result.is_error and result.remediation is None
    assert json.loads(result.content)["advisory_code"] == "plan_update_advisory"
    assert json.loads(result.content)["status"] == "not_applied"
    assert not controller._pending_remediations
    saved = controller.export_state()
    assert saved["items"] == before["items"] and saved["revision"] == before["revision"]
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Plan preserved", raw_data={"cayu_tau3_workflow": saved})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    repeated = restored.invoke_model_update(UPDATE_WORK_PLAN_TOOL, update)
    assert json.loads(repeated.content)["advisory_code"] == "plan_update_advisory_repeated"
    assert "Plan rejected" not in repeated.content
    assert restored.export_state() == saved
    independent = ToolCallPart(
        tool_call_id="read", tool_name="read_authorized_record", arguments={}
    )
    assert restored.review([independent])[0].allowed


def test_advisory_does_not_allow_a_stale_action_or_claim_completion():
    controller, ledger, _ = setup()
    read(ledger, "changed", status="CLOSED")
    controller.export_state()  # Reconcile the now-stale recorded assessment.
    result = controller.invoke_model_update(
        UPDATE_WORK_PLAN_TOOL,
        {
            "items": [
                {"id": "request", "obligations": [{"id": "eligibility", "status": "resolved"}]}
            ]
        },
    )
    assert (
        not result.is_error
        and json.loads(result.content)["advisory_code"] == "plan_update_advisory"
    )
    call = ToolCallPart(
        tool_call_id="new", tool_name="apply_adjustment", arguments={"account_id": "account-a"}
    )
    assert not controller.review([call])[0].allowed
    assert controller.export_state()["items"][0]["obligations"][0]["status"] == "ready"


def test_empty_checklist_never_claims_task_completion():
    controller = BankingWorkflowController(ConversationLedger(), planning_mode="outcomes")
    context = json.loads(
        controller.render_context().split("<workflow-plan>\n")[1].split("\n</workflow-plan>")[0]
    )
    contract = context["response_contract"]
    assert contract["task_complete"] is None
    assert not contract["plan_initialized"] and not contract["tracked_items_complete"]
