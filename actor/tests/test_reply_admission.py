"""Response admission must not infer intent from words or retry indefinitely."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tau2.data_model.message import UserMessage
from test_runtime_banking import assembly, call, text, work

from cayu_tau3_banking.admission_progress import AdmissionProgress, AdmissionStallError
from cayu_tau3_banking.bridge import TerminalOutputAdmission
from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL
from cayu_tau3_banking.runtime_agent import RuntimeCallbackExecutionError
from cayu_tau3_banking.runtime_banking import BankingReplyPolicy


def question_plan(planning_mode):
    item = work()
    item.update(
        id="identify-customer",
        goal="Collect the details needed to identify the customer",
        status="awaiting_customer",
        evidence="A searchable account identity has not yet been provided.",
        evidence_refs=[{"kind": "customer_message", "reference_id": "customer-message:1"}],
        next_operation={
            "owner": "customer",
            "kind": "customer_response",
            "tool_name": "",
            "arguments": {},
            "summary": "Ask for full name or account email before accessing customer records.",
        },
    )
    if planning_mode == "outcomes":
        item.pop("next_operation")
        item["obligations"] = [
            {
                "id": "identity-question",
                "goal": "Ask for a searchable profile identifier",
                "kind": "customer_input",
                "status": "ready",
                "evidence": item["evidence"],
                "evidence_refs": item["evidence_refs"],
            }
        ]
    return item


@pytest.mark.parametrize("planning_mode", ["exact", "outcomes"])
@pytest.mark.parametrize(
    "customer_text",
    [
        "I am not sure which ID you are referring to.",
        "You referred to my profile; which identifier do you need?",
        "Which account should I use to maximize my referral bonus?",
    ],
    ids=["unrelated-identity-question", "paraphrased-identity-question", "genuine-referral"],
)
def test_identity_question_reaches_customer_without_referral_veto(
    tmp_path, planning_mode, customer_text
):
    question = "Please provide your full name or account email so I can verify your identity."
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [question_plan(planning_mode)]}),
            text(question),
        ],
        planning_mode=planning_mode,
        native_recall=True,
        callback_timeout_s=60,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text(customer_text), state)
        assert reply.content == question
        assert not reply.tool_calls
        assert len(provider.requests) == 2
        assert not captured["state"].ledger.actions
        assert captured["state"].controller.export_state()["items"][0]["outcome"] == "unresolved"
        assert captured["state"].ledger.customer_messages == [customer_text]
    finally:
        state.stop()


def test_public_referral_information_does_not_require_private_account_access(tmp_path):
    response = (
        "Public program terms do not establish your eligibility; I can check after verification."
    )
    agent, provider, captured = assembly(
        tmp_path,
        [text(response)],
        planning_mode="outcomes",
        advice_mode="lean",
        callback_timeout_s=60,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Which referral program should I consider?"), state
        )
        assert reply.content == response
        assert len(provider.requests) == 1
        assert not captured["state"].ledger.actions
    finally:
        state.stop()


@pytest.mark.parametrize("planning_mode", ["exact", "outcomes"])
def test_rejected_drafts_stop_before_ninth_model_call_and_never_escape(tmp_path, planning_mode):
    agent, provider, captured = assembly(
        tmp_path,
        [text("Undelivered claim that work succeeded.") for _ in range(10)],
        planning_mode=planning_mode,
        persistent=True,
        max_steps=64,
        callback_timeout_s=120,
    )
    state = agent.get_init_state()
    try:
        with pytest.raises(RuntimeCallbackExecutionError) as failure:
            agent.generate_next_message(UserMessage.text("Perform the requested work."), state)
        assert failure.value.failure_type == "AdmissionStallError"
        assert "Undelivered claim" not in str(failure.value)
        assert len(provider.requests) == 8
        assert not captured["state"].ledger.actions
        assert captured["state"].ledger.customer_messages == ["Perform the requested work."]
        assert any(
            "Tool or response admission rejected" in str(request.messages)
            for request in provider.requests
        )
    finally:
        state.stop()


def test_repaired_response_is_delivered_without_resetting_the_denial_allowance(tmp_path):
    question = "Please provide the account email."
    agent, provider, captured = assembly(
        tmp_path,
        [
            text("Unrecorded success."),
            text("Same unrecorded success, rephrased."),
            call(SYNC_WORK_PLAN_TOOL, {"items": [question_plan("outcomes")]}),
            text(question),
        ],
        planning_mode="outcomes",
        callback_timeout_s=60,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Help with my account."), state)
        assert reply.content == question
        assert len(provider.requests) == 4
        assert captured["state"].admission_progress.warning is not None
        assert not captured["state"].ledger.actions
        assert captured["state"].ledger.customer_messages == ["Help with my account."]
    finally:
        state.stop()


@pytest.mark.parametrize("open_remediation", [False, True])
def test_both_reply_denial_paths_share_the_existing_round_budget(open_remediation):
    async def scenario():
        progress = AdmissionProgress()
        memory = SimpleNamespace(
            review_terminal_output=AsyncMock(
                return_value=TerminalOutputAdmission(
                    allowed=False, reason="Required work is unresolved.", correction_code="test"
                )
            )
        )
        state = SimpleNamespace(
            task_memory=None,
            native_recall=False,
            advisory_controller=False,
            remediations=SimpleNamespace(unresolved_count=int(open_remediation)),
            memory=memory,
            advice_mode="controller",
            admission_progress=progress,
        )
        policy = BankingReplyPolicy(state)
        for index in range(8):
            context = SimpleNamespace(
                step_result=SimpleNamespace(model_step_id=f"step-{index}", text_content="Draft")
            )
            # A tool denial or duplicate policy evaluation of the same step is
            # still one denied round, not three.
            progress.observe_denial(f"step-{index}")
            for _ in range(2):
                decision = await policy.before_stop(context)
                assert decision.action.value == "continue"
            if index < 7:
                progress.check()
        with pytest.raises(AdmissionStallError):
            progress.check()
        if open_remediation:
            memory.review_terminal_output.assert_not_awaited()

    asyncio.run(scenario())
