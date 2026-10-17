"""Confirmation semantics belong to the actor, not a substring classifier."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from tau2.data_model.message import UserMessage
from test_admission import confirm_effect, proposed

from cayu_tau3_banking.admission import BankingToolAdmissionGate
from cayu_tau3_banking.advice import lean_banking_prompt
from cayu_tau3_banking.agent import banking_system_prompt
from cayu_tau3_banking.cli import _parser, _run
from cayu_tau3_banking.controller_contract import advisory_banking_prompt
from cayu_tau3_banking.ledger import ConversationLedger
from cayu_tau3_banking.runtime_banking import BankingRuntimeToolPolicy
from cayu_tau3_banking.verification_guidance import VERIFICATION_GUIDANCE


def observed_identity(customer_text):
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text(customer_text))
    row = dict(
        user_id="user-7",
        name="Example",
        address="1 Main Street",
        email="example@example.com",
        phone_number="5550100",
        date_of_birth="01/02/1990",
    )
    confirm_effect(
        ledger, "lookup", "get_user_information_by_id", result=json.dumps(row), user_id="user-7"
    )
    confirm_effect(ledger, "time", "get_current_time", result="2025-11-14 03:40:00 EST")
    call = proposed("verify", "log_verification", **row, time_verified="2025-11-14 03:40:00 EST")
    return ledger, call


@pytest.mark.parametrize(
    "message",
    [
        "Yes, both details you just read are correct.",
        "My birthday is January second, nineteen ninety, and that is my current address.",
        "That is NOT my email example@example.com or my phone 5550100.",
        "For example, someone might give example@example.com and 5550100.",
        "I am unsure whether those details are correct.",
    ],
)
def test_gate_does_not_classify_customer_language(message):
    ledger, call = observed_identity(message)
    assert BankingToolAdmissionGate(ledger).review((call,))[0].allowed
    # Even for denials the gate does not judge semantics; proposing the call would
    # be an actor error. Admission alone must not manufacture a verified user.
    assert ledger.verification_logs_by_user == set()
    assert (
        not BankingToolAdmissionGate(ledger)
        .review(
            (
                proposed(
                    "protected", "change_user_email", user_id="user-7", new_email="new@example.com"
                ),
            )
        )[0]
        .allowed
    )


def test_record_mismatch_is_still_rejected_with_clear_confirmation():
    ledger, call = observed_identity("Yes, both details are correct.")
    wrong = proposed("wrong", "log_verification", **{**call.arguments, "user_id": "user-70"})
    assert not BankingToolAdmissionGate(ledger).review((wrong,))[0].allowed


@pytest.mark.parametrize(
    "prompt", [banking_system_prompt, lean_banking_prompt, advisory_banking_prompt]
)
def test_all_actor_modes_receive_semantic_confirmation_responsibility(prompt):
    assert VERIFICATION_GUIDANCE in prompt("Public policy")


def test_policy_audit_does_not_claim_that_semantics_were_validated(monkeypatch):
    from cayu_tau3_banking import runtime_banking

    ledger, call = observed_identity("Yes, both details are correct.")
    gate = BankingToolAdmissionGate(ledger)

    class AsyncGate:
        async def review(self, calls):
            return gate.review(calls)

    async def no_recall(state):
        pass

    monkeypatch.setattr(runtime_banking, "observe_delivered_recall", no_recall)
    state = SimpleNamespace(
        task_memory=None,
        host_names={"log_verification"},
        round_host_calls={},
        batch_gate=AsyncGate(),
        memory_gate=AsyncGate(),
        advisory_controller=True,
        ledger=ledger,
    )
    result = asyncio.run(
        BankingRuntimeToolPolicy(state).authorize(
            SimpleNamespace(
                tool_name=call.tool_name,
                tool_call_id=call.tool_call_id,
                arguments=call.arguments,
            )
        )
    )
    assert result.metadata["customer_confirmation_owner"] == "acting_model"
    assert result.metadata["customer_confirmation_semantically_validated"] is False
    assert result.metadata["latest_customer_message_id"] == ledger.customer_message_ids[-1]
    assert "example@example.com" not in json.dumps(result.metadata)


def test_lexical_navigation_is_explicit_and_rejects_bridge_before_dispatch():
    assert not _parser().parse_args([]).lexical_navigation
    args = _parser().parse_args(["--lexical-navigation", "--runtime", "bridge"])
    with pytest.raises(SystemExit, match="requires native"):
        _run(args)
