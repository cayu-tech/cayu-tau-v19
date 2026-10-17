from __future__ import annotations

import pytest
from cayu import ToolCallPart
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from cayu_tau3_banking.admission import BankingToolAdmissionGate
from cayu_tau3_banking.ledger import ConversationLedger
from cayu_tau3_banking.transfer_reason import customer_requests_human_assistance


def proposed(call_id: str, tool_name: str, **arguments: object) -> ToolCallPart:
    return ToolCallPart(
        tool_call_id=call_id,
        tool_name=tool_name,
        arguments=arguments,
    )


def confirm_effect(
    ledger: ConversationLedger,
    call_id: str,
    tool_name: str,
    *,
    result: str = "Success.",
    **arguments: object,
) -> None:
    ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id=call_id,
                    name=tool_name,
                    arguments=arguments,
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
            content=result,
        )
    )


def confirm_verification(ledger: ConversationLedger, *, call_id: str = "verify-1") -> None:
    confirm_effect(
        ledger,
        call_id,
        "log_verification",
        user_id="user-7",
        name="Example User",
        address="1 Main Street",
        email="example@example.com",
        phone_number="555-0100",
        date_of_birth="01/02/1990",
        time_verified="2025-11-14 03:40:00 EST",
    )


def test_discoverable_agent_tool_requires_a_confirmed_unlock() -> None:
    ledger = ConversationLedger()
    gate = BankingToolAdmissionGate(ledger)
    call = proposed(
        "call-1",
        "call_discoverable_agent_tool",
        agent_tool_name="specialized_action_1234",
        arguments="{}",
    )

    denied = gate.review((call,))[0]
    confirm_effect(
        ledger,
        "unlock-1",
        "unlock_discoverable_agent_tool",
        agent_tool_name="specialized_action_1234",
    )
    allowed = gate.review((call,))[0]

    assert denied.allowed is False
    assert "not been successfully unlocked" in denied.reason
    assert denied.diagnostic_code == "banking_discoverable_tool_locked"
    assert allowed.allowed is True


def test_confirmed_side_effect_cannot_be_repeated_with_exact_arguments() -> None:
    ledger = ConversationLedger()
    confirm_verification(ledger)
    confirm_effect(
        ledger,
        "email-1",
        "change_user_email",
        user_id="user-7",
        new_email="new@example.com",
    )
    gate = BankingToolAdmissionGate(ledger)

    repeated, changed = gate.review(
        (
            proposed(
                "email-2",
                "change_user_email",
                new_email="new@example.com",
                user_id="user-7",
            ),
            proposed(
                "email-3",
                "change_user_email",
                new_email="different@example.com",
                user_id="user-7",
            ),
        )
    )

    assert repeated.allowed is False
    assert "already confirmed successful" in repeated.reason
    assert changed.allowed is True


def test_failed_side_effect_can_be_retried() -> None:
    ledger = ConversationLedger()
    confirm_verification(ledger)
    ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="email-1",
                    name="change_user_email",
                    arguments={"user_id": "user-7", "new_email": "new@example.com"},
                    requestor="assistant",
                )
            ],
        )
    )
    ledger.observe_input(
        ToolMessage(
            id="email-1",
            role="tool",
            requestor="assistant",
            content="Error: update rejected",
        )
    )

    decision = BankingToolAdmissionGate(ledger).review(
        (
            proposed(
                "email-2",
                "change_user_email",
                user_id="user-7",
                new_email="new@example.com",
            ),
        )
    )[0]

    assert decision.allowed is True


def test_same_nonrepeatable_effect_is_denied_within_one_batch() -> None:
    gate = BankingToolAdmissionGate(ConversationLedger())
    gate.observe_knowledge_text("Use customer_action_1234 for this workflow.")
    first, second = gate.review(
        (
            proposed(
                "give-1",
                "give_discoverable_user_tool",
                discoverable_tool_name="customer_action_1234",
            ),
            proposed(
                "give-2",
                "give_discoverable_user_tool",
                discoverable_tool_name="customer_action_1234",
            ),
        )
    )

    assert first.allowed is True
    assert second.allowed is False
    assert "more than once in this batch" in second.reason
    assert gate.export_state() == {
        "denied_calls": 1,
        "denials_by_reason": {
            "the same non-repeatable side effect appears more than once in this batch": 1
        },
        "knowledge_identifier_count": 1,
        "knowledge_search_dialogue_sequences": [],
        "public_tool_identifier_count": 0,
        "public_tool_identifiers": [],
        "recovery_bypass_code_count": 0,
        "requires_knowledge_evidence": True,
        "reviewed_calls": 2,
        "schema_version": 3,
    }


def test_discoverable_name_must_come_from_delivered_cayu_knowledge() -> None:
    gate = BankingToolAdmissionGate(ConversationLedger())
    unlock = proposed(
        "unlock-1",
        "unlock_discoverable_agent_tool",
        agent_tool_name="order_replacement_card_7291",
    )

    before = gate.review((unlock,))[0]
    gate.observe_knowledge_text(
        "The public procedure says to unlock order_replacement_card_7291 before use."
    )
    after = gate.review((unlock,))[0]

    assert before.allowed is False
    assert "Cayu-retrieved public knowledge" in before.reason
    assert after.allowed is True


def test_public_capability_memory_keeps_only_bounded_tool_shaped_identifiers() -> None:
    gate = BankingToolAdmissionGate(ConversationLedger())
    gate.observe_knowledge_text(
        "Use get_bank_account_transactions_9173 for account chk_example_1234. "
        "Then call apply_savings_account_credit_6831 when policy permits it."
    )

    assert gate.public_tool_identifiers() == (
        "apply_savings_account_credit_6831",
        "get_bank_account_transactions_9173",
    )
    context = gate.render_public_capability_context()
    assert context is not None
    assert "get_bank_account_transactions_9173" in context
    assert "chk_example_1234" not in context
    assert "does not discover new tools" in context
    assert gate.knowledge_identifier_is_exposed("get_bank_account_transactions_9173")
    assert not gate.knowledge_identifier_is_exposed("get_card_last_4_digits")


def test_identical_discoverable_read_is_denied_only_in_same_customer_turn() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Review my accounts."))
    confirm_effect(
        ledger,
        "unlock-read",
        "unlock_discoverable_agent_tool",
        agent_tool_name="get_all_user_accounts_by_user_id_3847",
    )
    arguments = {
        "agent_tool_name": "get_all_user_accounts_by_user_id_3847",
        "arguments": '{"user_id":"user-7"}',
    }
    confirm_effect(
        ledger,
        "read-1",
        "call_discoverable_agent_tool",
        agent_tool_name="get_all_user_accounts_by_user_id_3847",
        arguments='{ "user_id": "user-7" }',
    )
    gate = BankingToolAdmissionGate(ledger)

    repeated = gate.review((proposed("read-2", "call_discoverable_agent_tool", **arguments),))[0]
    ledger.observe_input(UserMessage.text("Please refresh that account list."))
    refreshed = gate.review((proposed("read-3", "call_discoverable_agent_tool", **arguments),))[0]

    assert repeated.allowed is False
    assert "reuse the observation ledger" in repeated.reason
    assert refreshed.allowed is True


def test_later_discoverable_mutation_allows_same_turn_read_refresh() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Open an account, then refresh my accounts."))
    confirm_effect(
        ledger,
        "unlock-read",
        "unlock_discoverable_agent_tool",
        agent_tool_name="get_all_user_accounts_by_user_id_3847",
    )
    read_arguments = {
        "agent_tool_name": "get_all_user_accounts_by_user_id_3847",
        "arguments": '{"user_id":"user-7"}',
    }
    confirm_effect(ledger, "read-1", "call_discoverable_agent_tool", **read_arguments)
    confirm_effect(
        ledger,
        "write-1",
        "call_discoverable_agent_tool",
        agent_tool_name="open_bank_account_4821",
        arguments='{"user_id":"user-7","account_type":"checking"}',
    )
    gate = BankingToolAdmissionGate(ledger)

    decision = gate.review((proposed("read-2", "call_discoverable_agent_tool", **read_arguments),))[
        0
    ]

    assert decision.allowed is True


def test_discoverable_history_polling_and_same_batch_reads_are_denied() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Find my account activity."))
    confirm_effect(ledger, "list-1", "list_discoverable_agent_tools")
    confirm_effect(
        ledger,
        "unlock-read",
        "unlock_discoverable_agent_tool",
        agent_tool_name="get_bank_account_transactions_9173",
    )
    gate = BankingToolAdmissionGate(ledger)
    read = proposed(
        "read-1",
        "call_discoverable_agent_tool",
        agent_tool_name="get_bank_account_transactions_9173",
        arguments='{"account_id":"checking-1"}',
    )

    repeated_list = gate.review((proposed("list-2", "list_discoverable_agent_tools"),))[0]
    first, repeated_read = gate.review(
        (
            read,
            proposed(
                "read-2",
                "call_discoverable_agent_tool",
                agent_tool_name="get_bank_account_transactions_9173",
                arguments='{"account_id":"checking-1"}',
            ),
        )
    )

    assert repeated_list.allowed is False
    assert first.allowed is True
    assert repeated_read.allowed is False
    assert "same exact read" in repeated_read.reason


def test_distinct_customer_side_actions_must_be_delivered_sequentially() -> None:
    gate = BankingToolAdmissionGate(ConversationLedger())
    gate.observe_knowledge_text("Use customer_action_1234 for each selected record.")

    first, second = gate.review(
        (
            proposed(
                "give-1",
                "give_discoverable_user_tool",
                discoverable_tool_name="customer_action_1234",
                arguments='{"record_id":"one"}',
            ),
            proposed(
                "give-2",
                "give_discoverable_user_tool",
                discoverable_tool_name="customer_action_1234",
                arguments='{"record_id":"two"}',
            ),
        )
    )

    assert first.allowed is True
    assert second.allowed is False
    assert "only one customer-side action" in second.reason


def test_successfully_given_customer_tool_is_reused_without_regiving() -> None:
    ledger = ConversationLedger()
    confirm_effect(
        ledger,
        "give-1",
        "give_discoverable_user_tool",
        discoverable_tool_name="customer_action_1234",
        arguments='{"record_id":"one"}',
        result="Tool given to user: customer_action_1234",
    )
    gate = BankingToolAdmissionGate(ledger)
    gate.observe_knowledge_text("Use customer_action_1234 for each selected record.")

    decision = gate.review(
        (
            proposed(
                "give-2",
                "give_discoverable_user_tool",
                discoverable_tool_name="customer_action_1234",
                arguments='{"record_id":"two"}',
            ),
        )
    )[0]

    assert decision.allowed is False
    assert "already given successfully" in decision.reason
    assert "reuse the available tool name" in decision.reason


@pytest.mark.parametrize(
    "tool_name",
    [
        "get_credit_card_accounts_by_user",
        "get_credit_card_transactions_by_user",
        "get_referrals_by_user",
    ],
)
def test_customer_specific_base_tool_requires_verification_for_same_user(tool_name: str) -> None:
    ledger = ConversationLedger()
    gate = BankingToolAdmissionGate(ledger)
    call = proposed("cards-1", tool_name, user_id="user-7")

    denied = gate.review((call,))[0]
    confirm_verification(ledger)
    wrong_user, verified_user = gate.review(
        (
            proposed("cards-2", tool_name, user_id="user-8"),
            call,
        )
    )

    assert denied.allowed is False
    assert wrong_user.allowed is False
    assert verified_user.allowed is True


@pytest.mark.parametrize("prefix", ["", "Unrelated context. " * 100], ids=["short", "long"])
def test_verification_requires_public_identity_and_time_evidence(prefix: str) -> None:
    ledger = ConversationLedger()
    gate = BankingToolAdmissionGate(ledger)
    verification = proposed(
        "verify-1",
        "log_verification",
        user_id="user-7",
        name="Example User",
        address="1 Main Street",
        email="example@example.com",
        phone_number="+1 (555) 0100",
        date_of_birth="01/02/1990",
        time_verified="2025-11-14 03:40:00 EST",
    )

    missing_time = gate.review((verification,))[0]
    confirm_effect(
        ledger,
        "time-1",
        "get_current_time",
        result="The current time is 2025-11-14 03:40:00 EST.",
    )
    missing_identity = gate.review((verification,))[0]
    ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="lookup-1",
                    name="get_user_information_by_id",
                    arguments={"user_id": "user-7"},
                    requestor="assistant",
                )
            ],
        )
    )
    ledger.observe_input(
        ToolMessage(
            id="lookup-1",
            role="tool",
            requestor="assistant",
            content=(
                '{"user_id":"user-7","name":"Example User",'
                '"address":"1 Main Street","email":"example@example.com",'
                '"phone_number":"+1 (555) 0100","date_of_birth":"01/02/1990"}'
            ),
        )
    )
    ledger.observe_input(UserMessage.text(prefix + "My email is example@example.com."))
    one_customer_value = gate.review((verification,))[0]
    ledger.observe_input(UserMessage.text(prefix + "My phone number is +1 555 0100."))
    admitted = gate.review((verification,))[0]

    assert missing_time.allowed is False
    assert "get_current_time" in missing_time.reason
    assert missing_identity.allowed is False
    # The gate validates records/time; it no longer pretends to judge whether
    # customer language confirms two fields. That judgment belongs to the actor.
    assert one_customer_value.allowed is True
    assert admitted.allowed is True


def test_verification_cannot_be_logged_twice_in_one_conversation() -> None:
    ledger = ConversationLedger()
    confirm_verification(ledger)

    decision = BankingToolAdmissionGate(ledger).review(
        (
            proposed(
                "verify-2",
                "log_verification",
                user_id="user-8",
                name="Another User",
                address="2 Main Street",
                email="another@example.com",
                phone_number="555-0200",
                date_of_birth="02/03/1991",
                time_verified="2025-11-14 03:41:00 EST",
            ),
        )
    )[0]

    assert decision.allowed is False
    assert "already logged successfully" in decision.reason


def test_recovery_bypass_prevents_a_second_standard_verification() -> None:
    ledger = ConversationLedger()
    gate = BankingToolAdmissionGate(ledger)
    recovery_code = "RECOVERY9CODE7"
    bypass_arguments = {
        argument: recovery_code
        for argument in (
            "name",
            "user_id",
            "address",
            "email",
            "phone_number",
            "date_of_birth",
        )
    }
    bypass_arguments["time_verified"] = "2025-11-14 03:40:00 EST"
    confirm_effect(ledger, "bypass-1", "log_verification", **bypass_arguments)

    decision = gate.review(
        (
            proposed(
                "verify-2",
                "log_verification",
                user_id="user-7",
                name="Example User",
                address="1 Main Street",
                email="example@example.com",
                phone_number="555-0100",
                date_of_birth="01/02/1990",
                time_verified="2025-11-14 03:41:00 EST",
            ),
        )
    )[0]

    assert decision.allowed is False
    assert "already logged successfully" in decision.reason


def test_verification_cannot_combine_fields_from_different_lookup_records() -> None:
    ledger = ConversationLedger()
    confirm_effect(
        ledger,
        "time-1",
        "get_current_time",
        result="The current time is 2025-11-14 03:40:00 EST.",
    )
    confirm_effect(
        ledger,
        "lookup-1",
        "get_user_information_by_name",
        customer_name="Example User",
        result=(
            "Found 2 record(s) in 'users':\n\n"
            "1. Record ID: user-7\n"
            "   user_id: user-7\n"
            "   address: 1 Main Street\n"
            "   email: first@example.com\n\n"
            "2. Record ID: user-8\n"
            "   phone_number: 555-0100\n"
            "   date_of_birth: 01/02/1990\n"
        ),
    )
    ledger.observe_input(
        UserMessage.text("My email is first@example.com and my phone is 555-0100.")
    )

    decision = BankingToolAdmissionGate(ledger).review(
        (
            proposed(
                "verify-1",
                "log_verification",
                user_id="user-7",
                name="Example User",
                address="1 Main Street",
                email="first@example.com",
                phone_number="555-0100",
                date_of_birth="01/02/1990",
                time_verified="2025-11-14 03:40:00 EST",
            ),
        )
    )[0]

    assert decision.allowed is False
    assert "matching identity lookup" in decision.reason


def test_documented_account_recovery_bypass_can_verify_a_resolved_user() -> None:
    ledger = ConversationLedger()
    gate = BankingToolAdmissionGate(ledger)
    recovery_code = "RECOVERY9CODE7"
    gate.observe_knowledge_text(
        "Account Recovery Bypass Code instructions: use log_verification and put "
        f"{recovery_code} in every identity field."
    )
    confirm_effect(
        ledger,
        "time-1",
        "get_current_time",
        result="The current time is 2025-11-14 03:40:00 EST.",
    )
    confirm_effect(
        ledger,
        "lookup-1",
        "get_user_information_by_name",
        customer_name="Example User",
        result=(
            "Found 1 record(s) in 'users':\n\n"
            "1. Record ID: user-7\n"
            "   user_id: user-7\n"
            "   name: Example User"
        ),
    )
    ledger.observe_input(UserMessage.text(f"My recovery code is {recovery_code}."))
    bypass_arguments = {
        argument: recovery_code
        for argument in (
            "name",
            "user_id",
            "address",
            "email",
            "phone_number",
            "date_of_birth",
        )
    }
    bypass_arguments["time_verified"] = "2025-11-14 03:40:00 EST"

    bypass = proposed("bypass-1", "log_verification", **bypass_arguments)
    assert gate.review((bypass,))[0].allowed is True
    confirm_effect(ledger, "bypass-1", "log_verification", **bypass_arguments)

    resolved_user, unrelated_user = gate.review(
        (
            proposed(
                "email-1",
                "change_user_email",
                user_id="user-7",
                new_email="new@example.com",
            ),
            proposed(
                "email-2",
                "change_user_email",
                user_id="user-8",
                new_email="other@example.com",
            ),
        )
    )

    assert resolved_user.allowed is True
    assert unrelated_user.allowed is False


def test_account_recovery_bypass_requires_public_code_provenance() -> None:
    ledger = ConversationLedger()
    gate = BankingToolAdmissionGate(ledger)
    recovery_code = "RECOVERY9CODE7"
    confirm_effect(
        ledger,
        "time-1",
        "get_current_time",
        result="The current time is 2025-11-14 03:40:00 EST.",
    )
    bypass_arguments = {
        argument: recovery_code
        for argument in (
            "name",
            "user_id",
            "address",
            "email",
            "phone_number",
            "date_of_birth",
        )
    }
    bypass_arguments["time_verified"] = "2025-11-14 03:40:00 EST"
    bypass = proposed("bypass-1", "log_verification", **bypass_arguments)

    not_retrieved = gate.review((bypass,))[0]
    gate.observe_knowledge_text(
        f"Account Recovery Bypass Code instructions: call log_verification with {recovery_code}."
    )
    not_customer_supplied = gate.review((bypass,))[0]
    ledger.observe_input(UserMessage.text(f"The code is {recovery_code}."))
    allowed = gate.review((bypass,))[0]

    assert not_retrieved.allowed is False
    assert "Cayu-retrieved recovery code" in not_retrieved.reason
    assert not_customer_supplied.allowed is False
    assert allowed.allowed is True


def test_account_recovery_bypass_requires_uniform_documented_fields() -> None:
    ledger = ConversationLedger()
    gate = BankingToolAdmissionGate(ledger)
    recovery_code = "RECOVERY9CODE7"
    gate.observe_knowledge_text(
        "Account Recovery Bypass Code instructions: call log_verification with "
        f"{recovery_code} in every field."
    )
    confirm_effect(
        ledger,
        "time-1",
        "get_current_time",
        result="The current time is 2025-11-14 03:40:00 EST.",
    )
    ledger.observe_input(UserMessage.text(f"The code is {recovery_code}."))

    decision = gate.review(
        (
            proposed(
                "bypass-1",
                "log_verification",
                name=recovery_code,
                user_id=recovery_code,
                address=recovery_code,
                email="different@example.com",
                phone_number=recovery_code,
                date_of_birth=recovery_code,
                time_verified="2025-11-14 03:40:00 EST",
            ),
        )
    )[0]

    assert decision.allowed is False


def test_transfer_requires_an_exact_public_reason_code() -> None:
    decision = BankingToolAdmissionGate(ConversationLedger()).review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="Customer requested a specialist.",
                reason="made_up_reason",
            ),
        )
    )[0]

    assert decision.allowed is False
    assert "exact public transfer reason code" in decision.reason


def test_transfer_reason_must_come_from_delivered_cayu_knowledge() -> None:
    gate = BankingToolAdmissionGate(ConversationLedger())
    transfer = proposed(
        "transfer-1",
        "transfer_to_human_agents",
        summary="Customer requested a specialist.",
        reason="specialized_department_required",
    )

    before = gate.review((transfer,))[0]
    gate.observe_knowledge_text(
        "Use specialized_department_required when a request needs a specialist."
    )
    after = gate.review((transfer,))[0]

    assert before.allowed is False
    assert "Cayu-retrieved public knowledge" in before.reason
    assert after.allowed is True


def test_external_offer_uses_highest_supported_transfer_reason() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(
        UserMessage.text("I received a letter promising a special promotional offer.")
    )
    ledger.observe_assistant(
        AssistantMessage.text("That promotional offer does not exist in our system.")
    )
    ledger.observe_input(UserMessage.text("But the letter has my name on it."))
    ledger.observe_assistant(AssistantMessage.text("I cannot honor that offer."))
    ledger.observe_input(UserMessage.text("Transfer me to a human representative now."))

    gate = BankingToolAdmissionGate(ledger, require_knowledge_evidence=False)
    unavailable_refusal, external_communication = gate.review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="Customer requests review of an unverified mailed offer.",
                reason="customer_demands_after_unavailable_offer_refusal",
            ),
            proposed(
                "transfer-2",
                "transfer_to_human_agents",
                summary="Customer requests review of an unverified mailed offer.",
                reason="unconfirmed_external_communication",
            ),
        )
    )

    assert unavailable_refusal.allowed is True
    assert external_communication.allowed is False
    assert "higher priority" in external_communication.reason


def test_materially_contradicted_offer_overrides_earlier_uncertainty_in_same_response() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(
        UserMessage.text(
            "My flyer says the Business Bronze Rewards Card pays a $1,000 statement credit."
        )
    )
    ledger.observe_assistant(
        AssistantMessage.text(
            "I couldn't confirm that exact offer. The closest documented promotion is for "
            "the Business Bronze Rewards Card, but it provides a $500 statement credit, "
            "not $1,000."
        )
    )
    ledger.observe_input(UserMessage.text("But the flyer in my hand says $1,000."))
    ledger.observe_assistant(
        AssistantMessage.text("I still cannot verify a separate $1,000 promotion.")
    )
    ledger.observe_input(
        UserMessage.text("I still insist. I need to speak to someone who can help me.")
    )

    unavailable_refusal, external_communication = BankingToolAdmissionGate(
        ledger,
        require_knowledge_evidence=False,
    ).review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="Customer persisted after the documented offer contradicted the flyer.",
                reason="customer_demands_after_unavailable_offer_refusal",
            ),
            proposed(
                "transfer-2",
                "transfer_to_human_agents",
                summary="Customer requested review of an unverified flyer.",
                reason="unconfirmed_external_communication",
            ),
        )
    )

    assert unavailable_refusal.allowed is True
    assert external_communication.allowed is False
    assert "higher priority" in external_communication.reason


def test_unconfirmed_external_offer_requires_claim_then_agent_nonconfirmation() -> None:
    ledger = ConversationLedger()
    transfer = proposed(
        "transfer-1",
        "transfer_to_human_agents",
        summary="Customer requests review of an unverified mailed offer.",
        reason="unconfirmed_external_communication",
    )
    gate = BankingToolAdmissionGate(ledger, require_knowledge_evidence=False)

    before = gate.review((transfer,))[0]
    ledger.observe_input(
        UserMessage.text("I received a letter promising a special promotional offer.")
    )
    after_claim = gate.review((transfer,))[0]
    ledger.observe_assistant(
        AssistantMessage.text("I could not find or verify that offer in our public knowledge.")
    )
    after_nonconfirmation = gate.review((transfer,))[0]

    assert before.allowed is False
    assert "observed claim" in before.reason
    assert after_claim.allowed is False
    assert "agent response" in after_claim.reason
    assert after_nonconfirmation.allowed is True


def test_unconfirmed_external_offer_accepts_typographic_apostrophe() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(
        UserMessage.text("I received a letter promising a special promotional offer.")
    )
    ledger.observe_assistant(
        AssistantMessage.text("I couldn\u2019t confirm that promotional offer in our records.")
    )

    decision = BankingToolAdmissionGate(
        ledger,
        require_knowledge_evidence=False,
    ).review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="Customer requests review of an unverified mailed offer.",
                reason="unconfirmed_external_communication",
            ),
        )
    )[0]

    assert decision.allowed is True


def test_cayu_backed_unconfirmed_external_offer_requires_search_before_response() -> None:
    transfer = proposed(
        "transfer-1",
        "transfer_to_human_agents",
        summary="Customer requests review of an unverified mailed offer.",
        reason="unconfirmed_external_communication",
    )
    late_ledger = ConversationLedger()
    late_ledger.observe_input(
        UserMessage.text("I received a letter promising a special promotional offer.")
    )
    late_gate = BankingToolAdmissionGate(late_ledger)
    late_gate.observe_knowledge_text(
        "Use unconfirmed_external_communication for the documented transfer case."
    )
    late_ledger.observe_assistant(
        AssistantMessage.text("I could not find or verify that offer in our public knowledge.")
    )

    without_search = late_gate.review((transfer,))[0]
    late_gate.observe_knowledge_search()
    search_after_response = late_gate.review((transfer,))[0]

    valid_ledger = ConversationLedger()
    valid_ledger.observe_input(
        UserMessage.text("I received a letter promising a special promotional offer.")
    )
    valid_gate = BankingToolAdmissionGate(valid_ledger)
    valid_gate.observe_knowledge_text(
        "Use unconfirmed_external_communication for the documented transfer case."
    )
    valid_gate.observe_knowledge_search()
    valid_ledger.observe_assistant(
        AssistantMessage.text("I could not find or verify that offer in our public knowledge.")
    )
    valid = valid_gate.review((transfer,))[0]

    assert without_search.allowed is False
    assert "successful Cayu knowledge search" in without_search.reason
    assert search_after_response.allowed is False
    assert valid.allowed is True


def test_capability_gap_exception_requires_public_policy_search_and_current_human_request():
    ledger = ConversationLedger()
    gate = BankingToolAdmissionGate(ledger)
    ledger.observe_input(
        UserMessage.text("I received a letter promising a special promotional offer.")
    )
    gate.observe_knowledge_search()
    ledger.observe_assistant(
        AssistantMessage.text("I could not verify that offer in public knowledge.")
    )
    assert not gate.has_verified_offer_capability_gap()
    ledger.observe_input(UserMessage.text("Please transfer me to a human."))
    assert gate.render_offer_transfer_review_context() is None
    assert not gate.has_verified_offer_capability_gap()  # No delivered policy identity.
    gate.observe_knowledge_text("unconfirmed_external_communication")
    assert gate.has_verified_offer_capability_gap()
    before = gate.export_state()
    evidence = gate.render_offer_transfer_review_context()
    assert '"scope": "current_offer_human_review"' in evidence
    assert (
        '"supported_delivered_policy_reason_codes": ["unconfirmed_external_communication"]'
        in evidence
    )
    assert '"separate_banking_outcomes_remain_subject_to_review": true' in evidence
    assert gate.render_offer_transfer_review_context() == evidence
    assert gate.export_state() == before
    ledger.observe_assistant(AssistantMessage.text("The address update is ready to proceed."))
    ledger.observe_input(UserMessage.text("Please transfer me to a human instead."))
    assert not gate.has_verified_offer_capability_gap()  # An old offer gap is not current evidence.
    assert gate.render_offer_transfer_review_context() is None
    ledger.observe_input(UserMessage.text("Do not transfer me. Stop the transfer."))
    assert not gate.has_verified_offer_capability_gap()
    assert gate.render_offer_transfer_review_context() is None


@pytest.mark.parametrize(
    "explanation",
    [
        "The promotional offer cannot be verified. Would you like human help?",
        "There is no documented referral program matching those terms. Would you like human help?",
        "The offer could not be located after searching. Would you like human help?",
    ],
)
def test_offer_nonconfirmation_supports_passive_and_documented_absence(explanation):
    ledger = ConversationLedger()
    gate = BankingToolAdmissionGate(ledger)
    ledger.observe_input(UserMessage.text("I received a letter about a referral program."))
    gate.observe_knowledge_text("unconfirmed_external_communication")
    gate.observe_knowledge_search()
    ledger.observe_assistant(AssistantMessage.text(explanation))
    ledger.observe_input(UserMessage.text("Yes, please transfer me to a human agent."))
    assert gate.has_verified_offer_capability_gap()
    call = ToolCallPart(
        tool_call_id="review-mailing",
        tool_name="transfer_to_human_agents",
        arguments={"reason": "unconfirmed_external_communication", "summary": "Review mailing."},
    )
    assert gate.review([call])[0].allowed
    ledger.observe_input(UserMessage.text("Do not transfer me."))
    assert not gate.has_verified_offer_capability_gap()
    assert not gate.review([call])[0].allowed


def test_nonconfirmation_does_not_prove_definitive_offer_refusal():
    ledger = ConversationLedger()
    gate = BankingToolAdmissionGate(ledger)
    gate.observe_knowledge_text("customer_demands_after_unavailable_offer_refusal")
    ledger.observe_input(UserMessage.text("I received a letter about a referral program."))
    for _ in range(3):
        gate.observe_knowledge_search()
        ledger.observe_assistant(AssistantMessage.text("The offer cannot be verified."))
        ledger.observe_input(UserMessage.text("I still want this offer. Transfer me to a human."))
    decision = gate.review(
        [
            ToolCallPart(
                tool_call_id="review-mailing",
                tool_name="transfer_to_human_agents",
                arguments={
                    "reason": "customer_demands_after_unavailable_offer_refusal",
                    "summary": "Review.",
                },
            )
        ]
    )[0]
    assert not decision.allowed
    assert "definitively" in decision.reason


def test_cayu_knowledge_search_receipts_restore_fail_closed() -> None:
    history = [
        UserMessage.text("I received an email promising a special bonus offer."),
        AssistantMessage.text(
            "I could not verify that bonus offer in our public knowledge.",
            raw_data={
                "cayu_tau3_admission": {
                    "knowledge_search_dialogue_sequences": [1],
                    "schema_version": 2,
                }
            },
        ),
    ]
    ledger = ConversationLedger.from_history(history)
    restored = BankingToolAdmissionGate.from_history(history, ledger=ledger)
    restored.observe_knowledge_text(
        "Use unconfirmed_external_communication for the documented transfer case."
    )
    transfer = proposed(
        "transfer-1",
        "transfer_to_human_agents",
        summary="Customer requests review of an unverified emailed offer.",
        reason="unconfirmed_external_communication",
    )

    accepted = restored.review((transfer,))[0]
    malformed_history = [
        history[0],
        AssistantMessage.text(
            history[1].content,
            raw_data={
                "cayu_tau3_admission": {
                    "knowledge_search_dialogue_sequences": [2, 1],
                    "schema_version": 2,
                }
            },
        ),
    ]
    malformed_ledger = ConversationLedger.from_history(malformed_history)
    fail_closed = BankingToolAdmissionGate.from_history(
        malformed_history,
        ledger=malformed_ledger,
    )
    fail_closed.observe_knowledge_text(
        "Use unconfirmed_external_communication for the documented transfer case."
    )
    rejected = fail_closed.review((transfer,))[0]

    assert accepted.allowed is True
    assert restored.export_state()["knowledge_search_dialogue_sequences"] == [1]
    assert rejected.allowed is False
    assert fail_closed.export_state()["knowledge_search_dialogue_sequences"] == []


def test_public_capability_names_restore_from_valid_state_only() -> None:
    tool_name = "get_bank_account_transactions_9173"
    history = [
        UserMessage.text("Review my account activity."),
        AssistantMessage.text(
            "I am checking the procedure.",
            raw_data={
                "cayu_tau3_admission": {
                    "knowledge_search_dialogue_sequences": [1],
                    "public_tool_identifiers": [tool_name],
                    "schema_version": 3,
                }
            },
        ),
    ]
    ledger = ConversationLedger.from_history(history)
    restored = BankingToolAdmissionGate.from_history(history, ledger=ledger)

    assert restored.public_tool_identifiers() == (tool_name,)
    assert (
        restored.review(
            (
                proposed(
                    "unlock-1",
                    "unlock_discoverable_agent_tool",
                    agent_tool_name=tool_name,
                ),
            )
        )[0].allowed
        is True
    )

    malformed_history = [
        history[0],
        AssistantMessage.text(
            history[1].content,
            raw_data={
                "cayu_tau3_admission": {
                    "knowledge_search_dialogue_sequences": [1],
                    "public_tool_identifiers": ["not a valid tool"],
                    "schema_version": 3,
                }
            },
        ),
    ]
    malformed_ledger = ConversationLedger.from_history(malformed_history)
    fail_closed = BankingToolAdmissionGate.from_history(
        malformed_history,
        ledger=malformed_ledger,
    )
    assert fail_closed.public_tool_identifiers() == ()


def test_nonconfirmation_before_external_claim_does_not_authorize_transfer() -> None:
    ledger = ConversationLedger()
    ledger.observe_assistant(
        AssistantMessage.text("I could not find or verify that offer in our public knowledge.")
    )
    ledger.observe_input(
        UserMessage.text("I received a letter promising a special promotional offer.")
    )

    decision = BankingToolAdmissionGate(ledger, require_knowledge_evidence=False).review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="Customer requests review of an unverified mailed offer.",
                reason="unconfirmed_external_communication",
            ),
        )
    )[0]

    assert decision.allowed is False
    assert "and then an agent response" in decision.reason


def test_unrelated_nonconfirmation_after_external_claim_does_not_authorize_transfer() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(
        UserMessage.text("I received a letter promising a special promotional offer.")
    )
    ledger.observe_assistant(
        AssistantMessage.text("I could not verify your identity from those details.")
    )

    decision = BankingToolAdmissionGate(ledger, require_knowledge_evidence=False).review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="Customer requests review of an unverified mailed offer.",
                reason="unconfirmed_external_communication",
            ),
        )
    )[0]

    assert decision.allowed is False
    assert "agent response" in decision.reason


def test_uncertain_search_result_cannot_authorize_unavailable_offer_refusal_reason() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Is there a loyalty bonus available?"))
    ledger.observe_assistant(
        AssistantMessage.text("I couldn't find or verify that loyalty bonus in the knowledge base.")
    )
    ledger.observe_input(UserMessage.text("Please look again because I still expect the bonus."))
    ledger.observe_input(UserMessage.text("Transfer me to a human agent now."))

    decision = BankingToolAdmissionGate(ledger, require_knowledge_evidence=False).review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="Customer demands a human after an unsuccessful search.",
                reason="customer_demands_after_unavailable_offer_refusal",
            ),
        )
    )[0]

    assert decision.allowed is False
    assert "definitively said" in decision.reason


def test_unavailable_offer_refusal_counts_persistence_before_later_definitive_refusal() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("I received a flyer promising a statement credit offer."))
    ledger.observe_assistant(
        AssistantMessage.text("I could not confirm that statement credit offer.")
    )
    ledger.observe_input(UserMessage.text("But the statement credit is printed on my flyer."))
    ledger.observe_assistant(
        AssistantMessage.text("I still could not verify that statement credit offer.")
    )
    ledger.observe_input(UserMessage.text("Then apply the annual-fee waiver from the flyer."))
    ledger.observe_assistant(
        AssistantMessage.text(
            "I can't apply the flyer's annual-fee waiver because its application window ended."
        )
    )
    ledger.observe_input(
        UserMessage.text("I still insist. Transfer me to a human representative now.")
    )

    gate = BankingToolAdmissionGate(ledger, require_knowledge_evidence=False)
    unavailable_refusal, external_communication = gate.review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="Customer persisted after the bank refused the flyer offers.",
                reason="customer_demands_after_unavailable_offer_refusal",
            ),
            proposed(
                "transfer-2",
                "transfer_to_human_agents",
                summary="Customer requested review of an unverified flyer.",
                reason="unconfirmed_external_communication",
            ),
        )
    )

    assert unavailable_refusal.allowed is True
    assert external_communication.allowed is False
    assert "higher priority" in external_communication.reason


def test_conditional_offer_application_step_is_not_a_definitive_refusal() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("I received a flyer with a promotional offer."))
    ledger.observe_assistant(
        AssistantMessage.text("I can't apply the flyer offer until you provide its offer code.")
    )
    ledger.observe_input(UserMessage.text("But I still want the flyer offer."))
    ledger.observe_input(UserMessage.text("Transfer me to a human representative now."))

    decision = BankingToolAdmissionGate(ledger, require_knowledge_evidence=False).review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="Customer has not supplied the required offer code.",
                reason="customer_demands_after_unavailable_offer_refusal",
            ),
        )
    )[0]

    assert decision.allowed is False
    assert "definitively said" in decision.reason


def test_late_definitive_refusal_still_requires_multiple_persistence_turns() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("I received a flyer with an annual-fee waiver."))
    ledger.observe_assistant(
        AssistantMessage.text("That waiver's application window ended and cannot be reopened.")
    )
    ledger.observe_input(UserMessage.text("Transfer me to a human representative now."))

    decision = BankingToolAdmissionGate(ledger, require_knowledge_evidence=False).review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="Customer immediately requested a human after one refusal.",
                reason="customer_demands_after_unavailable_offer_refusal",
            ),
        )
    )[0]

    assert decision.allowed is False
    assert "after multiple customer persistence turns" in decision.reason


def test_unavailable_offer_refusal_reason_requires_full_observed_sequence() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Do you have a double-points promotion?"))
    ledger.observe_assistant(
        AssistantMessage.text("That promotion does not exist in our system and is not available.")
    )
    ledger.observe_input(UserMessage.text("But I still want that promotion."))
    ledger.observe_assistant(AssistantMessage.text("It remains unavailable."))
    ledger.observe_input(UserMessage.text("I insist. Transfer me to a human representative now."))

    decision = BankingToolAdmissionGate(ledger, require_knowledge_evidence=False).review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="Customer persisted after a confirmed unavailable-offer refusal.",
                reason="customer_demands_after_unavailable_offer_refusal",
            ),
        )
    )[0]

    assert decision.allowed is True


def test_human_assistance_detection_covers_natural_someone_requests() -> None:
    assert customer_requests_human_assistance(
        "I want to speak to someone who can actually override this."
    )
    assert customer_requests_human_assistance("Please put me through to someone now.")
    assert customer_requests_human_assistance(
        "I want you to escalate this and apply at least one of the offers."
    )
    assert not customer_requests_human_assistance(
        "Someone sent me a flyer, but I only want you to check it."
    )
    assert not customer_requests_human_assistance(
        "Can you explain what the escalation policy means?"
    )
    assert not customer_requests_human_assistance(
        "No—don\u2019t transfer me yet. Please answer my question."
    )
    assert not customer_requests_human_assistance(
        "No—stop transferring me. I want a yes-or-no answer."
    )
    assert customer_requests_human_assistance(
        "Don't transfer me yet—actually, please connect me to a human now."
    )


def test_latest_customer_transfer_refusal_blocks_every_reason() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please transfer me to a human agent."))
    ledger.observe_assistant(AssistantMessage.text("I requested that transfer."))
    ledger.observe_input(UserMessage.text("No—don\u2019t transfer me yet. Answer this first."))

    decision = BankingToolAdmissionGate(ledger, require_knowledge_evidence=False).review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="A system problem needs specialist review.",
                reason="technical_system_error",
            ),
        )
    )[0]

    assert decision.allowed is False
    assert "latest customer turn explicitly declines" in decision.reason


def test_customer_disposition_reason_cannot_reuse_stale_human_request() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please transfer me to a human agent."))
    ledger.observe_assistant(AssistantMessage.text("I requested that transfer."))
    ledger.observe_input(UserMessage.text("Can you answer whether the offer is available?"))

    decision = BankingToolAdmissionGate(ledger, require_knowledge_evidence=False).review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="Customer was previously frustrated.",
                reason="customer_frustrated_demands_human",
            ),
        )
    )[0]

    assert decision.allowed is False
    assert "latest customer turn" in decision.reason


def test_specific_unconfirmed_external_reason_precedes_generic_frustration() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("My flyer promises a promotional statement credit."))
    ledger.observe_assistant(
        AssistantMessage.text("I could not verify that statement-credit offer in our records.")
    )
    ledger.observe_input(UserMessage.text("This is frustrating. Connect me to a human agent."))

    decision = BankingToolAdmissionGate(ledger, require_knowledge_evidence=False).review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="Customer is frustrated about the unverified flyer.",
                reason="customer_frustrated_demands_human",
            ),
        )
    )[0]

    assert decision.allowed is False
    assert "unconfirmed_external_communication is fully established" in decision.reason


def test_bare_escalation_request_can_complete_unavailable_offer_sequence() -> None:
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("My flyer promises a $1,000 statement credit."))
    ledger.observe_assistant(
        AssistantMessage.text("That statement-credit offer is unavailable in our system.")
    )
    ledger.observe_input(UserMessage.text("But I still want the statement-credit offer."))
    ledger.observe_assistant(
        AssistantMessage.text("The flyer offer remains unavailable and cannot be applied.")
    )
    ledger.observe_input(UserMessage.text("I want you to escalate this and honor the flyer offer."))

    decision = BankingToolAdmissionGate(ledger, require_knowledge_evidence=False).review(
        (
            proposed(
                "transfer-1",
                "transfer_to_human_agents",
                summary="Customer persisted after the unavailable offer refusal.",
                reason="customer_demands_after_unavailable_offer_refusal",
            ),
        )
    )[0]

    assert decision.allowed is True
