"""Read-only capabilities stay available across resources in an outcome plan."""

import asyncio
import json

import pytest
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from test_runtime_banking import assembly, call, offered, text, work

from cayu_tau3_banking.admission import BankingToolAdmissionGate
from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.bridge import InternalToolResult
from cayu_tau3_banking.controller import (
    SYNC_WORK_PLAN_TOOL,
    UPDATE_WORK_PLAN_TOOL,
    BankingWorkflowController,
)
from cayu_tau3_banking.credit_card_dispute import (
    AUDIT_CREDIT_CARD_DISPUTES_TOOL,
    PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL,
)
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.ledger import ConversationLedger
from cayu_tau3_banking.reward_audit import (
    AUDIT_CREDIT_CARD_REWARDS_TOOL,
    PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL,
)
from cayu_tau3_banking.savings_interest_audit import (
    AUDIT_SAVINGS_INTEREST_TOOL,
    PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
)

POLICY = "doc_savings_accounts_gold_account_001"
HELPERS = (
    PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL,
    AUDIT_CREDIT_CARD_DISPUTES_TOOL,
    PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL,
    AUDIT_CREDIT_CARD_REWARDS_TOOL,
    PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
    AUDIT_SAVINGS_INTEREST_TOOL,
)


def write_policy(tmp_path):
    (tmp_path / f"{POLICY}.json").write_text(
        json.dumps({"id": POLICY, "title": "Gold Account", "content": "Base APY: 3.2%."})
    )


def observe(ledger, call_id, tool, arguments, content):
    ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": call_id,
                    "name": "call_discoverable_agent_tool",
                    "arguments": {"agent_tool_name": tool, "arguments": json.dumps(arguments)},
                }
            ],
        )
    )
    ledger.observe_input(ToolMessage(id=call_id, role="tool", content=content))


def seed_accounts(ledger):
    records = "\n\n".join(
        f"{index}. Record ID: {account}\n   account_id: {account}\n   user_id: customer\n"
        "   class: saving\n   level: Gold Account\n   status: OPEN\n"
        "   current_holdings: 12000.00"
        for index, account in enumerate(("savings-a", "savings-b"), 1)
    )
    observe(
        ledger,
        "accounts",
        "get_all_user_accounts_by_user_id_3847",
        {"user_id": "customer"},
        "Bank Accounts:\nFound 2 record(s) in 'accounts':\n\n"
        + records
        + "\n\nCredit Card Accounts:\nNo credit card accounts found.",
    )
    for account in ("savings-a", "savings-b"):
        observe(
            ledger,
            f"transactions-{account}",
            "get_bank_account_transactions_9173",
            {"account_id": account},
            f"Found 1 record(s) in 'bank_account_transaction_history':\n\n"
            f"1. Record ID: interest-{account}\n   transaction_id: interest-{account}\n"
            f"   account_id: {account}\n   date: 10/31/2025\n"
            "   description: MONTHLY INTEREST CREDIT\n   amount: 32.00\n"
            "   type: interest_credit\n   status: posted",
        )


def session(tmp_path):
    write_policy(tmp_path)
    ledger = ConversationLedger()
    controller = BankingWorkflowController(ledger, planning_mode="outcomes")
    return BankingMemoryRuntime(BankingKnowledgeCorpus.from_directory(tmp_path)).new_session(
        ledger=ledger,
        controller=controller,
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=False,
    )


@pytest.mark.parametrize("name", HELPERS)
def test_helper_success_does_not_remove_capability_in_outcome_mode(tmp_path, name):
    memory = session(tmp_path)
    before = memory.controller.revision
    result = memory._observe_successful_private_action(name, InternalToolResult(content="ok"))
    assert not result.is_error
    assert name in {schema["name"] for schema in memory.tool_schemas()}
    assert memory.controller.revision == before


def test_two_resources_can_be_prepared_and_audited_without_plan_or_observation_churn(tmp_path):
    memory = session(tmp_path)
    seed_accounts(memory.ledger)
    epoch = (memory.ledger.last_tool_action_sequence, memory.controller.revision)
    for account in ("savings-a", "savings-b"):
        assert PREPARE_SAVINGS_INTEREST_POLICY_TOOL in {
            schema["name"] for schema in memory.tool_schemas()
        }
        prepared = asyncio.run(
            memory.invoke(
                PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
                {"user_id": "customer", "account_id": account},
            )
        )
        assert not prepared.is_error, prepared.content
        dossier_id = json.loads(prepared.content.splitlines()[-1])["policy_dossier_id"]
        assert AUDIT_SAVINGS_INTEREST_TOOL in {schema["name"] for schema in memory.tool_schemas()}
        audited = asyncio.run(
            memory.invoke(
                AUDIT_SAVINGS_INTEREST_TOOL,
                {
                    "user_id": "customer",
                    "account_id": account,
                    "policy_dossier_id": dossier_id,
                    "base_or_tier_apy_percent": 3.2,
                    "checking_boosts": [],
                    "credit_card_bonuses": [],
                    "other_stackable_components": [],
                },
            )
        )
        assert not audited.is_error, audited.content
    assert (memory.ledger.last_tool_action_sequence, memory.controller.revision) == epoch


def test_exposing_helper_does_not_bypass_observation_validation(tmp_path):
    memory = session(tmp_path)
    result = asyncio.run(
        memory.invoke(
            PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
            {"user_id": "customer", "account_id": "unobserved-account"},
        )
    )
    assert result.is_error
    assert "No complete parsed successful all-accounts lookup" in result.content
    assert PREPARE_SAVINGS_INTEREST_POLICY_TOOL in {
        schema["name"] for schema in memory.tool_schemas()
    }


def test_native_provider_exposes_same_helper_for_second_resource(tmp_path):
    write_policy(tmp_path)
    initial = work()
    initial.pop("next_operation")
    final = work(done=True, reference=POLICY)
    final.pop("next_operation")
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [initial]}),
            call(
                PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
                {"user_id": "customer", "account_id": "savings-a"},
                "prepare-a",
            ),
            call(
                PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
                {"user_id": "customer", "account_id": "savings-b"},
                "prepare-b",
            ),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [final]}, "finish"),
            text("Both policies are available."),
        ],
        planning_mode="outcomes",
        native_recall=True,
    )
    state = agent.get_init_state()

    async def seed():
        seed_accounts(captured["state"].ledger)

    try:
        asyncio.run_coroutine_threadsafe(seed(), state._loop).result(10)
        reply, _ = agent.generate_next_message(
            UserMessage.text("Please look up policy for both savings accounts."), state
        )
        assert reply.content == "Both policies are available."
        assert len(provider.requests) == 5
        assert PREPARE_SAVINGS_INTEREST_POLICY_TOOL in offered(provider.requests[2])
        assert captured["state"].controller.revision == 2
    finally:
        state.stop()
