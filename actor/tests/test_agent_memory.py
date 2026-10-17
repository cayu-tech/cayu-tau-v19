from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from cayu import ToolCallPart
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from cayu_tau3_banking.action_verifier import (
    ActionCallReview,
    ActionReviewBatch,
    ActionReviewCorrection,
)
from cayu_tau3_banking.admission import BankingToolAdmissionGate
from cayu_tau3_banking.agent_memory import (
    _MAX_KNOWLEDGE_SEARCHES,
    _MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE,
    BankingMemoryRuntime,
)
from cayu_tau3_banking.controller import (
    PREPARE_RESPONSE_TOOL,
    SYNC_WORK_PLAN_TOOL,
    UPDATE_WORK_PLAN_TOOL,
    BankingWorkflowController,
)
from cayu_tau3_banking.credit_card_dispute import (
    AUDIT_CREDIT_CARD_DISPUTES_TOOL,
    FILE_CREDIT_CARD_DISPUTE_TOOL,
    PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL,
)
from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL, BankingKnowledgeCorpus
from cayu_tau3_banking.ledger import ConversationLedger
from cayu_tau3_banking.plan_verifier import PlanReviewCorrection, PlanReviewDecision
from cayu_tau3_banking.product_navigation import NAVIGATION_TOOLS
from cayu_tau3_banking.response_verifier import (
    ResponseReviewCorrection,
    ResponseReviewDecision,
)
from cayu_tau3_banking.reward_audit import (
    AUDIT_CREDIT_CARD_REWARDS_TOOL,
    PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL,
)
from cayu_tau3_banking.savings_interest_audit import (
    AUDIT_SAVINGS_INTEREST_TOOL,
    PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
)


def run(coroutine):
    return asyncio.run(coroutine)


def runtime(
    tmp_path: Path,
    *,
    plan_verifier=None,
    action_verifier=None,
    response_verifier=None,
) -> BankingMemoryRuntime:
    (tmp_path / "card_limits.json").write_text(
        json.dumps(
            {
                "id": "card_limits",
                "title": "Card purchase limits",
                "content": "The Sapphire card has a daily purchase limit of $4,500.",
            }
        ),
        encoding="utf-8",
    )
    return BankingMemoryRuntime(
        BankingKnowledgeCorpus.from_directory(tmp_path),
        plan_verifier=plan_verifier,
        action_verifier=action_verifier,
        response_verifier=response_verifier,
    )


def write_human_transfer_policy(tmp_path: Path) -> None:
    (tmp_path / "transfer_policy.json").write_text(
        json.dumps(
            {
                "id": "transfer_policy",
                "title": "Internal: Human Agent Transfer Reason Codes",
                "content": (
                    "Use transfer_to_human_agents with the highest applicable reason. "
                    "TIER 1 (HIGHEST PRIORITY): "
                    "customer_demands_after_unavailable_offer_refusal. "
                    "TIER 2: unconfirmed_external_communication."
                ),
            }
        ),
        encoding="utf-8",
    )


def write_credit_card_dispute_policies(tmp_path: Path) -> None:
    policies = (
        (
            "dispute_history",
            "Checking User Dispute History (Internal)",
            "Retrieve the user's complete dispute history before deciding eligibility.",
        ),
        (
            "dispute_filing",
            "Filing a Credit Card Transaction Dispute (Internal)",
            "Collect all required fields and file each requested transaction exactly once.",
        ),
        (
            "credit_card_number",
            "How to Find Your Rho-Bank Credit Card Numbers Online",
            (
                "Give the customer get_card_last_4_digits(credit_card_account_id: str). "
                "Providing this tool replaces manual retrieval of the last four digits."
            ),
        ),
        (
            "dispute_eligibility",
            "Provisional Credit Eligibility Guidelines (Internal)",
            """The account must be open for at least 60 days.
2. Dispute Reason Category:
- 'unauthorized_fraudulent_charge'
- 'duplicate_charge'
- 'goods_services_not_received' (only if purchase was made more than 30 days ago)
3. Dispute Amount: The transaction amount must be at least $25.00.
4. Previous Disputes: The customer has not filed more than 2 disputes.
- Premium Tier (Gold Rewards Card): Maximum $10000.00""",
        ),
    )
    for document_id, title, content in policies:
        (tmp_path / f"{document_id}.json").write_text(
            json.dumps({"id": document_id, "title": title, "content": content}),
            encoding="utf-8",
        )


def new_session(
    tmp_path: Path,
    *,
    workflow_controller_enabled: bool = False,
    action_verifier=None,
):
    memory_runtime = runtime(tmp_path, action_verifier=action_verifier)
    ledger = ConversationLedger()
    controller = BankingWorkflowController(ledger)
    return memory_runtime.new_session(
        ledger=ledger,
        controller=controller,
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=workflow_controller_enabled,
    )


class ScriptedPlanVerifier:
    def __init__(self, decisions: list[PlanReviewDecision]) -> None:
        self.decisions = list(decisions)
        self.calls: list[dict[str, object]] = []

    async def review(
        self,
        *,
        customer_messages,
        plan_state,
        reclassifiable_item_ids=(),
        dialogue_context="",
    ) -> PlanReviewDecision:
        self.calls.append(
            {
                "customer_messages": tuple(customer_messages),
                "plan_state": plan_state,
                "reclassifiable_item_ids": tuple(reclassifiable_item_ids),
                "dialogue_context": dialogue_context,
            }
        )
        return self.decisions.pop(0)


class ScriptedActionVerifier:
    def __init__(self, batches: list[ActionReviewBatch]) -> None:
        self.batches = list(batches)
        self.calls: list[dict[str, object]] = []

    async def review(
        self,
        *,
        customer_messages,
        observation_context,
        workflow_context,
        calls,
    ) -> ActionReviewBatch:
        self.calls.append(
            {
                "customer_messages": tuple(customer_messages),
                "observation_context": observation_context,
                "workflow_context": workflow_context,
                "calls": tuple(calls),
            }
        )
        return self.batches.pop(0)


class ScriptedResponseVerifier:
    def __init__(self, decisions: list[ResponseReviewDecision]) -> None:
        self.decisions = list(decisions)
        self.calls: list[dict[str, object]] = []

    async def review(
        self,
        *,
        customer_messages,
        observation_context,
        response_state,
        proposed_response,
    ) -> ResponseReviewDecision:
        self.calls.append(
            {
                "customer_messages": tuple(customer_messages),
                "observation_context": observation_context,
                "response_state": response_state,
                "proposed_response": proposed_response,
            }
        )
        return self.decisions.pop(0)


def test_prepare_context_publishes_work_context_and_retrieves_knowledge(
    tmp_path: Path,
) -> None:
    session = new_session(tmp_path, workflow_controller_enabled=True)
    session.ledger.observe_input(UserMessage.text("What is the Sapphire purchase limit?"))

    context = run(session.prepare_context())
    stored = run(session.work_context_store.load_work_context(session.task_id))

    assert stored is not None
    assert stored.revision == 1
    assert "Sapphire purchase limit" in stored.goal
    assert context is not None
    assert "<cayu-work-context>" in context
    assert "<banking-knowledge-recall>" in context
    assert "$4,500" in context
    assert session.export_state()["last_recall"]["injected_count"] == 1
    assert session.export_state()["knowledge_retrieval"] == {
        "embedding_dimensions": None,
        "embedding_model": None,
        "embedding_representation": None,
        "fusion_configuration": "cayu.tau3.banking.bm25-lexical.v1",
        "mode": "lexical",
    }
    assert session.export_state()["schema_version"] == 14
    assert session.export_state()["human_transfer_policy"] == {
        "available_documents": 0,
        "injected_this_turn": False,
    }
    assert session.export_state()["product_comparison_dossier"] == {
        "injected_documents_this_turn": 0,
    }
    assert session.admission_gate.export_state()["knowledge_identifier_count"] >= 1
    assert session.admission_gate.export_state()["knowledge_search_dialogue_sequences"] == [1]

    completed = run(
        session.invoke(
            SYNC_WORK_PLAN_TOOL,
            {
                "items": [
                    {
                        "id": "answer-limit",
                        "goal": "Answer the Sapphire purchase-limit question",
                        "completion_mode": "information",
                        "status": "completed",
                        "outcome": "answered",
                        "depends_on": [],
                        "next_operation": {
                            "owner": "none",
                            "kind": "none",
                            "tool_name": "",
                            "arguments": {},
                            "summary": "",
                        },
                        "evidence": "The answer is in the injected card-limit policy.",
                        "evidence_refs": [
                            {
                                "kind": "knowledge_revision",
                                "reference_id": "card_limits",
                                "version": "1",
                            }
                        ],
                    }
                ]
            },
        )
    )
    assert completed.is_error is False


def test_prepare_context_preserves_previously_exposed_public_capability_names(
    tmp_path: Path,
) -> None:
    session = new_session(tmp_path)
    session.admission_gate.observe_knowledge_text(
        "Use get_bank_account_transactions_9173 to retrieve the account history."
    )
    session.ledger.observe_input(UserMessage.text("How much cash did I withdraw today?"))

    context = run(session.prepare_context())

    assert context is not None
    assert "<public-capability-memory>" in context
    assert "get_bank_account_transactions_9173" in context
    assert "does not discover new tools" in context


def test_prepare_context_pins_complete_credit_card_dispute_policy(
    tmp_path: Path,
) -> None:
    write_credit_card_dispute_policies(tmp_path)
    session = new_session(tmp_path, workflow_controller_enabled=True)
    session.ledger.observe_input(UserMessage.text("These are charges on my Silver Rewards Card."))
    session.ledger.observe_input(UserMessage.text("Please dispute both charges."))

    context = run(session.prepare_context())

    assert context is not None
    for entry_id in (
        "dispute_history",
        "dispute_filing",
        "credit_card_number",
        "dispute_eligibility",
    ):
        assert context.count(f'"entry_id":"{entry_id}"') == 1
    assert "get_card_last_4_digits" in context
    assert session.admission_gate.knowledge_identifier_is_exposed("get_card_last_4_digits")


def test_credit_card_dispute_last_four_requires_customer_tool_handoff(
    tmp_path: Path,
) -> None:
    write_credit_card_dispute_policies(tmp_path)
    session = new_session(tmp_path, workflow_controller_enabled=True)
    session.ledger.observe_input(UserMessage.text("This transaction is on my Silver Rewards Card."))
    session.ledger.observe_input(UserMessage.text("Please dispute that charge."))
    run(session.prepare_context())
    session.ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "accounts-1",
                    "name": "get_credit_card_accounts_by_user",
                    "arguments": {"user_id": "user-7"},
                }
            ],
        )
    )
    session.ledger.observe_input(
        ToolMessage(
            id="accounts-1",
            role="tool",
            requestor="assistant",
            content=(
                "Found 1 record(s) in 'credit_card_accounts':\n\n"
                "1. Record ID: card-1\n"
                "   account_id: card-1\n"
                "   user_id: user-7\n"
                "   card_type: Silver Rewards Card\n"
                "   date_of_account_open: 03/15/2024"
            ),
        )
    )
    waiting = {
        "id": "dispute-charge",
        "goal": "File the requested credit-card transaction dispute",
        "completion_mode": "operation",
        "status": "awaiting_customer",
        "outcome": "unresolved",
        "depends_on": [],
        "next_operation": {
            "owner": "customer",
            "kind": "customer_response",
            "tool_name": "",
            "arguments": {},
            "summary": "Ask for the last four digits of the Silver Rewards Card.",
        },
        "evidence": "The customer requested the transaction dispute.",
        "evidence_refs": [{"kind": "customer_message", "reference_id": "customer-message:2"}],
    }
    assert run(session.invoke(SYNC_WORK_PLAN_TOOL, {"items": [waiting]})).is_error is False

    before_delivery = run(
        session.review_terminal_output("Please read the last four digits from your card.")
    )

    assert before_delivery.allowed is False
    assert before_delivery.correction_code == ("response_dispute_last_four_tool_delivery_required")
    assert before_delivery.required_tool_names == (UPDATE_WORK_PLAN_TOOL,)
    session.ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "give-last-four-tool",
                    "name": "give_discoverable_user_tool",
                    "arguments": {"discoverable_tool_name": "get_card_last_4_digits"},
                }
            ],
        )
    )
    session.ledger.observe_input(
        ToolMessage(
            id="give-last-four-tool",
            role="tool",
            requestor="assistant",
            content="Customer tool provided successfully.",
        )
    )

    before_handoff = run(
        session.review_terminal_output("Please read the last four digits from your card.")
    )

    assert before_handoff.allowed is False
    assert before_handoff.correction_code == ("response_dispute_last_four_tool_handoff_required")
    unobserved_handoff = {
        **waiting,
        "next_operation": {
            "owner": "customer",
            "kind": "tool",
            "tool_name": "get_card_last_4_digits",
            "arguments": {"credit_card_account_id": "invented-card"},
            "summary": "Run get_card_last_4_digits for the credit-card account.",
        },
    }
    assert (
        run(session.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [unobserved_handoff]})).is_error
        is False
    )
    unobserved = run(
        session.review_terminal_output(
            "Run get_card_last_4_digits with credit_card_account_id invented-card."
        )
    )
    assert unobserved.allowed is False
    assert unobserved.correction_code == ("response_dispute_last_four_tool_handoff_required")
    handoff = {
        **waiting,
        "next_operation": {
            "owner": "customer",
            "kind": "tool",
            "tool_name": "get_card_last_4_digits",
            "arguments": {"credit_card_account_id": "card-1"},
            "summary": "Run get_card_last_4_digits for the observed credit-card account.",
        },
    }
    assert run(session.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [handoff]})).is_error is False

    admitted = run(
        session.review_terminal_output(
            "Run get_card_last_4_digits with credit_card_account_id card-1."
        )
    )

    assert admitted.allowed is True


def test_credit_card_dispute_reuses_last_four_from_verified_account_result(
    tmp_path: Path,
) -> None:
    session = new_session(tmp_path, workflow_controller_enabled=True)
    session.ledger.observe_input(
        UserMessage.text("Please dispute the charge on my Silver Rewards Card.")
    )
    session.ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="verify-1",
                    name="log_verification",
                    arguments={"user_id": "user-7"},
                    requestor="assistant",
                ),
                ToolCall(
                    id="accounts-1",
                    name="get_credit_card_accounts_by_user",
                    arguments={"user_id": "user-7"},
                    requestor="assistant",
                ),
            ],
        )
    )
    session.ledger.observe_input(
        ToolMessage(
            id="verify-1",
            role="tool",
            requestor="assistant",
            content="Verification logged successfully.",
        )
    )
    session.ledger.observe_input(
        ToolMessage(
            id="accounts-1",
            role="tool",
            requestor="assistant",
            content=(
                "Found 1 record(s) in 'credit_card_accounts':\n\n"
                "1. Record ID: card-1\n"
                "   account_id: card-1\n"
                "   user_id: user-7\n"
                "   card_type: Silver Rewards Card\n"
                "   account_status: ACTIVE\n"
                "   card_last_4_digits: 7294"
            ),
        )
    )
    waiting = {
        "id": "dispute-charge",
        "goal": "File the requested Silver Rewards Card transaction dispute.",
        "completion_mode": "operation",
        "status": "awaiting_customer",
        "outcome": "unresolved",
        "depends_on": [],
        "next_operation": {
            "owner": "customer",
            "kind": "customer_response",
            "tool_name": "",
            "arguments": {},
            "summary": "Ask for the last four digits of the Silver Rewards Card.",
        },
        "evidence": "The customer requested the transaction dispute.",
        "evidence_refs": [{"kind": "customer_message", "reference_id": "customer-message:1"}],
    }
    assert run(session.invoke(SYNC_WORK_PLAN_TOOL, {"items": [waiting]})).is_error is False

    decision = run(session.review_terminal_output("What are the card's last four digits?"))

    assert decision.allowed is False
    assert decision.correction_code == "response_dispute_last_four_already_observed"
    assert decision.required_tool_names == (UPDATE_WORK_PLAN_TOOL,)
    assert "already contains the last four digits" in decision.reason


def test_terminal_rejects_generic_handoff_for_audited_dispute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = new_session(tmp_path, workflow_controller_enabled=True)
    session.ledger.observe_input(
        UserMessage.text("Please file my credit-card transaction dispute.")
    )
    waiting = {
        "id": "dispute-charge",
        "goal": "File the requested credit-card transaction dispute",
        "completion_mode": "operation",
        "status": "awaiting_customer",
        "outcome": "unresolved",
        "depends_on": [],
        "next_operation": {
            "owner": "customer",
            "kind": "customer_response",
            "tool_name": "",
            "arguments": {},
            "summary": "Submit the dispute through another support channel.",
        },
        "evidence": "The customer requested the transaction dispute.",
        "evidence_refs": [{"kind": "customer_message", "reference_id": "customer-message:1"}],
    }
    assert run(session.invoke(SYNC_WORK_PLAN_TOOL, {"items": [waiting]})).is_error is False
    monkeypatch.setattr(
        session._credit_card_dispute_action_gate,
        "terminal_handoff_denial",
        lambda _response_state: "One audited filing remains agent-executable.",
    )

    decision = run(session.review_terminal_output("Please submit through another support channel."))

    assert decision.allowed is False
    assert decision.correction_code == "response_audited_dispute_filing_required"
    assert decision.required_tool_names == (UPDATE_WORK_PLAN_TOOL,)


def test_referral_gate_allows_collecting_verification_facts(tmp_path: Path) -> None:
    session = new_session(tmp_path, workflow_controller_enabled=True)
    session.ledger.observe_input(
        UserMessage.text("Which account should I use to maximize my referral bonus?")
    )
    waiting = {
        "id": "referral-recommendation",
        "goal": "Recommend the best account for the customer's referral bonus.",
        "completion_mode": "information",
        "status": "awaiting_customer",
        "outcome": "unresolved",
        "depends_on": [],
        "next_operation": {
            "owner": "customer",
            "kind": "customer_response",
            "tool_name": "",
            "arguments": {},
            "summary": "Collect the customer's identity-verification facts.",
        },
        "evidence": "Identity facts are required before checking referral eligibility.",
        "evidence_refs": [{"kind": "customer_message", "reference_id": "customer-message:1"}],
    }
    assert run(session.invoke(SYNC_WORK_PLAN_TOOL, {"items": [waiting]})).is_error is False

    admission = run(session.review_terminal_output("Please provide your verification details."))

    assert admission.allowed is True


def test_prepare_context_adds_deterministic_product_comparison_coverage(
    tmp_path: Path,
) -> None:
    documents = (
        (
            "doc_credit_cards_ecocard_001",
            "EcoCard: Apply",
            "Minimum credit score: 0. Foreign transaction fee: 1.0%.",
        ),
        (
            "doc_credit_cards_ecocard_008",
            "EcoCard: Account Management",
            "Minimum monthly payment: 1.0%. Virtual card management: Yes.",
        ),
        (
            "unrelated",
            "Unrelated banking policy",
            "Routing numbers identify bank accounts.",
        ),
    )
    for document_id, title, content in documents:
        (tmp_path / f"{document_id}.json").write_text(
            json.dumps({"id": document_id, "title": title, "content": content}),
            encoding="utf-8",
        )
    for index in range(6):
        document_id = f"doc_credit_cards_test_product_{index}_001"
        (tmp_path / f"{document_id}.json").write_text(
            json.dumps(
                {
                    "id": document_id,
                    "title": f"Test Product {index}: Apply",
                    "content": (
                        "Minimum credit score: 700. Foreign transaction fee: 2.0%. "
                        "Minimum payment: 2.0%. Virtual card management: No."
                    ),
                }
            ),
            encoding="utf-8",
        )
    memory_runtime = BankingMemoryRuntime(BankingKnowledgeCorpus.from_directory(tmp_path))
    ledger = ConversationLedger()
    ledger.observe_input(
        UserMessage.text(
            "Which credit card options fit my credit score, a low foreign transaction fee, "
            "a low minimum payment, and virtual card management?"
        )
    )
    controller = BankingWorkflowController(ledger)
    session = memory_runtime.new_session(
        ledger=ledger,
        controller=controller,
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )

    context = run(session.prepare_context())

    assert context is not None
    assert "<banking-product-comparison-dossier>" in context
    assert '"entry_id":"doc_credit_cards_ecocard_001"' in context
    assert '"entry_id":"doc_credit_cards_ecocard_008"' in context
    assert session.export_state()["product_comparison_dossier"]["injected_documents_this_turn"] >= 1


def test_prepare_context_retains_active_referral_intent_across_fact_followup(
    tmp_path: Path,
) -> None:
    documents = (
        (
            "doc_bank_accounts_bank_accounts_(general)_009",
            "Internal: Retrieving Customer Account Information",
            "Use get_all_user_accounts_by_user_id_3847 for complete customer accounts.",
        ),
        (
            "doc_business_checking_accounts_world_blue_001",
            "Getting Started with World Blue International Checking",
            "A referred business must deposit at least $25,000.",
        ),
        (
            "doc_business_checking_accounts_world_blue_007",
            "World Blue Referral Bonuses",
            "Your referral bonus is $300 per successful referral.",
        ),
    )
    for document_id, title, content in documents:
        (tmp_path / f"{document_id}.json").write_text(
            json.dumps({"id": document_id, "title": title, "content": content}),
            encoding="utf-8",
        )
    session = new_session(tmp_path, workflow_controller_enabled=True)
    session.ledger.observe_input(
        UserMessage.text("Which business checking account should I refer for the biggest bonus?")
    )
    active_item = {
        "id": "referral-recommendation",
        "goal": "Recommend the best business checking account for the referral bonus.",
        "completion_mode": "information",
        "status": "awaiting_customer",
        "outcome": "unresolved",
        "depends_on": [],
        "next_operation": {
            "owner": "customer",
            "kind": "customer_response",
            "tool_name": "",
            "arguments": {},
            "summary": "Ask how much the referred business can deposit.",
        },
        "evidence": "The customer requested a business referral recommendation.",
        "evidence_refs": [{"kind": "customer_message", "reference_id": "customer-message:1"}],
    }
    assert run(session.invoke(SYNC_WORK_PLAN_TOOL, {"items": [active_item]})).is_error is False
    first_context = run(session.prepare_context())
    session.ledger.observe_input(UserMessage.text("They can deposit $30,000."))

    followup_context = run(session.prepare_context())
    stored = run(session.work_context_store.load_work_context(session.task_id))

    assert first_context is not None
    assert followup_context is not None
    assert '"entry_id":"doc_business_checking_accounts_world_blue_001"' in followup_context
    assert '"entry_id":"doc_business_checking_accounts_world_blue_007"' in followup_context
    assert '"entry_id":"doc_bank_accounts_bank_accounts_(general)_009"' in followup_context
    assert "get_all_user_accounts_by_user_id_3847" in followup_context
    assert stored is not None
    assert "business checking account" in stored.goal
    assert "Latest customer input: They can deposit $30,000." in stored.goal


def test_terminal_correction_revises_plan_when_search_budget_is_exhausted(
    tmp_path: Path,
) -> None:
    session = new_session(tmp_path, workflow_controller_enabled=True)
    session.ledger.observe_input(UserMessage.text("What is the requested policy?"))
    planned = run(
        session.invoke(
            SYNC_WORK_PLAN_TOOL,
            {
                "items": [
                    {
                        "id": "answer-policy",
                        "goal": "Answer the requested policy question",
                        "completion_mode": "information",
                        "status": "in_progress",
                        "outcome": "unresolved",
                        "depends_on": [],
                        "next_operation": {
                            "owner": "agent",
                            "kind": "knowledge_search",
                            "tool_name": "",
                            "arguments": {},
                            "summary": "Search for the requested policy.",
                        },
                        "evidence": "",
                        "evidence_refs": [],
                    }
                ]
            },
        )
    )
    searches = [
        run(session.invoke(KNOWLEDGE_SEARCH_TOOL, {"query": f"policy query {index}"}))
        for index in range(_MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE)
    ]
    repeated_search_plan = run(
        session.invoke(
            UPDATE_WORK_PLAN_TOOL,
            {
                "items": [
                    {
                        "id": "answer-policy",
                        "goal": "Answer the requested policy question",
                        "completion_mode": "information",
                        "status": "in_progress",
                        "outcome": "unresolved",
                        "depends_on": [],
                        "next_operation": {
                            "owner": "agent",
                            "kind": "knowledge_search",
                            "tool_name": "",
                            "arguments": {},
                            "summary": "Search again for the requested policy.",
                        },
                        "evidence": "",
                        "evidence_refs": [],
                    }
                ]
            },
        )
    )

    admission = run(session.review_terminal_output("Here is the answer."))

    assert planned.is_error is False
    assert all(not result.is_error for result in searches)
    assert repeated_search_plan.is_error is True
    assert repeated_search_plan.diagnostic_code == "sync_operation_shape_rejected"
    assert session.required_action_tool_names() == (UPDATE_WORK_PLAN_TOOL,)
    assert admission.allowed is False
    assert admission.correction_code == "response_actionable_items"
    assert admission.required_tool_names == (UPDATE_WORK_PLAN_TOOL,)
    assert "search capability is now exhausted" in admission.reason


def test_prepare_context_pins_transfer_taxonomy_when_customer_asks_for_someone(
    tmp_path: Path,
) -> None:
    write_human_transfer_policy(tmp_path)
    for index in range(8):
        (tmp_path / f"override_decision_{index}.json").write_text(
            json.dumps(
                {
                    "id": f"override_decision_{index}",
                    "title": f"Override decision procedure {index}",
                    "content": (
                        "Speak to someone about an override decision using this unrelated "
                        f"procedure number {index}."
                    ),
                }
            ),
            encoding="utf-8",
        )
    session = new_session(tmp_path)
    session.ledger.observe_input(
        UserMessage.text("I want to speak to someone who can override this decision.")
    )

    context = run(session.prepare_context())

    assert context is not None
    assert "<banking-human-transfer-policy>" in context
    assert "customer_demands_after_unavailable_offer_refusal" in context
    assert "unconfirmed_external_communication" in context
    assert session.export_state()["knowledge_searches"] == 0
    assert session.export_state()["human_transfer_policy"] == {
        "available_documents": 1,
        "injected_this_turn": True,
    }
    admission = session.admission_gate.export_state()
    assert admission["knowledge_identifier_count"] >= 3
    assert admission["knowledge_search_dialogue_sequences"] == [1]


def test_prepare_context_does_not_duplicate_a_recalled_transfer_taxonomy(
    tmp_path: Path,
) -> None:
    write_human_transfer_policy(tmp_path)
    session = new_session(tmp_path)
    session.ledger.observe_input(
        UserMessage.text(
            "Please transfer_to_human_agents so I can speak to someone; use "
            "customer_demands_after_unavailable_offer_refusal rather than "
            "unconfirmed_external_communication."
        )
    )

    context = run(session.prepare_context())

    assert context is not None
    assert context.count('"entry_id":"transfer_policy"') == 1
    assert session.export_state()["human_transfer_policy"] == {
        "available_documents": 1,
        "injected_this_turn": False,
    }


def test_semantic_plan_review_rejects_during_sync_and_caches_by_immutable_semantics(
    tmp_path: Path,
) -> None:
    verifier = ScriptedPlanVerifier(
        [
            PlanReviewDecision(
                accepted=False,
                correction=PlanReviewCorrection.MISSING_OPERATION_OUTCOME,
                required_new_item_count=1,
                prompt_tokens=20,
                completion_tokens=4,
            ),
            PlanReviewDecision(
                accepted=True,
                correction=PlanReviewCorrection.NONE,
                prompt_tokens=18,
                completion_tokens=2,
            ),
            PlanReviewDecision(
                accepted=True,
                correction=PlanReviewCorrection.NONE,
                prompt_tokens=18,
                completion_tokens=2,
            ),
        ]
    )
    memory_runtime = runtime(tmp_path, plan_verifier=verifier)
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please perform the requested operation."))
    controller = BankingWorkflowController(ledger)
    session = memory_runtime.new_session(
        ledger=ledger,
        controller=controller,
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    waiting = {
        "items": [
            {
                "id": "operation",
                "goal": "Perform the requested operation",
                "completion_mode": "operation",
                "status": "awaiting_customer",
                "outcome": "unresolved",
                "depends_on": [],
                "next_operation": {
                    "owner": "customer",
                    "kind": "customer_response",
                    "tool_name": "",
                    "arguments": {},
                    "summary": "Provide the required input.",
                },
                "evidence": "Customer input is required.",
                "evidence_refs": [],
            }
        ]
    }
    rejected_sync = run(session.invoke(SYNC_WORK_PLAN_TOOL, waiting))
    disposition = {
        "plan_revision": controller.revision,
        "items": [{"item_id": "operation", "kind": "ask_customer"}],
    }
    rejected_prepare = run(session.invoke(PREPARE_RESPONSE_TOOL, disposition))
    waiting["items"][0]["next_operation"]["summary"] = "Provide all required input."
    cached_rejection = run(session.invoke(SYNC_WORK_PLAN_TOOL, waiting))
    waiting["items"].append(
        {
            "id": "additional-operation",
            "goal": "Perform the additional requested operation",
            "completion_mode": "information",
            "status": "awaiting_customer",
            "outcome": "unresolved",
            "depends_on": [],
            "next_operation": {
                "owner": "customer",
                "kind": "customer_response",
                "tool_name": "",
                "arguments": {},
                "summary": "Provide the additional required input.",
            },
            "evidence": "Additional customer input is required.",
            "evidence_refs": [],
        }
    )
    wrong_direction = run(session.invoke(SYNC_WORK_PLAN_TOOL, waiting))
    waiting["items"][1]["completion_mode"] = "operation"
    accepted_sync = run(session.invoke(SYNC_WORK_PLAN_TOOL, waiting))
    waiting["items"][1]["next_operation"]["summary"] = "Provide every additional required input."
    operational_update = run(session.invoke(SYNC_WORK_PLAN_TOOL, waiting))
    revised_disposition = {
        "plan_revision": controller.revision,
        "items": [
            {"item_id": "operation", "kind": "ask_customer"},
            {"item_id": "additional-operation", "kind": "ask_customer"},
        ],
    }
    assert run(session.invoke(PREPARE_RESPONSE_TOOL, revised_disposition)).is_error is False
    accepted = run(session.review_terminal_output("Please provide all required input."))
    consumed = run(session.review_terminal_output("Please provide all required input."))

    assert rejected_sync.is_error is True
    assert rejected_sync.diagnostic_code == "sync_semantic_plan_missing_operation_outcome"
    assert "Correction class: missing_operation_outcome" in rejected_sync.content
    assert rejected_prepare.is_error is True
    assert cached_rejection.is_error is True
    assert wrong_direction.is_error is True
    assert "new operation-mode outcome" in wrong_direction.content
    assert accepted_sync.is_error is False
    assert operational_update.is_error is False
    assert accepted.allowed is True
    assert consumed.allowed is False
    assert consumed.correction_code == "response_without_new_customer_input"
    assert [call["reclassifiable_item_ids"] for call in verifier.calls] == [
        ("operation",),
        ("additional-operation",),
        (),
    ]
    assert controller.export_state()["customer_response_handoffs"] == {
        "additional-operation": 1,
        "operation": 1,
    }
    assert session.export_state()["semantic_plan_verifier"] == {
        "acceptances": 2,
        "cache_hits": 1,
        "completion_tokens": 8,
        "enabled": True,
        "explicit_operation_floor_rejections": 0,
        "prompt_tokens": 56,
        "provider_calls": 3,
        "rejections": 1,
    }


def test_semantic_action_review_rejects_discoverable_call_and_caches_decision(
    tmp_path: Path,
) -> None:
    verifier = ScriptedActionVerifier(
        [
            ActionReviewBatch(
                decisions=(
                    ActionCallReview(
                        call_index=0,
                        accepted=False,
                        correction=ActionReviewCorrection.ARGUMENT_SEMANTIC_MISMATCH,
                    ),
                ),
                prompt_tokens=31,
                completion_tokens=6,
            )
        ]
    )
    memory_runtime = runtime(tmp_path, action_verifier=verifier)
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Open the requested savings account."))
    controller = BankingWorkflowController(
        ledger,
        available_environment_tool_names={"call_discoverable_agent_tool"},
    )
    wrapper_arguments = {
        "agent_tool_name": "open_bank_account_4821",
        "arguments": '{"user_id":"user-1","account_type":"savings"}',
    }
    planned_item = {
        "id": "open-account",
        "goal": "Open the requested savings account",
        "completion_mode": "operation",
        "status": "in_progress",
        "outcome": "unresolved",
        "depends_on": [],
        "next_operation": {
            "owner": "agent",
            "kind": "tool",
            "tool_name": "call_discoverable_agent_tool",
            "arguments": wrapper_arguments,
            "summary": "Open the selected account.",
        },
        "evidence": "The customer requested this operation.",
        "evidence_refs": [],
    }
    plan = controller.invoke(
        SYNC_WORK_PLAN_TOOL,
        {"items": [planned_item]},
    )
    assert plan.is_error is False
    session = memory_runtime.new_session(
        ledger=ledger,
        controller=controller,
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    call = ToolCallPart(
        tool_call_id="open-1",
        tool_name="call_discoverable_agent_tool",
        arguments=wrapper_arguments,
    )

    first = run(session.review([call]))
    cached = run(session.review([call]))
    rewritten_item = {
        **planned_item,
        "next_operation": {
            **planned_item["next_operation"],
            "summary": "Use the available account-opening capability.",
        },
    }
    rewritten = controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [rewritten_item]})
    cached_after_plan_rewrite = run(session.review([call]))

    assert rewritten.is_error is False
    assert first[0].allowed is False
    assert "argument_semantic_mismatch" in first[0].reason
    assert cached == first
    assert cached_after_plan_rewrite == first
    assert len(verifier.calls) == 1
    assert "<observation-ledger>" in verifier.calls[0]["observation_context"]
    assert "<workflow-plan>" in verifier.calls[0]["workflow_context"]
    assert session.export_state()["semantic_action_verifier"] == {
        "acceptances": 0,
        "cache_hits": 2,
        "completion_tokens": 6,
        "enabled": True,
        "prompt_tokens": 31,
        "provider_calls": 1,
        "rejections": 1,
    }


def test_semantic_action_cache_changes_with_typed_workflow_state(tmp_path):
    from test_controller import invoke, item

    from cayu_tau3_banking.agent_memory import _action_review_key

    controller = BankingWorkflowController(ConversationLedger())
    planned = item("operation")
    assert not invoke(controller, [planned]).is_error
    before = controller.render_action_review_context()
    unchanged = {**planned, "evidence": "cosmetic wording"}
    assert not controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [unchanged]}).is_error
    assert controller.render_action_review_context() == before
    changed = {**planned, "status": "in_progress"}
    assert not controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [changed]}).is_error
    after = controller.render_action_review_context()
    assert before != after
    args = {
        "customer_messages": ["Resolve the request."],
        "observation_context": "Same observed facts.",
        "calls": [],
    }
    assert _action_review_key(**args, workflow_context=before) != _action_review_key(
        **args, workflow_context=after
    )


def test_semantic_action_review_receives_typed_evidence_for_referenced_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = ScriptedActionVerifier(
        [
            ActionReviewBatch(
                decisions=(
                    ActionCallReview(
                        call_index=0,
                        accepted=True,
                        correction=ActionReviewCorrection.NONE,
                    ),
                )
            )
        ]
    )
    memory_runtime = runtime(tmp_path, action_verifier=verifier)
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Replace my Silver Rewards Card."))
    ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="cards-1",
                    name="get_credit_card_accounts_by_user",
                    arguments={"user_id": "user-7"},
                    requestor="assistant",
                )
            ],
        )
    )
    ledger.observe_input(
        ToolMessage(
            id="cards-1",
            role="tool",
            requestor="assistant",
            content=(
                "Found 1 record(s) in 'credit_card_accounts':\n\n"
                "1. Record ID: silver-card\n"
                "   account_id: silver-card\n"
                "   user_id: user-7\n"
                "   card_type: Silver Rewards Card\n"
                "   date_of_account_open: 03/15/2024\n"
                "   current_balance: $100.00\n"
                "   reward_points: 10 points"
            ),
        )
    )
    wrapper_arguments = {
        "agent_tool_name": "order_replacement_credit_card_7291",
        "arguments": json.dumps(
            {
                "credit_card_account_id": "silver-card",
                "user_id": "user-7",
                "shipping_address": "1 Main Street",
                "reason": "damaged",
                "expedited_shipping": False,
            }
        ),
    }
    controller = BankingWorkflowController(
        ledger,
        available_environment_tool_names={"call_discoverable_agent_tool"},
    )
    plan = controller.invoke(
        SYNC_WORK_PLAN_TOOL,
        {
            "items": [
                {
                    "id": "replace-card",
                    "goal": "Replace the customer's Silver Rewards Card",
                    "completion_mode": "operation",
                    "status": "in_progress",
                    "outcome": "unresolved",
                    "depends_on": [],
                    "next_operation": {
                        "owner": "agent",
                        "kind": "tool",
                        "tool_name": "call_discoverable_agent_tool",
                        "arguments": wrapper_arguments,
                        "summary": "Order the requested replacement card.",
                    },
                    "evidence": "The customer requested a replacement.",
                    "evidence_refs": [],
                }
            ]
        },
    )
    assert plan.is_error is False
    session = memory_runtime.new_session(
        ledger=ledger,
        controller=controller,
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    monkeypatch.setattr(
        session._credit_card_dispute_action_gate,
        "render_context",
        lambda: "<credit-card-dispute-authorizations>audited</credit-card-dispute-authorizations>",
    )
    call = ToolCallPart(
        tool_call_id="replace-1",
        tool_name="call_discoverable_agent_tool",
        arguments=wrapper_arguments,
    )

    admission = run(session.review([call]))

    assert admission[0].allowed is True
    context = verifier.calls[0]["observation_context"]
    assert '"account_id": "silver-card"' in context
    assert '"card_type": "Silver Rewards Card"' in context
    assert '"user_id": "user-7"' in context
    assert "<credit-card-dispute-authorizations>audited" in context


def test_isolated_reviewers_receive_delivered_dialogue_and_invalidate_cached_verdict(tmp_path):
    action = ScriptedActionVerifier(
        [
            ActionReviewBatch(decisions=(ActionCallReview(0, True, ActionReviewCorrection.NONE),)),
            ActionReviewBatch(decisions=(ActionCallReview(0, True, ActionReviewCorrection.NONE),)),
        ]
    )
    response = ScriptedResponseVerifier(
        [
            ResponseReviewDecision(True, ResponseReviewCorrection.NONE),
            ResponseReviewDecision(True, ResponseReviewCorrection.NONE),
        ]
    )
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("How do I set up a travel notification?"))
    session = runtime(tmp_path, action_verifier=action, response_verifier=response).new_session(
        ledger=ledger,
        controller=BankingWorkflowController(ledger),
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    call = ToolCallPart(
        tool_call_id="handoff",
        tool_name="transfer_to_human_agents",
        arguments={
            "reason": "kb_search_unsuccessful_customer_requests_transfer",
            "summary": "Travel notification.",
        },
    )
    # Projection tests deliberately bypass unrelated structural plan gates.
    for _ in range(2):
        assert run(session._review_semantic_actions([call], (0,)))[0].allowed
        assert run(session._review_semantic_response("Would you like human help?")).allowed
    assert len(action.calls) == len(response.calls) == 1
    delivered = "I could not find the travel-notification instructions. Would you like human help?"
    session.admission_gate.observe_knowledge_search()
    ledger.observe_assistant(AssistantMessage.text(delivered))
    assert run(session._review_semantic_actions([call], (0,)))[0].allowed
    assert run(session._review_semantic_response("Would you like human help?")).allowed
    assert len(action.calls) == len(response.calls) == 2
    for verifier in (action, response):
        context = verifier.calls[-1]["observation_context"]
        assert delivered in context
        assert '"role": "assistant"' in context
        assert "<knowledge-search-observations>" in context
        assert '"dialogue_sequences": [1]' in context


def test_plan_review_uses_delivered_dialogue_and_refreshes_semantic_cache(tmp_path):
    verifier = ScriptedPlanVerifier(
        [
            PlanReviewDecision(accepted=True, correction=PlanReviewCorrection.NONE),
            PlanReviewDecision(accepted=True, correction=PlanReviewCorrection.NONE),
        ]
    )
    ledger = ConversationLedger()
    ledger.observe_input(
        UserMessage.text("I want the card with the highest rewards. Recommend one.")
    )
    controller = BankingWorkflowController(ledger)
    plan = {
        "items": [
            {
                "id": "compare",
                "goal": "Recommend an option",
                "completion_mode": "information",
                "status": "in_progress",
                "outcome": "unresolved",
                "depends_on": [],
                "next_operation": {
                    "owner": "agent",
                    "kind": "knowledge_search",
                    "tool_name": "",
                    "arguments": {},
                    "summary": "Compare public terms.",
                },
                "evidence": "",
                "evidence_refs": [],
            }
        ]
    }
    assert not controller.invoke(SYNC_WORK_PLAN_TOOL, plan).is_error
    session = runtime(tmp_path, plan_verifier=verifier).new_session(
        ledger=ledger,
        controller=controller,
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    assert run(session._review_semantic_plan()).allowed
    assert run(session._review_semantic_plan()).allowed
    assert len(verifier.calls) == 1
    ledger.observe_assistant(AssistantMessage.text("There are two options. Which do you prefer?"))
    assert run(session._review_semantic_plan()).allowed
    assert len(verifier.calls) == 2
    assert "There are two options" in verifier.calls[-1]["dialogue_context"]
    assert (
        session.export_state()["semantic_plan_verifier"]["explicit_operation_floor_rejections"] == 0
    )


def test_repeated_selection_is_a_semantic_rejection_not_a_regex_transition(tmp_path):
    verifier = ScriptedResponseVerifier(
        [
            ResponseReviewDecision(
                False, ResponseReviewCorrection.REPEATED_OR_STALLED_REQUEST, ("acquire",)
            ),
        ]
    )
    ledger = ConversationLedger()
    ledger.observe_assistant(AssistantMessage.text("Would you like the Jade Rewards Card?"))
    ledger.observe_input(UserMessage.text("Yes, that one. Please proceed."))
    controller = BankingWorkflowController(ledger)
    plan = {
        "items": [
            {
                "id": "acquire",
                "goal": "Obtain the selected product",
                "completion_mode": "operation",
                "status": "awaiting_customer",
                "outcome": "unresolved",
                "depends_on": [],
                "next_operation": {
                    "owner": "customer",
                    "kind": "customer_product_selection",
                    "tool_name": "",
                    "arguments": {},
                    "summary": "Ask which product to select.",
                },
                "evidence": "The customer replied to the proposed product.",
                "evidence_refs": [
                    {"kind": "customer_message", "reference_id": "customer-message:1"}
                ],
            }
        ]
    }
    assert not controller.invoke(SYNC_WORK_PLAN_TOOL, plan).is_error
    session = runtime(tmp_path, response_verifier=verifier).new_session(
        ledger=ledger,
        controller=controller,
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    result = run(session._review_semantic_response("Which product would you like?"))
    assert not result.allowed
    assert result.required_tool_names == (UPDATE_WORK_PLAN_TOOL,)
    assert result.correction_code == "semantic_response_repeated_or_stalled_request"
    assert "Would you like the Jade Rewards Card?" in verifier.calls[0]["observation_context"]
    assert "Yes, that one" in verifier.calls[0]["observation_context"]
    assert controller.export_state()["product_selection_transition_rejections"] == 0


def test_offer_policy_evidence_reaches_action_review_and_invalidates_cached_verdict(tmp_path):
    write_human_transfer_policy(tmp_path)
    verifier = ScriptedActionVerifier(
        [
            ActionReviewBatch(
                decisions=(ActionCallReview(0, False, ActionReviewCorrection.PREMATURE_HANDOFF),)
            ),
            ActionReviewBatch(decisions=(ActionCallReview(0, True, ActionReviewCorrection.NONE),)),
        ]
    )
    ledger = ConversationLedger()
    gate = BankingToolAdmissionGate(ledger)
    ledger.observe_input(UserMessage.text("I received a letter promising a promotional offer."))
    gate.observe_knowledge_search()
    ledger.observe_assistant(AssistantMessage.text("I could not verify that offer."))
    ledger.observe_input(UserMessage.text("Please transfer me to a human to review the offer."))
    session = runtime(tmp_path, action_verifier=verifier).new_session(
        ledger=ledger,
        controller=BankingWorkflowController(ledger),
        admission_gate=gate,
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    call = ToolCallPart(
        tool_call_id="offer-review",
        tool_name="transfer_to_human_agents",
        arguments={"reason": "unconfirmed_external_communication", "summary": "Review offer."},
    )
    # Isolate this observation projection from unrelated structural plan admission.
    assert not run(session._review_semantic_actions([call], (0,)))[0].allowed
    assert "<offer-transfer-policy-evidence>" not in verifier.calls[0]["observation_context"]
    assert "<banking-human-transfer-policy>" in verifier.calls[0]["observation_context"]
    assert '"text_complete":true' in verifier.calls[0]["observation_context"]
    assert not gate.knowledge_identifier_is_exposed("unconfirmed_external_communication")
    gate.observe_knowledge_text("unconfirmed_external_communication")
    assert run(session._review_semantic_actions([call], (0,)))[0].allowed
    assert "<offer-transfer-policy-evidence>" in verifier.calls[1]["observation_context"]
    assert "current_offer_human_review" in verifier.calls[1]["observation_context"]
    assert run(session._review_semantic_actions([call], (0,)))[0].allowed
    assert len(verifier.calls) == 2


@pytest.mark.parametrize("offer_policy_evidence", [False, True])
def test_semantic_action_review_rejects_premature_human_transfer(
    tmp_path: Path, offer_policy_evidence: bool
) -> None:
    verifier = ScriptedActionVerifier(
        [
            ActionReviewBatch(
                decisions=(
                    ActionCallReview(
                        call_index=0,
                        accepted=False,
                        correction=ActionReviewCorrection.PREMATURE_HANDOFF,
                    ),
                )
            )
        ]
    )
    memory_runtime = runtime(tmp_path, action_verifier=verifier)
    ledger = ConversationLedger()
    gate = BankingToolAdmissionGate(ledger)
    if offer_policy_evidence:
        ledger.observe_input(UserMessage.text("I received a letter promising a promotional offer."))
        gate.observe_knowledge_text("unconfirmed_external_communication")
        gate.observe_knowledge_search()
        ledger.observe_assistant(AssistantMessage.text("I could not verify that offer."))
    ledger.observe_input(
        UserMessage.text(
            "Please perform the requested banking operation, then transfer me to a human."
        )
    )
    controller = BankingWorkflowController(
        ledger,
        available_environment_tool_names={"transfer_to_human_agents"},
    )
    transfer_arguments = {
        "summary": "Perform the requested work and transfer the customer.",
        "reason": "unconfirmed_external_communication" if offer_policy_evidence else "other",
    }
    plan = controller.invoke(
        SYNC_WORK_PLAN_TOOL,
        {
            "items": [
                {
                    "id": "requested-operation",
                    "goal": "Perform the requested banking operation",
                    "completion_mode": "operation",
                    "status": "in_progress",
                    "outcome": "unresolved",
                    "depends_on": [],
                    "next_operation": {
                        "owner": "agent",
                        "kind": "tool",
                        "tool_name": "transfer_to_human_agents",
                        "arguments": transfer_arguments,
                        "summary": "Transfer without performing the requested operation.",
                    },
                    "evidence": "The customer requested both outcomes.",
                    "evidence_refs": [],
                }
            ]
        },
    )
    assert plan.is_error is False
    session = memory_runtime.new_session(
        ledger=ledger,
        controller=controller,
        admission_gate=gate,
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    transfer = ToolCallPart(
        tool_call_id="transfer-1",
        tool_name="transfer_to_human_agents",
        arguments=transfer_arguments,
    )

    reviewed = run(session.review([transfer]))

    assert reviewed[0].allowed is False
    assert "premature_handoff" in reviewed[0].reason
    assert [call.tool_name for call in verifier.calls[0]["calls"]] == ["transfer_to_human_agents"]
    assert session.export_state()["semantic_action_verifier"]["rejections"] == 1
    assert (
        "<offer-transfer-policy-evidence>" in verifier.calls[0]["observation_context"]
    ) is offer_policy_evidence


@pytest.mark.parametrize(
    "correction",
    [
        ResponseReviewCorrection.MISSING_REQUIRED_ITEM,
        ResponseReviewCorrection.CUSTOMER_INSTRUCTION_MISMATCH,
        ResponseReviewCorrection.REPEATED_OR_STALLED_REQUEST,
    ],
)
def test_semantic_response_review_requires_complete_customer_handoff(
    tmp_path: Path,
    correction: ResponseReviewCorrection,
) -> None:
    verifier = ScriptedResponseVerifier(
        [
            ResponseReviewDecision(
                accepted=False,
                correction=correction,
                affected_item_ids=("second",),
                prompt_tokens=37,
                completion_tokens=5,
            ),
            ResponseReviewDecision(
                accepted=True,
                correction=ResponseReviewCorrection.NONE,
                prompt_tokens=41,
                completion_tokens=4,
            ),
        ]
    )
    memory_runtime = runtime(tmp_path, response_verifier=verifier)
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please complete both requested operations."))
    controller = BankingWorkflowController(ledger)
    items = [
        {
            "id": item_id,
            "goal": f"Complete requested operation {item_id}",
            "completion_mode": "operation",
            "status": "awaiting_customer",
            "outcome": "unresolved",
            "depends_on": [],
            "next_operation": {
                "owner": "customer",
                "kind": "customer_response",
                "tool_name": "",
                "arguments": {},
                "summary": f"Provide missing value {item_id}.",
            },
            "evidence": "The customer must provide one missing value.",
            "evidence_refs": [],
        }
        for item_id in ("first", "second")
    ]
    assert controller.invoke(SYNC_WORK_PLAN_TOOL, {"items": items}).is_error is False
    session = memory_runtime.new_session(
        ledger=ledger,
        controller=controller,
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )

    rejected = run(session.review_terminal_output("Please provide the first value."))
    accepted = run(
        session.review_terminal_output("Please provide both the first and second values.")
    )

    assert rejected.allowed is False
    assert rejected.correction_code == f"semantic_response_{correction.value}"
    if correction is ResponseReviewCorrection.MISSING_REQUIRED_ITEM:
        assert rejected.required_tool_names == ()
    else:
        assert rejected.required_tool_names == (UPDATE_WORK_PLAN_TOOL,)
        assert "update the plan and advance that operation" in rejected.reason
        assert (
            "If the plan already matches the instruction, correct only the draft" in rejected.reason
        )
    assert "Affected workflow item IDs: second" in rejected.reason
    assert accepted.allowed is True
    assert controller.export_state()["customer_response_handoffs"] == {
        "first": 1,
        "second": 1,
    }
    assert verifier.calls[0]["response_state"]["items"] == [
        {
            "completion_mode": "operation",
            "evidence_refs": [],
            "goal": "Complete requested operation first",
            "id": "first",
            "next_operation": {
                "arguments": {},
                "kind": "customer_response",
                "owner": "customer",
                "summary": "Provide missing value first.",
                "tool_name": "",
            },
            "outcome": "unresolved",
            "response_kind": "ask_customer",
            "status": "awaiting_customer",
        },
        {
            "completion_mode": "operation",
            "evidence_refs": [],
            "goal": "Complete requested operation second",
            "id": "second",
            "next_operation": {
                "arguments": {},
                "kind": "customer_response",
                "owner": "customer",
                "summary": "Provide missing value second.",
                "tool_name": "",
            },
            "outcome": "unresolved",
            "response_kind": "ask_customer",
            "status": "awaiting_customer",
        },
    ]
    assert session.export_state()["semantic_response_verifier"] == {
        "acceptances": 1,
        "cache_hits": 0,
        "completion_tokens": 9,
        "enabled": True,
        "prompt_tokens": 78,
        "provider_calls": 2,
        "rejections": 1,
    }


def test_semantic_mode_rejection_can_reclassify_only_the_new_item(tmp_path: Path) -> None:
    verifier = ScriptedPlanVerifier(
        [
            PlanReviewDecision(
                accepted=False,
                correction=PlanReviewCorrection.WRONG_COMPLETION_MODE_TO_OPERATION,
                affected_item_ids=("operation",),
            ),
            PlanReviewDecision(
                accepted=True,
                correction=PlanReviewCorrection.NONE,
            ),
        ]
    )
    memory_runtime = runtime(tmp_path, plan_verifier=verifier)
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please perform the requested operation."))
    controller = BankingWorkflowController(ledger)
    session = memory_runtime.new_session(
        ledger=ledger,
        controller=controller,
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    plan = {
        "items": [
            {
                "id": "operation",
                "goal": "Perform the requested operation",
                "completion_mode": "information",
                "status": "awaiting_customer",
                "outcome": "unresolved",
                "depends_on": [],
                "next_operation": {
                    "owner": "customer",
                    "kind": "customer_response",
                    "tool_name": "",
                    "arguments": {},
                    "summary": "Provide the required input.",
                },
                "evidence": "Customer input is required.",
                "evidence_refs": [],
            }
        ]
    }

    rejected = run(session.invoke(SYNC_WORK_PLAN_TOOL, plan))
    plan["items"].append(
        {
            "id": "information-only-repair",
            "goal": "Explain the requested operation",
            "completion_mode": "information",
            "status": "awaiting_customer",
            "outcome": "unresolved",
            "depends_on": [],
            "next_operation": {
                "owner": "customer",
                "kind": "customer_response",
                "tool_name": "",
                "arguments": {},
                "summary": "Provide the required question.",
            },
            "evidence": "Customer input is required.",
            "evidence_refs": [],
        }
    )
    wrong_direction = run(session.invoke(SYNC_WORK_PLAN_TOOL, plan))
    plan["items"][0]["completion_mode"] = "operation"
    corrected = run(session.invoke(SYNC_WORK_PLAN_TOOL, plan))

    assert rejected.is_error is True
    assert "wrong_completion_mode_to_operation" in rejected.content
    assert wrong_direction.is_error is True
    assert "additional operation-mode outcome" in wrong_direction.content
    assert corrected.is_error is False
    assert len(verifier.calls) == 2


def test_semantic_reviewer_requires_missing_operation_without_a_language_floor(
    tmp_path: Path,
) -> None:
    verifier = ScriptedPlanVerifier(
        [
            PlanReviewDecision(
                accepted=False,
                correction=PlanReviewCorrection.MISSING_OPERATION_OUTCOME,
                required_new_item_count=1,
            ),
            PlanReviewDecision(accepted=True, correction=PlanReviewCorrection.NONE),
        ]
    )
    memory_runtime = runtime(tmp_path, plan_verifier=verifier)
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please help me apply for the selected product."))
    controller = BankingWorkflowController(ledger)
    session = memory_runtime.new_session(
        ledger=ledger,
        controller=controller,
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    plan = {
        "items": [
            {
                "id": "explain-selection",
                "goal": "Explain the selected product",
                "completion_mode": "information",
                "status": "pending",
                "outcome": "unresolved",
                "depends_on": [],
                "next_operation": {
                    "owner": "agent",
                    "kind": "knowledge_search",
                    "tool_name": "",
                    "arguments": {},
                    "summary": "Retrieve product information.",
                },
                "evidence": "",
                "evidence_refs": [],
            }
        ]
    }

    rejected = run(session.invoke(SYNC_WORK_PLAN_TOOL, plan))
    plan["items"].append(
        {
            "id": "apply-selection",
            "goal": "Apply for the selected product",
            "completion_mode": "operation",
            "status": "pending",
            "outcome": "unresolved",
            "depends_on": ["explain-selection"],
            "next_operation": {
                "owner": "agent",
                "kind": "knowledge_search",
                "tool_name": "",
                "arguments": {},
                "summary": "Retrieve the application procedure.",
            },
            "evidence": "",
            "evidence_refs": [],
        }
    )
    corrected = run(session.invoke(SYNC_WORK_PLAN_TOOL, plan))

    assert rejected.is_error is True
    assert "missing_operation_outcome" in rejected.content
    assert corrected.is_error is False
    assert len(verifier.calls) == 2
    assert session.export_state()["semantic_plan_verifier"] == {
        "acceptances": 1,
        "cache_hits": 0,
        "completion_tokens": 0,
        "enabled": True,
        "explicit_operation_floor_rejections": 0,
        "prompt_tokens": 0,
        "provider_calls": 2,
        "rejections": 1,
    }


def test_work_context_tracks_pending_tau_call_without_executing_it(tmp_path: Path) -> None:
    session = new_session(tmp_path)
    session.ledger.observe_input(UserMessage.text("Please look up account A-1"))
    session.ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "lookup_account",
                    "arguments": {"account_id": "A-1"},
                }
            ],
        )
    )

    run(session.prepare_context())
    pending = run(session.work_context_store.load_work_context(session.task_id))
    assert pending is not None
    assert pending.planned_action_ids == ("call-1",)
    assert pending.entity_ids == ("account_id:A-1",)
    assert pending.workflow_phase == "awaiting_environment_results"

    session.ledger.observe_input(
        ToolMessage(
            id="call-1",
            role="tool",
            requestor="assistant",
            content='{"account_id":"A-1"}',
        )
    )
    run(session.prepare_context())
    completed = run(session.work_context_store.load_work_context(session.task_id))
    assert completed is not None
    assert completed.revision == 2
    assert completed.planned_action_ids == ()
    assert completed.workflow_phase == "servicing_customer"


def test_internal_knowledge_tool_uses_cayu_recall_and_validates_arguments(tmp_path: Path) -> None:
    session = new_session(tmp_path)
    before = session.progress_token()

    result = run(session.invoke(KNOWLEDGE_SEARCH_TOOL, {"query": "Sapphire limit", "limit": 1}))
    after = session.progress_token()
    repeated = run(session.invoke(KNOWLEDGE_SEARCH_TOOL, {"query": "Sapphire limit", "limit": 1}))
    after_repeated = session.progress_token()
    invalid = run(session.invoke(KNOWLEDGE_SEARCH_TOOL, {"query": "Sapphire", "limit": True}))

    assert result.is_error is False
    assert json.loads(result.content)["documents"][0]["entry_id"] == "card_limits"
    assert repeated.is_error is False
    assert json.loads(repeated.content)["documents"] == []
    assert before != after
    assert after_repeated == after
    assert invalid.is_error is True
    assert session.export_state()["knowledge_searches"] == 2


def test_literal_navigation_never_counts_as_delivered_evidence(tmp_path: Path, monkeypatch):
    session = new_session(tmp_path)
    observed = []
    monkeypatch.setattr(session.admission_gate, "observe_knowledge_text", observed.append)
    delivered = set(session._turn_delivered_entry_ids)
    result = run(session.invoke(KNOWLEDGE_SEARCH_TOOL, {"query": "Sapphire", "mode": "literal"}))
    assert not result.is_error
    payload = json.loads(result.content)
    assert payload["navigation_only"]
    assert payload["documents"][0]["entry_id"] == "card_limits"
    assert observed == []
    assert session._turn_delivered_entry_ids == delivered
    assert session.export_state()["knowledge_searches"] == 1
    # Navigation must not cause the subsequent full-text search to omit this source.
    read = run(session.invoke(KNOWLEDGE_SEARCH_TOOL, {"query": "Sapphire", "mode": "lexical"}))
    assert not read.is_error
    assert json.loads(read.content)["documents"][0]["entry_id"] == "card_limits"
    assert observed


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_unavailable_embedding_modes_do_not_spend_search_budget(tmp_path, mode):
    session = new_session(tmp_path)
    result = run(session.invoke(KNOWLEDGE_SEARCH_TOOL, {"query": "Sapphire", "mode": mode}))
    assert result.is_error
    assert result.diagnostic_code == "embedding_search_unavailable"
    assert session.export_state()["knowledge_searches"] == 0
    result = run(
        session.invoke(
            KNOWLEDGE_SEARCH_TOOL,
            {
                "queries": [
                    {"query": "Sapphire", "mode": mode},
                    {"query": "Sapphire", "mode": "lexical"},
                ]
            },
        )
    )
    assert [item["is_error"] for item in json.loads(result.content)["batch"]] == [True, False]
    assert session.export_state()["knowledge_searches"] == 1


def test_partial_document_read_never_claims_full_delivery_or_unlocks(tmp_path, monkeypatch):
    session = new_session(tmp_path)
    observed = []
    monkeypatch.setattr(session.admission_gate, "observe_knowledge_text", observed.append)
    args = {
        "documents": [{"entry_id": "card_limits", "revision": 1}],
        "source_span": {"start": 0, "end": 12, "context_chars": 0},
    }
    result = run(session.invoke("cayu_read_product_documents", args))
    assert not result.is_error, result.content
    doc = json.loads(result.content)["documents"][0]
    assert doc["navigation_only"] and not doc["text_complete"]
    assert "evidence_ref" not in doc
    assert not session.controller.has_delivered_knowledge_revision("card_limits", "1")
    assert "card_limits" not in session._turn_delivered_entry_ids
    assert observed == []
    del args["source_span"]
    complete = run(session.invoke("cayu_read_product_documents", args))
    assert not complete.is_error
    assert session.controller.has_delivered_knowledge_revision("card_limits", "1")
    assert observed


def test_query_batch_preserves_order_errors_and_per_query_budget(tmp_path):
    session = new_session(tmp_path)
    result = run(
        session.invoke(
            KNOWLEDGE_SEARCH_TOOL,
            {
                "queries": [
                    {"query": "Sapphire", "mode": "literal"},
                    {"query": "Sapphire", "mode": "lexical"},
                    {"query": "invalid", "mode": "invalid"},
                ]
            },
        )
    )
    assert not result.is_error
    batch = json.loads(result.content)["batch"]
    assert [item["index"] for item in batch] == [0, 1, 2]
    assert [item["is_error"] for item in batch] == [False, False, True]
    assert batch[0]["result"]["returned"] == batch[1]["result"]["returned"] == 1
    assert session.export_state()["knowledge_searches"] == 2


def test_concurrent_batch_cannot_overspend_last_search_slot(tmp_path):
    session = new_session(tmp_path)
    session._knowledge_searches = _MAX_KNOWLEDGE_SEARCHES - 1
    result = run(
        session.invoke(
            KNOWLEDGE_SEARCH_TOOL,
            {
                "queries": [
                    {"query": "Sapphire", "mode": "lexical"},
                    {"query": "Sapphire", "mode": "lexical"},
                ]
            },
        )
    )
    batch = json.loads(result.content)["batch"]
    assert sum(not item["is_error"] for item in batch) == 1
    assert session._knowledge_searches == _MAX_KNOWLEDGE_SEARCHES


def test_batch_cancellation_propagates_and_drains_children(tmp_path, monkeypatch):
    async def exercise():
        session = new_session(tmp_path)
        started = asyncio.Event()
        live = set()

        async def blocked(query, **kwargs):
            task = asyncio.current_task()
            live.add(task)
            if len(live) == 2:
                started.set()
            try:
                await asyncio.Future()
            finally:
                live.remove(task)

        monkeypatch.setattr(session.corpus, "search_with_evidence", blocked)
        task = asyncio.create_task(
            session.invoke(
                KNOWLEDGE_SEARCH_TOOL,
                {
                    "queries": [
                        {"query": "one"},
                        {"query": "two"},
                    ]
                },
            )
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not live

    run(exercise())


@pytest.mark.parametrize(
    "arguments",
    [
        {"queries": []},
        {"queries": [{}] * 5},
        {"queries": [{"queries": []}]},
        {"queries": [{"query": "x", "limit": 5}]},
        {"queries": [{"query": "x"}], "query": "y"},
    ],
)
def test_query_batch_rejects_ambiguous_and_recursive_requests(tmp_path, arguments):
    session = new_session(tmp_path)
    assert run(session.invoke(KNOWLEDGE_SEARCH_TOOL, arguments)).is_error
    assert session._knowledge_searches == 0


def test_capability_directory_does_not_grant_evidence_until_document_read(tmp_path: Path) -> None:
    (tmp_path / "lookup.json").write_text(
        json.dumps(
            {
                "id": "lookup",
                "title": "Delivery lookup",
                "content": "Use read_delivery_status_1234.",
            }
        ),
        encoding="utf-8",
    )
    session = new_session(tmp_path)
    context = run(session.runtime_context())
    assert "read_delivery_status_1234" in context
    assert not session.admission_gate.knowledge_identifier_is_exposed("read_delivery_status_1234")
    result = run(
        session.invoke(
            KNOWLEDGE_SEARCH_TOOL,
            {
                "query": "read_delivery_status_1234",
                "mode": "capability",
            },
        )
    )
    assert not result.is_error
    assert json.loads(result.content)["documents"][0]["entry_id"] == "lookup"
    assert session.admission_gate.knowledge_identifier_is_exposed("read_delivery_status_1234")
    assert session.export_state()["knowledge_searches"] == 1
    repeated = run(
        session.invoke(
            KNOWLEDGE_SEARCH_TOOL,
            {
                "query": "read_delivery_status_1234",
                "mode": "capability",
            },
        )
    )
    assert not json.loads(repeated.content)["documents"]
    assert session.export_state()["knowledge_searches"] == 2


def test_internal_knowledge_tool_accepts_lexical_mode_and_shares_budget(tmp_path: Path) -> None:
    session = new_session(tmp_path)
    invalid = run(session.invoke(KNOWLEDGE_SEARCH_TOOL, {"query": "Sapphire", "mode": "bad"}))
    assert invalid.is_error
    assert "mode" in invalid.content
    assert session.export_state()["knowledge_searches"] == 0

    result = run(
        session.invoke(
            KNOWLEDGE_SEARCH_TOOL,
            {
                "query": "Sapphire limit",
                "mode": "lexical",
                "limit": 1,
            },
        )
    )
    assert not result.is_error
    assert json.loads(result.content)["retrieval"]["mode"] == "lexical"
    assert json.loads(result.content)["documents"][0]["entry_id"] == "card_limits"
    repeated = run(session.invoke(KNOWLEDGE_SEARCH_TOOL, {"query": "Sapphire limit"}))
    assert not repeated.is_error
    assert json.loads(repeated.content)["documents"] == []
    assert session.export_state()["knowledge_searches"] == 2


def test_workflow_requires_a_plan_before_internal_memory_helpers(tmp_path: Path) -> None:
    session = new_session(tmp_path, workflow_controller_enabled=True)

    blocked = run(session.invoke(KNOWLEDGE_SEARCH_TOOL, {"query": "Sapphire limit", "limit": 1}))
    planned = run(
        session.invoke(
            SYNC_WORK_PLAN_TOOL,
            {
                "items": [
                    {
                        "id": "answer-limit",
                        "goal": "Answer the Sapphire purchase-limit question",
                        "completion_mode": "information",
                        "status": "pending",
                        "outcome": "unresolved",
                        "depends_on": [],
                        "next_operation": {
                            "owner": "agent",
                            "kind": "knowledge_search",
                            "tool_name": "",
                            "arguments": {},
                            "summary": "Retrieve the relevant public policy.",
                        },
                        "evidence": "",
                        "evidence_refs": [],
                    }
                ]
            },
        )
    )
    searched = run(session.invoke(KNOWLEDGE_SEARCH_TOOL, {"query": "Sapphire limit", "limit": 1}))

    assert blocked.is_error is True
    assert "controller_sync_work_plan" in blocked.content
    assert planned.is_error is False
    assert searched.is_error is False


def test_successful_private_action_requires_plan_revision_before_reuse(tmp_path: Path) -> None:
    session = new_session(tmp_path, workflow_controller_enabled=True)
    initial_item = {
        "id": "answer-limit",
        "goal": "Answer the Sapphire purchase-limit question",
        "completion_mode": "information",
        "status": "in_progress",
        "outcome": "unresolved",
        "depends_on": [],
        "next_operation": {
            "owner": "agent",
            "kind": "knowledge_search",
            "tool_name": "",
            "arguments": {},
            "summary": "Retrieve the relevant public policy.",
        },
        "evidence": "",
        "evidence_refs": [],
    }
    planned = run(session.invoke(SYNC_WORK_PLAN_TOOL, {"items": [initial_item]}))

    assert planned.is_error is False
    assert session.required_action_tool_names() == (
        KNOWLEDGE_SEARCH_TOOL,
        UPDATE_WORK_PLAN_TOOL,
    )
    assert session.required_action_batch_size(KNOWLEDGE_SEARCH_TOOL) == 1

    searched = run(
        session.invoke(
            KNOWLEDGE_SEARCH_TOOL,
            {"query": "Sapphire purchase limit", "limit": 1},
        )
    )
    repeated = run(
        session.invoke(
            KNOWLEDGE_SEARCH_TOOL,
            {"query": "Sapphire purchase limit", "limit": 1},
        )
    )

    assert searched.is_error is False
    assert repeated.is_error is False
    assert session.required_action_tool_names() == (UPDATE_WORK_PLAN_TOOL,)
    assert session.required_action_batch_size(UPDATE_WORK_PLAN_TOOL) == 1

    revised_item = {
        **initial_item,
        "next_operation": {
            **initial_item["next_operation"],
            "summary": "Retrieve one additional relevant public policy.",
        },
        "evidence": "The first exact policy revision was retrieved.",
        "evidence_refs": [
            {
                "kind": "knowledge_revision",
                "reference_id": "card_limits",
                "version": "1",
            }
        ],
    }
    revised = run(session.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [revised_item]}))

    assert revised.is_error is False
    assert session.required_action_tool_names() == (
        KNOWLEDGE_SEARCH_TOOL,
        UPDATE_WORK_PLAN_TOOL,
    )


def test_explicit_search_omits_documents_already_in_automatic_recall(tmp_path: Path) -> None:
    session = new_session(tmp_path)
    session.ledger.observe_input(UserMessage.text("What is the Sapphire purchase limit?"))
    context = run(session.prepare_context())

    result = run(
        session.invoke(
            KNOWLEDGE_SEARCH_TOOL,
            {"query": "Sapphire purchase limit", "limit": 1},
        )
    )

    assert "<banking-knowledge-recall>" in context
    assert result.is_error is False
    assert json.loads(result.content)["documents"] == []


def test_internal_knowledge_tool_records_discoverable_name_evidence(tmp_path: Path) -> None:
    (tmp_path / "discoverable.json").write_text(
        json.dumps(
            {
                "id": "discoverable",
                "title": "Card replacement workflow",
                "content": "Unlock and call order_replacement_card_7291 for replacement.",
            }
        ),
        encoding="utf-8",
    )
    session = new_session(tmp_path)

    before = session.admission_gate.review(
        (proposed_unlock("unlock-before", "order_replacement_card_7291"),)
    )[0]
    run(session.invoke(KNOWLEDGE_SEARCH_TOOL, {"query": "replacement card", "limit": 2}))
    after = session.admission_gate.review(
        (proposed_unlock("unlock-after", "order_replacement_card_7291"),)
    )[0]

    assert before.allowed is False
    assert after.allowed is True


def test_explicit_knowledge_search_budget_stops_content_churn(tmp_path: Path) -> None:
    session = new_session(tmp_path)
    limit = session.export_state()["knowledge_search_limit"]
    per_customer_message_limit = session.export_state()[
        "knowledge_searches_per_customer_message_limit"
    ]

    results = []
    for index in range(limit):
        if index % per_customer_message_limit == 0:
            session.ledger.observe_input(UserMessage.text(f"Policy question batch {index}"))
            run(session.prepare_context())
        results.append(
            run(
                session.invoke(
                    KNOWLEDGE_SEARCH_TOOL,
                    {"query": f"Sapphire purchase limit variation {index}", "limit": 1},
                )
            )
        )
    progress_at_limit = session.progress_token()
    exhausted = run(
        session.invoke(
            KNOWLEDGE_SEARCH_TOOL,
            {"query": "one more distinct policy search", "limit": 1},
        )
    )

    assert all(result.is_error is False for result in results)
    assert limit == _MAX_KNOWLEDGE_SEARCHES == 576
    assert per_customer_message_limit == _MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE == 36
    assert exhausted.is_error is True
    assert "budget is exhausted" in exhausted.content
    assert session.export_state()["knowledge_searches"] == limit
    assert session.progress_token() == progress_at_limit
    assert KNOWLEDGE_SEARCH_TOOL not in {schema["name"] for schema in session.tool_schemas()}


def test_explicit_knowledge_search_budget_resets_only_for_a_new_customer_message(
    tmp_path: Path,
) -> None:
    session = new_session(tmp_path)
    session.ledger.observe_input(UserMessage.text("First policy question"))
    run(session.prepare_context())

    results = [
        run(
            session.invoke(
                KNOWLEDGE_SEARCH_TOOL,
                {"query": f"Sapphire purchase limit variation {index}", "limit": 1},
            )
        )
        for index in range(_MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE)
    ]
    exhausted = run(
        session.invoke(
            KNOWLEDGE_SEARCH_TOOL,
            {"query": "another policy query", "limit": 1},
        )
    )
    assert KNOWLEDGE_SEARCH_TOOL not in {schema["name"] for schema in session.tool_schemas()}
    run(session.prepare_context())
    assert KNOWLEDGE_SEARCH_TOOL not in {schema["name"] for schema in session.tool_schemas()}
    same_customer_message = run(
        session.invoke(
            KNOWLEDGE_SEARCH_TOOL,
            {"query": "another policy query", "limit": 1},
        )
    )
    session.ledger.observe_input(UserMessage.text("Here is new information"))
    run(session.prepare_context())
    assert KNOWLEDGE_SEARCH_TOOL in {schema["name"] for schema in session.tool_schemas()}
    next_customer_message = run(
        session.invoke(
            KNOWLEDGE_SEARCH_TOOL,
            {"query": "another policy query", "limit": 1},
        )
    )

    assert all(result.is_error is False for result in results)
    assert exhausted.is_error is True
    assert "latest customer message is exhausted" in exhausted.content
    assert same_customer_message.is_error is True
    assert next_customer_message.is_error is False


def test_session_identity_restores_only_from_its_public_history(tmp_path: Path) -> None:
    memory_runtime = runtime(tmp_path, plan_verifier=ScriptedPlanVerifier([]))
    original_ledger = ConversationLedger()
    original = memory_runtime.new_session(
        ledger=original_ledger,
        controller=BankingWorkflowController(original_ledger),
        admission_gate=BankingToolAdmissionGate(original_ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    history = [
        AssistantMessage.text(
            "Hello",
            raw_data={
                "cayu_tau3_memory": {
                    "knowledge_searches": 7,
                    "schema_version": 4,
                    "semantic_plan_verifier": {
                        "provider_calls": 5,
                        "acceptances": 3,
                        "rejections": 2,
                        "cache_hits": 1,
                        "prompt_tokens": 80,
                        "completion_tokens": 12,
                    },
                    "task_id": original.task_id,
                }
            },
        )
    ]

    restored_ledger = ConversationLedger.from_history(history)
    restored = memory_runtime.new_session(
        ledger=restored_ledger,
        controller=BankingWorkflowController.from_history(
            history,
            ledger=restored_ledger,
        ),
        admission_gate=BankingToolAdmissionGate(restored_ledger),
        message_history=history,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )

    assert restored.task_id == original.task_id
    assert restored.export_state()["knowledge_searches"] == 7
    assert restored.export_state()["semantic_plan_verifier"] == {
        "acceptances": 3,
        "cache_hits": 1,
        "completion_tokens": 12,
        "enabled": True,
        "explicit_operation_floor_rejections": 0,
        "prompt_tokens": 80,
        "provider_calls": 5,
        "rejections": 2,
    }


def test_current_memory_schema_restores_customer_message_search_budget(tmp_path: Path) -> None:
    memory_runtime = runtime(tmp_path)
    original_ledger = ConversationLedger()
    original_ledger.observe_input(UserMessage.text("Compare these policy details"))
    original = memory_runtime.new_session(
        ledger=original_ledger,
        controller=BankingWorkflowController(original_ledger),
        admission_gate=BankingToolAdmissionGate(original_ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=False,
    )
    run(original.prepare_context())
    for index in range(2):
        result = run(
            original.invoke(
                KNOWLEDGE_SEARCH_TOOL,
                {"query": f"policy detail {index}", "limit": 1},
            )
        )
        assert result.is_error is False
    history = [
        UserMessage.text("Compare these policy details"),
        AssistantMessage.text(
            "I am checking.",
            raw_data={"cayu_tau3_memory": original.export_state()},
        ),
    ]
    restored_ledger = ConversationLedger.from_history(history)
    restored = memory_runtime.new_session(
        ledger=restored_ledger,
        controller=BankingWorkflowController.from_history(history, ledger=restored_ledger),
        admission_gate=BankingToolAdmissionGate(restored_ledger),
        message_history=history,
        ledger_enabled=True,
        workflow_controller_enabled=False,
    )

    run(restored.prepare_context())
    state = restored.export_state()
    assert restored.task_id == original.task_id
    assert state["knowledge_searches"] == 2
    assert state["knowledge_searches_for_current_customer_message"] == 2
    for index in range(_MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE - 2):
        final_slot = run(
            restored.invoke(
                KNOWLEDGE_SEARCH_TOOL, {"query": f"remaining detail {index}", "limit": 1}
            )
        )
        assert final_slot.is_error is False
    assert KNOWLEDGE_SEARCH_TOOL not in {schema["name"] for schema in restored.tool_schemas()}


def test_current_memory_schema_restores_semantic_action_review_counters(
    tmp_path: Path,
) -> None:
    verifier = ScriptedActionVerifier([])
    memory_runtime = runtime(tmp_path, action_verifier=verifier)
    original_ledger = ConversationLedger()
    original = memory_runtime.new_session(
        ledger=original_ledger,
        controller=BankingWorkflowController(original_ledger),
        admission_gate=BankingToolAdmissionGate(original_ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    state = original.export_state()
    state["semantic_action_verifier"] = {
        "acceptances": 2,
        "cache_hits": 4,
        "completion_tokens": 13,
        "enabled": True,
        "prompt_tokens": 89,
        "provider_calls": 2,
        "rejections": 1,
    }
    history = [AssistantMessage.text("Continuing.", raw_data={"cayu_tau3_memory": state})]
    restored_ledger = ConversationLedger.from_history(history)
    restored = memory_runtime.new_session(
        ledger=restored_ledger,
        controller=BankingWorkflowController.from_history(history, ledger=restored_ledger),
        admission_gate=BankingToolAdmissionGate(restored_ledger),
        message_history=history,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )

    assert restored.export_state()["semantic_action_verifier"] == {
        "acceptances": 2,
        "cache_hits": 4,
        "completion_tokens": 13,
        "enabled": True,
        "prompt_tokens": 89,
        "provider_calls": 2,
        "rejections": 1,
    }


def test_current_memory_schema_restores_semantic_response_review_counters(
    tmp_path: Path,
) -> None:
    verifier = ScriptedResponseVerifier([])
    memory_runtime = runtime(tmp_path, response_verifier=verifier)
    original_ledger = ConversationLedger()
    original = memory_runtime.new_session(
        ledger=original_ledger,
        controller=BankingWorkflowController(original_ledger),
        admission_gate=BankingToolAdmissionGate(original_ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    state = original.export_state()
    state["semantic_response_verifier"] = {
        "acceptances": 2,
        "cache_hits": 3,
        "completion_tokens": 11,
        "enabled": True,
        "prompt_tokens": 73,
        "provider_calls": 3,
        "rejections": 1,
    }
    history = [AssistantMessage.text("Continuing.", raw_data={"cayu_tau3_memory": state})]
    restored_ledger = ConversationLedger.from_history(history)
    restored = memory_runtime.new_session(
        ledger=restored_ledger,
        controller=BankingWorkflowController.from_history(history, ledger=restored_ledger),
        admission_gate=BankingToolAdmissionGate(restored_ledger),
        message_history=history,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )

    assert restored.export_state()["semantic_response_verifier"] == {
        "acceptances": 2,
        "cache_hits": 3,
        "completion_tokens": 11,
        "enabled": True,
        "prompt_tokens": 73,
        "provider_calls": 3,
        "rejections": 1,
    }


def test_explicit_operation_floor_counter_restores_from_schema_five(tmp_path: Path) -> None:
    memory_runtime = runtime(tmp_path, plan_verifier=ScriptedPlanVerifier([]))
    original_ledger = ConversationLedger()
    original = memory_runtime.new_session(
        ledger=original_ledger,
        controller=BankingWorkflowController(original_ledger),
        admission_gate=BankingToolAdmissionGate(original_ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    history = [
        AssistantMessage.text(
            "Hello",
            raw_data={
                "cayu_tau3_memory": {
                    "knowledge_searches": 0,
                    "schema_version": 5,
                    "semantic_plan_verifier": {
                        "provider_calls": 2,
                        "acceptances": 1,
                        "rejections": 1,
                        "cache_hits": 0,
                        "prompt_tokens": 40,
                        "completion_tokens": 8,
                        "explicit_operation_floor_rejections": 1,
                    },
                    "task_id": original.task_id,
                }
            },
        )
    ]
    restored_ledger = ConversationLedger.from_history(history)
    restored = memory_runtime.new_session(
        ledger=restored_ledger,
        controller=BankingWorkflowController.from_history(history, ledger=restored_ledger),
        admission_gate=BankingToolAdmissionGate(restored_ledger),
        message_history=history,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )

    assert (
        restored.export_state()["semantic_plan_verifier"]["explicit_operation_floor_rejections"]
        == 1
    )


def test_tool_schema_combines_cayu_search_and_workflow_controller(tmp_path: Path) -> None:
    session = new_session(tmp_path, workflow_controller_enabled=True)

    assert {schema["name"] for schema in session.tool_schemas()} == {
        *NAVIGATION_TOOLS,
        KNOWLEDGE_SEARCH_TOOL,
        PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL,
        AUDIT_CREDIT_CARD_DISPUTES_TOOL,
        PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL,
        AUDIT_CREDIT_CARD_REWARDS_TOOL,
        AUDIT_SAVINGS_INTEREST_TOOL,
        PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
        "controller_sync_work_plan",
    }
    assert session.handles(AUDIT_CREDIT_CARD_REWARDS_TOOL) is True
    assert session.handles(AUDIT_SAVINGS_INTEREST_TOOL) is True
    uncontrolled = new_session(tmp_path)
    unavailable = run(
        uncontrolled.invoke(
            AUDIT_CREDIT_CARD_REWARDS_TOOL,
            {
                "user_id": "user-7",
                "policy_dossier_id": "a" * 64,
                "rates": [],
            },
        )
    )
    assert unavailable.is_error is True
    assert "No parsed successful" in unavailable.content
    savings_unavailable = run(
        uncontrolled.invoke(
            AUDIT_SAVINGS_INTEREST_TOOL,
            {
                "user_id": "user-7",
                "account_id": "savings-1",
                "policy_dossier_id": "a" * 64,
                "base_or_tier_apy_percent": 5.5,
                "checking_boosts": [],
                "credit_card_bonuses": [],
                "other_stackable_components": [],
            },
        )
    )
    assert savings_unavailable.is_error is True
    assert "Prepare this account's Cayu" in savings_unavailable.content


def test_savings_interest_policy_dossier_is_bound_to_observed_products(tmp_path: Path) -> None:
    for document_id, title, content in (
        (
            "doc_savings_accounts_gold_account_001",
            "Gold Savings Account: Complete Guide",
            "The Gold Account earns 5.5% APY.",
        ),
        (
            "doc_checking_accounts_green_account_(checking)_001",
            "Green Account at a glance",
            "The linked savings APY boost for a Gold Account is 0.75%.",
        ),
    ):
        (tmp_path / f"{document_id}.json").write_text(
            json.dumps({"id": document_id, "title": title, "content": content}),
            encoding="utf-8",
        )
    session = new_session(tmp_path)
    session.ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "accounts-1",
                    "name": "call_discoverable_agent_tool",
                    "arguments": {
                        "agent_tool_name": "get_all_user_accounts_by_user_id_3847",
                        "arguments": json.dumps({"user_id": "user-7"}),
                    },
                }
            ],
        )
    )
    session.ledger.observe_input(
        ToolMessage(
            id="accounts-1",
            role="tool",
            requestor="assistant",
            content=(
                "Bank Accounts:\n"
                "Found 2 record(s) in 'accounts':\n\n"
                "1. Record ID: savings-1\n"
                "   account_id: savings-1\n"
                "   user_id: user-7\n"
                "   class: saving\n"
                "   level: Gold Account\n"
                "   status: OPEN\n"
                "   current_holdings: 96000.00\n\n"
                "2. Record ID: checking-1\n"
                "   account_id: checking-1\n"
                "   user_id: user-7\n"
                "   class: checking\n"
                "   level: Green Account\n"
                "   status: OPEN\n"
                "   current_holdings: 8500.00\n\n"
                "Credit Card Accounts:\n"
                "No credit card accounts found."
            ),
        )
    )

    prepared = run(
        session.invoke(
            PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
            {"user_id": "user-7", "account_id": "savings-1"},
        )
    )
    dossier = json.loads(prepared.content.splitlines()[-1])
    attempted_audit = run(
        session.invoke(
            AUDIT_SAVINGS_INTEREST_TOOL,
            {
                "user_id": "user-7",
                "account_id": "savings-1",
                "policy_dossier_id": dossier["policy_dossier_id"],
                "base_or_tier_apy_percent": 5.5,
                "checking_boosts": [{"account_id": "checking-1", "apy_percent": 0.75}],
                "credit_card_bonuses": [],
                "other_stackable_components": [],
            },
        )
    )

    assert prepared.is_error is False
    assert dossier["savings_account_level"] == "Gold Account"
    assert dossier["checking_account_levels"] == ["Green Account"]
    assert {item["entry_id"] for item in dossier["documents"]} == {
        "doc_checking_accounts_green_account_(checking)_001",
        "doc_savings_accounts_gold_account_001",
    }
    assert attempted_audit.is_error is True
    assert "No parsed successful transaction lookup" in attempted_audit.content
    assert session.export_state()["savings_interest_policy_dossiers"] == 1


def test_savings_audit_authorizes_only_exact_ordered_environment_actions(
    tmp_path: Path,
) -> None:
    (tmp_path / "doc_savings_accounts_gold_account_001.json").write_text(
        json.dumps(
            {
                "id": "doc_savings_accounts_gold_account_001",
                "title": "Gold Savings Account: Complete Guide",
                "content": "The Gold Account earns 5.5% APY.",
            }
        ),
        encoding="utf-8",
    )
    session = new_session(tmp_path)
    session.ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "accounts-1",
                    "name": "call_discoverable_agent_tool",
                    "arguments": {
                        "agent_tool_name": "get_all_user_accounts_by_user_id_3847",
                        "arguments": json.dumps({"user_id": "user-7"}),
                    },
                }
            ],
        )
    )
    session.ledger.observe_input(
        ToolMessage(
            id="accounts-1",
            role="tool",
            requestor="assistant",
            content=(
                "Bank Accounts:\n"
                "Found 1 record(s) in 'accounts':\n\n"
                "1. Record ID: savings-1\n"
                "   account_id: savings-1\n"
                "   user_id: user-7\n"
                "   class: saving\n"
                "   level: Gold Account\n"
                "   status: OPEN\n"
                "   current_holdings: 96000.00\n\n"
                "Credit Card Accounts:\n"
                "No credit card accounts found."
            ),
        )
    )
    session.ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "transactions-1",
                    "name": "call_discoverable_agent_tool",
                    "arguments": {
                        "agent_tool_name": "get_bank_account_transactions_9173",
                        "arguments": json.dumps({"account_id": "savings-1"}),
                    },
                }
            ],
        )
    )
    session.ledger.observe_input(
        ToolMessage(
            id="transactions-1",
            role="tool",
            requestor="assistant",
            content=(
                "Transactions for account savings-1:\n"
                "Found 1 record(s) in 'bank_account_transaction_history':\n\n"
                "1. Record ID: interest-1\n"
                "   transaction_id: interest-1\n"
                "   account_id: savings-1\n"
                "   date: 10/31/2025\n"
                "   description: MONTHLY INTEREST CREDIT\n"
                "   amount: 400.00\n"
                "   type: interest_credit\n"
                "   status: posted"
            ),
        )
    )
    prepared = run(
        session.invoke(
            PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
            {"user_id": "user-7", "account_id": "savings-1"},
        )
    )
    dossier = json.loads(prepared.content.splitlines()[-1])
    audited = run(
        session.invoke(
            AUDIT_SAVINGS_INTEREST_TOOL,
            {
                "user_id": "user-7",
                "account_id": "savings-1",
                "policy_dossier_id": dossier["policy_dossier_id"],
                "base_or_tier_apy_percent": 5.5,
                "checking_boosts": [],
                "credit_card_bonuses": [],
                "other_stackable_components": [],
            },
        )
    )

    assert audited.is_error is False
    context = run(session.prepare_context())
    authorization_text = context.split("<savings-interest-action-authorizations>\n", 1)[1].split(
        "\n</savings-interest-action-authorizations>", 1
    )[0]
    authorizations = json.loads(authorization_text.splitlines()[-1])
    credit_arguments = {
        "agent_tool_name": "apply_savings_account_credit_6831",
        "arguments": (
            '{"account_id": "savings-1", "amount": 40.00, "credit_type": "interest_correction"}'
        ),
    }
    report_arguments = {
        "agent_tool_name": "submit_interest_discrepancy_report_7294",
        "arguments": (
            '{"account_id": "savings-1", "user_id": "user-7", '
            '"expected_apy": 5.5, "actual_apy": 5.0, "amount_difference": 40.00}'
        ),
    }
    assert authorizations["actions"] == [
        {
            "account_id": "savings-1",
            "credit_call": credit_arguments,
            "report_call": report_arguments,
        }
    ]

    wrong_credit = ToolCallPart(
        tool_call_id="wrong-credit",
        tool_name="call_discoverable_agent_tool",
        arguments={
            **credit_arguments,
            "arguments": credit_arguments["arguments"].replace("40.00", "41.00"),
        },
    )
    correct_credit = ToolCallPart(
        tool_call_id="credit-1",
        tool_name="call_discoverable_agent_tool",
        arguments=credit_arguments,
    )
    premature_report = ToolCallPart(
        tool_call_id="report-early",
        tool_name="call_discoverable_agent_tool",
        arguments=report_arguments,
    )
    wrong_report = ToolCallPart(
        tool_call_id="report-wrong",
        tool_name="call_discoverable_agent_tool",
        arguments={
            **report_arguments,
            "arguments": report_arguments["arguments"].replace("5.0", "5.0001"),
        },
    )

    assert session.review([wrong_credit])[0].allowed is False
    assert "copy this exact wrapper payload" in session.review([wrong_credit])[0].reason
    assert session.review([wrong_report])[0].allowed is False
    assert "copy this exact wrapper payload" in session.review([wrong_report])[0].reason
    assert session.review([correct_credit])[0].allowed is True
    assert session.review([premature_report])[0].allowed is False
    assert "only after" in session.review([premature_report])[0].reason
    duplicate_credit = ToolCallPart(
        tool_call_id="credit-duplicate",
        tool_name="call_discoverable_agent_tool",
        arguments=credit_arguments,
    )
    duplicate_batch = session.review([correct_credit, duplicate_credit])
    assert duplicate_batch[0].allowed is True
    assert duplicate_batch[1].allowed is False
    assert "more than once" in duplicate_batch[1].reason

    session.ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "credit-1",
                    "name": "call_discoverable_agent_tool",
                    "arguments": credit_arguments,
                }
            ],
        )
    )
    session.ledger.observe_input(
        ToolMessage(
            id="credit-1",
            role="tool",
            requestor="assistant",
            content='{"status":"success"}',
        )
    )

    assert session.review([correct_credit])[0].allowed is False
    assert "already succeeded" in session.review([correct_credit])[0].reason
    assert session.review([premature_report])[0].allowed is True


def test_credit_card_dispute_audit_is_integrated_with_memory_and_admission(
    tmp_path: Path,
) -> None:
    write_credit_card_dispute_policies(tmp_path)
    session = new_session(tmp_path)
    observations = (
        (
            "time-1",
            "get_current_time",
            {},
            "The current time is 2025-11-14 03:40:00 EST.",
        ),
        (
            "transactions-1",
            "get_credit_card_transactions_by_user",
            {"user_id": "user-7"},
            "Found 1 record(s) in 'credit_card_transaction_history':\n\n"
            "1. Record ID: txn-1\n"
            "   transaction_id: txn-1\n"
            "   user_id: user-7\n"
            "   credit_card_type: Gold Rewards Card\n"
            "   merchant_name: Example Air\n"
            "   transaction_amount: $126.77\n"
            "   transaction_date: 09/10/2025\n"
            "   category: Travel\n"
            "   status: COMPLETED\n"
            "   rewards_earned: 0 points",
        ),
        (
            "accounts-1",
            "get_credit_card_accounts_by_user",
            {"user_id": "user-7"},
            "Found 1 record(s) in 'credit_card_accounts':\n\n"
            "1. Record ID: card-1\n"
            "   account_id: card-1\n"
            "   user_id: user-7\n"
            "   card_type: Gold Rewards Card\n"
            "   date_of_account_open: 03/15/2024",
        ),
    )
    for call_id, tool_name, arguments, content in observations:
        session.ledger.observe_assistant(
            AssistantMessage(
                role="assistant",
                tool_calls=[{"id": call_id, "name": tool_name, "arguments": arguments}],
            )
        )
        session.ledger.observe_input(
            ToolMessage(
                id=call_id,
                role="tool",
                requestor="assistant",
                content=content,
            )
        )
    session.ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "history-1",
                    "name": "call_discoverable_agent_tool",
                    "arguments": {
                        "agent_tool_name": "get_user_dispute_history_7291",
                        "arguments": json.dumps({"user_id": "user-7"}),
                    },
                }
            ],
        )
    )
    session.ledger.observe_input(
        ToolMessage(
            id="history-1",
            role="tool",
            requestor="assistant",
            content=(
                "User transaction dispute history retrieved successfully.\n\n"
                "Executed: get_user_dispute_history_7291\n"
                "Transaction dispute history for user user-7:\n"
                "No transaction disputes found for this user."
            ),
        )
    )

    assert PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL in {
        schema["name"] for schema in session.tool_schemas()
    }
    prepared = run(session.invoke(PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL, {"user_id": "user-7"}))
    dossier = json.loads(prepared.content.splitlines()[-1])
    assert PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL not in {
        schema["name"] for schema in session.tool_schemas()
    }
    prepared_context = run(session.prepare_context())
    assert "<prepared-policy-memory>" in prepared_context
    assert dossier["policy_dossier_id"] in prepared_context
    assert PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL not in {
        schema["name"] for schema in session.tool_schemas()
    }
    audit_arguments = {
        "user_id": "user-7",
        "policy_dossier_id": dossier["policy_dossier_id"],
        "priority_transaction_ids": [],
        "disputes": [
            {
                "transaction_id": "txn-1",
                "dispute_reason": "unauthorized_fraudulent_charge",
                "contacted_merchant": False,
                "issue_noticed_date": "11/14/2025",
            }
        ],
    }
    audited = run(session.invoke(AUDIT_CREDIT_CARD_DISPUTES_TOOL, audit_arguments))
    audit = json.loads(audited.content.splitlines()[-1])
    filing_arguments = {
        **audit["audited_disputes"][0],
        "card_action": "keep_active",
        "card_last_4_digits": "1234",
        "full_name": "Example User",
        "phone": "555-0100",
        "email": "user@example.com",
        "address": "1 Main St",
        "resolution_requested": "full_refund",
    }
    filing_call = ToolCallPart(
        tool_call_id="filing-1",
        tool_name="call_discoverable_agent_tool",
        arguments={
            "agent_tool_name": FILE_CREDIT_CARD_DISPUTE_TOOL,
            "arguments": json.dumps(filing_arguments),
        },
    )

    assert prepared.is_error is False
    assert audited.is_error is False
    assert AUDIT_CREDIT_CARD_DISPUTES_TOOL not in {
        schema["name"] for schema in session.tool_schemas()
    }
    assert session.review([filing_call])[0].allowed is True
    assert "<credit-card-dispute-authorizations>" in run(session.prepare_context())
    assert session.export_state()["credit_card_dispute_policy_dossiers"] == 1
    assert session.export_state()["credit_card_dispute_audits"] == 1

    session.ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "filing-1",
                    "name": "call_discoverable_agent_tool",
                    "arguments": filing_call.arguments,
                }
            ],
        )
    )
    session.ledger.observe_input(
        ToolMessage(
            id="filing-1",
            role="tool",
            requestor="assistant",
            content="Credit card transaction dispute filed successfully.",
        )
    )

    run(session.prepare_context())
    assert PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL in {
        schema["name"] for schema in session.tool_schemas()
    }

    stale = run(session.invoke(AUDIT_CREDIT_CARD_DISPUTES_TOOL, audit_arguments))
    assert stale.is_error is True
    assert "Prepare the current" in stale.content


@pytest.mark.parametrize("advisory", [False, True])
def test_reward_audit_requires_and_uses_exact_cayu_policy_dossier(
    tmp_path: Path, advisory: bool
) -> None:
    for document_id, title, content in (
        (
            "silver_rates",
            "Silver Rewards Card: How to Earn 4% Cash Back",
            "Earn 4% on travel and software and 1% on other purchases.",
        ),
        (
            "reward_rounding",
            "Credit Card Rewards Points Rounding Policy",
            "All reward points are rounded down to a whole point.",
        ),
        (
            "reward_representation",
            "Internal: Credit Card Rewards - Points vs Cash Back Representation",
            "One reward point represents one cent of cash back.",
        ),
    ):
        (tmp_path / f"{document_id}.json").write_text(
            json.dumps({"id": document_id, "title": title, "content": content}),
            encoding="utf-8",
        )
    session = new_session(tmp_path)
    session.advisory_workflow = advisory
    session.ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "transactions-1",
                    "name": "get_credit_card_transactions_by_user",
                    "arguments": {"user_id": "user-7"},
                }
            ],
        )
    )
    session.ledger.observe_input(
        ToolMessage(
            id="transactions-1",
            role="tool",
            requestor="assistant",
            content=(
                "Found 1 record(s) in 'credit_card_transaction_history':\n\n"
                "1. Record ID: txn-1\n"
                "   transaction_id: txn-1\n"
                "   user_id: user-7\n"
                "   credit_card_type: Silver Rewards Card\n"
                "   merchant_name: Example Air\n"
                "   transaction_amount: $126.77\n"
                "   transaction_date: 09/10/2025\n"
                "   category: Travel\n"
                "   status: COMPLETED\n"
                "   rewards_earned: 507 points"
            ),
        )
    )

    before = run(
        session.invoke(
            AUDIT_CREDIT_CARD_REWARDS_TOOL,
            {
                "user_id": "user-7",
                "policy_dossier_id": "a" * 64,
                "rates": [{"transaction_id": "txn-1", "cash_back_percent": 4}],
            },
        )
    )
    prepared = run(session.invoke(PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL, {"user_id": "user-7"}))
    dossier = json.loads(prepared.content.splitlines()[-1])
    if advisory:
        assert all("text" not in doc for doc in dossier["documents"])
        assert all("evidence_ref" not in doc for doc in dossier["documents"])
        assert not session.controller.has_delivered_knowledge_revision("silver_rates", "1")
        assert all(not doc["text_complete"] for doc in dossier["documents"])
    audited = run(
        session.invoke(
            AUDIT_CREDIT_CARD_REWARDS_TOOL,
            {
                "user_id": "user-7",
                "policy_dossier_id": dossier["policy_dossier_id"],
                "rates": [{"transaction_id": "txn-1", "cash_back_percent": 4}],
            },
        )
    )

    assert before.is_error is True
    assert "Prepare this user's Cayu reward-policy dossier" in before.content
    assert prepared.is_error is False
    assert {item["entry_id"] for item in dossier["documents"]} == {
        "reward_representation",
        "reward_rounding",
        "silver_rates",
    }
    assert audited.is_error is False
    assert session.export_state()["reward_policy_dossiers"] == 1


def proposed_unlock(call_id: str, tool_name: str) -> ToolCallPart:
    return ToolCallPart(
        tool_call_id=call_id,
        tool_name="unlock_discoverable_agent_tool",
        arguments={"agent_tool_name": tool_name},
    )


def test_batch_limit_defaults_and_overrides_match_single_search(tmp_path, monkeypatch):
    session = new_session(tmp_path)
    seen = []
    original = session.corpus.search_with_evidence

    async def observe(query, **kwargs):
        seen.append(kwargs["limit"])
        return await original(query, **kwargs)

    monkeypatch.setattr(session.corpus, "search_with_evidence", observe)
    args = {
        "limit": 4,
        "queries": [
            {"query": "Sapphire", "mode": "lexical"},
            {"query": "Sapphire", "mode": "lexical", "limit": 2},
        ],
    }
    from jsonschema import Draft202012Validator

    assert Draft202012Validator(session.corpus.search_tool_schema()["input_schema"]).is_valid(args)
    result = run(session.invoke(KNOWLEDGE_SEARCH_TOOL, args))
    assert not result.is_error, result.content
    assert sorted(seen) == [2, 4]
    assert session._knowledge_searches == 2


def test_scoped_batch_dispatch_and_invalid_scope_refund(tmp_path):
    session = new_session(tmp_path)
    result = run(session.invoke(KNOWLEDGE_SEARCH_TOOL, {'queries': [
        {'query': 'Sapphire', 'mode': 'scoped', 'scope': {'entry_ids': ['card_limits']}},
        {'query': 'Sapphire', 'mode': 'scoped', 'scope': {'entry_ids': ['missing']}},
    ]}))
    batch = json.loads(result.content)['batch']
    assert [row['is_error'] for row in batch] == [False, True]
    assert session.export_state()['knowledge_searches'] == 1
    assert batch[0]['result']['documents'][0]['entry_id'] == 'card_limits'
    assert 'card_limits' in session._turn_delivered_entry_ids
