"""Per-conversation Cayu memory orchestration for the tau3 banking agent."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import uuid4

from cayu import (
    AgentWorkContext,
    AgentWorkContextConflict,
    AgentWorkContextStore,
    InMemoryAgentWorkContextStore,
    ToolCallPart,
    WorkEvidenceReference,
)
from tau2.data_model.message import AssistantMessage, Message

from cayu_tau3_banking.action_verifier import (
    ActionCallReview,
    ActionReviewBatch,
    SemanticActionVerifier,
)
from cayu_tau3_banking.admission import BankingToolAdmissionGate
from cayu_tau3_banking.bridge import (
    ExternalToolAdmission,
    InternalToolResult,
    TerminalOutputAdmission,
)
from cayu_tau3_banking.controller import (
    SYNC_WORK_PLAN_TOOL,
    UPDATE_WORK_PLAN_TOOL,
    BankingWorkflowController,
    WorkCompletionMode,
    WorkStatus,
)
from cayu_tau3_banking.credit_card_dispute import (
    AUDIT_CREDIT_CARD_DISPUTES_TOOL,
    PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL,
    CreditCardDisputeActionGate,
    CreditCardDisputePolicy,
    compile_dispute_policy_dossier,
    dispute_audit_tool_schema,
    dispute_policy_dossier_tool_schema,
    invoke_dispute_audit,
)
from cayu_tau3_banking.customer_intent import customer_requests_human_assistance
from cayu_tau3_banking.knowledge import (
    KNOWLEDGE_SEARCH_TOOL,
    BankingKnowledgeCorpus,
    BankingKnowledgeError,
    BankingRecall,
)
from cayu_tau3_banking.ledger import (
    ConversationLedger,
    CreditCardTransactionObservation,
    ToolActionStatus,
)
from cayu_tau3_banking.plan_verifier import (
    PlanReviewCorrection,
    PlanReviewDecision,
    SemanticPlanVerifier,
)
from cayu_tau3_banking.product_navigation import (
    BROWSE_TOOL,
    COMPARISON_TOOL,
    MAX_READ_RESULT_BYTES,
    NAVIGATION_TOOLS,
    READ_TOOL,
    RECOVERY_LANE,
    ProductNavigation,
)
from cayu_tau3_banking.product_navigation import (
    tool_schemas as navigation_tool_schemas,
)
from cayu_tau3_banking.response_verifier import (
    ResponseReviewCorrection,
    ResponseReviewDecision,
    SemanticResponseVerifier,
)
from cayu_tau3_banking.reward_audit import (
    AUDIT_CREDIT_CARD_REWARDS_TOOL,
    PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL,
    invoke_reward_audit,
    reward_audit_tool_schema,
    reward_policy_dossier_tool_schema,
)
from cayu_tau3_banking.reward_policy_view import progressive_reward_policy_view
from cayu_tau3_banking.savings_interest_actions import SavingsInterestActionGate
from cayu_tau3_banking.savings_interest_audit import (
    AUDIT_SAVINGS_INTEREST_TOOL,
    PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
    invoke_savings_interest_audit,
    savings_interest_audit_tool_schema,
    savings_interest_policy_dossier_tool_schema,
)

_TASK_ID_PATTERN = re.compile(r"^tau3-banking-[0-9a-f]{32}$")
_MAX_GOAL_CHARS = 3_500
_MAX_RETRIEVAL_QUERY_CHARS = 4_000
_MAX_ENTITY_IDS = 64
_MAX_ENTITY_ID_CHARS = 240
_MAX_RECENT_CONVERSATION = 4
_MAX_CONTEXT_PUBLICATION_ATTEMPTS = 3
_MAX_REWARD_USER_ID_CHARS = 128
# Keep one finite conversation-wide search ceiling independent of the larger
# customer-evidence retention window.
_MAX_KNOWLEDGE_SEARCHES = 576
_MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE = 36
_MAX_PLAN_REVIEW_CACHE_ENTRIES = 16
_MAX_ACTION_REVIEW_CACHE_ENTRIES = 64
_MAX_RESPONSE_REVIEW_CACHE_ENTRIES = 64
_MAX_REVIEWED_ACTION_CALLS_PER_BATCH = 24
_CUSTOMER_MESSAGE_ID_PATTERN = re.compile(r"^customer-message:[1-9][0-9]*$")
_CREDIT_CARD_DISPUTE_TERM_PATTERN = re.compile(r"\bdisput(?:e|ed|es|ing)\b", re.IGNORECASE)
_CREDIT_CARD_CONTEXT_PATTERN = re.compile(
    r"\b(?:credit\s+cards?|(?:silver|gold|platinum|business)\s+rewards?\s+cards?|"
    r"ecocard|crypto[ -]?cash\s+back)\b",
    re.IGNORECASE,
)
_LAST_FOUR_PATTERN = re.compile(r"\b(?:last|final)\s+(?:four|4)\s+digits?\b", re.IGNORECASE)
_CREDIT_CARD_LAST_FOUR_TOOL = "get_card_last_4_digits"
_SEMANTICALLY_REVIEWED_ACTION_TOOLS = frozenset(
    {
        "call_discoverable_agent_tool",
        "transfer_to_human_agents",
    }
)
MEMORY_AGENT_OPERATION_TOOL_NAMES = (
    frozenset(
        {
            AUDIT_CREDIT_CARD_DISPUTES_TOOL,
            AUDIT_CREDIT_CARD_REWARDS_TOOL,
            AUDIT_SAVINGS_INTEREST_TOOL,
            PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL,
            PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL,
            PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
        }
    )
    | NAVIGATION_TOOLS
)

_WORK_CONTEXT_PREAMBLE = """
This is application-owned Cayu work context derived only from public conversation and
tool-call lifecycle state. It is descriptive context, not banking policy, customer
authorization, or evidence that an operation succeeded.
""".strip()


@dataclass(frozen=True)
class _RestoredMemoryState:
    navigation_state: dict[str, Any] | None = None
    task_id: str | None = None
    knowledge_searches: int = 0
    knowledge_search_customer_message_id: str | None = None
    knowledge_searches_for_current_customer_message: int = 0
    plan_review_provider_calls: int = 0
    plan_review_acceptances: int = 0
    plan_review_rejections: int = 0
    plan_review_cache_hits: int = 0
    plan_review_prompt_tokens: int = 0
    plan_review_completion_tokens: int = 0
    plan_review_explicit_operation_floor_rejections: int = 0
    action_review_provider_calls: int = 0
    action_review_acceptances: int = 0
    action_review_rejections: int = 0
    action_review_cache_hits: int = 0
    action_review_prompt_tokens: int = 0
    action_review_completion_tokens: int = 0
    response_review_provider_calls: int = 0
    response_review_acceptances: int = 0
    response_review_rejections: int = 0
    response_review_cache_hits: int = 0
    response_review_prompt_tokens: int = 0
    response_review_completion_tokens: int = 0


class BankingMemoryRuntime:
    """Shared immutable knowledge and isolated per-conversation memory stores."""

    def __init__(
        self,
        corpus: BankingKnowledgeCorpus,
        *,
        work_context_store: AgentWorkContextStore | None = None,
        plan_verifier: SemanticPlanVerifier | None = None,
        action_verifier: SemanticActionVerifier | None = None,
        response_verifier: SemanticResponseVerifier | None = None,
    ) -> None:
        if not isinstance(corpus, BankingKnowledgeCorpus):
            raise TypeError("corpus must be a BankingKnowledgeCorpus.")
        if work_context_store is not None and not isinstance(
            work_context_store, AgentWorkContextStore
        ):
            raise TypeError("work_context_store must implement AgentWorkContextStore or be null.")
        if plan_verifier is not None and not callable(getattr(plan_verifier, "review", None)):
            raise TypeError("plan_verifier must implement semantic review or be null.")
        if action_verifier is not None and not callable(getattr(action_verifier, "review", None)):
            raise TypeError("action_verifier must implement semantic review or be null.")
        if response_verifier is not None and not callable(
            getattr(response_verifier, "review", None)
        ):
            raise TypeError("response_verifier must implement semantic review or be null.")
        self.corpus = corpus
        self.work_context_store = (
            InMemoryAgentWorkContextStore() if work_context_store is None else work_context_store
        )
        self.plan_verifier = plan_verifier
        self.action_verifier = action_verifier
        self.response_verifier = response_verifier

    def new_session(
        self,
        *,
        ledger: ConversationLedger,
        controller: BankingWorkflowController,
        admission_gate: BankingToolAdmissionGate,
        message_history: list[Message] | None,
        ledger_enabled: bool,
        workflow_controller_enabled: bool,
    ) -> BankingMemorySession:
        if not isinstance(ledger, ConversationLedger):
            raise TypeError("ledger must be a ConversationLedger.")
        if not isinstance(controller, BankingWorkflowController):
            raise TypeError("controller must be a BankingWorkflowController.")
        if not isinstance(admission_gate, BankingToolAdmissionGate):
            raise TypeError("admission_gate must be a BankingToolAdmissionGate.")
        if type(ledger_enabled) is not bool:
            raise TypeError("ledger_enabled must be a boolean.")
        if type(workflow_controller_enabled) is not bool:
            raise TypeError("workflow_controller_enabled must be a boolean.")
        restored = _restored_memory_state(message_history)
        return BankingMemorySession(
            corpus=self.corpus,
            work_context_store=self.work_context_store,
            ledger=ledger,
            controller=controller,
            admission_gate=admission_gate,
            task_id=restored.task_id or f"tau3-banking-{uuid4().hex}",
            initial_knowledge_searches=restored.knowledge_searches,
            initial_navigation_state=restored.navigation_state,
            initial_knowledge_search_customer_message_id=(
                restored.knowledge_search_customer_message_id
            ),
            initial_knowledge_searches_for_current_customer_message=(
                restored.knowledge_searches_for_current_customer_message
            ),
            initial_plan_review_provider_calls=restored.plan_review_provider_calls,
            initial_plan_review_acceptances=restored.plan_review_acceptances,
            initial_plan_review_rejections=restored.plan_review_rejections,
            initial_plan_review_cache_hits=restored.plan_review_cache_hits,
            initial_plan_review_prompt_tokens=restored.plan_review_prompt_tokens,
            initial_plan_review_completion_tokens=restored.plan_review_completion_tokens,
            initial_plan_review_explicit_operation_floor_rejections=(
                restored.plan_review_explicit_operation_floor_rejections
            ),
            initial_action_review_provider_calls=restored.action_review_provider_calls,
            initial_action_review_acceptances=restored.action_review_acceptances,
            initial_action_review_rejections=restored.action_review_rejections,
            initial_action_review_cache_hits=restored.action_review_cache_hits,
            initial_action_review_prompt_tokens=restored.action_review_prompt_tokens,
            initial_action_review_completion_tokens=restored.action_review_completion_tokens,
            initial_response_review_provider_calls=restored.response_review_provider_calls,
            initial_response_review_acceptances=restored.response_review_acceptances,
            initial_response_review_rejections=restored.response_review_rejections,
            initial_response_review_cache_hits=restored.response_review_cache_hits,
            initial_response_review_prompt_tokens=restored.response_review_prompt_tokens,
            initial_response_review_completion_tokens=(restored.response_review_completion_tokens),
            plan_verifier=self.plan_verifier,
            action_verifier=self.action_verifier,
            response_verifier=self.response_verifier,
            ledger_enabled=ledger_enabled,
            workflow_controller_enabled=workflow_controller_enabled,
        )


class BankingMemorySession:
    """Cayu recall, work context, and internal tools for one tau simulation."""

    def __init__(
        self,
        *,
        corpus: BankingKnowledgeCorpus,
        work_context_store: AgentWorkContextStore,
        ledger: ConversationLedger,
        controller: BankingWorkflowController,
        admission_gate: BankingToolAdmissionGate,
        task_id: str,
        initial_knowledge_searches: int,
        initial_navigation_state: dict[str, Any] | None = None,
        initial_knowledge_search_customer_message_id: str | None,
        initial_knowledge_searches_for_current_customer_message: int,
        initial_plan_review_provider_calls: int,
        initial_plan_review_acceptances: int,
        initial_plan_review_rejections: int,
        initial_plan_review_cache_hits: int,
        initial_plan_review_prompt_tokens: int,
        initial_plan_review_completion_tokens: int,
        initial_plan_review_explicit_operation_floor_rejections: int,
        initial_action_review_provider_calls: int,
        initial_action_review_acceptances: int,
        initial_action_review_rejections: int,
        initial_action_review_cache_hits: int,
        initial_action_review_prompt_tokens: int,
        initial_action_review_completion_tokens: int,
        initial_response_review_provider_calls: int,
        initial_response_review_acceptances: int,
        initial_response_review_rejections: int,
        initial_response_review_cache_hits: int,
        initial_response_review_prompt_tokens: int,
        initial_response_review_completion_tokens: int,
        plan_verifier: SemanticPlanVerifier | None,
        action_verifier: SemanticActionVerifier | None,
        response_verifier: SemanticResponseVerifier | None,
        ledger_enabled: bool,
        workflow_controller_enabled: bool,
    ) -> None:
        if not _TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError("task_id is not a Cayu tau3 banking task identity.")
        if (
            type(initial_knowledge_searches) is not int
            or not 0 <= initial_knowledge_searches <= _MAX_KNOWLEDGE_SEARCHES
        ):
            raise ValueError(
                "initial_knowledge_searches must be within the conversation search budget."
            )
        if initial_knowledge_search_customer_message_id is not None and (
            not isinstance(initial_knowledge_search_customer_message_id, str)
            or not _CUSTOMER_MESSAGE_ID_PATTERN.fullmatch(
                initial_knowledge_search_customer_message_id
            )
        ):
            raise ValueError("initial knowledge-search customer-message ID is invalid.")
        if (
            type(initial_knowledge_searches_for_current_customer_message) is not int
            or not 0
            <= initial_knowledge_searches_for_current_customer_message
            <= _MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE
            or initial_knowledge_searches_for_current_customer_message > initial_knowledge_searches
        ):
            raise ValueError("initial per-customer-message search state is invalid.")
        self.corpus = corpus
        self.work_context_store = work_context_store
        self.navigation = ProductNavigation(corpus, initial_navigation_state)
        self.ledger = ledger
        self.controller = controller
        self.admission_gate = admission_gate
        self.task_id = task_id
        self.retrieval_capture = None
        self.ledger_enabled = ledger_enabled
        self.workflow_controller_enabled = workflow_controller_enabled
        self.advisory_workflow = False
        self._work_context: AgentWorkContext | None = None
        self._last_recall: BankingRecall | None = None
        self._knowledge_searches = initial_knowledge_searches
        latest_customer_message_id = (
            self.ledger.customer_message_ids[-1] if self.ledger.customer_message_ids else None
        )
        if initial_knowledge_search_customer_message_id == latest_customer_message_id:
            self._customer_message_knowledge_searches = (
                initial_knowledge_searches_for_current_customer_message
            )
        else:
            self._customer_message_knowledge_searches = 0
        self._knowledge_search_customer_message_id = latest_customer_message_id
        self._turn_delivered_entry_ids: set[str] = set()
        self._human_transfer_policy_injected_this_turn = False
        self._product_comparison_documents_injected_this_turn = 0
        self._credit_card_dispute_policies: dict[
            str,
            tuple[CreditCardDisputePolicy, str],
        ] = {}
        self._credit_card_dispute_audits: dict[str, str] = {}
        self._credit_card_dispute_action_gate = CreditCardDisputeActionGate(self.ledger)
        self._reward_policy_dossiers: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._savings_interest_policy_dossiers: dict[
            tuple[str, str],
            tuple[str, str],
        ] = {}
        self._savings_interest_audits: dict[str, str] = {}
        self._savings_interest_action_gate = SavingsInterestActionGate(self.ledger)
        self._private_actions_completed_at_plan_revision: set[str] = set()
        self._private_action_plan_revision = self.controller.revision
        self._plan_verifier = plan_verifier
        self._action_verifier = action_verifier
        self._response_verifier = response_verifier
        self._plan_review_cache: dict[str, PlanReviewDecision] = {}
        self._action_review_cache: dict[str, tuple[ActionCallReview, ...]] = {}
        action_restored_counts = {
            "initial_action_review_provider_calls": initial_action_review_provider_calls,
            "initial_action_review_acceptances": initial_action_review_acceptances,
            "initial_action_review_rejections": initial_action_review_rejections,
            "initial_action_review_cache_hits": initial_action_review_cache_hits,
            "initial_action_review_prompt_tokens": initial_action_review_prompt_tokens,
            "initial_action_review_completion_tokens": initial_action_review_completion_tokens,
        }
        if any(type(value) is not int or value < 0 for value in action_restored_counts.values()):
            raise ValueError("initial action-review counters must be nonnegative integers.")
        if initial_action_review_acceptances + initial_action_review_rejections > (
            initial_action_review_provider_calls * _MAX_REVIEWED_ACTION_CALLS_PER_BATCH
        ):
            raise ValueError("initial action-review decisions exceed the provider-call bound.")
        self._action_review_provider_calls = initial_action_review_provider_calls
        self._action_review_acceptances = initial_action_review_acceptances
        self._action_review_rejections = initial_action_review_rejections
        self._action_review_cache_hits = initial_action_review_cache_hits
        self._action_review_prompt_tokens = initial_action_review_prompt_tokens
        self._action_review_completion_tokens = initial_action_review_completion_tokens
        self._response_review_cache: dict[str, ResponseReviewDecision] = {}
        response_restored_counts = {
            "initial_response_review_provider_calls": initial_response_review_provider_calls,
            "initial_response_review_acceptances": initial_response_review_acceptances,
            "initial_response_review_rejections": initial_response_review_rejections,
            "initial_response_review_cache_hits": initial_response_review_cache_hits,
            "initial_response_review_prompt_tokens": initial_response_review_prompt_tokens,
            "initial_response_review_completion_tokens": (
                initial_response_review_completion_tokens
            ),
        }
        if any(type(value) is not int or value < 0 for value in response_restored_counts.values()):
            raise ValueError("initial response-review counters must be nonnegative integers.")
        if initial_response_review_acceptances + initial_response_review_rejections > (
            initial_response_review_provider_calls
        ):
            raise ValueError("initial response-review decisions exceed provider calls.")
        self._response_review_provider_calls = initial_response_review_provider_calls
        self._response_review_acceptances = initial_response_review_acceptances
        self._response_review_rejections = initial_response_review_rejections
        self._response_review_cache_hits = initial_response_review_cache_hits
        self._response_review_prompt_tokens = initial_response_review_prompt_tokens
        self._response_review_completion_tokens = initial_response_review_completion_tokens
        restored_counts = {
            "initial_plan_review_provider_calls": initial_plan_review_provider_calls,
            "initial_plan_review_acceptances": initial_plan_review_acceptances,
            "initial_plan_review_rejections": initial_plan_review_rejections,
            "initial_plan_review_cache_hits": initial_plan_review_cache_hits,
            "initial_plan_review_prompt_tokens": initial_plan_review_prompt_tokens,
            "initial_plan_review_completion_tokens": initial_plan_review_completion_tokens,
            "initial_plan_review_explicit_operation_floor_rejections": (
                initial_plan_review_explicit_operation_floor_rejections
            ),
        }
        if any(type(value) is not int or value < 0 for value in restored_counts.values()):
            raise ValueError("initial plan-review counters must be nonnegative integers.")
        if initial_plan_review_acceptances + initial_plan_review_rejections > (
            initial_plan_review_provider_calls
        ):
            raise ValueError("initial plan-review decisions cannot exceed provider calls.")
        self._plan_review_provider_calls = initial_plan_review_provider_calls
        self._plan_review_acceptances = initial_plan_review_acceptances
        self._plan_review_rejections = initial_plan_review_rejections
        self._plan_review_cache_hits = initial_plan_review_cache_hits
        self._plan_review_prompt_tokens = initial_plan_review_prompt_tokens
        self._plan_review_completion_tokens = initial_plan_review_completion_tokens
        if initial_plan_review_explicit_operation_floor_rejections > initial_plan_review_rejections:
            raise ValueError(
                "explicit-operation floor rejections cannot exceed plan-review rejections."
            )
        self._plan_review_explicit_operation_floor_rejections = (
            initial_plan_review_explicit_operation_floor_rejections
        )
        self._synchronize_controller_capabilities()

    async def prepare_context(
        self, *, include_state: bool = True, automatic_recall: bool = True
    ) -> str | None:
        """Publish current work state and retrieve bounded knowledge for this turn."""

        # A private helper is single-use only while producing one outward turn.
        # New tau input starts a fresh turn; content-addressed caches below still
        # suppress policy work that remains valid across that boundary.
        self._private_action_plan_revision = self.controller.revision
        self._private_actions_completed_at_plan_revision.clear()
        latest_customer_message_id = (
            self.ledger.customer_message_ids[-1] if self.ledger.customer_message_ids else None
        )
        if latest_customer_message_id != self._knowledge_search_customer_message_id:
            self._knowledge_search_customer_message_id = latest_customer_message_id
            self._customer_message_knowledge_searches = 0
        self._synchronize_controller_capabilities()
        self._turn_delivered_entry_ids.clear()
        self._human_transfer_policy_injected_this_turn = False
        self._product_comparison_documents_injected_this_turn = 0
        work_context = await self._publish_work_context()
        query = self._current_query()
        recall = None
        if query is not None and automatic_recall:
            recall = await self.corpus.recall(
                query,
                recent_conversation=tuple(
                    self.ledger.customer_messages[-_MAX_RECENT_CONVERSATION:-1]
                ),
                work_context=self._recall_work_context(work_context),
            )
            self.admission_gate.observe_knowledge_search()
        self._last_recall = recall
        contexts: list[str] = []
        if recall is not None and recall.context is not None:
            self._turn_delivered_entry_ids.update(recall.record_ids)
            self.controller.observe_knowledge_evidence(recall.evidence_refs)
            self.admission_gate.observe_knowledge_text(recall.context)
            contexts.append(recall.context)
        # Native outcome agents choose explicit product scopes through navigation.
        # Do not infer a single comparison scope from customer-language markers.
        if query is not None and self.controller.planning_mode != "outcomes":
            (
                comparison_context,
                comparison_evidence,
            ) = await self.corpus.product_comparison_context_with_evidence(
                query,
                exclude_entry_ids=self._turn_delivered_entry_ids,
            )
            if comparison_context is not None:
                self._turn_delivered_entry_ids.update(
                    reference.reference_id for reference in comparison_evidence
                )
                self.controller.observe_knowledge_evidence(comparison_evidence)
                self.admission_gate.observe_knowledge_text(comparison_context)
                contexts.append(comparison_context)
                self._product_comparison_documents_injected_this_turn = len(comparison_evidence)
        if not self.advisory_workflow and _customer_requests_credit_card_dispute(self.ledger.customer_messages):
            (
                dispute_context,
                dispute_evidence,
            ) = await self.corpus.credit_card_dispute_policy_context_with_evidence(
                exclude_entry_ids=self._turn_delivered_entry_ids,
            )
            if dispute_context is not None:
                self._turn_delivered_entry_ids.update(
                    reference.reference_id for reference in dispute_evidence
                )
                self.controller.observe_knowledge_evidence(dispute_evidence)
                self.admission_gate.observe_knowledge_text(dispute_context)
                contexts.append(dispute_context)
        if not self.advisory_workflow and query is not None and customer_requests_human_assistance(query):
            (
                transfer_context,
                transfer_evidence,
            ) = await self.corpus.human_transfer_policy_context_with_evidence(
                exclude_entry_ids=self._turn_delivered_entry_ids,
            )
            if transfer_context is not None:
                self._turn_delivered_entry_ids.update(
                    reference.reference_id for reference in transfer_evidence
                )
                self.controller.observe_knowledge_evidence(transfer_evidence)
                self.admission_gate.observe_knowledge_text(transfer_context)
                contexts.append(transfer_context)
                self._human_transfer_policy_injected_this_turn = True
        if include_state:
            contexts.insert(0, await self.runtime_context(work_context=work_context))
        return "\n\n".join(contexts) or None

    async def runtime_context(self, *, work_context: AgentWorkContext | None = None) -> str:
        """Refresh work/evidence without recalling or resetting action/search budgets."""

        self._synchronize_controller_capabilities()
        if work_context is None:
            work_context = await self._publish_work_context()
        contexts = [self._render_work_context(work_context), self.research_capacity_context()]
        for reference in self.controller.active_knowledge_references():
            self.navigation.remember_document(reference.reference_id)
        contexts.append(self.navigation.context())
        comparison = await self.navigation.comparison_context(
            self.controller.has_delivered_knowledge_revision
        )
        self.controller.set_comparison_readiness(self.navigation.comparison_readiness())
        if comparison:
            contexts.append(comparison)
        if (directory := self.corpus.public_capability_directory()) is not None:
            contexts.append(directory)
        for gate in (
            ()
            if self.advisory_workflow
            else (self._credit_card_dispute_action_gate, self._savings_interest_action_gate)
        ):
            if (context := gate.render_context()) is not None:
                contexts.append(context)
        prepared_policy_context = self._render_prepared_policy_context()
        if prepared_policy_context is not None:
            contexts.append(prepared_policy_context)
        capability_context = self.admission_gate.render_public_capability_context()
        if capability_context is not None:
            contexts.append(capability_context)
        if self.ledger_enabled:
            contexts.append(self.ledger.render_context())
        if self.workflow_controller_enabled:
            contexts.append(self.controller.render_context(advisory=self.advisory_workflow))
        return "\n\n".join(contexts)

    def research_capacity_context(self) -> str:
        """Expose application limits without changing policy or task completion."""
        customer = (
            self.ledger.customer_message_ids[-1] if self.ledger.customer_message_ids else None
        )
        remaining = self.navigation.remaining(customer)
        remaining[KNOWLEDGE_SEARCH_TOOL] = {
            "customer_calls": max(
                0,
                _MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE
                - self._customer_message_knowledge_searches,
            ),
            "conversation_calls": max(0, _MAX_KNOWLEDGE_SEARCHES - self._knowledge_searches),
        }
        exhausted = all(
            min(remaining[name].values()) == 0
            for name in (READ_TOOL, RECOVERY_LANE, KNOWLEDGE_SEARCH_TOOL)
        )
        return (
            "<research-capacity>\n"
            + json.dumps(
                {
                    "remaining_calls": remaining,
                    "research_exhausted": exhausted,
                    "delivered_source_reuse_available": self.navigation.allow_delivered_reuse,
                    "new_source_reads_available": min(remaining[READ_TOOL].values()) > 0,
                    "source_recovery_available": self.navigation.allow_delivered_reuse
                    or min(remaining[RECOVERY_LANE].values()) > 0,
                    "ranked_search_available": min(remaining[KNOWLEDGE_SEARCH_TOOL].values()) > 0,
                    "directory_browse_available": min(remaining[BROWSE_TOOL].values()) > 0,
                }
            )
            + (
                "\nAlready delivered immutable document revisions may be reread locally without "
                "a research quota charge. Exact delivery records determine eligibility; any unread "
                "revision uses the bounded new-source lane. Per-call document/byte bounds and "
                "overall runtime limits still apply. Prefer existing context; reread only needed text. "
                if self.navigation.allow_delivered_reuse
                else ""
            )
            + "\nThese are application cost controls, not banking policy or benchmark rules. "
            "The read tool selects its lane from exact delivery records: all-known revisions use "
            "source recovery; a request with any unread revision uses new research. Each lane is "
            "independently bounded. Directory browsing gives titles, not document bodies. "
            "Plan edits and worksheet checks cannot restore exhausted capacity. "
            "Use already-delivered evidence for supported work. A worksheet checks supplied facts; "
            "it cannot search for or read a missing document. "
            "If a necessary fact is unavailable, identify the specific gap and report an honest "
            "limitation while preserving unresolved work. Document retrieval obligations use "
            "kind knowledge_search; represent the limited response as a ready communication "
            "obligation, not a completed banking action. Other "
            "executable bank actions must not be relabeled as lookups. Research exhaustion "
            "does not supply evidence, establish eligibility or authorize completion. "
            "Do not ask the customer to repeat a request just to replenish capacity.\n"
            "</research-capacity>"
        )

    def tool_schemas(self) -> Sequence[Mapping[str, Any]]:
        self._refresh_private_action_epoch()
        schemas: list[Mapping[str, Any]] = []
        customer = (
            self.ledger.customer_message_ids[-1] if self.ledger.customer_message_ids else None
        )
        schemas.extend(
            s for s in navigation_tool_schemas() if self.navigation.available(customer, s["name"])
        )
        if self.advisory_workflow:
            for schema in schemas:
                if schema["name"] == COMPARISON_TOOL:
                    schema["description"] = (
                        "Keep exact source passages across compaction. Copy passage_handles from document reads; "
                        "no extracted values, field names, quotes, coverage or calculations are needed. "
                        "Choose a small set distinguishing plausible alternatives, including conditions and conflicts. "
                        "Passages must come from fully delivered sources. Saves return compact receipts. Capacity is 96 passages across at most 16 products per worksheet. operation=replace atomically keeps only the supplied set (empty clears it). operation=read returns selected retained handles or all with an empty list, in bounded pages; repeat the same selection with cursor=next_cursor. "
                        "For recovery you can instead use operation=read with entry_ids copied from source citations; this returns only retained passages, not a new document search. "
                        "Duplicate handles are harmless. Pinning does not verify relevance, interpretation, eligibility or completeness."
                    )
                    schema["input_schema"] = {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "passage_handles": {"type": "array", "maxItems": 24,
                                "items": {"type": "string", "minLength": 1, "maxLength": 240}},
                            "entry_ids": {"type": "array", "minItems": 1, "maxItems": 16,
                                "items": {"type": "string", "minLength": 1, "maxLength": 240}},
                            "worksheet_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]{1,64}$"},
                            "operation": {"enum": ["append", "replace", "read"]},
                            "cursor": {"type": "string", "minLength": 1, "maxLength": 100},
                        },
                        "oneOf": [
                            {"required": ["passage_handles"], "not": {"required": ["entry_ids"]}},
                            {"required": ["entry_ids", "operation"], "properties": {"operation": {"const": "read"}},
                             "not": {"required": ["passage_handles"]}},
                        ],
                    }
                if schema["name"] == READ_TOOL:
                    schema["description"] = (
                        "Read complete public document revisions by exact documents or product_id. "
                        "Already delivered immutable revisions are local source reuse and do not "
                        "consume research or recovery quota. A batch containing any unread revision "
                        "uses the bounded new-source budget. Directory listings are not delivery. "
                        "Follow the schema's document count and scope constraints; the complete "
                        f"result must fit {MAX_READ_RESULT_BYTES} UTF-8 bytes. Split oversized batches. "
                        "A complete product packet does not establish customer eligibility, consent "
                        "or coverage of separate general procedures. Reread only text you need. "
                        "Optional source_span with ONE exact document reads a bounded text window "
                        "using copied Unicode start/end offsets and optional context_chars. Windows "
                        "are navigation only: no full-document evidence, delivery marking or tool "
                        "unlocks. Omit source_span to read the complete source. Full reads include a passages map from handle to [start,end] offsets "
                        "for direct retention and read_progress showing which sources were already delivered."
                    )
        remaining_searches = _MAX_KNOWLEDGE_SEARCHES - self._knowledge_searches
        remaining_customer_message_searches = min(
            _MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE
            - self._customer_message_knowledge_searches,
            remaining_searches,
        )
        if remaining_customer_message_searches > 0:
            search_schema = self.corpus.search_tool_schema()
            search_schema["description"] = (
                f"{search_schema['description']} Exactly "
                f"{remaining_customer_message_searches} successful slot(s) remain for the "
                f"latest customer message and {remaining_searches} remain in this conversation; "
                "tool-result turns do not reset either bound."
            )
            schemas.append(search_schema)
        current_dispute_policy = bool(self._current_credit_card_dispute_policies())
        private_schemas = [
            dispute_policy_dossier_tool_schema(),
            dispute_audit_tool_schema(),
            reward_policy_dossier_tool_schema(),
            reward_audit_tool_schema(),
            savings_interest_policy_dossier_tool_schema(),
            savings_interest_audit_tool_schema(),
        ]
        if self.advisory_workflow:
            for schema in private_schemas:
                if schema["name"] == PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL:
                    schema["description"] = (
                        "Prepare the complete public policy directory for observed card types. "
                        "Read applicable full sources before choosing rates; directory entries "
                        "are not delivered evidence. Retain its private policy_dossier_id for "
                        "controller_audit_credit_card_rewards."
                    )
                elif schema["name"] == AUDIT_CREDIT_CARD_REWARDS_TOOL:
                    schema["description"] = schema["description"].replace(
                        "First prepare and read the complete Cayu policy dossier",
                        "First prepare the policy directory and read applicable full sources",
                    )
        schemas.extend(
            schema
            for schema in private_schemas
            # Outcome plans describe customer goals, not one tool invocation.
            # A successful read-only helper for one resource must not remove
            # that capability for every other resource until a bookkeeping edit.
            # Exact-step plans retain their explicit continuation contract.
            if (
                self.controller.planning_mode == "outcomes"
                or schema["name"] not in self._private_actions_completed_at_plan_revision
            )
            and not (
                schema["name"] == PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL and current_dispute_policy
            )
        )
        if self.workflow_controller_enabled:
            schemas.extend(self.controller.tool_schemas())
        return tuple(schemas)

    def handles(self, name: str) -> bool:
        return (
            name
            in {
                KNOWLEDGE_SEARCH_TOOL,
                PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL,
                AUDIT_CREDIT_CARD_DISPUTES_TOOL,
                PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL,
                AUDIT_CREDIT_CARD_REWARDS_TOOL,
                PREPARE_SAVINGS_INTEREST_POLICY_TOOL,
                AUDIT_SAVINGS_INTEREST_TOOL,
            }
            or name in NAVIGATION_TOOLS
            or (self.workflow_controller_enabled and self.controller.handles(name))
        )

    def transcript_retention_key(self, name: str) -> str | None:
        """Delegate private workflow-snapshot compaction to the controller."""

        if not self.workflow_controller_enabled:
            return None
        return self.controller.transcript_retention_key(name)

    def render_internal_context(self) -> str | None:
        """Refresh authoritative workflow state during a private bridge loop."""

        if not self.workflow_controller_enabled:
            return None
        return self.controller.render_internal_context()

    def progress_token(self) -> str:
        """Return a stable token for useful memory/controller state changes."""

        payload = {
            "controller": self.controller.progress_token(),
            "credit_card_dispute_audits": dict(sorted(self._credit_card_dispute_audits.items())),
            "credit_card_dispute_policies": {
                user_id: [policy.dossier_id, observation_fingerprint]
                for user_id, (policy, observation_fingerprint) in sorted(
                    self._credit_card_dispute_policies.items()
                )
            },
            "reward_policy_dossiers": {
                user_id: [dossier_id, list(card_types)]
                for user_id, (dossier_id, card_types) in sorted(
                    self._reward_policy_dossiers.items()
                )
            },
            "savings_interest_audits": dict(sorted(self._savings_interest_audits.items())),
            "savings_interest_policy_dossiers": {
                f"{user_id}:{account_id}": [dossier_id, observation_fingerprint]
                for (user_id, account_id), (
                    dossier_id,
                    observation_fingerprint,
                ) in sorted(self._savings_interest_policy_dossiers.items())
            },
            "private_action_epoch": {
                "completed_tool_names": sorted(self._private_actions_completed_at_plan_revision),
                "plan_revision": self._private_action_plan_revision,
            },
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def required_action_tool_names(self) -> tuple[str, ...]:
        """Expose controller-authorized continuations to the provider bridge."""

        if not self.workflow_controller_enabled:
            return ()
        self._synchronize_controller_capabilities()
        self._refresh_private_action_epoch()
        return tuple(
            name
            for name in self.controller.required_action_tool_names()
            if name not in self._private_actions_completed_at_plan_revision
        )

    def required_action_batch_size(self, name: str) -> int:
        """Delegate bounded homogeneous execution batches to the controller."""

        if not self.workflow_controller_enabled:
            return 1
        return self.controller.required_action_batch_size(name)

    def allows_staged_external_calls(self, internal_tool_names: tuple[str, ...]) -> bool:
        """Delegate atomic plan-publication prefixes to the controller."""

        return self.workflow_controller_enabled and self.controller.allows_staged_external_calls(
            internal_tool_names
        )

    def diagnostic_counts(self) -> dict[str, int]:
        """Return content-free internal counters for saved and discarded attempts."""

        return self.controller.diagnostic_counts()

    def review(
        self,
        calls: Sequence[ToolCallPart],
    ) -> tuple[ExternalToolAdmission, ...] | Awaitable[tuple[ExternalToolAdmission, ...]]:
        """Compose deterministic admission with isolated semantic action review."""

        if self.controller.invalid_comparison_bindings():
            return self.controller.review(calls)
        deterministic = self.corpus.public_argument_contracts.review(calls)
        if any(not admission.allowed for admission in deterministic):
            return deterministic
        deterministic = self._credit_card_dispute_action_gate.review(calls)
        if any(not admission.allowed for admission in deterministic):
            return deterministic
        deterministic = self._savings_interest_action_gate.review(calls)
        if any(not admission.allowed for admission in deterministic):
            return deterministic
        if self._action_verifier is None:
            return deterministic
        structural = self.controller.review(calls)
        if any(not admission.allowed for admission in structural):
            return structural
        reviewed_indexes = tuple(
            index
            for index, call in enumerate(calls)
            if call.tool_name in _SEMANTICALLY_REVIEWED_ACTION_TOOLS
        )
        if not reviewed_indexes:
            return deterministic
        return self._review_semantic_actions(calls, reviewed_indexes)

    async def _review_semantic_actions(
        self,
        calls: Sequence[ToolCallPart],
        reviewed_indexes: tuple[int, ...],
    ) -> tuple[ExternalToolAdmission, ...]:
        """Review consequential calls against public intent without oracle access."""

        reviewed_calls = tuple(calls[index] for index in reviewed_indexes)
        observation_contexts = [
            self.ledger.render_context(
                referenced_arguments=tuple(call.arguments for call in reviewed_calls),
                include_dialogue=True,
            ),
            self.admission_gate.render_knowledge_search_review_context(),
        ]
        observation_contexts.extend(
            context
            for context in (
                self._credit_card_dispute_action_gate.render_context(),
                self._savings_interest_action_gate.render_context(),
            )
            if context is not None
        )
        if any(call.tool_name == "transfer_to_human_agents" for call in reviewed_calls):
            # The isolated reviewer needs the exact public taxonomy, not just
            # customer dialogue and model-authored plan references. Reading it
            # here grants neither actor exposure nor new execution authority.
            policy_context, _ = await self.corpus.human_transfer_policy_context_with_evidence()
            if policy_context is not None:
                observation_contexts.append(policy_context)
            transfer_context = self.admission_gate.render_offer_transfer_review_context()
            if transfer_context is not None:
                observation_contexts.append(transfer_context)
        observation_context = "\n\n".join(observation_contexts)
        workflow_context = self.controller.render_action_review_context()
        review_key = _action_review_key(
            customer_messages=self.ledger.customer_messages,
            observation_context=observation_context,
            workflow_context=workflow_context,
            calls=reviewed_calls,
        )
        decisions = self._action_review_cache.get(review_key)
        if decisions is None:
            batch = await self._action_verifier.review(
                customer_messages=tuple(self.ledger.customer_messages),
                observation_context=observation_context,
                workflow_context=workflow_context,
                calls=reviewed_calls,
            )
            if not isinstance(batch, ActionReviewBatch) or len(batch.decisions) != len(
                reviewed_calls
            ):
                raise TypeError("action_verifier returned an invalid review batch.")
            decisions = batch.decisions
            self._action_review_provider_calls += 1
            self._action_review_acceptances += sum(decision.accepted for decision in decisions)
            self._action_review_rejections += sum(not decision.accepted for decision in decisions)
            self._action_review_prompt_tokens += batch.prompt_tokens or 0
            self._action_review_completion_tokens += batch.completion_tokens or 0
            self._action_review_cache[review_key] = decisions
            if len(self._action_review_cache) > _MAX_ACTION_REVIEW_CACHE_ENTRIES:
                del self._action_review_cache[next(iter(self._action_review_cache))]
        else:
            self._action_review_cache_hits += 1
        decisions_by_call = {
            reviewed_indexes[decision.call_index]: decision for decision in decisions
        }
        return tuple(
            ExternalToolAdmission(tool_call_id=call.tool_call_id, allowed=True)
            if (decision := decisions_by_call.get(index)) is None or decision.accepted
            else ExternalToolAdmission(
                tool_call_id=call.tool_call_id,
                allowed=False,
                reason=(
                    "The isolated semantic action verifier rejected this proposed call. "
                    "Revise the plan and call to match public customer and tool evidence. "
                    f"Correction class: {decision.correction.value}."
                ),
                diagnostic_code=f"semantic_action_{decision.correction.value}",
            )
            for index, call in enumerate(calls)
        )

    async def review_terminal_output(
        self, text: str, *, allow_unplanned_advice: bool = False
    ) -> TerminalOutputAdmission:
        """Compose structural readiness with an optional read-only semantic review."""

        self._synchronize_controller_capabilities()
        await self.navigation.comparison_context(self.controller.has_delivered_knowledge_revision)
        self.controller.set_comparison_readiness(self.navigation.comparison_readiness())
        if gaps := self.controller.invalid_comparison_bindings():
            return TerminalOutputAdmission(
                allowed=False,
                reason="A resolved comparison no longer has complete revalidated declared coverage. Reopen it and address its missing facts/questions: "
                + ", ".join(gaps),
                correction_code="comparison_coverage_invalidated",
                required_tool_names=(UPDATE_WORK_PLAN_TOOL,),
            )
        unplanned = (
            allow_unplanned_advice
            and self.controller.planning_mode == "outcomes"
            and not self.controller.has_work_items()
        )
        structural = (
            TerminalOutputAdmission(allowed=True)
            if unplanned
            else self.controller.preview_terminal_output(text)
        )
        if not structural.allowed:
            return structural
        if self._plan_verifier is not None:
            semantic = await self._review_semantic_plan()
            if not semantic.allowed:
                return semantic
        customer_tool = self._review_credit_card_last_four_handoff()
        if not customer_tool.allowed:
            return customer_tool
        audited_filing = self._review_audited_credit_card_dispute_handoff()
        if not audited_filing.allowed:
            return audited_filing
        response = await self._review_semantic_response(text)
        if not response.allowed:
            return response
        return (
            TerminalOutputAdmission(allowed=True)
            if unplanned
            else self.controller.review_terminal_output(text)
        )

    def _review_credit_card_last_four_handoff(self) -> TerminalOutputAdmission:
        """Require the public customer tool instead of manual card-number collection."""

        response_state = self.controller.response_review_state()
        dispute_active = _customer_requests_credit_card_dispute(
            self.ledger.customer_messages
        ) or _response_contains_credit_card_dispute(response_state)
        if not dispute_active or not _response_requires_credit_card_last_four(response_state):
            return TerminalOutputAdmission(allowed=True)
        verified_accounts = tuple(
            account
            for user_id in self.ledger.verification_logs_by_user
            if (accounts := self.ledger.credit_card_accounts(user_id)) is not None
            for account in accounts
        )
        if verified_accounts and all(
            account.card_last_four is not None for account in verified_accounts
        ):
            return TerminalOutputAdmission(
                allowed=False,
                reason=(
                    "The verified customer's complete credit-card account result already "
                    "contains the last four digits for every observed card. Do not give or "
                    "handoff get_card_last_4_digits, and do not ask the customer for data "
                    "already present in authoritative tool evidence. Update the stable "
                    "dispute item to use the observed digits and advance its next prerequisite "
                    "or filing operation."
                ),
                correction_code="response_dispute_last_four_already_observed",
                required_tool_names=(UPDATE_WORK_PLAN_TOOL,),
            )
        exposed = self.admission_gate.knowledge_identifier_is_exposed(_CREDIT_CARD_LAST_FOUR_TOOL)
        if not exposed:
            return TerminalOutputAdmission(
                allowed=False,
                reason=(
                    "A credit-card dispute is waiting for the card's last four digits, but "
                    "the required public retrieval procedure has not been delivered. Update "
                    "the plan to retrieve that procedure before asking the customer."
                ),
                correction_code="response_dispute_last_four_policy_required",
                required_tool_names=(UPDATE_WORK_PLAN_TOOL,),
            )
        given = _CREDIT_CARD_LAST_FOUR_TOOL in self.ledger.tools_given_to_customer
        if not given:
            return TerminalOutputAdmission(
                allowed=False,
                reason=(
                    "The delivered credit-card dispute policy says the customer-side "
                    "get_card_last_4_digits tool replaces manual retrieval. Do not ask the "
                    "customer to inspect or share card digits yet. Update one stable dispute "
                    "or prerequisite item to the exact agent-owned give_discoverable_user_tool "
                    "operation, then execute it."
                ),
                correction_code="response_dispute_last_four_tool_delivery_required",
                required_tool_names=(UPDATE_WORK_PLAN_TOOL,),
            )
        handoff_account_id = _credit_card_last_four_tool_handoff_account_id(response_state)
        if (
            handoff_account_id is None
            or self.ledger.credit_card_account(handoff_account_id) is None
        ):
            return TerminalOutputAdmission(
                allowed=False,
                reason=(
                    "get_card_last_4_digits was already given successfully. Update one stable "
                    "dispute or prerequisite item to a customer-owned tool operation using that "
                    "exact name and the observed credit_card_account_id, then ask the customer "
                    "to run it; do not request manual card inspection."
                ),
                correction_code="response_dispute_last_four_tool_handoff_required",
                required_tool_names=(UPDATE_WORK_PLAN_TOOL,),
            )
        return TerminalOutputAdmission(allowed=True)

    def _review_audited_credit_card_dispute_handoff(self) -> TerminalOutputAdmission:
        """Keep audited filings on the agent execution path until resolved."""

        reason = self._credit_card_dispute_action_gate.terminal_handoff_denial(
            self.controller.response_review_state()
        )
        if reason is None:
            return TerminalOutputAdmission(allowed=True)
        return TerminalOutputAdmission(
            allowed=False,
            reason=reason,
            correction_code="response_audited_dispute_filing_required",
            required_tool_names=(UPDATE_WORK_PLAN_TOOL,),
        )

    async def _review_semantic_response(self, text: str) -> TerminalOutputAdmission:
        """Require one proposed response to cover its complete public disposition."""

        if self._response_verifier is None:
            return TerminalOutputAdmission(allowed=True)
        response_state = self.controller.response_review_state()
        observation_context = "\n\n".join(
            (
                self.ledger.render_context(include_dialogue=True),
                self.admission_gate.render_knowledge_search_review_context(),
            )
        )
        review_key = _response_review_key(
            customer_messages=self.ledger.customer_messages,
            observation_context=observation_context,
            response_state=response_state,
            proposed_response=text,
        )
        decision = self._response_review_cache.get(review_key)
        if decision is None:
            decision = await self._response_verifier.review(
                customer_messages=tuple(self.ledger.customer_messages),
                observation_context=observation_context,
                response_state=response_state,
                proposed_response=text,
            )
            if not isinstance(decision, ResponseReviewDecision):
                raise TypeError("response_verifier returned an invalid review decision.")
            self._response_review_provider_calls += 1
            self._response_review_acceptances += decision.accepted
            self._response_review_rejections += not decision.accepted
            self._response_review_prompt_tokens += decision.prompt_tokens or 0
            self._response_review_completion_tokens += decision.completion_tokens or 0
            self._response_review_cache[review_key] = decision
            if len(self._response_review_cache) > _MAX_RESPONSE_REVIEW_CACHE_ENTRIES:
                del self._response_review_cache[next(iter(self._response_review_cache))]
        else:
            self._response_review_cache_hits += 1
        if decision.accepted:
            return TerminalOutputAdmission(allowed=True)
        reconsider_plan = decision.correction in {
            ResponseReviewCorrection.CUSTOMER_INSTRUCTION_MISMATCH,
            ResponseReviewCorrection.REPEATED_OR_STALLED_REQUEST,
        }
        repair = (
            "Re-read the latest customer instruction and the affected work items. "
            "If a planned question was already answered, consent was already supplied, "
            "or the customer requested an actionable next operation, update the plan "
            "and advance that operation instead of rewriting the same question. "
            "A wrong plan or handoff cannot be repaired by response wording alone. "
            "If the plan already matches the instruction, correct only the draft. "
            if reconsider_plan
            else "Rewrite it to cover every controller response item without "
            "unsupported claims or repeated requests. "
        )
        return TerminalOutputAdmission(
            allowed=False,
            reason=(
                "The isolated semantic response verifier rejected this customer-facing "
                f"draft. {repair}"
                f"Correction class: {decision.correction.value}. "
                "Affected workflow item IDs: "
                f"{', '.join(decision.affected_item_ids)}."
            ),
            correction_code=f"semantic_response_{decision.correction.value}",
            required_tool_names=(UPDATE_WORK_PLAN_TOOL,) if reconsider_plan else (),
        )

    async def _review_semantic_plan(
        self,
        *,
        reclassifiable_item_ids: Sequence[str] = (),
    ) -> TerminalOutputAdmission:
        """Review immutable outcome coverage and lock a rejected plan revision."""

        plan_state = self.controller.semantic_review_state()
        if not plan_state["items"]:
            raise RuntimeError("Cannot semantically review an empty plan.")
        dialogue_context = self.ledger.render_dialogue_context()
        review_key = _plan_review_key(
            self.ledger.customer_messages,
            plan_state,
            reclassifiable_item_ids=reclassifiable_item_ids,
            dialogue_context=dialogue_context,
        )
        decision = self._plan_review_cache.get(review_key)
        if decision is None:
            decision = await self._plan_verifier.review(
                customer_messages=tuple(self.ledger.customer_messages),
                plan_state=plan_state,
                reclassifiable_item_ids=reclassifiable_item_ids,
                dialogue_context=dialogue_context,
            )
            if not isinstance(decision, PlanReviewDecision):
                raise TypeError("plan_verifier returned an invalid decision.")
            self._plan_review_provider_calls += 1
            self._plan_review_prompt_tokens += decision.prompt_tokens or 0
            self._plan_review_completion_tokens += decision.completion_tokens or 0
            if decision.accepted:
                self._plan_review_acceptances += 1
            else:
                self._plan_review_rejections += 1
            self._plan_review_cache[review_key] = decision
            if len(self._plan_review_cache) > _MAX_PLAN_REVIEW_CACHE_ENTRIES:
                del self._plan_review_cache[next(iter(self._plan_review_cache))]
        else:
            self._plan_review_cache_hits += 1
        if not decision.accepted:
            required_new_item_mode = {
                PlanReviewCorrection.MISSING_INFORMATION_OUTCOME: (WorkCompletionMode.INFORMATION),
                PlanReviewCorrection.MISSING_OPERATION_OUTCOME: WorkCompletionMode.OPERATION,
            }.get(decision.correction)
            require_operation_increase = (
                decision.correction is PlanReviewCorrection.WRONG_COMPLETION_MODE_TO_OPERATION
            )
            self.controller.reject_semantic_plan(
                reclassifiable_item_ids=(
                    tuple(
                        item_id
                        for item_id in decision.affected_item_ids
                        if item_id in reclassifiable_item_ids
                    )
                    if require_operation_increase
                    else ()
                ),
                required_new_item_mode=required_new_item_mode,
                required_new_item_count=decision.required_new_item_count,
                require_operation_increase=require_operation_increase,
            )
            return TerminalOutputAdmission(
                allowed=False,
                reason=(
                    "Semantic plan review rejected this plan revision. Changing only the "
                    "response wording cannot fix it; revise the work plan after re-reading "
                    f"the public customer request. Correction class: {decision.correction.value}. "
                    f"Required new outcome count: {decision.required_new_item_count}. "
                    "Affected workflow item IDs: "
                    f"{', '.join(decision.affected_item_ids) or 'none'}."
                ),
                correction_code=f"semantic_plan_{decision.correction.value}",
                required_tool_names=(UPDATE_WORK_PLAN_TOOL,),
            )
        return TerminalOutputAdmission(allowed=True)

    async def invoke(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        allow_unplanned_public_research: bool = False,
        search_exclude_entry_ids: frozenset[str] | None = None,
    ) -> InternalToolResult:
        """Execute a read-only memory search or validated private plan transition."""

        if (
            name == KNOWLEDGE_SEARCH_TOOL
            and isinstance(arguments, Mapping)
            and "queries" in arguments
        ):
            queries = arguments["queries"]
            if (
                not set(arguments).issubset({"queries", "limit", "mode"})
                or type(arguments.get("limit", 1)) is not int
                or not 1 <= arguments.get("limit", 1) <= 4
                or not isinstance(queries, list)
                or not 1 <= len(queries) <= 4
                or any(
                    not isinstance(query, dict)
                    or not set(query).issubset({"query", "mode", "cursor", "limit", "scope", "terms", "match"})
                    for query in queries
                )
            ):
                return InternalToolResult(
                    content="Use queries: 1-4 search requests, with optional limit per request or for the batch.", is_error=True
                )

            batch_exclusions = frozenset(self._turn_delivered_entry_ids)

            async def search_one(query):
                try:
                    return await self.invoke(
                        name,
                        {"limit": arguments.get("limit", 1), "mode": arguments.get("mode", "default"), **query},
                        allow_unplanned_public_research=allow_unplanned_public_research,
                        search_exclude_entry_ids=batch_exclusions,
                    )
                except Exception as error:
                    # Do not leak provider credentials or arbitrary exception text.
                    return InternalToolResult(
                        content="Search failed; retry this query separately.",
                        is_error=True,
                        diagnostic_code=f"search_batch_{type(error).__name__}",
                    )

            results = await asyncio.gather(*(search_one(query) for query in queries))
            return InternalToolResult(
                content=json.dumps(
                    {
                        "batch": [
                            {
                                "index": index,
                                "request": query,
                                "is_error": result.is_error,
                                "diagnostic_code": result.diagnostic_code,
                                "result": json.loads(result.content)
                                if not result.is_error
                                else result.content,
                            }
                            for index, (query, result) in enumerate(
                                zip(queries, results, strict=True)
                            )
                        ],
                        "per_query_page_limit": arguments.get("limit", 1),
                    }
                ),
                is_error=all(result.is_error for result in results),
            )

        from cayu_tau3_banking.advice import PUBLIC_RESEARCH_TOOLS

        self._synchronize_controller_capabilities()
        if (
            self.workflow_controller_enabled
            and not self.advisory_workflow
            and name != SYNC_WORK_PLAN_TOOL
            and not self.controller.has_work_items()
            and not (
                allow_unplanned_public_research
                and self.controller.planning_mode == "outcomes"
                and name in PUBLIC_RESEARCH_TOOLS
            )
        ):
            return InternalToolResult(
                content=(
                    "Create a populated workflow plan before using memory, controller, or "
                    "policy-audit helpers. Call controller_sync_work_plan first."
                ),
                is_error=True,
            )
        if name in NAVIGATION_TOOLS:
            try:
                args = dict(arguments)
                customer = (
                    self.ledger.customer_message_ids[-1]
                    if self.ledger.customer_message_ids
                    else None
                )
                charged_lane = self.navigation.begin_call(
                    name,
                    args,
                    customer,
                    delivered=self.controller.has_delivered_knowledge_revision,
                )
                if name == BROWSE_TOOL:
                    payload = self.navigation.browse(args)
                elif name == READ_TOOL:
                    documents = [
                        await self.corpus.read_document(d["entry_id"], d["revision"])
                        for d in self.navigation.resolve_read_documents(args)
                    ]
                    if len({d["entry_id"] for d in documents}) != len(documents):
                        raise ValueError("Read each source once per batch.")
                    for document in documents:
                        locator = self.navigation.locator(document["entry_id"])
                        document["product_id"] = locator["product_id"]
                        document["category"] = locator["category"]
                        document["source_handle"] = locator["source_handle"]
                    for document in documents:
                        document["passages"] = {p["passage_handle"]: [p["source_span"]["start"], p["source_span"]["end"]] for p in self.navigation.document_passages[document["entry_id"]]}
                    payload = {"documents": documents, "corpus_sha256": self.corpus.corpus_sha256}
                    previous = [d["entry_id"] for d in documents if
                                self.controller.has_delivered_knowledge_revision(d["entry_id"], str(d["entry_revision"]))]
                    payload["read_progress"] = {
                        "previously_delivered": previous,
                        "newly_delivered": [d["entry_id"] for d in documents if d["entry_id"] not in previous],
                        "all_sources_previously_delivered": len(previous) == len(documents),
                        "notice": "Source-delivery accounting, not semantic novelty. Reuse retained passages; reread only missing conditions. A repeated read remains allowed.",
                    }
                    if "source_span" in args:
                        from cayu_tau3_banking.document_windows import document_window

                        documents = [document_window(documents[0], args["source_span"])]
                        payload["documents"] = documents
                        payload["read_progress"]["newly_delivered"] = []
                        payload["read_progress"]["navigation_window_only"] = True
                    if "product_id" in args:
                        payload["product_scope"] = {
                            "product_id": args["product_id"],
                            "directory_document_count": len(documents),
                            "all_product_documents_included": True,
                            "general_policy_coverage_established": False,
                            "establishes_customer_eligibility_or_consent": False,
                        }
                else:
                    if "passage_handles" in args or "entry_ids" in args:
                        from cayu_tau3_banking.passage_retention import retain_passages

                        payload = await retain_passages(self.navigation, args, self.controller.has_delivered_knowledge_revision)
                    elif self.advisory_workflow:
                        from cayu_tau3_banking.incremental_comparison import check_incremental

                        payload = await check_incremental(
                            self.navigation, args, self.controller.has_delivered_knowledge_revision
                        )
                    else:
                        payload = await self.navigation.check(
                            args, self.controller.has_delivered_knowledge_revision
                        )
                    await self.navigation.comparison_context(
                        self.controller.has_delivered_knowledge_revision
                    )
                    self.controller.set_comparison_readiness(self.navigation.comparison_readiness())
                payload["navigation_budget"] = {
                    "customer": self.navigation.state["customer"],
                    "lanes": deepcopy(self.navigation.state["lanes"]),
                    "recovery": deepcopy(self.navigation.state["recovery"]),
                    "charged_lane": charged_lane,
                }
                payload["remaining_calls"] = self.navigation.remaining(customer)
                content = json.dumps(payload, ensure_ascii=False)
                if name == READ_TOOL:
                    if len(content.encode("utf-8")) > MAX_READ_RESULT_BYTES:
                        raise ValueError(
                            f"Complete read result exceeds {MAX_READ_RESULT_BYTES} UTF-8 bytes. "
                            "Split the batch. No document from this call was delivered or admitted as evidence."
                        )
                    references = tuple(
                        WorkEvidenceReference(**d["evidence_ref"])
                        for d in documents
                        if not d.get("navigation_only")
                    )
                    self.controller.observe_knowledge_evidence(references)
                    self._turn_delivered_entry_ids.update(r.reference_id for r in references)
                    for document in documents:
                        self.navigation.remember_document(document["entry_id"])
                    if "source_span" not in args:
                        self.admission_gate.observe_knowledge_text(content)
                return InternalToolResult(content=content)
            except (ValueError, TypeError) as error:
                return InternalToolResult(
                    content=f"Navigation/check rejected: {error}", is_error=True
                )
        if name == PREPARE_CREDIT_CARD_DISPUTE_POLICY_TOOL:
            result = await self._prepare_credit_card_dispute_policy_dossier(arguments)
            return self._observe_successful_private_action(name, result)
        if name == AUDIT_CREDIT_CARD_DISPUTES_TOOL:
            user_id = arguments.get("user_id") if isinstance(arguments, Mapping) else None
            normalized_user_id = user_id.strip() if isinstance(user_id, str) else None
            observation_fingerprint = (
                _credit_card_dispute_observation_fingerprint(
                    self.ledger,
                    user_id=normalized_user_id,
                )
                if normalized_user_id
                else None
            )
            prepared = (
                self._credit_card_dispute_policies.get(normalized_user_id)
                if normalized_user_id
                else None
            )
            policy = (
                prepared[0]
                if prepared is not None and prepared[1] == observation_fingerprint
                else None
            )
            result = invoke_dispute_audit(self.ledger, arguments, policy=policy)
            if not result.is_error and normalized_user_id is not None:
                self._credit_card_dispute_action_gate.observe_audit(
                    result,
                    expected_user_id=normalized_user_id,
                )
                self._credit_card_dispute_audits[normalized_user_id] = sha256(
                    result.content.encode("utf-8")
                ).hexdigest()
            return self._observe_successful_private_action(name, result)
        if name == PREPARE_CREDIT_CARD_REWARD_POLICY_TOOL:
            result = await self._prepare_reward_policy_dossier(arguments)
            return self._observe_successful_private_action(name, result)
        if name == AUDIT_CREDIT_CARD_REWARDS_TOOL:
            user_id = arguments.get("user_id") if isinstance(arguments, Mapping) else None
            normalized_user_id = user_id.strip() if isinstance(user_id, str) else None
            observations = (
                self.ledger.credit_card_transactions(normalized_user_id)
                if normalized_user_id
                else None
            )
            current_card_types = (
                _completed_card_types(observations) if observations is not None else None
            )
            prepared = (
                self._reward_policy_dossiers.get(normalized_user_id) if normalized_user_id else None
            )
            required_dossier_id = (
                prepared[0] if prepared is not None and prepared[1] == current_card_types else None
            )
            result = invoke_reward_audit(
                self.ledger, arguments, required_policy_dossier_id=required_dossier_id,
            )
            return self._observe_successful_private_action(name, result)
        if name == PREPARE_SAVINGS_INTEREST_POLICY_TOOL:
            result = await self._prepare_savings_interest_policy_dossier(arguments)
            return self._observe_successful_private_action(name, result)
        if name == AUDIT_SAVINGS_INTEREST_TOOL:
            user_id = arguments.get("user_id") if isinstance(arguments, Mapping) else None
            account_id = arguments.get("account_id") if isinstance(arguments, Mapping) else None
            normalized_user_id = user_id.strip() if isinstance(user_id, str) else None
            normalized_account_id = account_id.strip() if isinstance(account_id, str) else None
            observation_fingerprint = (
                _savings_policy_observation_fingerprint(
                    self.ledger,
                    user_id=normalized_user_id,
                    account_id=normalized_account_id,
                )
                if normalized_user_id and normalized_account_id
                else None
            )
            prepared = (
                self._savings_interest_policy_dossiers.get(
                    (normalized_user_id, normalized_account_id)
                )
                if normalized_user_id and normalized_account_id
                else None
            )
            required_dossier_id = (
                prepared[0]
                if prepared is not None and prepared[1] == observation_fingerprint
                else None
            )
            result = invoke_savings_interest_audit(
                self.ledger,
                arguments,
                required_policy_dossier_id=required_dossier_id,
                component_source_validator=lambda source: (
                    isinstance(source.get("entry_id"), str)
                    and type(source.get("revision")) is int
                    and isinstance(source.get("quote"), str)
                    and bool(source["quote"].strip())
                    and len(source["quote"]) <= 1600
                    and self.controller.has_delivered_knowledge_revision(
                        source["entry_id"], str(source["revision"])
                    )
                    and source["entry_id"] in self.corpus._entries
                    and source["quote"] in self.corpus._entries[source["entry_id"]].text
                ),
            )
            if not result.is_error and isinstance(account_id, str):
                normalized_account_id = account_id.strip()
                self._savings_interest_action_gate.observe_audit(
                    result,
                    expected_account_id=normalized_account_id,
                )
                self._savings_interest_audits[normalized_account_id] = sha256(
                    result.content.encode("utf-8")
                ).hexdigest()
            return self._observe_successful_private_action(name, result)
        if name != KNOWLEDGE_SEARCH_TOOL:
            if self.workflow_controller_enabled and self.controller.handles(name):
                if name in {SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL}:
                    await self.navigation.comparison_context(
                        self.controller.has_delivered_knowledge_revision
                    )
                    self.controller.set_comparison_readiness(self.navigation.comparison_readiness())
                previous_revision = self.controller.revision
                previous_item_ids = {
                    item["id"] for item in self.controller.semantic_review_state()["items"]
                }
                result = self.controller.invoke_model_update(name, arguments)
                if (
                    name in {SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL}
                    and not result.is_error
                    and self.controller.revision != previous_revision
                    and self._plan_verifier is not None
                ):
                    current_item_ids = {
                        item["id"] for item in self.controller.semantic_review_state()["items"]
                    }
                    semantic = await self._review_semantic_plan(
                        reclassifiable_item_ids=tuple(sorted(current_item_ids - previous_item_ids))
                    )
                    if not semantic.allowed:
                        return InternalToolResult(
                            content=semantic.reason or "",
                            is_error=True,
                            diagnostic_code=f"sync_{semantic.correction_code}",
                            resolutions=result.resolutions,
                        )
                return result
            return InternalToolResult(content=f"Unknown internal tool {name!r}.", is_error=True)
        if not isinstance(arguments, Mapping) or not set(arguments).issubset(
            {"query", "limit", "mode", "cursor", "scope", "terms", "match"}
        ):
            return InternalToolResult(
                content="Knowledge search accepts query and optional limit, mode, cursor, scope, terms, match. For a batch put these fields inside each queries item.",
                is_error=True,
            )
        if "query" not in arguments:
            return InternalToolResult(content="Knowledge search requires 'query'.", is_error=True)
        mode = arguments.get("mode", "default")
        if mode not in ("default", "hybrid", "semantic", "lexical", "capability", "literal", "scoped"):
            return InternalToolResult(
                content="Knowledge search mode must be default, hybrid, semantic, lexical, capability, literal, or scoped. Scope works with default/lexical/literal/scoped only.",
                is_error=True,
            )
        if mode in ("hybrid", "semantic") and self.corpus.retrieval_mode != "hybrid":
            return InternalToolResult(
                content="Embedding search is not configured. Use lexical, literal or capability mode.",
                is_error=True,
                diagnostic_code="embedding_search_unavailable",
            )
        limit = arguments.get("limit", self.corpus.max_tool_documents)
        if type(limit) is not int or not 1 <= limit <= self.corpus.max_tool_documents:
            return InternalToolResult(
                content=(
                    "Knowledge search limit must be an integer from 1 through "
                    f"{self.corpus.max_tool_documents}."
                ),
                is_error=True,
            )
        if self._knowledge_searches >= _MAX_KNOWLEDGE_SEARCHES:
            return InternalToolResult(
                content=(
                    "The explicit knowledge-search budget is exhausted. Use the public "
                    "knowledge already delivered by automatic recall and prior searches, "
                    "then continue only supported work. If a required fact is still missing, "
                    "report the limitation; do not guess or claim completion."
                ),
                is_error=True,
            )
        if (
            self._customer_message_knowledge_searches
            >= _MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE
        ):
            return InternalToolResult(
                content=(
                    "The explicit knowledge-search budget for the latest customer message is "
                    "exhausted. "
                    "Use the automatic recall and non-duplicate search results already delivered, "
                    "then continue only supported work. If a required fact is still missing, "
                    "report the limitation; do not guess or claim completion."
                ),
                is_error=True,
            )
        # Reserve before awaiting retrieval so concurrent batch members cannot
        # all observe and spend the same final quota slot.
        self._knowledge_searches += 1
        self._customer_message_knowledge_searches += 1
        try:
            content, evidence_refs = await self.corpus.search_with_evidence(
                arguments["query"],
                limit=limit,
                mode=mode,
                exclude_entry_ids=(
                    frozenset(self._turn_delivered_entry_ids)
                    if search_exclude_entry_ids is None
                    else search_exclude_entry_ids
                ),
                cursor=arguments.get("cursor"),
                capture=self.retrieval_capture,
                scope=arguments.get("scope"),
                terms=arguments.get("terms"),
                match=arguments.get("match", "any"),
            )
        except ValueError:
            self._knowledge_searches -= 1
            self._customer_message_knowledge_searches -= 1
            return InternalToolResult(
                content=(
                    "Knowledge search query must be nonblank text within its size bound. "
                    "Literal mode accepts at most 500 characters. Scope requires exact directory IDs and default/lexical/literal/scoped mode; "
                    "terms must be 1-8 nonblank strings of at most 100 characters and match any/all. "
                    "If using a cursor, repeat its original query, mode, scope and terms; if results changed "
                    "or the cursor is invalid, restart without cursor."
                ),
                is_error=True,
            )
        payload = json.loads(content)
        payload["search_budget"] = {
            "remaining_conversation": _MAX_KNOWLEDGE_SEARCHES - self._knowledge_searches,
            "remaining_customer_message": (
                _MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE
                - self._customer_message_knowledge_searches
            ),
            "can_search_again": (
                self._knowledge_searches < _MAX_KNOWLEDGE_SEARCHES
                and self._customer_message_knowledge_searches
                < _MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE
            ),
        }
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        self._synchronize_controller_capabilities()
        self.admission_gate.observe_knowledge_search()
        self._turn_delivered_entry_ids.update(reference.reference_id for reference in evidence_refs)
        self.controller.observe_knowledge_evidence(evidence_refs)
        if not payload.get("navigation_only", False):
            admitted_content = content
            if payload.get("retrieval", {}).get("mode") == "scoped":
                admitted_content = json.dumps({"documents": [
                    doc for doc in payload["documents"] if doc.get("text_complete")
                    and not doc.get("navigation_only")
                ]})
            self.admission_gate.observe_knowledge_text(admitted_content)
        return self._observe_successful_private_action(
            name,
            InternalToolResult(content=content),
        )

    def _synchronize_controller_capabilities(self) -> None:
        """Keep plan validation aligned with the currently offered search tool."""

        customer = (
            self.ledger.customer_message_ids[-1] if self.ledger.customer_message_ids else None
        )
        self.controller.set_knowledge_search_available(
            (
                self.controller.planning_mode == "outcomes"
                and any(self.navigation.available(customer, name) for name in (READ_TOOL,))
            )
            or (
                self._knowledge_searches < _MAX_KNOWLEDGE_SEARCHES
                and self._customer_message_knowledge_searches
                < _MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE
            )
        )

    def _refresh_private_action_epoch(self) -> None:
        """Reset completed private steps after an accepted workflow revision."""

        if self.controller.planning_mode == "outcomes":
            # Fresh authoritative input may require another audit without changing
            # the customer's outcome checklist. Do not force a bookkeeping edit
            # merely to re-expose a read-only helper after the underlying facts change.
            epoch = (
                self.ledger.last_customer_message_sequence,
                self.ledger.last_tool_action_sequence,
            )
            if getattr(self, "_private_action_observation_epoch", None) != epoch:
                self._private_action_observation_epoch = epoch
                self._private_actions_completed_at_plan_revision.clear()
        if self.controller.revision == self._private_action_plan_revision:
            return
        self._private_action_plan_revision = self.controller.revision
        self._private_actions_completed_at_plan_revision.clear()

    def _current_credit_card_dispute_policies(
        self,
    ) -> tuple[tuple[str, CreditCardDisputePolicy], ...]:
        """Return prepared policies still bound to current public observations."""

        current: list[tuple[str, CreditCardDisputePolicy]] = []
        for user_id, (policy, prepared_fingerprint) in sorted(
            self._credit_card_dispute_policies.items()
        ):
            observed_fingerprint = _credit_card_dispute_observation_fingerprint(
                self.ledger,
                user_id=user_id,
            )
            if observed_fingerprint == prepared_fingerprint:
                current.append((user_id, policy))
        return tuple(current)

    def _render_prepared_policy_context(self) -> str | None:
        """Expose stable private dossier handles without replaying full documents."""

        policies = self._current_credit_card_dispute_policies()
        if not policies:
            return None
        payload = {
            "credit_card_dispute": [
                {
                    "policy_dossier_id": policy.dossier_id,
                    "user_id": user_id,
                }
                for user_id, policy in policies
            ]
        }
        return (
            "These Cayu policy dossiers were already prepared from exact public revisions "
            "and remain bound to the current observations. Reuse their identifiers for the "
            "corresponding deterministic audit; do not prepare them again.\n"
            "<prepared-policy-memory>\n"
            f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n"
            "</prepared-policy-memory>"
        )

    def _observe_successful_private_action(
        self,
        name: str,
        result: InternalToolResult,
    ) -> InternalToolResult:
        """Require a workflow sync before repeating another planned private step."""

        if result.is_error:
            return result
        self._refresh_private_action_epoch()
        self._private_actions_completed_at_plan_revision.add(name)
        return result

    async def _prepare_credit_card_dispute_policy_dossier(
        self,
        arguments: Mapping[str, Any],
    ) -> InternalToolResult:
        if not isinstance(arguments, Mapping) or set(arguments) != {"user_id"}:
            return InternalToolResult(
                content="Credit-card dispute policy preparation requires exactly user_id.",
                is_error=True,
            )
        user_id = arguments["user_id"]
        if (
            not isinstance(user_id, str)
            or not user_id.strip()
            or len(user_id.strip()) > _MAX_REWARD_USER_ID_CHARS
        ):
            return InternalToolResult(
                content=(
                    "Credit-card dispute policy preparation requires a bounded nonblank user_id."
                ),
                is_error=True,
            )
        normalized_user_id = user_id.strip()
        observation_fingerprint = _credit_card_dispute_observation_fingerprint(
            self.ledger,
            user_id=normalized_user_id,
        )
        if observation_fingerprint is None:
            return InternalToolResult(
                content=(
                    "Before policy preparation, retrieve current time, this user's complete "
                    "credit-card transactions and accounts, and credit-card dispute history. "
                    "Every observed card used by a transaction must include its opening date."
                ),
                is_error=True,
            )
        try:
            (
                content,
                evidence_refs,
            ) = await self.corpus.credit_card_dispute_policy_dossier_with_evidence()
            policy = compile_dispute_policy_dossier(content)
        except (BankingKnowledgeError, ValueError):
            return InternalToolResult(
                content=(
                    "Could not prepare and compile the complete bounded credit-card dispute "
                    "policy family from exact public revisions."
                ),
                is_error=True,
            )
        self._credit_card_dispute_policies[normalized_user_id] = (
            policy,
            observation_fingerprint,
        )
        self._turn_delivered_entry_ids.update(reference.reference_id for reference in evidence_refs)
        self.controller.observe_knowledge_evidence(evidence_refs)
        self.admission_gate.observe_knowledge_text(content)
        return InternalToolResult(
            content=(
                "Cayu credit-card dispute policies prepared from exact public revisions. "
                "Read every document. Then call controller_audit_credit_card_disputes once "
                "with every transaction in the complete intended filing batch. The audit "
                "allocates the remaining provisional-credit slots deterministically and "
                "returns fields enforced at the external-action boundary.\n" + content
            )
        )

    async def _prepare_savings_interest_policy_dossier(
        self,
        arguments: Mapping[str, Any],
    ) -> InternalToolResult:
        if not isinstance(arguments, Mapping) or set(arguments) != {"user_id", "account_id"}:
            return InternalToolResult(
                content=(
                    "Savings-interest policy preparation requires exactly user_id and account_id."
                ),
                is_error=True,
            )
        user_id = arguments["user_id"]
        account_id = arguments["account_id"]
        if (
            not isinstance(user_id, str)
            or not user_id.strip()
            or len(user_id.strip()) > _MAX_REWARD_USER_ID_CHARS
            or not isinstance(account_id, str)
            or not account_id.strip()
            or len(account_id.strip()) > _MAX_REWARD_USER_ID_CHARS
        ):
            return InternalToolResult(
                content="Savings-interest policy preparation requires bounded nonblank IDs.",
                is_error=True,
            )
        normalized_user_id = user_id.strip()
        normalized_account_id = account_id.strip()
        bank_accounts = self.ledger.bank_accounts(normalized_user_id)
        credit_cards = self.ledger.credit_card_accounts(normalized_user_id)
        if bank_accounts is None or credit_cards is None:
            return InternalToolResult(
                content="No complete parsed successful all-accounts lookup exists for this user.",
                is_error=True,
            )
        savings_account = next(
            (
                item
                for item in bank_accounts
                if item.account_id == normalized_account_id
                and item.account_class.casefold() in {"saving", "savings"}
                and item.status.casefold() == "open"
            ),
            None,
        )
        if savings_account is None:
            return InternalToolResult(
                content="account_id is not an OPEN savings account in the latest lookup.",
                is_error=True,
            )
        checking_levels = tuple(
            item.level
            for item in bank_accounts
            if item.account_class.casefold() == "checking" and item.status.casefold() == "open"
        )
        card_types = tuple(
            item.card_type for item in credit_cards if item.status.casefold() == "active"
        )
        multiple_open_savings = (
            sum(
                item.account_class.casefold() in {"saving", "savings"}
                and item.status.casefold() == "open"
                for item in bank_accounts
            )
            > 1
        )
        try:
            (
                content,
                evidence_refs,
            ) = await self.corpus.savings_interest_policy_dossier_with_evidence(
                savings_account_level=savings_account.level,
                checking_account_levels=checking_levels,
                credit_card_types=card_types,
            )
        except (BankingKnowledgeError, ValueError):
            return InternalToolResult(
                content=(
                    "Could not prepare complete bounded savings-interest policy families for "
                    "the observed account products."
                ),
                is_error=True,
            )
        payload = json.loads(content)
        dossier_id = payload["policy_dossier_id"]
        observation_fingerprint = _savings_policy_observation_fingerprint(
            self.ledger,
            user_id=normalized_user_id,
            account_id=normalized_account_id,
        )
        assert observation_fingerprint is not None
        self._savings_interest_policy_dossiers[(normalized_user_id, normalized_account_id)] = (
            dossier_id,
            observation_fingerprint,
        )
        self._turn_delivered_entry_ids.update(reference.reference_id for reference in evidence_refs)
        self.controller.observe_knowledge_evidence(evidence_refs)
        self.admission_gate.observe_knowledge_text(content)
        relationship_guidance = (
            " The current account observation has multiple OPEN savings accounts and no "
            "account-specific relationship-eligibility field. Treat a conditional 0.025% "
            "relationship component as unproven and omit it from the audit."
            if multiple_open_savings
            else (
                " When this is the customer's only OPEN Silver savings account and another "
                "OPEN bank account or ACTIVE credit card is observed, the deterministic "
                "audit supplies Silver's documented 0.025% multi-product relationship "
                "component. Do not replace it with a different rate."
                if savings_account.level.strip().casefold() == "silver account"
                else ""
            )
        )
        return InternalToolResult(
            content=(
                "Cayu savings-interest policy families prepared from exact public revisions. "
                "Read every document, use only the target savings product's rates, assign one "
                "documented rate or zero to every observed checking account and credit card, "
                "then pass this policy_dossier_id to controller_audit_savings_interest."
                + relationship_guidance
                + "\n"
                + content
            )
        )

    async def _prepare_reward_policy_dossier(
        self,
        arguments: Mapping[str, Any],
    ) -> InternalToolResult:
        if not isinstance(arguments, Mapping) or set(arguments) != {"user_id"}:
            return InternalToolResult(
                content="Reward-policy preparation requires exactly 'user_id'.",
                is_error=True,
            )
        user_id = arguments["user_id"]
        if (
            not isinstance(user_id, str)
            or not user_id.strip()
            or len(user_id.strip()) > _MAX_REWARD_USER_ID_CHARS
        ):
            return InternalToolResult(
                content="Reward-policy preparation requires a nonblank user_id.",
                is_error=True,
            )
        normalized_user_id = user_id.strip()
        observations = self.ledger.credit_card_transactions(normalized_user_id)
        if observations is None:
            return InternalToolResult(
                content=(
                    "No parsed successful credit-card transaction lookup exists for this user."
                ),
                is_error=True,
            )
        card_types = _completed_card_types(observations)
        try:
            content, evidence_refs = await self.corpus.reward_policy_dossier_with_evidence(
                card_types
            )
        except (BankingKnowledgeError, ValueError):
            return InternalToolResult(
                content=(
                    "Could not prepare a complete bounded reward-policy dossier for the "
                    "observed card types."
                ),
                is_error=True,
            )
        payload = json.loads(content)
        dossier_id = payload["policy_dossier_id"]
        self._reward_policy_dossiers[normalized_user_id] = (dossier_id, card_types)
        if self.advisory_workflow:
            directory, delivered_ids = progressive_reward_policy_view(payload)
            delivered_refs = tuple(
                ref for ref in evidence_refs if ref.reference_id in delivered_ids
            )
            self._turn_delivered_entry_ids.update(delivered_ids)
            self.controller.observe_knowledge_evidence(delivered_refs)
            for document in directory["documents"]:
                if document["text_complete"]:
                    self.admission_gate.observe_knowledge_text(document["text"])
            return InternalToolResult(
                content=(
                    "Public policy directory with bounded complete introductory articles. "
                    "Only entries with text_complete=true are delivered full-text evidence; "
                    "other entries are navigation aids and require a full revision read. "
                    "Application articles can contain rates for existing accounts: do not "
                    "discard them by title. Use section headings to locate primary terms. "
                    "Establish earning rates, category rules, exclusions, promotions and "
                    "rounding from applicable sources. An unknown rate is not zero or a "
                    "default rate, and an unresolved transaction is not a verified match. "
                    "Keep unresolved coverage explicit. The policy_dossier_id is a private "
                    "calculator handle, not a banking tool_result reference. Pass it to "
                    "controller_audit_credit_card_rewards after establishing all rates.\n"
                    + json.dumps(directory, ensure_ascii=False, sort_keys=True)
                )
            )
        self._turn_delivered_entry_ids.update(reference.reference_id for reference in evidence_refs)
        self.controller.observe_knowledge_evidence(evidence_refs)
        self.admission_gate.observe_knowledge_text(content)
        return InternalToolResult(
            content=(
                "Cayu reward-policy dossier prepared from exact public knowledge revisions. "
                "Read every document, including exclusions and promotions, then pass this "
                "policy_dossier_id to controller_audit_credit_card_rewards.\n" + content
            )
        )

    def export_state(self) -> dict[str, Any]:
        recall = self._last_recall
        context = self._work_context
        return {
            "corpus_sha256": self.corpus.corpus_sha256,
            "navigation_state": self.navigation.snapshot(),
            "knowledge_retrieval": {
                "embedding_dimensions": self.corpus.embedding_dimensions,
                "embedding_model": self.corpus.embedding_model,
                "embedding_representation": self.corpus.embedding_representation,
                "fusion_configuration": self.corpus.fusion_configuration_version,
                "mode": self.corpus.retrieval_mode,
            },
            "knowledge_searches": self._knowledge_searches,
            "knowledge_search_limit": _MAX_KNOWLEDGE_SEARCHES,
            "knowledge_search_customer_message_id": (self._knowledge_search_customer_message_id),
            "knowledge_searches_for_current_customer_message": (
                self._customer_message_knowledge_searches
            ),
            "knowledge_searches_per_customer_message_limit": (
                _MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE
            ),
            "human_transfer_policy": {
                "available_documents": self.corpus.human_transfer_policy_document_count,
                "injected_this_turn": self._human_transfer_policy_injected_this_turn,
            },
            "product_comparison_dossier": {
                "injected_documents_this_turn": (
                    self._product_comparison_documents_injected_this_turn
                ),
            },
            "credit_card_dispute_audits": len(self._credit_card_dispute_audits),
            "credit_card_dispute_policy_dossiers": len(self._credit_card_dispute_policies),
            "reward_policy_dossiers": len(self._reward_policy_dossiers),
            "savings_interest_audits": len(self._savings_interest_audits),
            "savings_interest_policy_dossiers": len(self._savings_interest_policy_dossiers),
            "semantic_plan_verifier": {
                "acceptances": self._plan_review_acceptances,
                "cache_hits": self._plan_review_cache_hits,
                "completion_tokens": self._plan_review_completion_tokens,
                "enabled": self._plan_verifier is not None,
                "explicit_operation_floor_rejections": (
                    self._plan_review_explicit_operation_floor_rejections
                ),
                "prompt_tokens": self._plan_review_prompt_tokens,
                "provider_calls": self._plan_review_provider_calls,
                "rejections": self._plan_review_rejections,
            },
            "semantic_action_verifier": {
                "acceptances": self._action_review_acceptances,
                "cache_hits": self._action_review_cache_hits,
                "completion_tokens": self._action_review_completion_tokens,
                "enabled": self._action_verifier is not None,
                "prompt_tokens": self._action_review_prompt_tokens,
                "provider_calls": self._action_review_provider_calls,
                "rejections": self._action_review_rejections,
            },
            "semantic_response_verifier": {
                "acceptances": self._response_review_acceptances,
                "cache_hits": self._response_review_cache_hits,
                "completion_tokens": self._response_review_completion_tokens,
                "enabled": self._response_verifier is not None,
                "prompt_tokens": self._response_review_prompt_tokens,
                "provider_calls": self._response_review_provider_calls,
                "rejections": self._response_review_rejections,
            },
            "last_recall": (
                None
                if recall is None
                else {
                    "candidate_count": recall.candidate_count,
                    "injected_count": recall.injected_count,
                    "record_ids": list(recall.record_ids),
                    "result_sha256": recall.result_sha256,
                    "source_coverage": [list(item) for item in recall.source_coverage],
                }
            ),
            "schema_version": 14,
            "task_id": self.task_id,
            "work_context": (
                None
                if context is None
                else {
                    "content_sha256": context.content_sha256,
                    "revision": context.revision,
                    "workflow_iteration": context.workflow_iteration,
                    "workflow_phase": context.workflow_phase,
                }
            ),
        }

    async def _publish_work_context(self) -> AgentWorkContext:
        for _ in range(_MAX_CONTEXT_PUBLICATION_ATTEMPTS):
            current = await self.work_context_store.load_work_context(self.task_id)
            expected_revision = None if current is None else current.revision
            candidate = AgentWorkContext.create(
                task_id=self.task_id,
                goal=self._goal(),
                revision=1 if current is None else current.revision + 1,
                operation_id=f"tau3-work-context-{uuid4().hex}",
                published_by="cayu-tau3-banking-memory",
                published_at=datetime.now(UTC),
                scope_ids=("tau3:banking_knowledge",),
                workflow_id="tau3-banking-half-duplex",
                workflow_phase=self._workflow_phase(),
                workflow_iteration=len(self.ledger.actions),
                entity_ids=self._entity_ids(),
                planned_action_ids=tuple(
                    action.call_id
                    for action in self.ledger.actions
                    if action.status is ToolActionStatus.PENDING
                ),
            )
            if current is not None and candidate.content_sha256 == current.content_sha256:
                self._work_context = current
                return current
            try:
                receipt = await self.work_context_store.publish_work_context(
                    candidate,
                    expected_revision=expected_revision,
                )
            except AgentWorkContextConflict:
                continue
            self._work_context = receipt.context
            return receipt.context
        raise AgentWorkContextConflict("work_context_publication_retry_exhausted")

    def _goal(self) -> str:
        active_goals = self._active_work_goals()
        if active_goals:
            latest = self.ledger.customer_messages[-1] if self.ledger.customer_messages else None
            parts = []
            if latest is not None:
                parts.append(f"Latest customer input: {latest}")
            parts.append(
                "Resolve these active customer-requested outcomes: " + " | ".join(active_goals)
            )
            goal = " ".join(parts)
            return goal if len(goal) <= _MAX_GOAL_CHARS else f"{goal[: _MAX_GOAL_CHARS - 1]}…"
        messages = self.ledger.customer_messages[-_MAX_RECENT_CONVERSATION:]
        if not messages:
            return "Resolve the customer's banking request using authorized public observations."
        goal = "Resolve these customer-stated banking requests: " + " | ".join(messages)
        return goal if len(goal) <= _MAX_GOAL_CHARS else f"{goal[: _MAX_GOAL_CHARS - 1]}…"

    def _current_query(self) -> str | None:
        if not self.ledger.customer_messages:
            return None
        latest = self.ledger.customer_messages[-1]
        active_goals = self._active_work_goals()
        if not active_goals:
            return latest
        query = f"Latest customer input: {latest}\nActive unresolved work:\n- " + "\n- ".join(
            active_goals
        )
        return (
            query
            if len(query) <= _MAX_RETRIEVAL_QUERY_CHARS
            else f"{query[: _MAX_RETRIEVAL_QUERY_CHARS - 1]}…"
        )

    def _active_work_goals(self) -> tuple[str, ...]:
        """Return stable unresolved goals that must survive conversational follow-ups."""

        if not self.workflow_controller_enabled:
            return ()
        raw_items = self.controller.export_state().get("items")
        if not isinstance(raw_items, list):
            return ()
        goals: list[str] = []
        for item in raw_items:
            if (
                not isinstance(item, Mapping)
                or item.get("status") == WorkStatus.COMPLETED.value
                or not isinstance(goal := item.get("goal"), str)
                or not goal.strip()
            ):
                continue
            goals.append(goal.strip())
        return tuple(goals)

    def _workflow_phase(self) -> str:
        if any(action.status is ToolActionStatus.PENDING for action in self.ledger.actions):
            return "awaiting_environment_results"
        if self.workflow_controller_enabled:
            items = self.controller.export_state()["items"]
            if any(item["status"] != WorkStatus.COMPLETED.value for item in items):
                return "executing_customer_request"
        return "servicing_customer"

    def _entity_ids(self) -> tuple[str, ...]:
        identities: list[str] = []
        seen: set[str] = set()
        for action in self.ledger.actions:
            for key, value in _identity_arguments(action.arguments):
                identity = f"{key}:{value}"
                if identity in seen:
                    continue
                seen.add(identity)
                identities.append(identity)
                if len(identities) >= _MAX_ENTITY_IDS:
                    return tuple(identities)
        return tuple(identities)

    @staticmethod
    def _recall_work_context(context: AgentWorkContext) -> str:
        return json.dumps(
            {
                "entity_ids": list(context.entity_ids),
                "goal": context.goal,
                "workflow_phase": context.workflow_phase,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def _render_work_context(context: AgentWorkContext) -> str:
        body = json.dumps(
            {
                "entity_ids": list(context.entity_ids),
                "goal": context.goal,
                "planned_action_ids": list(context.planned_action_ids),
                "revision": context.revision,
                "workflow_iteration": context.workflow_iteration,
                "workflow_phase": context.workflow_phase,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return f"{_WORK_CONTEXT_PREAMBLE}\n<cayu-work-context>\n{body}\n</cayu-work-context>"


def _identity_arguments(value: object) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                continue
            key = raw_key.casefold()
            if (key == "id" or key.endswith("_id")) and (
                isinstance(item, str) or type(item) is int
            ):
                normalized = str(item).strip()
                if normalized:
                    bounded = normalized[:_MAX_ENTITY_ID_CHARS]
                    found.append((key, bounded))
            else:
                found.extend(_identity_arguments(item))
    elif isinstance(value, list | tuple):
        for item in value:
            found.extend(_identity_arguments(item))
    return found


def _completed_card_types(
    observations: Sequence[CreditCardTransactionObservation],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            observation.card_type
            for observation in observations
            if observation.status.casefold() == "completed"
        )
    )


def _credit_card_dispute_observation_fingerprint(
    ledger: ConversationLedger,
    *,
    user_id: str,
) -> str | None:
    current_date = ledger.current_date
    transactions = ledger.credit_card_transactions(user_id)
    accounts = ledger.credit_card_accounts(user_id)
    history = ledger.credit_card_disputes(user_id)
    if current_date is None or transactions is None or accounts is None or history is None:
        return None
    accounts_by_card_type: dict[str, list[Any]] = {}
    for account in accounts:
        card_type = _normalized_credit_card_type(account.card_type)
        accounts_by_card_type.setdefault(card_type, []).append(account)
    observed_card_types = {
        _normalized_credit_card_type(transaction.card_type) for transaction in transactions
    }
    if any(
        len(accounts_by_card_type.get(card_type, ())) != 1
        or accounts_by_card_type[card_type][0].opened_on is None
        for card_type in observed_card_types
    ):
        return None
    payload = {
        "accounts": sorted(
            [
                account.account_id,
                account.user_id,
                account.card_type,
                account.status,
                account.opened_on.isoformat() if account.opened_on is not None else None,
            ]
            for account in accounts
        ),
        "current_date": current_date.isoformat(),
        "dispute_history": sorted(
            [
                dispute.dispute_id,
                dispute.transaction_id,
                dispute.user_id,
                dispute.submitted_on.isoformat(),
                dispute.provisional_credit_given,
            ]
            for dispute in history
        ),
        "successful_session_filings": sorted(
            [
                filing.get("transaction_id"),
                filing.get("user_id"),
                filing.get("eligible_for_provisional_credit"),
            ]
            for filing in ledger.successful_credit_card_dispute_arguments(user_id)
        ),
        "transactions": sorted(
            [
                transaction.transaction_id,
                transaction.user_id,
                transaction.card_type,
                str(transaction.amount),
                transaction.transaction_date,
                transaction.status,
            ]
            for transaction in transactions
        ),
        "user_id": user_id,
    }
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _normalized_credit_card_type(value: str) -> str:
    return " ".join(value.casefold().split()).removesuffix(" card")


def _savings_policy_observation_fingerprint(
    ledger: ConversationLedger,
    *,
    user_id: str,
    account_id: str,
) -> str | None:
    bank_accounts = ledger.bank_accounts(user_id)
    credit_cards = ledger.credit_card_accounts(user_id)
    if bank_accounts is None or credit_cards is None:
        return None
    if not any(
        item.account_id == account_id
        and item.account_class.casefold() in {"saving", "savings"}
        and item.status.casefold() == "open"
        for item in bank_accounts
    ):
        return None
    payload = {
        "account_id": account_id,
        "bank_accounts": [
            [item.account_id, item.account_class, item.level, item.status] for item in bank_accounts
        ],
        "credit_cards": [[item.account_id, item.card_type, item.status] for item in credit_cards],
        "user_id": user_id,
    }
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _plan_review_key(
    customer_messages: Sequence[str],
    plan_state: Mapping[str, Any],
    *,
    reclassifiable_item_ids: Sequence[str],
    dialogue_context: str,
) -> str:
    encoded = json.dumps(
        {
            "customer_messages": list(customer_messages),
            "plan_state": dict(plan_state),
            "dialogue_context": dialogue_context,
            "reclassifiable_item_ids": list(reclassifiable_item_ids),
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _action_review_key(
    *,
    customer_messages: Sequence[str],
    observation_context: str,
    workflow_context: str,
    calls: Sequence[ToolCallPart],
) -> str:
    """Bind the decision to every input in the stable verifier projection."""

    encoded = json.dumps(
        {
            "calls": [{"arguments": call.arguments, "tool_name": call.tool_name} for call in calls],
            "customer_messages": list(customer_messages),
            "observation_context": observation_context,
            "workflow_context": workflow_context,
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _response_review_key(
    *,
    customer_messages: Sequence[str],
    observation_context: str,
    response_state: Mapping[str, Any],
    proposed_response: str,
) -> str:
    """Bind a response decision to its complete public semantic authority."""

    encoded = json.dumps(
        {
            "customer_messages": list(customer_messages),
            "observation_context": observation_context,
            "proposed_response": proposed_response,
            "response_state": dict(response_state),
        },
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _customer_requests_credit_card_dispute(customer_messages: Sequence[str]) -> bool:
    """Return whether recent customer text requests a credit-card dispute workflow."""

    if isinstance(customer_messages, str | bytes):
        raise TypeError("customer_messages must be a sequence of strings.")
    messages = tuple(customer_messages)
    if any(not isinstance(message, str) for message in messages):
        raise TypeError("customer_messages must contain only strings.")
    combined = "\n".join(messages)
    return bool(
        _CREDIT_CARD_DISPUTE_TERM_PATTERN.search(combined)
        and _CREDIT_CARD_CONTEXT_PATTERN.search(combined)
    )


def _response_contains_credit_card_dispute(response_state: Mapping[str, Any]) -> bool:
    """Return whether the response contract contains explicit credit-card dispute work."""

    raw_items = response_state.get("items")
    if not isinstance(raw_items, list):
        return False
    return any(
        isinstance(item, Mapping)
        and _CREDIT_CARD_DISPUTE_TERM_PATTERN.search(
            " ".join(
                value for value in (item.get("id"), item.get("goal")) if isinstance(value, str)
            )
        )
        and "debit" not in str(item.get("goal", "")).casefold()
        for item in raw_items
    )


def _response_requires_credit_card_last_four(response_state: Mapping[str, Any]) -> bool:
    """Return whether a dispute response is trying to collect credit-card digits."""

    raw_items = response_state.get("items")
    if not isinstance(raw_items, list):
        return False
    return any(
        isinstance(item, Mapping)
        and isinstance(operation := item.get("next_operation"), Mapping)
        and (
            operation.get("tool_name") == _CREDIT_CARD_LAST_FOUR_TOOL
            or (
                isinstance(summary := operation.get("summary"), str)
                and (_LAST_FOUR_PATTERN.search(summary) or _CREDIT_CARD_LAST_FOUR_TOOL in summary)
            )
        )
        for item in raw_items
    )


def _credit_card_last_four_tool_handoff_account_id(
    response_state: Mapping[str, Any],
) -> str | None:
    """Return the account ID from one exact public customer-tool handoff."""

    raw_items = response_state.get("items")
    if not isinstance(raw_items, list):
        return None
    for item in raw_items:
        if (
            not isinstance(item, Mapping)
            or item.get("response_kind") != "handoff_customer_tool"
            or not isinstance(operation := item.get("next_operation"), Mapping)
            or operation.get("owner") != "customer"
            or operation.get("kind") != "tool"
            or operation.get("tool_name") != _CREDIT_CARD_LAST_FOUR_TOOL
            or not isinstance(arguments := operation.get("arguments"), Mapping)
            or set(arguments) != {"credit_card_account_id"}
            or not isinstance(account_id := arguments["credit_card_account_id"], str)
            or not account_id.strip()
        ):
            continue
        return account_id.strip()
    return None


def _restored_memory_state(messages: list[Message] | None) -> _RestoredMemoryState:
    for message in reversed(messages or ()):
        if not isinstance(message, AssistantMessage) or not isinstance(message.raw_data, Mapping):
            continue
        memory = message.raw_data.get("cayu_tau3_memory")
        if not isinstance(memory, Mapping):
            continue
        schema_version = memory.get("schema_version")
        if schema_version not in {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
            continue
        task_id = memory.get("task_id")
        searches = memory.get("knowledge_searches", 0)
        if (
            isinstance(task_id, str)
            and _TASK_ID_PATTERN.fullmatch(task_id)
            and type(searches) is int
            and searches >= 0
        ):
            verifier = memory.get("semantic_plan_verifier")
            counts = (
                _restored_plan_review_counts(verifier, schema_version=schema_version)
                if schema_version in {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
                else {}
            )
            action_counts = (
                _restored_action_review_counts(memory.get("semantic_action_verifier"))
                if schema_version >= 12
                else {}
            )
            response_counts = (
                _restored_response_review_counts(memory.get("semantic_response_verifier"))
                if schema_version >= 13
                else {}
            )
            search_epoch = _restored_search_epoch(memory, schema_version=schema_version)
            if (
                search_epoch is None
                or search_epoch["knowledge_searches_for_current_customer_message"] > searches
            ):
                continue
            return _RestoredMemoryState(
                task_id=task_id,
                navigation_state=memory.get("navigation_state"),
                knowledge_searches=min(searches, _MAX_KNOWLEDGE_SEARCHES),
                **search_epoch,
                **counts,
                **action_counts,
                **response_counts,
            )
    return _RestoredMemoryState()


def _restored_plan_review_counts(value: object, *, schema_version: int) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    names = {
        "provider_calls": "plan_review_provider_calls",
        "acceptances": "plan_review_acceptances",
        "rejections": "plan_review_rejections",
        "cache_hits": "plan_review_cache_hits",
        "prompt_tokens": "plan_review_prompt_tokens",
        "completion_tokens": "plan_review_completion_tokens",
    }
    restored: dict[str, int] = {}
    for serialized_name, field_name in names.items():
        serialized = value.get(serialized_name)
        if type(serialized) is not int or serialized < 0:
            return {}
        restored[field_name] = serialized
    if (
        restored["plan_review_acceptances"] + restored["plan_review_rejections"]
        > restored["plan_review_provider_calls"]
    ):
        return {}
    floor_rejections = value.get("explicit_operation_floor_rejections", 0)
    if schema_version in {5, 6, 7, 8, 9, 10, 11, 12, 13, 14} and (
        type(floor_rejections) is not int
        or not 0 <= floor_rejections <= restored["plan_review_rejections"]
    ):
        return {}
    restored["plan_review_explicit_operation_floor_rejections"] = floor_rejections
    return restored


def _restored_action_review_counts(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    names = {
        "provider_calls": "action_review_provider_calls",
        "acceptances": "action_review_acceptances",
        "rejections": "action_review_rejections",
        "cache_hits": "action_review_cache_hits",
        "prompt_tokens": "action_review_prompt_tokens",
        "completion_tokens": "action_review_completion_tokens",
    }
    restored: dict[str, int] = {}
    for serialized_name, field_name in names.items():
        serialized = value.get(serialized_name)
        if type(serialized) is not int or serialized < 0:
            return {}
        restored[field_name] = serialized
    decisions = restored["action_review_acceptances"] + restored["action_review_rejections"]
    if decisions > restored["action_review_provider_calls"] * _MAX_REVIEWED_ACTION_CALLS_PER_BATCH:
        return {}
    return restored


def _restored_response_review_counts(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    names = {
        "provider_calls": "response_review_provider_calls",
        "acceptances": "response_review_acceptances",
        "rejections": "response_review_rejections",
        "cache_hits": "response_review_cache_hits",
        "prompt_tokens": "response_review_prompt_tokens",
        "completion_tokens": "response_review_completion_tokens",
    }
    restored: dict[str, int] = {}
    for serialized_name, field_name in names.items():
        serialized = value.get(serialized_name)
        if type(serialized) is not int or serialized < 0:
            return {}
        restored[field_name] = serialized
    decisions = restored["response_review_acceptances"] + restored["response_review_rejections"]
    if decisions > restored["response_review_provider_calls"]:
        return {}
    return restored


def _restored_search_epoch(
    value: Mapping[str, Any],
    *,
    schema_version: int,
) -> dict[str, str | int | None] | None:
    if schema_version < 9:
        return {
            "knowledge_search_customer_message_id": None,
            "knowledge_searches_for_current_customer_message": 0,
        }
    customer_message_id = value.get("knowledge_search_customer_message_id")
    searches = value.get("knowledge_searches_for_current_customer_message")
    if customer_message_id is not None and (
        not isinstance(customer_message_id, str)
        or not _CUSTOMER_MESSAGE_ID_PATTERN.fullmatch(customer_message_id)
    ):
        return None
    if (
        type(searches) is not int
        or not 0 <= searches <= _MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE
    ):
        return None
    return {
        "knowledge_search_customer_message_id": customer_message_id,
        "knowledge_searches_for_current_customer_message": searches,
    }


__all__ = ["BankingMemoryRuntime", "BankingMemorySession"]
