"""Banking application assembly on Cayu's native runtime extension points.

No model loop lives here. The application supplies banking policy/evidence
semantics; Cayu owns scheduling, tool authorization, transcript and completion.
"""

from __future__ import annotations

import json
import secrets
import tempfile
from collections.abc import Awaitable, Callable
from itertools import islice
from pathlib import Path
from typing import Any

from cayu import (
    AgentSpec,
    BeforeStopDecision,
    CayuApp,
    Environment,
    EnvironmentSpec,
    LoopPolicy,
    Message,
    RequestFootprintConfig,
    ToolCallPart,
)
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    ContextBuildResult,
    ContextRequest,
    ObservedDeltaContextEstimator,
    RuntimeManagedContextPolicy,
)
from cayu.memory.recall import RecallSourceUnavailable
from cayu.observability.hooks import (
    BeforeToolCallDecision,
    BeforeToolCallHookContext,
    RuntimeHook,
    ToolCallHookContext,
)
from cayu.providers import ModelProvider
from cayu.runtime.loop_policies import BeforeStopContext
from cayu.sessions.base import SessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.exposure import (
    ToolExposureDecision,
    ToolExposurePolicy,
    ToolExposurePolicyRequest,
)
from cayu.tools.policy import (
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from jsonschema import validators
from referencing import Registry
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from cayu_tau3_banking.active_evidence import (
    active_evidence_context,
    recent_tool_documentation_references,
)
from cayu_tau3_banking.admission import BankingToolAdmissionGate
from cayu_tau3_banking.admission_progress import AdmissionProgress
from cayu_tau3_banking.advice import PUBLIC_RESEARCH_TOOLS, lean_banking_prompt
from cayu_tau3_banking.agent import banking_system_prompt
from cayu_tau3_banking.agent_memory import MEMORY_AGENT_OPERATION_TOOL_NAMES, BankingMemoryRuntime
from cayu_tau3_banking.bridge import (
    CompositeExternalToolCallGate,
    InternalRemediationLifecycle,
    internal_tool_result_projection,
)
from cayu_tau3_banking.compaction import banking_compactor
from cayu_tau3_banking.context_pressure import (
    COMPACTION_TRIGGER,
    RECENT_CONTEXT_TARGET,
    WARNING_RESERVE,
    warn_before_compaction,
)
from cayu_tau3_banking.controller import (
    HUMAN_TRANSFER_TOOL,
    SYNC_WORK_PLAN_TOOL,
    UPDATE_WORK_PLAN_TOOL,
    BankingWorkflowController,
)
from cayu_tau3_banking.controller_contract import ExecutionContractGate, advisory_banking_prompt
from cayu_tau3_banking.customer_sources import customer_source_context
from cayu_tau3_banking.decision_notes import DECISION_NOTE_SCHEMA, DECISION_NOTE_TOOL, DecisionFacts
from cayu_tau3_banking.decision_packets import PACKET_INSTRUCTIONS, DecisionPacketGate
from cayu_tau3_banking.execution_receipts import ExecutionReceipts
from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL
from cayu_tau3_banking.ledger import ConversationLedger, ToolActionStatus
from cayu_tau3_banking.planning_progress import PlanningProgress, plan_fingerprint
from cayu_tau3_banking.runtime_agent import RuntimeResources
from cayu_tau3_banking.runtime_recall import banking_recall_policy, observe_delivered_recall
from cayu_tau3_banking.runtime_tools import HostToolBroker
from cayu_tau3_banking.saved_evidence import (
    SAVED_EVIDENCE_SCHEMA,
    SAVED_EVIDENCE_TOOL,
    SavedEvidence,
)
from cayu_tau3_banking.task_memory import (
    MARKDOWN_INSTRUCTIONS,
    UPDATE_CONSTRAINTS,
    TaskMemory,
    normalize_constraint_arguments,
)
from cayu_tau3_banking.task_memory import schemas as task_memory_schemas
from cayu_tau3_banking.transaction_binding import bind_transaction_fields
from cayu_tau3_banking.verification_guidance import VERIFICATION_GUIDANCE


def _validator(spec: ToolSpec):
    schema = spec.model_dump(mode="json")["input_schema"]
    validator_type = validators.validator_for(schema)
    validator_type.check_schema(schema)
    return validator_type(schema, registry=Registry())


class BankingRuntimeSession:
    """Loop-confined banking projections, shared by native policies and tools.

    Evidence uses Cayu's model call IDs, never transport request IDs. Customer
    evidence enters only through the adapter observer, not arbitrary transcript
    user messages (which can include runtime-authored correction instructions).
    """

    def __init__(
        self,
        memory_runtime: BankingMemoryRuntime,
        host_specs: tuple[ToolSpec, ...],
        *,
        planning_mode: str = "exact",
        advice_mode: str = "controller",
        controller_mode: str = "enforced",
        decision_packets: bool = False,
        task_memory_mode: str | None = None,
        task_memory_directory: Path | None = None,
    ):
        if controller_mode not in {"enforced", "advisory"}:
            raise ValueError("controller_mode must be enforced or advisory")
        if controller_mode == "advisory" and (planning_mode != "outcomes" or advice_mode != "lean"):
            raise ValueError("Advisory control requires native outcome planning and lean advice")
        if decision_packets and controller_mode != "advisory":
            raise ValueError("Decision packets require advisory outcome mode")
        if task_memory_mode is not None and (controller_mode != "advisory" or decision_packets):
            raise ValueError("Task memory candidates require advisory control without packets")
        self.task_memory_mode = task_memory_mode
        self.advisory_controller = controller_mode == "advisory"
        self.lightweight_workflow = self.advisory_controller and task_memory_mode == "constraints"
        if advice_mode not in {"controller", "lean"}:
            raise ValueError("advice_mode must be controller or lean")
        if advice_mode == "lean" and planning_mode != "outcomes":
            raise ValueError("Lean advice requires outcome planning")
        self.advice_mode = advice_mode
        if planning_mode == "outcomes" and any(
            (
                memory_runtime.plan_verifier,
                memory_runtime.action_verifier,
                memory_runtime.response_verifier,
            )
        ):
            raise ValueError("Outcome planning does not use legacy semantic veto reviewers")
        self.host_names = frozenset(spec.name for spec in host_specs)
        self.host_validators = {spec.name: _validator(spec) for spec in host_specs}
        self.ledger = ConversationLedger(interpret_customer_intent=not self.advisory_controller)
        self.admission = BankingToolAdmissionGate(
            self.ledger,
            require_knowledge_evidence=True,
            authoritative_only=self.advisory_controller,
        )
        self.controller = BankingWorkflowController(
            self.ledger,
            available_environment_tool_names=self.host_names,
            available_private_tool_names=MEMORY_AGENT_OPERATION_TOOL_NAMES,
            knowledge_search_available=True,
            verified_capability_gap=self.admission.has_verified_offer_capability_gap,
            planning_mode=planning_mode,
            enforce_comparison_coverage=not self.advisory_controller,
        )
        self.memory = memory_runtime.new_session(
            ledger=self.ledger,
            controller=self.controller,
            admission_gate=self.admission,
            message_history=None,
            ledger_enabled=True,
            workflow_controller_enabled=True,
        )
        self.controller.allow_retrospective_outcomes = task_memory_mode == "constraints"
        self.controller.lightweight_advisory = self.lightweight_workflow
        self.memory.advisory_workflow = self.advisory_controller
        self.memory.navigation.allow_delivered_reuse = self.advisory_controller
        execution_gate = ExecutionContractGate(memory_runtime.corpus, self.ledger)
        self.packet_gate = (
            DecisionPacketGate(self.controller, self.ledger) if decision_packets else None
        )
        self.execution_receipts = ExecutionReceipts(self.ledger, self.controller) if self.advisory_controller and not decision_packets else None
        self.operation_guidance = memory_runtime.corpus.operation_guidance
        self.decision_facts = DecisionFacts(self.ledger, self.controller) if self.advisory_controller else None
        self.task_memory = TaskMemory(task_memory_mode, task_memory_directory or Path(tempfile.mkdtemp(prefix="tau-task-memory-")), self.ledger, self.controller, knowledge_reference_resolver=self.memory.navigation.resolve_knowledge_reference) if task_memory_mode else None
        if self.lightweight_workflow:
            self.task_memory.enforce_notes = False
        if task_memory_mode == "markdown":
            self.execution_receipts = None
            self.decision_facts = None
            self.memory.workflow_controller_enabled = False
        self.saved_evidence = SavedEvidence(self.ledger, self.decision_facts) if self.decision_facts is not None else None
        execution_gate.decision_facts = None if self.lightweight_workflow else self.decision_facts
        packet_gates = (self.packet_gate,) if self.packet_gate else ()
        self.gate = CompositeExternalToolCallGate(
            (self.admission, self.operation_guidance, execution_gate, *packet_gates)
            if self.advisory_controller
            else (self.controller, self.admission, self.memory),
            denial_required_tool_names=()
            if self.advisory_controller
            else (UPDATE_WORK_PLAN_TOOL, SYNC_WORK_PLAN_TOOL),
        )
        self.batch_gate = CompositeExternalToolCallGate(
            (self.admission, self.operation_guidance, *packet_gates)
            if self.advisory_controller
            else (self.controller, self.admission)
        )
        self.memory_gate = CompositeExternalToolCallGate(
            (execution_gate,) if self.advisory_controller else (self.memory,)
        )
        self.remediations = InternalRemediationLifecycle()
        self.private_calls = 0
        self.planning_progress = PlanningProgress()
        self.admission_progress = AdmissionProgress(fatal=not self.advisory_controller)
        self.recall_dirty = True
        self.recall_context: str | None = None
        self.correction: str | None = None
        self.available_private_names: frozenset[str] = frozenset()
        self.model_step_id: str | None = None
        self.round_host_calls: dict[str, ToolCallPart] = {}
        self.native_recall = False
        self.focused_bundle = None
        self.lexical_navigation_reserved_tokens = 0
        self.session_store = None
        self.session_id = None
        self.observed_recall_exposures: set[str] = set()
        self.structured_findings: list[dict[str, Any]] = []
        self.refresh_exposure()

    def record_structured_findings(
        self, *, tool_name: str, tool_call_id: str, structured: dict[str, Any] | None
    ) -> None:
        """Retain tool-declared findings without interpreting free-form text."""
        if not structured:
            return
        findings = structured.get("findings")
        if not isinstance(findings, list):
            return
        for finding in findings[:32]:
            if not isinstance(finding, dict):
                continue
            self.structured_findings.append(
                {
                    # This is an internal calculation, not an executed banking
                    # call in the evidence ledger. Never advertise its private
                    # id as a tool_result citation the plan cannot validate.
                    "calculation_id": tool_call_id,
                    "finding": finding,
                    "kind": structured.get("finding_kind", "unspecified"),
                    "source_tool": tool_name,
                }
            )
        self.structured_findings = self.structured_findings[-128:]

    def refresh_exposure(self) -> None:
        # Mutating bookkeeping runs at application transitions, never inside
        # Cayu's pure ToolExposurePolicy.select callback.
        self.available_private_names = frozenset(
            schema["name"] for schema in self.memory.tool_schemas()
        )
        if self.decision_facts is not None:
            self.available_private_names |= {DECISION_NOTE_TOOL, SAVED_EVIDENCE_TOOL}
        if self.task_memory_mode == "markdown":
            self.available_private_names &= PUBLIC_RESEARCH_TOOLS - {"cayu_check_comparison_facts"}
        if self.task_memory is not None:
            self.available_private_names |= {x["name"] for x in task_memory_schemas(self.task_memory_mode)}

    async def history(self, messages: tuple[Message, ...]) -> None:
        for message in messages:
            text = "".join(part.text for part in message.content if part.type == "text")
            if message.role == "user":
                self.ledger.observe_input(UserMessage.text(text))
            elif message.role == "assistant":
                self.ledger.observe_assistant(AssistantMessage.text(text))
            else:
                raise ValueError("Banking history must contain only customer/assistant text.")

    async def customer(self, text: str) -> None:
        self.ledger.observe_input(UserMessage.text(text))
        self.controller.reconcile_conditions()
        self.planning_progress.reset()
        self.admission_progress.reset()
        self.recall_dirty = True
        self.correction = None
        self.refresh_exposure()

    async def reply(self, text: str) -> None:
        self.ledger.observe_assistant(AssistantMessage.text(text))


class BankingToolExposure(ToolExposurePolicy):
    def __init__(self, state: BankingRuntimeSession):
        self.state = state

    def select(self, request: ToolExposurePolicyRequest) -> ToolExposureDecision:
        names = self.state.available_private_names
        if self.state.advisory_controller or self.state.controller.has_work_items():
            names |= self.state.host_names
        else:
            names = frozenset({SYNC_WORK_PLAN_TOOL})
            if self.state.advice_mode == "lean":
                names |= self.state.available_private_names & PUBLIC_RESEARCH_TOOLS
        return ToolExposureDecision(
            profile_id=(
                "cayu-tau3:banking-runtime:v1:lean"
                if self.state.advice_mode == "lean"
                else "cayu-tau3:banking-runtime:v1"
            ),
            tool_names=tuple(tool.name for tool in request.eligible_tools if tool.name in names),
        )


class BankingRuntimeToolPolicy(ToolPolicy):
    def __init__(self, state: BankingRuntimeSession):
        self.state = state

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        state = self.state
        await observe_delivered_recall(state)
        try:
            bound_arguments = bind_transaction_fields(state.ledger, request.tool_name, request.arguments)
        except ValueError as error:
            state.correction = str(error)
            state.admission_progress.observe_denial(state.model_step_id or request.tool_call_id)
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason=str(error),
                                    metadata={"diagnostic_code": "transaction_source_mismatch"})
        if request.tool_name not in state.host_names:
            state.refresh_exposure()
            allowed = request.tool_name in state.available_private_names and (
                state.controller.has_work_items()
                or state.advisory_controller
                or request.tool_name == SYNC_WORK_PLAN_TOOL
                or (state.advice_mode == "lean" and request.tool_name in PUBLIC_RESEARCH_TOOLS)
            )
            reason = None if allowed else "Private operation is unavailable in the current plan."
            diagnostic_code = None if allowed else "private_tool_unavailable"
        else:
            call = ToolCallPart(
                tool_call_id=request.tool_call_id,
                tool_name=request.tool_name,
                arguments=bound_arguments,
            )
            previous = state.round_host_calls.get(request.tool_call_id)
            if previous is not None and previous != call:
                raise ValueError("An authorized round call changed its arguments.")
            state.round_host_calls[request.tool_call_id] = call
            # Cayu authorizes an entire round before executing it. Preserve
            # banking batch constraints (one item per call, one customer-tool
            # delivery per batch) instead of reviewing each proposal in isolation.
            decisions = await state.batch_gate.review(tuple(state.round_host_calls.values()))
            decision = next(d for d in decisions if d.tool_call_id == request.tool_call_id)
            if decision.allowed:
                # Only deterministic guards need the prefix. Re-reviewing every
                # prefix through optional model verifiers would multiply spend.
                (decision,) = await state.memory_gate.review((call,))
            allowed, reason = decision.allowed, decision.reason
            diagnostic_code = decision.diagnostic_code
            if allowed and state.task_memory is not None:
                memory_error = state.task_memory.pre_action_error(request.tool_name, bound_arguments)
                if memory_error:
                    allowed, reason, diagnostic_code = False, memory_error, "task_memory_review"
        if not allowed:
            state.admission_progress.observe_denial(state.model_step_id or request.tool_call_id)
            # Runtime policy denial text is intentionally privacy-safe. Supply
            # bounded application correction in model context, not event metadata.
            state.correction = (reason or "Operation denied by banking policy.")[:4000]
        return ToolPolicyResult(
            decision=ToolPolicyDecision.ALLOW if allowed else ToolPolicyDecision.DENY,
            reason=reason,
            metadata={
                "controller_mode": "advisory" if state.advisory_controller else "enforced",
                "owner": "banking_adapter",
                "diagnostic_code": diagnostic_code,
                "decision_class": "allow" if allowed else "hard_deny",
                **(
                    {
                        "customer_confirmation_owner": "acting_model",
                        "customer_confirmation_semantically_validated": False,
                        "retained_customer_message_count": len(state.ledger.customer_message_ids),
                        "latest_customer_message_id": (
                            state.ledger.customer_message_ids[-1]
                            if state.ledger.customer_message_ids
                            else None
                        ),
                    }
                    if request.tool_name == "log_verification"
                    else {}
                ),
            },
        )


class BankingEvidenceHook(RuntimeHook):
    def __init__(self, state: BankingRuntimeSession):
        self.state = state

    async def before_tool_call(
        self, context: BeforeToolCallHookContext
    ) -> BeforeToolCallDecision | None:
        try:
            arguments = bind_transaction_fields(self.state.ledger, context.tool_name, context.arguments)
        except ValueError as error:
            self.state.correction = str(error)
            return BeforeToolCallDecision(action="block", block_reason=str(error))
        if context.tool_name in self.state.host_names:
            if self.state.task_memory is not None:
                error = self.state.task_memory.pre_action_error(context.tool_name, arguments)
                if error:
                    self.state.correction = error
                    return BeforeToolCallDecision(action="block", block_reason=error)
            if not self.state.host_validators[context.tool_name].is_valid(arguments):
                self.state.admission_progress.observe_denial(
                    self.state.model_step_id or context.tool_call_id
                )
                return BeforeToolCallDecision(
                    action="block", block_reason="Host tool arguments do not satisfy its schema."
                )
            # Earlier calls/private plan updates in the same sequential round
            # may have consumed or changed the authority approved at round start.
            (decision,) = await self.state.gate.review(
                (
                    ToolCallPart(
                        tool_call_id=context.tool_call_id,
                        tool_name=context.tool_name,
                        arguments=arguments,
                    ),
                )
            )
            if not decision.allowed:
                self.state.admission_progress.observe_denial(
                    self.state.model_step_id or context.tool_call_id
                )
                self.state.correction = (decision.reason or "Banking authority changed.")[:4000]
                return BeforeToolCallDecision(
                    action="block",
                    block_reason=(
                        "Banking execution check rejected dispatch: "
                        + (decision.diagnostic_code or "unspecified_contract")
                    ),
                )
            self.state.ledger.observe_assistant(
                AssistantMessage.text(
                    "",
                    tool_calls=[
                        ToolCall(
                            id=context.tool_call_id,
                            name=context.tool_name,
                            arguments=arguments,
                        )
                    ],
                )
            )
            # Capture satisfied preconditions in both modes; this records history,
            # it does not turn advisory conditions into dispatch vetoes.
            self.state.controller.before_condition_execution(
                ToolCallPart(tool_call_id=context.tool_call_id,
                             tool_name=context.tool_name, arguments=arguments)
            )
        if arguments != context.arguments:
            return BeforeToolCallDecision(action="proceed_modified", modified_arguments=arguments)
        return None

    async def after_tool_call(self, context: ToolCallHookContext) -> None:
        state = self.state
        if context.tool_name not in state.host_names:
            return
        # A policy-denied call never reached the before-execution hook. Do not
        # turn a synthetic denial into an observed environment action/result.
        if state.ledger.tool_action_name(context.tool_call_id) is None:
            return
        state.ledger.observe_input(
            ToolMessage(
                role="tool",
                id=context.tool_call_id,
                content=context.result.content,
                error=context.result.is_error,
            )
        )
        state.controller.after_condition_execution(context.tool_call_id)
        state.controller.reconcile_conditions()
        if state.execution_receipts is not None:
            state.execution_receipts.observe_result(context.tool_call_id)
        if state.packet_gate is not None:
            state.packet_gate.observe_result(context.tool_call_id)
        if state.decision_facts is not None:
            state.decision_facts.observe_result(context.tool_call_id)
        if state.saved_evidence is not None:
            state.saved_evidence.observe_result(context.tool_call_id, context.result.content)
        if not context.result.is_error:
            state.planning_progress.reset()
        if (
            not state.advisory_controller
            and context.tool_name == HUMAN_TRANSFER_TOOL
            and state.ledger.tool_action_status(context.tool_call_id) is ToolActionStatus.SUCCEEDED
        ):
            state.controller.reconcile_successful_human_transfer(context.tool_call_id)
        state.recall_dirty = True
        state.refresh_exposure()


class BankingNativeTool(Tool):
    """One registered private memory/work operation; never forwarded to τ."""

    def __init__(self, spec: ToolSpec, state: BankingRuntimeSession):
        super().__init__(spec)
        self.state = state
        self.validator = _validator(spec)

    async def run(self, ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
        normalizations = []
        if self.name == UPDATE_CONSTRAINTS and self.state.task_memory_mode == "constraints":
            arguments, normalizations = normalize_constraint_arguments(arguments)
        validator = self.validator
        if self.name == DECISION_NOTE_TOOL:
            if self.state.task_memory_mode == "constraints" and isinstance(arguments.get("facts"), list) and any(f.get("conflict") for f in arguments["facts"] if isinstance(f, dict)):
                return ToolResult(content="Record adverse interactions once in cayu_update_constraints with relation=adverse_interaction and conflict. These are source-linked working interpretations, not execution permission; do not duplicate them in decision facts.", is_error=True)
            # Keep typed model-facing guidance, but validate fact entries separately
            # so one bad optional note cannot discard the remaining useful facts.
            import copy

            from jsonschema import Draft202012Validator
            transport = copy.deepcopy(DECISION_NOTE_SCHEMA["input_schema"])
            transport["properties"]["facts"]["items"] = {}
            validator = Draft202012Validator(transport)
        errors = list(islice(validator.iter_errors(arguments), 3))
        if errors:
            details = [
                {
                    "path": list(error.absolute_path),
                    "rule": error.validator,
                    "constraint": (
                        error.validator_value
                        if error.validator == "enum" or isinstance(error.validator_value, (str, int, float, bool))
                        else None
                    ),
                    "message": (
                        error.message[:400]
                        if error.validator in {"required", "additionalProperties"}
                        else f"Field violates {error.validator}; no operation executed."
                    ),
                }
                for error in errors
            ]
            return ToolResult(
                content="Private tool arguments do not satisfy its schema. Repair these fields: "
                + json.dumps(details, ensure_ascii=False),
                is_error=True,
            )
        state = self.state
        if self.name in {SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL} and "expected_revision" in arguments:
            if arguments["expected_revision"] != state.controller.revision:
                return ToolResult(content=f"Stale outcome-plan revision; current expected_revision is {state.controller.revision}. This is separate from task-note revision. Recheck the current plan before retrying. No plan was changed.", is_error=True)
            arguments = {k: v for k, v in arguments.items() if k != "expected_revision"}
        state.refresh_exposure()
        if self.name not in state.available_private_names:
            return ToolResult(content="Private operation is no longer available.", is_error=True)
        if state.task_memory is not None and self.name in {x["name"] for x in task_memory_schemas(state.task_memory_mode)}:
            try:
                result = state.task_memory.invoke(self.name, arguments)
            except ValueError as error:
                return ToolResult(content=str(error), is_error=True)
            state.correction = None
            if normalizations:
                result["normalizations"] = [*result.get("normalizations", []), *normalizations]
            return ToolResult(content=json.dumps(result))
        is_plan = self.name in {SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL}
        if self.name == DECISION_NOTE_TOOL:
            try:
                receipt = state.decision_facts.apply_patch(arguments)
            except ValueError as error:
                return ToolResult(content=str(error), is_error=True)
            return ToolResult(
                content=json.dumps(receipt),
                is_error=receipt["status"] == "rejected",
                structured=receipt,
            )
        if self.name == SAVED_EVIDENCE_TOOL:
            try:
                receipt = state.saved_evidence.read(arguments)
                return ToolResult(content=json.dumps(receipt), is_error=receipt.get("status") == "rejected")
            except ValueError as error:
                return ToolResult(content=str(error), is_error=True)
        before = plan_fingerprint(state.controller.export_state()["items"]) if is_plan else ""
        try:
            result = await state.memory.invoke(
                self.name,
                arguments,
                allow_unplanned_public_research=state.advice_mode == "lean",
            )
        except (RecallSourceUnavailable, ValueError) as error:
            if self.name != "cayu_search_banking_knowledge":
                raise
            cause = error
            seen = set()
            while cause is not None and id(cause) not in seen:
                seen.add(id(cause))
                if isinstance(cause, RecallSourceUnavailable):
                    break
                cause = cause.__cause__ or cause.__context__
            else:
                raise
            return ToolResult(
                content="Knowledge retrieval is unavailable. Retry within the remaining search budget, "
                "use lexical search, or report the missing evidence. No banking operation executed.",
                is_error=True,
                structured={"failure_category": "retrieval_unavailable"},
            )
        if state.controller.planning_mode == "outcomes":
            if is_plan:
                state.planning_progress.observe_plan(
                    before,
                    plan_fingerprint(state.controller.export_state()["items"]),
                    failed=result.is_error,
                )
            else:
                state.planning_progress.observe_private_result(
                    self.name, result.content, failed=result.is_error
                )
        if not result.is_error and self.name in {SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL}:
            state.correction = None
        state.private_calls += 1
        state.remediations.observe((result,), round_number=state.private_calls)
        content, structured = internal_tool_result_projection(result)
        state.record_structured_findings(
            tool_name=self.name,
            tool_call_id=ctx.idempotency_key or self.name,
            structured=structured,
        )
        if state.saved_evidence is not None and self.name == KNOWLEDGE_SEARCH_TOOL and not result.is_error:
            state.saved_evidence.observe_search_documents(content)
        if state.saved_evidence is not None and self.name == "cayu_read_product_documents" and not result.is_error:
            content = state.saved_evidence.project_documents(content, targeted="source_span" in arguments)
            # Do not retain a second full-body copy through structured projection.
            if structured is not None:
                structured = None
        state.refresh_exposure()
        return ToolResult(content=content, is_error=result.is_error, structured=structured)


class BankingRuntimeContext(RuntimeManagedContextPolicy):
    def __init__(
        self,
        state: BankingRuntimeSession,
        *,
        compaction: CheckpointCompactionContextPolicy,
    ):
        self.state = state
        self.base = compaction

    async def build_with_checkpoint(
        self, request: ContextRequest, *, checkpoint: dict[str, Any] | None
    ) -> ContextBuildResult:
        state = self.state
        if not state.advisory_controller:
            state.planning_progress.check()
        state.admission_progress.check()
        if request.model_step_id != state.model_step_id:
            state.session_id = request.session.id
            state.model_step_id = request.model_step_id
            state.round_host_calls.clear()
        if state.recall_dirty:
            state.recall_context = await state.memory.prepare_context(
                include_state=False, automatic_recall=not state.native_recall and state.focused_bundle is None
            )
            state.recall_dirty = False
        if state.task_memory_mode == "markdown":
            from cayu_tau3_banking.agent_memory import (
                _MAX_KNOWLEDGE_SEARCHES,
                _MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE,
            )
            customer_id = state.ledger.customer_message_ids[-1] if state.ledger.customer_message_ids else None
            capacity = {k: v for k, v in state.memory.navigation.remaining(customer_id).items() if k in state.available_private_names}
            capacity["cayu_search_banking_knowledge"] = {
                "customer_calls": _MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE - (state.memory._customer_message_knowledge_searches if state.memory._knowledge_search_customer_message_id == customer_id else 0),
                "conversation_calls": _MAX_KNOWLEDGE_SEARCHES - state.memory._knowledge_searches,
            }
            current = "\n".join((state.ledger.render_context(), "Remaining retrieval calls: " + json.dumps(capacity), state.memory.navigation.context()))
        else:
            current = await state.memory.runtime_context()
        state.refresh_exposure()
        # Put primary customer sources before derived plans and summaries in the
        # fresh projection; neither plan prose nor compaction can redefine them.
        pieces = []
        if state.advisory_controller:
            pieces.append(customer_source_context(state.ledger))
            from cayu.deadlines import current_execution_deadline

            step_limit = request.metadata.get("tau_runtime_max_steps")
            progress = f"Current acting step in this interaction: {request.step}."
            if type(step_limit) is int and step_limit > 0:
                progress += f" Limit: {step_limit}; remaining after this step: {max(0, step_limit - request.step)}."
            pieces.append(progress + " " + current_execution_deadline().model_context())
        pieces.append(current)
        if state.advisory_controller and state.task_memory_mode != "markdown":
            pieces.append(state.operation_guidance.context())
        if state.task_memory is not None:
            pieces.append(state.task_memory.context(state.model_step_id))
        if state.execution_receipts is not None and state.execution_receipts.receipts:
            pieces.append(state.execution_receipts.context())
        if state.decision_facts is not None and state.decision_facts.facts:
            pieces.append("<actor_facts meaning_verified_by_model_only='true'>" + json.dumps(state.decision_facts.context_snapshot()) + "</actor_facts>")
        if (state.decision_facts is not None and any(f.get("conflict") for f in state.decision_facts.facts.values())) or (state.task_memory is not None and state.task_memory_mode == "constraints" and any(r.get("conflict") for r in state.task_memory.data["constraints"].values())):
            pieces.append("<conflict_dialogue>" + state.ledger.render_dialogue_context() + "</conflict_dialogue>")
        if state.saved_evidence is not None:
            pieces.append("<saved_evidence>" + state.saved_evidence.context() + "</saved_evidence>")
        if state.focused_bundle is not None:
            await state.focused_bundle.prepare(state.memory.corpus)
            pieces.append(state.focused_bundle.text)
            state.controller.observe_knowledge_evidence(state.focused_bundle.references)
            state.admission.observe_knowledge_search()
        active_evidence = await active_evidence_context(
            state.memory.corpus, tuple(
                ref for ref in state.controller.active_knowledge_references()
                if state.focused_bundle is None
                or ref.reference_id not in {r.reference_id for r in state.focused_bundle.references}
            )
        )
        if active_evidence:
            pieces.append(active_evidence)
        tool_documentation = await active_evidence_context(
            state.memory.corpus,
            recent_tool_documentation_references(
                state.memory.corpus, state.ledger, state.controller
            ),
            max_documents=4,
            max_bytes=12_000,
            purpose="tool_contracts",
        )
        if tool_documentation:
            pieces.append(tool_documentation)
        if state.recall_context:
            pieces.append(state.recall_context)
        if state.structured_findings:
            pieces.append(
                "<structured_findings>\n"
                + json.dumps(
                    {
                        "notice": (
                            "These are derived findings from completed internal tools, "
                            "not customer authorization or a reference answer. "
                            "A calculation_id identifies private bookkeeping and is not a "
                            "plan evidence_ref. Cite the underlying executed banking "
                            "tool_result and delivered knowledge_revision sources in the plan. "
                            "Preserve material unresolved findings and ask the customer before "
                            "a consequential action when scope is unclear."
                        ),
                        "items": state.structured_findings,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n</structured_findings>"
            )
        if state.correction:
            pieces.append("Banking execution correction: " + state.correction)
        if state.packet_gate is not None and state.packet_gate.receipts:
            pieces.append("<execution_receipts>" + json.dumps(state.packet_gate.receipts) + "</execution_receipts>")
        if state.planning_progress.warning and not state.advisory_controller:
            pieces.append(state.planning_progress.warning)
        if state.admission_progress.warning:
            pieces.append(state.admission_progress.warning)
        context = Message.text(
            "user", "<banking_context>\n" + "\n\n".join(pieces) + "\n</banking_context>"
        )
        base = self.base
        if base.compact_after_estimated_context_tokens is not None:
            # Reserve room for the exact transient evidence projection without
            # inserting it into transcript-indexed compaction inputs. Those
            # indexes must continue referring to the durable Cayu transcript.
            evidence_tokens = ObservedDeltaContextEstimator().estimate_message_tokens(context)
            base = CheckpointCompactionContextPolicy(
                compactor=base.compactor,
                max_user_turns=base.max_user_turns,
                compact_after_messages=base.compact_after_messages,
                compact_after_estimated_context_tokens=base.compact_after_estimated_context_tokens,
                max_recent_context_tokens=base.max_recent_context_tokens,
                reserved_output_tokens=(
                    base.reserved_output_tokens
                    + evidence_tokens
                    + state.lexical_navigation_reserved_tokens
                    + (WARNING_RESERVE if state.advisory_controller else 0)
                ),
                reserved_summary_tokens=base.reserved_summary_tokens,
                summary_prefix=base.summary_prefix,
                max_attachment_results=base.max_attachment_results,
            )
        result = await base.build_with_checkpoint(request, checkpoint=checkpoint)
        # This is transient supporting context, not durable customer evidence.
        # Do not compact it into a lossy summary or discard the base policy's
        # durable checkpoint, cursor and compaction telemetry on the way out.
        if state.advisory_controller:
            result = warn_before_compaction(
                result, request, previous_checkpoint=checkpoint,
                threshold=self.base.compact_after_estimated_context_tokens,
                reserved_output=base.reserved_output_tokens,
            )
        return result.model_copy(update={"messages": [*result.messages, context]})


class BankingReplyPolicy(LoopPolicy):
    def __init__(self, state: BankingRuntimeSession):
        self.state = state

    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        state = self.state
        await observe_delivered_recall(state)
        if state.task_memory is not None and state.task_memory.enforce_notes and state.task_memory.exposed_revision != state.task_memory.revision:
            return BeforeStopDecision.continue_with(
                Message.text("user", state.task_memory.context(state.model_step_id) + "\nReview the current notes against actual results before delivering your reply."),
                reason="task_memory_changed_before_reply",
            )
        if state.advisory_controller:
            # A delivered reply does not mark any unresolved checklist item successful.
            return BeforeStopDecision.complete()
        if state.remediations.unresolved_count:
            state.admission_progress.observe_denial(context.step_result.model_step_id)
            return BeforeStopDecision.continue_with(
                Message.text(
                    "user",
                    "Your proposed response was NOT delivered to the customer. It cannot count "
                    "as a disclosure, offer, question or completed communication. Resolve the "
                    "open typed controller correction before replying. Keep internal correction "
                    "details out of the customer-facing response.",
                ),
                reason="banking_open_correction",
            )
        decision = await state.memory.review_terminal_output(
            context.step_result.text_content,
            allow_unplanned_advice=state.advice_mode == "lean",
        )
        if decision.allowed:
            return BeforeStopDecision.complete()
        # Count response vetoes in the same bounded allowance as tool denials.
        # The next context build fails closed before another model dispatch;
        # repeated drafts, plan changes and lookups cannot replenish the budget.
        state.admission_progress.observe_denial(context.step_result.model_step_id)
        correction = json.dumps(
            {
                "message": (
                    "Your proposed response was NOT delivered to the customer. It cannot count "
                    "as a disclosure, offer, question or other completed communication. "
                    + (decision.reason or "Reconcile the outstanding work before responding.")
                ),
                "correction_code": decision.correction_code,
                "required_tool_names": decision.required_tool_names,
            },
            ensure_ascii=True,
        )
        return BeforeStopDecision.continue_with(
            Message.text("user", correction), reason="banking_reply_not_ready"
        )


def build_banking_runtime_resources(
    broker: HostToolBroker,
    host_specs: tuple[ToolSpec, ...],
    *,
    provider: ModelProvider,
    model: str,
    domain_policy: str,
    memory_runtime: BankingMemoryRuntime,
    session_store: SessionStore | None = None,
    aclose: Callable[[], Awaitable[None]] | None = None,
    compaction: CheckpointCompactionContextPolicy | None = None,
    native_recall: bool = False,
    lexical_navigation: bool = False,
    provider_options: dict[str, Any] | None = None,
    planning_mode: str = "exact",
    advice_mode: str = "controller",
    context_capture_directory: Path | None = None,
    controller_mode: str = "enforced",
    decision_packets: bool = False,
    task_memory_mode: str | None = None,
    task_memory_directory: Path | None = None,
    system_prompt_transform: Callable[[str], str] | None = None,
    curated_context: str | None = None,
    focused_policy_file: Path | None = None,
) -> RuntimeResources:
    """Assemble on the owned loop; caller owns supplied resource lifetimes.

    Pass aclose to release provider/store resources at conversation stop. The
    caller must clean up if assembly itself fails. No default CLI is changed.
    """

    if lexical_navigation and not native_recall:
        raise ValueError("Lexical navigation requires native recall.")
    state = BankingRuntimeSession(
        memory_runtime,
        host_specs,
        planning_mode=planning_mode,
        advice_mode=advice_mode,
        controller_mode=controller_mode,
        decision_packets=decision_packets,
        task_memory_mode=task_memory_mode,
        task_memory_directory=task_memory_directory,
    )
    state.native_recall = native_recall
    if focused_policy_file is not None:
        if curated_context is not None:
            raise ValueError("Focused policies and legacy curated context are mutually exclusive")
        from cayu_tau3_banking.focused_context import FocusedPolicyBundle

        state.focused_bundle = FocusedPolicyBundle(focused_policy_file)
    if compaction is None:
        compaction = CheckpointCompactionContextPolicy(
            compactor=banking_compactor(provider, model),
            compact_after_estimated_context_tokens=COMPACTION_TRIGGER,
            max_recent_context_tokens=RECENT_CONTEXT_TARGET,
            reserved_output_tokens=8_000,
            reserved_summary_tokens=4_000,
        )
    schemas = {schema["name"]: schema for schema in state.memory.tool_schemas()}
    if state.decision_facts is not None:
        schemas[DECISION_NOTE_TOOL] = DECISION_NOTE_SCHEMA
        schemas[SAVED_EVIDENCE_TOOL] = SAVED_EVIDENCE_SCHEMA
        if "cayu_read_product_documents" in schemas:
            schemas["cayu_read_product_documents"]["description"] = (
                "First reads return source bodies; unchanged repeated reads return saved-source locators. "
                "Recover several missing sources in one cayu_read_saved_evidence kind=documents call "
                "using returned entry_ids and section offsets. Use source_span for a targeted public read. "
                + schemas["cayu_read_product_documents"]["description"]
            )
    if curated_context is not None:
        if not isinstance(curated_context, str) or not curated_context.strip():
            raise ValueError("curated_context must be nonblank text")
        schemas = {
            name: schema for name, schema in schemas.items() if name not in PUBLIC_RESEARCH_TOOLS
        }
    schemas.update(
        {schema["name"]: schema for schema in state.controller.registration_tool_schemas()}
    )
    if state.lightweight_workflow:
        from cayu_tau3_banking.workflow_profile import outcome_schemas
        schemas.update({schema["name"]: schema for schema in outcome_schemas(state.controller)})
    if task_memory_mode == "markdown":
        schemas = {name: schema for name, schema in schemas.items() if name in PUBLIC_RESEARCH_TOOLS - {"cayu_check_comparison_facts"}}
    if state.task_memory is not None:
        schemas.update({x["name"]: x for x in task_memory_schemas(task_memory_mode)})
    for plan_tool in (SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL):
        if plan_tool in schemas:
            schemas[plan_tool] = json.loads(json.dumps(schemas[plan_tool]))
            schemas[plan_tool]["input_schema"]["properties"]["expected_revision"] = {
                "type": "integer", "minimum": 0,
                "description": "Optional optimistic concurrency check against the outcome-plan revision, not the task-note revision. Omit if not supplying this check.",
            }
    if task_memory_mode == "constraints" and DECISION_NOTE_TOOL in schemas:
        schemas[DECISION_NOTE_TOOL] = json.loads(json.dumps(schemas[DECISION_NOTE_TOOL]))
        schemas[DECISION_NOTE_TOOL]["input_schema"]["properties"]["facts"]["items"]["properties"].pop("conflict", None)
    if state.host_names & schemas.keys():
        raise ValueError("Host tools collide with private banking tool names.")
    private_tools = [
        BankingNativeTool(
            ToolSpec(
                name=schema["name"],
                description=schema["description"],
                input_schema=schema["input_schema"],
                parallel_safe=False,
            ),
            state,
        )
        for schema in schemas.values()
    ]
    app = CayuApp(
        session_store=session_store,
        enable_logging=False,
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="banking-live-process-v1",
            fingerprint_key=secrets.token_hex(32),
        )
        if native_recall
        else None,
    )
    state.session_store = app.session_store
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="banking-public-corpus"),
            knowledge_store=memory_runtime.corpus.store,
        ),
        default=True,
    )
    context_policy = BankingRuntimeContext(state, compaction=compaction)
    if native_recall and state.focused_bundle is None:
        context_policy = banking_recall_policy(memory_runtime.corpus, context_policy)
    if lexical_navigation and state.focused_bundle is None:
        from cayu_tau3_banking.lexical_navigation import LexicalNavigationContext

        context_policy = LexicalNavigationContext(
            context_policy,
            memory_runtime.corpus,
            state.memory._current_query,
            lambda tokens: setattr(state, "lexical_navigation_reserved_tokens", tokens),
        )
    if context_capture_directory is not None:
        from cayu_tau3_banking.context_capture import CapturingContextPolicy
        from cayu_tau3_banking.retrieval_capture import RetrievalCapture

        context_policy = CapturingContextPolicy(context_policy, context_capture_directory)
        state.memory.retrieval_capture = RetrievalCapture(
            context_capture_directory, state.memory.task_id
        )
    selected_system_prompt = (
        advisory_banking_prompt(domain_policy)
        if state.advisory_controller
        else lean_banking_prompt(domain_policy)
        if advice_mode == "lean"
        else banking_system_prompt(domain_policy, planning_mode=planning_mode)
    )
    if task_memory_mode == "markdown":
        selected_system_prompt = MARKDOWN_INSTRUCTIONS + "\n" + VERIFICATION_GUIDANCE + "\nPublic domain policy:\n" + domain_policy
    elif state.lightweight_workflow:
        from cayu_tau3_banking.workflow_profile import banking_prompt
        selected_system_prompt = banking_prompt(domain_policy)
    if decision_packets:
        selected_system_prompt += "\n" + PACKET_INSTRUCTIONS
    if curated_context is not None:
        selected_system_prompt += (
            "\n\n<curated_diagnostic_context>\n"
            + curated_context
            + "\n</curated_diagnostic_context>\n"
            "Use this source-backed context for the diagnostic comparison. Do not search "
            "for additional public documents; ask the customer when authorization is unclear."
        )
    if system_prompt_transform is not None:
        selected_system_prompt = system_prompt_transform(selected_system_prompt)
        if not isinstance(selected_system_prompt, str) or not selected_system_prompt.strip():
            raise ValueError("system_prompt_transform must return nonblank text")
    app.register_agent(
        AgentSpec(
            name="banking",
            model=model,
            system_prompt=selected_system_prompt,
            provider_options=provider_options or {},
        ),
        tools=[*private_tools, *(broker.tool(spec) for spec in host_specs)],
        context_policy=context_policy,
        tool_exposure_policy=BankingToolExposure(state),
        tool_policy=BankingRuntimeToolPolicy(state),
        runtime_hooks=[BankingEvidenceHook(state)],
        loop_policies=[BankingReplyPolicy(state)],
    )
    return RuntimeResources(
        app=app,
        aclose=aclose,
        observer=state,
        diagnostic_snapshot=lambda: {
            "cayu_tau3_workflow": state.controller.export_state(),
            "cayu_tau3_memory": state.memory.export_state(),
            "cayu_tau3_admission": state.admission.export_state(),
            "cayu_task_memory": state.task_memory.snapshot() if state.task_memory else None,
            "cayu_decision_facts": state.decision_facts.snapshot() if state.decision_facts else None,
            "cayu_execution_receipts": state.execution_receipts.receipts if state.execution_receipts else [],
            "cayu_native_runtime": {
                "session_id": state.session_id,
                "native_recall": state.native_recall,
                "lexical_navigation": lexical_navigation,
                "advice_mode": state.advice_mode,
                "controller_mode": controller_mode,
                "decision_packets": decision_packets,
                "task_memory_mode": task_memory_mode,
                "packet_gate_counts": dict(state.packet_gate.counts) if state.packet_gate else {},
                "observed_recall_exposures": len(state.observed_recall_exposures),
                "focused_policy_sha256": state.focused_bundle.digest if state.focused_bundle else None,
                "automatic_recall_enabled": native_recall and state.focused_bundle is None,
                "lexical_navigation_enabled": lexical_navigation and state.focused_bundle is None,
            },
        },
        untracked_model_helpers=any(
            verifier is not None
            for verifier in (
                memory_runtime.plan_verifier,
                memory_runtime.action_verifier,
                memory_runtime.response_verifier,
            )
        ),
    )
