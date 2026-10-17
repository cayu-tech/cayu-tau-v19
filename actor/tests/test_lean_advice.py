import asyncio

import pytest
from cayu import ModelStreamEvent, ToolCallPart, ToolResultPart
from tau2.data_model.message import ToolMessage, UserMessage
from tau2.environment.tool import Tool
from test_admission import confirm_verification
from test_runtime_banking import assembly, call, offered, text, work

from cayu_tau3_banking import cli
from cayu_tau3_banking.advice import COMPARISON_COVERAGE, PUBLIC_RESEARCH_TOOLS, lean_banking_prompt
from cayu_tau3_banking.argument_contract import OPTIONAL_ARGUMENT_DEFAULTS
from cayu_tau3_banking.controller import (
    SYNC_WORK_PLAN_TOOL,
    UPDATE_WORK_PLAN_TOOL,
    TerminalOutputAdmission,
)
from cayu_tau3_banking.ledger import ToolActionStatus
from cayu_tau3_banking.product_navigation import READ_TOOL
from cayu_tau3_banking.runtime_agent import RuntimeAgentError


def build(tmp_path, steps, **kwargs):
    return assembly(tmp_path, steps, planning_mode="outcomes", advice_mode="lean", **kwargs)


def outcome(*, completed=False, reference="effect"):
    item = work(operation=True, done=completed, reference=reference)
    item.pop("next_operation")
    return item


def test_lean_prompt_keeps_shared_optional_contract_and_generic_comparison_guidance():
    from cayu_tau3_banking.argument_contract import SOURCE_BINDING_RECOVERY

    policy = "Fixture policy: preserve required overrides."
    prompt = lean_banking_prompt(policy)
    assert prompt.count(OPTIONAL_ARGUMENT_DEFAULTS) == 1
    assert prompt.count(SOURCE_BINDING_RECOVERY) == 1
    assert prompt.count(COMPARISON_COVERAGE) == 1
    assert prompt.endswith(f"<policy>\n{policy}\n</policy>")
    assert "Preserve meaningful" in prompt and "policy-required overrides" in prompt
    assert "Respect explicit exclusions and refusals" in prompt
    assert "qualify the recommendation" in prompt
    assert "Silver Rewards" not in prompt and "task_066" not in prompt
    assert "knowledge_revision evidence_ref" in prompt
    assert "policy denial is not an executed tool failure" in prompt
    assert "do not infer authorization" in prompt
    assert "action-specific" in prompt
    assert "magnitude_class" in prompt
    assert "one-point" in prompt


def test_lean_prompt_preserves_preparation_and_dependencies_without_task_rules():
    from cayu_tau3_banking.advice import OPERATION_PREPARATION
    from cayu_tau3_banking.condition_preservation import CONDITION_PRESERVATION

    prompt = lean_banking_prompt("Fixture policy")
    assert prompt.count(OPERATION_PREPARATION) == 1
    assert prompt.count(CONDITION_PRESERVATION) == 1
    assert "awaiting_customer" in prompt and "; waiting_customer " not in prompt
    assert "14 days" not in prompt and "Light Blue" not in prompt
    assert "knowledge_revision evidence_refs" in prompt


def test_lean_prompt_keeps_investigation_and_customer_needs_ahead_of_handoffs():
    from cayu_tau3_banking.agent import PRODUCT_NEEDS_INSTRUCTIONS, banking_system_prompt
    from cayu_tau3_banking.investigation import INVESTIGATION_OWNERSHIP

    prompt = lean_banking_prompt("Fixture policy")
    assert prompt.count(INVESTIGATION_OWNERSHIP) == 1
    assert prompt.count(PRODUCT_NEEDS_INSTRUCTIONS) == 1
    assert INVESTIGATION_OWNERSHIP in banking_system_prompt("Fixture policy")
    assert "optional for a simple source-backed recommendation" in prompt
    assert "customer_input" in PRODUCT_NEEDS_INSTRUCTIONS
    assert "task_017" not in prompt and "Roberto" not in prompt


def test_eligibility_guidance_distinguishes_public_terms_from_customer_record_proof():
    from cayu_tau3_banking.agent import banking_system_prompt

    for prompt in (lean_banking_prompt("Fixture policy"), banking_system_prompt("Fixture policy")):
        assert "Public documents establish the rule, not the customer's record" in prompt
        assert "use the authorized\nbank records after the required identity checks" in prompt
        assert "General explanations of public terms need no private lookup" in prompt
        assert "Obtain customer-owned facts\nand preferences from the customer" in prompt
        assert "does not waive a procedure's required" in prompt
        assert "World Blue" not in prompt and "Hunter Green" not in prompt
        assert "task_100" not in prompt


def test_public_read_and_reply_require_no_plan_but_no_bank_tools_are_exposed(tmp_path):
    agent, provider, captured = build(
        tmp_path,
        [
            call(READ_TOOL, {"documents": [{"entry_id": "limits", "revision": 1}]}, "read"),
            text("The purchase limit is $4,500."),
        ],
        persistent=True,
        native_recall=True,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Explain the limit, advice only."), state
        )
        assert reply.content == "The purchase limit is $4,500."
        assert not captured["state"].controller.has_work_items()
        assert not captured["state"].ledger.actions
        assert all(
            offered(r) <= PUBLIC_RESEARCH_TOOLS | {SYNC_WORK_PLAN_TOOL} for r in provider.requests
        )
        assert "Public research and advice do not require a checklist" in str(
            provider.requests[0].messages[0].content
        )
        for request in provider.requests:
            instructions = "\n".join(part.text for part in request.messages[0].content)
            assert OPTIONAL_ARGUMENT_DEFAULTS in instructions
            assert COMPARISON_COVERAGE in instructions
        assert reply.raw_data["cayu_native_runtime"]["advice_mode"] == "lean"
    finally:
        state.stop()


def test_operational_plan_is_still_required_by_authorizer_and_execution_gate(tmp_path):
    agent, provider, captured = build(tmp_path, [text("Which operation do you need?")])
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Explain the limit."), state)
        proposed = ToolCallPart(
            tool_call_id="unplanned", tool_name="host_action", arguments={"value": 7}
        )
        session = captured["state"]
        for gate in (session.batch_gate, session.gate):
            decision = asyncio.run(gate.review((proposed,)))[0]
            assert not decision.allowed and "plan" in decision.reason.lower()
        assert "host_action" not in offered(provider.requests[0])
    finally:
        state.stop()


def test_advice_can_transition_to_guarded_host_execution_in_same_session(tmp_path):
    agent, provider, captured = build(
        tmp_path,
        [
            text("The purchase limit is $4,500."),
            call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}, "plan"),
            text("Premature done."),
            call("host_action", {"value": 7}, "effect"),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [outcome(completed=True)]}, "complete"),
            text("The requested operation succeeded."),
        ],
        callback_timeout_s=60,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Explain the limit."), state)
        assert not reply.tool_calls
        reply, _ = agent.generate_next_message(
            UserMessage.text("Now perform the operation with value 7."), state
        )
        (external,) = reply.tool_calls
        assert external.name == "host_action" and external.arguments == {"value": 7}
        assert captured["state"].ledger.tool_action_status("effect") is ToolActionStatus.PENDING
        reply, _ = agent.generate_next_message(
            ToolMessage(role="tool", id=external.id, content="success"), state
        )
        assert reply.content == "The requested operation succeeded."
        assert [action.call_id for action in captured["state"].ledger.actions] == ["effect"]
        assert len(captured["state"].ledger.customer_message_ids) == 2
        assert "NOT delivered" in str(provider.requests[3].messages)
        assert captured["state"].controller.export_state()["items"][0]["outcome"] == "succeeded"
    finally:
        state.stop()


def test_public_advice_cannot_bypass_pending_operation_completion(tmp_path):
    agent, _, captured = build(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}),
            text("Everything is done."),
        ],
        max_steps=2,
    )
    state = agent.get_init_state()
    try:
        with pytest.raises(RuntimeAgentError):
            agent.generate_next_message(UserMessage.text("Perform the requested operation."), state)
        assert not captured["state"].ledger.actions
        assert captured["state"].controller.export_state()["items"][0]["outcome"] == "unresolved"
    finally:
        state.stop()


def test_lean_mode_retains_host_schema_check_and_repair(tmp_path):
    agent, provider, captured = build(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}),
            call("host_action", {"value": "not an integer"}, "bad"),
            call("host_action", {"value": 7}, "good"),
        ],
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Perform the requested operation."), state
        )
        (external,) = reply.tool_calls
        assert external.name == "host_action" and external.arguments == {"value": 7}
        assert [action.call_id for action in captured["state"].ledger.actions] == ["good"]
        assert any(
            part.is_error
            for m in provider.requests[-1].messages
            for part in m.content
            if isinstance(part, ToolResultPart)
        )
    finally:
        state.stop()


def test_nonpublic_memory_helpers_cannot_use_unplanned_research_permission(tmp_path):
    from cayu_tau3_banking.credit_card_dispute import AUDIT_CREDIT_CARD_DISPUTES_TOOL

    agent, _, captured = build(tmp_path, [text("Public answer.")])
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Public question."), state)
        result = asyncio.run(
            captured["state"].memory.invoke(
                AUDIT_CREDIT_CARD_DISPUTES_TOOL, {}, allow_unplanned_public_research=True
            )
        )
        assert result.is_error and "plan" in result.content
    finally:
        state.stop()


def test_lean_mode_keeps_customer_verification_gate_after_planning(tmp_path):
    agent, _, captured = build(
        tmp_path, [call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]})], max_steps=1
    )
    state = agent.get_init_state()
    try:
        with pytest.raises(RuntimeAgentError):
            agent.generate_next_message(UserMessage.text("Read my account details."), state)
        proposed = ToolCallPart(
            tool_call_id="private-read",
            tool_name="get_credit_card_accounts_by_user",
            arguments={"user_id": "user-7"},
        )
        for gate in (captured["state"].batch_gate, captured["state"].gate):
            (decision,) = asyncio.run(gate.review((proposed,)))
            assert not decision.allowed and "verif" in decision.reason.lower()
        assert not captured["state"].ledger.actions
    finally:
        state.stop()


def test_lean_mode_executes_duplicate_host_effect_only_once(tmp_path):
    # Use a real non-repeatable banking capability. The generic synthetic
    # host_action has no banking effect classification in the admission gate.
    def change_user_email(user_id: str, new_email: str) -> str:
        """Change a verified customer's email.

        Args:
            user_id: Customer identifier.
            new_email: Requested replacement email.
        """
        raise AssertionError("Only the test host executes environment tools")

    arguments = {"user_id": "user-7", "new_email": "new@example.com"}
    double = [
        ModelStreamEvent.tool_call(id=call_id, name="change_user_email", arguments=arguments)
        for call_id in ("effect", "duplicate")
    ] + [ModelStreamEvent.completed({"finish_reason": "tool_calls"})]
    agent, provider, captured = build(
        tmp_path,
        [
            text("I can help with that."),
            call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}),
            double,
            call(UPDATE_WORK_PLAN_TOOL, {"items": [outcome(completed=True)]}),
            text("Done."),
        ],
        host_tools=[Tool(change_user_email)],
    )
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Can you update my email?"), state)
        confirm_verification(captured["state"].ledger)
        reply, _ = agent.generate_next_message(UserMessage.text("Perform the operation."), state)
        (external,) = reply.tool_calls
        assert external.name == "change_user_email" and external.arguments == arguments
        reply, _ = agent.generate_next_message(
            ToolMessage(role="tool", id=external.id, content="success"), state
        )
        assert reply.content == "Done."
        assert [a.call_id for a in captured["state"].ledger.actions] == ["verify-1", "effect"]
        assert any(
            p.tool_call_id == "duplicate" and p.is_error
            for m in provider.requests[-1].messages
            for p in m.content
            if isinstance(p, ToolResultPart)
        )
    finally:
        state.stop()


@pytest.mark.parametrize(
    "guard_name",
    ["_review_credit_card_last_four_handoff", "_review_audited_credit_card_dispute_handoff"],
)
def test_unplanned_reply_still_consults_domain_evidence_checks(tmp_path, monkeypatch, guard_name):
    agent, _, captured = build(tmp_path, [text("Public answer.")])
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Public question."), state)
        memory = captured["state"].memory
        rejection = TerminalOutputAdmission(allowed=False, reason="Missing required evidence.")
        monkeypatch.setattr(memory, guard_name, lambda: rejection)
        assert (
            asyncio.run(
                memory.review_terminal_output("Public answer.", allow_unplanned_advice=True)
            )
            == rejection
        )
    finally:
        state.stop()


def test_configuration_is_opt_in_and_rejects_incompatible_runtime():
    assert cli._parser().parse_args([]).advice_mode == "controller"
    for args in (["--planning-mode", "exact"], ["--runtime", "bridge"]):
        parsed = cli._parser().parse_args(["--advice-mode", "lean", *args])
        with pytest.raises(SystemExit, match="Lean advice requires"):
            cli._run(parsed)
