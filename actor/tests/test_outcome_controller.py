"""Generic checklist contracts: no benchmark task inputs or reference answers."""

import asyncio
import json

import pytest
from cayu import ToolCallPart, WorkEvidenceReference
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from test_admission import confirm_effect, confirm_verification
from test_controller import item
from test_runtime_banking import assembly, call, text, work

from cayu_tau3_banking.admission import BankingToolAdmissionGate
from cayu_tau3_banking.bridge import CompositeExternalToolCallGate
from cayu_tau3_banking.controller import (
    SYNC_WORK_PLAN_TOOL,
    UPDATE_WORK_PLAN_TOOL,
    BankingWorkflowController,
)
from cayu_tau3_banking.ledger import ConversationLedger


def outcome(item_id="request", **kwargs):
    prior_report = {key: kwargs.pop(key) for key in ("prior_customer_report",) if key in kwargs}
    result = item(item_id, **kwargs)
    result.update(prior_report)
    result.pop("next_operation")
    return result


def setup_plan(*items):
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please perform the requested work."))
    controller = BankingWorkflowController(ledger, planning_mode="outcomes")
    assert not controller.invoke(
        SYNC_WORK_PLAN_TOOL, {"items": list(items or [outcome()])}
    ).is_error
    return controller, ledger


def test_information_withdrawal_requires_later_customer_not_fake_answer():
    controller, ledger = setup_plan(outcome(completion_mode="information"))
    patch = {
        "status": "completed",
        "outcome": "declined",
        "evidence": "Customer explicitly withdrew this research.",
        "evidence_refs": [{"kind": "customer_message", "reference_id": "customer-message:1"}],
    }
    assert update(controller, **patch).is_error
    ledger.observe_input(
        UserMessage.text("Stop comparing products; I no longer need that research.")
    )
    patch["evidence_refs"][0]["reference_id"] = "customer-message:2"
    assert not update(controller, **patch).is_error
    saved = controller.export_state()
    assert saved["items"][0]["outcome"] == "declined"
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Research withdrawn.", raw_data={"cayu_tau3_workflow": saved})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == saved


def test_information_parent_reuses_child_evidence_not_operation_proof():
    ref = {"kind": "knowledge_revision", "reference_id": "synthetic-policy", "version": "1"}
    for mode in ("information", "operation"):
        controller, _ = setup_plan(outcome(completion_mode=mode))
        controller.observe_knowledge_evidence([WorkEvidenceReference(**ref)])
        result = update(
            controller,
            status="completed",
            outcome="answered" if mode == "information" else "succeeded",
            evidence="Supported by the resolved factual prerequisite.",
            obligations=[obligation("resolved", evidence_refs=[ref])],
        )
        assert result.is_error == (mode == "operation")
        if mode == "information":
            assert (
                controller.export_state()["items"][0]["evidence_refs"][0]["reference_id"]
                == "synthetic-policy"
            )


def test_waiting_information_parent_needs_no_duplicate_child_evidence():
    controller, _ = setup_plan(outcome(completion_mode="information"))
    assert not update(
        controller, status="awaiting_customer", obligations=[obligation("waiting")]
    ).is_error
    saved = controller.export_state()["items"][0]
    assert saved["evidence"] == "See the explicitly cited child obligations."
    assert saved["evidence_refs"][0]["kind"] == "customer_message"


def test_new_outcome_can_record_explicit_prior_customer_report_and_restore():
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("I already submitted the delivery form successfully."))
    controller = BankingWorkflowController(ledger, planning_mode="outcomes")
    reported = outcome(
        "form",
        status="completed",
        outcome="succeeded",
        prior_customer_report=True,
        evidence="Customer reports the form was already submitted; not independently verified.",
        evidence_refs=[{"kind": "customer_message", "reference_id": "customer-message:1"}],
    )
    assert not controller.invoke(SYNC_WORK_PLAN_TOOL, {"items": [reported]}).is_error
    snapshot = controller.export_state()
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Noted.", raw_data={"cayu_tau3_workflow": snapshot})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == snapshot
    ledger.observe_input(UserMessage.text("Now please update my address."))
    assert not restored.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [outcome("address")]}).is_error
    assert restored.export_state()["items"][0]["prior_customer_report"] is True
    changed = restored.invoke(
        UPDATE_WORK_PLAN_TOOL,
        {
            "items": [
                {
                    "id": "form",
                    "prior_customer_report": False,
                }
            ]
        },
    )
    assert changed.is_error


@pytest.mark.parametrize(
    "patch",
    [
        {"status": "pending", "outcome": "unresolved"},
        {"completion_mode": "information", "outcome": "answered"},
        {"evidence_refs": []},
    ],
)
def test_prior_report_rejects_invalid_shape(patch):
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("The form is already submitted."))
    controller = BankingWorkflowController(ledger, planning_mode="outcomes")
    entry = outcome(
        status="completed",
        outcome="succeeded",
        prior_customer_report=True,
        evidence="Customer testimony",
        evidence_refs=[
            {
                "kind": "customer_message",
                "reference_id": "customer-message:1",
            }
        ],
    )
    entry.update(patch)
    assert controller.invoke(SYNC_WORK_PLAN_TOOL, {"items": [entry]}).is_error


def test_version_14_outcomes_restore_without_prior_report_declaration():
    controller, ledger = setup_plan()
    snapshot = controller.export_state()
    snapshot["schema_version"] = 14
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Working.", raw_data={"cayu_tau3_workflow": snapshot})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.has_work_items()
    assert restored.export_state()["schema_version"] == 16


@pytest.mark.parametrize("declaration", [False, "true", 1])
def test_initial_request_is_not_completion_without_typed_report_declaration(declaration):
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please submit the form."))
    controller = BankingWorkflowController(ledger, planning_mode="outcomes")
    result = controller.invoke(
        SYNC_WORK_PLAN_TOOL,
        {
            "items": [
                outcome(
                    status="completed",
                    outcome="succeeded",
                    prior_customer_report=declaration,
                    evidence="Request is not completion",
                    evidence_refs=[
                        {"kind": "customer_message", "reference_id": "customer-message:1"}
                    ],
                )
            ]
        },
    )
    assert result.is_error
    assert not controller.has_work_items()


def test_report_declaration_cannot_bypass_existing_handoff():
    controller, _ = setup_plan(
        outcome(status="awaiting_customer", evidence="Waiting for submission")
    )
    result = update(
        controller,
        status="completed",
        outcome="succeeded",
        prior_customer_report=True,
        evidence="Claimed prior report",
        evidence_refs=[{"kind": "customer_message", "reference_id": "customer-message:1"}],
    )
    assert result.is_error
    assert "cannot bypass a handoff" in result.content


def test_new_report_cannot_cite_stale_customer_message():
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("I submitted a form last year."))
    ledger.observe_input(UserMessage.text("Please submit a new form."))
    controller = BankingWorkflowController(ledger, planning_mode="outcomes")
    result = controller.invoke(
        SYNC_WORK_PLAN_TOOL,
        {
            "items": [
                outcome(
                    status="completed",
                    outcome="succeeded",
                    prior_customer_report=True,
                    evidence="Old report",
                    evidence_refs=[
                        {
                            "kind": "customer_message",
                            "reference_id": "customer-message:1",
                        }
                    ],
                )
            ]
        },
    )
    assert result.is_error


def observe(ledger, name="perform_operation", call_id="effect", error=False):
    ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[ToolCall(id=call_id, name=name, arguments={}, requestor="assistant")],
        )
    )
    ledger.observe_input(
        ToolMessage(role="tool", id=call_id, content="failed" if error else "ok", error=error)
    )


def update(controller, **fields):
    return controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [{"id": "request", **fields}]})


def obligation(status="ready", **fields):
    return {
        "id": "validate-address",
        "goal": "Validate the delivery address",
        "status": status,
        "evidence": "" if status == "ready" else "Observed prerequisite",
        "evidence_refs": []
        if status == "ready"
        else [{"kind": "customer_message", "reference_id": "customer-message:1"}],
        **fields,
    }


def shipping_plan(status="pending"):
    return [
        outcome("payment", status="awaiting_customer", evidence="Customer must pay"),
        {**outcome(status=status, depends_on=["payment"]), "obligations": [obligation()]},
    ]


@pytest.mark.parametrize("status", ["pending", "in_progress"])
def test_effect_dependency_does_not_suppress_preparation(status):
    controller, _ = setup_plan(*shipping_plan(status))
    decision = controller.preview_terminal_output("Waiting for payment.")
    assert not decision.allowed
    assert "request/validate-address" in decision.reason
    assert not update(controller, obligations=[obligation("waiting")]).is_error
    assert controller.preview_terminal_output("Please confirm your address and pay.").allowed
    assert "not proof of coverage" in controller.render_context()


def test_planned_action_packet_binds_outcome_mutation_arguments():
    planned = {
        "tool_name": "change_card_limit",
        "arguments": {"user_id": "u1", "card_id": "c1", "new_limit": 5000},
    }
    controller, _ = setup_plan(
        {**outcome(), "obligations": [obligation(planned_tool=planned)]}
    )
    matching = controller.review(
        (
            ToolCallPart(
                tool_call_id="call-1",
                tool_name="change_card_limit",
                arguments={"user_id": "u1", "card_id": "c1", "new_limit": 5000},
            ),
        )
    )
    assert matching[0].allowed
    mismatch = controller.review(
        (
            ToolCallPart(
                tool_call_id="call-2",
                tool_name="change_card_limit",
                arguments={"user_id": "u1", "card_id": "c2", "new_limit": 5000},
            ),
        )
    )
    assert not mismatch[0].allowed
    assert mismatch[0].diagnostic_code == "outcome_action_packet_mismatch"


def test_planned_action_packet_is_optional_for_legacy_outcome_obligations():
    controller, _ = setup_plan({**outcome(), "obligations": [obligation()]})
    decisions = controller.review(
        (
            ToolCallPart(
                tool_call_id="call-legacy",
                tool_name="change_card_limit",
                arguments={"card_id": "c1"},
            ),
        )
    )
    assert decisions[0].allowed


def test_in_progress_preparation_update_restores_without_completing_dependency():
    controller, ledger = setup_plan(*shipping_plan())
    result = update(controller, status="in_progress")
    assert not result.is_error, result.content
    snapshot = controller.export_state()
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Preparing shipment.", raw_data={"cayu_tau3_workflow": snapshot})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == snapshot
    assert not restored.preview_terminal_output("Finished.").allowed
    observe(ledger)
    rejected = update(
        restored,
        status="completed",
        outcome="succeeded",
        evidence="Shipment effect observed, but payment is still unconfirmed.",
        evidence_refs=[{"kind": "tool_result", "reference_id": "effect"}],
        obligations=[obligation("resolved")],
    )
    assert rejected.is_error
    assert "before dependencies complete" in rejected.content
    assert restored.export_state() == snapshot


@pytest.mark.parametrize("dependencies", [["unknown"], ["request"], ["dependent"]])
def test_in_progress_preparation_still_rejects_invalid_dependency_graph(dependencies):
    controller, _ = setup_plan(outcome(), outcome("dependent", depends_on=["request"]))
    snapshot = controller.export_state()
    result = update(controller, status="in_progress", depends_on=dependencies)
    assert result.is_error
    assert result.diagnostic_code == "sync_dependency_graph_rejected"
    assert controller.export_state() == snapshot


def test_exhausted_lookup_permits_handoff_without_resolving_work():
    controller, _ = setup_plan(
        {
            **outcome(),
            "obligations": [
                obligation(kind="knowledge_search"),
                obligation(
                    id="offer-choice", kind="communication", goal="Ask which option to pursue"
                ),
            ],
        }
    )
    assert not controller.preview_terminal_output("Which option?").allowed
    controller.set_knowledge_search_available(False)
    assert controller.preview_terminal_output(
        "I cannot verify the remaining detail. Which option?"
    ).allowed
    saved = controller.export_state()["items"][0]
    assert saved["status"] == "pending"
    assert saved["obligations"][0]["status"] == "ready"
    assert (
        '"resource_blocked_obligations": ["request/validate-address"]'
        in controller.render_context()
    )
    controller.set_knowledge_search_available(True)
    assert not controller.preview_terminal_output("Which option?").allowed


def test_exhausted_search_does_not_release_independent_action_or_complete_lookup():
    controller, _ = setup_plan(
        {
            **outcome(),
            "obligations": [
                obligation(kind="knowledge_search"),
                obligation(id="action", goal="Execute the authorized effect"),
                obligation(id="ask", kind="communication", goal="Ask a question"),
            ],
        }
    )
    controller.set_knowledge_search_available(False)
    assert not controller.preview_terminal_output("Finished.").allowed
    assert update(controller, status="completed", outcome="succeeded").is_error


@pytest.mark.parametrize("navigation_exhausted", [True, False])
def test_native_exhausted_lookup_can_deliver_real_choice(
    tmp_path, navigation_exhausted, monkeypatch
):
    from cayu_tau3_banking import agent_memory
    from cayu_tau3_banking import product_navigation as navigation
    from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL
    from cayu_tau3_banking.product_navigation import BROWSE_TOOL, READ_TOOL
    from cayu_tau3_banking.runtime_agent import RuntimeCallbackExecutionError

    monkeypatch.setitem(navigation.BUDGET_LIMITS, BROWSE_TOOL, (2, 32))
    monkeypatch.setitem(navigation.BUDGET_LIMITS, READ_TOOL, (2, 48))
    monkeypatch.setattr(navigation, "RECOVERY_LIMITS", (2, 24))
    monkeypatch.setattr(agent_memory, "_MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE", 3)
    for identifier, content in (("diagnostic_alpha", "Zircon."), ("diagnostic_beta", "Onyx.")):
        (tmp_path / f"{identifier}.json").write_text(
            json.dumps({"id": identifier, "title": identifier, "content": content})
        )

    agent, provider, captured = assembly(
        tmp_path,
        [
            call(
                SYNC_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            **outcome(),
                            "obligations": [
                                obligation(kind="knowledge_search"),
                                obligation(
                                    id="offer",
                                    kind="communication",
                                    goal="Ask which option to pursue",
                                ),
                            ],
                        }
                    ]
                },
            ),
            *[
                call(
                    KNOWLEDGE_SEARCH_TOOL,
                    {"query": "delivery policy", "mode": "lexical", "limit": 4},
                    f"search-{i}",
                )
                for i in range(3)
            ],
            *[
                call(BROWSE_TOOL, {}, f"browse-{i}")
                for i in range(2 if navigation_exhausted else 0)
            ],
            *[
                call(
                    READ_TOOL, {"documents": [{"entry_id": identifier, "revision": 1}]}, f"read-{i}"
                )
                for i, identifier in enumerate(
                    ("diagnostic_alpha", "diagnostic_beta", "diagnostic_alpha", "diagnostic_beta")
                    if navigation_exhausted
                    else ()
                )
            ],
            text("I could not verify that detail. Which option would you like to pursue?"),
        ],
        planning_mode="outcomes",
        native_recall=True,
        max_steps=32,
    )
    # Exercise actual new reads and recoveries under smaller fixture-only limits.
    agent.callback_timeout_s = 60
    state = agent.get_init_state()
    try:
        if not navigation_exhausted:
            # This unsupported blocker must not be delivered while exact reads
            # remain available. The scripted provider has no repair response.
            with pytest.raises(RuntimeCallbackExecutionError):
                agent.generate_next_message(UserMessage.text("Explain my options."), state)
            assert not captured["state"].ledger.actions
            return
        reply, _ = agent.generate_next_message(UserMessage.text("Explain my options."), state)
        assert "could not verify" in reply.content
        bank = captured["state"]
        assert bank.controller.export_state()["items"][0]["status"] == "pending"
        assert len(provider.requests) == 11
        assert bank.memory.navigation.state["recovery"]["turn"] == 2
        assert bank.memory.navigation.state["lanes"][READ_TOOL]["turn"] == 2
        assert not bank.ledger.actions
        assert not any(t["name"] == KNOWLEDGE_SEARCH_TOOL for t in provider.requests[-1].tools)
        context = str(provider.requests[-1].messages)
        assert '"research_exhausted": true' in context
        assert "application cost controls" in context
        assert "worksheet checks cannot restore exhausted capacity" in context
    finally:
        state.stop()


def test_repeated_successful_read_does_not_reset_native_plan_stalls(tmp_path):
    from cayu_tau3_banking.product_navigation import READ_TOOL

    read = {"documents": [{"entry_id": "limits", "revision": 1}]}
    steps = [
        call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}),
        call(READ_TOOL, read, "first-read"),
    ]
    for i in range(6):
        steps.extend(
            [
                call(
                    UPDATE_WORK_PLAN_TOOL,
                    {"items": [{"id": "request", "evidence": "Still working"}]},
                    f"plan-{i}",
                ),
                call(READ_TOOL, read, f"read-{i}"),
            ]
        )
    agent, provider, captured = assembly(
        tmp_path, steps, planning_mode="outcomes", native_recall=True, callback_timeout_s=60
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Please do the work."), state)
        assert agent.is_stop(reply)
        assert reply.raw_data["agent_termination"] == "planning_stall"
        assert len(provider.requests) == 11
        assert not captured["state"].ledger.actions
    finally:
        state.stop()


def test_native_plan_only_loop_stops_before_another_model_call(tmp_path):

    agent, provider, _ = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}),
            *[
                call(
                    UPDATE_WORK_PLAN_TOOL,
                    {"items": [{"id": "request", "evidence": f"Still working {i}"}]},
                    f"rewrite-{i}",
                )
                for i in range(8)
            ],
            text("Unverified success"),
        ],
        planning_mode="outcomes",
        native_recall=True,
        # Durable validation is slower on the qualification VM;
        # this test asserts model-call counts, not a 20-second performance SLA.
        callback_timeout_s=60,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Please do the work."), state)
        assert agent.is_stop(reply)
        assert reply.raw_data["agent_termination"] == "planning_stall"
        assert len(provider.requests) == 6
        assert any(
            "Planning is not advancing" in str(request.messages) for request in provider.requests
        )
    finally:
        state.stop()


def test_native_warning_recovers_after_real_host_result(tmp_path):
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}),
            *[
                call(
                    UPDATE_WORK_PLAN_TOOL,
                    {"items": [{"id": "request", "evidence": f"Working {i}"}]},
                    f"rewrite-{i}",
                )
                for i in range(2)
            ],
            call("host_action", {"value": 7}, "effect"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "request",
                            "status": "completed",
                            "outcome": "succeeded",
                            "evidence": "The host confirmed execution",
                            "evidence_refs": [{"kind": "tool_result", "reference_id": "effect"}],
                        }
                    ]
                },
            ),
            text("The operation succeeded."),
        ],
        planning_mode="outcomes",
        native_recall=True,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Execute the operation."), state)
        assert reply.tool_calls[0].name == "host_action"
        assert captured["state"].planning_progress.warning
        reply, _ = agent.generate_next_message(
            ToolMessage(role="tool", id=reply.tool_calls[0].id, content="Executed successfully"),
            state,
        )
        assert reply.content == "The operation succeeded."
        assert captured["state"].planning_progress.warning is None
        assert len(provider.requests) == 6
    finally:
        state.stop()


def test_ready_communication_can_be_delivered_without_claiming_completion():
    controller, ledger = setup_plan(
        {**outcome(), "obligations": [obligation(kind="communication", goal="Ask for consent")]}
    )
    assert controller.preview_terminal_output("Do you authorize this?").allowed
    before = controller.export_state()
    assert controller.review_terminal_output("Do you authorize this?").allowed
    saved = controller.export_state()["items"][0]
    assert saved == before["items"][0]
    assert saved["status"] == "pending"
    assert saved["obligations"][0]["status"] == "ready"
    assert not controller.preview_terminal_output("Do you authorize this?").allowed
    ledger.observe_input(UserMessage.text("Yes."))
    assert controller.preview_terminal_output("Acknowledged.").allowed


def test_communication_does_not_hide_independently_ready_action():
    controller, _ = setup_plan(
        {
            **outcome(),
            "obligations": [obligation(kind="communication"), obligation(id="inspect")],
        }
    )
    assert not controller.preview_terminal_output("A question").allowed
    assert not update(
        controller,
        obligations=[
            {
                "id": "inspect",
                "status": "waiting",
                "evidence": "Needs customer input",
                "evidence_refs": [
                    {"kind": "customer_message", "reference_id": "customer-message:1"}
                ],
            }
        ],
    ).is_error
    assert controller.preview_terminal_output("A question").allowed


def test_communication_kind_survives_patch_and_restore():
    controller, ledger = setup_plan()
    assert not update(controller, obligations=[obligation(kind="communication")]).is_error
    assert not update(
        controller, obligations=[{"id": "validate-address", "evidence": "Ready to ask"}]
    ).is_error
    restored = BankingWorkflowController.from_history(
        [
            AssistantMessage(
                role="assistant",
                content="Question",
                raw_data={"cayu_tau3_workflow": controller.export_state()},
            )
        ],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == controller.export_state()
    assert restored.preview_terminal_output("Question").allowed
    assert update(restored, obligations=[{"id": "validate-address", "kind": []}]).is_error


def test_native_runtime_delivers_communication_with_unfinished_operation(tmp_path):
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(
                SYNC_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            **outcome(),
                            "obligations": [
                                obligation(kind="communication", goal="Ask for consent")
                            ],
                        }
                    ]
                },
            ),
            text("Do you authorize this operation?"),
        ],
        planning_mode="outcomes",
        native_recall=True,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Help with my request."), state)
        assert reply.content == "Do you authorize this operation?"
        assert len(provider.requests) == 2
        assert captured["state"].controller.export_state()["items"][0]["outcome"] == "unresolved"
    finally:
        state.stop()


def test_renewed_request_gets_new_identity_without_rewriting_completed_history():
    controller, ledger = setup_plan()
    ledger.observe_input(UserMessage.text("I no longer want this operation."))
    assert not update(
        controller,
        status="completed",
        outcome="declined",
        evidence="Customer declined",
        evidence_refs=[{"kind": "customer_message", "reference_id": "customer-message:2"}],
    ).is_error
    completed = controller.export_state()["items"][0]
    ledger.observe_input(UserMessage.text("I have changed my mind; please proceed."))
    renewed = outcome(
        "renewed-request",
        evidence="Customer renewed the request",
        evidence_refs=[{"kind": "customer_message", "reference_id": "customer-message:3"}],
    )
    assert not controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [renewed]}).is_error
    assert controller.export_state()["items"][0] == completed
    assert not controller.preview_terminal_output("Done.").allowed


@pytest.mark.parametrize(
    "patch",
    [
        [obligation(goal="Different goal")],
        [obligation(evidence_refs=[{"kind": "tool_result", "reference_id": "invented"}])],
        [obligation("resolved", evidence_refs=[])],
        [obligation("waiting", evidence="")],
        [obligation(), obligation()],
        [obligation(status=[])],
        [obligation()] * 9,
    ],
)
def test_invalid_obligation_updates_are_atomic(patch):
    controller, _ = setup_plan(*shipping_plan())
    before = controller.export_state()
    assert update(controller, obligations=patch).is_error
    assert controller.export_state() == before


def test_obligation_field_patches_preserve_identity_evidence_and_omitted_entries():
    controller, _ = setup_plan(*shipping_plan())
    assert not update(controller, obligations=[obligation("waiting")]).is_error
    before = controller.export_state()
    assert not update(controller, obligations=[]).is_error
    assert controller.export_state() == before
    assert not update(
        controller, obligations=[{"id": "validate-address", "status": "ready"}]
    ).is_error
    saved = controller.export_state()["items"][1]["obligations"][0]
    assert saved["goal"] == obligation()["goal"]
    assert saved["evidence_refs"][0]["available"] is True
    assert not update(controller, obligations=[obligation(id="confirm-window")]).is_error
    assert len(controller.export_state()["items"][1]["obligations"]) == 2
    assert update(controller, obligations=[{"id": "missing-new-fields"}]).is_error


def test_only_ready_obligations_default_missing_evidence_fields():
    controller, _ = setup_plan()
    minimal = {"id": "inspect", "goal": "Inspect the record", "status": "ready"}
    assert not update(controller, obligations=[minimal]).is_error
    saved = controller.export_state()["items"][0]["obligations"][0]
    assert saved["evidence"] == "" and saved["evidence_refs"] == []
    for status in ("waiting", "resolved", "not_applicable"):
        assert update(controller, obligations=[{**minimal, "status": status}]).is_error


@pytest.mark.parametrize("status", ["pending", "in_progress"])
def test_new_nonterminal_outcomes_default_empty_evidence(status):
    controller, _ = setup_plan()
    new = outcome("new-request", status=status)
    del new["evidence"]
    del new["evidence_refs"]
    result = controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [new]})
    assert not result.is_error, result.content
    saved = controller.export_state()["items"][-1]
    assert saved["evidence"] == "" and saved["evidence_refs"] == []


@pytest.mark.parametrize("missing", ["goal", "completion_mode", "depends_on", "outcome"])
def test_outcome_shape_error_only_requests_public_fields(missing):
    controller, _ = setup_plan()
    new = outcome("new-request")
    del new[missing]
    result = controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [new]})
    if missing in {"depends_on", "outcome"}:
        # New unstarted work now has safe defaults for these optional fields.
        assert not result.is_error, result.content
        saved = next(i for i in controller.export_state()["items"] if i["id"] == "new-request")
        assert saved[missing] == ([] if missing == "depends_on" else "unresolved")
        return
    assert result.is_error and missing in result.content
    assert "next_operation" not in result.content
    assert "new-request" not in {i["id"] for i in controller.export_state()["items"]}


@pytest.mark.parametrize("status", ["awaiting_customer", "blocked", "completed"])
def test_nonactionable_new_outcomes_do_not_default_required_evidence(status):
    controller, _ = setup_plan()
    new = outcome("new-request", status=status)
    del new["evidence"]
    del new["evidence_refs"]
    result = controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [new]})
    assert result.is_error and "evidence" in result.content
    assert "next_operation" not in result.content


def test_native_runtime_delivers_two_calls_from_one_model_round(tmp_path):
    from cayu import ModelStreamEvent

    batch = [
        call("host_action", {"value": 7}, "first")[0],
        call("host_action", {"value": 8}, "second")[0],
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]
    agent, provider, _ = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}),
            batch,
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "request",
                            "status": "completed",
                            "outcome": "succeeded",
                            "evidence": "Both operations completed",
                            "evidence_refs": [{"kind": "tool_result", "reference_id": "second"}],
                        }
                    ]
                },
            ),
            text("Done."),
        ],
        planning_mode="outcomes",
        native_recall=True,
    )
    state = agent.get_init_state()
    try:
        first, _ = agent.generate_next_message(UserMessage.text("Perform both operations."), state)
        assert first.tool_calls[0].arguments == {"value": 7}
        second, _ = agent.generate_next_message(
            ToolMessage(role="tool", id=first.tool_calls[0].id, content="ok"), state
        )
        assert second.tool_calls[0].arguments == {"value": 8}
        assert len(provider.requests) == 2
        reply, _ = agent.generate_next_message(
            ToolMessage(role="tool", id=second.tool_calls[0].id, content="ok"), state
        )
        assert reply.content == "Done."
        assert len(provider.requests) == 4
    finally:
        state.stop()


def test_obligation_evidence_and_partial_update_survive_restore():
    controller, ledger = setup_plan(*shipping_plan())
    observe(ledger, call_id="address-read")
    assert not update(
        controller,
        obligations=[
            obligation(
                "resolved", evidence_refs=[{"kind": "tool_result", "reference_id": "address-read"}]
            )
        ],
    ).is_error
    assert not update(
        controller,
        evidence="Still waiting for payment",
        evidence_refs=[{"kind": "tool_result", "reference_id": "address-read"}],
    ).is_error
    history = [
        AssistantMessage(
            role="assistant",
            content="Waiting",
            raw_data={"cayu_tau3_workflow": controller.export_state()},
        )
    ]
    restored = BankingWorkflowController.from_history(
        history, ledger=ledger, planning_mode="outcomes"
    )
    assert restored.export_state() == controller.export_state()
    assert restored.preview_terminal_output("Please pay.").allowed
    assert not update(restored, obligations=[obligation()]).is_error
    assert not restored.preview_terminal_output("Waiting for payment.").allowed


@pytest.mark.parametrize("field", ["evidence_refs", "obligations"])
def test_model_cannot_supply_internal_evidence_availability(field):
    controller, _ = setup_plan(*shipping_plan())
    refs = [{"kind": "customer_message", "reference_id": "customer-message:1", "available": True}]
    value = refs if field == "evidence_refs" else [obligation(evidence_refs=refs)]
    assert update(controller, **{field: value}).is_error


def test_unresolved_preparation_prevents_parent_completion():
    controller, ledger = setup_plan()
    observe(ledger)
    result = update(
        controller,
        status="completed",
        outcome="succeeded",
        evidence="Effect done",
        evidence_refs=[{"kind": "tool_result", "reference_id": "effect"}],
        obligations=[obligation("waiting")],
    )
    assert result.is_error
    assert "unresolved preparatory" in str(result.content)


def test_tool_failure_cannot_resolve_preparation_and_transfer_preserves_handoff():
    controller, ledger = setup_plan(*shipping_plan())
    observe(ledger, error=True)
    assert update(
        controller,
        obligations=[
            obligation("resolved", evidence_refs=[{"kind": "tool_error", "reference_id": "effect"}])
        ],
    ).is_error
    observe(ledger, name="transfer_to_human_agents", call_id="transfer")
    controller.reconcile_successful_human_transfer("transfer")
    saved = controller.export_state()["items"][1]
    assert saved["outcome"] == "handed_off"
    assert saved["obligations"][0]["status"] == "not_applicable"
    assert controller.preview_terminal_output("Transferred, not completed.").allowed


@pytest.mark.parametrize("status", ["pending", "in_progress"])
def test_native_runtime_continues_preparation_despite_blocked_final_effect(tmp_path, status):
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": shipping_plan(status)}),
            text("Waiting for payment."),  # Gate must reject this premature response.
            call("host_action", {"value": 7}, "address-read"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "request",
                            "obligations": [
                                obligation(
                                    "resolved",
                                    evidence_refs=[
                                        {"kind": "tool_result", "reference_id": "address-read"}
                                    ],
                                )
                            ],
                        }
                    ]
                },
            ),
            text("Address validated. Please complete payment before shipment."),
        ],
        planning_mode="outcomes",
        native_recall=True,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Prepare my shipment."), state)
        assert reply.tool_calls[0].arguments == {"value": 7}
        reply, _ = agent.generate_next_message(
            ToolMessage(role="tool", id=reply.tool_calls[0].id, content="Address valid"), state
        )
        assert reply.content == "Address validated. Please complete payment before shipment."
        assert len(provider.requests) == 5
        assert captured["state"].controller.export_state()["items"][1]["status"] == status
    finally:
        state.stop()


def test_schema_and_partial_updates_do_not_require_future_calls():
    controller, _ = setup_plan()
    schema = controller.tool_schemas()[0]["input_schema"]["properties"]["items"]["items"]
    assert schema["required"] == ["id"]
    assert "next_operation" not in schema["properties"]
    assert not update(controller, status="in_progress").is_error
    assert not update(controller, status="pending").is_error
    revision = controller.revision
    assert not update(controller).is_error
    assert controller.revision == revision
    assert "next_operation" not in controller.render_context()
    decisions = controller.review(
        (ToolCallPart(tool_call_id="read", tool_name="read_record", arguments={}),)
    )
    assert decisions[0].allowed


@pytest.mark.parametrize(
    "kind", ["customer_message", "tool_result", "tool_error", "knowledge_revision"]
)
@pytest.mark.parametrize(
    "patch", [{}, {"status": "in_progress"}, {"evidence": "Updated explanation"}]
)
def test_partial_updates_preserve_populated_evidence(kind, patch):
    controller, ledger = setup_plan()
    if kind == "customer_message":
        reference = {"kind": kind, "reference_id": "customer-message:1"}
    elif kind == "knowledge_revision":
        reference = {"kind": kind, "reference_id": "policy", "version": "1", "digest": "a" * 64}
        controller.observe_knowledge_evidence((WorkEvidenceReference(**reference),))
    else:
        observe(ledger, error=kind == "tool_error")
        reference = {"kind": kind, "reference_id": "effect"}
    assert not update(controller, evidence="Observed evidence", evidence_refs=[reference]).is_error
    before = controller.export_state()["items"][0]["evidence_refs"]
    assert before[0]["available"] is True
    revision = controller.revision
    result = update(controller, **patch)
    assert not result.is_error, result.content
    assert controller.export_state()["items"][0]["evidence_refs"] == before
    assert controller.revision == revision + bool(patch)


def test_new_prerequisite_and_partial_dependency_update_retain_evidence_atomically():
    controller, _ = setup_plan()
    reference = {"kind": "customer_message", "reference_id": "customer-message:1"}
    assert not update(
        controller,
        status="awaiting_customer",
        evidence="Need prerequisite",
        evidence_refs=[reference],
    ).is_error
    result = controller.invoke(
        UPDATE_WORK_PLAN_TOOL,
        {
            "items": [
                outcome("prerequisite", status="in_progress", evidence_refs=[reference]),
                {"id": "request", "status": "pending", "depends_on": ["prerequisite"]},
            ]
        },
    )
    assert not result.is_error, result.content
    saved = controller.export_state()
    assert saved["items"][0]["evidence_refs"][0]["reference_id"] == "customer-message:1"
    assert saved["items"][0]["depends_on"] == ["prerequisite"]
    # Only trusted retained state is translated. Caller-supplied internal fields
    # and fabricated references still reject the entire batch without mutation.
    for invalid in (
        {**reference, "available": True},
        {"kind": "tool_result", "reference_id": "invented"},
    ):
        rejected = controller.invoke(
            UPDATE_WORK_PLAN_TOOL,
            {
                "items": [
                    {"id": "prerequisite", "evidence": "Must not be committed"},
                    {"id": "request", "evidence_refs": [invalid]},
                ]
            },
        )
        assert rejected.is_error
        assert controller.export_state() == saved


def test_incomplete_outcomes_prevent_completion_and_fabricated_evidence_is_atomic():
    controller, _ = setup_plan(outcome(), outcome("second"))
    revision = controller.revision
    assert not controller.preview_terminal_output("Done").allowed
    assert update(
        controller,
        status="completed",
        outcome="succeeded",
        evidence="Done",
        evidence_refs=[{"kind": "tool_result", "reference_id": "invented"}],
    ).is_error
    assert controller.revision == revision
    assert len(controller.export_state()["items"]) == 2


def test_real_effect_evidence_completes_without_predeclaring_tool_arguments():
    controller, ledger = setup_plan()
    observe(ledger)
    assert not update(
        controller,
        status="completed",
        outcome="succeeded",
        evidence="Operation returned success",
        evidence_refs=[{"kind": "tool_result", "reference_id": "effect"}],
    ).is_error
    assert controller.preview_terminal_output("Done").allowed
    assert update(controller, status="pending", outcome="unresolved").is_error


def test_failed_result_does_not_prove_success_but_can_support_an_unresolved_blocker():
    controller, ledger = setup_plan()
    observe(ledger, error=True)
    refs = [{"kind": "tool_error", "reference_id": "effect"}]
    assert update(
        controller, status="completed", outcome="succeeded", evidence="Failed", evidence_refs=refs
    ).is_error
    assert not update(
        controller,
        status="blocked",
        outcome="blocked",
        evidence="Service unavailable",
        evidence_refs=refs,
    ).is_error
    assert controller.preview_terminal_output("The service failed; this has not completed.").allowed
    assert controller.export_state()["items"][0]["status"] == "blocked"


def test_waiting_resumes_after_customer_input_without_status_repair_loop():
    controller, ledger = setup_plan()
    assert not update(
        controller, status="awaiting_customer", evidence="Need the target account"
    ).is_error
    assert controller.review_terminal_output("Which account?").allowed
    assert not controller.preview_terminal_output("Which account?").allowed
    ledger.observe_input(UserMessage.text("Use the savings account."))
    assert not update(controller, status="pending").is_error
    assert not controller.preview_terminal_output("Done").allowed


def test_dependent_work_waits_but_independent_work_still_prevents_closure():
    controller, _ = setup_plan(
        outcome(), outcome("dependent", depends_on=["request"]), outcome("independent")
    )
    assert not update(
        controller, status="awaiting_customer", evidence="Need customer input"
    ).is_error
    assert not controller.preview_terminal_output("Please provide input").allowed
    # Cycles and deleting an existing obligation remain invalid and atomic.
    assert update(controller, depends_on=["dependent"]).is_error
    assert controller.invoke(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}).is_error


def test_handoff_requires_real_transfer_and_reconciles_all_unfinished_outcomes():
    controller, ledger = setup_plan(outcome(), outcome("second"))
    observe(ledger)
    assert update(
        controller,
        status="completed",
        outcome="handed_off",
        evidence="Handed off",
        evidence_refs=[{"kind": "tool_result", "reference_id": "effect"}],
    ).is_error
    observe(ledger, "transfer_to_human_agents", "transfer")
    controller.reconcile_successful_human_transfer("transfer")
    assert {i["outcome"] for i in controller.export_state()["items"]} == {"handed_off"}


def test_restore_keeps_mode_and_partial_update_contract():
    controller, ledger = setup_plan()
    history = [
        AssistantMessage(
            role="assistant",
            content="Working",
            raw_data={"cayu_tau3_workflow": controller.export_state()},
        )
    ]
    restored = BankingWorkflowController.from_history(
        history, ledger=ledger, planning_mode="outcomes"
    )
    assert restored.export_state() == controller.export_state()
    assert not update(restored, status="in_progress").is_error
    with pytest.raises(ValueError, match="different planning mode"):
        BankingWorkflowController.from_history(history, ledger=ledger)


def test_checklist_does_not_bypass_discovery_or_duplicate_effect_admission():
    controller, ledger = setup_plan()
    admission = BankingToolAdmissionGate(ledger)
    gate = CompositeExternalToolCallGate((controller, admission))
    locked = ToolCallPart(
        tool_call_id="locked",
        tool_name="call_discoverable_agent_tool",
        arguments={"agent_tool_name": "perform_operation", "arguments": {}},
    )
    assert not asyncio.run(gate.review((locked,)))[0].allowed
    confirm_effect(
        ledger, "unlock", "unlock_discoverable_agent_tool", agent_tool_name="perform_operation"
    )
    assert asyncio.run(gate.review((locked,)))[0].allowed
    confirm_verification(ledger)
    confirm_effect(
        ledger, "prior-effect", "change_user_email", user_id="user-7", new_email="new@example.com"
    )
    duplicate = ToolCallPart(
        tool_call_id="duplicate",
        tool_name="change_user_email",
        arguments={"user_id": "user-7", "new_email": "new@example.com"},
    )
    denied = asyncio.run(gate.review((duplicate,)))[0]
    assert not denied.allowed
    assert "already confirmed successful" in denied.reason


def test_native_runtime_executes_two_host_calls_without_intermediate_plan_updates(tmp_path):
    initial = work(operation=True)
    initial.pop("next_operation")
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [initial]}),
            call("host_action", {"value": 7}, "first"),
            call("host_action", {"value": 8}, "second"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "outcome",
                            "status": "completed",
                            "outcome": "succeeded",
                            "evidence": "Requested operation completed",
                            "evidence_refs": [{"kind": "tool_result", "reference_id": "second"}],
                        }
                    ]
                },
                "finish-plan",
            ),
            text("Done."),
        ],
        planning_mode="outcomes",
        native_recall=True,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Perform both operations."), state)
        assert reply.tool_calls[0].arguments == {"value": 7}
        reply, _ = agent.generate_next_message(
            ToolMessage(role="tool", id=reply.tool_calls[0].id, content="ok"), state
        )
        assert reply.tool_calls[0].arguments == {"value": 8}
        reply, _ = agent.generate_next_message(
            ToolMessage(role="tool", id=reply.tool_calls[0].id, content="ok"), state
        )
        assert reply.content == "Done."
        assert len(provider.requests) == 5
        assert captured["state"].controller.revision == 2
        assert (
            json.loads(json.dumps(reply.raw_data))["cayu_tau3_workflow"]["planning_mode"]
            == "outcomes"
        )
    finally:
        state.stop()


def test_new_observations_reenable_private_helpers_without_plan_churn(tmp_path):
    agent, _, captured = assembly(tmp_path, [text("unused")], planning_mode="outcomes")
    state = agent.get_init_state()
    bank = captured["state"]

    async def scenario():
        await bank.customer("Please perform the requested work.")
        assert not (await bank.memory.invoke(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]})).is_error
        bank.memory._refresh_private_action_epoch()
        completed = bank.memory._private_actions_completed_at_plan_revision
        completed.add("example_read_only_helper")
        bank.memory._refresh_private_action_epoch()
        assert "example_read_only_helper" in completed
        revision = bank.controller.revision
        await bank.customer("Here are the missing details.")
        bank.memory._refresh_private_action_epoch()
        assert not completed
        assert bank.controller.revision == revision

    try:
        asyncio.run_coroutine_threadsafe(scenario(), state._loop).result(10)
    finally:
        state.stop()
