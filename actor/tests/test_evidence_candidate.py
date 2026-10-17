"""Behavioral regressions for the frozen-candidate requirement failures."""

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from test_decision_packets import packet_item
from test_outcome_controller import outcome, setup_plan
from test_runtime_banking import assembly, call, text

from cayu_tau3_banking.controller import UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.decision_notes import DECISION_NOTE_SCHEMA, DECISION_NOTE_TOOL, DecisionFacts
from cayu_tau3_banking.decision_packets import DecisionPacketGate


def record(ledger, name, args, *, call_id="executed", error=False):
    ledger.observe_assistant(
        AssistantMessage.text("", tool_calls=[ToolCall(id=call_id, name=name, arguments=args)])
    )
    ledger.observe_input(
        ToolMessage(
            role="tool",
            id=call_id,
            content="Operation succeeded" if not error else "Error: failed",
            error=error,
        )
    )


def test_advisory_runtime_never_calls_legacy_intent_matchers(tmp_path, monkeypatch):
    import cayu_tau3_banking.admission as admission
    import cayu_tau3_banking.agent_memory as memory
    import cayu_tau3_banking.ledger as ledger_module

    def forbidden(*args, **kwargs):
        raise AssertionError("Regex intent inference entered the candidate runtime")

    monkeypatch.setattr(memory, "customer_requests_human_assistance", forbidden)
    monkeypatch.setattr(memory, "_customer_requests_credit_card_dispute", forbidden)
    monkeypatch.setattr(ledger_module, "customer_requests_human_assistance", forbidden)
    monkeypatch.setattr(admission, "transfer_reason_denial", forbidden)
    monkeypatch.setattr(admission, "current_offer_capability_gap", forbidden)
    agent, _, captured = assembly(
        tmp_path,
        [text("Which transaction should I investigate?")],
        planning_mode="outcomes",
        advice_mode="lean",
        controller_mode="advisory",
        decision_packets=True,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Yes, I want a human review of this dispute."), state
        )
        assert reply.content == "Which transaction should I investigate?"
        assert captured["state"].admission.authoritative_only
        assert not captured["state"].ledger.interpret_customer_intent
        assert captured["state"].packet_gate is not None
    finally:
        state.stop()


def test_transfer_dispatch_does_not_require_regex_consent_phrase(tmp_path, monkeypatch):
    from tau2.environment.tool import Tool

    import cayu_tau3_banking.admission as admission
    import cayu_tau3_banking.ledger as ledger_module

    def forbidden(*args, **kwargs):
        raise AssertionError("Intent matcher must not determine consent")

    def transfer_to_human_agents(reason: str, summary: str) -> str:
        """Transfer the customer using a public reason and factual summary."""
        raise AssertionError("Only the external host executes this")

    monkeypatch.setattr(admission, "transfer_reason_denial", forbidden)
    monkeypatch.setattr(ledger_module, "customer_requests_human_assistance", forbidden)
    agent, _, _ = assembly(
        tmp_path,
        [
            call(
                "transfer_to_human_agents",
                {
                    "reason": "customer_requests_human_no_specific_reason",
                    "summary": "Customer requested human review.",
                },
                "transfer",
            )
        ],
        host_tools=[Tool(transfer_to_human_agents)],
        planning_mode="outcomes",
        advice_mode="lean",
        controller_mode="advisory",
        decision_packets=True,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Yes, I want a human review."), state
        )
        assert [c.name for c in reply.tool_calls] == ["transfer_to_human_agents"]
    finally:
        state.stop()


@pytest.mark.parametrize("error", [False, True])
def test_exact_packet_receipt_settles_only_real_success(error):
    from test_admission import proposed

    controller, ledger = setup_plan(packet_item())
    gate = DecisionPacketGate(controller, ledger, mutations={"host_action"})
    action = proposed("effect", "host_action", value=7)
    assert gate.review([action])[0].allowed
    record(ledger, "host_action", {"value": 7}, call_id="effect", error=error)
    gate.observe_result("effect")
    saved = controller.export_state()["items"][0]
    assert saved["obligations"][0]["status"] == ("ready" if error else "resolved")
    assert saved["status"] != "completed"
    assert gate.counts["settled_from_execution"] == (0 if error else 1)


def test_wrong_level_update_repairs_existing_child_without_new_outcome():
    controller, ledger = setup_plan(packet_item())
    child = controller.export_state()["items"][0]["obligations"][0]["id"]
    record(ledger, "host_action", {"value": 7})
    result = controller.invoke_model_update(
        UPDATE_WORK_PLAN_TOOL,
        {
            "items": [
                {
                    "id": child,
                    "status": "completed",
                    "outcome": "succeeded",
                    "completion_mode": "operation",
                    "evidence": "Actual operation completed",
                    "evidence_refs": [{"kind": "tool_result", "reference_id": "executed"}],
                }
            ]
        },
    )
    assert not result.is_error, result.content
    saved = controller.export_state()["items"]
    assert len(saved) == 1
    assert saved[0]["obligations"][0]["status"] == "resolved"


def test_wrong_level_update_cannot_invent_success():
    controller, _ = setup_plan(packet_item())
    child = controller.export_state()["items"][0]["obligations"][0]["id"]
    result = controller.invoke_model_update(
        UPDATE_WORK_PLAN_TOOL,
        {
            "items": [
                {
                    "id": child,
                    "status": "resolved",
                    "evidence": "Claimed success",
                    "evidence_refs": [{"kind": "tool_result", "reference_id": "invented"}],
                }
            ]
        },
    )
    assert result.is_error
    assert controller.export_state()["items"][0]["obligations"][0]["status"] == "ready"


@pytest.mark.parametrize("resource_field", ["account_id", "bank_account_id", "credit_card_account_id"])
def test_facts_preserve_unknown_conflicts_and_scoped_invalidation(resource_field):
    from jsonschema import validate
    controller, ledger = setup_plan(outcome())
    notes = DecisionFacts(ledger, controller)
    ledger.observe_input(UserMessage.text("The earlier date was incorrect."))
    refs = [{"kind": "customer_message", "reference_id": f"customer-message:{n}"} for n in [1, 2]]
    notes.update(
        {
            "facts": [
                {
                    "id": "eligibility",
                    "claim": "Account is eligible",
                    "state": "true",
                    "sources": refs[:1],
                    "invalidated_by": ["close_account"],
                    "resource": {"field": resource_field, "value": "A"},
                },
                {
                    "id": "date",
                    "claim": "Date is unresolved",
                    "state": "contradicted",
                    "sources": refs,
                },
                {"id": "unknown", "claim": "Missing status", "state": "unknown", "sources": []},
            ]
        }
    )
    validate({"facts": list(notes.facts.values())}, DECISION_NOTE_SCHEMA["input_schema"])
    record(ledger, "close_account", {resource_field: "B"}, call_id="other")
    notes.observe_result("other")
    assert notes.facts["eligibility"]["state"] == "true"
    record(ledger, "close_account", {resource_field: "A"})
    notes.observe_result("executed")
    assert notes.facts["eligibility"]["state"] == "unknown"
    assert notes.facts["date"]["state"] == "contradicted"
    assert notes.facts["unknown"]["state"] == "unknown"
    before = notes.snapshot()
    with pytest.raises(ValueError):
        notes.update(
            {
                "facts": [
                    {
                        "id": "bad",
                        "claim": "Fabricated",
                        "state": "true",
                        "sources": [{"kind": "tool_result", "reference_id": "fake"}],
                    }
                ]
            }
        )
    assert notes.snapshot() == before


def test_compact_notes_are_exposed_in_real_advisory_runtime(tmp_path):
    agent, _, captured = assembly(
        tmp_path,
        [
            call(
                DECISION_NOTE_TOOL,
                {
                    "facts": [
                        {
                            "id": "preference",
                            "claim": "Customer prefers no fee",
                            "state": "true",
                            "sources": [
                                {"kind": "customer_message", "reference_id": "customer-message:1"}
                            ],
                        }
                    ]
                },
            ),
            text("I will compare the applicable fees."),
        ],
        planning_mode="outcomes",
        advice_mode="lean",
        controller_mode="advisory",
        decision_packets=True,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("I prefer no annual fee."), state)
        assert reply.content == "I will compare the applicable fees."
        assert captured["state"].decision_facts.facts["preference"]["state"] == "true"
    finally:
        state.stop()


def test_admission_exhaustion_is_graded_but_provider_failure_is_not(monkeypatch):
    from cayu import ScriptedModelProvider
    from test_runtime_agent import make_agent

    from cayu_tau3_banking.runtime_agent import RuntimeCallbackExecutionError

    agent, _, _ = make_agent(ScriptedModelProvider([]), grade_budget_stops=True)
    state = agent.get_init_state([])
    try:

        def stall(message):
            raise RuntimeCallbackExecutionError("AdmissionStallError")

        monkeypatch.setattr(state, "advance", stall)
        reply, _ = agent.generate_next_message(UserMessage.text("Continue"), state)
        assert agent.is_stop(reply)
        assert reply.raw_data["agent_termination"] == "admission_stall"

        def outage(message):
            raise RuntimeCallbackExecutionError("ModelProviderError")

        monkeypatch.setattr(state, "advance", outage)
        with pytest.raises(RuntimeCallbackExecutionError):
            agent.generate_next_message(UserMessage.text("Continue"), state)
    finally:
        state.stop()


def test_profile_rejects_accidental_enforced_configuration():
    from cayu_tau3_banking.cli import _parser, _run

    args = _parser().parse_args(["--candidate-profile", "evidence-v2"])
    with pytest.raises(ValueError, match="evidence-v2 configuration mismatch"):
        _run(args)
