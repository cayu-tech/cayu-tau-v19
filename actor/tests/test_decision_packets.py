import json

import pytest
from cayu import WorkEvidenceReference
from tau2.data_model.message import UserMessage
from test_admission import proposed
from test_outcome_controller import obligation, outcome, setup_plan
from test_runtime_banking import assembly, call, text

from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.decision_packets import DecisionPacketGate, normalized_operation


def packet_item(*, value=7, effects=None, choice=None):
    result = outcome("change", completion_mode="operation")
    packet = {"tool_name": "host_action", "arguments": {"value": value}}
    if effects is not None:
        packet["outcome_effects"] = effects
    if choice:
        packet["customer_choice"] = choice
    result["obligations"] = [obligation(planned_tool=packet)]
    return result


def test_advisory_runtime_blocks_wrong_mutation_and_dispatches_only_repair(tmp_path, monkeypatch):
    # A synthetic host mutation exercises the real runtime dispatch path without
    # baking any benchmark task, resource, or answer into the regression.
    import cayu_tau3_banking.runtime_banking as runtime

    real_gate = DecisionPacketGate
    monkeypatch.setattr(
        runtime,
        "DecisionPacketGate",
        lambda controller, ledger: real_gate(controller, ledger, mutations={"host_action"}),
    )
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [packet_item()]}),
            call("host_action", {"value": 99}, "wrong"),
            call("host_action", {"value": 7}, "right"),
        ],
        controller_mode="advisory",
        planning_mode="outcomes",
        advice_mode="lean",
        decision_packets=True,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Make the change."), state)
        assert [(c.name, c.arguments) for c in reply.tool_calls] == [("host_action", {"value": 7})]
        assert [a.call_id for a in captured["state"].ledger.actions] == ["right"]
        assert captured["state"].packet_gate.counts["packet_missing_or_mismatch"] >= 1
        assert len(provider.requests) == 3
    finally:
        state.stop()


@pytest.mark.parametrize("payload", ["not JSON", "[]", '{"a":1,"a":2}', '{"a":NaN}'])
def test_wrapper_must_be_concrete_unambiguous_json(payload):
    with pytest.raises((ValueError, TypeError)):
        normalized_operation(
            "call_discoverable_agent_tool",
            {"agent_tool_name": "change_resource", "arguments": payload},
        )


def test_wrapper_and_inner_call_are_equivalent():
    assert normalized_operation("change_resource", {"b": 2, "a": 1}) == normalized_operation(
        "call_discoverable_agent_tool",
        {"agent_tool_name": "change_resource", "arguments": '{"a":1, "b":2}'},
    )
    assert normalized_operation("change_resource", {"a": True}) != normalized_operation(
        "change_resource", {"a": 1}
    )


def configured(*, effects=None, choice=None):
    first = packet_item(effects=effects, choice=choice)
    controller, ledger = setup_plan(first, outcome("other", completion_mode="operation"))
    ref = WorkEvidenceReference(kind="knowledge_revision", reference_id="limits", version="1")
    controller.observe_knowledge_evidence([ref])
    assert not controller.invoke(
        UPDATE_WORK_PLAN_TOOL,
        {
            "items": [
                {
                    "id": "change",
                    "obligations": [
                        {
                            "id": "validate-address",
                            "evidence": "Read shared resource requirements before changing it.",
                            "evidence_refs": [
                                ref.model_dump(
                                    mode="json", exclude_none=True, exclude={"available"}
                                )
                            ],
                        }
                    ],
                }
            ]
        },
    ).is_error
    return DecisionPacketGate(controller, ledger, mutations={"host_action"}), controller, ledger


@pytest.mark.parametrize(
    "effects,code",
    [
        (None, "packet_interference_unreviewed"),
        ({"other": "blocks"}, "packet_outcome_conflict"),
        ({"other": "unknown"}, "packet_outcome_conflict"),
    ],
)
def test_cross_outcome_preflight_denies_before_irreversible_call(effects, code):
    gate, _, _ = configured(effects=effects)
    decision = gate.review([proposed("change", "host_action", value=7)])[0]
    assert not decision.allowed
    assert decision.diagnostic_code == code


def test_independent_outcome_does_not_block_source_backed_mutation():
    gate, _, _ = configured(effects={"other": "preserves"})
    assert gate.review([proposed("change", "host_action", value=7)])[0].allowed


def test_customer_choice_requires_available_current_customer_reference():
    gate, controller, ledger = configured(
        effects={"other": "customer_accepted"}, choice="customer-message:2"
    )
    action = proposed("change", "host_action", value=7)
    assert not gate.review([action])[0].allowed
    ledger.observe_input(UserMessage.text("I accept delaying the other operation."))
    assert gate.review([action])[0].allowed
    ledger.observe_input(UserMessage.text("Actually, preserve that other operation."))
    assert not gate.review([action])[0].allowed
    # Packet effects survive workflow snapshot serialization.
    saved = controller.export_state()
    assert json.loads(json.dumps(saved))["items"][0]["obligations"][0]["planned_tool"][
        "outcome_effects"
    ] == {"other": "customer_accepted"}


def test_original_request_cannot_authorize_a_later_sequencing_consequence():
    gate, _, _ = configured(effects={"other": "customer_accepted"}, choice="customer-message:1")
    assert gate.review([proposed("change", "host_action", value=7)])[0].diagnostic_code == (
        "packet_customer_choice_missing"
    )


def test_reads_do_not_require_packets():
    gate, _, _ = configured()
    assert gate.review([proposed("read", "get_current_time")])[0].allowed


def test_same_action_with_missing_plan_is_denied():
    controller, ledger = setup_plan(outcome())
    gate = DecisionPacketGate(controller, ledger, mutations={"host_action"})
    assert not gate.review([proposed("change", "host_action", value=7)])[0].allowed


def test_preservation_declaration_requires_available_policy():
    controller, ledger = setup_plan(
        packet_item(effects={"other": "preserves"}),
        outcome("other", completion_mode="operation"),
    )
    gate = DecisionPacketGate(controller, ledger, mutations={"host_action"})
    decision = gate.review([proposed("change", "host_action", value=7)])[0]
    assert decision.diagnostic_code == "packet_policy_evidence_missing"


def test_new_outcome_invalidates_old_scope_review():
    gate, controller, _ = configured(effects={"other": "preserves"})
    action = proposed("change", "host_action", value=7)
    assert gate.review([action])[0].allowed
    assert not controller.invoke(
        UPDATE_WORK_PLAN_TOOL, {"items": [outcome("new", completion_mode="operation")]}
    ).is_error
    assert gate.review([action])[0].diagnostic_code == "packet_interference_unreviewed"


def test_declared_dependency_cannot_be_bypassed_by_matching_packet():
    first = packet_item(effects={"other": "preserves"})
    first["depends_on"] = ["other"]
    controller, ledger = setup_plan(first, outcome("other", completion_mode="operation"))
    gate = DecisionPacketGate(controller, ledger, mutations={"host_action"})
    assert gate.review([proposed("change", "host_action", value=7)])[0].diagnostic_code == (
        "packet_prerequisite_pending"
    )


def test_production_supported_wrapper_is_bound_to_inner_packet():
    first = packet_item()
    packet = first["obligations"][0]["planned_tool"]
    packet.update(tool_name="freeze_debit_card_3892", arguments={"card_id": "synthetic-card"})
    controller, ledger = setup_plan(first)
    gate = DecisionPacketGate(controller, ledger)
    assert gate.review(
        [
            proposed(
                "call",
                "call_discoverable_agent_tool",
                agent_tool_name="freeze_debit_card_3892",
                arguments='{"card_id": "synthetic-card"}',
            )
        ]
    )[0].allowed
    assert not gate.review(
        [
            proposed(
                "call",
                "call_discoverable_agent_tool",
                agent_tool_name="freeze_debit_card_3892",
                arguments='{"card_id": "different-card"}',
            )
        ]
    )[0].allowed


def test_runtime_conflict_stops_dispatch_but_allows_customer_question(tmp_path, monkeypatch):
    import cayu_tau3_banking.runtime_banking as runtime

    def gate_with_observed_policy(controller, ledger):
        # Seed a synthetic read receipt; this test covers tool-policy dispatch,
        # not knowledge retrieval. Production reference validation is unchanged.
        controller.observe_knowledge_evidence(
            [WorkEvidenceReference(kind="knowledge_revision", reference_id="limits", version="1")]
        )
        return DecisionPacketGate(controller, ledger, mutations={"host_action"})

    monkeypatch.setattr(runtime, "DecisionPacketGate", gate_with_observed_policy)
    first = packet_item(effects={"other": "blocks"})
    first["obligations"][0]["evidence_refs"] = [
        {"kind": "knowledge_revision", "reference_id": "limits", "version": "1"}
    ]
    agent, _, captured = assembly(
        tmp_path,
        [
            call(
                SYNC_WORK_PLAN_TOOL,
                {"items": [first, outcome("other", completion_mode="operation")]},
            ),
            call("host_action", {"value": 7}, "would-block-other"),
            text(
                "This change would prevent the other requested operation. Which should take priority?"
            ),
        ],
        controller_mode="advisory",
        planning_mode="outcomes",
        advice_mode="lean",
        decision_packets=True,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Perform both operations."), state)
        assert not reply.tool_calls
        assert "priority" in reply.content
        assert not captured["state"].ledger.actions
        assert captured["state"].packet_gate.counts["packet_outcome_conflict"] >= 1
    finally:
        state.stop()
