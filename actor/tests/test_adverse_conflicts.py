"""Source-backed generic ordering conflicts; no benchmark identities or answers."""

import copy
import json
from types import SimpleNamespace

import pytest
from cayu import WorkEvidenceReference
from jsonschema import validate
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from test_evidence_candidate import record
from test_outcome_controller import outcome, setup_plan, update

from cayu_tau3_banking.controller import BankingWorkflowController
from cayu_tau3_banking.controller_contract import ExecutionContractGate
from cayu_tau3_banking.decision_notes import DECISION_NOTE_SCHEMA, DecisionFacts
from cayu_tau3_banking.public_argument_contracts import PublicArgumentContracts


def setup():
    controller, ledger = setup_plan(outcome())
    ref = dict(kind="knowledge_revision", reference_id="fictional-procedure", version="1")
    controller.observe_knowledge_evidence([WorkEvidenceReference(**ref)])
    facts = DecisionFacts(ledger, controller)
    fact = dict(
        id="temporary-lock",
        claim="Locking the resource prevents finishing the requested change",
        state="true",
        sources=[ref],
        conflict=dict(
            affected_outcome="request",
            operation="lock_resource",
            arguments={"resource_id": "A"},
            prerequisite="Resource must remain unlocked until the change is complete",
        ),
    )
    return controller, ledger, facts, fact


@pytest.mark.parametrize("state", ["true", "unknown", "contradicted"])
def test_only_exact_declared_scope_is_blocked(state):
    _, _, facts, fact = setup()
    fact["state"] = state
    validate({"facts": [fact]}, DECISION_NOTE_SCHEMA["input_schema"])
    facts.update({"facts": [fact]})
    assert facts.conflict_error("lock_resource", {"resource_id": "A"})
    assert facts.conflict_error("lock_resource", {"resource_id": "B"}) is None
    assert facts.conflict_error("read_resource", {"resource_id": "A"}) is None
    assert facts.conflict_error("lock_resource", {"other_id": "A"}) is None
    fact["state"] = "false"
    facts.update({"facts": [fact]})
    assert facts.conflict_error("lock_resource", {"resource_id": "A"}) is None


def test_source_and_existing_outcome_required_atomically():
    _, _, facts, fact = setup()
    bad = copy.deepcopy(fact)
    bad["conflict"]["affected_outcome"] = "invented"
    with pytest.raises(ValueError, match="existing"):
        facts.update({"facts": [fact, {**bad, "id": "bad"}]})
    assert facts.snapshot()["facts"] == []
    fact["sources"] = [dict(kind="customer_message", reference_id="customer-message:1")]
    with pytest.raises(ValueError, match="policy source"):
        facts.update({"facts": [fact]})


def test_urgency_before_disclosure_cannot_be_used_as_informed_choice():
    _, ledger, facts, fact = setup()
    ledger.observe_assistant(
        AssistantMessage.text(
            "Locking now delays the other requested change. Which should go first?"
        )
    )
    fact["conflict"]["customer_choice"] = dict(disclosure_sequence=2, response_sequence=1)
    with pytest.raises(ValueError, match="followed"):
        facts.update({"facts": [fact]})
    ledger.observe_input(UserMessage.text("I understand the delay; lock it first."))
    fact["conflict"]["customer_choice"]["response_sequence"] = 3
    facts.update({"facts": [fact]})
    assert facts.conflict_error("lock_resource", {"resource_id": "A"}) is None


def test_success_resolves_conflict_but_bank_denial_does_not():
    for disposition in ["succeeded", "denied"]:
        controller, ledger, facts, fact = setup()
        facts.update({"facts": [fact]})
        record(ledger, "change_resource", {"resource_id": "A"})
        result = update(
            controller,
            status="completed",
            outcome=disposition,
            evidence="Observed decision",
            evidence_refs=[dict(kind="tool_result", reference_id="executed")],
        )
        assert not result.is_error, result.content
        assert bool(facts.conflict_error("lock_resource", {"resource_id": "A"})) == (
            disposition == "denied"
        )


def test_wrapper_and_native_gate_enforce_same_scope():
    _, ledger, facts, fact = setup()
    facts.update({"facts": [fact]})
    corpus = SimpleNamespace(
        document_directory=lambda: [], public_argument_contracts=PublicArgumentContracts({})
    )
    gate = ExecutionContractGate(corpus, ledger, facts)
    for name, args in [
        ("lock_resource", {"resource_id": "A"}),
        (
            "call_discoverable_agent_tool",
            {"agent_tool_name": "lock_resource", "arguments": json.dumps({"resource_id": "A"})},
        ),
    ]:
        call = SimpleNamespace(tool_name=name, tool_call_id="call", arguments=args)
        assert not gate.review([call])[0].allowed


def test_denied_needs_bank_result_and_round_trips():
    controller, ledger, _, _ = setup()
    patch = dict(
        status="completed",
        outcome="denied",
        evidence="Bank declined the requested change",
        evidence_refs=[dict(kind="customer_message", reference_id="customer-message:1")],
    )
    assert update(controller, **patch).is_error
    ledger.observe_assistant(
        AssistantMessage.text(
            "", tool_calls=[ToolCall(id="decision", name="submit_request", arguments={})]
        )
    )
    ledger.observe_input(
        ToolMessage(
            role="tool", id="decision", content="Request processed. Decision: denied.", error=False
        )
    )
    patch["evidence_refs"] = [dict(kind="tool_result", reference_id="decision")]
    result = update(controller, **patch)
    assert not result.is_error, result.content
    saved = controller.export_state()
    assert saved["items"][0]["outcome"] == "denied"
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("The bank denied it.", raw_data={"cayu_tau3_workflow": saved})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == saved


def test_runtime_preserves_conflict_through_compaction_and_allows_independent_work(tmp_path):
    from cayu import ScriptedModelProvider
    from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
    from test_runtime_banking import assembly, call, text

    from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL

    summaries = ScriptedModelProvider([text('Earlier details omitted.') for _ in range(12)])
    agent, provider, captured = assembly(
        tmp_path,
        [text('I will check the sequencing before acting.'),
         call('host_action', {'value': 7}, 'blocked'),
         call('host_action', {'value': 8}, 'independent')],
        controller_mode='advisory', planning_mode='outcomes', advice_mode='lean',
        persistent=True,
        compaction=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=summaries, model='summary'),
            compact_after_messages=2, max_user_turns=1),
    )
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text('Perform both outcomes.'), state)
        runtime = captured['state']
        runtime.controller.invoke(SYNC_WORK_PLAN_TOOL, {'items': [outcome()]})
        ref = dict(kind='knowledge_revision', reference_id='limits', version='1')
        runtime.controller.observe_knowledge_evidence([WorkEvidenceReference(**ref)])
        runtime.decision_facts.update({'facts': [dict(
            id='conflict', claim='This action blocks the other requested change', state='true', sources=[ref],
            conflict=dict(affected_outcome='request', operation='host_action', arguments={'value': 7}, prerequisite='Finish the other change first'))]})
        reply, _ = agent.generate_next_message(UserMessage.text('Continue the independent work.'), state)
        assert [(c.name, c.arguments) for c in reply.tool_calls] == [('host_action', {'value': 8})]
        assert summaries.requests
        projected = '\n'.join(p.text for m in provider.requests[-1].messages for p in m.content if p.type == 'text')
        assert 'Finish the other change first' in projected
        assert 'conflict_dialogue' in projected
        assert not any(a.arguments == {'value': 7} for a in runtime.ledger.actions)
    finally:
        state.stop()
