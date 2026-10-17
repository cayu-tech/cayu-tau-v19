import pytest
from cayu import ScriptedModelProvider, ToolResultPart
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from tau2.data_model.message import ToolMessage, UserMessage
from test_evidence_candidate import record
from test_outcome_controller import outcome, setup_plan
from test_runtime_banking import assembly, call, offered, text
from test_runtime_compaction import message_texts

from cayu_tau3_banking.task_memory import EDIT_NOTES, READ_NOTES, UPDATE_CONSTRAINTS, TaskMemory


def memory(tmp_path, mode="markdown"):
    controller, ledger = setup_plan(outcome())
    ledger.observe_input(UserMessage.text("Complete both outcomes; preserve eligibility."))
    return TaskMemory(mode, tmp_path, ledger, controller)


def edit(notes, new, old=""):
    return notes.invoke(
        EDIT_NOTES, {"expected_revision": notes.revision, "old_text": old, "new_text": new}
    )


def constraint_args(notes, **extra):
    return {
        "expected_revision": notes.revision,
        "customer_message_id": notes.ledger.customer_message_ids[-1],
        "ordering_decision": "Confirm the change in priority before modifying the resource.",
        **extra,
    }


def test_markdown_edits_are_isolated_atomic_and_revision_checked(tmp_path):
    a, b = memory(tmp_path), memory(tmp_path)
    edit(a, "# Outcomes\nKeep the original card until the export is completed.\n")
    assert b.path.read_text() == "" and b.revision == 0
    edit(a, "the report", "the export")
    assert "the report" in a.path.read_text()
    before = a.path.read_text()
    with pytest.raises(ValueError, match="Stale"):
        a.invoke(EDIT_NOTES, {"expected_revision": 0, "old_text": "", "new_text": "bad"})
    with pytest.raises(ValueError, match="exactly once"):
        edit(a, "x", "missing")
    with pytest.raises(ValueError, match="16000"):
        edit(a, "x" * 16001)
    assert a.path.read_text() == before and a.revision == 2
    assert a.invoke(READ_NOTES, {})["markdown"] == before


def test_unknown_and_contradiction_sources_are_not_invented(tmp_path):
    n = memory(tmp_path, "constraints")
    fact = {
        "id": "eligibility",
        "relation": "other",
        "rule": "Eligibility after changing the card is unknown.",
        "affects": ["export"],
        "state": "unknown",
        "sources": [],
    }
    n.invoke(UPDATE_CONSTRAINTS, constraint_args(n, constraints=[fact]))
    bad = {**fact, "state": "active"}
    with pytest.raises(ValueError, match="actual source"):
        n.invoke(UPDATE_CONSTRAINTS, constraint_args(n, constraints=[bad]))
    bad["sources"] = [{"kind": "tool_result", "reference_id": "invented"}]
    with pytest.raises(ValueError, match="not been observed"):
        n.invoke(UPDATE_CONSTRAINTS, constraint_args(n, constraints=[bad]))
    assert n.data["constraints"]["eligibility"]["state"] == "unknown"
    with pytest.raises(ValueError, match="latest real customer"):
        n.invoke(UPDATE_CONSTRAINTS, {**constraint_args(n), "customer_message_id": "not-real"})


def test_notes_need_context_refresh_only_when_the_file_changes(tmp_path):
    n = memory(tmp_path)
    assert n.pre_action_error("close_debit_card_4721", {})
    assert n.pre_action_error("get_current_time", {}) is None
    edit(n, "Customer chose closure; verify all prerequisites before acting.")
    assert n.pre_action_error("close_debit_card_4721", {})
    n.context("step-1")
    assert n.pre_action_error("close_debit_card_4721", {}) is None
    # No mandatory new note/packet per operation.
    assert n.pre_action_error("order_debit_card_5739", {}) is None
    edit(n, "\nReplacement remains unknown; ask before proceeding.")
    assert n.pre_action_error("order_debit_card_5739", {})
    n.context("step-2")
    assert n.pre_action_error("order_debit_card_5739", {}) is None


@pytest.mark.parametrize("mode", ["constraints", "markdown"])
def test_real_runtime_notes_survive_lossy_compaction_and_customer_changes(tmp_path, mode):
    summary = "Older details deliberately omitted."
    summary_provider = ScriptedModelProvider([text(summary) for _ in range(25)])
    compaction = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(provider=summary_provider, model="lossy"),
        compact_after_messages=2,
        max_user_turns=1,
    )
    original = "Export must finish before retiring the original resource; retirement removes export eligibility."
    updated = "Customer now cancels export and authorizes retirement only; the export constraint is resolved."
    if mode == "markdown":
        first = call(EDIT_NOTES, {"expected_revision": 0, "old_text": "", "new_text": original})
        second = call(
            EDIT_NOTES, {"expected_revision": 1, "old_text": original, "new_text": updated}
        )
    else:
        first = call(
            UPDATE_CONSTRAINTS,
            {
                "expected_revision": 0,
                "customer_message_id": "customer-message:1",
                "ordering_decision": original,
                "constraints": [
                    {
                        "id": "preserve_export",
                        "relation": "other",
                        "rule": original,
                        "affects": ["export", "retire"],
                        "state": "active",
                        "sources": [
                            {"kind": "customer_message", "reference_id": "customer-message:1"}
                        ],
                    }
                ],
            },
        )
        second = call(
            UPDATE_CONSTRAINTS,
            {
                "expected_revision": 1,
                "customer_message_id": "customer-message:2",
                "ordering_decision": updated,
                "constraints": [
                    {
                        "id": "preserve_export",
                        "relation": "other",
                        "rule": updated,
                        "affects": ["retire"],
                        "state": "resolved",
                        "sources": [
                            {"kind": "customer_message", "reference_id": "customer-message:2"}
                        ],
                    }
                ],
            },
        )
    agent, provider, captured = assembly(
        tmp_path,
        [
            first,
            text("I will preserve the export path."),
            second,
            call("host_action", {"value": 7}, "retire"),
            text("The requested change completed."),
        ],
        planning_mode="outcomes",
        advice_mode="lean",
        controller_mode="advisory",
        compaction=compaction,
        task_memory_mode=mode,
        task_memory_directory=tmp_path / "notes",
        callback_timeout_s=60,
    )
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text(original), state)
        reply, _ = agent.generate_next_message(UserMessage.text(updated), state)
        assert reply.tool_calls
        agent.generate_next_message(
            ToolMessage(role="tool", id=reply.tool_calls[0].id, content="completed"), state
        )
        assert summary_provider.requests
        assert any(summary in t for request in provider.requests for t in message_texts(request))
        assert original in "\n".join(message_texts(provider.requests[1]))
        assert updated in "\n".join(message_texts(provider.requests[-1]))
        assert captured["state"].task_memory.revision == 2
        if mode == "markdown":
            for req in provider.requests:
                names = offered(req)
                assert EDIT_NOTES in names and READ_NOTES in names
                assert not any(n.startswith("controller_") for n in names)
                assert "cayu_update_decision_facts" not in names
                assert "cayu_check_comparison_facts" not in names
            assert captured["state"].execution_receipts is None
            assert captured["state"].decision_facts is None
            assert captured["state"].controller.export_state()["items"] == []
        assert not any(
            part.is_error
            for req in provider.requests
            for msg in req.messages
            for part in msg.content
            if isinstance(part, ToolResultPart)
        )
    finally:
        state.stop()


def test_retrospective_handoff_requires_real_success_but_not_prior_plan():
    from cayu_tau3_banking.controller import UPDATE_WORK_PLAN_TOOL

    controller, ledger = setup_plan(outcome())
    controller.allow_retrospective_outcomes = True
    record(ledger, "transfer_to_human_agents", {"reason": "technical_system_error"})
    item = outcome()
    item.update(
        id="later_note",
        completion_mode="operation",
        status="completed",
        outcome="handed_off",
        evidence="The requested transfer already succeeded.",
        evidence_refs=[{"kind": "tool_result", "reference_id": "executed"}],
    )
    result = controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [item]})
    assert not result.is_error, result.content
    item["id"] = "fabricated_note"
    item["evidence_refs"][0]["reference_id"] = "invented"
    result = controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [item]})
    assert result.is_error


def test_same_round_note_edit_cannot_execute_action_before_updated_context(tmp_path):
    from cayu import ModelStreamEvent
    from tau2.environment.tool import Tool

    def order_debit_card_5739(value: int) -> str:
        """Synthetic banking mutation.

        Args:
            value: Requested value.
        """
        raise AssertionError("Only the test host executes this")

    batch = [
        ModelStreamEvent.tool_call(
            id="edit-again",
            name=EDIT_NOTES,
            arguments={
                "expected_revision": 1,
                "old_text": "Initial decision",
                "new_text": "Updated customer choice",
            },
        ),
        ModelStreamEvent.tool_call(
            id="premature", name="order_debit_card_5739", arguments={"value": 7}
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(
                EDIT_NOTES, {"expected_revision": 0, "old_text": "", "new_text": "Initial decision"}
            ),
            batch,
            call("order_debit_card_5739", {"value": 7}, "after-reading"),
            text("Done."),
        ],
        planning_mode="outcomes",
        advice_mode="lean",
        controller_mode="advisory",
        task_memory_mode="markdown",
        task_memory_directory=tmp_path / "notes",
        host_tools=[Tool(order_debit_card_5739)],
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Please perform the requested action."), state
        )
        assert len(reply.tool_calls) == 1
        bank = captured["state"]
        assert [a.call_id for a in bank.ledger.actions] == ["after-reading"]
        assert "Updated customer choice" in "\n".join(message_texts(provider.requests[2]))
        agent.generate_next_message(
            ToolMessage(role="tool", id=reply.tool_calls[0].id, content="completed"), state
        )
        assert bank.task_memory.revision == 2
    finally:
        state.stop()


def test_partial_constraint_patch_preserves_definition_and_is_atomic(tmp_path):
    n = memory(tmp_path, 'constraints')
    fact = {'id': 'gap', 'rule': 'Eligibility unknown', 'relation': 'other', 'affects': ['export'], 'state': 'unknown', 'sources': []}
    n.invoke(UPDATE_CONSTRAINTS, constraint_args(n, constraints=[fact]))
    n.invoke(UPDATE_CONSTRAINTS, constraint_args(n, constraints=[{'id': 'gap', 'state': 'resolved'}]))
    assert n.data['constraints']['gap'] == {**fact, 'state': 'resolved'}
    before, revision = n.path.read_text(), n.revision
    for patch in ([{'id': 'new', 'state': 'unknown'}], [{'id': 'gap', 'relation': 'adverse_interaction'}], [{'id': 'gap', 'state': 'active'}], [{'id': 'gap', 'state': 'unknown'}, {'id': 'new', 'state': 'unknown'}], [{'id': 'gap', 'unexpected': 'x'}], [{'id': 'gap', 'state': 'active', 'sources': [{'kind': 'tool_result', 'reference_id': 'invented'}]}]):
        with pytest.raises(ValueError): n.invoke(UPDATE_CONSTRAINTS, constraint_args(n, constraints=patch))
        assert n.path.read_text() == before and n.revision == revision
