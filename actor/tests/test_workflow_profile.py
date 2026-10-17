"""Regression checks through the actual advisory constraints runtime assembly."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from jsonschema import validate
from tau2.data_model.message import UserMessage
from test_runtime_banking import assembly, call, text

from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL
from cayu_tau3_banking.public_argument_contracts import (
    PublicArgumentContract,
    PublicArgumentContracts,
)
from cayu_tau3_banking.task_memory import UPDATE_CONSTRAINTS


def candidate(tmp_path, steps, **kwargs):
    return assembly(
        tmp_path, steps, planning_mode="outcomes", advice_mode="lean",
        controller_mode="advisory", task_memory_mode="constraints", **kwargs,
    )


def test_assembled_prompt_and_tools_select_the_lightweight_interface(tmp_path):
    agent, provider, captured = candidate(tmp_path, [text("Hello.")])
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Compare my options."), state)
        request = provider.requests[0]
        system = "\n".join(p.text for m in request.messages if m.role == "system"
                           for p in m.content if p.type == "text")
        assert "add scope to normal cayu_search_banking_knowledge" in system
        assert "BEFORE a recommendation" in system
        assert "Notes are optional" in system
        assert "customer-owned steps" in system
        assert "Before the first consequential operation, use cayu_update_constraints" not in system
        schemas = {tool["name"]: tool for tool in request.tools}
        search = schemas[KNOWLEDGE_SEARCH_TOOL]["input_schema"]
        for args in ({"query": "limit", "scope": {"entry_ids": ["limits"]}},
                     {"queries": [{"query": "limit", "mode": "literal",
                                   "scope": {"entry_ids": ["limits"]}}]}):
            validate(args, search)
        plan = schemas[SYNC_WORK_PLAN_TOOL]
        item = plan["input_schema"]["properties"]["items"]["items"]
        assert "obligations" not in item["properties"]
        assert "planned_tool" not in item["properties"]
        assert len(json.dumps(plan)) < 6000
        assert not captured["state"].task_memory.enforce_notes
        assert not captured["state"].admission_progress.fatal
    finally:
        state.stop()


def test_eight_denials_allow_corrected_call_without_weakening_contract(tmp_path):
    def configure(corpus):
        source = "host_action parameter value must be an integer no greater than 7."
        contract = PublicArgumentContract(
            "host_action", "synthetic_api", 1, hashlib.sha256(source.encode()).hexdigest(),
            source, json.dumps({"type": "object", "properties": {"value": {"maximum": 7}}}),
        )
        corpus.public_argument_contracts = PublicArgumentContracts(
            {"synthetic_api": SimpleNamespace(revision=1, text=source)}, [contract]
        )

    steps = []
    for i in range(8):
        steps.append(call("host_action", {"value": 9}, f"denied-{i}"))
        if i < 7:
            steps.append(call("cayu_update_decision_facts", {"facts": [{
                "id": f"fact-{i}", "claim": "Need the authoritative limit.",
                "state": "unknown", "sources": [],
            }]}, f"note-{i}"))
    steps.append(call("host_action", {"value": 7}, "corrected"))
    agent, provider, captured = candidate(tmp_path, steps, configure_corpus=configure,
                                           callback_timeout_s=120)
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Perform the action."), state)
        assert len(provider.requests) == 16
        assert [(c.name, c.arguments) for c in reply.tool_calls] == [("host_action", {"value": 7})]
        assert [a.call_id for a in captured["state"].ledger.actions] == ["corrected"]
        assert len(captured["state"].decision_facts.facts) == 7
        assert "stop this attempt" not in captured["state"].admission_progress.warning
    finally:
        state.stop()


def test_outcome_field_patches_keep_evidence_checks_and_reject_invented_success(tmp_path):
    agent, provider, captured = candidate(tmp_path, [
        call(SYNC_WORK_PLAN_TOOL, {"items": [{"id": "action", "goal": "Perform the action",
                                            "completion_mode": "operation"}]}),
        call(UPDATE_WORK_PLAN_TOOL, {"items": [{"id": "action", "status": "in_progress"}]}, "patch"),
        call(UPDATE_WORK_PLAN_TOOL, {"items": [{"id": "action", "status": "completed",
             "outcome": "succeeded", "evidence": "Claimed execution", "evidence_refs": [
                 {"kind": "tool_result", "reference_id": "invented"}]}]}, "bad-proof"),
        text("I have not completed the action."),
    ])
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Perform the action."), state)
        item = captured["state"].controller.export_state()["items"][0]
        assert item["status"] == "in_progress" and item["outcome"] == "unresolved"
        assert "not completed" in reply.content and len(provider.requests) == 4
    finally:
        state.stop()


def test_unexposed_optional_notes_do_not_veto_reply_or_independent_action(tmp_path):
    import asyncio

    from cayu_tau3_banking.runtime_banking import BankingReplyPolicy

    agent, _, captured = candidate(tmp_path, [call("host_action", {"value": 7}, "valid")])
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Perform the action."), state)
        assert reply.tool_calls[0].arguments == {"value": 7}
        runtime = captured["state"]
        runtime.task_memory.invoke(UPDATE_CONSTRAINTS, {
            "expected_revision": 0, "ordering_decision": "Continue independently.",
            "constraints": [],
        })
        assert runtime.task_memory.exposed_revision != runtime.task_memory.revision
        assert runtime.task_memory.pre_action_error("close_debit_card_4721", {}) is None
        decision = asyncio.run(BankingReplyPolicy(runtime).before_stop(SimpleNamespace()))
        assert decision == type(decision).complete()
        assert "Continue independently." in runtime.task_memory.context("next")
    finally:
        state.stop()


@pytest.mark.parametrize("status", ["active", "unknown", "contradicted"])
def test_model_constraints_preserve_uncertainty_without_becoming_authority(tmp_path, status):
    from test_canonical_constraints import prepared, save
    memory, row = prepared(tmp_path)
    memory.enforce_notes = False
    row["state"] = status
    save(memory, row)
    assert memory.pre_action_error("lock_resource", {"resource_id": "A"}) is None
    assert memory.data["constraints"][row["id"]]["state"] == status
    context = memory.context("next")
    assert row["rule"] in context and 'authoritative="false"' in context
