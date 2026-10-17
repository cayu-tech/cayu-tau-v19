import copy
import json

import pytest
from jsonschema import validate
from tau2.data_model.message import AssistantMessage, UserMessage
from test_adverse_conflicts import setup
from test_task_memory import constraint_args

from cayu_tau3_banking.task_memory import UPDATE_CONSTRAINTS, TaskMemory, schemas


def prepared(tmp_path):
    controller, ledger, _, fact = setup()
    memory = TaskMemory("constraints", tmp_path, ledger, controller)
    row = dict(
        id=fact["id"],
        rule=fact["claim"],
        state="active",
        relation="adverse_interaction",
        affects=["request"],
        sources=fact["sources"],
        conflict=fact["conflict"],
    )
    return memory, row


def save(memory, row):
    patch = constraint_args(memory, constraints=[row])
    validate(patch, schemas("constraints")[0]["input_schema"])
    memory.invoke(UPDATE_CONSTRAINTS, patch)
    memory.context("step-" + str(memory.revision))


def test_canonical_constraint_directly_blocks_exact_scope_and_preserves_independent_work(tmp_path):
    memory, row = prepared(tmp_path)
    save(memory, row)
    assert memory.pre_action_error("lock_resource", {"resource_id": "A"})
    assert memory.pre_action_error("lock_resource", {"resource_id": "B"}) is None
    assert memory.pre_action_error("read_resource", {"resource_id": "A"}) is None
    assert memory.pre_action_error(
        "call_discoverable_agent_tool",
        {"agent_tool_name": "lock_resource", "arguments": json.dumps({"resource_id": "A"})},
    )
    assert "conflict" in memory.data["constraints"][row["id"]]


def test_urgency_is_not_a_later_informed_choice_and_failed_patch_is_atomic(tmp_path):
    memory, row = prepared(tmp_path)
    save(memory, row)
    before = memory.path.read_text()
    memory.ledger.observe_assistant(
        AssistantMessage.text("Locking now delays the requested change. Choose an order.")
    )
    bad = copy.deepcopy(row)
    bad["conflict"]["customer_choice"] = dict(disclosure_sequence=2, response_sequence=1)
    with pytest.raises(ValueError, match="followed"):
        save(memory, bad)
    assert memory.path.read_text() == before
    memory.ledger.observe_input(UserMessage.text("I understand that delay; lock it first."))
    good = copy.deepcopy(row)
    good["conflict"]["customer_choice"] = dict(disclosure_sequence=2, response_sequence=3)
    save(memory, good)
    assert memory.pre_action_error("lock_resource", {"resource_id": "A"}) is None


def test_adverse_record_requires_structural_scope_and_real_policy(tmp_path):
    memory, row = prepared(tmp_path)
    bad = copy.deepcopy(row)
    del bad["conflict"]
    # Patch schema permits omitted fields; merged/new-record validation enforces them.
    with pytest.raises(ValueError, match="adverse interaction"):
        save(memory, bad)
    bad = copy.deepcopy(row)
    bad["sources"] = [{"kind": "customer_message", "reference_id": "customer-message:1"}]
    with pytest.raises(ValueError, match="policy source"):
        save(memory, bad)
    assert memory.revision == 0


def test_corrected_constraint_releases_gate(tmp_path):
    memory, row = prepared(tmp_path)
    save(memory, row)
    corrected = {
        **row,
        "state": "resolved",
        "rule": "The cited prerequisite does not apply to this resource.",
    }
    save(memory, corrected)
    assert memory.pre_action_error("lock_resource", {"resource_id": "A"}) is None


def test_scoped_reconciliation_is_retained_and_recoverable_after_projection(tmp_path):
    from test_saved_evidence import fact
    from test_saved_evidence import setup as evidence_setup

    memory, _, facts = evidence_setup()
    rows = [dict(fact(str(i)), scope="other") for i in range(20)]
    reconciliation = dict(
        fact("account-a"),
        scope="A",
        comparison={
            "alternative": "Non-overlapping domestic fee rule",
            "calculation": "Overcharge +4; missing fee -1; existing rebate -1; conditional net +2.",
            "conditions": "Applies only to the three identified domestic rows; foreign ownership remains unknown.",
            "next_missing_fact": "Authoritative ownership of the separate foreign charge.",
        },
    )
    facts.apply_patch({"facts": [reconciliation, *rows]})
    assert any(f["id"] == "account-a" for f in facts.context_snapshot()["facts"])
    recovered = memory.read({"kind": "notes", "scope": "A"})["rows"]
    assert recovered == [reconciliation]
    assert memory.read({"kind": "notes", "scope": "absent"})["rows"] == []


@pytest.mark.parametrize(
    "placeholder",
    [
        None,
        {"disclosure_sequence": 0, "response_sequence": 0},
        {"disclosure_sequence": 2, "response_sequence": 0},
    ],
)
def test_native_constraint_warns_without_granting_self_authored_veto_or_consent(tmp_path, placeholder):
    from test_runtime_banking import assembly, call, text

    (tmp_path / "policy.json").write_text(
        json.dumps(
            {
                "id": "policy",
                "title": "Resource prerequisite",
                "content": "Changing value to 7 blocks the unfinished customer request.",
            }
        )
    )
    row = dict(
        id="block",
        rule="The proposed change blocks the request.",
        relation="adverse_interaction",
        affects=["outcome"],
        state="active",
        sources=[{"kind": "knowledge_revision", "reference_id": "policy", "version": "1"}],
        conflict={
            "affected_outcome": "outcome",
            "operation": "host_action",
            "arguments": {"value": 7},
            "prerequisite": "Keep value unchanged until the request is complete.",
        },
    )
    if placeholder is not None:
        row["conflict"]["customer_choice"] = placeholder
    agent, _provider, captured = assembly(
        tmp_path,
        [
            call(
                "controller_sync_work_plan",
                {
                    "items": [
                        {
                            "id": "outcome",
                            "goal": "Finish the requested change",
                            "completion_mode": "operation",
                            "status": "pending",
                            "outcome": "unresolved",
                        }
                    ]
                },
                "plan",
            ),
            call("cayu_read_product_documents", {"documents": [{"entry_id": "policy"}]}, "read"),
            call(
                UPDATE_CONSTRAINTS,
                {
                    "expected_revision": 0,
                    "ordering_decision": "Explain the adverse interaction first.",
                    "constraints": [row],
                },
                "notes",
            ),
            call("host_action", {"value": 7}, "blocked"),
            text("The change would delay your other request. Which outcome should go first?"),
        ],
        controller_mode="advisory",
        planning_mode="outcomes",
        advice_mode="lean",
        task_memory_mode="constraints",
        task_memory_directory=tmp_path / "notes",
        persistent=True,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Change it urgently and finish my other request."), state
        )
        assert [(c.name, c.arguments) for c in reply.tool_calls] == [("host_action", {"value": 7})]
        runtime = captured["state"]
        assert not runtime.decision_facts.facts
        assert (
            runtime.task_memory.data["constraints"]["block"]["conflict"]["operation"]
            == "host_action"
        )
        assert runtime.task_memory.pre_action_error("host_action", {"value": 7}) is None
        conflict = runtime.task_memory.data["constraints"]["block"]["conflict"]
        assert "customer_choice" not in conflict
        assert runtime.task_memory.data["constraints"]["block"]["state"] == "active"
        assert row["rule"] in runtime.task_memory.context("next")
    finally:
        state.stop()
