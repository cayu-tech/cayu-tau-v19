import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from cayu import ToolCallPart
from jsonschema import Draft202012Validator
from tau2.data_model.message import UserMessage
from test_decision_packets import packet_item
from test_evidence_candidate import record
from test_outcome_controller import outcome, setup_plan
from test_runtime_banking import assembly, call, text

from cayu_tau3_banking.cli import _parser, _run
from cayu_tau3_banking.execution_receipts import ExecutionReceipts
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.operation_guidance import CARDS, PublicOperationGuidance

DOCS = (
    Path(__file__).resolve().parents[1]
    / ".vendor/tau2/data/tau2/domains/banking_knowledge/documents"
)


def guidance():
    entries = {}
    for c in CARDS:
        d = json.loads((DOCS / (c["source"] + ".json")).read_text())
        entries[c["source"]] = SimpleNamespace(text=d["content"].strip(), revision=1)
    return PublicOperationGuidance(entries)


def proposal(wrapper, tool):
    field = (
        "discoverable_tool_name" if wrapper == "give_discoverable_user_tool" else "agent_tool_name"
    )
    return ToolCallPart(tool_call_id="test", tool_name=wrapper, arguments={field: tool})


def test_public_customer_ownership_and_valid_handoff():
    g = guidance()
    bad = g.review([proposal("unlock_discoverable_agent_tool", "get_card_last_4_digits")])[0]
    assert not bad.allowed and "give_discoverable_user_tool" in bad.reason
    assert g.review([proposal("give_discoverable_user_tool", "get_card_last_4_digits")])[0].allowed
    assert not g.review(
        [proposal("give_discoverable_user_tool", "update_transaction_rewards_3847")]
    )[0].allowed
    assert g.review([proposal("unlock_discoverable_agent_tool", "undocumented_example")])[0].allowed


def test_changed_public_source_fails_closed():
    c = CARDS[0]
    with pytest.raises(ValueError, match="source changed"):
        PublicOperationGuidance({c["source"]: SimpleNamespace(text="changed", revision=1)})


def test_directory_does_not_claim_full_source_delivery():
    content = guidance().context()
    assert 'full_documents_delivered="false"' in content
    assert "cash_back_disputes" in content and "highest applicable tier" in content
    assert "evidence_ref" not in content


@pytest.mark.parametrize("error", [False, True])
def test_receipts_do_not_need_a_plan_and_preserve_failure(error):
    controller, ledger = setup_plan(outcome())
    receipts = ExecutionReceipts(ledger, controller)
    record(ledger, "host_action", {"value": 7}, error=error)
    receipts.observe_result("executed")
    receipts.observe_result("executed")
    assert len(receipts.receipts) == 1
    assert receipts.receipts[0]["execution_status"] == ("failed" if error else "succeeded")
    assert controller.export_state()["items"][0]["status"] != "completed"


@pytest.mark.parametrize("error", [False, True])
def test_optional_matching_step_settles_only_on_success(error):
    controller, ledger = setup_plan(packet_item())
    receipts = ExecutionReceipts(ledger, controller)
    record(ledger, "host_action", {"value": 7}, error=error)
    receipts.observe_result("executed")
    step = controller.export_state()["items"][0]["obligations"][0]
    assert step["status"] == ("ready" if error else "resolved")
    assert controller.export_state()["items"][0]["status"] != "completed"


def test_content_error_cannot_create_success_receipt():
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage

    c, ledger = setup_plan(outcome())
    r = ExecutionReceipts(ledger, c)
    ledger.observe_assistant(
        AssistantMessage.text(
            "",
            tool_calls=[
                ToolCall(
                    id="bad",
                    name="unlock_discoverable_agent_tool",
                    arguments={"agent_tool_name": "bad"},
                )
            ],
        )
    )
    ledger.observe_input(
        ToolMessage(role="tool", id="bad", content="Error: Unknown agent tool", error=False)
    )
    r.observe_result("bad")
    assert r.receipts[0]["execution_status"] == "failed"


def test_search_repair_examples_match_schema():
    v = Draft202012Validator(BankingKnowledgeCorpus.search_tool_schema(None)["input_schema"])
    assert v.is_valid({"queries": [{"query": "q", "mode": "lexical"}]})
    assert v.is_valid({"query": "q", "limit": 4})
    assert v.is_valid({"queries": [{"query": "q", "limit": 4}]})
    assert not v.is_valid({"queries": [{"query": "q", "limit": 0}]})


def test_simplified_profile_refuses_packets():
    args = _parser().parse_args(
        [
            "--candidate-profile",
            "simplified-v3",
            "--runtime",
            "native",
            "--planning-mode",
            "outcomes",
            "--controller-mode",
            "advisory",
            "--advice-mode",
            "lean",
            "--decision-packets",
        ]
    )
    with pytest.raises(ValueError, match="decision_packets"):
        _run(args)


def test_actual_runtime_has_receipts_without_packet_gate(tmp_path):
    agent, _, captured = assembly(
        tmp_path,
        [text("Which account?")],
        planning_mode="outcomes",
        advice_mode="lean",
        controller_mode="advisory",
    )
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Please close my account"), state)
        assert captured["state"].packet_gate is None
        assert captured["state"].execution_receipts is not None
    finally:
        state.stop()


def test_actual_runtime_injects_receipt_after_host_result(tmp_path):
    from tau2.data_model.message import ToolMessage

    agent, provider, _ = assembly(
        tmp_path,
        [call("host_action", {"value": 7}, "observed-action"), text("Done.")],
        planning_mode="outcomes",
        advice_mode="lean",
        controller_mode="advisory",
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Please perform the action"), state)
        assert reply.tool_calls[0].name == "host_action"
        agent.generate_next_message(
            ToolMessage(role="tool", id=reply.tool_calls[0].id, content="completed"), state
        )
        context = "\n".join(
            part.text
            for message in provider.requests[-1].messages
            for part in message.content
            if part.type == "text"
        )
        assert "observed_bank_receipts" in context
        assert '"execution_status": "succeeded"' in context
        assert "observed-action" in context
        assert 'full_documents_delivered="false"' in context
    finally:
        state.stop()
