import json

from tau2.data_model.message import AssistantMessage, UserMessage

from cayu_tau3_banking.customer_sources import customer_source_context
from cayu_tau3_banking.ledger import ConversationLedger


def view(ledger, **kwargs):
    return json.loads(customer_source_context(ledger, **kwargs).split("\n", 1)[1])


def test_exact_per_resource_instructions_survive_other_resource_corrections():
    ledger = ConversationLedger(interpret_customer_intent=False)
    original = "A: partial refund $69.99. B: refund never processed, partial amount unknown."
    correction = "For B only: request a full refund."
    ledger.observe_input(UserMessage.text(original))
    ledger.observe_assistant(AssistantMessage.text("All inputs complete."))
    ledger.observe_input(UserMessage.text(correction))
    result = view(ledger)
    assert [x["text"] for x in result["messages"]] == [original, correction]
    assert [x["source"] for x in result["messages"]] == ["customer-message:1", "customer-message:2"]
    assert result["complete_history"]


def test_byte_bound_omits_whole_messages_explicitly_not_partial_instructions():
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("é" * 100))
    ledger.observe_input(UserMessage.text("Cancel that request."))
    result = view(ledger, max_bytes=100)
    assert result["source_bytes"] <= 100
    assert [x["text"] for x in result["messages"]] == ["Cancel that request."]
    assert result["omitted_retained_message_ids"] == ["customer-message:1"]
    assert not result["complete_history"]


def test_older_retention_loss_is_disclosed():
    ledger = ConversationLedger()
    for i in range(20):
        ledger.observe_input(UserMessage.text(str(i)))
    result = view(ledger)
    assert result["older_messages_outside_retention"] == 4
    assert not result["complete_history"]


def test_runtime_reprojects_original_customer_sources_after_lossy_compaction(tmp_path):
    from cayu import ScriptedModelProvider
    from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
    from test_runtime_banking import assembly, text
    from test_runtime_compaction import message_texts

    summarizer = ScriptedModelProvider(
        [text("Earlier amounts and reasons omitted.") for _ in range(12)]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(provider=summarizer, model="lossy"),
        compact_after_messages=2,
        max_user_turns=1,
    )
    agent, provider, _ = assembly(
        tmp_path,
        [text("Understood.") for _ in range(4)],
        compaction=policy,
        planning_mode="outcomes",
        advice_mode="lean",
        controller_mode="advisory",
    )
    state = agent.get_init_state()
    original = "Transaction A: partial refund $69.99. Transaction B: refund never processed."
    try:
        for request in [
            original,
            "For B only: full refund.",
            "Keep both accounts active.",
            "What remains?",
        ]:
            agent.generate_next_message(UserMessage.text(request), state)
        assert summarizer.requests
        rendered = "\n".join(message_texts(provider.requests[-1]))
        assert original in rendered
        assert "For B only: full refund." in rendered
        assert "Exact customer-source excerpts" in rendered
    finally:
        state.stop()
