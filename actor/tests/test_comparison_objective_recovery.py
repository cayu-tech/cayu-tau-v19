"""Evidence survives a wrong summary; this does not test live-model obedience."""

import asyncio

import pytest
from cayu import ScriptedModelProvider
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from tau2.data_model.message import UserMessage
from test_product_navigation import document
from test_runtime_banking import assembly, call, text
from test_runtime_compaction import message_texts

from cayu_tau3_banking.product_navigation import COMPARISON_TOOL, READ_TOOL, ProductNavigation


@pytest.mark.parametrize("memory_mode", [None, "constraints"])
def test_two_alternatives_and_exact_objective_survive_biased_compaction(tmp_path, memory_mode):
    first = "doc_savings_accounts_nimbus_001"
    second = "doc_savings_accounts_aurora_001"
    terms = {
        first: "With the Aster card, the bonus is 0.7%. Annual fee is $18.",
        second: "With the Iris card, the bonus is 0.4%. Annual fee is $0.",
    }
    for entry, content in terms.items():
        document(tmp_path, entry, content)
    steps = []

    def configure(corpus):
        nav = ProductNavigation(corpus)
        steps.extend([
            call(READ_TOOL, {"documents": [{"entry_id": e} for e in terms]}, "read"),
            call(COMPARISON_TOOL, {"passage_handles": [p["passage_handle"] for entry in terms for p in nav.document_passages[entry]]}, "save"),
            text("I have the inputs for both alternatives."),
            text("Understood."),
            text("I will compare the retained terms."),
        ])

    wrong_summary = "Customer wants purchase rewards. Only Nimbus matters; other terms omitted."
    summarizer = ScriptedModelProvider([text(wrong_summary) for _ in range(16)])
    policy = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(provider=summarizer, model="biased-summary"),
        compact_after_messages=2, max_user_turns=1,
    )
    agent, provider, _ = assembly(
        tmp_path, steps, configure_corpus=configure, compaction=policy,
        planning_mode="outcomes", advice_mode="lean", controller_mode="advisory",
        task_memory_mode=memory_mode, task_memory_directory=tmp_path / "notes",
    )
    objective = "Compare annual deposit interest minus required fees; exclude purchase rewards."
    state = agent.get_init_state()
    try:
        for request in [objective, "Keep my checking account.", "What is the comparison?"]:
            agent.generate_next_message(UserMessage.text(request), state)
        rendered = message_texts(provider.requests[-1])
        assert summarizer.requests
        assert wrong_summary in "\n".join(rendered)
        projection = next(t for t in reversed(rendered) if t.startswith("<banking_context>"))
        assert objective in projection
        for content in terms.values():
            assert content in projection
        assert projection.index("Exact customer-source excerpts") < projection.index("Retained source-checked comparison")
        assert '"semantic_entailment_checked": false' in projection
        assert '"authorizes_completion": false' in projection
    finally:
        state.stop()


@pytest.mark.parametrize("same_document", [False, True])
def test_conflicting_assertions_survive_patches_restore_and_projection(tmp_path, same_document):
    from cayu_tau3_banking.incremental_comparison import check_incremental
    from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus

    first = "doc_savings_accounts_nimbus_001"
    second = first if same_document else "doc_savings_accounts_nimbus_002"
    terms = "Stated annual yield is 4.2%."
    example = "The worked example uses annual yield of 4.8%."
    document(tmp_path, first, terms + "\n\n" + example if same_document else terms)
    if not same_document:
        document(tmp_path, second, example)
    corpus = BankingKnowledgeCorpus.from_directory(tmp_path)
    nav = ProductNavigation(corpus)
    def delivered(entry, revision):
        return entry in {first, second} and str(revision) == "1"
    for entry, field, value in [(first, "yield_terms", "4.2%"), (second, "yield_example", "4.8%")]:
        result = asyncio.run(check_incremental(nav, {"capture": True, "facts": [{
            "source_handle": nav.locator(entry)["source_handle"], "field": field,
            "value": value, "interpretation": "Conflicting assertions about annual yield; unresolved.",
        }]}, delivered))
        assert not result["authorizes_completion"]
    restored = ProductNavigation(corpus, nav.snapshot())
    # An ordinary correction to the same assertion must not discard its competitor.
    asyncio.run(check_incremental(restored, {"capture": True, "facts": [{
        "source_handle": restored.locator(first)["source_handle"], "field": "yield_terms",
        "value": "4.2%", "interpretation": "Stated rate; the example still conflicts.",
    }]}, delivered))
    facts = restored.snapshot()["worksheet"]["arguments"]["facts"]
    assert {(f["field"], f["value"]) for f in facts} == {
        ("yield_terms", "4.2%"), ("yield_example", "4.8%"),
    }
    context = asyncio.run(restored.comparison_context(delivered))
    assert terms in context and example in context
    assert "yield_terms" in context and "yield_example" in context
    assert '"semantic_entailment_checked": false' in context
    assert '"authorizes_completion": false' in context
