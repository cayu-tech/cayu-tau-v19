import asyncio
import json

import pytest
from test_incremental_comparison import run
from test_product_navigation import ENTRY, PRODUCT
from test_product_navigation import navigation as navigation
from test_read_batch_capacity import make_session

from cayu_tau3_banking.product_navigation import COMPARISON_TOOL, READ_TOOL, ProductNavigation


def capture(nav, field="base_apy", value="3.25"):
    return {
        "capture": True,
        "facts": [
            {"source_handle": nav.locator(ENTRY)["source_handle"], "field": field, "value": value}
        ],
    }


def test_early_capture_and_patch_survive_restore_without_coverage_claim(navigation):
    result = run(navigation, capture(navigation))
    assert result["facts"][0]["product_id"] == PRODUCT
    assert not result["declared_coverage_complete"]
    restored = ProductNavigation(navigation.corpus, navigation.snapshot())
    result = run(restored, capture(restored, "annual_fee", "20"))
    assert len(result["facts"]) == 2
    context = asyncio.run(restored.comparison_context(lambda *_: True))
    assert "3.25" in context and "Annual fee is $20" in context
    assert not result["declared_coverage_complete"]
    assert not result["authorizes_completion"]


def test_handle_does_not_grant_delivery_or_accept_invented_values(navigation):
    result = run(navigation, capture(navigation), lambda *_: False)
    assert not result["facts"]
    result = run(navigation, capture(navigation, value="999"))
    assert not result["facts"]


def test_bad_handle_and_product_mismatch_leave_saved_state_intact(navigation):
    run(navigation, capture(navigation))
    before = navigation.snapshot()
    args = capture(navigation)
    args["facts"][0]["source_handle"] = "src_invented"
    with pytest.raises(ValueError, match="Unknown source"):
        run(navigation, args)
    args = capture(navigation)
    args["facts"][0]["product_id"] = "wrong/product"
    with pytest.raises(ValueError, match="disagree"):
        run(navigation, args)
    assert navigation.snapshot() == before


def test_read_handle_and_omitted_revision_resolve_exactly(navigation):
    handle = navigation.locator(ENTRY)["source_handle"]
    for ref in [{"source_handle": handle}, {"entry_id": ENTRY}]:
        assert navigation.resolve_read_documents({"documents": [ref]}) == [
            {"entry_id": ENTRY, "revision": 1}
        ]
    assert (
        navigation.resolve_read_documents({"documents": [{"entry_id": ENTRY, "revision": 9}]})[0][
            "revision"
        ]
        == 9
    )
    with pytest.raises(ValueError, match="Unknown source"):
        navigation.resolve_read_documents({"documents": [{"entry_id": "doc_wrong/slashes"}]})


def test_handles_are_bound_to_corpus_not_reusable_after_source_change(tmp_path):
    from test_product_navigation import document

    from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus

    document(tmp_path)
    first = ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))
    handle = first.locator(ENTRY)["source_handle"]
    document(tmp_path, content="Changed policy.")
    second = ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))
    with pytest.raises(ValueError, match="Unknown source"):
        second.resolve_read_documents({"documents": [{"source_handle": handle}]})


def test_runtime_capture_projects_facts_without_transcript(tmp_path):
    session = make_session(tmp_path, count=1)
    session.advisory_workflow = True
    entry = "doc_savings_accounts_nimbus_000"
    read = asyncio.run(session.invoke(READ_TOOL, {"documents": [{"entry_id": entry}]}))
    assert not read.is_error, read.content
    handle = json.loads(read.content)["documents"][0]["source_handle"]
    result = asyncio.run(
        session.invoke(
            COMPARISON_TOOL,
            {"capture": True, "facts": [{"source_handle": handle, "field": "apy", "value": "3"}]},
        )
    )
    assert not result.is_error, result.content
    session.navigation = ProductNavigation(session.corpus, session.navigation.snapshot())
    context = asyncio.run(session.runtime_context())
    assert "Base APY is 3%." in context
    assert "apy" in context


def test_captured_finding_reaches_model_after_lossy_compaction(tmp_path):
    from cayu import ScriptedModelProvider
    from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
    from tau2.data_model.message import UserMessage
    from test_product_navigation import document
    from test_runtime_banking import assembly, call, text
    from test_runtime_compaction import message_texts

    document(tmp_path)
    steps = []

    def configure(corpus):
        handle = ProductNavigation(corpus).locator(ENTRY)["source_handle"]
        steps.extend(
            [
                call(READ_TOOL, {"documents": [{"source_handle": handle}]}, "read"),
                call(
                    COMPARISON_TOOL,
                    {"passage_handles": [x["passage_handle"] for x in ProductNavigation(corpus).document_passages[ENTRY]]},
                    "capture",
                ),
                text("I have saved the relevant rate."),
                text("Understood."),
                text("The saved rate is available."),
            ]
        )

    summarizer = ScriptedModelProvider([text("Earlier product facts omitted.") for _ in range(16)])
    policy = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(provider=summarizer, model="lossy"),
        compact_after_messages=2,
        max_user_turns=1,
    )
    agent, provider, _ = assembly(
        tmp_path,
        steps,
        configure_corpus=configure,
        compaction=policy,
        planning_mode="outcomes",
        advice_mode="lean",
        controller_mode="advisory",
    )
    state = agent.get_init_state()
    try:
        for message in ["Compare this account.", "Keep my account open.", "What was the rate?"]:
            agent.generate_next_message(UserMessage.text(message), state)
        assert summarizer.requests
        rendered = "\n".join(message_texts(provider.requests[-1]))
        assert "Base APY is 3.25%. Annual fee is $20." in rendered
        assert "passage_handle" in rendered
    finally:
        state.stop()
