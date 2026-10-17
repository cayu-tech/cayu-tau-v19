import asyncio
import json

from cayu import ToolResultPart
from tau2.data_model.message import UserMessage
from test_incremental_comparison import run
from test_product_navigation import document
from test_research_capture import capture
from test_runtime_banking import assembly, call, text

from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL, BankingKnowledgeCorpus
from cayu_tau3_banking.product_navigation import ProductNavigation


def test_native_runtime_batch_defaults_reach_search_and_bad_shape_has_current_error(tmp_path):
    args = {
        "queries": [{"query": "purchase", "limit": 2}, {"query": "limit"}],
        "mode": "lexical",
        "limit": 4,
    }
    agent, provider, _ = assembly(
        tmp_path,
        [
            call(KNOWLEDGE_SEARCH_TOOL, args, "valid"),
            call(KNOWLEDGE_SEARCH_TOOL, {**args, "cursor": "wrong"}, "invalid"),
            text("Done."),
        ],
        planning_mode="outcomes",
        advice_mode="lean",
        controller_mode="advisory",
    )
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Explain the purchase limit."), state)
        results = {
            part.tool_call_id: part
            for req in provider.requests
            for msg in req.messages
            for part in msg.content
            if isinstance(part, ToolResultPart)
        }
        assert not results["valid"].is_error, results["valid"].content
        payload = json.loads(results["valid"].content)
        assert len(payload["batch"]) == 2
        assert not any(item["is_error"] for item in payload["batch"])
        assert results["invalid"].is_error
        assert "Do not supply limit" not in results["invalid"].content
        assert "schema" in results["invalid"].content
    finally:
        state.stop()


def test_repeated_values_return_exact_choices_then_patch_without_reread(tmp_path):
    body = "Base APY is 3.25% if eligible.\n\nExample: 3.25% excludes promotional bonuses."
    document(tmp_path, content=body)
    nav = ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))
    result = run(nav, capture(nav))
    assert not result["facts"]
    repair = result["cell_errors"][0]["source_repair"]
    assert len(repair["passages"]) == 2
    assert not repair["facts_accepted"]
    for passage in repair["passages"]:
        span = passage["source_span"]
        assert body[span["start"] : span["end"]] == passage["text"]
    patch = capture(nav)
    patch["facts"][0]["source_span"] = repair["passages"][0]["source_span"]
    repaired = run(nav, patch)
    assert len(repaired["facts"]) == 1
    assert not repaired["declared_coverage_complete"]
    restored = ProductNavigation(nav.corpus, nav.snapshot())
    context = asyncio.run(restored.comparison_context(lambda *_: True))
    assert "if eligible" in context


def test_missing_literal_is_not_repaired_into_fact(tmp_path):
    document(tmp_path)
    nav = ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))
    result = run(nav, capture(nav, value="999"))
    assert not result["facts"]
    assert result["cell_errors"][0]["source_repair"]["passages"] == []


def test_repair_payload_has_aggregate_bound_and_does_not_truncate_passages(tmp_path):
    paragraph = "Rate is 3.25%. " + "é" * 1100
    body = "\n\n".join([paragraph + str(i) for i in range(6)])
    document(tmp_path, content=body)
    nav = ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))
    args = capture(nav)
    args["facts"] = [{**args["facts"][0], "field": f"field_{i}"} for i in range(12)]
    result = run(nav, args)
    repairs = [c["source_repair"] for c in result["cell_errors"] if "source_repair" in c]
    assert sum(len(json.dumps(r, ensure_ascii=False).encode()) for r in repairs) <= 12_000
    assert not result["facts"]
    for repair in repairs:
        assert repair["omitted_matching_paragraphs"] > 0
        for passage in repair["passages"]:
            span = passage["source_span"]
            assert body[span["start"] : span["end"]] == passage["text"]
