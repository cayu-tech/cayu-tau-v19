import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.search_paging import document_page, parse_cursor


def page(documents, *, base=None, cursor=None, excluded=frozenset(), limit=2):
    return document_page(
        base or {"query": "example"},
        documents,
        limit=limit,
        exclude_entry_ids=excluded,
        cursor=cursor,
    )


@pytest.fixture
def documents():
    return [{"entry_id": f"doc-{i}", "entry_revision": 1, "text": f"Fact {i}"} for i in range(7)]


def test_offset_is_not_relative_to_changing_exclusions(documents):
    first = page(documents)
    second = page(
        documents, cursor=first["navigation"]["next_cursor"], excluded={"doc-0", "doc-1", "doc-3"}
    )
    assert [d["entry_id"] for d in second["documents"]] == ["doc-2", "doc-4"]
    assert second["navigation"]["already_delivered_in_pool"] == 3
    assert second["navigation"]["already_delivered_skipped_on_page"] == 1
    assert second["navigation"]["remaining_unseen_in_pool"] == 2
    last = page(documents, cursor=second["navigation"]["next_cursor"])
    assert [d["entry_id"] for d in last["documents"]] == ["doc-5", "doc-6"]
    assert last["navigation"]["next_cursor"] is None
    assert last["navigation"]["pool_exhaustion_proves_answer_complete"] is False


def test_empty_and_all_delivered_are_distinguishable(documents):
    empty = page([])
    delivered = page(documents, excluded={d["entry_id"] for d in documents})
    assert empty["returned"] == delivered["returned"] == 0
    assert empty["navigation"]["candidate_documents_in_pool"] == 0
    assert delivered["navigation"]["already_delivered_in_pool"] == 7
    assert not delivered["navigation"]["has_more_in_pool"]


@pytest.mark.parametrize(
    "cursor", ["", True, 12, {}, "v1:x:2", "v1:" + "a" * 64 + ":0", "x" * 1000]
)
def test_malformed_cursor_rejected(cursor):
    with pytest.raises(ValueError, match="cursor"):
        parse_cursor(cursor)


def test_cursor_binds_query_order_content_and_revision(documents):
    cursor = page(documents)["navigation"]["next_cursor"]
    variants = [
        list(reversed(documents)),
        [{**documents[0], "text": "Changed"}, *documents[1:]],
        [{**documents[0], "entry_revision": 2}, *documents[1:]],
    ]
    for changed in variants:
        with pytest.raises(ValueError, match="no longer matches"):
            page(changed, cursor=cursor)
    with pytest.raises(ValueError, match="no longer matches"):
        page(documents, base={"query": "other"}, cursor=cursor)
    with pytest.raises(ValueError):
        page(documents, cursor=cursor.rsplit(":", 1)[0] + ":999999")


@pytest.mark.parametrize("limit", [True, 0, 5, "2"])
def test_page_bounds(documents, limit):
    with pytest.raises(ValueError):
        page(documents, limit=limit)


@pytest.fixture
def corpus(tmp_path):
    for index in range(20):
        (tmp_path / f"doc-{index}.json").write_text(
            json.dumps(
                {
                    "id": f"doc-{index}",
                    "title": f"Delivery guide {index}",
                    "content": f"Delivery package instructions {index}. Call read_package_status_1234.",
                }
            )
        )
    return BankingKnowledgeCorpus.from_directory(tmp_path)


def test_literal_search_pages_entire_corpus_without_ranking_or_evidence(corpus):
    async def exercise():
        corpus.discovery_engine.recall = AsyncMock(
            side_effect=AssertionError("ranking must not run")
        )
        cursor = None
        seen = []
        while True:
            content, refs = await corpus.search_with_evidence(
                "Delivery",
                mode="literal",
                cursor=cursor,
                exclude_entry_ids=set(corpus._entries),
            )
            payload = json.loads(content)
            assert payload["navigation_only"]
            assert refs == ()
            assert payload["retrieval_bounds"]["total_matching_documents"] == 20
            assert payload["retrieval_bounds"]["total_literal_occurrences"] == 40
            assert not payload["retrieval_bounds"]["retrieval_truncated"]
            assert payload["returned"] <= 4
            seen.extend(d["entry_id"] for d in payload["documents"])
            cursor = payload["navigation"]["next_cursor"]
            if cursor is None:
                break
        assert len(seen) == len(set(seen)) == 20
        corpus.discovery_engine.recall.assert_not_awaited()

    asyncio.run(exercise())


def test_literal_cursor_cannot_switch_query_or_mode(corpus):
    async def exercise():
        text, _ = await corpus.search_with_evidence("Delivery", mode="literal")
        cursor = json.loads(text)["navigation"]["next_cursor"]
        with pytest.raises(ValueError, match="cursor"):
            await corpus.search_with_evidence("instructions", mode="literal", cursor=cursor)
        with pytest.raises(ValueError, match="cursor"):
            await corpus.search_with_evidence("Delivery", mode="lexical", cursor=cursor)
        empty, refs = await corpus.search_with_evidence("does not exist", mode="literal")
        assert json.loads(empty)["retrieval_bounds"]["total_matching_documents"] == 0
        assert refs == ()

    asyncio.run(exercise())


def test_real_corpus_cache_provenance_limits_and_restart(corpus):
    async def exercise():
        original = corpus.discovery_engine.recall
        corpus.discovery_engine.recall = AsyncMock(wraps=original)
        first_text, refs = await corpus.search_with_evidence("Delivery", limit=2)
        first = json.loads(first_text)
        assert len(refs) == 2
        assert first["retrieval_bounds"]["total_matching_documents"] is None
        assert first["navigation"]["candidate_documents_in_pool"] == 20
        assert first["retrieval_bounds"]["fusion_omitted_identities"] == 0
        assert first["retrieval_bounds"]["omitted_by_result_bytes"] == 0
        cursor = first["navigation"]["next_cursor"]
        second_text, refs2 = await corpus.search_with_evidence(
            "Delivery", limit=2, cursor=cursor, exclude_entry_ids={r.reference_id for r in refs}
        )
        assert corpus.discovery_engine.recall.await_count == 1  # no repeated retrieval/embedding
        assert not ({r.reference_id for r in refs} & {r.reference_id for r in refs2})
        second = json.loads(second_text)
        assert {r.reference_id for r in refs2} == {d["entry_id"] for d in second["documents"]}
        for ref, document in zip(refs2, second["documents"], strict=True):
            assert ref.digest == document["content_sha256"]
        corpus._search_snapshots.clear()  # process restart/eviction
        replay = json.loads(await corpus.search("Delivery", limit=2, cursor=cursor))
        assert replay["documents"] == second["documents"]
        assert corpus.discovery_engine.recall.await_count == 2
        with pytest.raises(ValueError, match="cursor"):
            await corpus.search("Other query", cursor=cursor)
        with pytest.raises(ValueError, match="cursor"):
            await corpus.search("Delivery", mode="capability", cursor=cursor)

    asyncio.run(exercise())


def test_ranked_inventory_retains_all_matches_and_only_reads_selected_page(corpus):
    async def exercise():
        reader = AsyncMock(wraps=corpus.read_document)
        corpus.read_document = reader
        cursor = None
        seen = set()
        while True:
            payload = json.loads(await corpus.search("Delivery", limit=4, cursor=cursor))
            assert payload["retrieval_bounds"]["per_channel_candidate_limit"] == 100
            assert payload["retrieval_bounds"]["corpus_exhaustive"] is False
            for document in payload["documents"]:
                assert document["entry_id"] not in seen
                seen.add(document["entry_id"])
                assert document["channel_matches"]
                assert document["text_complete"]
                channels = [match["channel"] for match in document["channel_matches"]]
                assert len(channels) == len(set(channels))
            assert reader.await_count == len(seen)
            cursor = payload["navigation"]["next_cursor"]
            if cursor is None:
                break
        assert len(seen) == 20
        # Explicit discovery did not widen the automatic recall source budget.
        assert all(source.candidate_limit == 16 for source in corpus.recall_sources)
        assert corpus.admission_policy.max_injected_items == 6

    asyncio.run(exercise())


def test_changed_results_after_eviction_require_restart(corpus):
    async def exercise():
        first = json.loads(await corpus.search("Delivery", limit=1))
        original = await corpus.discovery_engine.recall(corpus._situation("Delivery"))
        corpus._search_snapshots.clear()
        corpus.discovery_engine.recall = AsyncMock(
            return_value=original.model_copy(update={"candidates": ()})
        )
        with pytest.raises(ValueError, match="no longer matches"):
            await corpus.search("Delivery", cursor=first["navigation"]["next_cursor"])

    asyncio.run(exercise())


def test_capability_pages_exhaust_exact_index(corpus):
    async def exercise():
        cursor = None
        seen = set()
        for _ in range(5):
            payload = json.loads(
                await corpus.search("read_package_status_1234", mode="capability", cursor=cursor)
            )
            assert payload["matched_documents"] == 20
            new = {d["entry_id"] for d in payload["documents"]}
            assert not seen & new
            seen |= new
            cursor = payload["navigation"]["next_cursor"]
        assert cursor is None and len(seen) == 20

    asyncio.run(exercise())


def test_snapshot_cache_is_bounded(corpus):
    async def exercise():
        for i in range(12):
            await corpus.search(f"Delivery instructions {i}", limit=1)
        assert len(corpus._search_snapshots) == 8

    asyncio.run(exercise())


def test_agent_pages_share_budget_and_only_deliver_selected_evidence(tmp_path, monkeypatch):
    from test_agent_memory import new_session

    from cayu_tau3_banking import agent_memory
    from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL

    # Exercise budget exhaustion before the bounded result snapshot runs out.
    # Production allowance values are asserted independently in test_agent_memory.
    per_message_limit = 3
    monkeypatch.setattr(
        agent_memory, "_MAX_KNOWLEDGE_SEARCHES_PER_CUSTOMER_MESSAGE", per_message_limit
    )
    for i in range(2 * per_message_limit + 4):
        (tmp_path / f"delivery-{i}.json").write_text(
            json.dumps(
                {
                    "id": f"delivery-{i}",
                    "title": f"Delivery instruction {i}",
                    "content": f"Delivery procedure step {i}.",
                }
            )
        )
    session = new_session(tmp_path)

    async def exercise():
        cursor = None
        seen = set()
        for index in range(per_message_limit):
            result = await session.invoke(
                KNOWLEDGE_SEARCH_TOOL,
                {"query": "Delivery", "limit": 1, **({"cursor": cursor} if cursor else {})},
            )
            assert not result.is_error
            payload = json.loads(result.content)
            ids = {d["entry_id"] for d in payload["documents"]}
            assert not seen & ids
            seen |= ids
            cursor = payload["navigation"]["next_cursor"]
            assert (
                payload["search_budget"]["remaining_customer_message"]
                == per_message_limit - 1 - index
            )
        assert cursor is not None
        assert not payload["search_budget"]["can_search_again"]
        assert session._turn_delivered_entry_ids == seen
        blocked = await session.invoke(
            KNOWLEDGE_SEARCH_TOOL, {"query": "Delivery", "cursor": cursor}
        )
        assert blocked.is_error and "do not guess" in blocked.content
        assert session.export_state()["knowledge_searches"] == per_message_limit

    asyncio.run(exercise())
