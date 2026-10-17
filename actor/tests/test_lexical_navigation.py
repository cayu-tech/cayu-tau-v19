import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from cayu import Message
from cayu.context.base import ContextBuildResult, ContextPressureOverhead

from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.lexical_navigation import (
    CHECKPOINT_KEY,
    MAX_NAVIGATION_BYTES,
)
from cayu_tau3_banking.lexical_navigation import (
    LexicalNavigationContext as NavigationPolicy,
)


def navigation(base, corpus, query_provider):
    base.reservations = []
    return NavigationPolicy(base, corpus, query_provider, base.reservations.append)


def setup(tmp_path, *, focus=False, suppressed=False):
    for i in range(6):
        (tmp_path / f"doc{i}.json").write_text(
            json.dumps(
                {
                    "id": f"doc{i}",
                    "title": f"Archive protocol {i}",
                    "content": "Archive protocol requires owner approval. " * 20,
                }
            )
        )
    corpus = BankingKnowledgeCorpus.from_directory(tmp_path)
    original = ContextBuildResult(
        messages=[Message.text("user", "Unchanged history")],
        checkpoint={
            "context_compaction": {
                "summary": "Preserved summary",
                "compacted_transcript_cursor": 5,
            },
            "automatic_recall": {
                "projection": {
                    "focus": {
                        "items": [
                            {
                                "text_complete": True,
                                "locator_json": json.dumps(
                                    {"entry_id": "doc0", "entry_revision": 1}
                                ),
                            }
                        ]
                        if focus
                        else []
                    }
                },
                "delta_state": {"original_projection_suppressed": suppressed},
            },
        },
    )

    class Base:
        async def build_with_checkpoint(self, request, *, checkpoint):
            self.request = request
            return original

    base = Base()
    request = SimpleNamespace(
        knowledge_store=corpus.store,
        knowledge_access_scope=corpus.scope,
        model_step_id="model1",
        interaction_id="turn1",
        pressure_overhead=ContextPressureOverhead(),
        count_input_tokens=None,
    )
    request.model_copy = lambda *, update: SimpleNamespace(**{**vars(request), **update})
    return corpus, base, request, original


def test_bounded_projection_checkpoint_and_no_full_read_evidence(tmp_path):
    corpus, base, request, original = setup(tmp_path)
    before = copy.deepcopy(original.model_dump())
    policy = navigation(base, corpus, lambda: "Archive protocol")
    result = asyncio.run(policy.build_with_checkpoint(request, checkpoint=None))
    assert original.model_dump() == before
    assert result.messages[:-1] == original.messages
    text = result.messages[-1].content[0].text
    assert len(text.encode()) <= MAX_NAVIGATION_BYTES
    docs = json.loads(text)["documents"]
    assert len(docs) == 4 and all(d["is_read_evidence"] is False for d in docs)
    assert all(len(d["preview"].encode()) <= 240 for d in docs)
    assert result.checkpoint["automatic_recall"] == original.checkpoint["automatic_recall"]
    assert result.checkpoint["context_compaction"] == original.checkpoint["context_compaction"]
    placement = result.checkpoint[CHECKPOINT_KEY]
    assert placement["status"] == "composed_not_acknowledged"
    assert placement["model_step_id"] == "model1"
    assert base.reservations[0] > 0 and base.reservations[-1] == 0
    assert base.request.pressure_overhead.structured_output_instruction is None
    assert request.pressure_overhead.structured_output_instruction is None


@pytest.mark.parametrize("suppressed", [False, True])
def test_dedup_only_current_complete_focus(tmp_path, suppressed):
    corpus, base, request, _ = setup(tmp_path, focus=True, suppressed=suppressed)
    result = asyncio.run(
        navigation(base, corpus, lambda: "Archive").build_with_checkpoint(request, checkpoint=None)
    )
    ids = [d["entry_id"] for d in result.checkpoint[CHECKPOINT_KEY]["items"]]
    assert ("doc0" in ids) is suppressed


def test_repeated_build_uses_cache_without_accumulating_messages(tmp_path, monkeypatch):
    corpus, base, request, _ = setup(tmp_path)
    policy = navigation(base, corpus, lambda: "Archive")

    async def check():
        first = await policy.build_with_checkpoint(request, checkpoint=None)

        async def forbidden(*args, **kwargs):
            raise AssertionError("Unnecessary repeat search")

        monkeypatch.setattr(corpus.store, "search", forbidden)
        second = await policy.build_with_checkpoint(request, checkpoint=first.checkpoint)
        assert len(first.messages) == len(second.messages) == 2
        assert first.checkpoint[CHECKPOINT_KEY] == second.checkpoint[CHECKPOINT_KEY]

    asyncio.run(check())


def test_wrong_store_is_rejected(tmp_path):
    corpus, base, request, _ = setup(tmp_path)
    request.knowledge_store = object()
    with pytest.raises(ValueError, match="different knowledge store"):
        asyncio.run(
            navigation(base, corpus, lambda: "Archive").build_with_checkpoint(
                request, checkpoint=None
            )
        )


def test_existing_offer_is_not_duplicated(tmp_path):
    corpus, base, request, original = setup(tmp_path)
    original.checkpoint["automatic_recall"]["projection"]["offer"] = {
        "items": [{"locator_json": json.dumps({"entry_id": "doc0", "entry_revision": 1})}]
    }
    result = asyncio.run(
        navigation(base, corpus, lambda: "Archive").build_with_checkpoint(request, checkpoint=None)
    )
    assert "doc0" not in [item["entry_id"] for item in result.checkpoint[CHECKPOINT_KEY]["items"]]


def test_no_query_clears_previous_placement_without_changing_history(tmp_path):
    corpus, base, request, original = setup(tmp_path)
    result = asyncio.run(
        navigation(base, corpus, lambda: None).build_with_checkpoint(
            request, checkpoint={CHECKPOINT_KEY: {"items": ["stale"]}}
        )
    )
    assert result.messages == original.messages
    assert result.checkpoint[CHECKPOINT_KEY]["items"] == []
    assert result.checkpoint[CHECKPOINT_KEY]["projection_sha256"] is None


def test_wrong_scope_is_rejected_before_cached_search(tmp_path):
    corpus, base, request, _ = setup(tmp_path)
    request.knowledge_access_scope = None
    with pytest.raises(ValueError, match="access scope"):
        asyncio.run(
            navigation(base, corpus, lambda: "Archive").build_with_checkpoint(
                request, checkpoint=None
            )
        )


def test_unicode_clipping_and_projection_bounds():
    from cayu_tau3_banking.lexical_navigation import _clip, _encode

    assert len(_clip("界" * 1000, 160).encode()) <= 160
    items = [
        {
            "entry_id": "a" * 256,
            "revision": 1,
            "title": _clip("界" * 1000, 160),
            "preview": _clip("界" * 1000, 240),
            "is_read_evidence": False,
        }
        for _ in range(4)
    ]
    assert len(_encode(items).encode()) <= MAX_NAVIGATION_BYTES


def test_reservation_is_released_when_base_context_fails(tmp_path):
    corpus, base, request, _ = setup(tmp_path)

    async def fail(request, *, checkpoint):
        assert base.reservations[-1] > 0
        assert request.pressure_overhead.structured_output_instruction is None
        raise RuntimeError("Context failed")

    base.build_with_checkpoint = fail
    policy = navigation(base, corpus, lambda: "Archive")
    with pytest.raises(RuntimeError, match="Context failed"):
        asyncio.run(policy.build_with_checkpoint(request, checkpoint=None))
    assert base.reservations[-1] == 0


def test_navigation_with_advisory_controller_is_not_customer_or_read_evidence(tmp_path):
    from tau2.data_model.message import UserMessage
    from test_runtime_banking import assembly, text

    def configure(corpus):
        corpus.admission_policy = corpus.admission_policy.model_copy(
            update={"minimum_inject_score": 1.0, "minimum_offer_score": 1.0}
        )

    agent, provider, captured = assembly(
        tmp_path,
        [text("I need to read the procedure before relying on it.")],
        native_recall=True,
        lexical_navigation=True,
        controller_mode="advisory",
        planning_mode="outcomes",
        advice_mode="lean",
        configure_corpus=configure,
    )
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("What is the purchase limit?"), state)
        assert len(provider.requests) == 1
        hints = [
            p.text
            for m in provider.requests[0].messages
            for p in m.content
            if p.type == "text" and "Additional lexical navigation hints" in p.text
        ]
        assert len(hints) == 1 and len(hints[0].encode()) <= MAX_NAVIGATION_BYTES
        runtime = captured["state"]
        assert runtime.ledger.customer_evidence_messages == ("What is the purchase limit?",)
        assert not runtime.controller.has_delivered_knowledge_revision("limits", "1")
        assert runtime.lexical_navigation_reserved_tokens == 0
    finally:
        state.stop()


def test_native_complete_recall_does_not_gain_duplicate_navigation(tmp_path):
    from tau2.data_model.message import UserMessage
    from test_runtime_banking import assembly, call, text, work

    from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL

    agent, provider, _ = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [work()]}, "create"),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [work(done=True)]}, "complete"),
            text("The purchase limit is $4,500."),
        ],
        native_recall=True,
        lexical_navigation=True,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("What is the purchase limit?"), state
        )
        assert reply.content == "The purchase limit is $4,500."
        assert not any(
            "Additional lexical navigation hints" in part.text
            for message in provider.requests[0].messages
            for part in message.content
            if part.type == "text"
        )
    finally:
        state.stop()


def test_native_runtime_composes_navigation_without_new_evidence_authority(tmp_path):
    from tau2.data_model.message import UserMessage
    from test_runtime_banking import assembly, call, text, work

    from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL

    def configure(corpus):
        corpus.admission_policy = corpus.admission_policy.model_copy(
            update={"minimum_inject_score": 1.0, "minimum_offer_score": 1.0}
        )

    agent, provider, _captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [work()]}, "create"),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [work(done=True)]}, "premature"),
            call(
                "cayu_read_product_documents",
                {"documents": [{"entry_id": "limits", "revision": 1}]},
                "read",
            ),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [work(done=True)]}, "complete"),
            text("The purchase limit is $4,500."),
        ],
        native_recall=True,
        lexical_navigation=True,
        configure_corpus=configure,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("What is the purchase limit?"), state
        )
        assert reply.content == "The purchase limit is $4,500."
        assert len(provider.requests) == 5
        assert any(
            "Additional lexical navigation hints" in part.text
            for message in provider.requests[0].messages
            for part in message.content
            if part.type == "text"
        )
        failures = [
            part
            for request in provider.requests
            for message in request.messages
            for part in message.content
            if part.type == "tool_result" and part.tool_call_id == "premature"
        ]
        assert failures and all(part.is_error for part in failures)
    finally:
        state.stop()
