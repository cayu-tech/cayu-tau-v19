import asyncio
import json

import pytest
from jsonschema import Draft202012Validator

from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus


@pytest.fixture
def corpus(tmp_path):
    for name, body in {
        "one": "Orchid eligibility and paper fee.",
        "two": "Orchid bonus requires active card.",
        "noise": "bonus bonus bonus credit interest irrelevant.",
        "long": "😀" * 2400 + " Orchid bonus end.",
    }.items():
        (tmp_path / (name + ".json")).write_text(
            json.dumps({"id": name, "title": name, "content": body})
        )
    return BankingKnowledgeCorpus.from_directory(tmp_path)


def search(corpus, **kw):
    content, refs = asyncio.run(corpus.search_with_evidence("bonus", mode="scoped", **kw))
    return json.loads(content), refs


def test_scope_before_rank_and_complete_evidence(corpus):
    payload, refs = search(corpus, scope={"entry_ids": ["one", "two"]})
    assert [d["entry_id"] for d in payload["documents"]] == ["two"]
    assert len(refs) == 1 and payload["documents"][0]["text_complete"]
    assert payload["retrieval_bounds"]["documents_in_scope"] == 2


def test_any_all_and_unknown_scope(corpus):
    args = dict(scope={"entry_ids": ["one", "two"]}, terms=["paper", "bonus"])
    assert len(search(corpus, **args)[0]["documents"]) == 2
    assert search(corpus, **args, match="all")[0]["documents"] == []
    assert search(corpus, scope={"entry_ids": ["one"]}, terms=["paper fee"])[0]["documents"]
    for scope in ({"entry_ids": ["missing"]}, {}, {"category": "unknown"}, {"product_ids": []}):
        with pytest.raises(ValueError):
            search(corpus, scope=scope)


def test_cursor_scope_binding_and_growing_exclusions(corpus):
    args = dict(scope={"entry_ids": ["one", "two"]}, terms=["Orchid"], limit=1)
    first, _ = search(corpus, **args)
    cursor = first["navigation"]["next_cursor"]
    second, _ = search(corpus, **args, cursor=cursor, exclude_entry_ids={"one"})
    assert second["documents"][0]["entry_id"] == "two"
    with pytest.raises(ValueError):
        search(corpus, **{**args, "terms": ["Orchid", "different"]}, cursor=cursor)
    with pytest.raises(ValueError):
        search(corpus, **{**args, "scope": {"entry_ids": ["two"]}}, cursor=cursor)


def test_long_unicode_window_is_not_complete_evidence(corpus):
    payload, refs = search(corpus, scope={"entry_ids": ["long"]}, terms=["bonus"])
    doc = payload["documents"][0]
    full = asyncio.run(corpus.read_document("long", doc["entry_revision"]))
    span = doc["source_window"]
    assert doc["text"] == full["text"][span["start"] : span["end"]]
    assert "bonus" in doc["text"] and not doc["text_complete"]
    assert not refs and payload["navigation_only"] and "evidence_ref" not in doc


def test_schema_exposes_batch_scope(corpus):
    schema = corpus.search_tool_schema()["input_schema"]
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    child = {"query": "bonus", "mode": "scoped", "scope": {"entry_ids": ["two"]}}
    assert validator.is_valid(child) and validator.is_valid({"queries": [child]})
    assert not validator.is_valid({"queries": [child], "scope": child["scope"]})


@pytest.mark.parametrize("mode", ["default", "lexical", "literal"])
def test_normal_search_accepts_scope_and_preserves_evidence_boundary(corpus, mode):
    content, refs = asyncio.run(corpus.search_with_evidence(
        "bonus", mode=mode, scope={"entry_ids": ["two", "long"]},
    ))
    payload = json.loads(content)
    assert {d["entry_id"] for d in payload["documents"]} == {"two", "long"}
    assert {r.reference_id for r in refs} == {"two"}
    assert payload["retrieval_bounds"]["scope_applied_before_ranking"]


def test_scope_cannot_silently_claim_global_semantic_retrieval(corpus):
    for mode in ("semantic", "hybrid", "capability"):
        with pytest.raises(ValueError):
            asyncio.run(corpus.search_with_evidence("bonus", mode=mode, scope={"entry_ids": ["two"]}))


def test_literal_scope_keeps_cursor_query_binding_and_rejects_conflicting_terms(corpus):
    kw = dict(mode="literal", scope={"entry_ids": ["one", "two"]}, limit=1)
    content, _ = asyncio.run(corpus.search_with_evidence("Orchid", **kw))
    cursor = json.loads(content)["navigation"]["next_cursor"]
    content, _ = asyncio.run(corpus.search_with_evidence("Orchid", cursor=cursor, **kw))
    assert json.loads(content)["documents"][0]["entry_id"] == "two"
    for query, extra in (("bonus", {"cursor": cursor}), ("x" * 501, {}),
                         ("bonus", {"terms": ["other"]})):
        with pytest.raises(ValueError):
            asyncio.run(corpus.search_with_evidence(query, **kw, **extra))


def test_batch_normal_scope_admits_only_full_evidence_and_refunds_invalid_scope(corpus):
    from cayu_tau3_banking.admission import BankingToolAdmissionGate
    from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
    from cayu_tau3_banking.controller import BankingWorkflowController
    from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL
    from cayu_tau3_banking.ledger import ConversationLedger

    ledger = ConversationLedger(interpret_customer_intent=False)
    gate = BankingToolAdmissionGate(ledger, authoritative_only=True)
    controller = BankingWorkflowController(ledger, planning_mode="outcomes")
    session = BankingMemoryRuntime(corpus).new_session(
        ledger=ledger, controller=controller, admission_gate=gate,
        message_history=None, ledger_enabled=True, workflow_controller_enabled=True,
    )
    session.advisory_workflow = True
    observed = []
    gate.observe_knowledge_text = observed.append
    result = asyncio.run(session.invoke(KNOWLEDGE_SEARCH_TOOL, {"queries": [
        {"query": "bonus", "scope": {"entry_ids": ["two", "long"]}, "limit": 4},
        {"query": "bonus", "scope": {"entry_ids": ["invented"]}},
    ]}))
    batch = json.loads(result.content)["batch"]
    assert not batch[0]["is_error"] and batch[1]["is_error"]
    assert session._knowledge_searches == 1
    assert session._turn_delivered_entry_ids == {"two"}
    assert len(observed) == 1
    assert [d["entry_id"] for d in json.loads(observed[0])["documents"]] == ["two"]
    assert controller.delivered_knowledge_reference("long", "1") is None
