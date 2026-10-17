import json

import pytest
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from test_outcome_controller import setup_plan

from cayu_tau3_banking.decision_notes import DECISION_NOTE_TOOL, DecisionFacts
from cayu_tau3_banking.saved_evidence import SAVED_EVIDENCE_TOOL, SavedEvidence


def setup():
    controller, ledger = setup_plan()
    facts = DecisionFacts(ledger, controller)
    return SavedEvidence(ledger, facts), ledger, facts


def result(memory, ledger, name, args, content="Done", call_id="effect", error=False):
    ledger.observe_assistant(
        AssistantMessage.text(
            "",
            tool_calls=[
                ToolCall(
                    id=call_id,
                    name="call_discoverable_agent_tool",
                    arguments={"agent_tool_name": name, "arguments": json.dumps(args)},
                )
            ],
        )
    )
    ledger.observe_input(ToolMessage(role="tool", id=call_id, content=content, error=error))
    memory.observe_result(call_id, content)


def history(account="A", amount="-125.50"):
    return (
        "Found 1 record(s) in 'bank_account_transaction_history':\n\n"
        f"1. Record ID: tx-{account}\n   transaction_id: tx-{account}\n   account_id: {account}\n"
        f"   date: 01/03/2025\n   description: ATM WITHDRAWAL\n   amount: {amount}\n"
        "   type: atm_withdrawal\n   status: posted\n"
    )


def fact(key="amount"):
    return dict(id=key, claim="Unresolved classification", state="unknown", sources=[])


def test_rows_are_exact_cited_paginated_and_not_interpreted():
    memory, ledger, _ = setup()
    result(
        memory,
        ledger,
        "get_bank_account_transactions_9173",
        {"account_id": "A"},
        history(),
        "read-A",
    )
    r = memory.read({"kind": "transactions", "account_id": "A"})
    assert r["source"]["reference_id"] == "read-A"
    assert r["rows"][0]["amount"] == "-125.50"
    assert r["rows"][0]["description"] == "ATM WITHDRAWAL"
    assert "fee_owner" not in r["rows"][0]
    assert r["next_offset"] is None
    assert (
        memory.read({"kind": "transactions", "account_id": "A", "ids": ["missing"]})["rows"] == []
    )
    r["rows"][0]["amount"] = "0"
    assert (
        memory.read({"kind": "transactions", "account_id": "A"})["rows"][0]["amount"] == "-125.50"
    )


def test_scoped_mutation_invalidates_only_matching_snapshot_and_refresh_restores():
    memory, ledger, _ = setup()
    for account in ["A", "B"]:
        result(
            memory,
            ledger,
            "get_bank_account_transactions_9173",
            {"account_id": account},
            history(account),
            "read-" + account,
        )
    result(memory, ledger, "apply_checking_account_credit_5829", {"account_id": "A", "amount": 10})
    assert (
        memory.read({"kind": "transactions", "account_id": "A"})["freshness"] == "refresh_required"
    )
    assert (
        memory.read({"kind": "transactions", "account_id": "B"})["freshness"]
        == "latest_observed_snapshot"
    )
    result(
        memory,
        ledger,
        "get_bank_account_transactions_9173",
        {"account_id": "A"},
        history("A", "-100.25"),
        "refresh-A",
    )
    r = memory.read({"kind": "transactions", "account_id": "A"})
    assert (
        r["freshness"] == "latest_observed_snapshot" and r["source"]["reference_id"] == "refresh-A"
    )


def test_malformed_new_lookup_never_relabels_old_rows_as_new_evidence():
    memory, ledger, _ = setup()
    result(
        memory, ledger, "get_bank_account_transactions_9173", {"account_id": "A"}, history(), "old"
    )
    result(
        memory,
        ledger,
        "get_bank_account_transactions_9173",
        {"account_id": "A"},
        "Unrecognized format",
        "new",
    )
    r = memory.read({"kind": "transactions", "account_id": "A"})
    assert r["freshness"] == "refresh_required" and r["source"]["reference_id"] == "old"
    result(memory, ledger, "unknown_mutation", {"user_id": "u"}, call_id="unknown")
    assert "unknown" in memory.read({"kind": "transactions", "account_id": "A"})["invalidated_by"]


def document(text=None):
    return {
        "documents": [
            {
                "entry_id": "policy",
                "entry_revision": 1,
                "title": "Fictional policy",
                "text": text or ("## Fees\n" + "Exact observed policy. " * 500),
                "text_complete": True,
                "evidence_ref": {
                    "kind": "knowledge_revision",
                    "reference_id": "policy",
                    "version": "1",
                },
            }
        ]
    }


def test_compact_repeat_delivery_and_saved_recovery_are_lossless():
    memory, _, _ = setup()
    body = json.dumps(document())
    first = json.loads(memory.project_documents(body))["documents"][0]
    second = json.loads(memory.project_documents(body))["documents"][0]
    assert first["text_complete"] and not second["text_complete"]
    assert second["text"] == ""
    assert second["saved_source"]["sections"][0]["offset"] == 0
    recovered = ""
    offset = 0
    while offset is not None:
        r = memory.read(
            {"kind": "document", "entry_id": "policy", "offset": offset, "characters": 2000}
        )
        recovered += r["text"]
        offset = r["next_offset"]
    assert recovered == first["text"]
    with pytest.raises(ValueError):
        memory.read({"kind": "document", "entry_id": "unseen"})
    changed = json.loads(memory.project_documents(json.dumps(document("## Fees\nChanged source."))))
    assert changed["documents"][0]["text_complete"]


def test_targeted_windows_are_not_replaced_or_saved_as_complete():
    memory, _, _ = setup()
    body = json.dumps(document("A specific passage"))
    assert memory.project_documents(body, targeted=True) == body
    assert not memory.documents


def test_search_deliveries_are_saved_without_promoting_snippets_or_failed_batches():
    memory, _, _ = setup()
    body = document("Complete policy from a search result.")
    memory.observe_search_documents(json.dumps(body))
    assert memory.read({"kind": "document", "entry_id": "policy"})["text"] == body["documents"][0]["text"]
    snippet = document("Only a snippet")
    snippet["documents"][0]["text_complete"] = False
    memory.observe_search_documents(json.dumps(snippet))
    assert memory.read({"kind": "document", "entry_id": "policy"})["text"] == body["documents"][0]["text"]
    missing_flag = document("Completeness unknown")
    del missing_flag["documents"][0]["text_complete"]
    memory.observe_search_documents(json.dumps(missing_flag))
    assert memory.read({"kind": "document", "entry_id": "policy"})["text"] == body["documents"][0]["text"]
    batch_doc = document("A second delivered document")
    batch_doc["documents"][0]["entry_id"] = "batch-policy"
    batch_doc["documents"][0]["evidence_ref"]["reference_id"] = "batch-policy"
    memory.observe_search_documents(json.dumps({"batch": [
        {"is_error": False, "result": batch_doc},
        {"is_error": True, "result": document("Failed delivery")},
    ]}))
    assert memory.read({"kind": "document", "entry_id": "batch-policy"})["text"] == "A second delivered document"
    assert memory.read({"kind": "document", "entry_id": "policy"})["text"] == body["documents"][0]["text"]
    with pytest.raises(ValueError):
        memory.read({"kind": "document", "entry_id": "undelivered"})


@pytest.mark.parametrize("batched", [False, True])
def test_search_document_is_recoverable_after_runtime_compaction(tmp_path, batched):
    from cayu import ScriptedModelProvider
    from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
    from test_runtime_banking import assembly, call, text

    source = "The silver moon account has a recovery limit of exactly 125.50."
    (tmp_path / "search-policy.json").write_text(json.dumps({
        "id": "search-policy", "title": "Silver moon recovery", "content": source,
    }))
    summaries = ScriptedModelProvider([text("Earlier context omitted.") for _ in range(8)])
    query = {"query": "silver moon recovery", "mode": "lexical", "limit": 4}
    agent, _provider, captured = assembly(tmp_path, [
        call("cayu_search_banking_knowledge", {"queries": [query]} if batched else query, "search"),
        text("Found the source."),
        call(SAVED_EVIDENCE_TOOL, {"kind": "document", "entry_id": "search-policy", "revision": 1}, "recover-search"),
        text("Recovered."),
    ], controller_mode="advisory", planning_mode="outcomes", advice_mode="lean", persistent=True,
        compaction=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=summaries, model="summary"), compact_after_messages=2, max_user_turns=1,
        ))
    state = agent.get_init_state()
    try:
        # Keep automatic recall from pre-delivering this fixture: this regression
        # specifically verifies the explicit search-result delivery path.
        agent.generate_next_message(UserMessage.text("Hello."), state)
        assert captured["state"].saved_evidence.read({"kind": "document", "entry_id": "search-policy"})["text"] == source
        reply, _ = agent.generate_next_message(UserMessage.text("Recover it."), state)
        assert "Recovered" in reply.content and summaries.requests
    finally:
        state.stop()


def test_independent_notes_survive_bad_entry_and_rejected_overwrite():
    _, _, facts = setup()
    good = fact()
    bad = {
        **fact("bad"),
        "state": "true",
        "sources": [{"kind": "tool_result", "reference_id": "invented"}],
    }
    receipt = facts.apply_patch({"facts": [good, bad]})
    assert receipt["accepted_ids"] == ["amount"] and receipt["rejected"][0]["id"] == "bad"
    facts.apply_patch({"facts": [{**bad, "id": "amount"}]})
    assert facts.facts["amount"] == good
    receipt = facts.apply_patch({"facts": [fact("duplicate"), fact("duplicate"), fact("valid")]})
    assert receipt["accepted_ids"] == ["valid"] and len(receipt["rejected"]) == 2


def test_notes_beyond_projection_are_recoverable_without_bloating_context():
    memory, _, facts = setup()
    facts.apply_patch({"facts": [fact(str(i)) for i in range(24)]})
    view = facts.context_snapshot()
    assert view["stored_fact_count"] == 24 and len(view["facts"]) == 16
    assert len(json.dumps(view)) < 15000
    assert memory.read({"kind": "notes", "ids": ["0"]})["rows"][0]["id"] == "0"


def test_partial_note_acceptance_and_exact_evidence_survive_runtime_compaction(tmp_path):
    from cayu import ScriptedModelProvider
    from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
    from test_runtime_banking import assembly, call, text

    summaries = ScriptedModelProvider([text("Details deliberately omitted.") for _ in range(20)])
    bad = {**fact("bad"), "state": "invalid"}
    agent, _provider, captured = assembly(
        tmp_path,
        [
            call(DECISION_NOTE_TOOL, {"facts": [fact(), bad]}, "notes"),
            text("Ready."),
            call(SAVED_EVIDENCE_TOOL, {"kind": "transactions", "account_id": "A"}, "recover"),
            call(SAVED_EVIDENCE_TOOL, {"kind": "notes"}, "recover-notes"),
            text("Recovered exact data."),
        ],
        controller_mode="advisory",
        planning_mode="outcomes",
        advice_mode="lean",
        persistent=True,
        compaction=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=summaries, model="summary"),
            compact_after_messages=2,
            max_user_turns=1,
        ),
    )
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Review the account."), state)
        runtime = captured["state"]
        assert (
            "amount" in runtime.decision_facts.facts and "bad" not in runtime.decision_facts.facts
        )
        result(
            runtime.saved_evidence,
            runtime.ledger,
            "get_bank_account_transactions_9173",
            {"account_id": "A"},
            history(),
            "bank-read",
        )
        reply, _ = agent.generate_next_message(UserMessage.text("Continue."), state)
        assert "Recovered" in reply.content and summaries.requests
        assert (
            runtime.saved_evidence.read({"kind": "transactions", "account_id": "A"})["rows"][0][
                "amount"
            ]
            == "-125.50"
        )
        assert runtime.decision_facts.facts["amount"]["state"] == "unknown"
    finally:
        state.stop()


def test_repeated_document_runtime_delivery_is_compact_and_recoverable(tmp_path):
    from test_runtime_banking import assembly, call, text

    source = "## Relevant rule\n" + "Keep the exact observed number 125.50.\n" * 200
    (tmp_path / "memory-policy.json").write_text(
        json.dumps({"id": "memory-policy", "title": "Observed public procedure", "content": source})
    )
    agent, _provider, captured = assembly(
        tmp_path,
        [
            call(
                "cayu_read_product_documents",
                {"documents": [{"entry_id": "memory-policy"}]},
                "first",
            ),
            call(
                "cayu_read_product_documents",
                {"documents": [{"entry_id": "memory-policy"}]},
                "repeat",
            ),
            call(
                SAVED_EVIDENCE_TOOL,
                {"kind": "document", "entry_id": "memory-policy", "characters": 1000},
                "recover",
            ),
            text("Recovered the exact source."),
        ],
        controller_mode="advisory",
        planning_mode="outcomes",
        advice_mode="lean",
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Read the procedure."), state)
        assert "Recovered" in reply.content
        store = captured["state"].saved_evidence
        assert (
            store.read({"kind": "document", "entry_id": "memory-policy", "characters": 1000})[
                "text"
            ]
            == source[:1000]
        )
        assert len(store.documents) == 1
    finally:
        state.stop()


def test_rejected_note_receipt_identifies_field_without_echoing_long_value():
    _, _, facts = setup()
    long_claim = "x" * 500
    receipt = facts.apply_patch({"facts": [
        {**fact("too-long"), "claim": long_claim},
        {"claim": "No ID", "state": "unknown", "sources": []},
    ]})
    assert receipt["status"] == "rejected"
    assert receipt["stored_fact_count"] == 0
    assert receipt["rejected"][0]["errors"][0]["path"] == ["facts", 0, "claim"]
    assert receipt["rejected"][0]["errors"][0]["constraint"] == 360
    assert receipt["rejected"][1]["index"] == 1
    assert "id" in receipt["rejected"][1]["errors"][0]["message"]
    assert long_claim not in json.dumps(receipt)
    receipt = facts.apply_patch({"facts": [fact("good"), {**fact("bad"), "claim": long_claim}]})
    assert receipt["status"] == "partial" and receipt["accepted_ids"] == ["good"]


@pytest.mark.parametrize("partial", [False, True])
def test_note_rejection_runtime_envelope_and_repair(tmp_path, partial):
    from cayu import ToolResultPart
    from test_runtime_banking import assembly, call, text

    entries = [{**fact("bad"), "claim": "x" * 500}]
    if partial:
        entries.append(fact("good"))
    agent, provider, captured = assembly(tmp_path, [
        call(DECISION_NOTE_TOOL, {"facts": entries}, "invalid-notes"),
        call(DECISION_NOTE_TOOL, {"facts": [fact("bad")]}, "repair-note"),
        text("Saved."),
    ], controller_mode="advisory", planning_mode="outcomes", advice_mode="lean")
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Keep notes."), state)
        parts = [p for request in provider.requests for m in request.messages for p in m.content
                 if isinstance(p, ToolResultPart) and p.tool_call_id == "invalid-notes"]
        assert parts and all(p.is_error == (not partial) for p in parts)
        assert captured["state"].decision_facts.facts["bad"] == fact("bad")
        if partial:
            assert captured["state"].decision_facts.facts["good"] == fact("good")
    finally:
        state.stop()


def test_batch_repeat_read_restores_exact_sources_after_compaction(tmp_path):
    from cayu import ScriptedModelProvider, ToolResultPart
    from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
    from test_runtime_banking import assembly, call, text

    sources = {"policy-a": "## Rule A\nAmount 125.50; eligibility unknown.",
               "policy-b": "## Rule B\nFee 10.00 when the balance is below 5000."}
    for key, source in sources.items():
        (tmp_path / (key + ".json")).write_text(json.dumps({"id": key, "title": key, "content": source}))
    docs = {"documents": [{"entry_id": key} for key in sources]}
    summaries = ScriptedModelProvider([text("Sources omitted from this summary.") for _ in range(20)])
    agent, provider, _ = assembly(tmp_path, [
        call("cayu_read_product_documents", docs, "first-batch"), text("Read."),
        call(SAVED_EVIDENCE_TOOL, {"kind": "documents", "documents": [{"entry_id": key, "characters": 2000} for key in sources]}, "repeat-batch"), text("Restored."),
    ], controller_mode="advisory", planning_mode="outcomes", advice_mode="lean", persistent=True,
        compaction=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=summaries, model="summary"),
            compact_after_messages=2, max_user_turns=1,
        ))
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Read both sources."), state)
        agent.generate_next_message(UserMessage.text("Recover both sources."), state)
        assert summaries.requests
        parts = [p for request in provider.requests for m in request.messages for p in m.content
                 if isinstance(p, ToolResultPart) and p.tool_call_id == "repeat-batch"]
        assert parts
        payload = json.loads(parts[-1].content)
        assert payload["status"] == "accepted"
        docs = [item["document"] for item in payload["results"]]
        assert {d["entry_id"]: d["text"] for d in docs} == sources
        assert all(d["text_complete"] and d["source"] for d in docs)
    finally:
        state.stop()


def test_batch_recovery_is_bounded_and_does_not_hide_failed_members():
    memory, _, _ = setup()
    memory.project_documents(json.dumps(document("## Fee\nExactly 12.50.")))
    receipt = memory.read({"kind": "documents", "documents": [
        {"entry_id": "policy", "revision": 1}, {"entry_id": "unseen"},
    ]})
    assert receipt["status"] == "partial"
    assert receipt["results"][0]["document"]["text"] == "## Fee\nExactly 12.50."
    assert receipt["results"][1]["is_error"]
    assert memory.read({"kind": "documents", "documents": [{"entry_id": "policy", "revision": 2}]})["status"] == "rejected"
    with pytest.raises(ValueError, match="16000"):
        memory.read({"kind": "documents", "documents": [{"entry_id": "policy", "characters": 6000}] * 3})
    with pytest.raises(ValueError, match="1-8"):
        memory.read({"kind": "documents", "documents": []})


def test_older_comparison_survives_unrelated_note_churn_and_overflow_is_visible():
    _, _, facts = setup()
    def comparison(i):
        return {**fact(str(i)), "comparison": {"alternative": "Option " + str(i),
            "calculation": "Unknown until conditions resolved", "conditions": "Unresolved",
            "next_missing_fact": "Applicable tier"}}
    facts.apply_patch({"facts": [comparison(0), comparison(1)]})
    facts.apply_patch({"facts": [fact("ordinary-" + str(i)) for i in range(24)]})
    snapshot = facts.context_snapshot()
    assert {"0", "1"} <= {f["id"] for f in snapshot["facts"]}
    facts.apply_patch({"facts": [comparison(i) for i in range(2, 20)]})
    snapshot = facts.context_snapshot()
    assert len(snapshot["facts"]) == 16
    assert {f["id"] for f in snapshot["facts"]} | {f["id"] for f in snapshot["omitted_comparisons"]} == {str(i) for i in range(20)}


def test_comparison_remains_in_actor_context_after_lossy_compaction(tmp_path):
    from cayu import ScriptedModelProvider
    from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
    from test_runtime_banking import assembly, call, text

    comparison = {**fact("contender"), "comparison": {
        "alternative": "Previously observed contender Z",
        "calculation": "Unknown until fee established", "conditions": "Fee unresolved",
        "next_missing_fact": "Applicable fee",
    }}
    notes = [comparison, *[fact("ordinary-" + str(i)) for i in range(24)]]
    summaries = ScriptedModelProvider([text("All details omitted.") for _ in range(20)])
    agent, provider, _ = assembly(tmp_path, [
        call(DECISION_NOTE_TOOL, {"facts": notes}, "save-comparison"),
        text("Ready."), text("Continue with the retained comparison."),
    ], controller_mode="advisory", planning_mode="outcomes", advice_mode="lean",
        persistent=True, compaction=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=summaries, model="summary"),
            compact_after_messages=2, max_user_turns=1,
        ))
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Compare the options."), state)
        agent.generate_next_message(UserMessage.text("Continue."), state)
        assert summaries.requests
        visible = repr(provider.requests[-1].messages)
        assert "Previously observed contender Z" in visible
        assert "Unknown until fee established" in visible
        assert "Applicable fee" in visible
    finally:
        state.stop()
