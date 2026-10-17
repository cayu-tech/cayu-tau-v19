import asyncio
import json

import pytest
from tau2.data_model.message import UserMessage

from cayu_tau3_banking import product_navigation as navigation
from cayu_tau3_banking.admission import BankingToolAdmissionGate
from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.controller import BankingWorkflowController
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.ledger import ConversationLedger


def make_session(tmp_path, count=8, content="Base APY is 3%."):
    for index in range(count):
        (tmp_path / f"doc_savings_accounts_nimbus_{index:03d}.json").write_text(
            json.dumps(
                {
                    "id": f"doc_savings_accounts_nimbus_{index:03d}",
                    "title": f"Nimbus terms {index}",
                    "content": content,
                }
            )
        )
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Compare the accounts."))
    return BankingMemoryRuntime(BankingKnowledgeCorpus.from_directory(tmp_path)).new_session(
        ledger=ledger,
        controller=BankingWorkflowController(ledger),
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=False,
    )


def request(count=8):
    return {
        "documents": [
            {"entry_id": f"doc_savings_accounts_nimbus_{index:03d}", "revision": 1}
            for index in range(count)
        ]
    }


def test_eight_document_candidate_delivers_exact_revisions_in_one_call(tmp_path, monkeypatch):
    monkeypatch.setattr(navigation, "MAX_READ_DOCUMENTS", 8)
    session = make_session(tmp_path)
    result = asyncio.run(session.invoke(navigation.READ_TOOL, request()))
    assert not result.is_error
    payload = json.loads(result.content)
    assert len(payload["documents"]) == 8
    assert len(result.content.encode()) <= navigation.MAX_READ_RESULT_BYTES
    assert all(d["text_complete"] and d["text"] == "Base APY is 3%." for d in payload["documents"])
    assert all(
        session.controller.has_delivered_knowledge_revision(d["entry_id"], "1")
        for d in payload["documents"]
    )
    assert payload["navigation_budget"]["lanes"][navigation.READ_TOOL]["turn"] == 1


@pytest.mark.parametrize("content", ["x" * 10_000, "é" * 5_000], ids=["ascii", "utf8"])
def test_oversized_batch_is_atomic_and_counts_utf8_bytes(tmp_path, monkeypatch, content):
    monkeypatch.setattr(navigation, "MAX_READ_DOCUMENTS", 8)
    session = make_session(tmp_path, content=content)
    result = asyncio.run(session.invoke(navigation.READ_TOOL, request()))
    assert result.is_error and "Split the batch" in result.content
    assert not session._turn_delivered_entry_ids
    assert not session.navigation.state["documents"]
    assert not session.controller.has_delivered_knowledge_revision(
        request()["documents"][0]["entry_id"], "1"
    )
    # A bounded repair delivers complete sources, never a silently truncated batch.
    repaired = asyncio.run(session.invoke(navigation.READ_TOOL, request(4)))
    assert not repaired.is_error
    assert len(json.loads(repaired.content)["documents"]) == 4


def test_invalid_late_revision_does_not_publish_earlier_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(navigation, "MAX_READ_DOCUMENTS", 8)
    session = make_session(tmp_path)
    arguments = request()
    arguments["documents"][-1]["revision"] = 2
    result = asyncio.run(session.invoke(navigation.READ_TOOL, arguments))
    assert result.is_error
    assert not session._turn_delivered_entry_ids
    assert not session.navigation.state["documents"]


def test_complete_product_packet_preserves_every_source_and_revision(tmp_path):
    session = make_session(tmp_path, count=13)
    result = asyncio.run(
        session.invoke(navigation.READ_TOOL, {"product_id": "personal_savings/nimbus"})
    )
    assert not result.is_error
    payload = json.loads(result.content)
    assert len(payload["documents"]) == 13
    assert payload["product_scope"] == {
        "product_id": "personal_savings/nimbus",
        "directory_document_count": 13,
        "all_product_documents_included": True,
        "general_policy_coverage_established": False,
        "establishes_customer_eligibility_or_consent": False,
    }
    assert all(d["text_complete"] and d["text"] == "Base APY is 3%." for d in payload["documents"])
    assert all(
        session.controller.has_delivered_knowledge_revision(d["entry_id"], "1")
        for d in payload["documents"]
    )
    assert payload["navigation_budget"]["lanes"][navigation.READ_TOOL]["turn"] == 1


@pytest.mark.parametrize("count,content", [(17, "Rate 3%."), (13, "é" * 5000)])
def test_oversized_product_packet_rejects_without_partial_delivery(tmp_path, count, content):
    session = make_session(tmp_path, count=count, content=content)
    result = asyncio.run(
        session.invoke(navigation.READ_TOOL, {"product_id": "personal_savings/nimbus"})
    )
    assert result.is_error
    assert not session._turn_delivered_entry_ids
    assert not session.navigation.state["documents"]
    assert not session.controller.has_delivered_knowledge_revision(
        request(1)["documents"][0]["entry_id"], "1"
    )
    repaired = asyncio.run(session.invoke(navigation.READ_TOOL, request(4)))
    assert not repaired.is_error


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"product_id": "unknown"},
        {"product_id": "Nimbus"},
        {"product_id": "personal_savings/nimbus", "documents": request(1)["documents"]},
    ],
)
def test_invalid_product_read_scope_never_delivers_evidence(tmp_path, arguments):
    session = make_session(tmp_path)
    result = asyncio.run(session.invoke(navigation.READ_TOOL, arguments))
    assert result.is_error
    assert not session._turn_delivered_entry_ids


def test_product_packet_locators_cannot_mutate_the_directory(tmp_path):
    session = make_session(tmp_path, count=2)
    resolved = session.navigation.resolve_read_documents({"product_id": "personal_savings/nimbus"})
    assert resolved == request(2)["documents"]
    resolved[0]["revision"] = 99
    assert (
        session.navigation.resolve_read_documents({"product_id": "personal_savings/nimbus"})
        == request(2)["documents"]
    )


@pytest.mark.parametrize("limit", [4, 8, 16])
def test_native_runtime_exposes_matching_read_schema(tmp_path, monkeypatch, limit):
    from cayu import ScriptedModelProvider
    from test_runtime_banking import call, text, work

    from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
    from cayu_tau3_banking.navigation_probe import run_case

    monkeypatch.setattr(navigation, "MAX_READ_DOCUMENTS", limit)
    pending, completed = work(), work(done=True, reference=request()["documents"][0]["entry_id"])
    for item in (pending, completed):
        item.pop("next_operation")
    count = max(8, limit)
    requested = request(count)["documents"]
    steps = [call(SYNC_WORK_PLAN_TOOL, {"items": [pending]}, "plan")]
    steps.extend(
        call(navigation.READ_TOOL, {"documents": requested[start : start + limit]}, f"read-{start}")
        for start in range(0, count, limit)
    )
    steps.extend(
        [call(UPDATE_WORK_PLAN_TOOL, {"items": [completed]}, "done"), text("The APY is 3%.")]
    )
    provider = ScriptedModelProvider(steps)
    result = run_case(
        tmp_path,
        {"id": "schema", "customer": "What is the APY? Advice only."},
        documents=tuple((d["entry_id"], "Nimbus terms", "The APY is 3%.") for d in requested),
        provider_factory=lambda: provider,
        model="scripted",
    )
    assert result["reply"] == "The APY is 3%."
    assert len(provider.requests) == 3 + count // limit
    for tool in provider.requests[1].tools:
        if tool["name"] == navigation.READ_TOOL:
            assert tool["input_schema"]["properties"]["documents"]["maxItems"] == limit
            break
    else:
        pytest.fail("Native read tool was not exposed")


def test_native_runtime_delivers_complete_product_packet(tmp_path):
    from cayu import ScriptedModelProvider, ToolResultPart
    from test_runtime_banking import call, text, work

    from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
    from cayu_tau3_banking.navigation_probe import run_case

    requested = request(13)["documents"]
    pending = work()
    completed = work(done=True, reference=requested[-1]["entry_id"])
    for item in (pending, completed):
        item.pop("next_operation")
    provider = ScriptedModelProvider(
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [pending]}, "plan"),
            call(navigation.READ_TOOL, {"product_id": "personal_savings/nimbus"}, "packet"),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [completed]}, "done"),
            text("All thirteen public documents were delivered."),
        ]
    )
    result = run_case(
        tmp_path,
        {"id": "packet", "customer": "Check all Nimbus terms. Advice only."},
        documents=tuple(
            (d["entry_id"], f"Nimbus terms {index}", f"Source marker {index}.")
            for index, d in enumerate(requested)
        ),
        provider_factory=lambda: provider,
        model="scripted",
    )
    assert result["reply"] == "All thirteen public documents were delivered."
    assert len(provider.requests) == 4
    results = [
        part
        for message in provider.requests[2].messages
        for part in message.content
        if isinstance(part, ToolResultPart) and part.tool_call_id == "packet"
    ]
    assert len(results) == 1 and not results[0].is_error
    packet = json.loads(results[0].content)
    assert [d["text"] for d in packet["documents"]] == [
        f"Source marker {index}." for index in range(13)
    ]


def test_sixteen_document_default_is_complete_and_seventeenth_is_rejected(tmp_path):
    session = make_session(tmp_path, count=17)
    rejected = asyncio.run(session.invoke(navigation.READ_TOOL, request(17)))
    assert rejected.is_error
    assert not session._turn_delivered_entry_ids
    result = asyncio.run(session.invoke(navigation.READ_TOOL, request(16)))
    assert not result.is_error, result.content
    docs = json.loads(result.content)["documents"]
    assert len(docs) == 16
    assert all(d["text_complete"] for d in docs)
    assert all(
        session.controller.has_delivered_knowledge_revision(d["entry_id"], "1") for d in docs
    )
    assert not session.controller.has_delivered_knowledge_revision(
        request(17)["documents"][-1]["entry_id"], "1"
    )
