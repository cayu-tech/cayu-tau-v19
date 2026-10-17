import asyncio
import json

import pytest
from cayu import ScriptedModelProvider
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.environment.tool import Tool
from test_outcome_controller import obligation, outcome, setup_plan
from test_runtime_banking import call, text

from cayu_tau3_banking.active_evidence import (
    active_evidence_context,
    recent_tool_documentation_references,
)
from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.resource_outcome_probe import run_case


@pytest.fixture
def corpus(tmp_path):
    for index in range(6):
        identifier = f"definition_{index}"
        (tmp_path / f"{identifier}.json").write_text(
            json.dumps(
                {
                    "id": identifier,
                    "title": identifier,
                    "content": f"list_owner_records_{4100 + index} returns ALL records, including closed rows.",
                }
            )
        )
    (tmp_path / "second.json").write_text(
        json.dumps(
            {
                "id": "second",
                "title": "Another use",
                "content": "list_owner_records_4100 also supplies ownership checks.",
            }
        )
    )
    return BankingKnowledgeCorpus.from_directory(tmp_path)


def deliver(corpus, controller, identifier):
    entry = asyncio.run(corpus.store.get_entry(identifier, revision=1, access_scope=corpus.scope))
    ref = corpus.delivered_revision_evidence(identifier, 1, delivered_text=entry.text)
    controller.observe_knowledge_evidence((ref,))
    return ref


def observe(ledger, index, *, completed=True, failed=False, wrapper=True):
    name = f"list_owner_records_{4100 + index}"
    ledger.observe_assistant(
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id=f"read-{index}",
                    requestor="assistant",
                    name="call_discoverable_agent_tool" if wrapper else name,
                    arguments={"agent_tool_name": name, "arguments": "{}"} if wrapper else {},
                )
            ],
        )
    )
    if completed:
        ledger.observe_input(
            ToolMessage(
                role="tool",
                id=f"read-{index}",
                content="Failed" if failed else "All records: []",
                error=failed,
            )
        )


def test_retains_previously_delivered_tool_scope_without_plan_citation(corpus):
    controller, ledger = setup_plan()
    reference = deliver(corpus, controller, "definition_0")
    observe(ledger, 0)
    before = controller.export_state()
    assert controller.active_knowledge_references() == ()
    refs = recent_tool_documentation_references(corpus, ledger, controller)
    assert refs == (reference,)
    context = asyncio.run(active_evidence_context(corpus, refs, purpose="tool_contracts"))
    assert "active-tool-documentation" in context and "including closed rows" in context
    assert controller.export_state() == before


@pytest.mark.parametrize(
    "completed,failed,delivered", [(False, False, True), (True, True, True), (True, False, False)]
)
def test_pending_failed_and_undelivered_contracts_are_excluded(
    corpus, completed, failed, delivered
):
    controller, ledger = setup_plan()
    if delivered:
        deliver(corpus, controller, "definition_0")
    observe(ledger, 0, completed=completed, failed=failed)
    assert recent_tool_documentation_references(corpus, ledger, controller) == ()


def test_focused_lookup_contract_does_not_pin_a_workflow_using_the_same_lookup(tmp_path):
    docs = {
        "definition": "list_owner_records_4100 returns all resources, including retired resources.",
        "unrelated_workflow": "To replace a damaged resource, call list_owner_records_4100 and replace_owner_resource_5100. This procedure is not an inventory contract.",
    }
    for identifier, content in docs.items():
        (tmp_path / f"{identifier}.json").write_text(
            json.dumps({"id": identifier, "title": identifier, "content": content})
        )
    corpus = BankingKnowledgeCorpus.from_directory(tmp_path)
    controller, ledger = setup_plan()
    for identifier in docs:
        deliver(corpus, controller, identifier)
    observe(ledger, 0)
    refs = recent_tool_documentation_references(corpus, ledger, controller)
    assert [r.reference_id for r in refs] == ["definition"]
    assert {i for i, _ in corpus.capability_document_locators("list_owner_records_4100")} == set(
        docs
    )


def test_general_contract_remains_a_fallback_when_focused_source_was_not_delivered(tmp_path):
    for identifier, content in {
        "focused": "list_owner_records_4100 returns all records.",
        "general": "Use list_owner_records_4100 before replace_owner_resource_5100.",
    }.items():
        (tmp_path / f"{identifier}.json").write_text(
            json.dumps({"id": identifier, "title": identifier, "content": content})
        )
    corpus = BankingKnowledgeCorpus.from_directory(tmp_path)
    controller, ledger = setup_plan()
    deliver(corpus, controller, "general")
    observe(ledger, 0)
    assert [
        r.reference_id for r in recent_tool_documentation_references(corpus, ledger, controller)
    ] == ["general"]


def test_recency_round_robin_and_document_bound_omissions(corpus):
    controller, ledger = setup_plan()
    for index in (5, 4, 3, 2, 1, 0):
        deliver(corpus, controller, f"definition_{index}")
        observe(ledger, index, wrapper=index != 1)
    deliver(corpus, controller, "second")
    refs = recent_tool_documentation_references(corpus, ledger, controller)
    assert [r.reference_id for r in refs] == [
        "definition_0",
        "definition_1",
        "definition_2",
        "definition_3",
        "second",
    ]
    context = asyncio.run(
        active_evidence_context(
            corpus, refs, max_documents=4, max_bytes=12_000, purpose="tool_contracts"
        )
    )
    payload = json.loads(context.split("<active-tool-documentation>\n")[1].split("\n</")[0])
    assert len(payload["documents"]) == 4
    assert payload["omitted"] == [
        {"entry_id": "second", "entry_revision": "1", "reason": "document_limit"}
    ]


@pytest.mark.parametrize(
    "limits", [{"max_tools": 5}, {"max_tools": 0}, {"documents_per_tool": 3}, {"max_tools": True}]
)
def test_invalid_selection_bounds_rejected(corpus, limits):
    controller, ledger = setup_plan()
    with pytest.raises(ValueError):
        recent_tool_documentation_references(corpus, ledger, controller, **limits)


def list_owner_records_4100(owner_id: str) -> str:
    """Read the owner's registry records; consult the public scope documentation."""
    raise AssertionError("Only synthetic host may execute")


def test_native_context_keeps_scope_after_result_and_customer_handoff(tmp_path):
    class Host:
        def __init__(self):
            self.calls = []

        def execute(self, name, args):
            assert name == "list_owner_records_4100" and args == {"owner_id": "sample"}
            return {"records": []}

    provider = ScriptedModelProvider(
        [
            call(
                SYNC_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            **outcome(completion_mode="information"),
                            "obligations": [obligation(kind="communication")],
                        }
                    ]
                },
            ),
            call("list_owner_records_4100", {"owner_id": "sample"}, "registry-read"),
            text("The registry has no records. What detail do you need?"),
            text("The complete registry has no records, including closed ones."),
        ]
    )
    run_case(
        tmp_path,
        mismatch=False,
        provider_factory=lambda: provider,
        model="scripted",
        host=Host(),
        tools=[Tool(list_owner_records_4100)],
        policy="The customer owns sample. No banking verification applies.",
        customers=("Read my registry records.", "Does that include closed records?"),
        documents=(
            (
                "scope",
                "Registry lookup scope",
                "list_owner_records_4100 returns all registry records including closed ones.",
            ),
        ),
        assessment=lambda host, plans: {},
    )
    for request in provider.requests[-2:]:
        rendered = request.model_dump_json()
        assert "active-tool-documentation" in rendered
        assert "all registry records including closed ones" in rendered
