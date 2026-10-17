import asyncio

import pytest
from cayu import ToolResultPart
from tau2.data_model.message import UserMessage
from test_incremental_comparison import run
from test_product_navigation import ENTRY, document
from test_product_navigation import navigation as navigation
from test_research_capture import capture
from test_runtime_banking import assembly, call, text
from test_task_memory import constraint_args, memory

from cayu_tau3_banking.product_navigation import READ_TOOL, ProductNavigation
from cayu_tau3_banking.task_memory import UPDATE_CONSTRAINTS


def test_mixed_unknown_handle_does_not_discard_independent_valid_fact(navigation):
    args = capture(navigation)
    args["facts"].append({"source_handle": "src_invented", "field": "other", "value": "999"})
    result = run(navigation, args)
    assert len(result["facts"]) == 1
    assert result["cell_errors"][0]["source_handle"] == "src_invented"
    assert not result["original_request_complete"]
    assert not result["declared_coverage_complete"]
    restored = ProductNavigation(navigation.corpus, navigation.snapshot())
    assert "3.25" in asyncio.run(restored.comparison_context(lambda *_: True))
    unread = run(navigation, args, delivered=lambda *_: False)
    assert not unread["facts"]


def test_alias_cannot_grant_delivery_or_fake_distinct_contradiction_sources(tmp_path, navigation):
    notes = memory(tmp_path, "constraints")
    notes.knowledge_reference_resolver = navigation.resolve_knowledge_reference
    handle = navigation.locator(ENTRY)["source_handle"]
    ref = {"kind": "knowledge_revision", "reference_id": handle}
    row = {
        "id": "rate",
        "rule": "The rate is unresolved.",
        "relation": "other",
        "affects": ["compare"],
        "state": "active",
        "sources": [ref],
    }
    with pytest.raises(ValueError, match="not been observed"):
        notes.invoke(UPDATE_CONSTRAINTS, constraint_args(notes, constraints=[row]))
    with pytest.raises(ValueError, match="disagree"):
        navigation.resolve_knowledge_reference({**ref, "version": "2"})
    row.update(
        state="contradicted",
        sources=[ref, {"kind": "knowledge_revision", "reference_id": ENTRY, "version": "1"}],
    )
    with pytest.raises(ValueError, match="two distinct"):
        notes.invoke(UPDATE_CONSTRAINTS, constraint_args(notes, constraints=[row]))
    assert notes.revision == 0


def test_native_constraint_tool_accepts_delivered_handle_and_stores_canonical_reference(tmp_path):
    document(tmp_path)
    steps = []

    def configure(corpus):
        handle = ProductNavigation(corpus).locator(ENTRY)["source_handle"]
        steps.extend(
            [
                call(READ_TOOL, {"documents": [{"source_handle": handle}]}, "read"),
                call(
                    UPDATE_CONSTRAINTS,
                    {
                        "expected_revision": 0,
                        "customer_message_id": "customer-message:1",
                        "ordering_decision": "Compare before making changes.",
                        "constraints": [
                            {
                                "id": "rate",
                                "rule": "Published rate is 3.25%; eligibility unresolved.",
                                "relation": "other",
                                "affects": ["compare"],
                                "state": "active",
                                "sources": [
                                    {
                                        "kind": "knowledge_revision",
                                        "reference_id": handle,
                                        "version": "1",
                                    }
                                ],
                            }
                        ],
                    },
                    "notes",
                ),
                text("Saved."),
            ]
        )

    agent, provider, captured = assembly(
        tmp_path,
        steps,
        configure_corpus=configure,
        planning_mode="outcomes",
        advice_mode="lean",
        controller_mode="advisory",
        task_memory_mode="constraints",
        task_memory_directory=tmp_path / "notes",
    )
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Compare the account."), state)
        result = [
            part
            for req in provider.requests
            for msg in req.messages
            for part in msg.content
            if isinstance(part, ToolResultPart) and part.tool_call_id == "notes"
        ]
        assert result and not result[-1].is_error, result
        assert captured["state"].task_memory.data["constraints"]["rate"]["sources"] == [
            {"kind": "knowledge_revision", "reference_id": ENTRY, "version": "1"}
        ]
    finally:
        state.stop()
