import asyncio

import pytest
from test_product_navigation import PRODUCT, worksheet
from test_product_navigation import navigation as navigation

from cayu_tau3_banking.incremental_comparison import check_incremental
from cayu_tau3_banking.product_navigation import ProductNavigation


def run(nav, args, delivered=lambda *_: True):
    return asyncio.run(check_incremental(nav, args, delivered))


def test_bad_cell_preserves_other_cells_but_never_certifies_coverage(navigation):
    args = worksheet()
    args["facts"][1]["value"] = "999"
    result = run(navigation, args)
    assert result["partial_recovery"] and result["retained_for_compaction"]
    assert [f["field"] for f in result["facts"]] == ["base_apy"]
    assert result["cell_errors"][0]["field"] == "annual_fee"
    assert not result["declared_coverage_complete"]
    assert not result["declared_inputs_complete"]
    assert not result["arithmetic_checked"]
    assert not result["authorizes_completion"]
    restored = ProductNavigation(navigation.corpus, state=navigation.snapshot())
    context = asyncio.run(restored.comparison_context(lambda *_: True))
    assert "3.25" in context and "Base APY" in context
    repair = {"worksheet_id": "default", "facts": [worksheet()["facts"][1]]}
    result = run(restored, repair)
    assert len(result["facts"]) == 2
    assert not result["declared_coverage_complete"]


def test_bad_update_removes_old_value_instead_of_silently_reusing_it(navigation):
    run(navigation, worksheet())
    fact = worksheet()["facts"][0]
    fact["value"] = "999"
    result = run(navigation, {"worksheet_id": "default", "facts": [fact]})
    assert [f["field"] for f in result["facts"]] == ["annual_fee"]
    assert result["missing"] == [{"product_id": PRODUCT, "field": "base_apy"}]


def test_duplicate_cell_is_not_last_writer_wins(navigation):
    args = worksheet()
    args["facts"].append(dict(args["facts"][0]))
    result = run(navigation, args)
    assert [f["field"] for f in result["facts"]] == ["annual_fee"]
    assert result["cell_errors"]


def test_unread_sources_cannot_be_saved_as_valid(navigation):
    result = run(navigation, worksheet(), lambda *_: False)
    assert not result["facts"]
    assert len(result["cell_errors"]) == 2
    assert not result["declared_inputs_complete"]


def test_wrong_product_source_does_not_gain_authority(navigation):
    args = worksheet()
    args["facts"][1]["entry_id"] = "doc_credit_cards_aster_001"
    result = run(navigation, args)
    assert [f["field"] for f in result["facts"]] == ["base_apy"]


def test_bad_coverage_scope_preserves_valid_source_facts(navigation):
    args = worksheet()
    args["product_ids"].append("personal_savings/nimbus_plus")
    args["required_fields"] = []
    args["required_cells"] = [
        {"product_id": PRODUCT, "field": f} for f in ["base_apy", "annual_fee"]
    ]
    result = run(navigation, args)
    assert result["partial_recovery"]
    assert len(result["facts"]) == 2
    assert not result["declared_coverage_complete"]


def test_malformed_request_does_not_change_previous_state(navigation):
    run(navigation, worksheet())
    before = navigation.snapshot()
    args = worksheet()
    args["facts"][0]["quote"] = "x"
    with pytest.raises(ValueError):
        run(navigation, args)
    assert navigation.snapshot() == before


def test_failed_retention_preserves_previous_snapshot(navigation):
    run(navigation, worksheet())
    before = navigation.snapshot()
    navigation.MAX_WORKSHEET_BYTES = 1
    result = run(navigation, worksheet())
    assert not result["retained_for_compaction"]
    assert navigation.snapshot() == before


def test_advisory_tool_path_returns_partial_evidence_and_rehydrates_it(tmp_path, navigation):
    import json

    from test_agent_memory import new_session

    from cayu_tau3_banking.product_navigation import COMPARISON_TOOL, READ_TOOL

    memory = new_session(tmp_path)
    memory.advisory_workflow = True
    read = asyncio.run(
        memory.invoke(
            READ_TOOL,
            {"documents": [{"entry_id": "doc_savings_accounts_nimbus_001", "revision": 1}]},
        )
    )
    assert not read.is_error
    args = worksheet()
    args["facts"][1]["value"] = "999"
    response = asyncio.run(memory.invoke(COMPARISON_TOOL, args))
    assert not response.is_error, response.content
    result = json.loads(response.content)
    assert result["partial_recovery"]
    assert not result["original_request_complete"]
    assert [fact["field"] for fact in result["facts"]] == ["base_apy"]
    assert not any(memory.navigation.comparison_readiness().values())
    assert "3.25" in asyncio.run(memory.runtime_context())


def test_unavailable_revision_does_not_discard_other_valid_cells(navigation):
    args = worksheet()
    args["facts"][1]["revision"] = 999
    result = run(navigation, args)
    assert [fact["field"] for fact in result["facts"]] == ["base_apy"]
    assert result["cell_errors"][0]["field"] == "annual_fee"
    assert not result["original_request_complete"]


def test_omitted_open_questions_retains_facts_with_unknown_coverage(navigation):
    args = worksheet()
    args["coverage"] = {
        "objective": "Compare the documented cost and rate.",
        "alternatives": [
            {"product_ids": [PRODUCT], "status": "compared", "reason": "Sources checked."}
        ],
    }
    result = run(navigation, args)
    assert len(result["facts"]) == 2
    assert result["retained_for_compaction"]
    assert result["coverage"]["open_questions"]
    assert not result["declared_coverage_complete"]
    restored = ProductNavigation(navigation.corpus, state=navigation.snapshot())
    again = run(restored, {"worksheet_id": "default", "facts": []})
    assert not again["declared_coverage_complete"]
    assert again["coverage"]["open_questions"] == result["coverage"]["open_questions"]
    args["coverage"]["open_questions"] = []
    completed = run(restored, args)
    assert completed["declared_coverage_complete"]
    assert not completed["authorizes_completion"]
