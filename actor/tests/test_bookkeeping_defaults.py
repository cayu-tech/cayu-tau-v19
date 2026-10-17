"""Small model-facing updates preserve checked state, without guessing semantics."""

import asyncio
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator
from test_outcome_controller import setup_plan
from test_product_navigation import document, worksheet

from cayu_tau3_banking.controller import (
    SYNC_WORK_PLAN_TOOL,
    UPDATE_WORK_PLAN_TOOL,
    BankingWorkflowController,
)
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.product_navigation import COMPARISON_TOOL, ProductNavigation


def test_minimal_creation_then_field_patch_retains_other_items():
    controller, ledger = setup_plan()
    old = deepcopy(controller.export_state()["items"][0])
    new = {"id": "compare", "goal": "Compare suitable accounts", "completion_mode": "information"}
    schemas = {s["name"]: s["input_schema"] for s in controller.registration_tool_schemas()}
    args = {"items": [new]}
    assert Draft202012Validator(schemas[SYNC_WORK_PLAN_TOOL]).is_valid(args)
    assert Draft202012Validator(schemas[UPDATE_WORK_PLAN_TOOL]).is_valid(args)
    empty = BankingWorkflowController(ledger, planning_mode="outcomes")
    assert not empty.invoke(SYNC_WORK_PLAN_TOOL, args).is_error
    assert not controller.invoke(UPDATE_WORK_PLAN_TOOL, args).is_error
    assert new == args["items"][0] and "status" not in new
    created = controller.export_state()["items"][1]
    assert created["status"] == "pending"
    assert created["outcome"] == "unresolved"
    assert created["depends_on"] == []
    result = controller.invoke(
        UPDATE_WORK_PLAN_TOOL, {"items": [{"id": "compare", "status": "in_progress"}]}
    )
    assert not result.is_error, result.content
    items = controller.export_state()["items"]
    assert items[0] == old
    assert items[1]["goal"] == new["goal"]
    assert items[1]["status"] == "in_progress"


@pytest.mark.parametrize(
    "change",
    [
        {"status": None},
        {"outcome": None},
        {"depends_on": None},
        {"status": "completed"},
        {"status": "completed", "outcome": "answered"},
        {"completion_mode": "invented"},
    ],
)
def test_creation_defaults_never_repair_invalid_values_or_invent_completion(change):
    controller, _ = setup_plan()
    before = controller.export_state()
    args = {
        "items": [{"id": "new", "goal": "Research", "completion_mode": "information", **change}]
    }
    assert controller.invoke(UPDATE_WORK_PLAN_TOOL, args).is_error
    assert controller.export_state() == before


def navigation(tmp_path):
    document(tmp_path)
    return ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))


def test_named_worksheet_patch_needs_only_changed_facts_and_revalidates(tmp_path):
    nav = navigation(tmp_path)
    original = {**worksheet(), "worksheet_id": "comparison"}
    first = asyncio.run(nav.check(original, lambda *_: True))
    restored = ProductNavigation(nav.corpus, nav.snapshot())
    patch = {"worksheet_id": "comparison", "facts": []}
    assert restored.validators[COMPARISON_TOOL].is_valid(patch)
    result = asyncio.run(restored.check(patch, lambda *_: True))
    assert result["calculation_results"] == first["calculation_results"]
    assert "reuse_checked_facts" not in patch
    bad_fact = {**worksheet()["facts"][0], "quote": "Invented evidence"}
    before = restored.snapshot()
    with pytest.raises(ValueError, match="literally"):
        asyncio.run(restored.check({**patch, "facts": [bad_fact]}, lambda *_: True))
    assert restored.snapshot() == before
    with pytest.raises(ValueError, match="Read the exact source revision"):
        asyncio.run(restored.check(patch, lambda *_: False))
    assert restored.snapshot() == before


def test_partial_worksheet_cannot_create_unknown_sheet_or_disable_reuse(tmp_path):
    nav = navigation(tmp_path)
    patch = {"worksheet_id": "unknown", "facts": []}
    with pytest.raises(ValueError, match="No retained worksheet"):
        asyncio.run(nav.check(patch, lambda *_: True))
    assert not nav.validators[COMPARISON_TOOL].is_valid({**patch, "reuse_checked_facts": False})


def test_full_worksheet_still_replaces_unless_explicitly_reused(tmp_path):
    nav = navigation(tmp_path)
    args = {**worksheet(), "worksheet_id": "comparison"}
    asyncio.run(nav.check(args, lambda *_: True))
    replacement = {
        **args,
        "required_fields": ["base_apy"],
        "facts": args["facts"][:1],
        "calculations": [],
    }
    with pytest.raises(ValueError, match="cannot remove"):
        asyncio.run(nav.check({**replacement, "reuse_checked_facts": True}, lambda *_: True))
    result = asyncio.run(nav.check(replacement, lambda *_: True))
    assert result["declared_cells_complete"]


def test_persisted_worksheet_must_be_complete_not_an_unresolved_patch(tmp_path):
    nav = navigation(tmp_path)
    asyncio.run(nav.check({**worksheet(), "worksheet_id": "comparison"}, lambda *_: True))
    snapshot = nav.snapshot()
    snapshot["named_worksheets"]["comparison"]["arguments"] = {
        "worksheet_id": "comparison",
        "facts": [],
    }
    with pytest.raises(ValueError, match="Invalid retained comparison worksheet"):
        ProductNavigation(nav.corpus, snapshot)
