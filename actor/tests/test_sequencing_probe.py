from copy import deepcopy

import pytest
from cayu import ScriptedModelProvider
from tau2.environment.tool import Tool
from test_runtime_banking import call, text

from cayu_tau3_banking.agent import banking_system_prompt
from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.product_navigation import READ_TOOL
from cayu_tau3_banking.resource_outcome_probe import run_case
from cayu_tau3_banking.sequencing_probe import (
    CASES,
    POLICY,
    MigrationHost,
    assess,
    catalogue,
    create_storage,
    export_archive,
    first_response_metrics,
    information_delivered,
    retire_credential,
)


@pytest.mark.parametrize("missing", [None, "receipt", "preparation", "evidence", "question"])
def test_information_delivery_matches_ready_communication_contract(missing):
    ref = {"kind": "knowledge_revision", "reference_id": "public-price", "version": "1"}
    item = {
        "id": "advice",
        "completion_mode": "information",
        "status": "in_progress",
        "outcome": "unresolved",
        "obligations": [
            {"kind": "knowledge_search", "status": "resolved", "evidence_refs": [deepcopy(ref)]},
            {"kind": "communication", "status": "ready", "evidence_refs": [deepcopy(ref)]},
        ],
    }
    plan = {"customer_response_handoff_sequences": {"advice": 2}}
    if missing == "receipt":
        plan["customer_response_handoff_sequences"] = {}
    elif missing == "preparation":
        item["obligations"][0]["status"] = "ready"
    elif missing == "evidence":
        item["obligations"][1]["evidence_refs"] = []
    elif missing == "question":
        item["obligations"][1]["kind"] = "customer_input"
    assert information_delivered(item, plan) is (missing is None)


def plan(*, missing):
    items = []
    for identifier, dependency in (
        ("choose", None),
        ("create", "choose"),
        ("export", "create"),
        ("retire", "export"),
    ):
        item = {
            "id": identifier,
            "goal": identifier,
            "completion_mode": "information" if identifier == "choose" else "operation",
            "status": "pending",
            "outcome": "unresolved",
            "depends_on": [dependency] if dependency else [],
        }
        if missing and identifier == "choose":
            item["status"] = "in_progress"
            item["obligations"] = [
                {
                    "id": "region",
                    "goal": "Ask for the required data region",
                    "kind": "communication",
                    "status": "ready",
                }
            ]
        items.append(item)
    return items


def completed(identifier, call_id=None):
    return {
        "id": identifier,
        "status": "completed",
        "outcome": "succeeded" if call_id else "answered",
        "evidence": "Verified service result"
        if call_id
        else "Verified public price and requirements",
        "evidence_refs": [{"kind": "tool_result", "reference_id": call_id}]
        if call_id
        else [
            {"kind": "knowledge_revision", "reference_id": "storage_dune_prices", "version": "1"},
            {
                "kind": "knowledge_revision",
                "reference_id": "storage_dune_requirements",
                "version": "1",
            },
        ],
    }


def run(tmp_path, steps, missing):
    provider = ScriptedModelProvider(steps)
    host = MigrationHost()
    result = run_case(
        tmp_path,
        mismatch=False,
        provider_factory=lambda: provider,
        model="scripted",
        host=host,
        policy=POLICY,
        documents=catalogue()[0],
        tools=[Tool(create_storage), Tool(export_archive), Tool(retire_credential)],
        customers=(CASES["missing"][0],) if missing else CASES["complete"],
        assessment=assess,
    )
    return result, provider


def test_catalogue_has_a_unique_winner_and_real_region_capacity_exclusions():
    documents, plans = catalogue()
    assert len(documents) == 35 and len(plans) == 16
    eligible = {
        sku: p for sku, p in plans.items() if p["region"] == "EU" and p["capacity_gb"] >= 2000
    }
    assert min(eligible, key=lambda s: eligible[s]["annual_fee"]) == "store_dune"
    assert sum(p["annual_fee"] == 101 for p in eligible.values()) == 1
    assert len(eligible) == 6


def test_preservation_guidance_keeps_actual_inspections_and_avoids_fixture_rules():
    prompt = banking_system_prompt("Public policy", planning_mode="outcomes")
    assert "a constraint, not a ready action" in prompt
    assert "record that concrete inspection separately" in prompt
    assert "store_dune" not in prompt and "credential K9" not in prompt


def test_retirement_is_not_itself_blocked_but_destroys_requested_migration():
    host = MigrationHost()
    assert host.execute("retire_credential", {"credential_id": "K9"})["success"]
    host.execute("create_storage", {"plan_sku": "store_dune"})
    assert not host.execute(
        "export_archive", {"archive_id": "A7", "credential_id": "K9", "workspace_id": "W8"}
    )["success"]
    assert not assess(host, [])["structural_pass"]


def test_native_missing_input_handoff_does_not_require_future_research(tmp_path):
    result, provider = run(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": plan(missing=True)}, "plan"),
            text("Which data region must your archive stay in: EU or US?"),
        ],
        missing=True,
    )
    assert not result["calls"] and len(provider.requests) == 2
    assert all(i["outcome"] == "unresolved" for i in result["plans"][0]["items"])
    metrics = first_response_metrics(tmp_path)
    assert metrics["first_interaction_model_calls"] == 2
    assert metrics["first_interaction_tools"] == [SYNC_WORK_PLAN_TOOL]


def test_native_waiting_operation_reuses_child_blocker_at_first_handoff(tmp_path):
    items = plan(missing=False)[1:]
    items[0].update(
        status="awaiting_customer",
        depends_on=[],
        obligations=[
            {
                "id": "region",
                "goal": "Obtain the required data region",
                "kind": "communication",
                "status": "ready",
            },
            {
                "id": "selection",
                "goal": "Select the cheapest eligible plan in the permitted region",
                "kind": "knowledge_search",
                "status": "waiting",
                "evidence": "The customer's request omits the required data region.",
                "evidence_refs": [
                    {"kind": "customer_message", "reference_id": "customer-message:1"}
                ],
            },
        ],
    )
    result, provider = run(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": items}, "plan"),
            text("Which data region must the archive stay in, EU or US?"),
        ],
        missing=True,
    )
    assert len(provider.requests) == 2 and not result["calls"]
    assert result["plans"][0]["items"][0]["status"] == "awaiting_customer"
    assert first_response_metrics(tmp_path)["first_interaction_tool_errors"] == 0


def test_native_sufficient_input_can_complete_effects_without_extra_handoff(tmp_path):
    steps = [
        call(SYNC_WORK_PLAN_TOOL, {"items": plan(missing=False)}, "plan"),
        call(
            READ_TOOL,
            {
                "documents": [
                    {"entry_id": "storage_dune_prices", "revision": 1},
                    {"entry_id": "storage_dune_requirements", "revision": 1},
                ]
            },
            "read",
        ),
        call(UPDATE_WORK_PLAN_TOOL, {"items": [completed("choose")]}, "choose"),
    ]
    for identifier, tool, args in (
        ("create", "create_storage", {"plan_sku": "store_dune"}),
        (
            "export",
            "export_archive",
            {"archive_id": "A7", "credential_id": "K9", "workspace_id": "W8"},
        ),
        ("retire", "retire_credential", {"credential_id": "K9"}),
    ):
        steps.extend(
            [
                call(tool, args, identifier + "-effect"),
                call(
                    UPDATE_WORK_PLAN_TOOL,
                    {"items": [completed(identifier, identifier + "-effect")]},
                    identifier + "-done",
                ),
            ]
        )
    steps.append(text("Created W8, migrated A7, then retired K9."))
    result, _ = run(tmp_path, steps, missing=False)
    assert result["structural_pass"]
    assert result["calls"][0]["result"]["plan_sku"] == "store_dune"
