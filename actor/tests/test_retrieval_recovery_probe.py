import json

import pytest
from cayu import ScriptedModelProvider
from test_outcome_controller import outcome
from test_runtime_banking import call, text

from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL
from cayu_tau3_banking.retrieval_recovery_probe import assess, run_case


@pytest.mark.parametrize("case", ["complete", "recoverable", "unavailable"])
def test_native_recovery_probe_executes_and_scores_outside_actor_context(tmp_path, case):
    steps = [
        call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}),
        call(KNOWLEDGE_SEARCH_TOOL, {"query": "Harbor parcel mass"}, "search1"),
    ]
    if case != "complete":
        steps.append(
            call(
                KNOWLEDGE_SEARCH_TOOL,
                {"query": "Harbor maximum kilograms", "mode": "lexical"},
                "search2",
            )
        )
    report = {
        "maximum_kg": None if case == "unavailable" else 17,
        "evidence_entry_ids": [] if case == "unavailable" else ["specification"],
        "unresolved_reason": "No Harbor mass specification found" if case == "unavailable" else "",
    }
    steps.extend(
        [
            call("record_research_result", report, "report"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "request",
                            "status": "completed",
                            "outcome": "succeeded",
                            "evidence": "Research finding recorded",
                            "evidence_refs": [{"kind": "tool_result", "reference_id": "report"}],
                        }
                    ]
                },
            ),
            text("Recorded the finding."),
        ]
    )
    result = run_case(
        tmp_path, case=case, provider_factory=lambda: ScriptedModelProvider(steps), model="scripted"
    )
    assert result["passed"]
    first = json.loads((tmp_path / "search-01.json").read_text())
    assert first["response"]["documents"][0]["entry_id"] == (
        "specification" if case == "complete" else "reception"
    )
    assert (tmp_path / "runtime.sqlite3").exists()


def test_assessment_rejects_wrong_product_unseen_evidence_and_duplicate_reports():
    searches = [{"response": {"documents": [{"entry_id": "other-service"}]}}] * 2
    report = {"maximum_kg": 38, "evidence_entry_ids": ["other-service"], "unresolved_reason": ""}
    assert not assess("recoverable", [report], searches)["passed"]
    report = {**report, "maximum_kg": 17, "evidence_entry_ids": ["specification"]}
    assert not assess("recoverable", [report], searches)["passed"]
    assert not assess("recoverable", [report, report], searches)["passed"]
