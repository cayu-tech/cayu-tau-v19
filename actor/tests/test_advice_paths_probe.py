import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest
from cayu import ScriptedModelProvider
from test_runtime_banking import call, text

from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.product_navigation import COMPARISON_TOOL, READ_TOOL

spec = importlib.util.spec_from_file_location(
    "probe_advice_paths", Path(__file__).parents[1] / "scripts/probe_advice_paths.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

DOC = "doc_savings_accounts_quartz_001"
PRODUCT = "personal_savings/quartz"
SOURCE = "Quartz Savings APY is 3.9%. Annual account fee is $0. Minimum balance is $100."
REFERENCE = {"kind": "knowledge_revision", "reference_id": DOC, "version": "1"}


def execute(tmp_path, arm, steps, *, max_steps=10):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / (DOC + ".json")).write_text(
        json.dumps({"id": DOC, "title": "Quartz terms", "content": SOURCE})
    )
    output = tmp_path / "run"
    output.mkdir()
    provider = ScriptedModelProvider(steps)
    result = probe.run_case(
        output,
        {"id": "fixture", "customer": "Compare Quartz terms. Advice only."},
        arm=arm,
        corpus_directory=corpus,
        policy=probe.POLICY,
        provider_factory=lambda: provider,
        model="scripted",
        timeout=90,  # Functional scripted rounds, not a wall-clock latency assertion.
        max_steps=max_steps,
    )
    return result, provider, output


def read_step(identifier=DOC, call_id="read"):
    return call(READ_TOOL, {"documents": [{"entry_id": identifier, "revision": 1}]}, call_id)


def worksheet():
    return {
        "product_ids": [PRODUCT],
        "required_fields": ["rate"],
        "facts": [
            {
                "product_id": PRODUCT,
                "field": "rate",
                "value": "3.9",
                "entry_id": DOC,
                "revision": 1,
                "quote": "Quartz Savings APY is 3.9%.",
            }
        ],
        "calculations": [
            {
                "label": "annual interest on 10000",
                "terms": [{"product_id": PRODUCT, "field": "rate", "coefficient": "100"}],
            }
        ],
    }


def test_minimal_runs_real_read_and_continues_to_final_reply(tmp_path):
    result, provider, output = execute(
        tmp_path, "minimal", [read_step(), text("Quartz yields 3.9%.")]
    )
    assert result["status"] == "answered" and result["reply"] == "Quartz yields 3.9%."
    assert result["plan"]["items"] == []
    assert len(provider.requests) == 2
    assert all({t["name"] for t in r.tools} <= probe.READ_ONLY_TOOLS for r in provider.requests)
    last = json.dumps([m.model_dump(mode="json") for m in provider.requests[-1].messages])
    assert SOURCE in last
    profile = json.loads((output / "profile.json").read_text())
    assert profile["event_counts"]["tool.call.completed:" + READ_TOOL] == 1
    assert profile["event_counts"]["model.completed"] == 2
    with sqlite3.connect(output / "runtime.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM cayu_events").fetchone()[0] > 0


def test_controller_runs_plan_read_worksheet_update_then_final_reply(tmp_path):
    plan = {
        "id": "advice",
        "goal": "Explain Quartz terms",
        "completion_mode": "information",
        "status": "in_progress",
        "outcome": "unresolved",
        "depends_on": [],
        "obligations": [
            {"id": "research", "goal": "Read terms", "kind": "knowledge_search", "status": "ready"},
            {"id": "answer", "goal": "Explain terms", "kind": "communication", "status": "ready"},
        ],
    }
    result, provider, output = execute(
        tmp_path,
        "controller",
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [plan]}, "sync"),
            read_step(),
            call(COMPARISON_TOOL, worksheet(), "check"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "advice",
                            "obligations": [
                                {
                                    "id": "research",
                                    "status": "resolved",
                                    "evidence": "Exact public terms read",
                                    "evidence_refs": [REFERENCE],
                                }
                            ],
                        }
                    ]
                },
                "update",
            ),
            text("Quartz yields $390 on $10000 for the year."),
        ],
    )
    assert result["status"] == "answered", result
    assert len(provider.requests) == 5
    profile = json.loads((output / "profile.json").read_text())
    for name in (SYNC_WORK_PLAN_TOOL, READ_TOOL, COMPARISON_TOOL, UPDATE_WORK_PLAN_TOOL):
        assert profile["event_counts"]["tool.call.completed:" + name] == 1
    last = json.dumps([m.model_dump(mode="json") for m in provider.requests[-1].messages])
    assert "calculation_results" in last and "390" in last
    assert result["plan"]["items"][0]["obligations"][0]["status"] == "resolved"


def test_real_read_error_is_returned_and_can_be_repaired(tmp_path):
    result, provider, _output = execute(
        tmp_path,
        "minimal",
        [
            read_step("missing"),
            read_step(call_id="repair"),
            text("Quartz yields 3.9%."),
        ],
    )
    assert result["status"] == "answered" and len(provider.requests) == 3
    second = json.dumps([m.model_dump(mode="json") for m in provider.requests[1].messages])
    assert "is_error" in second and "Navigation/check rejected" in second


def test_no_final_reply_at_step_limit_is_not_scored_as_an_answer(tmp_path):
    result, _, output = execute(tmp_path, "minimal", [read_step()], max_steps=1)
    assert result["status"] == "error" and "reply" not in result
    assert (output / "transcript.json").exists()


def test_minimal_rejects_registration_of_bank_actions():
    with pytest.raises(ValueError, match="cannot register bank actions"):
        probe.build_minimal_resources(
            None,
            (object(),),
            provider=None,
            model="fixture",
            domain_policy="",
            memory_runtime=None,
            session_store=None,
            aclose=None,
            compaction=None,
            context_capture_directory=None,
            provider_options={},
        )


def test_case_expectations_cannot_enter_runtime(tmp_path):
    with pytest.raises(ValueError, match="actor-visible"):
        probe.run_case(
            tmp_path,
            {"id": "x", "customer": "Question", "expected": "Secret answer"},
            arm="minimal",
            corpus_directory=tmp_path,
            policy="",
            provider_factory=lambda: None,
        )


def test_preparation_freezes_four_cases_two_paths_and_all_sources(tmp_path):
    root = probe.prepare(tmp_path)
    protocol = probe.verify(root)
    cases = json.loads((root / "cases.json").read_text())
    expected = json.loads((root / "expectations-evaluator-only.json").read_text())
    assert len(cases) == 4 and protocol["arms"] == ["controller", "minimal"]
    assert all(set(c) == {"id", "customer"} for c in cases)
    assert set(expected) == {c["id"] for c in cases}
    assert len(list((root / "corpus").glob("*.json"))) == 8
    assert not (root / "started.json").exists()
    (root / "cases.json").write_text("[]")
    with pytest.raises(ValueError, match="Frozen input"):
        probe.verify(root)
