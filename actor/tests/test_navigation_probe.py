import json

from cayu import ScriptedModelProvider
from test_runtime_banking import call, text, work

from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL
from cayu_tau3_banking.navigation_probe import CASES, DOCUMENTS, run_case
from cayu_tau3_banking.product_navigation import READ_TOOL
from cayu_tau3_banking.retrieval_audit import audit_session
from cayu_tau3_banking.runtime_profile import profile_session


def test_probe_uses_native_runtime_and_keeps_expectations_out_of_context(tmp_path):
    entry = "doc_savings_accounts_quartz_001"
    pending = work()
    completed = work(done=True, reference=entry)
    pending.pop("next_operation")
    completed.pop("next_operation")
    provider = ScriptedModelProvider(
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [pending]}, "plan"),
            call(READ_TOOL, {"documents": [{"entry_id": entry, "revision": 1}]}, "read"),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [completed]}, "complete"),
            text("Quartz yields $390 annually."),
        ]
    )
    case = {**CASES[0], "expected": "ORACLE_MUST_NOT_REACH_ACTOR"}
    policy = "SYNTHETIC_POLICY_SENTINEL: advice only, never perform customer operations."
    result = run_case(
        tmp_path, case, provider_factory=lambda: provider, model="scripted", policy=policy
    )
    assert result["reply"] == "Quartz yields $390 annually."
    assert len(list((tmp_path / "corpus").glob("*.json"))) == len(DOCUMENTS)
    for request in provider.requests:
        assert "ORACLE_MUST_NOT_REACH_ACTOR" not in request.model_dump_json()
    assert any("SYNTHETIC_POLICY_SENTINEL" in r.model_dump_json() for r in provider.requests)
    profile = profile_session(tmp_path / "runtime.sqlite3")
    assert profile["event_counts"]["model.completed"] == 4
    assert profile["event_counts"].get("tool.call.failed", 0) == 0
    assert json.loads((tmp_path / "result.json").read_text())["assessment"].startswith("Unscored")


def test_native_actor_search_window_then_full_read_is_reconstructable(tmp_path):
    entry = "doc_savings_accounts_quartz_001"
    provider = ScriptedModelProvider(
        [
            call(
                KNOWLEDGE_SEARCH_TOOL,
                {
                    "queries": [
                        {"query": "Quartz", "mode": "literal"},
                        {"query": "zzzxqnomatch", "mode": "lexical"},
                    ]
                },
                "search",
            ),
            call(
                READ_TOOL,
                {
                    "documents": [{"entry_id": entry, "revision": 1}],
                    "source_span": {"start": 0, "end": 20, "context_chars": 10},
                },
                "window",
            ),
            call(READ_TOOL, {"documents": [{"entry_id": entry, "revision": 1}]}, "full"),
            text("Quartz pays 3.9% APY with no annual fee."),
        ]
    )
    run_case(
        tmp_path,
        CASES[0],
        provider_factory=lambda: provider,
        model="scripted",
        advice_mode="lean",
        controller_mode="advisory",
    )
    rows = audit_session(tmp_path / "runtime.sqlite3")["rows"]
    assert len(rows) == 4
    assert rows[0]["documents"][0]["entry_id"] == entry
    assert rows[1]["documents"] == []
    assert rows[2]["documents"][0]["navigation_only"]
    assert not rows[2]["documents"][0]["text_complete"]
    assert rows[3]["documents"][0]["text_complete"]
    assert all(row["arguments_exact"] for row in rows)
    traces = {
        trace["trace_id"]: trace
        for path in (tmp_path / "contexts").glob("retrieval-*/*.json")
        for trace in [json.loads(path.read_text())]
    }
    assert len(traces) == 2
    assert all(row["retrieval_trace_id"] in traces for row in rows[:2])
    assert all(trace["inventory"] is not None for trace in traces.values())
