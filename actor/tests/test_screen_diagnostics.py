import json
import sqlite3
from types import SimpleNamespace

from cayu_tau3_banking.screen_diagnostics import (
    private_tool_diagnostics,
    termination_diagnostics,
)


def test_private_errors_are_distinct_from_public_and_deduplicated(tmp_path):
    with sqlite3.connect(tmp_path / "session.sqlite3") as db:
        db.execute("CREATE TABLE cayu_events(event_type TEXT, tool_name TEXT, payload_json TEXT)")
        for name, identity, error, structured in [
            ("controller_update_work_plan", "bad", True, None),
            ("controller_update_work_plan", "bad", True, None),
            ("cayu_read_saved_evidence", "read", False, None),
            ("host_action", "bank-error", True, None),
            ("cayu_update_decision_facts", "partial", False,
             {"status": "partial", "rejected": [{"id": "unsupported"}]}),
        ]:
            db.execute("INSERT INTO cayu_events VALUES (?, ?, ?)", (
                "tool.call.failed" if error else "tool.call.completed", name,
                json.dumps({"tool_call_id": identity,
                            "result": {"is_error": error, "structured": structured}}),
            ))
    result = private_tool_diagnostics(tmp_path)
    assert result["private_diagnostics_status"] == "complete"
    assert result["private_tool_errors"] == 1 and result["private_tool_calls"] == 3
    assert result["private_partial_patch_rejections"] == 1
    assert result["private_tool_errors_by_name"] == {"controller_update_work_plan": 1}


def test_missing_diagnostics_never_claim_zero_errors(tmp_path):
    assert private_tool_diagnostics(tmp_path)["private_tool_errors"] is None
    (tmp_path / "bad.sqlite3").write_text("invalid database")
    assert private_tool_diagnostics(tmp_path)["private_diagnostics_status"] == "incomplete"


def test_simulator_error_is_reported_without_altering_score():
    sim = SimpleNamespace(termination_reason=SimpleNamespace(value="user_error"),
                          reward_info=SimpleNamespace(reward=0.0))
    report = termination_diagnostics(sim, None)
    assert report["simulator_protocol_error"]
    assert report["failure_category"] == "simulator_protocol_error"
    assert not report["infrastructure_error"] and not report["agent_resource_stop"]
    assert sim.reward_info.reward == 0.0
    sim.termination_reason.value = "agent_stop"
    report = termination_diagnostics(sim, "agent_time_budget")
    assert report["agent_resource_stop"] and not report["simulator_protocol_error"]
