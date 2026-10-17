import json
import sqlite3

from cayu_tau3_banking.runtime_profile import profile_session


def test_phase_profile_matches_attempt_ids_and_excludes_unfinished_calls(tmp_path):
    path = tmp_path / "session.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE cayu_events (sequence INTEGER, session_id TEXT, event_type TEXT, timestamp TEXT, tool_name TEXT, payload_json TEXT)"
        )
        events = [
            ("model.started", 0, None, {"model_attempt_id": "a"}),
            ("model.completed", 5, None, {"model_attempt_id": "a"}),
            (
                "tool.call.started",
                5,
                "controller_sync_work_plan",
                {"model_attempt_id": "a", "tool_call_id": "p"},
            ),
            (
                "tool.call.completed",
                6,
                "controller_sync_work_plan",
                {"tool_call_id": "p", "result": "DO_NOT_EXPORT"},
            ),
            ("model.started", 7, None, {"model_attempt_id": "b"}),
        ]
        for index, (kind, seconds, tool, payload) in enumerate(events):
            db.execute(
                "INSERT INTO cayu_events VALUES(?,?,?,?,?,?)",
                (
                    index,
                    "session",
                    kind,
                    f"2026-09-08T00:00:{seconds:02d}+00:00",
                    tool,
                    json.dumps(payload),
                ),
            )
    result = profile_session(path)
    assert result["completed_model_seconds"] == 5
    assert result["model_seconds_containing_plan_tools"] == 5
    assert result["incomplete_model_attempts"] == 1
    assert result["tool_execution_seconds"]["controller_sync_work_plan"] == 1
    assert "DO_NOT_EXPORT" not in json.dumps(result)
