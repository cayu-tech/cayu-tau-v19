import json
import sqlite3

from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL
from cayu_tau3_banking.overhead_profile import returned_document_revisions
from cayu_tau3_banking.product_navigation import READ_TOOL
from cayu_tau3_banking.retrieval_audit import audit_session


def test_window_is_not_counted_as_full_revision_delivery():
    document = {
        "entry_id": "a",
        "entry_revision": 1,
        "text_complete": False,
        "navigation_only": True,
    }
    assert (
        returned_document_revisions({"result": {"content": json.dumps({"documents": [document]})}})
        == []
    )


def test_audit_reports_partial_reads_and_missing_payload_without_inventing_data(tmp_path):
    path = tmp_path / "session.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE cayu_events(session_id,sequence,tool_name,payload_json,event_type)"
        )
        for index, content in enumerate(
            [
                json.dumps(
                    {
                        "documents": [
                            {
                                "entry_id": "a",
                                "entry_revision": 1,
                                "text_complete": False,
                                "navigation_only": True,
                            }
                        ]
                    }
                ),
                "unavailable",
            ]
        ):
            db.execute(
                "INSERT INTO cayu_events VALUES(?,?,?,?,?)",
                (
                    "s",
                    index,
                    READ_TOOL,
                    json.dumps({"result": {"content": content}}),
                    "tool.call.completed",
                ),
            )
    report = audit_session(path)
    assert report["rows"][0]["documents"][0]["navigation_only"]
    assert not report["rows"][1]["inventory_available"]
    assert not report["rows"][0]["arguments_exact"]


def test_batch_error_is_not_hidden_by_successful_outer_result(tmp_path):
    path = tmp_path / "session.sqlite3"
    payload = {
        "result": {
            "is_error": False,
            "content": json.dumps(
                {
                    "batch": [
                        {"index": 0, "is_error": False, "result": {"documents": []}},
                        {"index": 1, "is_error": True, "result": "unavailable"},
                    ]
                }
            ),
        }
    }
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE cayu_events(session_id,sequence,tool_name,payload_json,event_type)"
        )
        db.execute(
            "INSERT INTO cayu_events VALUES(?,?,?,?,?)",
            ("s", 1, KNOWLEDGE_SEARCH_TOOL, json.dumps(payload), "tool.call.completed"),
        )
    rows = audit_session(path)["rows"]
    assert [row["is_error"] for row in rows] == [False, True]
    assert not rows[1]["inventory_available"]
