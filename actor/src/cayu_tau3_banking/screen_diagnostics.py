"""Observational diagnostics; never rewrite upstream grades or retry outcomes."""

import json
import sqlite3


def private_tool_diagnostics(directory):
    paths = sorted(directory.glob("*.sqlite3"))
    counts = {}
    calls = set()
    partial_entries = 0
    batch_errors = 0
    if not paths:
        return {"private_diagnostics_status": "unavailable", "private_tool_errors": None}
    try:
        for path in paths:
            with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
                rows = db.execute(
                    "SELECT event_type, tool_name, payload_json FROM cayu_events "
                    "WHERE event_type IN ('tool.call.failed', 'tool.call.completed')"
                )
                for event, name, raw in rows:
                    if not name or not name.startswith(("cayu_", "controller_")):
                        continue
                    data = json.loads(raw)
                    identity = (str(path), data["tool_call_id"])
                    if identity in calls:
                        continue
                    calls.add(identity)
                    result = data.get("result") or {}
                    if event == "tool.call.failed" or result.get("is_error"):
                        counts[name] = counts.get(name, 0) + 1
                    structured = result.get("structured") or {}
                    if structured.get("status") == "partial":
                        partial_entries += len(structured.get("rejected", []))
                    try:
                        content = json.loads(result.get("content", ""))
                    except (ValueError, TypeError):
                        content = None
                    if isinstance(content, dict) and isinstance(content.get("batch"), list):
                        batch_errors += sum(
                            bool(item.get("is_error")) for item in content["batch"]
                            if isinstance(item, dict)
                        )
    except (sqlite3.Error, ValueError, KeyError, TypeError):
        return {"private_diagnostics_status": "incomplete", "private_tool_errors": None}
    return {
        "private_diagnostics_status": "complete",
        "private_tool_errors": sum(counts.values()),
        "private_tool_errors_by_name": counts,
        "private_tool_calls": len(calls),
        "private_partial_patch_rejections": partial_entries,
        "private_batch_item_errors": batch_errors,
    }


def termination_diagnostics(sim, failure_category):
    termination = sim.termination_reason.value
    return {
        "score_source": "upstream_tau",
        "simulator_protocol_error": termination == "user_error",
        "agent_resource_stop": failure_category in {
            "planning_stall", "admission_stall", "runtime_limit", "agent_time_budget"
        },
        "infrastructure_error": termination == "infrastructure_error",
        "failure_category": (
            "simulator_protocol_error" if termination == "user_error" else failure_category
        ),
    }
