"""Read-only, content-free phase timings from a single native runtime session.

Model time is wall time between persisted start/completion events, not CPU time.
Calls containing a plan tool can include other reasoning; this is not time spent
solely writing JSON. Helper-model durations are not recorded in these events.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL


def profile_session(path: Path) -> dict:
    starts = {}
    completions = {}
    planning = set()
    tools = {}
    counts = Counter()
    durations = Counter()
    relevant = (
        "model.started",
        "model.completed",
        "model.failed",
        "tool.call.started",
        "tool.call.completed",
        "tool.call.failed",
        "tool.call.blocked",
        "session.interrupted",
    )
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT session_id,event_type,timestamp,tool_name,payload_json FROM cayu_events "
            "WHERE event_type IN (" + ",".join("?" for _ in relevant) + ") ORDER BY sequence",
            relevant,
        )
        sessions = set()
        for session, kind, timestamp, tool, raw in rows:
            sessions.add(session)
            payload = json.loads(raw)
            at = datetime.fromisoformat(timestamp)
            attempt = payload.get("model_attempt_id")
            counts[kind] += 1
            if kind == "model.started" and attempt:
                starts[attempt] = at
            elif kind == "model.completed" and attempt:
                completions[attempt] = at
            elif kind == "tool.call.started":
                tools[payload["tool_call_id"]] = at
                if tool in {SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL}:
                    planning.add(attempt)
            elif kind in {"tool.call.completed", "tool.call.failed"}:
                counts[f"{kind}:{tool}"] += 1
                start = tools.get(payload.get("tool_call_id"))
                if start is not None:
                    durations[tool] += (at - start).total_seconds()
    if len(sessions) != 1:
        raise ValueError("Expected exactly one nonempty native runtime session")
    matched = {
        key: (end - starts[key]).total_seconds()
        for key, end in completions.items()
        if key in starts
    }
    return {
        "schema_version": 1,
        "event_counts": dict(sorted(counts.items())),
        "completed_model_calls_with_timing": len(matched),
        "incomplete_model_attempts": len(starts.keys() - completions.keys()),
        "model_calls_containing_plan_tools": len(matched.keys() & planning),
        "completed_model_seconds": round(sum(matched.values()), 3),
        "model_seconds_containing_plan_tools": round(
            sum(v for k, v in matched.items() if k in planning), 3
        ),
        "tool_execution_seconds": {
            key: round(value, 3) for key, value in sorted(durations.items())
        },
        "limitations": [
            "Plan-associated model calls include reasoning beyond plan creation.",
            "Unfinished model durations and isolated reviewer durations are not included.",
            "Overlapping tool durations must not be summed as exclusive wall time.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    print(json.dumps(profile_session(args.session), indent=2))


if __name__ == "__main__":
    main()
