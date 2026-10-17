"""Summarize completed paired runs without inspecting reference answers."""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json


def audit_arm(root: Path) -> dict:
    manifest = json.loads((root / "source-manifest.json").read_text())["files"]
    for name, expected in manifest.items():
        path = (root / name).resolve(strict=True)
        if (
            not path.is_relative_to(root.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Frozen source integrity check failed")
    rows = json.loads((root / "task_064/assessment.json").read_text())["rows"]
    if len(rows) != 1:
        raise ValueError("Expected exactly one trial per arm")
    row = rows[0]
    counts = Counter()
    tools = Counter()
    first_started = None
    first_completed = None
    sessions = list((root / "task_064/sessions").glob("*.sqlite3"))
    if len(sessions) > 1:
        raise ValueError("Expected at most one session per arm")
    for path in sessions:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            for kind, timestamp, tool in db.execute(
                "SELECT event_type,timestamp,tool_name FROM cayu_events ORDER BY sequence"
            ):
                counts[kind] += 1
                if kind in {"tool.call.completed", "tool.call.failed"}:
                    tools[tool] += 1
                if kind == "interaction.started" and first_started is None:
                    first_started = datetime.fromisoformat(timestamp)
                if kind == "interaction.completed" and first_completed is None:
                    first_completed = datetime.fromisoformat(timestamp)
    return {
        "assessment": row,
        "source_verified": True,
        "first_response_seconds": (
            round((first_completed - first_started).total_seconds(), 3)
            if first_started is not None and first_completed is not None
            else None
        ),
        "completed_model_calls": counts["model.completed"],
        "model_errors": counts["model.error"],
        "tool_rejections": counts["tool.call.failed"],
        "compactions": counts["context.compaction.completed"],
        "tool_calls": dict(tools),
    }


def compare(baseline: dict, candidate: dict) -> dict:
    def completed(row):
        return row["termination"] in {"user_stop", "agent_stop"} and row["reward"] is not None

    old = baseline["assessment"]
    new = candidate["assessment"]
    comparable = completed(old) and completed(new) and old["duration_seconds"] > 0
    improvement = 1 - new["duration_seconds"] / old["duration_seconds"] if comparable else None
    return {
        "wall_time_reduction_fraction": improvement,
        "meets_exploratory_threshold": (
            comparable and old["reward"] == 1 and new["reward"] == 1 and improvement >= 0.25
        ),
        "statistical_or_general_reliability_claim": False,
        "next": "Inspect trajectories before drawing conclusions; retain all failures",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = {arm: audit_arm(args.root / arm) for arm in ("baseline", "candidate")}
    result["comparison"] = compare(result["baseline"], result["candidate"])
    save_private_json(args.root / "paired-audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
