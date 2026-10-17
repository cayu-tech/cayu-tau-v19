"""Content-free monitor that distinguishes actor and compaction failures."""

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

EVENT_TYPES = (
    "model.started",
    "model.completed",
    "model.error",
    "model.failed",
    "context.compaction.completed",
    "context.compaction.failed",
    "tool.call.completed",
    "tool.call.failed",
    "interaction.completed",
    "interaction.failed",
    "session.failed",
)


def summarize_events(rows):
    counts = Counter()
    tools = Counter()
    for kind, name, payload in rows:
        counts[kind] += 1
        if kind == "model.started":
            role = "compactor" if payload.get("purpose") == "context_compaction" else "actor"
            counts[f"{role}_model_started"] += 1
        elif kind == "model.completed":
            if payload.get("purpose") == "context_compaction":
                # Compactor terminal events use model.completed even when the
                # request failed. A terminal event is not a successful call.
                counts["compactor_model_terminal"] += 1
                if (
                    payload.get("error_type")
                    or payload.get("compaction_outcome") == "provider_error"
                ):
                    counts["compactor_model_failed"] += 1
            elif payload.get("status") == "completed":
                counts["actor_model_completed"] += 1
        if kind in {"tool.call.completed", "tool.call.failed"}:
            tools[name] += 1
            result = payload.get("result") or {}
            if kind == "tool.call.failed" or result.get("is_error"):
                counts["tool_result_errors"] += 1
    return {
        "actor_model_started": counts["actor_model_started"],
        "actor_model_completed": counts["actor_model_completed"],
        "actor_model_error_events": counts["model.error"] + counts["model.failed"],
        "compactor_model_started": counts["compactor_model_started"],
        "compactor_model_terminal": counts["compactor_model_terminal"],
        "compactor_model_failed": counts["compactor_model_failed"],
        "compactions_completed": counts["context.compaction.completed"],
        "compactions_failed": counts["context.compaction.failed"],
        "interactions_completed": counts["interaction.completed"],
        "interactions_failed": counts["interaction.failed"],
        "sessions_failed": counts["session.failed"],
        "runtime_failure": bool(
            counts["session.failed"]
            or counts["interaction.failed"]
            or counts["context.compaction.failed"]
            or counts["compactor_model_failed"]
            or counts["model.error"]
            or counts["model.failed"]
        ),
        "tool_result_errors": counts["tool_result_errors"],
        "tool_counts": dict(tools),
    }


def summarize_captures(directory):
    """Read capture health only; never read private provider error text."""
    result = {
        "status_files": 0,
        "records_written": 0,
        "sink_failures": 0,
        "records_dropped": 0,
        "invalid_status_files": 0,
    }
    for path in directory.glob("provider-*/*.capture.json"):
        result["status_files"] += 1
        try:
            value = json.loads(path.read_text())
            if (
                not isinstance(value, dict)
                or value.get("schema") != "cayu-tau3.provider-capture.v1"
            ):
                raise ValueError("Invalid capture status")
            for key in ("records_written", "sink_failures", "records_dropped"):
                if type(value.get(key)) is not int or value[key] < 0:
                    raise ValueError("Invalid capture counter")
            for key in ("records_written", "sink_failures", "records_dropped"):
                result[key] += value[key]
        except (OSError, ValueError):
            result["invalid_status_files"] += 1
    result["incomplete"] = bool(
        result["sink_failures"] or result["records_dropped"] or result["invalid_status_files"]
    )
    return result


def monitor(root):
    protocol = json.loads((root / "protocol.json").read_text())
    results = []
    for task in protocol["task_ids"]:
        directory = root / task
        started = directory / "started.json"
        row = {"task": task, "started": started.exists()}
        assessment = directory / "assessment.json"
        row["assessment_recorded"] = assessment.exists()
        if started.exists() and not assessment.exists():
            row["seconds_since_start"] = round(
                time.time() - json.loads(started.read_text())["started_unix"]
            )
        events = []
        for path in directory.glob("sessions/*.sqlite3"):
            with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
                events.extend(
                    (kind, name, json.loads(raw))
                    for kind, name, raw in db.execute(
                        "SELECT event_type,tool_name,payload_json FROM cayu_events "
                        "WHERE event_type IN (" + ",".join("?" for _ in EVENT_TYPES) + ") "
                        "ORDER BY sequence",
                        EVENT_TYPES,
                    )
                )
        row.update(summarize_events(events))
        row["provider_diagnostic_capture"] = summarize_captures(directory / "provider-diagnostics")
        if assessment.exists():
            row["assessment"] = json.loads(assessment.read_text())
        row["launcher_error_recorded"] = (directory / "error.json").exists()
        results.append(row)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(monitor(args.root.resolve()), indent=2))
