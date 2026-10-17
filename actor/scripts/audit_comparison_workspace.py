"""Read-only extraction of frozen comparison evidence; no semantic auto-grading."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


def audit(output: Path) -> dict:
    result = {
        "case": output.name,
        "semantic_grade": "requires_manual_review",
        "finished": (output / "result.json").exists(),
    }
    for name in ("qualification", "error"):
        path = output / f"{name}.json"
        if path.exists():
            result[name] = json.loads(path.read_text())
    database = output / "runtime.sqlite3"
    if not database.exists():
        return result
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as conn:
        events = conn.execute(
            "SELECT event_type, payload_json FROM cayu_events ORDER BY sequence"
        ).fetchall()
        messages = conn.execute(
            "SELECT message_json FROM cayu_transcript_messages ORDER BY sequence"
        ).fetchall()
    result["events"] = dict(Counter(kind for kind, _ in events))
    calls, rejected, worksheets = {}, [], []
    for (raw,) in messages:
        for part in json.loads(raw).get("content", []):
            if part.get("type") == "tool_call":
                calls[part["tool_call_id"]] = part
                if part["tool_name"] == "cayu_check_comparison_facts":
                    worksheets.append(
                        {"call_id": part["tool_call_id"], "arguments": part["arguments"]}
                    )
            elif part.get("type") == "tool_result" and part.get("is_error"):
                rejected.append(
                    {
                        "call_id": part["tool_call_id"],
                        "tool": part["tool_name"],
                        "reason": part.get("content"),
                    }
                )
    result["tool_counts"] = dict(Counter(call["tool_name"] for call in calls.values()))
    result["tool_rejections"] = rejected
    result["worksheet_proposals"] = worksheets
    result["provider_captures"] = [
        json.loads(path.read_text())
        for path in sorted(output.glob("provider-diagnostics/*/*.capture.json"))
    ]
    transcript = output / "transcript.json"
    if transcript.exists():
        result["customer_facing_messages"] = [
            {"role": message["role"], "content": message.get("content")}
            for message in json.loads(transcript.read_text())
            if message["role"] in {"user", "assistant"} and not message.get("tool_calls")
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.case_directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
