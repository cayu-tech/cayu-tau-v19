"""Read-only reconstruction of context, rereads, limits and comparison state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def documents(value, depth=0):
    if depth > 20:
        raise ValueError("Unexpected nested source representation")
    found = []
    if isinstance(value, dict):
        if "entry_id" in value and "text" in value and value.get("text_complete"):
            found.append(value)
        for child in value.values():
            found.extend(documents(child, depth + 1))
    elif isinstance(value, list):
        for child in value:
            found.extend(documents(child, depth + 1))
    elif isinstance(value, str):
        try:
            nested = json.loads(value)
        except json.JSONDecodeError:
            nested = None
        if isinstance(nested, (dict, list)):
            found.extend(documents(nested, depth + 1))
        for tag in ("active-work-evidence", "active-tool-documentation"):
            for match in re.finditer(f"<{tag}>(.*?)</{tag}>", value, re.S):
                found.extend(documents(json.loads(match[1]), depth + 1))
    return found


def body_key(doc):
    return doc["entry_id"], hashlib.sha256(doc["text"].encode()).hexdigest()


def section(text, tag):
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.S)
    if not match:
        raise ValueError(f"Missing {tag}")
    return match[1]


def analyze(root):
    manifest = json.loads((root.parent / "source-manifest.json").read_text())["files"]
    assert all(digest(root.parent / p) == h for p, h in manifest.items())
    database = next(root.glob("sessions/*.sqlite3"))
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        events = [
            (s, k, t, n, json.loads(v))
            for s, k, t, n, v in db.execute(
                "SELECT sequence,event_type,timestamp,tool_name,payload_json FROM cayu_events ORDER BY sequence"
            )
        ]
    contexts = {}
    for path in root.glob("contexts/*/*.json"):
        value = json.loads(path.read_text())
        contexts[value["model_step_id"]] = (path, value)
    captures = list(root.glob("provider-diagnostics/*/*.request.json"))
    seen = set()
    reads = []
    boundaries = []
    for sequence, kind, timestamp, name, value in events:
        if kind == "tool.call.completed":
            returned = documents(value["result"]["content"])
            keys = {body_key(d) for d in returned}
            if name == "cayu_read_product_documents" and 800 < sequence < 1185:
                context_path, context = contexts[value["model_step_id"]]
                visible = {body_key(d) for d in documents(context["messages"])}
                reads.append(
                    {
                        "event_sequence": sequence,
                        "arguments": value["arguments"],
                        "returned_complete_documents": len(keys),
                        "exact_documents_previously_delivered_by_tools": len(keys & seen),
                        "exact_documents_in_current_structured_source_blocks": len(keys & visible),
                        "context": str(context_path.relative_to(root)),
                        "context_sha256": digest(context_path),
                    }
                )
            seen.update(keys)
        if kind == "model.started" and sequence in (1038, 1142, 1185):
            model_step = value["model_step_id"]
            completion = next(
                (t, v)
                for _, k, t, _, v in events
                if k == "model.completed" and v.get("model_step_id") == model_step
            )
            begin, end = (datetime.fromisoformat(t).timestamp() for t in (timestamp, completion[0]))
            paths = [p for p in captures if begin <= p.stat().st_mtime <= end]
            assert len(paths) == 1
            capture = json.loads(paths[0].read_text())
            assert not capture["payload_redacted"] and capture["omitted_field_count"] == 0
            payload = capture["payload"]
            memory = next(
                p["text"]
                for m in payload["input"]
                if m.get("role") == "user"
                for p in m["content"]
                if "<banking_context>" in p.get("text", "")
            )
            summary = payload["input"][0]["content"][0]["text"]
            capacity = json.loads(section(memory, "research-capacity").splitlines()[1])
            plan = json.loads(section(memory, "workflow-plan"))
            boundaries.append(
                {
                    "event_sequence": sequence,
                    "model_step_id": model_step,
                    "provider_reported_input_tokens": completion[1]["usage"]["input_tokens"],
                    "capture": str(paths[0].relative_to(root)),
                    "capture_sha256": digest(paths[0]),
                    "capacity": capacity,
                    "tools": [t["name"] for t in payload["tools"]],
                    "silver_bonus_in_summary": "Silver +0.45%" in summary,
                    "no_refusal_in_summary": "No refusals stated" in summary,
                    "summary_characters": len(summary),
                    "memory_characters": len(memory),
                    "memory_section_characters": {
                        tag: len(section(memory, tag))
                        for tag in (
                            "workflow-plan",
                            "active-work-evidence",
                            "active-tool-documentation",
                            "observation-ledger",
                        )
                    },
                    "comparison_obligations": [
                        {"item_id": item["id"], **ob}
                        for item in plan["items"]
                        for ob in item.get("obligations", [])
                        if ob["id"] in ("checking-comparison", "savings-comparison")
                    ],
                    "retained_work_documents": [
                        d["entry_id"]
                        for d in json.loads(section(memory, "active-work-evidence"))["documents"]
                    ],
                    "retained_tool_documents": [
                        d["entry_id"]
                        for d in json.loads(section(memory, "active-tool-documentation"))[
                            "documents"
                        ]
                    ],
                }
            )
    assert len(reads) == 12 and len(boundaries) == 3
    last_four = reads[-4:]
    assert all(
        r["returned_complete_documents"] == r["exact_documents_previously_delivered_by_tools"]
        for r in last_four
    )
    assert all(r["exact_documents_in_current_structured_source_blocks"] == 0 for r in last_four)
    for boundary in boundaries[1:]:
        assert boundary["silver_bonus_in_summary"]
        assert "cayu_read_product_documents" not in boundary["tools"]
        assert "cayu_search_banking_knowledge" not in boundary["tools"]
    return {
        "scope": "Post-run diagnosis, not actor inputs or benchmark grading",
        "frozen_source_files_verified": len(manifest),
        "reads": reads,
        "boundaries": boundaries,
        "confirmed": [
            "The comparison turn used all 12 application document-read calls and 3 searches.",
            "The final four read calls returned 33 exact documents previously delivered; their complete bodies were absent from the structured source blocks in the immediately preceding composed requests.",
            "Do not describe those repeats as reading documents already fully in context: the summaries retained selected facts but compaction had removed the original bodies.",
            "Before the plan was closed, neither read nor search schema remained available. Directory browsing remained; it does not return document bodies.",
            "The Silver +0.45% lead survived compaction. Silver eligibility/cost documents were never requested in the episode.",
            "The comparison was marked resolved on a no-new-card basis; the customer had not excluded all cards. The eventual answer did offer a Gold-card alternative, so this was selective omission, not categorical refusal to consider any card.",
            "An unrelated debit-card dispute procedure was retained after capability lookup for get_bank_account_transactions_9173 because it mentions that tool. This is the agent's capability-index/retention path, not evidence of a vector-ranking failure.",
        ],
        "limits": [
            "Presence in a request does not establish model attention; context overload is not disproven.",
            "Prepared payloads correlated to model steps using a unique file timestamp inside each step interval, not a shared capture ID.",
            "Structured source extraction does not reconstruct opaque provider reasoning; summaries may retain many facts without full source bodies.",
            "The trace proves budget exhaustion and exposed instructions/state, not the private causal reason for every model choice.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.task_root)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {
                "reads": len(result["reads"]),
                "boundaries": len(result["boundaries"]),
                "source_files_verified": result["frozen_source_files_verified"],
            }
        )
    )
