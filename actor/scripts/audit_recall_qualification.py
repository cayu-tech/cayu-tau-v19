"""Read completed fictional qualification traces; never feed results to an actor."""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json


def active_procedure_present(texts, procedure):
    for text in texts:
        if "<active-work-evidence>" not in text:
            continue
        payload = text.split("<active-work-evidence>", 1)[1].split("</active-work-evidence>", 1)[0]
        packet = json.loads(payload)
        if any(
            d.get("text_complete") is True and d.get("text") == procedure.strip()
            for d in packet.get("documents", [])
        ):
            return True
    return False


def audit_case(root):
    result = json.loads((root / "result.json").read_text())
    procedure = json.loads((root / "corpus/archive-procedure.json").read_text())["content"]
    with sqlite3.connect(
        (root / "runtime.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
    ) as db:
        events = [
            (seq, kind, name, json.loads(raw))
            for seq, kind, name, raw in db.execute(
                "SELECT sequence,event_type,tool_name,payload_json FROM cayu_events ORDER BY sequence"
            )
        ]
    starts = {v["model_step_id"]: seq for seq, k, _, v in events if k == "model.started"}
    counts = Counter(k for _, k, _, _ in events)
    contexts = []
    for path in root.glob("contexts/*/*.json"):
        value = json.loads(path.read_text())
        text = [
            part["text"]
            for message in value["messages"]
            for part in message.get("content", [])
            if part.get("type") == "text"
        ]
        contexts.append(
            {
                "model_step_id": value["model_step_id"],
                "sequence": starts.get(value["model_step_id"]),
                "path": str(path.relative_to(root)),
                "compacted": bool(value["compaction"]),
                "full_procedure_literal_present": any(procedure.strip() in t for t in text),
                "exact_active_procedure_present": active_procedure_present(text, procedure),
                "recovery_delta_present": any("<cayu_memory_delta" in t for t in text),
                "active_evidence_present": any("<active-work-evidence" in t for t in text),
            }
        )
    contexts.sort(key=lambda v: v["sequence"] or 0)
    error_events = [
        {"sequence": seq, "type": k, "tool": name}
        for seq, k, name, v in events
        if k
        in {
            "session.failed",
            "interaction.failed",
            "model.error",
            "model.failed",
            "context.compaction.failed",
            "tool.call.failed",
        }
        or (k == "tool.call.completed" and (v.get("result") or {}).get("is_error"))
    ]
    writes = []
    for seq, kind, name, _ in events:
        if kind != "tool.call.started" or name not in {"export_archive", "retire_credential"}:
            continue
        prior = [v for v in contexts if v["sequence"] is not None and v["sequence"] < seq]
        writes.append({"sequence": seq, "tool": name, "context": prior[-1] if prior else None})
    transcript = json.loads((root / "transcript.json").read_text())
    return {
        "case": result["case"],
        "structural_pass": result["structural_pass"],
        "seconds": result["seconds"],
        "writes": result["writes"],
        "research_only_phase_mutations": result["research_only_phase_mutations"],
        "models": sorted({v["model"] for _, k, _, v in events if k == "model.started"}),
        "model_starts": counts["model.started"],
        "model_completions": counts["model.completed"],
        "compactions": counts["context.compaction.completed"],
        "errors": error_events,
        "contexts": contexts,
        "write_contexts": writes,
        "dependency_edges": result["dependency_edges"],
        "replies": [
            m["content"] for m in transcript if m["role"] == "assistant" and m.get("content")
        ],
        "limits": "Composed-context snapshots, not provider wire captures. Full literal containment is reported, not inferred from a receipt. Actual calls and final state have separate structural checks. Replies require human review. This is not a benchmark score or a causal ablation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    completed = json.loads((args.root / "completed.json").read_text())
    rows = [audit_case(args.root / row["case"]) for row in completed["results"]]
    save_private_json(args.root / "audit.json", {"cases": rows})
    for row in rows:
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "case",
                        "structural_pass",
                        "seconds",
                        "writes",
                        "model_starts",
                        "compactions",
                        "errors",
                        "write_contexts",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
