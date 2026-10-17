"""Read-only reconstruction of explicit retrieval delivery from runtime events."""

import argparse
import json
import sqlite3
from pathlib import Path

from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL
from cayu_tau3_banking.product_navigation import READ_TOOL


def audit_session(path: Path) -> dict:
    rows = []
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        events = db.execute(
            "SELECT session_id, sequence, tool_name, payload_json FROM cayu_events "
            "WHERE event_type='tool.call.completed' AND tool_name IN (?,?) "
            "ORDER BY session_id,sequence",
            (KNOWLEDGE_SEARCH_TOOL, READ_TOOL),
        )
        for session, sequence, tool, raw in events:
            event = json.loads(raw)
            result = event.get("result", {})
            base = {
                "session_id": session,
                "sequence": sequence,
                "tool": tool,
                "tool_call_id": event.get("tool_call_id"),
                "model_step_id": event.get("model_step_id"),
                "arguments_exact": event.get("arguments_exact", False),
                "arguments": event.get("arguments"),
                "is_error": result.get("is_error"),
            }
            try:
                payload = json.loads(result["content"])
            except (KeyError, TypeError, ValueError):
                rows.append({**base, "inventory_available": False})
                continue
            if not isinstance(payload, dict):
                rows.append({**base, "inventory_available": False})
                continue
            pages = payload.get("batch", [{"index": 0, "result": payload}])
            for page in pages:
                page_base = {
                    **base,
                    "batch_index": page.get("index"),
                    "is_error": page.get("is_error", base["is_error"]),
                }
                content = page.get("result")
                if not isinstance(content, dict):
                    rows.append({**page_base, "inventory_available": False})
                    continue
                rows.append(
                    {
                        **page_base,
                        "inventory_available": isinstance(content.get("documents"), list),
                        "retrieval": content.get("retrieval"),
                        "retrieval_trace_id": content.get("retrieval_trace_id"),
                        "bounds": content.get("retrieval_bounds"),
                        "navigation": content.get("navigation"),
                        "source_coverage": content.get("source_coverage"),
                        "documents": [
                            {
                                key: doc[key]
                                for key in (
                                    "entry_id",
                                    "entry_revision",
                                    "text_complete",
                                    "navigation_only",
                                    "channel_matches",
                                    "fused_rank",
                                    "source_window",
                                )
                                if key in doc
                            }
                            for doc in content.get("documents", [])
                        ],
                    }
                )
    return {
        "schema_version": 1,
        "rows": rows,
        "limitations": [
            "Explicit completed search/read events only; automatic recall, failed/blocked events and model attention are not reconstructed.",
            "Per-channel candidates outside delivered pages are not persisted by this interface.",
            "Arguments may contain customer query text; keep this report private.",
            "A returned document or exact argument does not prove the actor understood it.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_session(args.session), indent=2))
