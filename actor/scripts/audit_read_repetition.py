"""Compare successful reads with their exact pre-request composed contexts."""

import argparse
import asyncio
import json
import sqlite3
from collections import Counter
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.overhead_profile import returned_document_revisions


def text_leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from text_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from text_leaves(item)
    elif isinstance(value, str):
        yield value
        if value.lstrip().startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except ValueError:
                return
            yield from text_leaves(decoded)


def body_present(body, messages):
    normalized = " ".join(body.split())
    return bool(normalized) and any(
        normalized in " ".join(text.split()) for text in text_leaves(messages)
    )


async def audit(root, documents):
    contexts = {}
    for path in root.glob("contexts/*/*.json"):
        context = json.loads(path.read_text())
        contexts.setdefault(context["model_step_id"], []).append((path, context))
    corpus = BankingKnowledgeCorpus.from_directory(documents)
    seen = Counter()
    rows = []
    for path in sorted(root.glob("sessions/*.sqlite3")):
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            events = db.execute(
                "SELECT sequence,payload_json FROM cayu_events "
                "WHERE event_type='tool.call.completed' AND tool_name='cayu_read_product_documents' "
                "ORDER BY sequence"
            ).fetchall()
        for sequence, raw in events:
            payload = json.loads(raw)
            candidates = contexts.get(payload["model_step_id"], [])
            if len(candidates) != 1:
                raise ValueError(f"Expected one exact composed context for event {sequence}")
            context_path, context = candidates[0]
            for identity in returned_document_revisions(payload):
                document = await corpus.read_document(*identity)
                rows.append(
                    {
                        "sequence": sequence,
                        "model_step_id": payload["model_step_id"],
                        "context_path": str(context_path.relative_to(root)),
                        "entry_id": identity[0],
                        "revision": identity[1],
                        "prior_read_count": seen[identity],
                        "full_body_in_composed_messages": body_present(
                            document["text"], context["messages"]
                        ),
                        "compacted_cursor": context["compaction"].get(
                            "compacted_transcript_cursor"
                        ),
                    }
                )
                seen[identity] += 1
    return {
        "scope": "Exact full-body presence in pre-request composed messages; no reference answers.",
        "limits": "Absence does not mean all decision-relevant facts were absent: summaries can retain them. "
        "Presence does not prove attention or whether changed customer constraints justify reconsideration. "
        "Composed context is not a provider-wire capture.",
        "document_reads": len(rows),
        "unique_revisions": len(seen),
        "repeat_reads": sum(r["prior_read_count"] > 0 for r in rows),
        "repeat_reads_body_still_present": sum(
            r["prior_read_count"] > 0 and r["full_body_in_composed_messages"] for r in rows
        ),
        "all_reads_body_still_present": sum(r["full_body_in_composed_messages"] for r in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(audit(args.root, args.documents))
    save_private_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
