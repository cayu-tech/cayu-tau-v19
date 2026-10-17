"""Evaluate a frozen query manifest without embeddings or external requests."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path


async def evaluate(args):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus

    manifest = json.loads(args.cases.read_text())
    corpus = BankingKnowledgeCorpus.from_directory(args.documents)
    rows = []
    if not manifest["cases"]:
        raise ValueError("Expected nonempty cases")
    for case in manifest["cases"]:
        ids = []
        cursor = None
        seen_cursors = set()
        while True:
            page = json.loads(await corpus.search(case["query"], mode="lexical", cursor=cursor))
            ids.extend(document["entry_id"] for document in page["documents"])
            if not page["source_coverage"] or any(
                source["status"] not in {"complete", "partial"}
                or not set((source["reason"] or "").split("+"))
                <= {"", "lexical_truncated", "semantic_unsupported"}
                for source in page["source_coverage"]
            ):
                raise RuntimeError("Incomplete lexical source; cannot qualify retrieval")
            cursor = page["navigation"]["next_cursor"]
            if cursor is None:
                break
            if cursor in seen_cursors:
                raise RuntimeError("Pagination cycle")
            seen_cursors.add(cursor)
        required = set(case["required"])
        if not required:
            raise ValueError("Required source list must be nonempty")
        rows.append(
            {
                "id": case["id"],
                "required_ranks": {
                    key: ids.index(key) + 1 if key in ids else None for key in sorted(required)
                },
                "recall4": len(required.intersection(ids[:4])) / len(required),
                "inventory_recall": len(required.intersection(ids)) / len(required),
                "inventory_documents": len(ids),
                "source_coverage": page["source_coverage"],
            }
        )
    report = {
        "scope": manifest["scope"],
        "rows": rows,
        "case_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "corpus_sha256": corpus.corpus_sha256,
        "external_requests": 0,
        "limits": "Lexical only; no semantic vectors or banking-task success measured.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(evaluate(parser.parse_args()))
