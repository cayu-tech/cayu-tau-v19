"""Compare explicit search modes using existing development cases, without network IO."""

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from cayu import TextEmbeddingProvider


class OfflineOnly(TextEmbeddingProvider):
    name = "offline-only"

    def __init__(self):
        self.misses = 0

    async def embed_texts(self, request):
        self.misses += 1
        raise RuntimeError("Missing cached vector; external requests are disabled")


async def evaluate(args):
    # tau2's imports otherwise ask LiteLLM to fetch a public price map.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from evaluate_section_embeddings import DIMENSIONS, MODEL, CachedEmbeddings

    from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus

    offline = OfflineOnly()
    cache = CachedEmbeddings(offline)
    manifest = json.loads((args.cache / "manifest.json").read_text())
    cache.restore(args.cache, manifest["corpus_sha256"])
    if args.query_vectors:
        extra = json.loads(args.query_vectors.read_text())
        if extra["model"] != MODEL or extra["dimensions"] != DIMENSIONS:
            raise ValueError("Query vector model/dimension mismatch")
        for text, vector in extra["vectors"].items():
            array = np.asarray(vector, dtype=float)
            if array.shape != (DIMENSIONS,) or not np.isfinite(array).all():
                raise ValueError("Invalid query vector")
            if text in cache.vectors and not np.array_equal(cache.vectors[text], array):
                raise ValueError("Conflicting query vector")
            cache.vectors[text] = vector
    corpus = await BankingKnowledgeCorpus.from_directory_with_embeddings(
        args.documents,
        embedding_provider=cache,
        embedding_model=MODEL,
        embedding_dimensions=DIMENSIONS,
    )
    if corpus.corpus_sha256 != manifest["corpus_sha256"]:
        raise ValueError("Corpus fingerprint mismatch")
    cases = {}
    for path in args.cases:
        manifest_cases = json.loads(path.read_text())
        for case in manifest_cases["cases"] if isinstance(manifest_cases, dict) else manifest_cases:
            previous = cases.get(case["id"])
            if previous is not None and (
                previous["query"] != case["query"]
                or set(previous["required"]) != set(case["required"])
            ):
                raise ValueError("Conflicting duplicate case ID")
            cases[case["id"]] = case
    if not cases:
        raise ValueError("At least one evaluation case is required")
    rows = []
    for case in cases.values():
        required = set(case["required"])
        if not required:
            raise ValueError("Required documents must not be empty")
        for mode in ("lexical", "semantic", "hybrid"):
            misses = offline.misses
            start = time.perf_counter()
            raw = await corpus.search(case["query"], mode=mode, limit=4)
            first = json.loads(raw)
            first_ms = 1000 * (time.perf_counter() - start)
            page = first
            ids = []
            cursors = set()
            while True:
                ids.extend(doc["entry_id"] for doc in page["documents"])
                cursor = page["navigation"]["next_cursor"]
                if cursor is None:
                    break
                if cursor in cursors:
                    raise RuntimeError("Search cursor cycle")
                cursors.add(cursor)
                page = json.loads(
                    await corpus.search(
                        case["query"],
                        mode=mode,
                        limit=4,
                        cursor=cursor,
                    )
                )
            allowed_reasons = {"", "lexical_truncated", "semantic_truncated", "candidate_limit"}
            if mode == "lexical":
                allowed_reasons.add("semantic_unsupported")
            usable = (
                offline.misses == misses
                and bool(first["source_coverage"])
                and all(
                    source["status"] in ("complete", "partial")
                    and set((source["reason"] or "").split("+")) <= allowed_reasons
                    for source in first["source_coverage"]
                )
            )
            row = {
                "id": case["id"],
                "mode": mode,
                "usable": usable,
                "required_ranks": {
                    key: ids.index(key) + 1 if key in ids else None for key in sorted(required)
                },
                "recall": {
                    str(k): len(required.intersection(ids[:k])) / len(required) for k in (4, 6, 16)
                },
                "inventory_recall": len(required.intersection(ids)) / len(required),
                "inventory_documents": len(ids),
                "first_page_ms": first_ms,
                "first_page_bytes": len(raw.encode()),
                "bounds": first["retrieval_bounds"],
                "sources": first["source_coverage"],
                "cache_misses": offline.misses - misses,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "scope": "Existing development queries; not held-out evaluation or benchmark success",
        "corpus_sha256": corpus.corpus_sha256,
        "rows": rows,
        "external_requests": 0,
        "cache_misses": offline.misses,
        "case_file_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.cases],
        "implementation_sha256": {
            name: hashlib.sha256(
                (
                    Path(__file__).resolve().parents[1] / "src" / "cayu_tau3_banking" / name
                ).read_bytes()
            ).hexdigest()
            for name in ("knowledge.py", "search_excerpts.py", "search_paging.py")
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if offline.misses or any(not row["usable"] for row in rows):
        raise RuntimeError("Incomplete evaluation; inspect source coverage and cache misses")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--query-vectors", type=Path)
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(evaluate(parser.parse_args()))
