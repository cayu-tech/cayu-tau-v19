"""Offline public-document retrieval diagnostics; never imported by the acting agent.

Required-evidence labels are deliberately not exhaustive relevance judgments. A
scope-prefix miss is a coarse diagnostic, not proof that a document is irrelevant.
Date cases measure delivery of the source window, not temporal reasoning accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus


class EvidenceValidationError(ValueError):
    """Source-contract failure with payload-free diagnostic fields."""

    def __init__(self, *, case_index: int, result_rank: int, checks: dict[str, bool]):
        super().__init__("Retrieved evidence is not the complete canonical document.")
        self.case_index = case_index
        self.result_rank = result_rank
        self.checks = dict(checks)


def safe_error_diagnostic(error: BaseException) -> dict[str, Any]:
    """Capture locations, never exception messages, source lines, or local variables."""

    from cayu_tau3_banking.embedding_diagnostics import EmbeddingInitializationError
    from cayu_tau3_banking.failure_categories import classify_exception

    frames = []
    trace = error.__traceback__
    while trace is not None:
        frames.append(
            {
                "file": Path(trace.tb_frame.f_code.co_filename).name,
                "line": trace.tb_lineno,
                "function": trace.tb_frame.f_code.co_name,
            }
        )
        trace = trace.tb_next
    diagnostic: dict[str, Any] = {
        "error_type": type(error).__name__,
        "frames": frames,
        "failure_category": classify_exception(error),
    }
    if isinstance(error, EmbeddingInitializationError):
        diagnostic["embedding_failure"] = error.diagnostic
    if isinstance(error, EvidenceValidationError):
        diagnostic.update(
            case_index=error.case_index,
            result_rank=error.result_rank,
            checks=error.checks,
        )
    return diagnostic


def validate_suite(suite: dict[str, Any], documents: dict[str, dict[str, str]]) -> None:
    if set(suite) != {"version", "corpus_sha256", "provenance", "cases"}:
        raise ValueError("Unexpected suite fields; only public retrieval labels are supported.")
    if not isinstance(suite["cases"], list) or not suite["cases"]:
        raise ValueError("Cases must be a nonempty list.")
    seen = set()
    for case in suite["cases"]:
        required_fields = {"id", "group", "query", "required"}
        if not required_fields <= case.keys() or case.keys() - required_fields - {"scope_prefix"}:
            raise ValueError("Invalid case fields.")
        for field in ("id", "group", "query"):
            if not isinstance(case[field], str) or not case[field].strip():
                raise ValueError(f"Invalid {field}.")
        if case["id"] in seen:
            raise ValueError("Duplicate case id.")
        seen.add(case["id"])
        if not isinstance(case["required"], dict) or not case["required"]:
            raise ValueError("Each case needs required public evidence.")
        if "scope_prefix" in case and (
            not isinstance(case["scope_prefix"], str) or not case["scope_prefix"]
        ):
            raise ValueError("Invalid scope prefix.")
        for ident, quote in case["required"].items():
            if ident not in documents or not isinstance(quote, str) or not quote:
                raise ValueError("Unknown document or invalid evidence quote.")
            if quote not in documents[ident]["content"]:
                raise ValueError(f"Evidence quote absent from public document {ident}.")


def score_ids(ids: list[str], required: set[str], *, k: int) -> dict[str, float]:
    if type(k) is not int or k < 1 or not required:
        raise ValueError("Positive k and nonempty required evidence are necessary.")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate retrieved identities.")
    selected = ids[:k]
    found = set(selected) & required
    first = next((i for i, ident in enumerate(selected, 1) if ident in required), None)
    return {
        "required_recall": len(found) / len(required),
        "all_required": float(found == required),
        "reciprocal_rank": 0.0 if first is None else 1.0 / first,
    }


async def evaluate(
    corpus: BankingKnowledgeCorpus,
    suite: dict[str, Any],
    documents: dict[str, dict[str, str]],
    *,
    k: int = 4,
) -> dict[str, Any]:
    validate_suite(suite, documents)
    if suite["corpus_sha256"] != corpus.corpus_sha256:
        raise ValueError("Corpus fingerprint differs from the frozen suite.")
    if type(k) is not int or not 1 <= k <= corpus.max_tool_documents:
        raise ValueError("k exceeds the agent's public search limit.")
    rows = []
    for case_index, case in enumerate(suite["cases"]):
        started = time.perf_counter()
        content = await asyncio.wait_for(corpus.search(case["query"], limit=k), timeout=30)
        elapsed = (time.perf_counter() - started) * 1000
        result = json.loads(content)
        returned = result["documents"]
        ids = [d["entry_id"] for d in returned]
        if len(ids) > k:
            raise ValueError("Retriever exceeded requested result bound.")
        for result_rank, entry in enumerate(returned, 1):
            source = documents[entry["entry_id"]]["content"].strip()
            checks = {
                "canonical_text_matches": entry["text"] == source,
                "canonical_hash_matches": entry["content_sha256"]
                == hashlib.sha256(source.encode()).hexdigest(),
                "text_complete": entry["text_complete"] is True,
            }
            if not all(checks.values()):
                raise EvidenceValidationError(
                    case_index=case_index, result_rank=result_rank, checks=checks
                )
        # A fresh, stateless automatic-recall probe, not a durable-session exposure test.
        automatic = await asyncio.wait_for(corpus.recall(case["query"]), timeout=30)
        required = set(case["required"])
        prefix = case.get("scope_prefix")
        rows.append(
            {
                "id": case["id"],
                "group": case["group"],
                "query": case["query"],
                "required_ids": sorted(required),
                "returned_ids": ids,
                "missing_ids": sorted(required - set(ids)),
                **score_ids(ids, required, k=k),
                "required_quote_coverage": sum(
                    any(d["entry_id"] == ident and quote in d["text"] for d in returned)
                    for ident, quote in case["required"].items()
                )
                / len(required),
                "off_scope_fraction": (
                    sum(not ident.startswith(prefix) for ident in ids) / len(ids)
                    if prefix and ids
                    else None
                ),
                "automatic_ids": list(automatic.record_ids),
                "automatic_required_recall": len(set(automatic.record_ids) & required)
                / len(required),
                "source_coverage": result["source_coverage"],
                "search_ms": elapsed,
            }
        )
    return {
        "schema_version": 1,
        "suite": suite["version"],
        "suite_sha256": hashlib.sha256(json.dumps(suite, sort_keys=True).encode()).hexdigest(),
        "corpus_sha256": corpus.corpus_sha256,
        "document_count": corpus.document_count,
        "python_version": platform.python_version(),
        "retriever_source_sha256": hashlib.sha256(
            Path(__file__).with_name("knowledge.py").read_bytes()
        ).hexdigest(),
        "retrieval_mode": corpus.retrieval_mode,
        "fusion_configuration": corpus.fusion_configuration_version,
        "k": k,
        "metrics": {
            key: statistics.mean(row[key] for row in rows)
            for key in (
                "required_recall",
                "all_required",
                "reciprocal_rank",
                "required_quote_coverage",
                "automatic_required_recall",
            )
        },
        "median_search_ms": statistics.median(row["search_ms"] for row in rows),
        "interpretation": "Public-document development diagnostics; not task success, not held-out, not exhaustive relevance labels. Automatic recall is stateless and does not prove live exposure. Dates are source-delivery checks, not generated-answer checks.",
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new evidence path.")
    suite = json.loads(args.suite.read_text())
    documents = {
        d["id"]: d for path in args.documents.glob("*.json") if (d := json.loads(path.read_text()))
    }
    corpus = BankingKnowledgeCorpus.from_directory(args.documents)
    report = asyncio.run(evaluate(corpus, suite, documents, k=args.k))
    # Exclusive output prevents replacing earlier experimental evidence.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"]}))


if __name__ == "__main__":
    main()
