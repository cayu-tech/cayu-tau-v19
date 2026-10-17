"""Frozen representation ablation. No agent defaults or benchmark tasks are changed.

Run from the repository root with PYTHONPATH=src. --live authorizes embeddings;
without it only the corpus inventory and frozen experiment manifest are produced.
"""

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from cayu import (
    KNOWLEDGE_SEMANTIC_CHANNEL,
    InMemoryEmbeddingKnowledgeStore,
    KnowledgeRecallSource,
    RecallEngine,
    RecallSource,
    RecallSourceResult,
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)
from cayu.providers import OpenAIProvider

from cayu_tau3_banking.document_recall import DocumentRecallSource
from cayu_tau3_banking.embedding_diagnostics import embedding_failure_details
from cayu_tau3_banking.knowledge import (
    BankingKnowledgeCorpus,
    _load_corpus_material,
    _title_body_embedding_chunk,
    _unique_entry_candidates,
)
from cayu_tau3_banking.section_embeddings import VERSION, section_embedding_chunks

MODEL = "text-embedding-3-large"
DIMENSIONS = 3072
DOCUMENTS = Path(".vendor/tau2/data/tau2/domains/banking_knowledge/documents")
SUITES = (
    "public-retrieval-v1",
    "public-retrieval-validation-v1",
    "public-retrieval-fresh-v2",
)


def save_private_json(path, value):
    """Write generated experiment data without importing the simulator stack."""
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


class CachedEmbeddings(TextEmbeddingProvider):
    """One run, one model; reuse exact text vectors across both experimental arms."""

    name = "section-experiment-cache"

    def __init__(self, provider):
        self.provider = provider
        self.vectors = {}
        self.input_tokens = 0
        self.calls = 0

    def restore(self, directory, corpus_sha256):
        """Reuse only validated, same-model public vectors from a previous attempt."""
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest["model"] != MODEL
            or manifest["dimensions"] != DIMENSIONS
            or manifest["corpus_sha256"] != corpus_sha256
        ):
            raise ValueError("Vector cache model, dimensions, or corpus mismatch.")
        texts = json.loads((directory / "vector-texts.json").read_text())
        with np.load(directory / "vectors.npz", allow_pickle=False) as archive:
            vectors = archive["vectors"]
        if vectors.shape != (len(texts), DIMENSIONS) or not np.isfinite(vectors).all():
            raise ValueError("Invalid cached vector shape or values.")
        restored = {}
        for row, vector in zip(texts, vectors, strict=True):
            text = row["text"]
            if hashlib.sha256(text.encode()).hexdigest() != row["sha256"] or text in restored:
                raise ValueError("Invalid cached text hash or duplicate text.")
            restored[text] = vector.tolist()
        self.vectors.update(restored)

    async def embed_texts(self, request):
        if request.model != MODEL or request.dimensions != DIMENSIONS or request.options:
            raise ValueError("Unexpected embedding configuration.")
        missing = list(dict.fromkeys(t for t in request.texts if t not in self.vectors))
        for start in range(0, len(missing), 64):
            batch = missing[start : start + 64]
            result = await self.provider.embed_texts(
                TextEmbeddingRequest(model=MODEL, dimensions=DIMENSIONS, texts=batch)
            )
            if result.model != MODEL or {e.index for e in result.embeddings} != set(
                range(len(batch))
            ):
                raise ValueError("Embedding response identity/count mismatch.")
            for embedding in result.embeddings:
                if len(embedding.vector) != DIMENSIONS:
                    raise ValueError("Embedding response dimension mismatch.")
                self.vectors[batch[embedding.index]] = embedding.vector
            self.calls += 1
            self.input_tokens += (result.usage.input_tokens or 0) if result.usage else 0
            print(
                json.dumps(
                    {
                        "embedding_calls": self.calls,
                        "cached_texts": len(self.vectors),
                        "input_tokens": self.input_tokens,
                    }
                ),
                flush=True,
            )
        return TextEmbeddingResult(
            model=MODEL,
            embeddings=[
                TextEmbedding(index=i, vector=self.vectors[text])
                for i, text in enumerate(request.texts)
            ],
        )


class FrozenSource(RecallSource):
    def __init__(self, source, result):
        super().__init__(required=source.required, candidate_limit=source.candidate_limit)
        self.name, self.channel_names = source.name, source.channel_names
        self.result = result

    async def retrieve(self, situation):
        return self.result


def replace_semantic(baseline, section):
    """Change only the ordinary semantic channel; freeze every other channel."""
    channels = tuple(
        next(c for c in section.channels if c.channel == KNOWLEDGE_SEMANTIC_CHANNEL)
        if c.channel == KNOWLEDGE_SEMANTIC_CHANNEL
        else c
        for c in baseline.channels
    )
    records = {r.identity.sort_key(): r for r in (*baseline.records, *section.records)}
    needed = {h.identity.sort_key() for c in channels for h in c.hits}
    reasons = sorted(
        {
            reason
            for result in (baseline, section)
            for reason in (result.partial_reason or "").split("+")
            if reason
        }
    )
    return RecallSourceResult(
        source=baseline.source,
        channels=channels,
        records=tuple(records[k] for k in sorted(needed)),
        coverage_complete=baseline.coverage_complete and section.coverage_complete,
        partial_reason="+".join(reasons) or None,
    )


def require_usable_source(result):
    """Bounded top-k truncation is expected; execution/index failures are not."""
    if result.coverage_complete:
        return
    reasons = set((result.partial_reason or "").split("+"))
    if not reasons <= {"lexical_truncated", "semantic_truncated"}:
        raise RuntimeError(f"Recall source failed: {result.source}: {result.partial_reason}")


def rank_documents(chunks, vectors, query):
    q = np.asarray(vectors[query], dtype=np.float64)
    rows = []
    for entry_id, parts in chunks.items():
        scores = []
        for part in parts:
            v = np.asarray(vectors[part.text], dtype=np.float64)
            scores.append(float(np.dot(v, q) / (np.linalg.norm(v) * np.linalg.norm(q))))
        best = max(range(len(parts)), key=lambda i: scores[i])
        rows.append(
            {
                "entry_id": entry_id,
                "cosine": scores[best],
                "normalized_score": (scores[best] + 1) / 2,
                "chunk_id": parts[best].id,
                "text": parts[best].text,
            }
        )
    return sorted(rows, key=lambda r: (-r["cosine"], r["entry_id"]))


async def run(root, live, resume_vectors=None):
    material = _load_corpus_material(DOCUMENTS)
    whole = {e.id: [_title_body_embedding_chunk(e)] for e in material.entries}
    sections = {e.id: section_embedding_chunks(e) for e in material.entries}
    cases = [
        dict(case, suite=suite)
        for suite in SUITES
        for case in json.loads(Path(f"evals/{suite}.json").read_text())["cases"]
    ]
    old = json.loads(
        Path("artifacts/tier-ranking-diagnostic/run-_1tirrky/manifest.json").read_text()
    )["cases"][0]
    cases.append(dict(old, suite="observed-regression"))
    texts = list(
        dict.fromkeys(
            p.text for group in (whole, sections) for parts in group.values() for p in parts
        )
    )
    # Corpus-only preflight: no silent truncation or token-limit-dependent splitting.
    # UTF-8 byte counts conservatively bound byte-level token counts; no tokenizer
    # download is needed. Public banking inputs must fit without truncation.
    counts = [len(text.encode("utf-8")) for text in texts]
    if max(counts) > 8000 or sum(counts) > 4_000_000:
        raise ValueError("Frozen embedding input budget exceeded.")
    manifest = {
        "version": VERSION,
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "corpus_sha256": material.corpus_sha256,
        "cases": cases,
        "whole_chunks": sum(map(len, whole.values())),
        "section_chunks": sum(map(len, sections.values())),
        "unique_embedding_texts": len(texts),
        "embedding_input_token_upper_bound": sum(counts),
        "max_input_token_upper_bound": max(counts),
        "controls": "Same queries/model/weights/limits. Canonical document identities in BOTH paired arms. All lexical and offer-facet channels frozen from whole-document control. Only ordinary semantic channel replaced. Raw original fusion is a separate reference, not the paired control.",
        "gate": "No per-query final recall@4 regression versus paired whole-document control; observed missing evidence must improve. No retuning or automatic promotion. Retrieval only, not a tau task pass.",
        "limitations": "Existing small development suites, not blind validation. No automatic admission or runtime test. Facet semantic representation intentionally held constant.",
    }
    save_private_json(root / "manifest.json", manifest)
    save_private_json(
        root / "projections.json",
        {
            name: {key: [p.model_dump(mode="json") for p in parts] for key, parts in group.items()}
            for name, group in (("whole", whole), ("sections", sections))
        },
    )
    print(json.dumps(manifest), flush=True)
    if not live:
        return
    provider = OpenAIProvider(timeout_s=60)
    cached = CachedEmbeddings(provider)
    try:
        if resume_vectors is not None:
            cached.restore(resume_vectors, material.corpus_sha256)
            save_private_json(
                root / "cache-origin.json",
                {
                    "directory": str(resume_vectors.resolve()),
                    "reused_texts": len(cached.vectors),
                    "previous_usage_not_included_in_this_run": True,
                },
            )
        await cached.embed_texts(
            TextEmbeddingRequest(model=MODEL, dimensions=DIMENSIONS, texts=texts)
        )
        corpus = await BankingKnowledgeCorpus.from_directory_with_embeddings(
            DOCUMENTS,
            embedding_provider=cached,
            embedding_model=MODEL,
            embedding_dimensions=DIMENSIONS,
        )
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=material.scope,
            embedding_provider=cached,
            embedding_model=MODEL,
            embedding_dimensions=DIMENSIONS,
        )
        for entry in material.entries:
            await store.create_entry(entry, chunks=sections[entry.id], access_scope=material.scope)
        cursor = None
        indexed = 0
        while True:
            result = await store.backfill_embeddings(
                access_scope=material.scope, limit=500, cursor=cursor
            )
            if result.failed_records:
                raise RuntimeError("Section store backfill failed.")
            indexed += result.indexed_records
            if result.next_cursor is None:
                break
            if result.next_cursor == cursor:
                raise RuntimeError("Section backfill cursor did not advance.")
            cursor = result.next_cursor
        if indexed != manifest["section_chunks"]:
            raise RuntimeError("Section backfill incomplete.")
        source = KnowledgeRecallSource(
            store,
            candidate_limit=16,
            max_bytes=96_000,
            max_record_bytes=8000,
            semantic_timeout_seconds=10,
        )
        rows = []
        for case in cases:
            situation = corpus._situation(case["query"])
            for entry_id, snippet in case["required"].items():
                assert snippet in corpus._entries[entry_id].text
            raw = [await s.retrieve(situation) for s in corpus.recall_sources]
            section_raw = await source.retrieve(situation)
            for result in (*raw, section_raw):
                require_usable_source(result)
            treatment = [raw[0], replace_semantic(raw[1], section_raw), *raw[2:]]
            profiles = {}
            for name, results in (("original", raw), ("whole", raw), ("sections", treatment)):
                frozen = tuple(
                    FrozenSource(s, r) for s, r in zip(corpus.recall_sources, results, strict=True)
                )
                sources = (
                    frozen
                    if name == "original"
                    else tuple(DocumentRecallSource(s, corpus.store) for s in frozen)
                )
                result = await RecallEngine(
                    sources, fusion_config=corpus.fusion_config, config=corpus.engine_config
                ).recall(situation)
                ids = [
                    c.record.locator["entry_id"]
                    for c in _unique_entry_candidates(result.candidates, limit=16)
                ]
                required = set(case["required"])
                profiles[name] = {
                    "ids": ids,
                    "recall4": len(required.intersection(ids[:4])) / len(required),
                }
            rankings = {
                name: rank_documents(group, cached.vectors, situation.retrieval_text())
                for name, group in (("whole", whole), ("sections", sections))
            }
            semantic = {}
            for name, ranks in rankings.items():
                semantic[name] = {
                    "recall16": len(
                        set(case["required"]).intersection(r["entry_id"] for r in ranks[:16])
                    )
                    / len(case["required"]),
                    "required": {
                        entry_id: {"rank": i + 1, **r}
                        for i, r in enumerate(ranks)
                        for entry_id in case["required"]
                        if r["entry_id"] == entry_id
                    },
                }
            row = {
                "id": case["id"],
                "suite": case["suite"],
                "profiles": profiles,
                "semantic": semantic,
            }
            rows.append(row)
            save_private_json(
                root / f"{case['suite']}--{case['id']}.json",
                {
                    "row": row,
                    "rankings": rankings,
                    "sources": [r.model_dump(mode="json") for r in raw],
                    "section_source": section_raw.model_dump(mode="json"),
                },
            )
            print(
                json.dumps(
                    {
                        "case": case["id"],
                        "recall4": {n: p["recall4"] for n, p in profiles.items()},
                        "semantic_recall16": {n: v["recall16"] for n, v in semantic.items()},
                    }
                ),
                flush=True,
            )
        groups = {
            suite: {
                name: sum(r["profiles"][name]["recall4"] for r in rows if r["suite"] == suite)
                / sum(r["suite"] == suite for r in rows)
                for name in ("original", "whole", "sections")
            }
            for suite in (*SUITES, "observed-regression")
        }
        regressions = [
            r["id"]
            for r in rows
            if r["profiles"]["sections"]["recall4"] < r["profiles"]["whole"]["recall4"]
        ]
        save_private_json(
            root / "summary.json",
            {"rows": rows, "groups": groups, "regressions": regressions, "default_changed": False},
        )
        print(json.dumps({"groups": groups, "regressions": regressions}), flush=True)
    except Exception as error:
        save_private_json(root / "failure.json", embedding_failure_details(error))
        print(json.dumps({"stopped": embedding_failure_details(error)}), flush=True)
        raise SystemExit(1) from None
    finally:
        # Public-text/vector artifacts only: no credential or provider response body.
        items = list(cached.vectors.items())
        save_private_json(
            root / "vector-texts.json",
            [{"text": t, "sha256": hashlib.sha256(t.encode()).hexdigest()} for t, _ in items],
        )
        np.savez_compressed(
            root / "vectors.npz", vectors=np.asarray([v for _, v in items], dtype=np.float64)
        )
        save_private_json(
            root / "usage.json", {"calls": cached.calls, "input_tokens": cached.input_tokens}
        )
        await provider.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--resume-vectors", type=Path)
    args = parser.parse_args()
    parent = Path("artifacts/section-embedding-validation")
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    print(root, flush=True)
    asyncio.run(run(root, args.live, args.resume_vectors))
