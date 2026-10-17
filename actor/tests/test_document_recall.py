from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from cayu import (
    RecallSource,
    RetrievalCandidateIdentity,
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingResult,
)

from cayu_tau3_banking.document_recall import DocumentRecallSource, document_fusion_candidate
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus


class Embeddings(TextEmbeddingProvider):
    name = "document-test"

    async def embed_texts(self, request):
        return TextEmbeddingResult(
            model=request.model,
            embeddings=[
                TextEmbedding(index=i, vector=[1.0, 0.0]) for i in range(len(request.texts))
            ],
        )


async def corpus(path: Path):
    (path / "example.json").write_text(
        json.dumps(
            {"id": "example", "title": "Parcel service", "content": "A parcel requires postage."}
        )
    )
    return await BankingKnowledgeCorpus.from_directory_with_embeddings(
        path,
        embedding_provider=Embeddings(),
        embedding_model="document-test",
        embedding_dimensions=2,
    )


def test_document_votes_merge_without_changing_the_passing_engine(tmp_path):
    async def check():
        memory = await corpus(tmp_path)
        original_engine = memory.engine
        original_config = memory.fusion_config.model_dump()
        result = await document_fusion_candidate(memory).recall(memory._situation("parcel postage"))
        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert candidate.fused.channel_count == 3
        assert candidate.record.identity.record_type == "knowledge_entry"
        assert candidate.record.identity.revision == "1"
        assert (
            candidate.record.content_hash
            == hashlib.sha256(b"A parcel requires postage.").hexdigest()
        )
        assert candidate.record.text_complete
        assert any(
            "knowledge_chunk:example:r1:0" in explanation
            for match in candidate.fused.matches
            for explanation in match.explanations
        )
        assert memory.engine is original_engine
        assert memory.fusion_config.model_dump() == original_config
        assert candidate.fused.score == pytest.approx(2.5 / 61)

    asyncio.run(check())


def test_projection_requires_scope_and_rechecks_authoritative_access(tmp_path):
    async def check():
        memory = await corpus(tmp_path)
        source = memory.recall_sources[0]
        wrapper = DocumentRecallSource(source, memory.store)
        situation = memory._situation("parcel")
        with pytest.raises(ValueError, match="access scope"):
            await wrapper.retrieve(situation.model_copy(update={"knowledge_access_scope": None}))
        calls = []

        async def denied(entry_id, **kwargs):
            calls.append((entry_id, kwargs))
            return None

        wrapper = DocumentRecallSource(source, SimpleNamespace(get_entry=denied))
        with pytest.raises(ValueError, match="unavailable or denied"):
            await wrapper.retrieve(situation)
        assert calls[0][0] == "example"
        assert calls[0][1]["revision"] == 1
        assert calls[0][1]["access_scope"] == situation.knowledge_access_scope

    asyncio.run(check())


def test_projection_rejects_pageable_sources(tmp_path):
    async def check():
        memory = await corpus(tmp_path)
        source = memory.recall_sources[0]
        source.continuation_channels = source.channel_names
        with pytest.raises(ValueError, match="pageable"):
            DocumentRecallSource(source, memory.store)

    asyncio.run(check())


def test_multiple_chunks_do_not_multiply_one_channels_vote(tmp_path):
    async def check():
        memory = await corpus(tmp_path)
        source = memory.recall_sources[0]
        situation = memory._situation("parcel")
        raw = await source.retrieve(situation)
        records, hits = [], []
        for rank in (2, 1):
            identity = RetrievalCandidateIdentity(
                record_type="knowledge_chunk", record_id=f"example:chunk:{rank}", revision="1"
            )
            records.append(
                raw.records[0].model_copy(
                    update={
                        "identity": identity,
                        "locator": {
                            "entry_id": "example",
                            "entry_revision": 1,
                            "chunk_id": identity.record_id,
                        },
                    }
                )
            )
            hits.append(
                raw.channels[0].hits[0].model_copy(update={"identity": identity, "rank": rank})
            )
        packet = raw.model_copy(
            update={
                "records": tuple(records),
                "channels": (
                    raw.channels[0].model_copy(update={"hits": tuple(hits), "truncated": True}),
                ),
                "coverage_complete": False,
                "partial_reason": "candidate_limit",
            }
        )

        class Frozen(RecallSource):
            name, channel_names = source.name, source.channel_names

            async def retrieve(self, situation):
                return packet

        projected = await DocumentRecallSource(
            Frozen(required=True, candidate_limit=16),
            memory.store,
        ).retrieve(situation)
        assert len(projected.records) == 1
        assert len(projected.channels[0].hits) == 1
        assert projected.channels[0].hits[0].rank == 1
        assert projected.channels[0].truncated
        assert not projected.coverage_complete
        assert projected.partial_reason == "candidate_limit"

    asyncio.run(check())


def test_projection_rejects_wrong_revision_from_store(tmp_path):
    async def check():
        memory = await corpus(tmp_path)
        entry = await memory.store.get_entry("example", revision=1, access_scope=memory.scope)

        async def wrong_revision(*args, **kwargs):
            return entry.model_copy(update={"revision": 2})

        wrapper = DocumentRecallSource(
            memory.recall_sources[0], SimpleNamespace(get_entry=wrong_revision)
        )
        with pytest.raises(ValueError, match="does not match"):
            await wrapper.retrieve(memory._situation("parcel"))

    asyncio.run(check())
