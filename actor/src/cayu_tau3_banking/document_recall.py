"""Opt-in document-oriented recall candidate; the passing profile is unchanged."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from cayu import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    RecallEngine,
    RecallRecord,
    RecallSituation,
    RecallSource,
    RecallSourceResult,
    RetrievalCandidateIdentity,
    WeightedReciprocalRankFusionConfig,
)

if TYPE_CHECKING:
    from cayu import KnowledgeStore

    from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus

DOCUMENT_FUSION_VERSION = "cayu.tau3.document-balanced.v1"


class DocumentRecallSource(RecallSource):
    """Project bounded non-pageable hits to authoritative document revisions.

    Only the public immutable banking corpus uses this adapter. Chunk matches
    nominate a document; the document body is read again with the request's access
    scope. Original match identity/hash are retained in fusion explanations, not
    misrepresented as hashes of the expanded document. No cross-request cache.
    """

    def __init__(self, source: RecallSource, store: KnowledgeStore) -> None:
        if source.continuation_channels:
            raise ValueError("Document projection does not support pageable sources.")
        super().__init__(required=source.required, candidate_limit=source.candidate_limit)
        self.name = source.name
        self.channel_names = source.channel_names
        self._source = source
        self._store = store

    async def retrieve(self, situation: RecallSituation) -> RecallSourceResult:
        if situation.knowledge_access_scope is None:
            raise ValueError("Document projection requires an access scope.")
        result = await self._source.retrieve(situation)
        records = {r.identity.sort_key(): r for r in result.records}
        canonical: dict[tuple[str, int], RecallRecord] = {}
        channels = []
        for channel in result.channels:
            seen = set()
            hits = []
            for hit in sorted(channel.hits, key=lambda h: h.rank):
                original = records[hit.identity.sort_key()]
                entry_id = original.locator.get("entry_id")
                revision = original.locator.get("entry_revision")
                if type(entry_id) is not str or type(revision) is not int:
                    raise ValueError("Document projection requires an exact entry revision.")
                key = (entry_id, revision)
                if key in seen:
                    continue
                seen.add(key)
                if key not in canonical:
                    entry = await self._store.get_entry(
                        entry_id,
                        revision=revision,
                        max_bytes=64_000,
                        access_scope=situation.knowledge_access_scope,
                    )
                    if entry is None:
                        raise ValueError("Projected document revision is unavailable or denied.")
                    if entry.id != entry_id or entry.revision != revision:
                        raise ValueError("Projected document revision does not match the hit.")
                    text = entry.text.encode("utf-8")[:8_000].decode("utf-8", errors="ignore")
                    canonical[key] = RecallRecord(
                        identity=RetrievalCandidateIdentity(
                            record_type="knowledge_entry",
                            record_id=entry.id,
                            revision=str(entry.revision),
                        ),
                        representation="entry_text",
                        text=text,
                        text_complete=text == entry.text,
                        content_hash=hashlib.sha256(entry.text.encode("utf-8")).hexdigest(),
                        title=(
                            entry.title.encode("utf-8")[:1024].decode("utf-8", errors="ignore")
                            or None
                        )
                        if entry.title
                        else None,
                        locator={"entry_id": entry.id, "entry_revision": entry.revision},
                    )
                record = canonical[key]
                hits.append(
                    hit.model_copy(
                        update={
                            "identity": record.identity,
                            "representation": record.representation,
                            "content_hash": record.content_hash,
                            "explanations": (
                                *hit.explanations,
                                f"document_projection source={hit.identity.record_type}:{hit.identity.record_id} "
                                f"revision={hit.identity.revision} sha256={hit.content_hash}",
                            ),
                        }
                    )
                )
            channels.append(
                channel.model_copy(
                    update={
                        "hits": tuple(hits),
                        "index_version": channel.index_version + ":documents-v1",
                    }
                )
            )
        return RecallSourceResult(
            source=result.source,
            channels=tuple(channels),
            records=tuple(canonical[key] for key in sorted(canonical)),
            coverage_complete=result.coverage_complete,
            partial_reason=result.partial_reason,
        )


def document_fusion_candidate(corpus: BankingKnowledgeCorpus) -> RecallEngine:
    """Build the fixed candidate for comparative evaluation, without mutating corpus."""
    from cayu_tau3_banking.knowledge import (
        BANKING_BM25_CHANNEL,
        BANKING_OFFER_FACET_LEXICAL_CHANNEL,
        BANKING_OFFER_FACET_SEMANTIC_CHANNEL,
    )

    if corpus.retrieval_mode != "hybrid":
        raise ValueError("The document-fusion candidate requires the hybrid corpus.")
    weights = {
        BANKING_BM25_CHANNEL: 1.0,
        KNOWLEDGE_LEXICAL_CHANNEL: 0.5,
        KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
        BANKING_OFFER_FACET_LEXICAL_CHANNEL: 0.25,
        BANKING_OFFER_FACET_SEMANTIC_CHANNEL: 0.5,
    }
    config = WeightedReciprocalRankFusionConfig(
        configuration_version=DOCUMENT_FUSION_VERSION,
        channel_weights=weights,
        max_candidates_per_channel=16,
        fused_head_limit=16,
    )
    return RecallEngine(
        tuple(DocumentRecallSource(source, corpus.store) for source in corpus.recall_sources),
        fusion_config=config,
        config=corpus.engine_config,
    )
