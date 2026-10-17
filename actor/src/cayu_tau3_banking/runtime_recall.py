"""Native recall configuration and delivered-evidence projection for banking."""

from __future__ import annotations

from cayu import (
    AutomaticRecallContextPolicy,
    AutomaticRecallSourceConfig,
    AutomaticRecallSourceDescriptor,
    AutomaticRecallSourceRegistration,
    MemoryDeltaPolicy,
)
from cayu.memory.evidence import ContextExposureState, RecallEvidenceQuery, RecallItemAdmission

from cayu_tau3_banking.knowledge import BANKING_KNOWLEDGE_NAMESPACE, BankingKnowledgeCorpus


def banking_recall_policy(corpus: BankingKnowledgeCorpus, base_policy):
    registrations = []
    for source in corpus.recall_sources:
        if source.name == "knowledge":
            continue

        async def factory(context, source=source):
            if context.knowledge_store is not corpus.store:
                raise ValueError("Banking recall source belongs to another corpus.")
            # Corpus/index is immutable after ingestion. Each source honors the
            # native RecallSituation access scope; no query state is retained.
            return source

        registrations.append(
            AutomaticRecallSourceRegistration(
                descriptor=AutomaticRecallSourceDescriptor(
                    name=source.name,
                    channel_names=source.channel_names,
                    continuation_channels=source.continuation_channels,
                    required=source.required,
                    candidate_limit=source.candidate_limit,
                    configuration_version=f"banking-native-v1:{corpus.corpus_sha256}:{corpus.fusion_configuration_version}",
                ),
                factory=factory,
            )
        )
    knowledge = next(source for source in corpus.recall_sources if source.name == "knowledge")
    return AutomaticRecallContextPolicy(
        base_policy,
        admission_policy=corpus.admission_policy,
        fusion_config=corpus.fusion_config,
        engine_config=corpus.engine_config,
        sources=AutomaticRecallSourceConfig(
            include_transcript=False,
            knowledge_namespace=BANKING_KNOWLEDGE_NAMESPACE,
            knowledge_candidate_limit=knowledge.candidate_limit,
            knowledge_max_bytes=96_000,
            knowledge_max_record_bytes=8_000,
            semantic_timeout_seconds=30.0 if corpus.retrieval_mode == "hybrid" else 1.0,
        ),
        custom_sources=registrations,
        # Custom sources have no change-feed contract. Recovery only revalidates
        # exact, previously exposed knowledge against the canonical store.
        delta_policy=MemoryDeltaPolicy(
            refresh_on_knowledge_change=False,
            reanchor_on_projection_loss=True,
            reanchor_relevance_policy="cayu.query_concepts.v5",
        ),
    )


async def observe_delivered_recall(state) -> None:
    """Bind banking evidence only to the runtime's actual model-step exposure.

    Called before tool authorization and reply admission, never from a source
    factory or merely because a candidate was retrieved.
    """
    if not state.native_recall or state.model_step_id is None:
        return
    page = await state.session_store.list_context_exposures(
        RecallEvidenceQuery(
            session_id=state.session_id, model_step_id=state.model_step_id, limit=20
        )
    )
    if page.next_cursor is not None:
        raise RuntimeError("Unexpected recall exposure fan-out.")
    checkpoint = await state.session_store.load_checkpoint(state.session_id)
    frame = (checkpoint or {}).get("automatic_recall") or {}
    for exposure in page.items:
        if exposure.exposure_id in state.observed_recall_exposures:
            continue
        if exposure.state not in {
            ContextExposureState.ACKNOWLEDGED,
            ContextExposureState.COMPLETED,
        }:
            continue
        items = await state.session_store.load_recall_item_exposures(
            state.session_id, exposure.exposure_id
        )
        for item in items:
            if item.admission is RecallItemAdmission.ADMITTED:
                section, text_key = "focus", "text"
            elif item.admission is RecallItemAdmission.OFFERED:
                section, text_key = "offer", "preview"
            else:
                continue
            if item.receipt_id == frame.get("receipt_id"):
                projection = frame.get("projection") or {}
                candidates = (projection.get(section) or {}).get("items", [])
            else:
                deltas = [
                    delta
                    for delta in (frame.get("delta_state") or {}).get("deltas", [])
                    if delta.get("receipt_id") == item.receipt_id
                ]
                if len(deltas) != 1 or section != "focus":
                    raise RuntimeError("Delivered banking recall lost its exact projection.")
                # Suppressed/expired placements have no projection. Historical
                # receipt identity alone must never become current exposure.
                projection = deltas[0].get("projection") or {}
                candidates = projection.get("items", [])
            matching = next(
                (
                    value
                    for value in candidates
                    if value["identity"] == item.identity.model_dump(mode="json")
                    and value["content_hash"] == item.content_sha256
                ),
                None,
            )
            if matching is None:
                raise RuntimeError("Delivered banking recall lacks its exact projected text.")
            locator = item.locator
            if locator.kind not in {"knowledge_entry", "knowledge_chunk"}:
                continue
            delivered_text = matching.get(text_key)
            if section == "offer" and matching.get("preview_complete") is not True:
                continue
            if not isinstance(delivered_text, str) or not delivered_text:
                continue
            reference = state.memory.corpus.delivered_revision_evidence(
                locator.entry_id,
                locator.entry_revision,
                delivered_text=delivered_text,
            )
            # Recall ranking admission is not a statement of source completeness.
            # A short, fully exposed offer can be the entire immutable document.
            # Never upgrade a locator, redacted preview or partial chunk to full
            # evidence; the corpus verifies the exact bytes independently.
            if section == "offer" and reference is None:
                continue
            if reference is not None:
                state.controller.observe_knowledge_evidence((reference,))
            state.admission.observe_knowledge_search()
            state.admission.observe_knowledge_text(delivered_text)
        state.observed_recall_exposures.add(exposure.exposure_id)
