import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from cayu import WorkEvidenceReference
from cayu.memory.evidence import ContextExposureState, RecallItemAdmission
from test_outcome_controller import outcome, setup_plan, update

from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus, BankingKnowledgeError
from cayu_tau3_banking.runtime_recall import observe_delivered_recall


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "policy.json").write_text(
        json.dumps(
            {
                "id": "policy",
                "title": "Example policy",
                "content": "The requested action requires consent.",
            }
        )
    )
    return BankingKnowledgeCorpus.from_directory(tmp_path)


BODY = "The requested action requires consent."
CHUNK = "Example policy\n\n" + BODY


def test_full_representations_share_canonical_digest_without_promoting_partial_text(corpus):
    canonical = corpus.delivered_revision_evidence("policy", 1, delivered_text=BODY)
    assert canonical.digest == hashlib.sha256(BODY.encode()).hexdigest()
    assert corpus.delivered_revision_evidence("policy", 1, delivered_text=CHUNK) == canonical
    for text in (BODY[:10], CHUNK[:15], "unrelated", ""):
        assert corpus.delivered_revision_evidence("policy", 1, delivered_text=text) is None
    with pytest.raises(BankingKnowledgeError, match="pinned revision"):
        corpus.delivered_revision_evidence("policy", 2, delivered_text=BODY)


def test_native_chunk_recall_does_not_invalidate_previously_accepted_body_citation(corpus):
    controller, _ = setup_plan(outcome())
    canonical = corpus.delivered_revision_evidence("policy", 1, delivered_text=BODY)
    controller.observe_knowledge_evidence((canonical,))
    refs = [
        {
            "kind": "knowledge_revision",
            "reference_id": "policy",
            "version": "1",
            "digest": canonical.digest,
        }
    ]
    assert not update(
        controller, status="awaiting_customer", evidence="Need consent", evidence_refs=refs
    ).is_error
    identity = {"record_id": "policy:r1:0", "record_type": "knowledge_chunk", "revision": "1"}
    chunk_digest = hashlib.sha256(CHUNK.encode()).hexdigest()
    exposure = SimpleNamespace(exposure_id="exposure", state=ContextExposureState.COMPLETED)
    item = SimpleNamespace(
        admission=RecallItemAdmission.ADMITTED,
        receipt_id="receipt",
        identity=SimpleNamespace(model_dump=lambda **_: identity),
        content_sha256=chunk_digest,
        locator=SimpleNamespace(kind="knowledge_chunk", entry_id="policy", entry_revision=1),
    )
    store = SimpleNamespace(
        list_context_exposures=AsyncMock(
            return_value=SimpleNamespace(items=[exposure], next_cursor=None)
        ),
        load_recall_item_exposures=AsyncMock(return_value=[item]),
        load_checkpoint=AsyncMock(
            return_value={
                "automatic_recall": {
                    "receipt_id": "receipt",
                    "projection": {
                        "focus": {
                            "items": [
                                {"identity": identity, "content_hash": chunk_digest, "text": CHUNK}
                            ]
                        }
                    },
                }
            }
        ),
    )
    state = SimpleNamespace(
        native_recall=True,
        model_step_id="mstep_" + "1" * 32,
        session_id="session",
        session_store=store,
        memory=SimpleNamespace(corpus=corpus),
        controller=controller,
        admission=Mock(),
        observed_recall_exposures=set(),
    )
    asyncio.run(observe_delivered_recall(state))
    assert not update(controller, evidence="Consent still needed", evidence_refs=refs).is_error
    assert state.observed_recall_exposures == {"exposure"}
    bad_refs = [{**refs[0], "digest": chunk_digest}]
    assert update(controller, evidence_refs=bad_refs).is_error
    asyncio.run(observe_delivered_recall(state))
    assert store.load_recall_item_exposures.await_count == 1


def test_registered_revision_digest_cannot_be_overwritten_or_erased(corpus):
    controller, _ = setup_plan()
    canonical = corpus.delivered_revision_evidence("policy", 1, delivered_text=BODY)
    controller.observe_knowledge_evidence((canonical,))
    controller.observe_knowledge_evidence(
        (WorkEvidenceReference(kind="knowledge_revision", reference_id="policy", version="1"),)
    )
    with pytest.raises(ValueError, match="Conflicting canonical digest"):
        controller.observe_knowledge_evidence((canonical.model_copy(update={"digest": "0" * 64}),))
    assert not update(
        controller,
        evidence_refs=[
            {
                "kind": "knowledge_revision",
                "reference_id": "policy",
                "version": "1",
                "digest": canonical.digest,
            }
        ],
    ).is_error


def offered_state(corpus, *, preview=BODY, complete=True):
    identity = {"record_id": "policy", "record_type": "knowledge_entry", "revision": "1"}
    digest = hashlib.sha256(BODY.encode()).hexdigest()
    projected = {
        "identity": identity,
        "content_hash": digest,
        "preview": preview,
        "preview_complete": complete,
    }
    exposure = SimpleNamespace(exposure_id="offer-exposure", state=ContextExposureState.COMPLETED)
    item = SimpleNamespace(
        admission=RecallItemAdmission.OFFERED,
        receipt_id="receipt",
        identity=SimpleNamespace(model_dump=lambda **_: identity),
        content_sha256=digest,
        locator=SimpleNamespace(kind="knowledge_entry", entry_id="policy", entry_revision=1),
    )
    frame = {"receipt_id": "receipt", "projection": {"offer": {"items": [projected]}}}
    store = SimpleNamespace(
        list_context_exposures=AsyncMock(
            return_value=SimpleNamespace(items=[exposure], next_cursor=None)
        ),
        load_recall_item_exposures=AsyncMock(return_value=[item]),
        load_checkpoint=AsyncMock(return_value={"automatic_recall": frame}),
    )
    controller, _ = setup_plan(outcome())
    state = SimpleNamespace(
        native_recall=True,
        model_step_id="mstep_" + "2" * 32,
        session_id="session",
        session_store=store,
        memory=SimpleNamespace(corpus=corpus),
        controller=controller,
        admission=Mock(),
        observed_recall_exposures=set(),
    )
    return state, exposure, item, projected, frame


@pytest.mark.parametrize("preview", [BODY, CHUNK])
def test_complete_exposed_offer_registers_canonical_revision_once(corpus, preview):
    state, _, _, _, _ = offered_state(corpus, preview=preview)
    asyncio.run(observe_delivered_recall(state))
    assert state.controller.has_delivered_knowledge_revision("policy", "1")
    state.admission.observe_knowledge_text.assert_called_once_with(preview)
    asyncio.run(observe_delivered_recall(state))
    state.admission.observe_knowledge_text.assert_called_once()


@pytest.mark.parametrize(
    "preview,complete",
    [
        (BODY, False),
        (BODY[:10], True),
        ("Example policy", True),
        (None, False),
        (BODY.replace("consent", "[REDACTED]"), True),
        (CHUNK[:20], True),
    ],
)
def test_partial_or_metadata_offer_never_becomes_revision_evidence(corpus, preview, complete):
    state, _, _, _, _ = offered_state(corpus, preview=preview, complete=complete)
    asyncio.run(observe_delivered_recall(state))
    assert not state.controller.has_delivered_knowledge_revision("policy", "1")
    state.admission.observe_knowledge_text.assert_not_called()


def test_complete_offer_without_acknowledged_exposure_is_not_evidence(corpus):
    state, exposure, _, _, _ = offered_state(corpus)
    exposure.state = ContextExposureState.PLANNED
    asyncio.run(observe_delivered_recall(state))
    assert not state.controller.has_delivered_knowledge_revision("policy", "1")
    assert not state.observed_recall_exposures


@pytest.mark.parametrize("mismatch", ["receipt", "hash", "identity", "revision"])
def test_offer_requires_exact_exposed_receipt_identity_hash_and_revision(corpus, mismatch):
    state, _, item, projected, frame = offered_state(corpus)
    if mismatch == "receipt":
        frame["receipt_id"] = "another-receipt"
    elif mismatch == "hash":
        projected["content_hash"] = "0" * 64
    elif mismatch == "identity":
        projected["identity"] = {**projected["identity"], "record_id": "other"}
    else:
        item.locator.entry_revision = 2
    with pytest.raises((RuntimeError, BankingKnowledgeError)):
        asyncio.run(observe_delivered_recall(state))
    assert not state.controller.has_delivered_knowledge_revision("policy", "1")


def recovered_state(corpus):
    state, exposure, item, projected, frame = offered_state(corpus)
    item.admission = RecallItemAdmission.ADMITTED
    item.receipt_id = "recovery-receipt"
    projected.pop("preview")
    projected.pop("preview_complete")
    projected["text"] = BODY
    projected["text_complete"] = True
    frame["projection"] = None
    delta = {"receipt_id": item.receipt_id, "projection": {"items": [projected]}}
    frame["delta_state"] = {"deltas": [delta]}
    return state, exposure, item, projected, delta, frame


def test_reanchored_exposed_knowledge_uses_its_own_receipt_and_projection(corpus):
    state, _, _, _, _, _ = recovered_state(corpus)
    asyncio.run(observe_delivered_recall(state))
    assert state.controller.has_delivered_knowledge_revision("policy", "1")
    state.admission.observe_knowledge_text.assert_called_once_with(BODY)
    asyncio.run(observe_delivered_recall(state))
    state.admission.observe_knowledge_text.assert_called_once()


@pytest.mark.parametrize(
    "mismatch", ["receipt", "identity", "hash", "expired", "duplicate", "offered"]
)
def test_recovery_evidence_requires_exact_current_projection(corpus, mismatch):
    state, _, item, projected, delta, frame = recovered_state(corpus)
    if mismatch == "receipt":
        delta["receipt_id"] = "other"
    elif mismatch == "identity":
        projected["identity"] = {**projected["identity"], "record_id": "other"}
    elif mismatch == "hash":
        projected["content_hash"] = "0" * 64
    elif mismatch == "expired":
        delta["projection"] = None
    elif mismatch == "duplicate":
        frame["delta_state"]["deltas"].append(dict(delta))
    else:
        item.admission = RecallItemAdmission.OFFERED
    with pytest.raises(RuntimeError):
        asyncio.run(observe_delivered_recall(state))
    assert not state.controller.has_delivered_knowledge_revision("policy", "1")
    state.admission.observe_knowledge_text.assert_not_called()


def test_recovery_without_positive_exposure_is_not_evidence(corpus):
    state, exposure, _, _, _, _ = recovered_state(corpus)
    exposure.state = ContextExposureState.PLANNED
    asyncio.run(observe_delivered_recall(state))
    assert not state.controller.has_delivered_knowledge_revision("policy", "1")


def test_native_runtime_accepts_complete_offer_without_redundant_read(tmp_path):
    import sqlite3

    from cayu import ScriptedModelProvider
    from test_runtime_banking import call, text, work

    from cayu_tau3_banking.comparison_workflow_probe import catalogue_case
    from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
    from cayu_tau3_banking.navigation_probe import run_case

    documents, case, _ = catalogue_case()
    case = {**case, "followups": []}
    pending = work()
    completed = work(done=True, reference="doc_credit_cards_fern_001")
    for item in (pending, completed):
        item.pop("next_operation")
    provider = ScriptedModelProvider(
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [pending]}, "create"),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [completed]}, "complete"),
            text("Fern's annual fee is $20."),
        ]
    )
    result = run_case(
        tmp_path, case, documents=documents, provider_factory=lambda: provider, model="scripted"
    )
    assert result["plan"]["items"][0]["status"] == "completed"
    assert len(provider.requests) == 3
    with sqlite3.connect((tmp_path / "runtime.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        admissions = db.execute(
            "SELECT DISTINCT json_extract(item_json,'$.admission') FROM cayu_recall_item_exposures "
            "WHERE json_extract(item_json,'$.identity.record_id')='doc_credit_cards_fern_001'"
        ).fetchall()
    assert admissions == [("offered",)]
