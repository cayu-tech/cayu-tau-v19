import asyncio
import copy
import json

import pytest
from cayu import TextEmbedding, TextEmbeddingProvider, TextEmbeddingResult

from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus, BankingKnowledgeError
from cayu_tau3_banking.retrieval_eval import (
    EvidenceValidationError,
    evaluate,
    safe_error_diagnostic,
    score_ids,
    validate_suite,
)


def fixture():
    documents = {"doc_a": {"id": "doc_a", "title": "Transfer timing", "content": "Three days"}}
    suite = {
        "version": "synthetic",
        "corpus_sha256": "pending",
        "provenance": "synthetic",
        "cases": [
            {
                "id": "timing",
                "group": "timing",
                "query": "Transfer timing",
                "required": {"doc_a": "Three days"},
            }
        ],
    }
    return documents, suite


def test_metrics_use_required_evidence_and_respect_rank_cutoff():
    assert score_ids(["noise", "a", "b"], {"a", "b"}, k=2) == {
        "required_recall": 0.5,
        "all_required": 0.0,
        "reciprocal_rank": 0.5,
    }
    assert score_ids([], {"a"}, k=4)["required_recall"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        score_ids(["a", "a"], {"a"}, k=4)


@pytest.mark.parametrize("mutation", ["quote", "unknown", "duplicate", "hidden"])
def test_invalid_or_nonpublic_labels_fail_closed(mutation):
    docs, suite = fixture()
    if mutation == "quote":
        suite["cases"][0]["required"]["doc_a"] = "invented"
    elif mutation == "unknown":
        suite["cases"][0]["required"] = {"missing": "Three days"}
    elif mutation == "duplicate":
        suite["cases"].append(copy.deepcopy(suite["cases"][0]))
    else:
        suite["cases"][0]["expected_actions"] = []
    with pytest.raises(ValueError):
        validate_suite(suite, docs)


def test_public_corpus_evaluation_and_fingerprint_check(tmp_path):
    docs, suite = fixture()
    (tmp_path / "doc_a.json").write_text(json.dumps(docs["doc_a"]))
    corpus = BankingKnowledgeCorpus.from_directory(tmp_path)
    with pytest.raises(ValueError, match="fingerprint"):
        asyncio.run(evaluate(corpus, suite, docs))
    suite["corpus_sha256"] = corpus.corpus_sha256
    report = asyncio.run(evaluate(corpus, suite, docs))
    assert report["metrics"]["required_recall"] == 1.0
    assert report["metrics"]["required_quote_coverage"] == 1.0
    assert report["cases"][0]["returned_ids"] == ["doc_a"]
    assert report["cases"][0]["off_scope_fraction"] is None


def test_diagnostics_do_not_capture_exception_payload_or_locals():
    secret = "synthetic-sensitive-value"
    try:
        raise ValueError(secret)
    except ValueError as error:
        diagnostic = safe_error_diagnostic(error)
    assert diagnostic["error_type"] == "ValueError"
    assert diagnostic["frames"][-1]["file"] == "test_retrieval_eval.py"
    assert secret not in json.dumps(diagnostic)


def test_hybrid_projection_resolves_to_canonical_evidence(tmp_path):
    class OfflineEmbeddings(TextEmbeddingProvider):
        name = "offline-contract-test"

        async def embed_texts(self, request):
            return TextEmbeddingResult(
                model=request.model,
                embeddings=[
                    TextEmbedding(index=i, vector=[1.0, 0.0]) for i, _ in enumerate(request.texts)
                ],
            )

    docs, suite = fixture()
    (tmp_path / "doc_a.json").write_text(json.dumps(docs["doc_a"]))

    async def check():
        corpus = await BankingKnowledgeCorpus.from_directory_with_embeddings(
            tmp_path,
            embedding_provider=OfflineEmbeddings(),
            embedding_model="offline-contract-test",
            embedding_dimensions=2,
        )
        suite["corpus_sha256"] = corpus.corpus_sha256
        result = json.loads(await corpus.search(suite["cases"][0]["query"]))
        assert result["documents"][0]["text"] == "Three days"
        assert all("reason" in source for source in result["source_coverage"])
        report = await evaluate(corpus, suite, docs)
        assert report["metrics"]["required_recall"] == 1.0
        recalled = await corpus.recall(suite["cases"][0]["query"])
        assert '"text":"Three days"' in recalled.context

    asyncio.run(check())


def test_evidence_diagnostics_preserve_structured_checks():
    error = EvidenceValidationError(
        case_index=0, result_rank=1, checks={"canonical_text_matches": False}
    )
    diagnostic = safe_error_diagnostic(error)
    assert diagnostic["case_index"] == 0
    assert diagnostic["result_rank"] == 1
    assert diagnostic["checks"] == {"canonical_text_matches": False}


@pytest.mark.parametrize(
    "locator",
    [
        {"entry_id": "missing", "entry_revision": 1},
        {"entry_id": "doc_a", "entry_revision": 2},
        {"entry_id": "doc_a", "entry_revision": True},
    ],
)
def test_canonical_payload_rejects_unknown_or_stale_revision(tmp_path, locator):
    docs, _ = fixture()
    (tmp_path / "doc_a.json").write_text(json.dumps(docs["doc_a"]))
    corpus = BankingKnowledgeCorpus.from_directory(tmp_path)
    result = asyncio.run(corpus.engine.recall(corpus._situation("Transfer timing")))
    candidate = result.candidates[0]
    changed = candidate.model_copy(
        update={"record": candidate.record.model_copy(update={"locator": locator})}
    )
    with pytest.raises(BankingKnowledgeError, match="pinned entry revision"):
        corpus._candidate_payload(changed, rank=1)


def test_canonical_payload_preserves_multibyte_bound(tmp_path):
    docs, _ = fixture()
    docs["doc_a"]["content"] = "Transfer timing " + "界" * 4000
    (tmp_path / "doc_a.json").write_text(json.dumps(docs["doc_a"]))
    corpus = BankingKnowledgeCorpus.from_directory(tmp_path)
    result = json.loads(asyncio.run(corpus.search("Transfer timing")))
    document = result["documents"][0]
    assert len(document["text"].encode()) <= 8000
    assert docs["doc_a"]["content"].startswith(document["text"])
    assert document["text_complete"] is False
