"""Exercise the complete experiment runner with a local fake embedding provider."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
from cayu import RecallSourceResult, TextEmbedding, TextEmbeddingResult


@pytest.fixture
def experiment():
    path = Path(__file__).parents[1] / "scripts/evaluate_section_embeddings.py"
    spec = importlib.util.spec_from_file_location("section_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_truncation_is_not_an_execution_failure(experiment):
    for reason in (
        "lexical_truncated",
        "semantic_truncated",
        "lexical_truncated+semantic_truncated",
    ):
        experiment.require_usable_source(
            RecallSourceResult(
                source="test",
                channels=(),
                records=(),
                coverage_complete=False,
                partial_reason=reason,
            )
        )
    for reason in (
        "semantic_timeout",
        "semantic_failed",
        "semantic_index_partial",
        "lexical_truncated+semantic_failed",
    ):
        with pytest.raises(RuntimeError, match="Recall source failed"):
            experiment.require_usable_source(
                RecallSourceResult(
                    source="test",
                    channels=(),
                    records=(),
                    coverage_complete=False,
                    partial_reason=reason,
                )
            )


def test_complete_runner_and_cached_resume_offline(experiment, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    documents = tmp_path / experiment.DOCUMENTS
    documents.mkdir(parents=True)
    # More than the candidate limit, so normal truncation actually occurs.
    for i in range(20):
        (documents / f"doc{i}.json").write_text(
            json.dumps(
                {
                    "id": f"doc{i}",
                    "title": f"Example {i}",
                    "content": "# Overview\nGeneral material.\n## Detail\nThe needle is here.",
                }
            )
        )
    case = {"id": "probe", "query": "needle", "required": {"doc0": "needle"}}
    (tmp_path / "evals").mkdir()
    for suite in experiment.SUITES:
        (tmp_path / f"evals/{suite}.json").write_text(json.dumps({"cases": [case]}))
    old = tmp_path / "artifacts/tier-ranking-diagnostic/run-_1tirrky"
    old.mkdir(parents=True)
    (old / "manifest.json").write_text(json.dumps({"cases": [case]}))

    class FakeProvider:
        calls = 0

        def __init__(self, **kwargs):
            pass

        async def embed_texts(self, request):
            FakeProvider.calls += 1
            return TextEmbeddingResult(
                model=request.model,
                embeddings=[
                    TextEmbedding(
                        index=i,
                        vector=([1.0, 0.0] if "needle" in text else [0.0, 1.0])
                        + [0.0] * (experiment.DIMENSIONS - 2),
                    )
                    for i, text in enumerate(request.texts)
                ],
            )

        async def aclose(self):
            pass

    monkeypatch.setattr(experiment, "OpenAIProvider", FakeProvider)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    asyncio.run(experiment.run(first, True))
    summary = json.loads((first / "summary.json").read_text())
    assert len(summary["rows"]) == 4
    assert not summary["default_changed"]
    result = json.loads((first / "public-retrieval-v1--probe.json").read_text())
    assert result["sources"][1]["partial_reason"] == "lexical_truncated+semantic_truncated"
    calls = FakeProvider.calls
    asyncio.run(experiment.run(second, True, first))
    assert FakeProvider.calls == calls
    assert json.loads((second / "summary.json").read_text()) == summary
    manifest = json.loads((first / "manifest.json").read_text())
    manifest["model"] = "wrong-model"
    (first / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mismatch"):
        experiment.CachedEmbeddings(FakeProvider()).restore(first, manifest["corpus_sha256"])
