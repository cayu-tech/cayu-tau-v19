from cayu_tau3_banking.knowledge import _load_corpus_material
from cayu_tau3_banking.section_embeddings import section_embedding_chunks


def entry(tmp_path, body):
    import json

    (tmp_path / "doc.json").write_text(
        json.dumps({"id": "doc", "title": "Example", "content": body})
    )
    return _load_corpus_material(tmp_path).entries[0]


def test_sections_preserve_table_and_ancestor_context(tmp_path):
    document = entry(
        tmp_path, "# Root\nIntro\n## Table\n| A | B |\n|---|---|\n| X | Y |\n## Next\nEnd\n"
    )
    chunks = section_embedding_chunks(document)
    assert len(chunks) == 3
    assert "# Root\n\n## Table\n| A | B |\n|---|---|\n| X | Y |" in chunks[1].text
    assert "## Table" not in chunks[2].text
    assert (
        "".join(
            document.text[c.metadata["body_start_char"] : c.metadata["body_end_char"]]
            for c in chunks
        )
        == document.text
    )
    assert section_embedding_chunks(document) == chunks
    assert all(c.entry_id == document.id and c.entry_revision == 1 for c in chunks)
    assert len({c.id for c in chunks}) == 3


def test_fenced_headings_do_not_split_sections(tmp_path):
    document = entry(tmp_path, "Intro\n~~~python\n# Not a heading\n~~~\n## Real\nBody\n")
    chunks = section_embedding_chunks(document)
    assert len(chunks) == 2
    assert "# Not a heading" in chunks[0].text
    assert chunks[1].text == "Example\n\n## Real\nBody"


def test_plain_document_remains_one_projection(tmp_path):
    document = entry(tmp_path, "Plain body with café and no headings.")
    chunks = section_embedding_chunks(document)
    assert len(chunks) == 1
    assert chunks[0].text == f"Example\n\n{document.text}"


def test_nested_headings_and_unicode_offsets(tmp_path):
    document = entry(tmp_path, "# Café\n## Parent\n### Child\nA naïve example.\n# Other\nEnd")
    chunks = section_embedding_chunks(document)
    assert "# Café\n\n## Parent\n\n### Child" in chunks[2].text
    assert "Café" not in chunks[3].text
    assert (
        "".join(
            document.text[c.metadata["body_start_char"] : c.metadata["body_end_char"]]
            for c in chunks
        )
        == document.text
    )


def test_cayu_indexes_every_section_and_returns_best_matching_section(tmp_path):
    import asyncio

    from cayu import (
        InMemoryEmbeddingKnowledgeStore,
        KnowledgeQuery,
        TextEmbedding,
        TextEmbeddingProvider,
        TextEmbeddingResult,
    )

    class Embeddings(TextEmbeddingProvider):
        name = "section-test"

        async def embed_texts(self, request):
            return TextEmbeddingResult(
                model=request.model,
                embeddings=[
                    TextEmbedding(index=i, vector=[1.0, 0.0] if "needle" in text else [0.0, 1.0])
                    for i, text in enumerate(request.texts)
                ],
            )

    async def check():
        document = entry(tmp_path, "# Introduction\nGeneral text\n## Detail\nThe needle is here.")
        material = _load_corpus_material(tmp_path)
        chunks = section_embedding_chunks(document)
        store = InMemoryEmbeddingKnowledgeStore(
            access_scope=material.scope,
            embedding_provider=Embeddings(),
            embedding_model="test",
            embedding_dimensions=2,
        )
        await store.create_entry(document, chunks=chunks, access_scope=material.scope)
        indexed = await store.backfill_embeddings(access_scope=material.scope)
        assert indexed.indexed_records == 2
        assert indexed.failed_records == 0
        result = await store.search(
            KnowledgeQuery(text="needle", mode="semantic", namespace=document.namespace),
            access_scope=material.scope,
        )
        assert len(result.hits) == 1
        assert result.hits[0].chunk.id == chunks[1].id

    asyncio.run(check())
