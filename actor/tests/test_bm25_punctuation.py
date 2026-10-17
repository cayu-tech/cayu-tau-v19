import asyncio
import json

import pytest

from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus, _search_tokens


@pytest.mark.parametrize("apostrophe", ["'", "\u2018", "\u2019"])
def test_typographic_apostrophes_preserve_words(apostrophe):
    assert _search_tokens(f"I{apostrophe}d inspect the owner{apostrophe}s archive") == [
        "i'd",
        "inspect",
        "the",
        "owner's",
        "archive",
    ]


def test_sentence_punctuation_does_not_join_words():
    assert _search_tokens("Hi—I'd inspect it.") == _search_tokens("Hi, I\u2019d inspect it.")
    assert _search_tokens("customer-owned archive") == ["customer-owned", "archive"]


def test_bm25_ranking_is_invariant_for_typographic_apostrophes(tmp_path):
    for index, content in enumerate(
        [
            "I'd inspect the owner's archive before retiring its credentials.",
            "The owner\u2019s archive contains an inspection history.",
            "I need a new archive. D class storage has a lower price.",
        ]
    ):
        (tmp_path / f"doc_{index}.json").write_text(
            json.dumps(
                {"id": f"doc_{index}", "title": f"Archive guide {index}", "content": content}
            )
        )
    corpus = BankingKnowledgeCorpus.from_directory(tmp_path)
    source = next(s for s in corpus.recall_sources if s.name == "tau3_banking_bm25")

    async def compare():
        results = []
        for query in [
            "I'd inspect the owner's archive",
            "I\u2019d inspect the owner\u2019s archive",
        ]:
            result = await source.retrieve(corpus._situation(query))
            results.append(result.channels)
        assert results[0] == results[1]

    asyncio.run(compare())
