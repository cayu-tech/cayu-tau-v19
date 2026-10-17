from types import SimpleNamespace

import pytest

from cayu_tau3_banking.literal_search import literal_candidates


def entry(identifier, text, title="Guide", revision=1):
    return SimpleNamespace(id=identifier, text=text, title=title, revision=revision)


def test_exhaustive_beyond_sixteen_and_deterministic_order():
    entries = [entry(f"doc-{i:03}", "Needle") for i in range(80)]
    results = literal_candidates(reversed(entries), "needle")
    assert len(results) == 80
    assert [row["entry_id"] for row in results] == [item.id for item in entries]


def test_regex_metacharacters_are_literal_and_offsets_reference_original_text():
    text = "é😀 prefix [A+B]. suffix [a+b]."
    result = literal_candidates([entry("one", text)], "[a+b].")[0]
    match = result["matches"][0]
    assert match["occurrences"] == 2
    assert text[match["first_match"]["start"] : match["first_match"]["end"]] == "[A+B]."
    assert match["excerpt"] == text[match["excerpt_start"] : match["excerpt_end"]]
    assert not literal_candidates([entry("two", "AAABx")], "[a+b].")


def test_title_only_and_body_matches_have_separate_counts():
    results = literal_candidates([entry("one", "other", title="Needle")], "needle")
    assert results[0]["matches"][0]["field"] == "title"
    result = literal_candidates([entry("two", "needle needle", title="needle")], "needle")[0]
    assert result["literal_occurrences"] == 3
    assert len(result["matches"]) == 2


def test_excerpt_bounded_and_not_full_document_evidence():
    result = literal_candidates([entry("one", "x" * 1000 + "needle" + "x" * 1000)], "needle")[0]
    assert len(result["matches"][0]["excerpt"]) == 246
    assert not result["matches"][0]["excerpt_complete"]
    assert "evidence_ref" not in result
    assert "text" not in result


def test_literal_whitespace_is_not_rewritten():
    assert literal_candidates([entry("one", "two  spaces")], "two  spaces")
    assert not literal_candidates([entry("one", "two  spaces")], "two spaces")


@pytest.mark.parametrize("query", [None, True, "", "  ", "x" * 501])
def test_invalid_query(query):
    with pytest.raises(ValueError):
        literal_candidates([], query)
