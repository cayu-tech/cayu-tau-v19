from cayu_tau3_banking.search_excerpts import matching_excerpt


def test_excerpt_targets_matching_section_with_exact_unicode_offsets():
    text = (
        "# Introduction\n"
        + "unrelated 界\n" * 100
        + "# Requirements\nCard age must exceed sixty days.\n"
    )
    result = matching_excerpt(text, "card age requirements")
    assert result["heading"] == "# Requirements"
    assert result["text"] == text[result["start"] : result["end"]]
    assert "sixty days" in result["text"]
    assert not result["complete_document"]
    assert len(result["text"]) <= 500


def test_no_lexical_hit_does_not_invent_semantic_evidence():
    assert matching_excerpt("A monetary incentive", "welcome bonus") is None
    assert matching_excerpt("ordinary policy", "[]") is None
    assert matching_excerpt("discarded transaction", "card") is None


def test_excerpt_is_original_source_not_reformatted_summary():
    text = "# Tier table\n| Crypto-Cash | Entry |\n"
    result = matching_excerpt(text, "Crypto-Cash Entry")
    assert result["text"] == text[result["start"] : result["end"]]
    assert result["kind"] == "lexical_navigation_excerpt"
