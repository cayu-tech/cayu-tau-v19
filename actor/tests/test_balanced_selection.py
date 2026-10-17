import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def selector(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "balanced_selection", scripts / "evaluate_balanced_selection.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_balanced_sources_deduplicate_and_backfill(selector):
    lex = [("a", "1"), ("b", "1"), ("d", "1")]
    sem = [("a", "1"), ("c", "1")]
    assert selector.merge_families(lex, sem, mode="balanced_lexical_first") == [
        lex[0],
        sem[1],
        lex[1],
        lex[2],
    ]
    for mode in ("lexical_first", "balanced_lexical_first", "balanced_semantic_first"):
        assert set(selector.merge_families(lex, sem, mode=mode)) == set(lex + sem)
        assert selector.merge_families([], [], mode=mode) == []
        assert selector.merge_families([], sem, mode=mode) == sem
    with pytest.raises(ValueError):
        selector.merge_families([], [], mode="unknown")


def test_family_does_not_double_vote_or_merge_revisions(selector):
    lanes = {"a": {("doc", "1"): 2, ("other", "1"): 1}, "b": {("doc", "1"): 2, ("doc", "2"): 3}}
    assert selector.family_order(lanes, ("a", "b")) == [("other", "1"), ("doc", "1"), ("doc", "2")]
    assert selector.family_order(lanes, ("b", "a")) == selector.family_order(lanes, ("a", "b"))
