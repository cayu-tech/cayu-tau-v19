import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def selector():
    path = Path(__file__).parents[1] / "scripts/evaluate_multiscale_selection.py"
    spec = importlib.util.spec_from_file_location("multiscale_selection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_union_deduplication_empty_sources_and_determinism(selector):
    lanes = {name: {} for name in selector.ORDER}
    lanes[selector.WHOLE] = {("a", "1"): 1, ("b", "1"): 2}
    lanes[selector.SECTION] = {("a", "1"): 1, ("c", "1"): 2}
    expected = {("a", "1"), ("b", "1"), ("c", "1")}
    for mode in ("split_semantic", "max_semantic"):
        result = selector.weighted_selection(lanes, mode=mode)
        assert set(result) == expected and len(result) == 3
        assert result == selector.weighted_selection(dict(reversed(list(lanes.items()))), mode=mode)
    assert selector.round_robin(lanes) == [("a", "1"), ("c", "1"), ("b", "1")]
    assert selector.round_robin({name: {} for name in selector.ORDER}) == []


def test_max_semantic_does_not_double_vote(selector):
    lanes = {name: {} for name in selector.ORDER}
    lanes[selector.WHOLE] = {("shared", "1"): 2, ("only_whole", "1"): 1}
    lanes[selector.SECTION] = {("shared", "1"): 2}
    assert selector.weighted_selection(lanes, mode="max_semantic")[0] == ("only_whole", "1")
    assert selector.weighted_selection(lanes, mode="split_semantic")[0] == ("shared", "1")


def test_invalid_selector_is_rejected(selector):
    with pytest.raises(ValueError, match="Unknown selector"):
        selector.weighted_selection({}, mode="unexpected")
