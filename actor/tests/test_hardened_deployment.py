"""Qualification partitions are exhaustive and never dispatch live models."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "hardened_deployment", Path(__file__).parents[1] / "scripts/deploy_hardened_census_vm.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_every_module_including_nested_tests_is_assigned_once(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    for index in range(11):
        (tests / f"test_{index:02d}.py").touch()
    (tests / "nested").mkdir()
    (tests / "nested/test_nested.py").touch()
    shards = module.test_shards(tmp_path)
    flattened = [name for shard in shards for name in shard]
    assert len(shards) == 4
    assert len(flattened) == len(set(flattened)) == 12
    assert sorted(flattened) == sorted(str(p.relative_to(tmp_path)) for p in tests.rglob("*.py"))


def test_no_empty_or_invalid_qualification(tmp_path):
    with pytest.raises(ValueError):
        module.test_shards(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_one.py").touch()
    assert module.test_shards(tmp_path) == [["tests/test_one.py"]]
    with pytest.raises(ValueError):
        module.test_shards(tmp_path, 0)
