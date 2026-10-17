import tomllib
from pathlib import Path

from cayu_tau3_banking.provenance import (
    CAYU_COMMIT,
    CAYU_REPOSITORY,
    TAU2_COMMIT,
    assert_source_provenance,
)


def test_bootstrapped_sources_match_frozen_revisions() -> None:
    tau2_repository = Path(__file__).resolve().parents[1] / ".vendor" / "tau2"

    assert_source_provenance(tau2_repository)
    assert TAU2_COMMIT == "fc0055dc4e0a316c3f83133267fbd6faaa770992"
    assert CAYU_COMMIT == "d8e530acafa3cb7ab9596340182cffc8d9467d5d"
    assert CAYU_REPOSITORY == "file:///Users/talgatt/Desktop/Projects/cayu-tau-performance-20260927"


def test_package_dependency_matches_runtime_provenance() -> None:
    manifest = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with manifest.open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["dependencies"]
    assert f"cayu @ git+{CAYU_REPOSITORY}@{CAYU_COMMIT}" in dependencies
