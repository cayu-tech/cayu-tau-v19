"""Fail-closed source provenance checks for benchmark execution."""

from __future__ import annotations

import argparse
import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import cayu
import tau2

TAU2_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
CAYU_COMMIT = "d8e530acafa3cb7ab9596340182cffc8d9467d5d"
# Upstream main frozen for the separate 2026-09-23 candidate; r3 stays unchanged.
CAYU_REPOSITORY = "file:///Users/talgatt/Desktop/Projects/cayu-tau-performance-20260927"


class ProvenanceError(RuntimeError):
    """Installed source does not match the frozen benchmark manifest."""


def assert_source_provenance(tau2_repository: Path) -> None:
    repository = tau2_repository.resolve(strict=True)
    if not (repository / ".git").exists():
        raise ProvenanceError(f"tau2 repository is not a Git checkout: {repository}")

    actual_commit = _git(repository, "rev-parse", "HEAD")
    if actual_commit != TAU2_COMMIT:
        raise ProvenanceError(f"tau2 source mismatch: expected {TAU2_COMMIT}, got {actual_commit}")
    if _git(repository, "status", "--porcelain", "--untracked-files=no"):
        raise ProvenanceError(f"tau2 source has tracked modifications: {repository}")

    required_paths = (
        repository / "src" / "tau2" / "__init__.py",
        repository / "data" / "tau2" / "domains" / "banking_knowledge" / "tasks.json",
        repository / "data" / "tau2" / "domains" / "banking_knowledge" / "db.json",
        repository / "data" / "tau2" / "domains" / "banking_knowledge" / "documents",
    )
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise ProvenanceError(f"tau2 banking checkout is incomplete: {', '.join(missing_paths)}")

    tau2_source = Path(tau2.__file__).resolve(strict=True)
    expected_tau2_source = repository / "src" / "tau2"
    if not tau2_source.is_relative_to(expected_tau2_source):
        raise ProvenanceError(
            f"tau2 imports from {tau2_source}, expected source under {expected_tau2_source}"
        )

    _assert_cayu_distribution_provenance()


def _assert_cayu_distribution_provenance() -> None:
    try:
        distribution = metadata.distribution("cayu")
    except metadata.PackageNotFoundError as exc:
        raise ProvenanceError("Cannot verify Cayu: the distribution is not installed.") from exc

    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise ProvenanceError("Cannot verify Cayu: direct_url.json is missing.")
    try:
        direct_url: Any = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise ProvenanceError("Cannot verify Cayu: direct_url.json is invalid.") from exc
    if not isinstance(direct_url, dict):
        raise ProvenanceError("Cannot verify Cayu: direct_url.json is not an object.")

    repository = direct_url.get("url")
    vcs_info = direct_url.get("vcs_info")
    # Official deployments install the already-verified framework checkout from
    # a local file URL on the benchmark VM; portable environments use GitHub.
    # In both cases the exact commit below is mandatory.
    if (
        not isinstance(repository, str)
        or (repository != CAYU_REPOSITORY and not repository.startswith("file://"))
        or not isinstance(vcs_info, dict)
    ):
        raise ProvenanceError(
            f"Cayu source mismatch: expected Git repository {CAYU_REPOSITORY} at {CAYU_COMMIT}."
        )
    if vcs_info.get("vcs") != "git" or vcs_info.get("commit_id") != CAYU_COMMIT:
        raise ProvenanceError(
            f"Cayu source mismatch: expected {CAYU_COMMIT}, got {vcs_info.get('commit_id')!r}."
        )

    cayu_source = Path(cayu.__file__).resolve(strict=True)
    installed_source = Path(distribution.locate_file("cayu/__init__.py")).resolve(strict=True)
    if cayu_source != installed_source:
        raise ProvenanceError(
            f"Cayu imports from {cayu_source}, expected installed source {installed_source}"
        )


def _git(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or "unknown Git failure"
        raise ProvenanceError(f"Cannot verify tau2 source: {detail}")
    return process.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-repository", type=Path, required=True)
    args = parser.parse_args()
    assert_source_provenance(args.tau2_repository)
    print(f"Verified tau2 {TAU2_COMMIT} and Cayu {CAYU_COMMIT} source provenance.")


if __name__ == "__main__":
    main()
