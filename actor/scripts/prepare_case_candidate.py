"""Freeze the isolated case-state candidate; never mutate the ongoing census."""

from __future__ import annotations

import difflib
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from cayu_tau3_banking.provenance import (
    CAYU_COMMIT,
    CAYU_REPOSITORY,
    TAU2_COMMIT,
    assert_source_provenance,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    baseline = project.parent / "cayu-tau3-choice/artifacts/evidence-case-evaluation-h3cbw44g"
    manifest = json.loads((baseline / "source-manifest.json").read_text())["files"]
    for name, expected in manifest.items():
        path = baseline / name
        if not path.resolve().is_relative_to(baseline.resolve()) or digest(path) != expected:
            raise ValueError("Prior qualified candidate changed")
    assert_source_provenance((project / ".vendor/tau2").resolve())
    (project / "artifacts").mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="case-candidate-", dir=project / "artifacts"))
    for directory in ("src", "tests", "scripts"):
        shutil.copytree(
            project / directory,
            root / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(project / name, root / name)
    shutil.copy2(project / "scripts/run_lean_advice_task.py", root / "run_pilot.py")
    (root / ".vendor").symlink_to((project / ".vendor").resolve(), target_is_directory=True)
    prior = json.loads((baseline / "protocol.json").read_text())
    protocol = {
        key: prior[key]
        for key in (
            "seed",
            "trials",
            "whole_task_retries",
            "actor",
            "simulator",
            "retrieval",
            "runtime",
            "limits",
            "navigation_limits",
            "credential_handling",
            "tau_repository",
            "corpus_embedding_cache",
            "cli_arguments",
        )
    }
    protocol.update(
        framework_commit=CAYU_COMMIT,
        framework_repository=CAYU_REPOSITORY,
        tau_commit=TAU2_COMMIT,
        baseline=str(baseline),
        baseline_manifest_sha256=digest(baseline / "source-manifest.json"),
        task_ids=["task_017", "task_056", "task_066"],
        max_parallel_task_processes=1,
        purpose="Qualify resource-bound prerequisites, dependent reassessment, evidence ergonomics and merged terminal-stream lifecycle.",
        selection="Known development regressions: 017 for legitimate customer questions; 056/066 for research/evidence and resource sequencing. Not holdouts and not part of the frozen census score.",
        treatment=[
            "Merged Cayu terminal-stream fix, retaining the existing recall extension",
            "Explicit procedure/resource checks and dependency reassessment in the existing plan",
            "Hook-owned pre-execution receipts distinguish consumed prerequisites from current action authority",
            "Typed failure categories alongside unchanged official grades",
        ],
        causal_limit="Combined integration treatment, not a single-variable accuracy ablation. Synthetic negative controls establish individual mechanisms; live task changes alone cannot attribute an improvement to one component.",
        qualification="Pending a full offline suite on this immutable snapshot; no live result claimed.",
        stopping_policy="Do not compete with the active census. After qualification and provider preflight, one serial attempt per task with existing limits and no whole-task retries. Preserve failures and stop on execution/provider errors.",
    )
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    patch = []
    changes = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or ".vendor" in path.parts
            or path.suffix not in {".py", ".toml", ".md"}
        ):
            continue
        name = str(path.relative_to(root))
        previous = baseline / name
        if previous.is_file() and digest(previous) == digest(path):
            continue
        changes.append(name)
        before = previous.read_text().splitlines(keepends=True) if previous.is_file() else []
        patch.extend(
            difflib.unified_diff(
                before, path.read_text().splitlines(keepends=True), f"a/{name}", f"b/{name}"
            )
        )
    (root / "candidate.patch").write_text("".join(patch))
    files = {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".vendor" not in path.parts
    }
    (root / "source-manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    print(
        json.dumps(
            {
                "root": str(root),
                "manifest_sha256": digest(root / "source-manifest.json"),
                "changed_files": changes,
                "files": len(files),
            }
        )
    )


if __name__ == "__main__":
    main()
