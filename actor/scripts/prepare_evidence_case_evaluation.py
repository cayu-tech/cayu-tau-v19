"""Freeze source-reference and retained-case improvements separately from the census."""

import difflib
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    project = Path(__file__).resolve().parents[1]
    baseline = project / "artifacts/reply-guard-evaluation-cu39l22_"
    manifest = json.loads((baseline / "source-manifest.json").read_text())["files"]
    if any(digest(baseline / name) != value for name, value in manifest.items()):
        raise ValueError("Original census snapshot changed")
    changed = {
        str(path.relative_to(project))
        for path in (project / "src").rglob("*.py")
        if not (baseline / path.relative_to(project)).is_file()
        or digest(path) != digest(baseline / path.relative_to(project))
    }
    expected = {
        "src/cayu_tau3_banking/product_navigation.py",
        "src/cayu_tau3_banking/controller.py",
        "src/cayu_tau3_banking/advice.py",
    }
    if changed != expected:
        raise ValueError(f"Unexpected actor changes: {sorted(changed)}")
    root = Path(tempfile.mkdtemp(prefix="evidence-case-evaluation-", dir=project / "artifacts"))
    for directory in ("src", "tests", "scripts"):
        shutil.copytree(
            project / directory,
            root / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for filename in ("README.md", "pyproject.toml"):
        shutil.copy2(project / filename, root / filename)
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
            "framework_commit",
            "tau_commit",
            "tau_repository",
            "corpus_embedding_cache",
            "cli_arguments",
        )
    }
    protocol.update(
        task_ids=[],
        purpose="Qualify evidence-reference ergonomics and retained outcome sources; no live score claim",
        selection="Offline synthetic regressions only. Live evaluation must be selected separately, without interfering with the frozen census.",
        treatment=[
            "Reuse delivered source references and copy unambiguous exact source paragraphs",
            "Patch saved worksheets; retain facts but reopen conclusions when their declared inputs change",
            "Retain resolved research sources for unfinished parent outcomes",
            "Separate unverified model interpretation from literal source values; return bounded source recovery for automatic-quotation failures",
        ],
        baseline=str(baseline),
        baseline_manifest_sha256=digest(baseline / "source-manifest.json"),
        max_parallel_task_processes=1,
        stopping_policy="Require offline qualification and a successful connectivity check. One attempt each, serially, no whole-task retries or deadline extensions. Stop on execution/provider errors; preserve failures.",
        qualification="Pending offline qualification. No live task coordinates selected.",
    )
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    patch = []
    for path in sorted(root.rglob("*.py")):
        name = str(path.relative_to(root))
        previous = baseline / name
        old = previous.read_text().splitlines(keepends=True) if previous.is_file() else []
        patch.extend(
            difflib.unified_diff(
                old, path.read_text().splitlines(keepends=True), f"a/{name}", f"b/{name}"
            )
        )
    (root / "candidate.patch").write_text("".join(patch))
    files = {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".vendor" not in path.parts
    }
    (root / "source-manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    print(json.dumps({"root": str(root), "actor_changes": sorted(changed), "files": len(files)}))


if __name__ == "__main__":
    main()
