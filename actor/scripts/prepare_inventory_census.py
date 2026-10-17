"""Freeze current sources and explicit LB settings; never dispatch models locally."""

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from run_full_benchmark import digest, verify, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expected-tests", type=int, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    project = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=project).strip():
        raise ValueError("Commit and qualify sources before freezing")
    verify(args.baseline)
    previous = json.loads((args.baseline / "protocol.json").read_text())
    settings = json.loads((project / "docs/candidate-settings-20260917.json").read_text())
    options = settings["cli_arguments"]
    if options[options.index("--provider") + 1] != "codex-lb":
        raise ValueError("This census must use Codex LB, never subscription fallback")
    catalog = Path(previous["tau_repository"]) / "data/tau2/domains/banking_knowledge/tasks.json"
    ids = sorted(t["id"] for t in json.loads(catalog.read_text()))
    if (
        len(ids) != len(set(ids))
        or len(ids) != 97
        or not all(re.fullmatch(r"task_\d{3}", t) for t in ids)
    ):
        raise ValueError("Expected exact 97-task catalog")
    artifacts = project / "artifacts"
    artifacts.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="inventory-full97-", dir=artifacts))
    # Paths are identical on the dedicated VM, but its checkout stays isolated.
    for folder in ("src", "tests", "scripts", "docs"):
        shutil.copytree(
            project / folder, root / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(project / name, root / name)
    for source, target in (
        ("run_full_benchmark.py", "run_benchmark.py"),
        ("run_lean_advice_task.py", "run_pilot.py"),
    ):
        shutil.copy2(project / "scripts" / source, root / target)
    # Retain the reviewed LB-only connectivity check and non-deleting archiver.
    for name in ("connectivity_check.py", "archive_support.py"):
        shutil.copy2(args.baseline / name, root / name)
    protocol = {
        **{key: previous[key] for key in ("tau_repository", "corpus_embedding_cache")},
        **{
            key: settings[key]
            for key in (
                "actor",
                "simulator",
                "runtime",
                "limits",
                "navigation_limits",
                "cli_arguments",
                "framework_commit",
                "tau_commit",
            )
        },
        "agent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip(),
        "seed": 300,
        "trials": 1,
        "whole_task_retries": 0,
        "task_ids": ids,
        "task_catalog_sha256": digest(catalog),
        "max_parallel_task_processes": 4,
        "per_task_concurrency": 1,
        "expected_tests": args.expected_tests,
        "worker_watchdog_seconds": 7500,
        "minimum_free_disk_bytes": 2 * 1024**3,
        "pause_after_consecutive_unscored": None,
        "report": str(project.parent / "agent-documents" / (root.name + ".html")),
        "qualification_receipt": str(root / "linux-qualified.json"),
        "purpose": "Measure frozen retrieval/controller candidate on all 97 public banking tasks.",
        "selection": "All public tasks, including previously inspected development cases; not an unseen holdout.",
        "retrieval": "Hybrid cached vectors plus paginated per-channel discovery; unchanged bounded automatic injection.",
        "scoring": "One attempt per task. Operational passes/97 retains unscored failures. Official pass^1 only when all 97 receive binary grades.",
        "stopping_policy": "Pause on shared credential/quota/rate errors, source mismatch, low disk or STOP; no whole-task retries or in-flight tuning.",
        "framework_status": "Exact experimental draft PR 1782 dependency, not claimed upstream or merge-ready.",
    }
    write_json(root / "protocol.json", protocol)
    files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(root / "source-manifest.json", {"files": files})
    write_json(
        root / "progress.json",
        {"status": "prepared", "rows": {}, "running": {}, "pause_reason": None},
    )
    archive = root.with_suffix(".tar.gz")
    with tarfile.open(archive, "x:gz") as output:
        for name in [*files, "source-manifest.json", "progress.json"]:
            output.add(root / name, arcname=name, recursive=False)
    receipt = {
        "root": str(root),
        "archive": str(archive),
        "archive_sha256": digest(archive),
        "manifest_sha256": digest(root / "source-manifest.json"),
    }
    write_json(root / "freeze-receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
