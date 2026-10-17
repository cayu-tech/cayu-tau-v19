"""Freeze a new full census; all qualification and model work runs on the dedicated VM."""

from __future__ import annotations

import difflib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

from run_full_benchmark import GIB, digest, publish, verify, write_json

from cayu_tau3_banking.provenance import CAYU_COMMIT, TAU2_COMMIT, assert_source_provenance


def main():
    project = Path(__file__).resolve().parents[1]
    # The preserved qualification baseline lives in the case-tracking checkout;
    # this development checkout contains the current candidate source.
    baseline = project.parent / "cayu-tau3-case-tracking/artifacts/followup-eval-1z_iwpn0"
    verify(baseline)
    prior = json.loads((baseline / "protocol.json").read_text())
    tau = Path(prior["tau_repository"])
    assert_source_provenance(tau)
    task_file = tau / "data/tau2/domains/banking_knowledge/tasks.json"
    ids = sorted(row["id"] for row in json.loads(task_file.read_text()))
    if (
        len(ids) != 97
        or len(set(ids)) != 97
        or not all(re.fullmatch(r"task_\d{3}", t) for t in ids)
    ):
        raise ValueError("Expected the exact public 97-task catalog")
    root = Path(tempfile.mkdtemp(prefix="hardened-full97-", dir=project / "artifacts"))
    for name in ("src", "tests", "scripts"):
        shutil.copytree(
            project / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(project / name, root / name)
    shutil.copy2(project / "scripts/run_lean_advice_task.py", root / "run_pilot.py")
    shutil.copy2(project / "scripts/run_full_benchmark.py", root / "run_benchmark.py")
    shutil.copy2(baseline / "connectivity_check.py", root / "connectivity_check.py")
    (root / ".vendor").symlink_to(tau.parent, target_is_directory=True)
    keys = (
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
    protocol = {key: prior[key] for key in keys}
    options = protocol["cli_arguments"]
    for flag, value in (
        ("--completion-timeout-seconds", "7200"),
        ("--simulation-timeout-seconds", "7200"),
        ("--runtime-max-steps", "192"),
    ):
        options[options.index(flag) + 1] = value
    options.extend(["--runtime-model-max-attempts", "3"])
    protocol["limits"].update(
        simulation_seconds=7200,
        adapter_callback_seconds=7200,
        runtime_steps=192,
        transient_model_attempts=3,
        unknown_model_attempts=1,
    )
    report = project.parent / "agent-documents" / (root.name + ".html")
    protocol.update(
        framework_commit=CAYU_COMMIT,
        framework_repository=prior["framework_repository"],
        tau_commit=TAU2_COMMIT,
        task_ids=ids,
        task_catalog_sha256=digest(task_file),
        max_parallel_task_processes=8,
        per_task_concurrency=1,
        purpose="Complete post-hardening census: one frozen actor, all 97 tasks, no task-specific changes during execution.",
        selection="All public tasks, including previously inspected development cases. Not an unseen holdout.",
        baseline=str(baseline),
        baseline_manifest_sha256=digest(baseline / "source-manifest.json"),
        treatment=[
            "Previously qualified case/dependency state and removal of lexical response veto",
            "Compact source-span evidence, atomic worksheet retention, simpler optional comparison workflow",
            "Documented created-account field normalization and scoped pre-execution receipts",
            "Generic investigation ownership and needs-first guidance, no customer-text classifiers",
            "Bounded typed transient provider retries and durable deadline classification",
            "Two-hour conversation allowance and 192 runtime steps; public tau step limit unchanged",
        ],
        causal_limit="Combined treatment and larger execution allowance, not a single-variable ablation. Offline regressions prove specific invariants, not task success.",
        qualification="Full immutable-snapshot Linux suite must pass with zero failures, errors or skips, followed by provider connectivity checks before launch.",
        scoring="One attempt per task; official pass^1 only if all 97 receive official binary grades. Operational passes/97 keeps unscored failures in the denominator. No completed-only average or best-of retries.",
        stopping_policy="Continue scored failures and individual execution failures. Pause only shared auth/quota/rate-limit failures, source-integrity changes, low disk or STOP. No whole-task retries or source/config edits in flight.",
        pause_after_consecutive_unscored=None,
        worker_watchdog_seconds=7500,
        minimum_free_disk_bytes=2 * GIB,
        archiving="Lossless diagnostic ZIP, every member hash verified before removing only this new run's expanded diagnostics; official results remain readable.",
        report=str(report),
        qualification_receipt=str(root / "linux-tests.xml"),
    )
    write_json(root / "protocol.json", protocol)
    patch = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or ".vendor" in path.parts
            or path.suffix not in {".py", ".md", ".toml"}
        ):
            continue
        name = str(path.relative_to(root))
        old = baseline / name
        before = old.read_text().splitlines(keepends=True) if old.exists() else []
        patch.extend(
            difflib.unified_diff(
                before, path.read_text().splitlines(keepends=True), f"a/{name}", f"b/{name}"
            )
        )
    (root / "candidate.patch").write_text("".join(patch))
    files = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".vendor" not in p.parts
    }
    write_json(root / "source-manifest.json", {"files": files})
    publish(root, protocol, {"status": "prepared", "rows": {}, "running": {}, "pause_reason": None})
    archive_path = root.parent / (root.name + ".tar.gz")
    with tarfile.open(archive_path, "x:gz") as archive:
        for name in [*files, "source-manifest.json", "progress.json"]:
            archive.add(root / name, arcname=name, recursive=False)
    print(
        json.dumps(
            {
                "root": str(root),
                "manifest_sha256": digest(root / "source-manifest.json"),
                "files": len(files),
                "archive": str(archive_path),
                "archive_sha256": digest(archive_path),
            }
        )
    )


if __name__ == "__main__":
    main()
