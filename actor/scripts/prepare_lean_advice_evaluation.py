"""Freeze a lean-advice real-task candidate after successful offline qualification."""

import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.provenance import CAYU_COMMIT, TAU2_COMMIT, assert_source_provenance

TASKS = ["task_066"]


def main():
    project = Path(__file__).resolve().parents[1]
    previous = project / "artifacts/lean-advice-task-ugwr69rh"
    receipt = project / "artifacts/merged-cayu-offline-qualified-20260914.xml"
    suites = ET.parse(receipt).getroot().findall("testsuite")
    counts = {
        name: sum(int(s.get(name, "0")) for s in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    if counts["tests"] < 1000 or any(counts[k] for k in ("failures", "errors", "skipped")):
        raise ValueError("Successful full offline qualification is required")
    tau = (project / ".vendor/tau2").resolve()
    assert_source_provenance(tau)
    root = Path(tempfile.mkdtemp(prefix="lean-advice-task-", dir=project / "artifacts"))
    for name in ("src", "tests", "scripts"):
        shutil.copytree(
            project / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(project / name, root / name)
    shutil.copy2(project / "scripts/run_lean_advice_task.py", root / "run_pilot.py")
    shutil.copy2(project / "scripts/monitor_frozen_task.py", root / "monitor.py")
    shutil.copy2(receipt, root / "offline-tests.xml")
    (root / ".vendor").symlink_to((project / ".vendor").resolve(), target_is_directory=True)
    old = json.loads((previous / "protocol.json").read_text())
    protocol = {
        k: old[k]
        for k in (
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
        )
    }
    protocol.update(
        {
            "task_ids": TASKS,
            "framework_commit": CAYU_COMMIT,
            "tau_commit": TAU2_COMMIT,
            "tau_repository": str(tau),
            "corpus_embedding_cache": str(
                project / "artifacts/section-embedding-validation/run-eo81fta9"
            ),
            "purpose": "Requalify the existing lean-advice agent after pinning merged bounded-compaction continuation PR #1710",
            "selection": "Known-failing development task_066 chosen before reading this run's scenario or oracle. Not holdout.",
            "comparison": "Merged framework update only for actor behavior; unchanged actor model, prompts, retrieval, tool configuration, diagnostic capture and limits. Framework contains intervening main changes; not a single-fix ablation.",
            "baseline": str(previous / "task_066"),
            "advice_mode": "lean",
            "max_parallel_task_processes": 1,
            "qualification": counts,
            "stopping_policy": "One seed, one attempt, no whole-task retries, no mid-run actor changes or deadline extensions.",
            "limits_note": "One-hour absolute conversation deadline and research limits are application-owned, not tau requirements.",
            "review": "Inspect official score, public dialogue, real calls/results, context and timing after completion. Never supply reference actions to actor.",
            "cli_arguments": old["cli_arguments"],
        }
    )
    save_private_json(root / "protocol.json", protocol)
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    save_private_json(root / "source-manifest.json", {"files": files})
    print(json.dumps({"frozen": str(root), "files": len(files), "tasks": TASKS}), flush=True)


if __name__ == "__main__":
    main()
