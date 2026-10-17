"""Freeze the qualified current checkout for two preregistered development tasks."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.provenance import CAYU_COMMIT, TAU2_COMMIT, assert_source_provenance

TASKS = ["task_066", "task_067"]


def main():
    project = Path(__file__).resolve().parents[1]
    previous = project / "artifacts/parameter-binding-task_065-zdq53mm4"
    qualified = project / "artifacts/product-packets-m26d9_9i"
    qualification = json.loads((qualified / "protocol.json").read_text())
    conclusion = json.loads((qualified / "conclusion.json").read_text())
    if conclusion["offline_tests"]["passed"] != 1047 or conclusion["active_runs"]:
        raise ValueError("Completed qualification is required")
    for name, digest in qualification["source_hashes"].items():
        if (
            hashlib.sha256((project / "src/cayu_tau3_banking" / name).read_bytes()).hexdigest()
            != digest
        ):
            raise ValueError(f"Actor changed after qualification: {name}")
    tau = (project / ".vendor/tau2").resolve()
    assert_source_provenance(tau)
    # Only task IDs are used for preregistration, never scenario/answer content.
    ids = {
        t["id"]
        for t in json.loads((tau / "data/tau2/domains/banking_knowledge/tasks.json").read_text())
    }
    if not set(TASKS) <= ids:
        raise ValueError("Preselected IDs are missing")
    root = Path(tempfile.mkdtemp(prefix="product-packet-batch-", dir=project / "artifacts"))
    for name in ("src", "tests", "scripts"):
        shutil.copytree(
            project / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(project / name, root / name)
    for name in ("run_pilot.py", "monitor.py", "audit_pilot.py"):
        shutil.copy2(previous / name, root / name)
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
            "purpose": "Broader frozen development evaluation of the current complete-product-read agent",
            "selection": "Next two numeric IDs after deferred task_065, selected before reading their scenarios "
            "or reference actions in this batch. Earlier public benchmark development may have exposed them; "
            "not a pristine holdout or leaderboard estimate.",
            "max_parallel_task_processes": 2,
            "per_task_concurrency": 1,
            "product_packet_limits": {"documents": 16, "complete_result_bytes": 72000},
            "qualification": {
                "path": str(qualified),
                "tests": conclusion["offline_tests"],
                "conclusion_sha256": hashlib.sha256(
                    (qualified / "conclusion.json").read_bytes()
                ).hexdigest(),
            },
            "actor_definition": "Exact qualified normal checkout. Do NOT copy the experimental condition-preservation "
            "paragraph from task_065. No unqualified prompt-only selection guidance.",
            "comparison_limits": "Different tasks and normal-checkout prompt; not a single-factor comparison to task_065.",
            "stopping_policy": "One attempt per selected task; no whole-task retries or mid-run changes. "
            "Stop new launches on infrastructure errors. In-flight isolated tasks may finish within their original limits.",
            "review": "After completion inspect official reward, delivered advice/consent, real mutations, "
            "tool/provider errors, source coverage and timing. Reference actions may be inspected only after scoring, "
            "never supplied to the actor.",
            "limits_note": "Same application-owned one-hour absolute conversation deadline; not a tau-imposed deadline.",
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
