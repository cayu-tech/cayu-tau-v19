"""Freeze the already-qualified V9 actor for one fresh attempt on all 97 tasks."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    project = Path(__file__).resolve().parents[1]
    baseline = project / "artifacts/live-screen-v9-20260925/candidate-300"
    manifest = json.loads((baseline / "source-manifest.json").read_text())["files"]
    assert all(digest(baseline / name) == expected for name, expected in manifest.items())
    protocol = json.loads((baseline / "protocol.json").read_text())
    task_file = Path(protocol["tau_repository"]) / "data/tau2/domains/banking_knowledge/tasks.json"
    assert digest(task_file) == protocol["task_catalog_sha256"]
    ids = sorted(row["id"] for row in json.loads(task_file.read_text()))
    assert len(ids) == len(set(ids)) == 97
    root = project / "artifacts/full97-v9-20260925"
    root.mkdir(exist_ok=False)
    for name in manifest:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline / name, target)
    # Supervisor/reporting only; the acting code and per-task launcher stay exact.
    runner = (baseline / "scripts/run_full_benchmark.py").read_text()
    runner = runner.replace("Official one-trial pass^1:", "One-trial validation score:")
    runner = runner.replace("An entirely scored official pass^1 is unavailable.", "A fully scored one-trial result is unavailable.")
    runner = runner.replace('"Three consecutive unscored tasks; inspect infrastructure before continuing."',
                            'f"{limit} consecutive unscored task(s); inspect infrastructure before continuing."')
    (root / "run_benchmark.py").write_text(runner)
    protocol.update(
        task_ids=ids, expected_trajectories=97,
        purpose="One-trial 97-task validation of the unchanged V9 actor; not a four-trial official campaign or leaderboard submission.",
        selection="All 97 public tasks, including prior development tasks and recorded ambiguities. No task exclusions or reused screen results.",
        causal_limit="Previously inspected public tasks, not an unseen holdout; one stochastic attempt per task.",
        actor_source_identical=True, actor_snapshot_reference=str(baseline),
        baseline_manifest_sha256=digest(baseline / "source-manifest.json"),
        max_parallel_task_processes=8, per_task_concurrency=1,
        runner="Eight isolated task processes supervised independently of SSH.",
        whole_task_retries=0,
        scoring="Upstream binary grades; report passes/97, scored failures and unscored failures separately. A retry is never substituted silently or counted as best-of success.",
        stopping_policy="Continue scored failures. Pause new scheduling after the first unscored result, shared auth/quota/rate-limit failure, low disk or STOP; allow in-flight tasks to finish. No automatic retries or in-flight candidate edits.",
        pause_after_consecutive_unscored=1,
        worker_watchdog_seconds=8400, minimum_free_disk_bytes=8 * 1024**3,
        archiving="Only new completed diagnostics are losslessly ZIP-compressed; verify every member hash before removing expanded duplicates. Preserve all prior runs and readable official results.",
        report=str(root / "report.html"),
        qualification="Actor is byte-identical to the V9 screen that passed 130 Linux tests and 4/4 tasks. Census requires manifest/catalog checks and Linux supervisor tests before dispatch.",
        acceptance="Report the full 97-task result and failure categories before selecting an official four-trial campaign.",
    )
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    actor_files = {n: h for n, h in manifest.items() if n.startswith("src/") or n == "run_pilot.py"}
    assert all(digest(root / name) == expected for name, expected in actor_files.items())
    files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}
    (root / "source-manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    (root / "progress.json").write_text(json.dumps({"status": "prepared", "rows": {}, "running": {}, "pause_reason": None}, indent=2) + "\n")
    receipt = {"root": str(root), "candidate_manifest_sha256": digest(root / "source-manifest.json"),
               "baseline_manifest_sha256": protocol["baseline_manifest_sha256"],
               "actor_source_identical": True, "actor_files_verified": len(actor_files),
               "tasks": 97, "seed": protocol["seed"], "workers": 8, "trials": 1}
    (root / "freeze-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    archive = project / "artifacts/full97-v9-20260925.tar.gz"
    with tarfile.open(archive, "x:gz") as output:
        output.add(root, arcname=root.name)
    print(json.dumps({**receipt, "archive": str(archive)}))


if __name__ == "__main__":
    main()
