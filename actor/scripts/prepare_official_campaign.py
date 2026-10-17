"""Freeze sources for the authorized four-trial custom banking submission run."""

import json
import shutil
import tarfile
from pathlib import Path

from run_full_benchmark import digest, verify, write_json


def main():
    project = Path(__file__).resolve().parents[1]
    baseline = project.parent / "cayu-tau3-memory-20260926/artifacts/memory-v12-screen3-20260926"
    verify(baseline)
    root = project / "artifacts/official-banking-4trials-v14-repair1-20260926"
    root.mkdir(parents=True, exist_ok=False)
    for name in ("src", "scripts", "tests"):
        shutil.copytree(project / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(project / name, root / name)
    protocol = json.loads((baseline / "protocol.json").read_text())
    options = protocol["cli_arguments"]
    for flag, value in (("--num-trials", "4"), ("--max-concurrency", "8"),
                        ("--runtime-model-max-attempts", "10"),
                        ("--runtime-conversation-timeout-seconds", "9000"),
                        ("--simulation-timeout-seconds", "9300"),
                        ("--completion-timeout-seconds", "9150")):
        options[options.index(flag) + 1] = value
    protocol.update(
        candidate_revision="release-v14-repair1-20260926",
        actor_snapshot_reference=str(baseline),
        trials=4, expected_trajectories=388,
        baseline=str(baseline), baseline_manifest_sha256=digest(baseline / "source-manifest.json"),
        purpose="Full banking evaluation for a custom leaderboard submission; not yet maintainer accepted.",
        submission_type="custom", retrieval_config="cayu_hybrid",
        upstream_retrieval_config="no_knowledge (Cayu supplies its own hybrid retrieval)",
        runner="Unmodified upstream run_domain; full base task set, no task filters; eight concurrent fresh agents.",
        seed_policy="Unmodified upstream random.Random seeding from 300 for all four trials.",
        whole_task_retries="Upstream checkpoint resume for infrastructure only, at most three completed retry rounds; scored failures never retried.",
        treatment=[*protocol["treatment"],
                   "Complete documents delivered in single/batched search results retained for exact recovery; snippets never promoted to complete evidence",
                   "Official supervisor pauses only explicit typed quota/auth outages, not generic transient throttling",
                   "One canonical typed adverse-interaction constraint drives admission; no duplicate conflict facts",
                   "Empty/zero consent placeholders normalize to no choice with a receipt; optional current-turn metadata and checked plan revision parameter reduce repair overhead",
                   "Scoped recovery of existing reconciliation notes and explicit applicability/missing-evidence guidance",
                   "User-authorized 10 total transient model attempts (up to nine retries) and 150-minute agent task budget, with 152.5-minute callback and 155-minute simulation safety timeouts"],
        qualification="All local regressions and the complete VM suite, then actor/simulator/embedding connectivity.",
        stopping_policy="Pause on explicit shared auth/quota, low disk or STOP. Recheck connectivity before resuming. Generic throttling uses bounded recovery. Retain failed checkpoints. No scored-result retry.",
        archiving="Preserve old runs and gzip every checkpoint before infrastructure resume; retain original diagnostics.",
        causal_limit="New combined reliability treatment. No score or eligibility acceptance claimed in advance.",
    )
    protocol["limits"].update(simulation_seconds=9300, adapter_callback_seconds=9150,
        agent_conversation_seconds=9000, transient_model_attempts=10, summary_model_attempts=4, recall_source_seconds=45, recall_overall_seconds=60, recall_semantic_seconds=30)
    protocol.update(scoring="All 388 trajectories; upstream scores unchanged, including scored failures. Preserve original infrastructure attempts and disclose retries.",
                    selection="All 97 tasks in the pinned upstream base split, all four trials; no diagnostic task filtering.",
                    acceptance="Await V14 qualification and limited live comparison; this prepared snapshot is not a launch or a maintainer acceptance.")
    for key in ("report", "qualification_receipt", "worker_watchdog_seconds", "max_parallel_task_processes", "per_task_concurrency", "task_ids", "diagnostic_failures", "regression_controls", "exploratory_tasks", "excluded_before_launch", "pause_after_consecutive_unscored"):
        protocol.pop(key, None)
    write_json(root / "protocol.json", protocol)
    files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(root / "source-manifest.json", {"files": files})
    verify(root)
    archive_path = root.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "x:gz") as archive:
        for name in [*files, "source-manifest.json"]:
            archive.add(root / name, arcname=name, recursive=False)
    print(json.dumps({"root": str(root), "archive": str(archive_path),
        "manifest_sha256": digest(root / "source-manifest.json"), "archive_sha256": digest(archive_path)}))


if __name__ == "__main__":
    main()
