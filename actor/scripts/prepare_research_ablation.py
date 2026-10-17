"""Freeze a paired whole-task comparison; never inspect task/reference content."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.provenance import assert_source_provenance


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve(strict=True)
    manifest = json.loads((baseline / "source-manifest.json").read_text())["files"]
    for name, checksum in manifest.items():
        source = (baseline / name).resolve(strict=True)
        if not source.is_relative_to(baseline) or digest(source) != checksum:
            raise ValueError("Baseline manifest mismatch")
    protocol = json.loads((baseline / "protocol.json").read_text())
    if protocol["task_ids"] != ["task_064"]:
        raise ValueError("This preregistered comparison uses development task 064")
    assert_source_provenance(project / ".vendor" / "tau2")
    baseline_sources = {name for name in manifest if name.startswith("src/")}
    candidate_sources = {str(p.relative_to(project)) for p in (project / "src").rglob("*.py")}
    if candidate_sources != baseline_sources:
        raise ValueError("Actor source file set changed; this is not an instruction-only treatment")
    changed = [
        name
        for name, checksum in manifest.items()
        if name.startswith("src/") and digest(project / name) != checksum
    ]
    if changed != ["src/cayu_tau3_banking/agent.py"]:
        raise ValueError(f"Expected an instruction-only treatment; changed files: {changed}")
    root = Path(tempfile.mkdtemp(prefix="research-pair-task_064-", dir=project / "artifacts"))
    for arm, source_root in (("baseline", baseline), ("candidate", project)):
        target = root / arm
        target.mkdir()
        for directory in ("src", "tests", "scripts"):
            for source in (source_root / directory).rglob("*.py"):
                destination = target / source.relative_to(source_root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        shutil.copyfile(source_root / "pyproject.toml", target / "pyproject.toml")
        for name in ("run_pilot.py", "monitor.py", "audit_pilot.py"):
            shutil.copyfile(baseline / name, target / name)
        arm_protocol = {
            **protocol,
            "purpose": "Paired exploratory research-workflow ablation on development task 064",
            "arm": arm,
            "selection": "Known development task; no claim of untouched holdout performance",
            "changes": (
                ["Unchanged previously passing actor snapshot"]
                if arm == "baseline"
                else [
                    "Optional worksheet; evidence-first targeted research; decision sequencing last"
                ]
            ),
            "qualification": "852-test baseline; candidate test results recorded with experiment report",
            "pair_directory": str(root),
        }
        save_private_json(target / "protocol.json", arm_protocol)
        files = [
            *sorted(target.rglob("*.py")),
            target / "pyproject.toml",
            target / "protocol.json",
        ]
        save_private_json(
            target / "source-manifest.json",
            {"files": {str(path.relative_to(target)): digest(path) for path in files}},
        )
    save_private_json(
        root / "experiment.json",
        {
            "purpose": "Instruction-only paired exploratory comparison, not statistical proof",
            "baseline_snapshot": str(baseline),
            "changed_actor_files": changed,
            "task": "task_064",
            "seed": 300,
            "trials_per_arm": 1,
            "maximum_parallel_evaluations": 2,
            "models_limits_corpus_and_tools": "identical in both copied runners",
            "whole_task_retries": 0,
            "primary_outcome": "Official final-database reward; incomplete runs are not passes",
            "secondary_outcomes": [
                "wall time",
                "first response time",
                "model calls",
                "tool repairs",
            ],
            "decision": (
                "Consider candidate promising only if it passes and improves paired wall time "
                "by at least 25%; inspect correctness and then validate on task 065. "
                "Do not call one stochastic pair a proven speedup or reliability result."
            ),
            "stopping": "One run per arm; record all failures, do not silently replace them",
        },
    )
    print(root)


if __name__ == "__main__":
    main()
