"""Freeze an unchanged actor with only the existing batching CLI option enabled."""

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    if re.fullmatch(r"task_[0-9]{3}", args.task) is None:
        parser.error("Expected task_NNN")
    project = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve(strict=True)
    files = json.loads((baseline / "source-manifest.json").read_text())["files"]
    for name, expected in files.items():
        path = (baseline / name).resolve(strict=True)
        if not path.is_relative_to(baseline) or digest(path) != expected:
            raise ValueError("Baseline integrity mismatch")
    root = Path(tempfile.mkdtemp(prefix=f"tool-batching-{args.task}-", dir=project / "artifacts"))
    for directory in ("src", "tests", "scripts"):
        for source in (baseline / directory).rglob("*.py"):
            target = root / source.relative_to(baseline)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for name in ("run_pilot.py", "monitor.py", "audit_pilot.py", "pyproject.toml"):
        shutil.copyfile(baseline / name, root / name)
    shutil.copyfile(project / "scripts/run_batched_pilot.py", root / "run_batched_pilot.py")
    protocol = json.loads((baseline / "protocol.json").read_text())
    protocol.update(
        {
            "purpose": "Single-factor model tool-proposal batching evaluation",
            "baseline_snapshot": str(baseline),
            "task_ids": [args.task],
            "selection": (
                "Known task 064 development regression"
                if args.task == "task_064"
                else "Next task selected numerically without new scenario/reference inspection"
            ),
            "changes": ["Existing --runtime-parallel-tools flag enabled; identical actor source"],
            "qualification": "Existing multi-call authorization tests rerun; report records results",
            "runtime_parallel_tools": True,
            "banking_execution": "Existing tool safety declarations and guards unchanged",
            "whole_task_retries": 0,
        }
    )
    save_private_json(root / "protocol.json", protocol)
    paths = [*sorted(root.rglob("*.py")), root / "pyproject.toml", root / "protocol.json"]
    save_private_json(
        root / "source-manifest.json",
        {"files": {str(path.relative_to(root)): digest(path) for path in paths}},
    )
    print(root)


if __name__ == "__main__":
    main()
