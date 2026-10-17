"""Freeze one development trial only after the four synthetic binding controls pass."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.parameter_binding_probe import CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    previous = project / "artifacts/condition-preservation-task_065-zLYvjf"
    qualification = args.qualification.resolve()
    results = []
    for case in CASES:
        path = qualification / case / "result.json"
        row = json.loads(path.read_text())
        if row["structural_pass"] is not True:
            raise ValueError("Synthetic gate failed")
        results.append(
            {
                "case": case,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "seconds": row["seconds"],
            }
        )
    old = json.loads((previous / "protocol.json").read_text())
    root = Path(tempfile.mkdtemp(prefix="parameter-binding-task_065-", dir=project / "artifacts"))
    for name in ("src", "tests", "scripts"):
        shutil.copytree(
            project / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(project / name, root / name)
    # Preserve the already-qualified cross-outcome treatment exactly. The normal
    # checkout still uses the baseline; no live source or previous result is edited.
    for name in ("src/cayu_tau3_banking/agent.py", "tests/test_dependency_preservation_probe.py"):
        shutil.copy2(previous / name, root / name)
    for name in ("run_pilot.py", "monitor.py", "audit_pilot.py"):
        shutil.copy2(previous / name, root / name)
    (root / ".vendor").symlink_to((project / ".vendor").resolve(), target_is_directory=True)
    protocol = {
        key: old[key]
        for key in (
            "task_ids",
            "seed",
            "trials",
            "whole_task_retries",
            "framework_commit",
            "actor",
            "simulator",
            "retrieval",
            "runtime",
            "limits",
            "navigation_limits",
            "credential_handling",
            "selection",
        )
    }
    protocol.update(
        {
            "purpose": "One frozen development trial of public parameter validation plus optional-field discipline; not a leaderboard claim",
            "changes": [
                "Source-digest-checked public savings name format before dispatch; no alias map or auto-rewrite",
                "Generic instruction: optional fields express operational distinctions, not decorative narrative; meaningful overrides remain allowed",
            ],
            "retained": "Prior source-complete condition-preservation paragraph, same models/effort/corpus/retrieval, no semantic reviewers",
            "qualification": results,
            "prior_trial": str(previous),
            "limits_note": "Same application-owned one-hour absolute conversation deadline; not a tau-imposed deadline",
            "stopping_policy": "Exactly one task/trial, zero automatic whole-task retries, no mid-run changes or oracle-fed corrections",
            "interpretation_limits": "Public format validity does not prove the correct product was selected. Valid optional reasons remain valid even if exact DB scoring penalizes them. The actor never receives a hidden closure default.",
        }
    )
    save_private_json(root / "protocol.json", protocol)
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    save_private_json(root / "source-manifest.json", {"files": files})
    print(json.dumps({"frozen": str(root), "files": len(files)}))


if __name__ == "__main__":
    main()
