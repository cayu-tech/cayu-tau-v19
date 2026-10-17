"""Freeze diagnostics plus the already-qualified actor, without changing census inputs."""

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    project = Path(__file__).resolve().parents[1]
    baseline = project / "artifacts/case-candidate-e5z1j5vf"
    prior_manifest = json.loads((baseline / "source-manifest.json").read_text())["files"]
    for name, expected in prior_manifest.items():
        source = baseline / name
        if not source.resolve().is_relative_to(baseline) or digest(source) != expected:
            raise ValueError("Qualified baseline source changed")
        # No production actor change is smuggled into this diagnostic preparation.
        if name.startswith("src/") and digest(project / name) != expected:
            raise ValueError(f"Existing actor source changed: {name}")
    root = Path(tempfile.mkdtemp(prefix="followup-eval-", dir=project / "artifacts"))
    for directory in ("src", "tests", "scripts"):
        shutil.copytree(
            project / directory,
            root / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(project / name, root / name)
    shutil.copy2(project / "scripts/run_lean_advice_task.py", root / "run_pilot.py")
    shutil.copy2(
        project.parent / "cayu-tau3-choice/artifacts/full97-cloud-20260915/connectivity_check.py",
        root / "connectivity_check.py",
    )
    protocol = json.loads((baseline / "protocol.json").read_text())
    cases = (
        "missing_priority",
        "missing_priority_reversed",
        "explicit_first",
        "explicit_second",
        "sufficient_capacity",
        "independent_pools",
    )
    schedule = [
        {"case": case, "arm": arm}
        for index, case in enumerate(cases)
        for arm in (
            ("baseline", "shared_scope") if index % 2 == 0 else ("shared_scope", "baseline")
        )
    ]
    protocol.update(
        baseline=str(baseline),
        baseline_manifest_sha256=digest(baseline / "source-manifest.json"),
        purpose="Separate shared-quota instruction contrast and live validation of the already-qualified case/evidence actor",
        synthetic_schedule=schedule,
        synthetic_max_sessions=12,
        synthetic_attempts_per_case_arm=1,
        synthetic_max_parallel=1,
        synthetic_conversation_timeout_seconds=480,
        synthetic_worker_watchdog_seconds=540,
        qualification="Actor source unchanged from offline-qualified e5z1j5vf; new diagnostic harness tests and Linux qualification required before model dispatch.",
        stopping_policy="Wait for original census status=completed with zero running tasks. Linux offline tests and existing connectivity preflight must pass. Stop on execution/provider errors; continue ordinary scored failures. Never promote an instruction treatment automatically. One serial attempt per registered case/arm and banking task; no whole-task retries.",
        treatment_scope="shared_scope changes only the synthetic session instructions. Banking tasks use the unchanged qualified actor; no candidate guidance is promoted by this runner.",
        causal_limit="Single observations per fictional case/arm test a proposed mechanism; not a statistical reliability result. Three known banking regressions are not a leaderboard estimate.",
    )
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (root / ".vendor").symlink_to((project / ".vendor").resolve(), target_is_directory=True)
    files = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".vendor" not in p.parts
    }
    (root / "source-manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    with tarfile.open(root.parent / (root.name + ".tar.gz"), "x:gz") as archive:
        for name in [*files, "source-manifest.json"]:
            archive.add(root / name, arcname=name, recursive=False)
    print(
        json.dumps(
            {
                "root": str(root),
                "files": len(files),
                "manifest_sha256": digest(root / "source-manifest.json"),
                "archive_sha256": digest(root.parent / (root.name + ".tar.gz")),
            }
        )
    )


if __name__ == "__main__":
    main()
