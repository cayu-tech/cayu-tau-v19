"""Dedicated-VM serial follow-up runner; no census or other resources are mutated."""

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CENSUS = Path("/Users/talgatt/Desktop/Projects/cayu-tau3-choice/artifacts/full97-0u4lzs9c")


def verify(root):
    root = root.resolve(strict=True)
    for name, expected in json.loads((root / "source-manifest.json").read_text())["files"].items():
        path = (root / name).resolve(strict=True)
        if (
            not path.is_relative_to(root)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Follow-up source changed")


def require_ready(root, census_path):
    verify(root)
    census = json.loads(census_path.read_text())
    summary = census["summary"]
    if (
        census["status"] != "completed"
        or census["running"]
        or summary["finished"] != summary["total"]
        or summary["total"] != 97
    ):
        raise ValueError("Original census must finish before follow-up execution")
    manifest_digest = hashlib.sha256((root / "source-manifest.json").read_bytes()).hexdigest()
    qualification = json.loads((root / "linux-qualified.json").read_text())
    if (
        qualification.get("passed") is not True
        or qualification.get("manifest_sha256") != manifest_digest
    ):
        raise ValueError("Matching Linux qualification is required")
    if json.loads((root / "connectivity.json").read_text()).get("passed") is not True:
        raise ValueError("Connectivity preflight is required")


def write_progress(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2))
    pending.replace(path)


def execute(command, log_path, *, cwd, timeout):
    with log_path.open("x") as log:
        worker = subprocess.Popen(
            command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return worker.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
            return 124


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stage", choices=("synthetic", "banking"), required=True)
    args = parser.parse_args()
    os.umask(0o077)
    root = args.root.resolve(strict=True)
    require_ready(root, CENSUS / "progress.json")
    protocol = json.loads((root / "protocol.json").read_text())
    marker = root / f"{args.stage}-started.json"
    with marker.open("x") as stream:
        json.dump(
            {"started_unix": time.time(), "pid": os.getpid(), "whole_task_retries": 0}, stream
        )
    operations = root / f"{args.stage}-operations"
    operations.mkdir()
    if args.stage == "synthetic":
        schedule = protocol["synthetic_schedule"]
        results = root / "synthetic-results"
        results.mkdir()
    else:
        schedule = [{"task": task} for task in protocol["task_ids"]]
    completed = []
    for index, row in enumerate(schedule):
        verify(root)
        if args.stage == "synthetic":
            command = [
                sys.executable,
                "-m",
                "cayu_tau3_banking.shared_quota_probe",
                "--case",
                row["case"],
                "--arm",
                row["arm"],
                "--output-parent",
                str(results),
            ]
            timeout = protocol["synthetic_worker_watchdog_seconds"]
        else:
            command = [
                sys.executable,
                str(root / "run_pilot.py"),
                "--task",
                row["task"],
                "--openai-api-key-file",
                "/opt/cayu-tau3-credentials/openai_key.txt",
            ]
            timeout = 3900
        print(json.dumps({"starting": row}), flush=True)
        code = execute(command, operations / f"{index:02d}.log", cwd=root, timeout=timeout)
        verify(root)
        completed.append({**row, "exit_code": code, "finished_unix": time.time()})
        write_progress(
            operations / "progress.json",
            {
                "completed": completed,
                "total": len(schedule),
                "status": "failed" if code else "running",
            },
        )
        print(json.dumps(completed[-1]), flush=True)
        if code:
            raise SystemExit(code)
    write_progress(
        operations / "progress.json",
        {"completed": completed, "total": len(schedule), "status": "completed"},
    )


if __name__ == "__main__":
    main()
