"""Run a frozen development screen in separate task processes; preserve every attempt."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_full_benchmark import atomic_text, digest, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--openai-api-key-file", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    verify(root)
    protocol = json.loads((root / "protocol.json").read_text())
    # Exclusive receipt prevents accidental duplicate launch or automatic retries.
    with (root / "screen-started.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "started_unix": time.time()}, stream)
    pending = list(protocol["task_ids"])
    active = {}
    rows = {}
    pause_reason = None
    while pending or active:
        for task, (process, log, started) in list(active.items()):
            watchdog = time.monotonic() - started > protocol["worker_watchdog_seconds"]
            if process.poll() is None and watchdog:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            if process.poll() is None:
                continue
            log.close()
            verify(root)
            assessment = root / task / "assessment.json"
            official = root / task / "official/results.json"
            row = {"task": task, "reward": None, "termination": "worker_watchdog" if watchdog else "launcher_error"}
            if assessment.exists() and official.exists():
                result = json.loads(assessment.read_text())
                simulations = json.loads(official.read_text())["simulations"]
                assert result["source_unchanged"] and len(result["rows"]) == len(simulations) == 1
                measured = result["rows"][0]
                sim = simulations[0]
                assert measured["task"] == sim["task_id"] == task
                assert measured["reward"] == (sim.get("reward_info") or {}).get("reward")
                row.update(measured, official_sha256=digest(official))
            row.update(exit_code=process.returncode, supervisor_wall_seconds=time.monotonic() - started)
            rows[task] = row
            del active[task]
            print(json.dumps(row), flush=True)
            if row["reward"] is None:
                pause_reason = "Unscored attempt: inspect before launching remaining tasks."
        if (root / "STOP").exists():
            pause_reason = "STOP requested; active attempts may finish."
        while pending and not pause_reason and len(active) < protocol["max_parallel_task_processes"]:
            verify(root)
            task = pending.pop(0)
            if (root / task).exists():
                raise RuntimeError("Refuse to overwrite an existing task")
            log = (root / f"{task}-launcher.log").open("xb")
            process = subprocess.Popen(
                [sys.executable, str(root / "run_pilot.py"), "--task", task,
                 "--openai-api-key-file", str(args.openai_api_key_file)],
                cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONUNBUFFERED": "1"},
            )
            active[task] = (process, log, time.monotonic())
        state = {"status": "running" if active else "paused" if pending else "finished",
                 "rows": rows, "running": list(active), "pending": pending,
                 "pause_reason": pause_reason, "updated_unix": time.time(),
                 "scope": "Development screen; not an official benchmark score"}
        atomic_text(root / "progress.json", json.dumps(state, indent=2) + "\n")
        if not active:
            break
        time.sleep(3)


if __name__ == "__main__":
    main()
