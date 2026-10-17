"""Run preregistered task pairs without changing the ongoing full benchmark."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def write(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def verify(root):
    for name, expected in json.loads((root / "source-manifest.json").read_text())["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {name}")


def reviewed_resume(root, config, previous):
    """Continue only unstarted pairs; retain every reviewed failed attempt."""
    if previous.get("status") != "needs_infrastructure_review":
        raise ValueError("Only a reviewed infrastructure stop can resume")
    completed = previous["completed"]
    width = len(config["arms"])
    if not completed or len(completed) % width:
        raise ValueError("Incomplete pair requires separate recovery review")
    count = len(completed) // width
    if count >= len(config["schedule"]):
        raise ValueError("No unstarted pairs remain")
    for index, pair in enumerate(config["schedule"][:count]):
        rows = completed[index * width:(index + 1) * width]
        if {row["arm"] for row in rows} != set(config["arms"]):
            raise ValueError("Prior pair arms do not match schedule")
        for row in rows:
            if any(row[key] != value for key, value in pair.items()):
                raise ValueError("Prior results do not match schedule")
            actor = root / f"{row['arm']}-{row['seed']}"
            assessment = actor / row["task"] / "assessment.json"
            if not assessment.exists() or json.loads(assessment.read_text())["rows"] != row["rows"]:
                raise ValueError("Prior assessment missing or changed")
    for pair in config["schedule"][count:]:
        for arm in config["arms"]:
            if (root / f"{arm}-{pair['seed']}" / pair["task"]).exists():
                raise ValueError("An allegedly unstarted task already has output")
    return completed, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume-reviewed", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    os.umask(0o077)
    config = json.loads((root / "comparison.json").read_text())
    qualification = json.loads((root / "vm-qualification.json").read_text())
    if not qualification.get("passed"):
        raise ValueError("VM qualification required")
    for arm in config["arms"]:
        for seed in config["seeds"]:
            verify(root / f"{arm}-{seed}")
    completed = []
    start_index = 0
    status = root / "status.json"
    if status.exists():
        if not args.resume_reviewed:
            raise ValueError("Comparison already started; inspect evidence before resuming")
        previous = json.loads(status.read_text())
        completed, start_index = reviewed_resume(root, config, previous)
        with (root / f"reviewed-stop-{time.time_ns()}.json").open("x") as receipt:
            json.dump(previous, receipt, indent=2)
    elif args.resume_reviewed:
        raise ValueError("No prior stopped comparison to resume")
    for index, pair in enumerate(config["schedule"]):
        if index < start_index:
            continue
        workers = []
        write(status, {"status": "starting_pair", "pair": pair, "completed": completed,
                       "expected": len(config["schedule"]) * 2, "updated_unix": time.time()})
        try:
            arms = config["arms"] if index % 2 == 0 else list(reversed(config["arms"]))
            for arm in arms:
                actor = root / f"{arm}-{pair['seed']}"
                verify(actor)
                output = actor / pair["task"]
                if output.exists():
                    raise ValueError("Existing attempt will not be overwritten or selectively retried")
                log = (root / f"pair-{index:02d}-{arm}.log").open("x")
                process = subprocess.Popen(
                    [config["python"][arm], str(actor / "run_pilot.py"),
                     "--task", pair["task"], "--openai-api-key-file", config["api_key_file"]],
                    cwd=actor, env={**os.environ, "PYTHONPATH": str(actor / "src")},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                )
                workers.append((arm, actor, process, log))
            started = time.monotonic()
            while any(p.poll() is None for _, _, p, _ in workers):
                write(status, {"status": "running", "pair": pair, "completed": completed,
                               "expected": len(config["schedule"]) * 2,
                               "workers": [{"arm": a, "pid": p.pid, "exit_code": p.poll()}
                                           for a, _, p, _ in workers], "updated_unix": time.time()})
                if (root / "STOP").exists() or time.monotonic() - started > 7900:
                    raise RuntimeError("Comparison stopped or exceeded its outer watchdog")
                time.sleep(15)
            failed = False
            for arm, actor, process, _ in workers:
                verify(actor)
                assessment = actor / pair["task"] / "assessment.json"
                rows = json.loads(assessment.read_text())["rows"] if assessment.exists() else []
                failed |= process.returncode != 0 or not rows
                completed.append({**pair, "arm": arm, "exit_code": process.returncode, "rows": rows})
            write(status, {"status": "needs_infrastructure_review" if failed else "pair_completed",
                           "completed": completed, "expected": len(config["schedule"]) * 2,
                           "updated_unix": time.time()})
            if failed:
                return  # Preserve both arms; never retry until a preferred score appears.
        finally:
            for _, _, process, log in workers:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                log.close()
    write(status, {"status": "completed", "completed": completed,
                   "expected": len(config["schedule"]) * 2, "updated_unix": time.time()})


if __name__ == "__main__":
    main()
