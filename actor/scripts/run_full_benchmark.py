"""Freeze and supervise a one-attempt census, without changing the acting agent.

The supervisor uses only the standard library. Individual isolated workers use
the existing frozen launcher. Completed diagnostics are losslessly archived;
official results remain directly readable. No task is silently retried.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

GIB = 1024**3
DIAGNOSTIC_DIRECTORIES = ("contexts", "provider-diagnostics", "sessions")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_text(path, content):
    fd, temporary = tempfile.mkstemp(prefix=".benchmark-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify(root):
    for name, expected in json.loads((root / "source-manifest.json").read_text())["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError("Frozen source mismatch")


def prepare(project, baseline, report):
    protocol = json.loads((baseline / "protocol.json").read_text())
    # This census deliberately uses the currently qualified actor, without fixes.
    current = {
        str(p.relative_to(project / "src")): digest(p) for p in (project / "src").rglob("*.py")
    }
    previous = {
        str(p.relative_to(baseline / "src")): digest(p) for p in (baseline / "src").rglob("*.py")
    }
    if current != previous:
        raise ValueError("Actor differs from the last qualified snapshot")
    task_file = Path(protocol["tau_repository"]) / "data/tau2/domains/banking_knowledge/tasks.json"
    ids = sorted(t["id"] for t in json.loads(task_file.read_text()))
    if (
        len(ids) != 97
        or len(set(ids)) != 97
        or not all(re.fullmatch(r"task_\d{3}", t) for t in ids)
    ):
        raise ValueError("Expected the exact 97-task census")
    if report.exists():
        raise FileExistsError("Use a new report path; do not overwrite another run")
    root = Path(tempfile.mkdtemp(prefix="full97-", dir=project / "artifacts"))
    for name in ("src", "tests", "scripts"):
        shutil.copytree(
            project / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(project / name, root / name)
    shutil.copy2(project / "scripts/run_lean_advice_task.py", root / "run_pilot.py")
    shutil.copy2(project / "scripts/run_full_benchmark.py", root / "run_benchmark.py")
    (root / ".vendor").symlink_to((project / ".vendor").resolve(), target_is_directory=True)
    receipt = Path(protocol["qualification_receipt"])
    shutil.copy2(receipt, root / "prior-offline-tests.xml")
    for key in ("baseline", "comparison", "qualification_note"):
        protocol.pop(key, None)
    protocol.update(
        task_ids=ids,
        task_catalog_sha256=digest(task_file),
        purpose="Current-agent score across every public banking task; no fixes or tuning during this census.",
        selection="All 97 public task IDs in numeric order. Previously used development tasks are included; not an unseen holdout.",
        max_parallel_task_processes=2,
        per_task_concurrency=1,
        actor_snapshot_reference=str(baseline),
        actor_matches_reference=True,
        scoring="One fresh trial per task. Official pass^1 only if every task receives an official binary reward. Also report operational passes/97 with unscored failures explicitly separated. Never report a completed-only average as the full score.",
        stopping_policy="No whole-task retries. Agent failures remain recorded. Pause new launches for shared authentication/quota/rate-limit problems, three consecutive unscored tasks, low disk, source changes, or a STOP file. In-flight tasks finish under their original limits.",
        worker_watchdog_seconds=3900,
        minimum_free_disk_bytes=GIB,
        archiving="After process exit, zip contexts, provider diagnostics and sessions; verify every member SHA-256 before removing only those new run's expanded files. Official results remain outside the archive.",
        report=str(report.resolve()),
        qualification_receipt=str(root / "prior-offline-tests.xml"),
    )
    write_json(root / "protocol.json", protocol)
    files = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    write_json(root / "source-manifest.json", {"files": files})
    state = {
        "status": "prepared",
        "created_unix": time.time(),
        "rows": {},
        "running": {},
        "pause_reason": None,
    }
    publish(root, protocol, state)
    return root


def summarize(ids, rows):
    if set(rows) - set(ids):
        raise ValueError("Unregistered task result")
    rewards = [r.get("reward") for r in rows.values()]
    if any(
        r is not None and (type(r) not in (int, float) or not math.isfinite(r) or r not in (0, 1))
        for r in rewards
    ):
        raise ValueError("Unexpected nonbinary reward")
    scored = sum(r is not None for r in rewards)
    passed = sum(r == 1 for r in rewards)
    finished = len(rows) == len(ids)
    return {
        "total": len(ids),
        "finished": len(rows),
        "scored": scored,
        "passed": passed,
        "failed_scored": scored - passed,
        "unscored": len(rows) - scored,
        "official_pass1_percent": 100 * passed / len(ids)
        if finished and scored == len(ids)
        else None,
        "operational_pass_percent": 100 * passed / len(ids) if finished else None,
        "confirmed_pass_lower_bound_percent": 100 * passed / len(ids),
    }


def publish(root, protocol, state):
    state["updated_unix"] = time.time()
    state["summary"] = summarize(protocol["task_ids"], state["rows"])
    write_json(root / "progress.json", state)
    s = state["summary"]
    score = "Pending — do not quote a partial result as the 97-task score."
    if s["official_pass1_percent"] is not None:
        score = f"Official one-trial pass^1: {s['passed']}/97 = {s['official_pass1_percent']:.2f}%."
    elif s["operational_pass_percent"] is not None:
        score = f"Operational success: {s['passed']}/97 = {s['operational_pass_percent']:.2f}%; {s['unscored']} tasks unscored. An entirely scored official pass^1 is unavailable."
    table = []
    for task in protocol["task_ids"]:
        row = state["rows"].get(task)
        running = state["running"].get(task)
        if row:
            outcome = (
                "Pass"
                if row["reward"] == 1
                else "Fail"
                if row["reward"] == 0
                else "Unscored failure"
            )
            detail = html.escape(row["termination"])
            if row.get("failure_category"):
                detail += " / " + html.escape(row["failure_category"])
            duration = f"{row['wall_seconds'] / 60:.1f} min"
        elif running:
            outcome, detail = "Running", ""
            duration = f"{(time.time() - running['started_unix']) / 60:.1f} min"
        else:
            outcome, detail, duration = "Queued", "", ""
        table.append(
            f"<tr><td>{task}</td><td>{outcome}</td><td>{duration}</td><td>{detail}</td></tr>"
        )
    artifact_link = html.escape(
        os.path.relpath(root / "progress.json", Path(protocol["report"]).parent)
    )
    protocol_link = html.escape(
        os.path.relpath(root / "protocol.json", Path(protocol["report"]).parent)
    )
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="60">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Cayu τ³ — current 97-task score</title><style>
body{{font:16px/1.6 system-ui;background:#f4f6f8;color:#172a37;margin:0}}main{{max-width:1000px;margin:auto;padding:28px}}
section{{background:white;padding:20px;margin:18px 0;border-radius:10px}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:7px;border-bottom:1px solid #ddd}}code{{overflow-wrap:anywhere}}a{{color:#145c95}}</style></head><body><main>
<h1>Cayu τ³-Banking — all 97 tasks</h1><p>Frozen current-agent census · No leaderboard submission</p>
<section><h2>{html.escape(state["status"].replace("_", " ").title())}</h2><p>{score}</p>
<p>Finished {s["finished"]}/97 · Passed {s["passed"]} · Scored failures {s["failed_scored"]} · Unscored failures {s["unscored"]} · Running {len(state["running"])}</p>
<p>Updated {time.strftime("%Y-%m-%d %H:%M:%S %Z")}. {html.escape(state.get("pause_reason") or "")}</p></section>
<section><h2>Protocol</h2><p>Actor: {html.escape(protocol.get("actor", "See frozen protocol"))}. Simulator: {html.escape(protocol.get("simulator", "See frozen protocol"))}. {protocol["max_parallel_task_processes"]} isolated workers; one attempt per task; no whole-task retries or mid-run changes. Application conversation limit: {protocol.get("limits", {}).get("simulation_seconds", "see protocol")} seconds; worker watchdog: {protocol["worker_watchdog_seconds"]} seconds. These are application limits, not benchmark-imposed deadlines.</p>
<p>{html.escape(protocol.get("purpose", "Measure the frozen candidate."))} This includes previously studied development tasks and is not an unseen holdout. One trial measures pass^1 only, not pass^2 through pass^4 reliability. Unscored errors are disclosed separately and never dropped from the operational denominator.</p>
<p>Supervision runs on the execution host, independently of the dashboard. Diagnostic archives are lossless and checksummed; no older artifacts are removed. Credential values are not copied into protocol or progress files. Actual monetary costs may be unavailable from the providers; no spend total is inferred.</p>
<p><a href="{artifact_link}">Machine-readable progress</a> · <a href="{protocol_link}">Frozen protocol</a></p></section>
<section><h2>Task results</h2><table><thead><tr><th>Task</th><th>Outcome</th><th>Elapsed</th><th>Termination</th></tr></thead><tbody>{"".join(table)}</tbody></table></section>
</main></body></html>"""
    atomic_text(Path(protocol["report"]), report)


def archive_diagnostics(directory):
    """Only completed, newly produced diagnostic files; verify before unlinking."""
    paths = []
    folders = []
    for name in DIAGNOSTIC_DIRECTORIES:
        folder = directory / name
        if not folder.exists():
            continue
        if folder.is_symlink():
            raise ValueError("Refuse a diagnostic symlink")
        folders.append(folder)
        for path in folder.rglob("*"):
            if path.is_symlink():
                raise ValueError("Refuse a diagnostic symlink")
            if path.is_file():
                paths.append(path)
    if not paths:
        return None
    hashes = {str(p.relative_to(directory)): digest(p) for p in paths}
    archive = directory / "diagnostics.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zipped:
        os.chmod(archive, 0o600)
        for path in paths:
            zipped.write(path, str(path.relative_to(directory)))
    with zipfile.ZipFile(archive) as zipped:
        if set(zipped.namelist()) != set(hashes) or len(zipped.namelist()) != len(hashes):
            raise ValueError("Archive file-set mismatch")
        for name, expected in hashes.items():
            with zipped.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError("Archive content mismatch")
    receipt = {
        "archive": archive.name,
        "sha256": digest(archive),
        "files": hashes,
        "verified": True,
    }
    write_json(directory / "archive-manifest.json", receipt)
    # The actor process has exited. These exact files belong only to this task;
    # all bytes now have a verified recoverable copy in the archive.
    for path in paths:
        if digest(path) != hashes[str(path.relative_to(directory))]:
            raise ValueError("Diagnostic changed during archiving")
    for path in paths:
        path.unlink()
    for folder in folders:
        for sub in sorted(
            (p for p in folder.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True
        ):
            sub.rmdir()
        folder.rmdir()
    return {"archive": archive.name, "sha256": receipt["sha256"], "files": len(hashes)}


def shared_failure(directory):
    """Latch only explicit capacity/auth failures, never recoverable throttling.

    Worker-owned bounded retry/backoff handles generic 429 and overload. A
    terminal unscored result still invokes the separate configured pause policy.
    Historical retry diagnostics must not permanently stop healthy dispatch.
    Never return exception messages or credentials.
    """
    paths = [directory / "error.json"]
    paths.extend((directory / "provider-diagnostics").glob("*/*.error.json"))
    codes = (
        "insufficient_quota",
        "invalid_api_key",
        "authenticationerror",
        "authentication_error",
        "permissiondeniederror",
        "usage_limit_reached",
        "billing_hard_limit_reached",
        "account_deactivated",
        "organization_deactivated",
    )
    errors = []
    for path in paths:
        if path.is_file():
            try:
                errors.append(json.loads(path.read_text()))
            except (FileNotFoundError, json.JSONDecodeError):
                # Live workers may still be writing. Retry on the next poll.
                continue
    official = directory / "official/results.json"
    if official.is_file():
        try:
            simulations = json.loads(official.read_text()).get("simulations", [])
        except (FileNotFoundError, json.JSONDecodeError):
            simulations = []
        for simulation in simulations:
            if simulation.get("termination_reason") == "infrastructure_error":
                info = simulation.get("info") or {}
                errors.append({key: info[key] for key in ("error_type", "error") if key in info})
    for error in errors:
        if not isinstance(error, dict):
            continue
        nested = error.get("error")
        identities = [error.get(key) for key in ("type", "error_type", "code")]
        if isinstance(nested, dict):
            identities.extend(nested.get(key) for key in ("type", "code"))
        for identity in identities:
            if isinstance(identity, str) and identity.lower() in codes:
                return f"Shared credential/quota failure: {identity.lower()}"
    return None


def active_shared_failure(root, tasks):
    """Observe provider diagnostics before workers exit; never terminate workers."""
    for task in tasks:
        reason = shared_failure(root / task)
        if reason:
            return reason
    return None


def finish_task(root, task, code, started, watchdog=False):
    directory = root / task
    row = {
        "task": task,
        "reward": None,
        "termination": "worker_timeout" if watchdog else "launcher_error",
        "failure_category": "worker_watchdog" if watchdog else "unclassified",
        "exit_code": code,
        "wall_seconds": time.time() - started,
    }
    official = directory / "official/results.json"
    assessment = directory / "assessment.json"
    if assessment.exists() and official.exists():
        data = json.loads(assessment.read_text())
        sims = json.loads(official.read_text())["simulations"]
        if data.get("source_unchanged") is not True or len(data["rows"]) != 1 or len(sims) != 1:
            raise ValueError("Invalid single-task assessment")
        result, sim = data["rows"][0], sims[0]
        reward = (sim.get("reward_info") or {}).get("reward")
        if result["task"] != task or sim["task_id"] != task or reward != result["reward"]:
            raise ValueError("Assessment/official result mismatch")
        row.update(
            reward=reward,
            termination=result["termination"],
            failure_category="worker_watchdog" if watchdog else result.get("failure_category"),
            official_duration_seconds=result["duration_seconds"],
            official_sha256=digest(official),
        )
    reason = shared_failure(directory)
    row["archive"] = archive_diagnostics(directory) if directory.exists() else None
    write_json(root / f"{task}-outcome.json", row)
    return row, reason


def run(root, key_file):
    root = root.resolve()
    verify(root)
    protocol = json.loads((root / "protocol.json").read_text())
    state = json.loads((root / "progress.json").read_text())
    if state["status"] != "prepared":
        raise ValueError("This run has already started; do not duplicate attempts")
    if not key_file.is_file():
        raise ValueError("Credential file missing")
    # Children, not the supervisor's JSON/logs, read the selected secret file.
    lock = (root / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state.update(status="running", started_unix=time.time(), supervisor_pid=os.getpid())
    publish(root, protocol, state)
    wake = (
        subprocess.Popen(
            ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if sys.platform == "darwin"
        else None
    )
    active = {}
    queue = list(protocol["task_ids"])
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONUNBUFFERED": "1"}
    consecutive_unscored = 0
    try:
        while queue or active:
            if (root / "STOP").exists():
                state["pause_reason"] = "STOP requested; in-flight tasks may finish."
            if shutil.disk_usage(root).free < protocol["minimum_free_disk_bytes"]:
                state["pause_reason"] = "Low disk reserve; no new tasks launched."
            if not state["pause_reason"]:
                state["pause_reason"] = active_shared_failure(root, active)
            for task, (process, stream, began) in list(active.items()):
                watchdog = time.time() - began > protocol["worker_watchdog_seconds"]
                if process.poll() is None and watchdog:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                del active[task]
                del state["running"][task]
                verify(root)
                row, reason = finish_task(root, task, code, began, watchdog)
                state["rows"][task] = row
                consecutive_unscored = consecutive_unscored + 1 if row["reward"] is None else 0
                limit = protocol.get("pause_after_consecutive_unscored", 3)
                if reason or (limit is not None and consecutive_unscored >= limit):
                    state["pause_reason"] = (
                        reason
                        or f"{limit} consecutive unscored task(s); inspect infrastructure before continuing."
                    )
                print(
                    json.dumps(
                        {
                            "finished": task,
                            "reward": row["reward"],
                            "termination": row["termination"],
                        }
                    ),
                    flush=True,
                )
            while (
                queue
                and len(active) < protocol["max_parallel_task_processes"]
                and not state["pause_reason"]
            ):
                # Also inspect in-flight workers before filling a free slot.
                # This closes dispatch on quota before task cleanup completes.
                reason = active_shared_failure(root, active)
                if reason:
                    state["pause_reason"] = reason
                    break
                verify(root)
                task = queue.pop(0)
                if (root / task).exists():
                    raise ValueError("Refuse an existing task output")
                stream = (root / f"{task}-launcher.log").open("xb")
                os.chmod(stream.name, 0o600)
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(root / "run_pilot.py"),
                        "--task",
                        task,
                        "--openai-api-key-file",
                        str(key_file.resolve()),
                    ],
                    cwd=root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                began = time.time()
                active[task] = (process, stream, began)
                state["running"][task] = {"pid": process.pid, "started_unix": began}
                print(json.dumps({"starting": task, "pid": process.pid}), flush=True)
            state["status"] = (
                "pausing"
                if state["pause_reason"] and active
                else "paused"
                if state["pause_reason"]
                else "running"
            )
            publish(root, protocol, state)
            if not active and state["pause_reason"]:
                break
            if queue or active:
                time.sleep(10)
        if not queue and not active:
            state["status"] = "completed"
        publish(root, protocol, state)
        print(json.dumps({"status": state["status"], **state["summary"]}), flush=True)
    except BaseException as error:
        # Preserve task process IDs and evidence; never silently restart a task.
        state.update(
            status="supervisor_error",
            pause_reason=f"Supervisor stopped: {type(error).__name__}. Existing worker PIDs remain recorded; inspect before any restart.",
        )
        publish(root, protocol, state)
        raise
    finally:
        if not active and wake is not None:
            wake.terminate()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--baseline", type=Path, required=True)
    preparation.add_argument("--report", type=Path, required=True)
    execution = sub.add_parser("run")
    execution.add_argument("--root", type=Path, required=True)
    execution.add_argument("--openai-api-key-file", type=Path, required=True)
    launch = sub.add_parser("launch")
    launch.add_argument("--root", type=Path, required=True)
    launch.add_argument("--openai-api-key-file", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(
            prepare(
                Path(__file__).resolve().parents[1], args.baseline.resolve(), args.report.resolve()
            )
        )
    elif args.command == "run":
        run(args.root, args.openai_api_key_file)
    else:
        root = args.root.resolve()
        verify(root)
        if json.loads((root / "progress.json").read_text())["status"] != "prepared":
            raise ValueError("Run already started")
        with (root / "supervisor.log").open("xb") as stream:
            os.chmod(stream.name, 0o600)
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(root / "run_benchmark.py"),
                    "run",
                    "--root",
                    str(root),
                    "--openai-api-key-file",
                    str(args.openai_api_key_file.resolve()),
                ],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps({"supervisor_pid": process.pid, "root": str(root)}))


if __name__ == "__main__":
    main()
