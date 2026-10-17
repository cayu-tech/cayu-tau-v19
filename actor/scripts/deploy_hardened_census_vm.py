"""Verify, qualify and run a fresh full census ONLY on the dedicated benchmark VM."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tarfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path("/Users/talgatt/Desktop/Projects")
PROJECT = BASE / "cayu-tau3-case-tracking"
STAGE = Path("/opt/cayu-tau3-staging/hardened-20260916")
UV = Path("/opt/cayu-tau3-staging/bootstrap/bin/uv")
FRAMEWORK = BASE / "cayu-case-framework-integration"
TAU = BASE / "cayu-tau3-runtime/.vendor/tau2"
PIN = "8784d320775ddeedff6c2cb8ccd0ac1941314037"


def command(*args, cwd=None, timeout=1800):
    subprocess.run([str(arg) for arg in args], cwd=cwd, check=True, timeout=timeout)


def test_shards(root, count=4):
    """Every test module exactly once; independent interpreters avoid shared globals."""
    files = sorted(str(path.relative_to(root)) for path in (root / "tests").rglob("test_*.py"))
    if not files or count < 1:
        raise ValueError("Missing tests or invalid shard count")
    return [files[index::count] for index in range(min(count, len(files)))]


def qualify(root, python):
    processes = []
    logs = []
    deadline = time.monotonic() + 2400
    try:
        for index, files in enumerate(test_shards(root)):
            log = (root / f"linux-tests-{index}.log").open("x")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        str(python),
                        "-m",
                        "pytest",
                        "-q",
                        "--tb=short",
                        "-p",
                        "no:cacheprovider",
                        *files,
                        "--junitxml=" + str(root / f"linux-tests-{index}.xml"),
                    ],
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
            )
        codes = [process.wait(timeout=max(1, deadline - time.monotonic())) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in logs:
            log.close()
    combined = ET.Element("testsuites")
    for index in range(len(processes)):
        for suite in ET.parse(root / f"linux-tests-{index}.xml").getroot().findall("testsuite"):
            combined.append(suite)
    ET.ElementTree(combined).write(root / "linux-tests.xml", encoding="utf-8", xml_declaration=True)
    totals = {
        key: sum(int(s.get(key, 0)) for s in combined)
        for key in ("tests", "failures", "errors", "skipped")
    }
    print(json.dumps({"linux_test_totals": totals, "shard_exit_codes": codes}), flush=True)
    if (
        any(codes)
        or totals["tests"] < 1380
        or any(totals[k] for k in ("failures", "errors", "skipped"))
    ):
        raise ValueError("Incomplete Linux qualification; inspect per-shard logs and XML")
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--archive-sha256", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"hardened-full97-[a-z0-9_]+", args.candidate):
        raise ValueError("Unexpected candidate name")
    os.umask(0o077)
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/id",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        if response.read().decode() != "6392883163248811602":
            raise ValueError("Not the authorized benchmark VM")
    census = BASE / "cayu-tau3-choice/artifacts/full97-0u4lzs9c/progress.json"
    previous = json.loads(census.read_text())
    if (
        previous["status"] != "completed"
        or previous["running"]
        or previous["summary"]["finished"] != 97
    ):
        raise ValueError("Original census must be complete and idle")
    archive_path = STAGE / (args.candidate + ".tar.gz")
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != args.archive_sha256:
        raise ValueError("Transfer digest mismatch")
    root = PROJECT / "artifacts" / args.candidate
    root.mkdir(mode=0o700)  # Refuse overwriting any existing source or results.
    with tarfile.open(archive_path) as archive:
        names, size = set(), 0
        for member in archive.getmembers():
            size += member.size
            if (
                not member.isfile()
                or not (root / member.name).resolve().is_relative_to(root)
                or member.name in names
                or size > 64 * 1024 * 1024
            ):
                raise ValueError("Invalid source archive member")
            names.add(member.name)
        archive.extractall(root, filter="data")
    for directory, pin in ((FRAMEWORK, PIN), (TAU, "fc0055dc4e0a316c3f83133267fbd6faaa770992")):
        actual = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
        if actual != pin or dirty:
            raise ValueError("Framework/benchmark provenance mismatch")
    (root / ".vendor").symlink_to(TAU.parent, target_is_directory=True)
    spec = importlib.util.spec_from_file_location("supervisor", root / "run_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify(root)
    os.environ.update(
        UV_PYTHON_INSTALL_DIR="/opt/cayu-tau3-staging/python",
        PYTHONPATH=str(root / "src"),
        CAYU_HOME="/opt/cayu-tau3-credentials/cayu",
        PYTHONUNBUFFERED="1",
    )
    python = root / ".venv/bin/python"
    command(UV, "venv", "--python", "3.13.1", root / ".venv")
    command(
        UV,
        "pip",
        "install",
        "--python",
        python,
        "--no-deps",
        "-r",
        BASE / "cayu-tau3-choice/artifacts/full97-cloud-20260915/dependencies.txt",
    )
    command(
        UV,
        "pip",
        "install",
        "--python",
        python,
        "--no-deps",
        f"cayu @ git+file://{FRAMEWORK}@{PIN}",
    )
    command(UV, "pip", "install", "--python", python, "--no-deps", "-e", TAU, "-e", root)
    totals = qualify(root, python)
    module.verify(root)
    module.write_json(
        root / "linux-qualified.json",
        {
            "passed": True,
            "totals": totals,
            "framework_commit": PIN,
            "manifest_sha256": module.digest(root / "source-manifest.json"),
        },
    )
    command(python, root / "connectivity_check.py", cwd=root, timeout=240)
    receipt = json.loads((root / "connectivity.json").read_text())
    if receipt.get("passed") is not True or len(receipt["results"]) != 3:
        raise ValueError("Connectivity preflight incomplete")
    module.verify(root)
    os.chdir(root)
    os.execv(
        python,
        [
            str(python),
            str(root / "run_benchmark.py"),
            "run",
            "--root",
            str(root),
            "--openai-api-key-file",
            "/opt/cayu-tau3-credentials/openai_key.txt",
        ],
    )


if __name__ == "__main__":
    main()
