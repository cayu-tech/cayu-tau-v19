"""Install and run an isolated follow-up on the single authorized VM, after census."""

import argparse
import hashlib
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
FRAMEWORK = BASE / "cayu-case-framework-integration"
ORIGINAL = BASE / "cayu-tau3-choice"
CENSUS = ORIGINAL / "artifacts/full97-0u4lzs9c"
STAGE = Path("/opt/cayu-tau3-staging/followup-20260915")
PIN = "8784d320775ddeedff6c2cb8ccd0ac1941314037"
UV = Path("/opt/cayu-tau3-staging/bootstrap/bin/uv")
TAU = BASE / "cayu-tau3-runtime/.vendor/tau2"


def command(*args, cwd=None, timeout=1800):
    subprocess.run([str(a) for a in args], cwd=cwd, check=True, timeout=timeout)


def identify_vm():
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/id",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        if response.read().decode() != "6392883163248811602":
            raise ValueError("Not the authorized benchmark VM")


def wait_for_census():
    deadline = time.monotonic() + 6 * 3600
    while time.monotonic() < deadline:
        value = json.loads((CENSUS / "progress.json").read_text())
        if (
            value["status"] == "completed"
            and not value["running"]
            and value["summary"]["finished"] == value["summary"]["total"] == 97
        ):
            return
        print(json.dumps({"waiting_for_census": value["summary"]["finished"]}), flush=True)
        time.sleep(60)
    raise TimeoutError("Census did not finish; no follow-up was started")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"followup-eval-[a-z0-9_]+", args.candidate):
        raise ValueError("Unexpected candidate directory")
    os.umask(0o077)
    identify_vm()
    wait_for_census()
    # No environment installation, testing or model dispatch competes with the census.
    if PROJECT.exists() or FRAMEWORK.exists():
        raise ValueError("Isolated installation paths must be unused; never overwrite them")
    archive_path = STAGE / (args.candidate + ".tar.gz")
    bundle_path = STAGE / "framework.bundle"
    for path, digest in ((archive_path, args.archive_sha256), (bundle_path, args.bundle_sha256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Transfer digest mismatch")
    root = PROJECT / "artifacts" / args.candidate
    root.mkdir(parents=True)
    with tarfile.open(archive_path) as archive:
        names = set()
        total = 0
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            total += member.size
            if (
                not member.isfile()
                or not target.is_relative_to(root)
                or member.name in names
                or total > 64 * 1024 * 1024
            ):
                raise ValueError("Unexpected archive member")
            names.add(member.name)
        archive.extractall(root, filter="data")
    command(
        "git", "clone", "--no-hardlinks", BASE / "cayu-custom-source-recall-recovery", FRAMEWORK
    )
    command("git", "-C", FRAMEWORK, "fetch", bundle_path, "agent-case-integration-20260915")
    command("git", "-C", FRAMEWORK, "checkout", "--detach", PIN)
    for directory, expected in (
        (FRAMEWORK, PIN),
        (TAU, "fc0055dc4e0a316c3f83133267fbd6faaa770992"),
    ):
        actual = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
        if actual != expected or dirty:
            raise ValueError("Framework/benchmark provenance mismatch")
    for directory in (PROJECT, root):
        (directory / ".vendor").symlink_to(TAU.parent, target_is_directory=True)
    os.environ["UV_PYTHON_INSTALL_DIR"] = "/opt/cayu-tau3-staging/python"
    os.environ["PYTHONPATH"] = str(root / "src")
    os.environ["CAYU_HOME"] = "/opt/cayu-tau3-credentials/cayu"
    python = PROJECT / ".venv/bin/python"
    command(UV, "venv", "--python", "3.13.1", PROJECT / ".venv")
    command(
        UV,
        "pip",
        "install",
        "--python",
        python,
        "--no-deps",
        "-r",
        ORIGINAL / "artifacts/full97-cloud-20260915/dependencies.txt",
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
    command(
        python,
        "-c",
        "import sys;from pathlib import Path;from cayu_tau3_banking import cli;assert Path(cli.__file__).resolve().is_relative_to(Path(sys.argv[1])/'src')",
        root,
        cwd=root,
    )
    command(
        python,
        "-m",
        "pytest",
        "-q",
        "tests",
        "--junitxml=" + str(root / "linux-tests.xml"),
        cwd=root,
        timeout=2400,
    )
    suites = ET.parse(root / "linux-tests.xml").getroot().findall("testsuite")
    totals = {
        name: sum(int(s.get(name, 0)) for s in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    if totals["tests"] < 1380 or any(totals[k] for k in ("failures", "errors", "skipped")):
        raise ValueError("Incomplete Linux qualification")
    (root / "linux-qualified.json").write_text(
        json.dumps(
            {
                "passed": True,
                "totals": totals,
                "framework_commit": PIN,
                "manifest_sha256": hashlib.sha256(
                    (root / "source-manifest.json").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    command(python, root / "connectivity_check.py", cwd=root, timeout=240)
    for stage in ("synthetic", "banking"):
        command(
            python,
            root / "scripts/run_followup_evaluation.py",
            "--root",
            root,
            "--stage",
            stage,
            cwd=root,
            timeout=4 * 3600,
        )
    print(json.dumps({"status": "completed", "root": str(root)}), flush=True)


if __name__ == "__main__":
    main()
