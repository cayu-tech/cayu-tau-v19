"""Qualify and launch a new immutable census on the authorized VM only."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    args = parser.parse_args()
    os.umask(0o077)
    name = args.archive.name.removesuffix(".tar.gz")
    if not re.fullmatch(r"inventory-full97-[a-z0-9_]+", name):
        raise ValueError("Unexpected candidate name")
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/id",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        if response.read().decode() != "6392883163248811602":
            raise ValueError("Not the authorized VM")
    if not os.environ.get("CODEX_LB_API_KEY") or os.geteuid() == 0:
        raise ValueError("Run as benchmark owner with explicit LB credential")
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != args.archive_sha256:
        raise ValueError("Archive digest mismatch")
    base = Path("/Users/talgatt/Desktop/Projects")
    root = base / "cayu-tau3-development/artifacts" / name
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700)  # Never overwrite or resume an existing attempt.
    with tarfile.open(args.archive) as archive:
        names, size = set(), 0
        for member in archive.getmembers():
            size += member.size
            if (
                not member.isfile()
                or not (root / member.name).resolve().is_relative_to(root)
                or member.name in names
                or size > 64 * 1024**2
            ):
                raise ValueError("Unsafe archive")
            names.add(member.name)
        archive.extractall(root, filter="data")
    protocol = json.loads((root / "protocol.json").read_text())
    sys.path.insert(0, str(root))
    runner = load("run_benchmark", root / "run_benchmark.py")
    runner.verify(root)
    tau = Path(protocol["tau_repository"])
    if (
        subprocess.check_output(["git", "-C", str(tau), "rev-parse", "HEAD"], text=True).strip()
        != protocol["tau_commit"]
    ):
        raise ValueError("Benchmark pin mismatch")
    if subprocess.check_output(
        ["git", "-C", str(tau), "status", "--porcelain", "--untracked-files=no"]
    ).strip():
        raise ValueError("Modified benchmark checkout")
    if (
        runner.digest(tau / "data/tau2/domains/banking_knowledge/tasks.json")
        != protocol["task_catalog_sha256"]
    ):
        raise ValueError("Task catalog mismatch")
    (root / ".vendor").symlink_to(tau.parent, target_is_directory=True)
    os.environ.update(
        UV_PYTHON_INSTALL_DIR="/opt/cayu-tau3-staging/python",
        PYTHONPATH=str(root / "src"),
        CAYU_HOME=str(root / "unused-subscription-home"),
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        PYTHONUNBUFFERED="1",
    )
    # The VM has the explicitly authorized exact private-source mirror, not
    # GitHub login credentials. Use it only for this installation's Git fetch.
    # Keep the declared repository URL and verify the content-addressed pin.
    mirror = base / "cayu-case-framework-integration"
    if (
        subprocess.check_output(["git", "-C", str(mirror), "rev-parse", "HEAD"], text=True).strip()
        != protocol["framework_commit"]
    ):
        raise ValueError("Authorized framework mirror pin mismatch")
    if subprocess.check_output(
        ["git", "-C", str(mirror), "status", "--porcelain", "--untracked-files=no"]
    ).strip():
        raise ValueError("Modified authorized framework mirror")
    install_env = {
        **os.environ,
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": f"url.file://{mirror}.insteadOf",
        "GIT_CONFIG_VALUE_0": "https://github.com/cayu-tech/cayu.git",
    }
    uv = "/opt/cayu-tau3-staging/bootstrap/bin/uv"
    python = root / ".venv/bin/python"
    commands = [
        [uv, "venv", "--python", "3.13.1", str(root / ".venv")],
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "-r",
            str(base / "cayu-tau3-choice/artifacts/full97-cloud-20260915/dependencies.txt"),
        ],
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "cayu @ git+https://github.com/cayu-tech/cayu.git@" + protocol["framework_commit"],
        ],
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "-e",
            str(tau),
            "-e",
            str(root),
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True, timeout=1800, env=install_env)
    deployment = load("deployment", root / "scripts/deploy_hardened_census_vm.py")
    totals = deployment.qualify(root, python)
    if totals != {"tests": protocol["expected_tests"], "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError("Linux qualification incomplete")
    runner.verify(root)
    runner.write_json(
        root / "linux-qualified.json",
        {
            "passed": True,
            "totals": totals,
            "manifest_sha256": runner.digest(root / "source-manifest.json"),
        },
    )
    subprocess.run(
        [str(python), str(root / "connectivity_check.py")], cwd=root, check=True, timeout=240
    )
    receipt = json.loads((root / "connectivity.json").read_text())
    if receipt.get("passed") is not True or len(receipt["results"]) != 3:
        raise ValueError("Connectivity preflight incomplete")
    # Reexec with the qualified interpreter; no model requests before all gates.
    os.chdir(root)
    os.execv(
        python, [str(python), str(root / "scripts/start_inventory_census.py"), "--root", str(root)]
    )


if __name__ == "__main__":
    main()
