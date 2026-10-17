"""Freeze one model-configuration qualification; never change production defaults."""

import argparse
import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.product_navigation import RECOVERY_LIMITS
from cayu_tau3_banking.provenance import CAYU_COMMIT, TAU2_COMMIT, assert_source_provenance


def replace_model(arguments, model, effort):
    result = list(arguments)
    for flag, expected, value in (
        ("--model", "gpt-5.6-sol", model),
        ("--reasoning-effort", "xhigh", effort),
    ):
        if result.count(flag) != 1:
            raise ValueError("Expected one baseline model/effort setting")
        i = result.index(flag) + 1
        if i >= len(result) or result[i] != expected:
            raise ValueError("Unexpected baseline setting")
        result[i] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["gpt-5.6-sol", "gpt-6-astra"], required=True)
    parser.add_argument("--effort", choices=["medium", "high", "xhigh"], required=True)
    parser.add_argument("--qualification-receipt", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--purpose",
        default="Full-task qualification following matched model/effort next-decision probes",
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    previous = (args.baseline or project / "artifacts/lean-advice-task-wo692nzl").resolve()
    receipt = (
        args.qualification_receipt
        or (project / "artifacts/lean-guidance-all-qualified-20260914.xml")
    ).resolve()
    suites = ET.parse(receipt).getroot().findall("testsuite")
    counts = {
        name: sum(int(s.get(name, "0")) for s in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    if counts["tests"] < 1100 or any(counts[k] for k in ("failures", "errors", "skipped")):
        raise ValueError("Existing successful full-suite qualification is required")
    tau = (project / ".vendor/tau2").resolve()
    assert_source_provenance(tau)
    root = Path(tempfile.mkdtemp(prefix="model-task-", dir=project / "artifacts"))
    for name in ("src", "tests", "scripts"):
        shutil.copytree(
            project / name, root / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(project / name, root / name)
    shutil.copy2(project / "scripts/run_lean_advice_task.py", root / "run_pilot.py")
    shutil.copy2(project / "scripts/monitor_frozen_task.py", root / "monitor.py")
    shutil.copy2(receipt, root / "prior-offline-tests.xml")
    (root / ".vendor").symlink_to((project / ".vendor").resolve(), target_is_directory=True)
    protocol = json.loads((previous / "protocol.json").read_text())
    protocol.update(
        task_ids=["task_066"],
        framework_commit=CAYU_COMMIT,
        tau_commit=TAU2_COMMIT,
        tau_repository=str(tau),
        actor=f"subscription {args.model} {args.effort}",
        purpose=args.purpose,
        selection="Known-failing development task_066, not a holdout or aggregate score.",
        comparison="Current generic guidance and model configuration; historical full run used earlier guidance. This fresh episode is not an isolated model-only causal comparison with that historical run. The separately recorded portable probes changed only model and effort.",
        baseline=str(previous / "task_066"),
        compactor=f"Existing runtime uses the actor model family ({args.model}) at fixed low effort. No new reviewer or helper is enabled.",
        qualification=counts,
        qualification_receipt=str(receipt),
        qualification_note="Caller-supplied full-suite receipt; source remains separately frozen and checked."
        if args.qualification_receipt
        else "Prior full-suite receipt for current actor source; diagnostic-only additions qualified separately. This is not a fresh whole-suite run.",
        cli_arguments=replace_model(protocol["cli_arguments"], args.model, args.effort),
    )
    protocol["navigation_limits"] = {
        **protocol.get("navigation_limits", {}),
        "exact_source_recovery": list(RECOVERY_LIMITS),
    }
    save_private_json(root / "protocol.json", protocol)
    save_private_json(
        root / "source-manifest.json",
        {
            "files": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            }
        },
    )
    print(root)


if __name__ == "__main__":
    main()
