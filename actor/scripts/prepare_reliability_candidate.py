"""Package the reliability candidate without launching or modifying any run."""

import difflib
import json
import shutil
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

from run_full_benchmark import digest, verify, write_json


def main():
    project = Path(__file__).resolve().parents[1]
    baseline = project.parent / "cayu-tau3-development/artifacts/hardened-full97-3btzc1d8"
    verify(baseline)
    receipts = {
        "local-tests.xml": Path("/private/tmp/tau-reliability-tests.xml"),
        "local-recheck.xml": Path("/private/tmp/tau-reliability-recheck.xml"),
    }
    outcomes = {}
    for receipt in receipts.values():
        for case in ET.parse(receipt).iter("testcase"):
            outcomes[(case.get("classname"), case.get("name"))] = not any(
                case.find(kind) is not None for kind in ("failure", "error", "skipped"))
    if not outcomes or not all(outcomes.values()):
        raise ValueError("Local qualification contains unresolved failures, errors or skips")
    root = project / "artifacts/reliability-20260922"
    root.mkdir(parents=True, exist_ok=False)
    for name in ("src", "tests", "scripts"):
        shutil.copytree(project / name, root / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(project / name, root / name)
    shutil.copy2(project / "scripts/run_lean_advice_task.py", root / "run_pilot.py")
    shutil.copy2(project / "scripts/run_full_benchmark.py", root / "run_benchmark.py")
    shutil.copy2(baseline / "connectivity_check.py", root / "connectivity_check.py")
    protocol = json.loads((baseline / "protocol.json").read_text())
    protocol.update(
        baseline=str(baseline),
        baseline_manifest_sha256=digest(baseline / "source-manifest.json"),
        purpose="Qualification of generic reliability fixes; no new benchmark started by packaging.",
        treatment=[
            "Bounded retry eligibility for typed interrupted subscription streams with local functions only",
            "Apply stateless summary recovery to the subscription provider",
            "Return typed recall outages as repairable knowledge-search errors",
            "Preserve typed runtime interruption evidence",
            "Terminate planning stalls through benchmark agent_stop with usage and reason retained",
            "Select provider diagnostic envelopes by modification time rather than UUID ordering",
        ],
        causal_limit="No demonstrated score improvement; offline regression tests only until live qualification.",
        qualification="Local full suite receipt included; isolated VM suite and connectivity remain required before benchmark launch.",
        scoring="Use upstream task grades. Report scored failures and unscored errors separately; passes/97 is the operational lower bound. This protocol is our configuration, not a statement of official submission rules.",
        report=str(project.parent / "agent-documents/tau-reliability-20260922.html"),
        qualification_receipt=str(root / "linux-tests.xml"),
    )
    write_json(root / "protocol.json", protocol)
    for name, receipt in receipts.items():
        shutil.copy2(receipt, root / name)
    write_json(root / "local-qualification.json", {
        "unique_tests_passing_after_recheck": len(outcomes),
        "receipts_in_order": list(receipts),
        "note": "Initial suite lacked the vendor symlink and loaded an outdated assertion before usage metadata was added. Original failures are retained; affected tests were rerun after correction.",
    })
    shutil.copy2("/private/tmp/tau-reliability-final-focused.log", root / "recovery-tests.log")
    patch = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".toml", ".md"}:
            relative = str(path.relative_to(root))
            old = baseline / relative
            patch.extend(difflib.unified_diff(
                old.read_text().splitlines(keepends=True) if old.exists() else [],
                path.read_text().splitlines(keepends=True), f"a/{relative}", f"b/{relative}"))
    (root / "candidate.patch").write_text("".join(patch))
    files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(root / "source-manifest.json", {"files": files})
    write_json(root / "progress.json", {"status": "prepared", "rows": {}, "running": {}, "pause_reason": None})
    verify(root)
    archive_path = root.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "x:gz") as archive:
        for name in [*files, "source-manifest.json", "progress.json"]:
            archive.add(root / name, arcname=name, recursive=False)
    print(json.dumps({"root": str(root), "archive": str(archive_path),
                      "manifest_sha256": digest(root / "source-manifest.json"),
                      "archive_sha256": digest(archive_path)}))


if __name__ == "__main__":
    main()
