"""Freeze a preselected bounded evaluation without task/reference content inspection."""

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.provenance import CAYU_COMMIT, assert_source_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--development-regression", action="store_true")
    parser.add_argument("--infrastructure-restart-of", type=Path)
    parser.add_argument("--validation-after", type=Path)
    parser.add_argument("--prior-run", type=Path)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--change", action="append", default=[])
    parser.add_argument("--qualification-file", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch(r"task_[0-9]{3}", args.task) is None:
        parser.error("Expected an exact task_NNN identifier")
    project = Path(__file__).resolve().parents[1]
    restart = args.infrastructure_restart_of
    if restart is not None:
        restart = restart.resolve(strict=True)
        prior_task = restart / args.task
        rows = json.loads((prior_task / "assessment.json").read_text())["rows"]
        if len(rows) != 1 or rows[0]["termination"] != "infrastructure_error":
            parser.error("Restart requires one infrastructure-error assessment")
        for path in (prior_task / "sessions").glob("*.sqlite3"):
            with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
                if db.execute(
                    "SELECT count(*) FROM cayu_events WHERE event_type='model.started'"
                ).fetchone()[0]:
                    parser.error("Restart requires no acting-model starts")
        prior_manifest = json.loads((restart / "source-manifest.json").read_text())["files"]
        for name, digest in prior_manifest.items():
            if (
                name.startswith("src/")
                and hashlib.sha256((project / name).read_bytes()).hexdigest() != digest
            ):
                parser.error("Actor source changed since the infrastructure failure")
    assert_source_provenance(project / ".vendor" / "tau2")
    qualification = json.loads(args.qualification_file.read_text())
    if qualification.get("framework_commit") != CAYU_COMMIT:
        parser.error("Qualification must identify the exact current framework commit")
    base = project / "artifacts/admission-bound-evaluation-xmwUbCwr"
    root = Path(
        tempfile.mkdtemp(prefix=f"independent-budgets-{args.task}-", dir=project / "artifacts")
    )
    for directory in ("src", "tests", "scripts"):
        for source in (project / directory).rglob("*.py"):
            target = root / source.relative_to(project)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for filename in ("run_pilot.py", "audit_pilot.py", "monitor.py"):
        shutil.copyfile(base / filename, root / filename)
    shutil.copyfile(project / "pyproject.toml", root / "pyproject.toml")
    protocol = json.loads((base / "protocol.json").read_text())
    protocol.update(
        {
            "purpose": args.purpose,
            "task_ids": [args.task],
            "framework_commit": CAYU_COMMIT,
            "selection": (
                f"{args.task} development regression after its previous run and diagnostic audit, including reference-state inspection; NOT an untouched holdout. Production treatment must not embed reference actions or products."
                if args.development_regression
                else f"{args.task} selected numerically without new task/reference inspection; not a pristine historical holdout claim."
            ),
            "changes": [
                f"Framework exact integration {CAYU_COMMIT}; upstream ancestry and retained extensions recorded in qualification",
                "Compactor enables completed-blank recovery, maximum two attempts; unknown-error allowance remains one",
                "Independent discovery, read and comparison budgets; batched product discovery",
                "Immediate new-customer tool exposure refresh",
                "Numeric coverage and incomplete-comparison instructions",
                "Generic needs elicitation in shared agent prompt; direct answers when needs are sufficient",
                "Bounded exact locator retention outside lossy prose summaries and exact recovery mappings",
                "Independent multi-outcome handoff baseline passed without production behavioral changes",
                "Up to four independent source-revalidated worksheets, 12000 bytes each / 24000 aggregate",
                "Comparison check allowance six per customer / 24 total, supporting partial results across categories",
                "Compactor explicitly uses low reasoning effort; actor remains xhigh",
                "Compact navigation budget replies; no duplicated full locator inventories",
                "Ask for decision-changing customer inputs before broad research; represent actual dependencies rather than speculative research obligations",
                "Reserve comparison worksheets for multi-product quantitative decisions, not every read or clarification",
                "Preserve the customer's stated optimization objective instead of broadening it with optional adjacent benefits",
                "Validate and calculate literal grouped numbers without requiring incompatible normalized quotes; reject partial-number matches",
                "Explain that private comparison call IDs are not authoritative bank-tool results; use returned knowledge references",
                "Rejected literal quotes return a bounded access-checked excerpt from the already-delivered revision; no automatic correction or acceptance",
                "Invalid private arguments identify schema paths and constraints instead of an opaque schema failure",
                "Evidence-first targeted research; comparison worksheet optional rather than mandatory",
                "Needs sequencing follows tool instructions; no runtime batching enabled in this candidate",
            ],
            "qualification": "Prior single-worksheet live checks passed without rereads; offline multi-scope, source validation and compaction regressions tested. Qualification counts recorded separately. This does not establish banking success.",
            "navigation_limits": {"discovery": [8, 32], "read": [12, 48], "check": [6, 24]},
            "max_concurrent_processes": 1,
            "whole_task_retries": 0,
            "treatment_changes": args.change,
            "credential_handling": "Caller loads user-specified local secret into child environment only; never copied to artifacts",
        }
    )
    if restart is not None:
        protocol["infrastructure_restart"] = {
            "prior_artifacts": str(restart),
            "reason": "Sandbox DNS prevented simulator connectivity before any acting-model start; verified SDK HTTPS transport works outside sandbox. Restart with network permission and unchanged actor source.",
            "prior_failure_retained": True,
            "task_or_reference_content_inspected": False,
        }
        protocol["stopping_policy"] += (
            " This launch is an explicitly recorded pre-actor infrastructure restart, not an adaptive retry after a task outcome."
        )
    if args.validation_after is not None:
        prior = args.validation_after.resolve(strict=True)
        prior_rows = json.loads((prior / args.task / "assessment.json").read_text())["rows"]
        if len(prior_rows) != 1 or prior_rows[0]["termination"] != "infrastructure_error":
            parser.error("Validation-after requires a retained infrastructure-failure assessment")
        protocol["prior_failed_validation"] = {
            "artifacts": str(prior),
            "failure_retained": True,
            "new_treatment": "Current-main framework integration with bounded compaction recovery",
            "not_a_resume_or_replacement": True,
        }
    if args.prior_run is not None:
        prior = args.prior_run.resolve(strict=True)
        assessment = json.loads((prior / args.task / "assessment.json").read_text())
        protocol["prior_development_run"] = {
            "artifacts": str(prior),
            "assessment": assessment,
            "original_retained": True,
            "not_a_resume_or_replacement": True,
        }
    save_private_json(root / "protocol.json", protocol)
    save_private_json(root / "qualification.json", qualification)
    paths = [
        *sorted(root.rglob("*.py")),
        root / "pyproject.toml",
        root / "protocol.json",
        root / "qualification.json",
    ]
    save_private_json(
        root / "source-manifest.json",
        {
            "files": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            },
        },
    )
    print(root)


if __name__ == "__main__":
    main()
