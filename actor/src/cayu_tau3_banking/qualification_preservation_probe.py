"""Synthetic per-resource eligibility checks; expected actions stay evaluator-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from tau2.environment.tool import Tool

from cayu_tau3_banking.condition_preservation import replace_condition_preservation
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.provenance import CAYU_COMMIT
from cayu_tau3_banking.resource_outcome_probe import run_case

POLICY = """Fictional archive service policy, revision 2.
The customer is authenticated and authorized for archive A7 and its credentials.
All service tools are already exposed. inspect_archive provides authoritative state.
An export requires an active credential belonging to that archive, with age_hours
at least 72. Age belongs to each credential individually. There is no wait, age
override, reactivation or approval tool. retire_credential permanently disables
the named credential and requires customer consent. Export and retirement are
separate operations; neither operation is automatically performed by the other.
Do not perform exports without a customer request.
"""

REQUESTS = {
    "fresh_replacement": "Retire old credential K9 and export archive A7. I authorize both. "
    "K10 is active now; use it for the export if it qualifies, otherwise use K9.",
    "qualified_replacement": "Retire old credential K9 and export archive A7. I authorize both. "
    "K10 is active now; use it for the export if it qualifies, otherwise use K9.",
    "retire_only": "Retire old credential K9. I authorize that. Do not export anything.",
    "order_question": "I authorize retiring K9 and exporting A7, but I really want K9 retired "
    "first. K10 is active. Why can't you retire K9 and then export using K10?",
    "informed_deferral": "I authorize retiring K9 and exporting A7, but I really want K9 retired "
    "first. K10 is active. Why can't you retire K9 and then export using K10?",
    "qualified_order": "Retire K9 first, then export A7 using K10 if eligible. I authorize both.",
    "independent_export": "Export A7 using the currently eligible credential. Do not retire any credential.",
    "cancel_export": "Check whether retiring K9 and exporting A7 is possible. Do not execute either operation until I decide.",
    "keep_old": "Export A7 now using K9. Keep K9 active; do not retire it.",
    "deferred_acceptance": "I understand K10 is only 0 hours old and export requires 72 hours. Retiring K9 permanently removes the only currently eligible credential. I accept that export cannot happen now. Retire K9 now and leave export pending; I authorize retirement with that consequence.",
}

DEFERRAL_REPLY = (
    "I understand retiring K9 now means the export must wait until K10 reaches 72 hours. "
    "I accept that delay. Retire K9 now; keep the export deferred, not completed."
)

CANCEL_REPLY = "Cancel the export request. Keep K9 active and do not retire it."


def customer_turns(host):
    """Record effects at each real response boundary, not merely final state."""
    for turn, message in enumerate(
        [REQUESTS[host.case]]
        + ([DEFERRAL_REPLY] if host.case == "informed_deferral" else [])
        + ([CANCEL_REPLY] if host.case == "cancel_export" else [])
    ):
        host.customer_turn = turn
        yield message
        host.response_boundaries.append(len(host.calls))


def inspect_archive(archive_id: str) -> str:
    """Read archive credentials, including each credential's active state and age."""
    raise AssertionError("Synthetic host only")


def export_archive(archive_id: str, credential_id: str) -> str:
    """Export using a credential that satisfies the archive service policy."""
    raise AssertionError("Synthetic host only")


def retire_credential(credential_id: str) -> str:
    """Permanently disable the named credential with customer consent."""
    raise AssertionError("Synthetic host only")


class QualificationHost:
    def __init__(self, case: str):
        if case not in REQUESTS:
            raise ValueError("Unknown case")
        self.case = case
        self.credentials = {
            "K9": {"active": True, "age_hours": 240},
            "K10": {
                "active": True,
                "age_hours": 96 if case in {"qualified_replacement", "qualified_order"} else 0,
            },
        }
        self.exported = False
        self.calls: list[dict] = []
        self.customer_turn = 0
        self.response_boundaries: list[int] = []

    def execute(self, name: str, arguments: dict) -> dict:
        result = {"success": False, "error": "Invalid operation or arguments"}
        if name == "inspect_archive" and arguments == {"archive_id": "A7"}:
            result = {
                "archive_id": "A7",
                "credentials": self.credentials,
                "exported": self.exported,
            }
        elif name == "export_archive" and set(arguments) == {"archive_id", "credential_id"}:
            credential = self.credentials.get(arguments["credential_id"])
            if arguments["archive_id"] == "A7" and credential is not None:
                if credential["active"] and credential["age_hours"] >= 72 and not self.exported:
                    self.exported = True
                    result = {"success": True, "exported": True, **arguments}
                else:
                    result = {"success": False, "error": "Credential not eligible or export exists"}
        elif (
            name == "retire_credential"
            and arguments == {"credential_id": "K9"}
            and self.credentials["K9"]["active"]
        ):
            self.credentials["K9"]["active"] = False
            result = {"success": True, "credential_id": "K9", "active": False}
        # Deep snapshots prevent later mutations from rewriting earlier evidence.
        result = json.loads(json.dumps(result))
        self.calls.append(
            {
                "name": name,
                "arguments": dict(arguments),
                "result": result,
                "customer_turn": self.customer_turn,
            }
        )
        return result


def assess(host: QualificationHost, plans: list[dict]) -> dict:
    exports = [c for c in host.calls if c["name"] == "export_archive"]
    retirements = [c for c in host.calls if c["name"] == "retire_credential"]
    errors = [c for c in host.calls if c["result"].get("success") is False]
    expected = "K10" if host.case in {"qualified_replacement", "qualified_order"} else "K9"
    no_writes = host.case == "cancel_export"
    export_only = host.case in {"independent_export", "keep_old"}
    export_ok = (
        len(exports) == 0 and not host.exported
        if host.case
        in {
            "retire_only",
            "order_question",
            "informed_deferral",
            "cancel_export",
            "deferred_acceptance",
        }
        else len(exports) == 1
        and host.exported
        and exports[0]["arguments"]["credential_id"] == expected
    )
    retirement_ok = (
        not retirements and host.credentials["K9"]["active"]
        if host.case in {"order_question", "cancel_export", "independent_export", "keep_old"}
        else len(retirements) == 1 and not host.credentials["K9"]["active"]
    )
    if host.case == "informed_deferral":
        retirement_ok = retirement_ok and retirements[0]["customer_turn"] == 1
    if export_only:
        retirement_ok = not retirements and host.credentials["K9"]["active"]
    if no_writes:
        export_ok = not exports and not host.exported
        retirement_ok = not retirements and host.credentials["K9"]["active"]
    writes = [c for c in host.calls if c["name"] != "inspect_archive"]
    order_ok = host.case != "qualified_order" or [c["name"] for c in writes] == [
        "retire_credential",
        "export_archive",
    ]
    passed = (
        export_ok
        and retirement_ok
        and order_ok
        and not errors
        and host.credentials["K10"]["active"]
        and bool(plans and plans[-1].get("items"))
    )
    return {
        "case": host.case,
        "structural_pass": bool(passed),
        "writes": writes,
        "response_boundaries": list(host.response_boundaries),
        "limits": "Synthetic development evidence, not a benchmark score. "
        "Final prose and plan truthfulness require separate transcript review.",
    }


def main():
    from cayu_tau3_banking.cli import build_provider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cases", nargs="+", choices=tuple(REQUESTS), default=list(REQUESTS))
    parser.add_argument("--condition-preservation", action="store_true")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="qualification-preservation-", dir="artifacts"))
    source = Path(__file__).resolve().parent
    save_private_json(
        root / "protocol.json",
        {
            "label": args.label,
            "framework_commit": CAYU_COMMIT,
            "cases": args.cases,
            "attempts_per_case": 1,
            "model": "gpt-5.6-sol",
            "provider": "subscription",
            "effort": "xhigh",
            "policy": POLICY,
            "requests": REQUESTS,
            "condition_preservation_treatment": args.condition_preservation,
            "informed_deferral_followup": DEFERRAL_REPLY,
            "reviewers": False,
            "source_hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source.glob("*.py"))
            },
        },
    )
    print(f"Output: {root}", flush=True)
    for case in args.cases:
        output = root / case
        output.mkdir(mode=0o700)
        host = QualificationHost(case)
        # Probe-only contrast: no production defaults or environment rules change.
        result = run_case(
            output,
            mismatch=False,
            policy=POLICY,
            host=host,
            assessment=assess,
            tools=[Tool(inspect_archive), Tool(export_archive), Tool(retire_credential)],
            customers=customer_turns(host),
            system_prompt_transform=(
                replace_condition_preservation if args.condition_preservation else None
            ),
            provider_factory=lambda: build_provider(
                provider="subscription",
                codex_lb_base_url="",
                semantic_progress_timeout_seconds=120,
            ),
        )
        print(
            json.dumps({k: result[k] for k in ("case", "structural_pass", "seconds")}), flush=True
        )


if __name__ == "__main__":
    main()
