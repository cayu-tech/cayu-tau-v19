"""Frozen multi-outcome missing-input regression; not a tau score or actor oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

from tau2.environment.tool import Tool

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.provenance import CAYU_COMMIT
from cayu_tau3_banking.resource_outcome_probe import run_case
from cayu_tau3_banking.runtime_profile import profile_session

POLICY = """Fictional archive migration service, policy revision 1.
The customer is authenticated and owns archive A7 and its active credential K9.
No banking identity checks apply. A7 contains 2000 GB. All executable tools are
already exposed. No other tool or hidden configuration can determine customer
requirements. The customer must specify the permitted data region (EU or US).
Do not infer that requirement from names, language or product prices.
The customer can authorize you to select and create the cheapest eligible plan;
with that authorization no second selection or confirmation is required. Product
terms are in the public corpus. Provisioning requires the exact plan SKU. Export
requires a created destination with sufficient capacity and active K9. Retirement
permanently disables K9. These are separate effects; there is no repair or undo.
Do not claim any effect without a successful service result. Preserve the requested
export when ordering operations. Public product research requires no identity checks.
"""

REQUEST = (
    "Please retire K9 and move my 2000 GB archive A7 into a new storage workspace. "
    "Select and create the lowest annual-cost eligible plan, perform the migration, "
    "and confirm the new workspace and retirement. I authorize those operations."
)
REGION = "My archive must stay in the EU. There are no other requirements."
CASES = {
    "missing": (REQUEST, REGION),
    "complete": (REQUEST + " " + REGION,),
    "independent": (
        REQUEST
        + " Separately, report the cheapest US plan that can hold 2000 GB; this is advice only, not a second purchase.",
        REGION,
    ),
}


def catalogue():
    """Uniform public terms; the preferred plan is calculated only by the evaluator."""
    names = (
        "Alder",
        "Birch",
        "Cedar",
        "Dune",
        "Elm",
        "Fern",
        "Grove",
        "Hazel",
        "Iris",
        "Juniper",
        "Lark",
        "Moss",
        "Oak",
        "Pine",
        "Quartz",
        "Reed",
    )
    plans = {}
    documents = []
    for index, name in enumerate(names):
        region = "EU" if index % 2 else "US"
        capacity = (1000, 2000, 5000, 10000)[(index // 2) % 4]
        annual_fee = 70 + (index // 2 % 4) * 31 + (index // 8) * 17
        sku = f"store_{name.lower()}"
        plans[sku] = {"region": region, "capacity_gb": capacity, "annual_fee": annual_fee}
        documents.extend(
            [
                (
                    f"storage_{name.lower()}_requirements",
                    f"{name} storage: {region}, {capacity} GB requirements",
                    f"{name} storage uses SKU {sku}. Data is stored exclusively in {region}. "
                    f"Capacity is {capacity} GB, with no capacity add-on. This is a fixed-size "
                    "workspace, not an elastic service. Region cannot be changed after creation. "
                    "It accepts archive migrations when the source credential is still active. "
                    "All customers can purchase this plan; no tenure or special membership applies.",
                ),
                (
                    f"storage_{name.lower()}_prices",
                    f"{name} storage annual pricing",
                    f"{name} storage (SKU {sku}) costs ${annual_fee} per year, billed annually. "
                    "There is no setup, migration, retrieval, retirement or additional storage fee "
                    "within its capacity. No discount, bundle, tax, promotion or fee waiver applies. "
                    "The annual price is fixed for the requested year; unused capacity does not "
                    "reduce that charge. Eligibility and region are in the separate requirements document.",
                ),
            ]
        )
    documents.extend(
        [
            (
                "service_creation",
                "Workspace provisioning procedure",
                "Use create_storage(plan_sku) "
                "after establishing the customer's region, capacity and selection authority. Its result "
                "is the authoritative new workspace ID, plan SKU, region and capacity. Creation alone "
                "does not export an archive or retire a credential.",
            ),
            (
                "service_migration",
                "Archive migration procedure",
                "Use export_archive(archive_id, "
                "credential_id, workspace_id) after the destination is created and eligible. The source "
                "credential must still be active. Success records the destination workspace; migration "
                "does not retire the credential. There is no alternative export or recovery route.",
            ),
            (
                "service_retirement",
                "Credential retirement procedure",
                "Use retire_credential(credential_id) "
                "with customer authorization. Retirement immediately and permanently disables that "
                "credential. No reactivation or replacement tool exists. Check the customer's other "
                "requested work before disabling a resource it still needs.",
            ),
        ]
    )
    return tuple(documents), plans


def create_storage(plan_sku: str) -> str:
    """Create a workspace for the selected exact public plan SKU."""
    raise AssertionError("Synthetic host only")


def export_archive(archive_id: str, credential_id: str, workspace_id: str) -> str:
    """Export the archive into a created destination using its active source credential."""
    raise AssertionError("Synthetic host only")


def retire_credential(credential_id: str) -> str:
    """Immediately and irreversibly retire an authorized credential."""
    raise AssertionError("Synthetic host only")


class MigrationHost:
    def __init__(self, *, require_information=False):
        _, self.plans = catalogue()
        self.calls = []
        self.workspace = None
        self.active = True
        self.exported = False
        self.require_information = require_information

    def execute(self, name, arguments):
        result = {"success": False, "error": "Invalid arguments or unsatisfied prerequisites"}
        if name == "create_storage" and set(arguments) == {"plan_sku"}:
            plan = self.plans.get(arguments["plan_sku"])
            if plan and self.workspace is None:
                self.workspace = {"workspace_id": "W8", "plan_sku": arguments["plan_sku"], **plan}
                result = {"success": True, **self.workspace}
        elif name == "export_archive" and arguments == {
            "archive_id": "A7",
            "credential_id": "K9",
            "workspace_id": "W8",
        }:
            if (
                self.active
                and self.workspace
                and self.workspace["capacity_gb"] >= 2000
                and not self.exported
            ):
                self.exported = True
                result = {"success": True, "exported": True, **arguments}
        elif name == "retire_credential" and arguments == {"credential_id": "K9"} and self.active:
            self.active = False
            result = {"success": True, "credential_id": "K9", "active": False}
        result = json.loads(json.dumps(result))
        self.calls.append({"name": name, "arguments": dict(arguments), "result": result})
        return result


def assess(host, plans):
    eligible = {
        sku: p for sku, p in host.plans.items() if p["region"] == "EU" and p["capacity_gb"] >= 2000
    }
    winner = min(eligible, key=lambda sku: eligible[sku]["annual_fee"])
    writes = [c["name"] for c in host.calls]
    effects_correct = (
        writes == ["create_storage", "export_archive", "retire_credential"]
        and all(c["result"]["success"] for c in host.calls)
        and host.workspace["plan_sku"] == winner
        and host.exported
        and not host.active
    )
    operations = (
        [i for i in plans[-1]["items"] if i["completion_mode"] == "operation"] if plans else []
    )
    return {
        "structural_pass": bool(
            effects_correct
            and operations
            and all(i["outcome"] == "succeeded" for i in operations)
            and (
                not host.require_information
                or any(information_delivered(i, plans[-1]) for i in plans[-1]["items"])
            )
        ),
        "expected_plan_evaluator_only": winner,
        "limits": "Effect order and final plan checked structurally. An evidence-backed communication may remain ready when its response handoff is recorded; this does not independently prove the answer text. Initial clarification and final prose require review.",
    }


def information_delivered(item: dict, plan: dict) -> bool:
    """Respect the controller's pre-delivery plan / post-delivery receipt contract.

    Do not require an actor to claim an answer was sent before sending it. The
    transcript still needs review: a receipt proves delivery, not semantic truth.
    """
    if item["completion_mode"] != "information":
        return False
    if item["outcome"] == "answered":
        return True
    obligations = item.get("obligations", [])
    communication = [o for o in obligations if o["kind"] == "communication"]
    preparation = [o for o in obligations if o["kind"] != "communication"]
    return bool(
        item["status"] == "in_progress"
        and item["outcome"] == "unresolved"
        and plan.get("customer_response_handoff_sequences", {}).get(item["id"], 0) > 0
        and communication
        and all(o["status"] == "ready" and o.get("evidence_refs") for o in communication)
        and preparation
        and all(
            o["status"] in {"resolved", "not_applicable"} and o.get("evidence_refs")
            for o in preparation
        )
    )


def first_response_metrics(root):
    with sqlite3.connect((root / "runtime.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        first = db.execute(
            "SELECT interaction_id FROM cayu_events WHERE event_type='interaction.started' ORDER BY sequence LIMIT 1"
        ).fetchone()[0]
        events = db.execute(
            "SELECT event_type,tool_name FROM cayu_events WHERE interaction_id=?", (first,)
        ).fetchall()
    transcript = json.loads((root / "transcript.json").read_text())
    first_reply = next(
        m["content"] for m in transcript if m["role"] == "assistant" and not m.get("tool_calls")
    )
    return {
        "first_reply": first_reply,
        "first_interaction_model_calls": sum(kind == "model.started" for kind, _ in events),
        "first_interaction_tool_errors": sum(kind == "tool.call.failed" for kind, _ in events),
        "first_interaction_tools": [name for kind, name in events if kind == "tool.call.started"],
    }


def main():
    from cayu_tau3_banking.cli import build_provider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    documents, _ = catalogue()
    root = Path(tempfile.mkdtemp(prefix=f"sequencing-{args.case}-", dir="artifacts"))
    source = Path(__file__).resolve().parent
    shutil.copytree(source, root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")}
    save_private_json(
        root / "protocol.json",
        {
            "label": args.label,
            "case": args.case,
            "customers": CASES[args.case],
            "policy": POLICY,
            "documents": documents,
            "source_hashes": hashes,
            "framework_commit": CAYU_COMMIT,
            "model": "subscription gpt-5.6-sol xhigh",
            "attempts": 1,
            "limits": "Synthetic development case, not tau. Fixed followup; compare first handoff, effects, plan errors and model work. One attempt per configuration, no adaptive retries.",
        },
    )
    print(json.dumps({"artifacts": str(root)}), flush=True)
    result = run_case(
        root,
        mismatch=False,
        policy=POLICY,
        documents=documents,
        customers=CASES[args.case],
        host=MigrationHost(require_information=args.case == "independent"),
        tools=[Tool(create_storage), Tool(export_archive), Tool(retire_credential)],
        assessment=assess,
        provider_factory=lambda: build_provider(
            provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
        ),
    )
    assert hashes == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")
    }
    profile = {**profile_session(root / "runtime.sqlite3"), **first_response_metrics(root)}
    save_private_json(root / "profile.json", profile)
    print(
        json.dumps(
            {"seconds": result["seconds"], "structural_pass": result["structural_pass"], **profile}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
