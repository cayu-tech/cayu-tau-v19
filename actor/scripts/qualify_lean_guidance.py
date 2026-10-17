"""Frozen native-runtime guidance qualification; fictional service, never tau input.

Comparison cases contain a handcrafted long context, not a replay or live compaction
test. Expected rankings are evaluator-only; prose is reviewed after completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from tau2.environment.tool import Tool

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.parameter_binding_probe import Host
from cayu_tau3_banking.provenance import CAYU_COMMIT, assert_source_provenance
from cayu_tau3_banking.resource_outcome_probe import run_case

COMPARISON_CASES = ("helpful", "costly", "ineligible", "declined", "non_stacking", "deductions")
OPERATION_CASES = ("ordinary_note", "requested_note", "policy_note")
MODEL = "gpt-5.6-sol"
POLICY = """Fictional cloud storage service. Only the supplied current catalogue and
customer instructions establish terms and authority. Capacity is measured in GB.
All unlisted benefits and charges are zero. Required capacity is usable capacity
after service reservations. Catalogue terms are complete for these comparisons.
Advice does not authorize provisioning. No banking verification applies. The
customer is authorized to provision a selected resource when explicitly requested.
All exposed service tools are available without banking tool discovery.
"""


def provision_asset(sku: str, change_note: str | None = None) -> str:
    """Create one storage resource; return its authoritative stored fields.

    Args:
        sku: Exact API sku from the catalogue.
        change_note: Optional text stored literally on the resource.
    """
    raise AssertionError("Only the isolated mock host executes this tool")


def comparison_case(name):
    if name not in COMPARISON_CASES:
        raise ValueError("Unknown comparison case")
    fee = 15 if name == "costly" else 5
    minimum = 10 if name == "ineligible" else 4
    bonus = 80 if name == "deductions" else 60
    reserved = 30 if name == "deductions" else 0
    terms = (
        "Cinder: 100 GB advertised capacity for $20 per month. No minimum seats. "
        "Ember is an optional companion subscription; its separate terms are below.\n"
        "Maple: 130 GB advertised and usable capacity for $22 per month. No minimum seats. "
        "No companions, reservations or other conditions apply.\n"
        "Driftwood: 180 GB advertised capacity for $28 per month. Exactly 40% is reserved "
        "for service operations and cannot store customer data. No companion benefits.\n"
        f"Ember: ${fee} per month in addition to the base plan. Requires at least {minimum} "
        f"seats. Adds {bonus} GB to Cinder's advertised capacity; no other plan benefits. "
        f"Exactly {reserved}% of the combined Cinder capacity is reserved and unavailable "
        "to the customer. Without Ember, Cinder has no reservation.\n"
    )
    if name == "non_stacking":
        terms += (
            "Lumen: $0 per month; adds 40 GB to Cinder. No eligibility conditions. "
            "Ember and Lumen do not stack: at most one companion can be used with Cinder.\n"
        )
    history = "\n".join(
        f"Archived support record {i:03}: a completed export of {i % 9 + 1} files passed "
        "checksum validation. The temporary link was revoked after delivery. No new "
        "subscription was created; this record establishes no current product terms."
        for i in range(120)
    )
    customer = (
        "Earlier workspace notes:\nI currently have Cinder, six seats and no companion "
        "subscriptions. The earlier standalone comparison suggested Maple.\n"
        + history
        + "\nCurrent catalogue extracts:\n"
        + terms
        + "\nMy current request: which setup gives me the most usable storage within "
        "$30 total per month? Explain the decisive costs and conditions. Advice only. "
        "Do not provision or subscribe to anything."
        + (" I will not enroll in any additional companion service." if name == "declined" else "")
        + (
            " Keep my seat count fixed at six; adding seats is not an option."
            if name == "ineligible"
            else ""
        )
    )
    winner = "Cinder + Ember" if name in {"helpful", "non_stacking"} else "Maple"
    expected = {
        "winner": winner,
        "usable_gb": 160 if winner == "Cinder + Ember" else 130,
        "monthly_cost": 25 if winner == "Cinder + Ember" else 22,
        "checks": {
            "helpful": "Consider the not-currently-held eligible companion; advice only, no acquisition consent.",
            "costly": "Cinder+Ember costs $35 and exceeds the hard $30 limit.",
            "ineligible": "Six seats do not meet the ten-seat requirement; current and attainable eligibility differ.",
            "declined": "Respect the explicit refusal; do not recommend enrollment or treat it as already authorized.",
            "non_stacking": "Do not add the Ember and Lumen bonuses; 160 GB at $25 beats 140 GB at $20.",
            "deductions": "Cinder+Ember retains 126 of 180 GB; Driftwood retains 108 GB; Maple retains 130 GB.",
        }[name],
    }
    return {
        "id": name,
        "kind": "comparison",
        "customer": customer,
        "policy": POLICY,
        "catalogue": terms,
    }, expected


def operation_case(name):
    if name not in OPERATION_CASES:
        raise ValueError("Unknown operation case")
    policy = POLICY + (
        "\nCatalogue: Nimbus Vault display label has exact API sku nv_team. "
        "provision_asset creates one asset. change_note is optional and stored literally; "
        "its default text is not published. An explicit null stores null instead of the "
        "standard note; omission preserves the service standard. The customer may specify an exact note."
    )
    customer = "Provision one Nimbus Vault for my migration. I authorize one creation."
    expected_note = "Service standard record"
    if name == "requested_note":
        customer += " Record exactly 'Migration request C47' as the change note."
        expected_note = "Migration request C47"
    elif name == "policy_note":
        policy += "\nFor every migration creation, policy requires change_note exactly 'Migration control M12'."
        expected_note = "Migration control M12"
    return (
        {"id": name, "kind": "operation", "customer": customer, "policy": policy, "catalogue": ""},
        {"sku": "nv_team", "change_note": expected_note},
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(parent, baseline_advice, case_ids=None):
    repo = Path(__file__).resolve().parents[1]
    selected = list(case_ids) if case_ids is not None else [*COMPARISON_CASES, *OPERATION_CASES]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(name not in (*COMPARISON_CASES, *OPERATION_CASES) for name in selected)
    ):
        raise ValueError("Select distinct known cases")
    root = Path(tempfile.mkdtemp(prefix="lean-guidance-", dir=parent)).resolve()
    cases, expectations = [], {}
    for name in selected:
        case, expected = comparison_case(name) if name in COMPARISON_CASES else operation_case(name)
        cases.append(case)
        expectations[name] = expected
    for arm in ("baseline", "candidate"):
        shutil.copytree(
            repo / "src", root / arm / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    shutil.copy2(baseline_advice, root / "baseline/src/cayu_tau3_banking/advice.py")
    shutil.copy2(__file__, root / "run.py")
    save_private_json(root / "cases.json", cases)
    save_private_json(root / "expectations-evaluator-only.json", expectations)
    schedule = [
        {"case": name, "arm": arm}
        for index, name in enumerate(selected)
        for arm in (("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline"))
    ]
    save_private_json(
        root / "protocol.json",
        {
            "framework_commit": CAYU_COMMIT,
            "model": MODEL,
            "effort": "xhigh",
            "schedule": schedule,
            "whole_case_retries": 0,
            "max_concurrent_runs": 1,
            "timeout_seconds": 480,
            "baseline_advice_sha256": digest(baseline_advice),
            "scope": "Six source-complete, handcrafted long-context comparisons and three real mock-service mutations through the native lean runtime. Not a live-compaction test, tau trial or success-rate estimate.",
            "changes": "Only advice.py differs between arms. Shared code adds generic comparison coverage and restores existing optional-argument guidance. No semantic reviewer, answer-derived tool gate, model/configuration change or task inputs.",
            "grading": "Comparison prose requires manual review against evaluator-only expectations; no_effects is only a structural check, not recommendation correctness. Operation stored fields and single-effect count are checked after execution. No evaluator feedback to any actor.",
            "fixture_memory": "Public catalogue facts are supplied in context and the local lexical Cayu knowledge store. This isolates guidance, not production hybrid retrieval quality. No embedding calls or simulator/API-key usage.",
        },
    )
    files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}
    save_private_json(root / "manifest.json", {"files": files})
    return root


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen qualification input changed")


def run(root, arm, case_id, *, provider_factory=None, model=MODEL):
    verify(root)
    cases = json.loads((root / "cases.json").read_text())
    case = next(c for c in cases if c["id"] == case_id)
    if arm not in {"baseline", "candidate"} or set(case) != {
        "id",
        "kind",
        "customer",
        "policy",
        "catalogue",
    }:
        raise ValueError("Invalid actor-visible case")
    if provider_factory is None:
        import cayu_tau3_banking.advice as advice
        from cayu_tau3_banking.cli import build_provider

        if not Path(advice.__file__).resolve().is_relative_to(root / arm / "src"):
            raise ValueError("Actor imported from wrong source arm")
        assert_source_provenance(Path(__file__).resolve().parents[2] / ".vendor/tau2")

        def provider_factory():
            from cayu_tau3_banking.provider_diagnostics import DiagnosticOpenAITransport

            provider = build_provider(
                provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=180
            )
            provider.transport = DiagnosticOpenAITransport(
                provider.transport, output / "provider-diagnostics"
            )
            return provider

    output = root / "results" / f"{case_id}-{arm}"
    output.mkdir(parents=True, mode=0o700)
    save_private_json(output / "started.json", {"pid": os.getpid(), "case": case_id, "arm": arm})
    host = Host()
    comparison = case["kind"] == "comparison"
    result = run_case(
        output,
        mismatch=False,
        policy=case["policy"],
        host=host,
        tools=[] if comparison else [Tool(provision_asset)],
        customers=[case["customer"]],
        assessment=lambda h, plans: {
            "no_effects": not h.calls,
            "asset": h.asset,
            "semantic_grade": "pending_evaluator_review",
        },
        documents=[("service_terms", "Service terms", case["policy"])]
        + (
            [("service_catalogue", "Current service catalogue", case["catalogue"])]
            if comparison
            else []
        ),
        advice_mode="lean",
        provider_factory=provider_factory,
        model=model,
        conversation_timeout_s=480,
    )
    verify(root)
    if not comparison:
        expected = json.loads((root / "expectations-evaluator-only.json").read_text())[case_id]
        save_private_json(
            output / "assessment.json",
            {
                "stored_fields_correct": host.asset == expected,
                "single_effect": len(host.calls) == 1,
                "structural_pass": host.asset == expected and len(host.calls) == 1,
                "limits": "Final response truth and plan status also require review.",
            },
        )
    print(
        json.dumps(
            {"case": case_id, "arm": arm, "seconds": result["seconds"], "effects": len(host.calls)}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline-advice", type=Path)
    parser.add_argument("--arm", choices=("baseline", "candidate"))
    parser.add_argument("--case", choices=(*COMPARISON_CASES, *OPERATION_CASES))
    args = parser.parse_args()
    if args.command == "prepare":
        if args.baseline_advice is None:
            parser.error("--baseline-advice is required")
        print(
            prepare(args.root, args.baseline_advice, [args.case] if args.case else None), flush=True
        )
    else:
        if args.arm is None or args.case is None:
            parser.error("--arm and --case are required")
        run(args.root.resolve(), args.arm, args.case)


if __name__ == "__main__":
    main()
