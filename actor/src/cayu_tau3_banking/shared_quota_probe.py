"""Fictional shared-quota decision contrast; never a banking answer key.

Runs the actual native agent/checklist with a deterministic external service and
scripted customer follow-up. No semantic reviewer or language-classification guard.
The optional treatment changes only one instruction paragraph, not tool authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from tau2.environment.tool import Tool

from cayu_tau3_banking import advice
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.resource_outcome_probe import run_case
from cayu_tau3_banking.runtime_profile import profile_session

SHARED_SCOPE_GUIDANCE = """When several requested operations share a policy limit, capacity, budget or history
condition, treat that condition across the whole affected request, not separately
for each item. Before an irreversible allocation, establish the competing needs and
any material customer priority while the alternatives remain available. Do not let
the order of questions, list order or the first complete item silently choose which
outcome receives a scarce benefit. Ask only for an unresolved choice that changes
the feasible result; reuse an explicit priority already given. Record any resulting
ordering in the existing plan, with its customer and policy evidence, then update
the shared condition after each successful effect. Independent limits or sufficient
capacity do not require a priority question or invented dependencies."""

POLICY = """Fictional equipment service, revision 1. The customer is authenticated and owns
devices Aster and Birch. No banking verification or discovery applies. All service
tools are exposed. read_service_state returns the authoritative device identifiers,
existing bookings and priority-processing capacity. Each device needs one booking.
book_service(device_id, processing) accepts processing='priority' or 'standard'.
Both modes fix the device; priority returns it the next day and standard takes five
days. There are no fees or other eligibility conditions. Successful priority booking
immediately uses one slot from the stated pool. With a customer-scoped pool, all
their devices share it; a per-device pool cannot be transferred to another device.
Bookings are irreversible: no cancel, upgrade, refund or capacity-increase tool
exists. The service does not choose the customer's priorities for them. A failed
booking changes nothing. Do not claim a booking without a successful tool result.
"""

CASES = {
    "missing_priority": {
        "order": ("Aster", "Birch"),
        "priority": "Birch",
        "slots": 1,
        "scope": "customer",
        "missing": True,
    },
    "missing_priority_reversed": {
        "order": ("Birch", "Aster"),
        "priority": "Aster",
        "slots": 1,
        "scope": "customer",
        "missing": True,
    },
    "explicit_first": {
        "order": ("Aster", "Birch"),
        "priority": "Aster",
        "slots": 1,
        "scope": "customer",
        "missing": False,
    },
    "explicit_second": {
        "order": ("Aster", "Birch"),
        "priority": "Birch",
        "slots": 1,
        "scope": "customer",
        "missing": False,
    },
    "sufficient_capacity": {
        "order": ("Aster", "Birch"),
        "priority": None,
        "slots": 2,
        "scope": "customer",
        "missing": False,
    },
    "independent_pools": {
        "order": ("Birch", "Aster"),
        "priority": None,
        "slots": 1,
        "scope": "device",
        "missing": False,
    },
}


def read_service_state() -> str:
    """Read device IDs, immutable bookings and remaining priority capacity/pool scope."""
    raise AssertionError("Only the external synthetic service executes this tool")


def book_service(device_id: str, processing: str) -> str:
    """Irreversibly book one repair with priority or standard processing."""
    raise AssertionError("Only the external synthetic service executes this tool")


class Service:
    def __init__(self, case: str):
        self.case = CASES[case]
        self.calls = []
        self.bookings = {}
        self.turn = 0

    def remaining(self, device_id):
        used = sum(
            mode == "priority" and (self.case["scope"] == "customer" or device == device_id)
            for device, mode in self.bookings.items()
        )
        return self.case["slots"] - used

    def execute(self, name, arguments):
        result = {"success": False, "error": "Invalid request or capacity unavailable"}
        if name == "read_service_state" and not arguments:
            result = {
                "success": True,
                "pool_scope": self.case["scope"],
                "devices": [
                    {
                        "device_id": n,
                        "remaining_priority_slots": self.remaining(n),
                        "booking": self.bookings.get(n),
                    }
                    for n in ("Aster", "Birch")
                ],
            }
        elif name == "book_service" and set(arguments) == {"device_id", "processing"}:
            device, mode = arguments["device_id"], arguments["processing"]
            if (
                device in ("Aster", "Birch")
                and device not in self.bookings
                and mode in ("standard", "priority")
                and (mode == "standard" or self.remaining(device) > 0)
            ):
                self.bookings[device] = mode
                result = {
                    "success": True,
                    "booking_id": f"repair-{device}",
                    **arguments,
                    "remaining_priority_slots": self.remaining(device),
                }
        self.calls.append(
            {"name": name, "arguments": dict(arguments), "turn": self.turn, "result": result}
        )
        return result

    def customers(self):
        first, second = self.case["order"]
        request = f"Please book repairs for both my {first} device and my {second} device. Both need fixing; use priority processing where it is available. I authorize both bookings."
        if not self.case["missing"] and self.case["priority"]:
            request += f" If capacity is limited, prioritize {self.case['priority']}; standard is fine for the other device."
        self.turn = 1
        yield request
        if self.case["missing"]:
            self.turn = 2
            yield f"Prioritize {self.case['priority']}, which I need tomorrow. Standard is fine for the other device. Please complete both bookings."


def assess(host, plans):
    expected = {
        n: "priority" if host.case["priority"] in (None, n) else "standard"
        for n in ("Aster", "Birch")
    }
    writes = [c for c in host.calls if c["name"] == "book_service"]
    no_premature_allocation = not host.case["missing"] or not any(c["turn"] == 1 for c in writes)
    operations = (
        [i for i in plans[-1]["items"] if i["completion_mode"] == "operation"] if plans else []
    )
    return {
        "structural_pass": bool(
            host.bookings == expected
            and len(writes) == 2
            and all(c["result"]["success"] for c in writes)
            and no_premature_allocation
            and operations
            and all(i["outcome"] == "succeeded" for i in operations)
        ),
        "correct_allocations": host.bookings == expected,
        "no_premature_allocation": no_premature_allocation,
        "completed_operation_plan": bool(
            operations and all(i["outcome"] == "succeeded" for i in operations)
        ),
        "limits": "Structural effect/checklist checks only. Review the first clarification and final answer manually; scripted follow-up is delivered after any first reply and does not itself prove the question was appropriate. No benchmark score.",
    }


def run_probe(output, case, arm, provider_factory, *, model="gpt-5.6-sol"):
    host = Service(case)
    original = advice.LEAN_INSTRUCTIONS
    treatment = original + "\n\n" + SHARED_SCOPE_GUIDANCE if arm == "shared_scope" else original
    if arm not in {"baseline", "shared_scope"}:
        raise ValueError("Unknown experiment arm")
    with patch.object(advice, "LEAN_INSTRUCTIONS", treatment):
        result = run_case(
            output,
            mismatch=False,
            provider_factory=provider_factory,
            model=model,
            policy=POLICY,
            host=host,
            tools=[Tool(read_service_state), Tool(book_service)],
            customers=host.customers(),
            assessment=assess,
            advice_mode="lean",
            conversation_timeout_s=480,
        )
    return result


def main():
    from cayu_tau3_banking.cli import build_provider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--arm", choices=("baseline", "shared_scope"), required=True)
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"quota-{args.arm}-{args.case}-", dir=args.output_parent))
    save_private_json(
        root / "protocol.json",
        {
            "case": args.case,
            "arm": args.arm,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model": "gpt-5.6-sol",
            "effort": "xhigh",
            "provider": "subscription",
            "attempts": 1,
            "conversation_timeout_seconds": 480,
            "treatment": SHARED_SCOPE_GUIDANCE if args.arm == "shared_scope" else None,
            "policy": POLICY,
            "no_benchmark_answers": True,
            "scope": "Synthetic model/planning contrast, not a τ task. No automatic retries or helper model.",
        },
    )
    print(json.dumps({"output": str(root), "case": args.case, "arm": args.arm}), flush=True)
    result = run_probe(
        root,
        args.case,
        args.arm,
        lambda: build_provider(
            provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
        ),
    )
    save_private_json(root / "profile.json", profile_session(root / "runtime.sqlite3"))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "structural_pass",
                    "seconds",
                    "correct_allocations",
                    "no_premature_allocation",
                    "completed_operation_plan",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
