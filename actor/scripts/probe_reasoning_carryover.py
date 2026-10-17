"""Bounded follow-up: identical visible context, with/without opaque prior state.

No prior reasoning is inspected or rewritten; only whole reasoning items are
excluded in the fresh-state arm. New reasoning remains xhigh in both arms.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import tempfile
import time
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh_state(payload):
    candidate = copy.deepcopy(payload)
    original = candidate["input"]
    candidate["input"] = [item for item in original if item.get("type") != "reasoning"]
    if len(original) == len(candidate["input"]):
        raise ValueError("No opaque reasoning items to contrast")
    return candidate


def prepare(parent):
    if json.loads((parent / "completed.json").read_text())["scheduled_calls_completed"] != 8:
        raise ValueError("The original registered experiment must finish first")
    original = json.loads((parent / "0-both-request.json").read_text())
    candidates = {"carryover": original, "fresh_reasoning": fresh_state(original)}
    root = Path(tempfile.mkdtemp(prefix="reasoning-carryover-", dir=parent.parent)).resolve()
    schedule = []
    for repeat, order in enumerate((candidates, reversed(candidates))):
        for arm in order:
            name = f"{repeat}-{arm}"
            path = root / f"{name}-request.json"
            save_private_json(path, candidates[arm])
            schedule.append({"name": name, "arm": arm, "request_sha256": digest(path)})
    save_private_json(
        root / "protocol.json",
        {
            "parent_experiment": str(parent.resolve()),
            "source_request": str((parent / "0-both-request.json").resolve()),
            "source_request_sha256": digest(parent / "0-both-request.json"),
            "code_hashes": {
                str(Path(__file__).resolve()): digest(Path(__file__)),
                str(Path(__file__).with_name("probe_fee_plan_context.py").resolve()): digest(
                    Path(__file__).with_name("probe_fee_plan_context.py")
                ),
            },
            "parent_result_hashes": {
                p.name: digest(p) for p in sorted(parent.glob("*-result.json"))
            },
            "schedule": schedule,
            "model": original["model"],
            "reasoning": original["reasoning"],
            "opaque_items_removed_in_fresh_arm": len(original["input"])
            - len(candidates["fresh_reasoning"]["input"]),
            "timeout_seconds_per_call": 300,
            "experiment_retries": 0,
            "purpose": "Check whether retaining opaque pre-decision reasoning explains persistence after both visible-context treatments failed.",
            "review": "Same target as parent: substantive alternative card-bonus evaluation versus omission; unknown eligibility/approval/consent preserved; all proposed calls recorded without execution.",
            "limits": "Exploratory follow-up, not part of the original 2x2 registration and not pooled with it. Four calls, two per arm, reverse order. Removing opaque items also reduces input length; any improvement would not separate state contents from length. All visible input, original instructions, tools, fees, plan, research budgets and xhigh effort stay identical. No tool executes, no reference answer is supplied, no production change.",
        },
    )
    return root


async def run(root):
    from probe_fee_plan_context import CapturedTransport, decide

    from cayu_tau3_banking.cli import build_provider

    p = json.loads((root / "protocol.json").read_text())
    for path, expected in p["code_hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError("Diagnostic code changed after registration")
    if digest(Path(p["source_request"])) != p["source_request_sha256"]:
        raise ValueError("Parent request changed")
    for row in p["schedule"]:
        if digest(root / f"{row['name']}-request.json") != row["request_sha256"]:
            raise ValueError("Registered arm changed")
    save_private_json(root / "started.json", {"time_unix": time.time(), "scheduled_calls": 4})
    for row in p["schedule"]:
        name = row["name"]
        payload = json.loads((root / f"{name}-request.json").read_text())
        provider = build_provider(
            provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=180
        )
        provider.transport = CapturedTransport(provider.transport, payload)
        started = time.monotonic()
        print(json.dumps({"starting": name}), flush=True)
        try:
            result = await decide(provider, payload, timeout=p["timeout_seconds_per_call"])
            result.update({**row, "seconds": time.monotonic() - started})
            save_private_json(root / f"{name}-result.json", result)
            print(
                json.dumps(
                    {
                        "completed": name,
                        "seconds": round(result["seconds"], 2),
                        "proposed_tools": len(result["calls"]),
                    }
                ),
                flush=True,
            )
        except Exception as error:
            save_private_json(root / f"{name}-error.json", {"error_type": type(error).__name__})
            raise
        finally:
            await provider.aclose()
    save_private_json(
        root / "completed.json", {"scheduled_calls_completed": 4, "tools_executed": 0}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.root.resolve()))
    else:
        asyncio.run(run(args.root.resolve()))


if __name__ == "__main__":
    main()
