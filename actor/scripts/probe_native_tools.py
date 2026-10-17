"""Two-call follow-up removing application helper schemas, not banking tools.

This separates answer quality from optional plan/worksheet detours. No new
instructions or evidence are added and no proposed tool executes.
"""

import argparse
import asyncio
import copy
import json
import tempfile
import time
from pathlib import Path

from probe_fee_plan_context import CapturedTransport, decide, digest

from cayu_tau3_banking.decision_probe import save_private_json

HELPERS = frozenset(
    {
        "cayu_check_comparison_facts",
        "controller_update_work_plan",
        "controller_prepare_credit_card_dispute_policy",
        "controller_audit_credit_card_disputes",
        "controller_prepare_credit_card_reward_policy",
        "controller_audit_credit_card_rewards",
        "controller_prepare_savings_interest_policy",
        "controller_audit_savings_interest",
    }
)


def native_tools(payload):
    candidate = copy.deepcopy(payload)
    names = [t["name"] for t in candidate["tools"]]
    if len(names) != len(set(names)) or not set(names).issuperset(HELPERS):
        raise ValueError("Expected original helper schemas exactly once")
    candidate["tools"] = [t for t in candidate["tools"] if t["name"] not in HELPERS]
    if not candidate["tools"]:
        raise ValueError("Must preserve native banking tools")
    return candidate


def prepare(parent):
    if json.loads((parent / "completed.json").read_text()).get("scheduled_calls_completed") != 6:
        raise ValueError("Complete the original six-call experiment first")
    source = parent / "0-lean-request.json"
    original = json.loads(source.read_text())
    candidate = native_tools(original)
    root = Path(tempfile.mkdtemp(prefix="native-tools-", dir=parent.parent)).resolve()
    schedule = []
    for repeat in range(2):
        name = f"{repeat}-native_tools"
        path = root / f"{name}-request.json"
        save_private_json(path, candidate)
        schedule.append({"name": name, "request_sha256": digest(path)})
    save_private_json(
        root / "protocol.json",
        {
            "source_request": str(source.resolve()),
            "source_request_sha256": digest(source),
            "code_hashes": {
                str(p.resolve()): digest(p)
                for p in (Path(__file__), Path(__file__).with_name("probe_fee_plan_context.py"))
            },
            "schedule": schedule,
            "model": candidate["model"],
            "reasoning": candidate["reasoning"],
            "removed_tools": sorted(HELPERS),
            "retained_tools": [t["name"] for t in candidate["tools"]],
            "timeout_seconds_per_call": 300,
            "experiment_retries": 0,
            "purpose": "Both lean calls proposed optional arithmetic worksheets, so final recommendation quality was not observable in that one-step experiment. Check the next decision without application helper schemas.",
            "changes": "Only remove the eight listed application helper schemas. Preserve every native bank schema, input item, customer fact, source, research-capacity block, original bank policy and lean instructions. No new guidance or source selection.",
            "review": "Substantive omitted Silver Rewards bonus comparison; respect funds, customer preferences, uncertain subscription/eligibility/approval/consent and conflicting source terms. Tool proposals remain unexecuted; report a bank-tool-only response as no final answer, not a wrong answer. No reference product matching.",
            "limits": "Separate exploratory two-call follow-up, not pooled with six registered calls. Changes available helper tools and their descriptive guidance, not length alone; no contemporary repeated helper-enabled control beyond the immediately preceding two lean calls. Not a full task, reliability estimate, or production recommendation to remove helpers. No tools execute.",
        },
    )
    return root


async def run(root):
    from cayu_tau3_banking.cli import build_provider

    protocol = json.loads((root / "protocol.json").read_text())
    for p, h in protocol["code_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError("Code changed")
    if digest(Path(protocol["source_request"])) != protocol["source_request_sha256"]:
        raise ValueError("Source changed")
    for row in protocol["schedule"]:
        if digest(root / f"{row['name']}-request.json") != row["request_sha256"]:
            raise ValueError("Request changed")
    save_private_json(root / "started.json", {"time_unix": time.time(), "scheduled_calls": 2})
    for row in protocol["schedule"]:
        name = row["name"]
        payload = json.loads((root / f"{name}-request.json").read_text())
        provider = build_provider(
            provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=180
        )
        provider.transport = CapturedTransport(provider.transport, payload)
        started = time.monotonic()
        print(json.dumps({"starting": name}), flush=True)
        try:
            result = await decide(provider, payload, timeout=protocol["timeout_seconds_per_call"])
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
            save_private_json(
                root / f"{name}-error.json",
                {"error_type": type(error).__name__, "seconds": time.monotonic() - started},
            )
            raise
        finally:
            await provider.aclose()
    save_private_json(
        root / "completed.json", {"scheduled_calls_completed": 2, "tools_executed": 0}
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
