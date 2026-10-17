"""Frozen, one-decision context reduction; never executes proposed banking tools.

Only structural sections are selected. No customer-language matching, product
ranking, new source selection, expected answer, or actor policy is implemented.
"""

import argparse
import asyncio
import copy
import json
import tempfile
import time
from pathlib import Path

from probe_fee_plan_context import CapturedTransport, decide, digest, memory_block

from cayu_tau3_banking.decision_probe import save_private_json

ARMS = ("full", "plan_reduced", "lean")
SUMMARY_BOUNDARY = "\n## Plan/tool state\n"
INSTRUCTION_BOUNDARY = "\n\nGive each discoverable customer-side tool name only once;"


def section(text, tag):
    opening, closing = f"<{tag}>", f"</{tag}>"
    if text.count(opening) != 1 or text.count(closing) != 1:
        raise ValueError(f"Expected one {tag} section")
    prefix, rest = text.split(opening)
    body, suffix = rest.split(closing)
    return prefix, body, suffix


def reduce_plan(payload):
    candidate = copy.deepcopy(payload)
    if any(i.get("type") == "reasoning" for i in candidate["input"]):
        raise ValueError("All arms must start without opaque prior reasoning")
    summary_parts = [
        part
        for item in candidate["input"]
        if item.get("role") == "user" and isinstance(item.get("content"), list)
        for part in item["content"]
        if part.get("text", "").startswith("Previous session context summary:\n")
    ]
    if len(summary_parts) != 1:
        raise ValueError("Expected one historical summary")
    summary = summary_parts[0]["text"]
    if summary.count(SUMMARY_BOUNDARY) != 1:
        raise ValueError("Expected one boundary after observed facts")
    factual_prefix, removed = summary.split(SUMMARY_BOUNDARY)
    if (
        "\n## Observed facts\n" not in factual_prefix
        or "\n## Pending work\n" not in removed
        or "\n## Tentative proposals—not required procedures\n" not in removed
    ):
        raise ValueError("Historical summary structure changed")
    summary_parts[0]["text"] = factual_prefix
    calls = [
        item
        for item in candidate["input"]
        if item.get("type") == "function_call" and item.get("name") == "controller_update_work_plan"
    ]
    if len(calls) != 1 or not calls[0].get("call_id"):
        raise ValueError("Expected one historical plan call")
    call_id = calls[0]["call_id"]
    outputs = [
        item
        for item in candidate["input"]
        if item.get("type") == "function_call_output" and item.get("call_id") == call_id
    ]
    if len(outputs) != 1:
        raise ValueError("Expected the historical plan call's output")
    candidate["input"] = [
        item
        for item in candidate["input"]
        if not (
            item.get("type") in {"function_call", "function_call_output"}
            and item.get("call_id") == call_id
        )
    ]
    memory = memory_block(candidate)
    prefix, _, suffix = section(memory["text"], "workflow-plan")
    memory["text"] = prefix + suffix
    return candidate


def reduce_instructions(payload):
    candidate = copy.deepcopy(payload)
    before, instructions, after = section(candidate["instructions"], "instructions")
    if instructions.count(INSTRUCTION_BOUNDARY) != 1:
        raise ValueError("Expected the original generic instruction prefix")
    # The exact opening rules and entire bank policy remain. No new advice is added.
    generic = instructions.split(INSTRUCTION_BOUNDARY)[0]
    section(after, "policy")
    candidate["instructions"] = before + "<instructions>" + generic + "\n</instructions>" + after
    return candidate


def variants(payload):
    reduced = reduce_plan(payload)
    return {
        "full": copy.deepcopy(payload),
        "plan_reduced": reduced,
        "lean": reduce_instructions(reduced),
    }


def measurements(payload):
    return {
        "instructions_characters": len(payload["instructions"]),
        "input_json_characters": len(json.dumps(payload["input"], ensure_ascii=False)),
        "tools_json_characters": len(json.dumps(payload["tools"], ensure_ascii=False)),
        "note": "Character counts, not measured token usage.",
    }


def prepare(parent, output_parent):
    completed = json.loads((parent / "completed.json").read_text())
    if completed.get("scheduled_calls_completed") != 4:
        raise ValueError("Requires the completed carryover experiment")
    source = parent / "0-fresh_reasoning-request.json"
    original = json.loads(source.read_text())
    if original["model"] != "gpt-5.6-sol" or original["reasoning"]["effort"] != "xhigh":
        raise ValueError("Unexpected actor model/configuration")
    candidates = variants(original)
    root = Path(tempfile.mkdtemp(prefix="lean-context-", dir=output_parent)).resolve()
    schedule = []
    for repeat, order in enumerate((ARMS, reversed(ARMS))):
        for arm in order:
            name = f"{repeat}-{arm}"
            path = root / f"{name}-request.json"
            save_private_json(path, candidates[arm])
            schedule.append({"name": name, "arm": arm, "request_sha256": digest(path)})
    save_private_json(
        root / "protocol.json",
        {
            "source_request": str(source.resolve()),
            "source_request_sha256": digest(source),
            "code_hashes": {
                str(path.resolve()): digest(path)
                for path in (Path(__file__), Path(__file__).with_name("probe_fee_plan_context.py"))
            },
            "schedule": schedule,
            "model": original["model"],
            "reasoning": original["reasoning"],
            "measurements": {arm: measurements(p) for arm, p in candidates.items()},
            "timeout_seconds_per_call": 300,
            "experiment_retries": 0,
            "changes": {
                "full": "Identical to prior fee-restored, two-claim-neutralized request with opaque prior reasoning already removed.",
                "plan_reduced": "Remove the single historical plan call/output, displayed workflow-plan, and summary tail from Plan/tool state onward. Preserve the complete customer-constraints and observed-facts prefix verbatim. All other input and instructions remain.",
                "lean": "Additionally retain only the original generic instruction opening before the discoverable-tool guidance. Preserve the full original bank policy and every tool schema. No new instruction is added.",
            },
            "preserved": "All public source bodies, research limits, customer ledger messages, factual summary prefix, observation ledger, navigation directories and tool schemas. No document is selected or excluded based on the target answer. No opaque prior reasoning in any arm.",
            "manual_review": [
                "Does the response substantively compare the omitted Silver Rewards savings bonus, merely mention it, or omit it?",
                "Does it preserve the total funds limit, changed checking preference, no-diversion constraint, unknown subscription price, unknown card eligibility/approval and missing execution consent?",
                "Does it preserve relevant source contradictions instead of inventing a resolved price or rate?",
                "Record proposed tools, plan-only responses and research requests separately; no tool will execute.",
                "Distinguish a changed recommendation from a supported improvement. Do not grade by matching the benchmark reference product choice.",
            ],
            "limits": "Six one-step development probes, two per arm in reverse order. Not a task replay or reliable success rate. The lean treatment changes both the amount and content of guidance, not length alone. The plan reduction also removes model-derived interpretations, not solely duplicate bytes. Source prose and factual summary can still frame the choice. Historical plan evidence may contain interpretations not retained elsewhere; this is not a production-safe compaction implementation. No independent decision replication, runtime integration, or framework defect is established by this experiment.",
        },
    )
    return root


async def run(root):
    from cayu_tau3_banking.cli import build_provider

    protocol = json.loads((root / "protocol.json").read_text())
    for path, expected in protocol["code_hashes"].items():
        if digest(Path(path)) != expected:
            raise ValueError("Diagnostic code changed after registration")
    if digest(Path(protocol["source_request"])) != protocol["source_request_sha256"]:
        raise ValueError("Parent request changed")
    for row in protocol["schedule"]:
        if digest(root / f"{row['name']}-request.json") != row["request_sha256"]:
            raise ValueError("Prepared arm changed")
    save_private_json(root / "started.json", {"time_unix": time.time(), "scheduled_calls": 6})
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
        root / "completed.json", {"scheduled_calls_completed": 6, "tools_executed": 0}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("parent", type=Path)
    p.add_argument("--output-parent", type=Path, default=Path("artifacts"))
    p = sub.add_parser("run")
    p.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.parent.resolve(), args.output_parent))
    else:
        asyncio.run(run(args.root.resolve()))


if __name__ == "__main__":
    main()
