"""Preregistered 2x2 saved-decision experiment; proposed tools never execute.

This is diagnostic code, not an actor policy. Exact plan IDs select two observed
claims in one historical request; no customer-language matching is involved.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import tempfile
import time
from contextlib import aclosing
from pathlib import Path

from cayu import Message
from cayu.providers.base import ModelRequest, ModelStreamEventType

from cayu_tau3_banking.captured_argument_probe import CapturedTransport
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.policy_grounding_probe import ProbeProviderError

TARGETS = {
    ("recommend_checking", "deliver_revised_combo_recommendation"),
    ("recommend_savings", "deliver_savings_recommendation"),
}
NEUTRAL = (
    "The customer's revised comparison request has not yet been answered. "
    "The cited sources and unresolved facts remain the basis for formulating that answer."
)
ARMS = ("baseline", "fee_only", "plan_only", "both")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory_block(payload):
    matches = [
        part
        for item in payload["input"]
        if item.get("role") == "user" and isinstance(item.get("content"), list)
        for part in item["content"]
        if "<banking_context>" in part.get("text", "")
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one application memory block")
    return matches[0]


def selected_claims(plan):
    claims = {}
    for item in plan["items"]:
        for obligation in item.get("obligations", []):
            key = (item["id"], obligation["id"])
            if key in TARGETS:
                if key in claims or not obligation.get("evidence"):
                    raise ValueError("Ambiguous or empty target evidence")
                claims[key] = obligation["evidence"]
    if claims.keys() != TARGETS:
        raise ValueError("Missing target plan claims")
    return claims


def replace_claim(raw, before):
    # Replace only JSON string tokens, preserving all other formatting and bytes.
    tokens = {json.dumps(before), json.dumps(before, ensure_ascii=False)}
    matches = [token for token in tokens if token in raw]
    if len(matches) != 1 or raw.count(matches[0]) != 1 or before == NEUTRAL:
        raise ValueError("Claim is not an unambiguous original JSON string")
    return raw.replace(matches[0], json.dumps(NEUTRAL), 1)


def neutralize_plan(payload):
    candidate = copy.deepcopy(payload)
    calls = [
        item
        for item in candidate["input"]
        if item.get("type") == "function_call" and item.get("name") == "controller_update_work_plan"
    ]
    if len(calls) != 1:
        raise ValueError("Expected one captured plan update")
    call = calls[0]
    claims = selected_claims(json.loads(call["arguments"]))
    memory = memory_block(candidate)
    text = memory["text"]
    opening, closing = "<workflow-plan>", "</workflow-plan>"
    if text.count(opening) != 1 or text.count(closing) != 1:
        raise ValueError("Expected one visible checklist")
    prefix, tail = text.split(opening)
    plan_raw, suffix = tail.split(closing)
    if selected_claims(json.loads(plan_raw)) != claims:
        raise ValueError("Captured update and visible checklist disagree")
    for before in claims.values():
        call["arguments"] = replace_claim(call["arguments"], before)
        plan_raw = replace_claim(plan_raw, before)
    memory["text"] = prefix + opening + plan_raw + closing + suffix
    return candidate


def restore_fee(payload, document):
    if not isinstance(document, dict) or any(
        not isinstance(document.get(k), str) or not document[k].strip()
        for k in ("id", "title", "content")
    ):
        raise ValueError("Requires one complete public source")
    candidate = copy.deepcopy(payload)
    block = memory_block(candidate)
    if (
        block["text"].count("</banking_context>") != 1
        or "<restored-public-source>" in block["text"]
    ):
        raise ValueError("Ambiguous or already treated memory")
    source = json.dumps({k: document[k] for k in ("id", "title", "content")}, ensure_ascii=False)
    block["text"] = block["text"].replace(
        "</banking_context>",
        "\n<restored-public-source>\n" + source + "\n</restored-public-source>\n</banking_context>",
        1,
    )
    return candidate


def variants(payload, document):
    neutral = neutralize_plan(payload)
    return {
        "baseline": copy.deepcopy(payload),
        "fee_only": restore_fee(payload, document),
        "plan_only": neutral,
        "both": restore_fee(neutral, document),
    }


def prepare(capture_path, document_path, parent):
    capture = json.loads(capture_path.read_text())
    if (
        capture.get("schema") != "cayu-tau3.provider-request.v1"
        or capture.get("omitted_field_count") != 0
    ):
        raise ValueError("Requires a complete prepared request")
    payload = capture["payload"]
    if payload["model"] != "gpt-5.6-sol" or payload["reasoning"]["effort"] != "xhigh":
        raise ValueError("Unexpected actor model/configuration")
    candidates = variants(payload, json.loads(document_path.read_text()))
    root = Path(tempfile.mkdtemp(prefix="fee-plan-context-", dir=parent)).resolve()
    schedule = []
    for repeat in range(2):
        for arm in ARMS if repeat == 0 else reversed(ARMS):
            name = f"{repeat}-{arm}"
            path = root / f"{name}-request.json"
            save_private_json(path, candidates[arm])
            schedule.append(
                {"name": name, "arm": arm, "repeat": repeat, "request_sha256": digest(path)}
            )
    protocol = {
        "capture": str(capture_path.resolve()),
        "capture_sha256": digest(capture_path),
        "public_document": str(document_path.resolve()),
        "public_document_sha256": digest(document_path),
        "script_sha256": digest(Path(__file__)),
        "schedule": schedule,
        "model": payload["model"],
        "reasoning": payload["reasoning"],
        "timeout_seconds_per_call": 300,
        "experiment_retries": 0,
        "fee_treatment": "Append the full previously delivered public fee/eligibility document to the same application memory role; no new system instruction.",
        "plan_treatment": "Replace only two premature best-fallback evidence claims with neutral pending-answer statements in the recorded plan call and displayed checklist. Keep statuses, goals, citations and budgets unchanged.",
        "manual_review": [
            "Does the output substantively evaluate the Silver Plus card bonus, merely mention it, or omit it?",
            "Does it preserve unknown approval, customer choice and credit-check consent?",
            "Does it propose an action, unavailable research or plan update instead of an answer?",
        ],
        "limits": "Eight scheduled one-step decisions, not a full task, reliability estimate or context-length experiment. No tools execute. No evaluator/reference actions are supplied. Original instructions, tools and opaque prior reasoning items remain identical in all arms; residual prior reasoning and other plan wording can still carry the earlier conclusion. Factor effects include the changed text's placement/salience. All outputs are reviewed, including baseline successes.",
    }
    save_private_json(root / "protocol.json", protocol)
    return root


async def decide(provider, payload, timeout=300):
    request = ModelRequest(
        model=payload["model"],
        messages=[Message.text("user", "Isolated diagnostic replay")],
        tools=[
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t["parameters"],
            }
            for t in payload["tools"]
        ],
    )
    calls, text, usage = [], [], None
    completed = 0
    async with asyncio.timeout(timeout):
        async with aclosing(provider.runtime_stream(request)) as stream:
            async for event in stream:
                if event.type is ModelStreamEventType.TOOL_CALL:
                    calls.append(event.payload)
                elif event.type is ModelStreamEventType.TEXT_DELTA:
                    text.append(event.delta)
                elif event.type is ModelStreamEventType.COMPLETED:
                    completed += 1
                    if event.completion is not None:
                        value = event.completion.model_dump(mode="json").get("usage")
                        if value is not None:
                            usage = value
                elif event.type is ModelStreamEventType.ERROR:
                    raise ProbeProviderError(event.payload)
    if completed != 1 or (not calls and not any(text)):
        raise ValueError("Expected one complete, nonempty decision")
    return {"calls": calls, "text": "".join(text), "usage": usage, "tools_executed": 0}


async def run(root):
    from cayu_tau3_banking.cli import build_provider

    protocol = json.loads((root / "protocol.json").read_text())
    if digest(Path(__file__)) != protocol["script_sha256"]:
        raise ValueError("Experiment code changed after preregistration")
    for field in ("capture", "public_document"):
        if digest(Path(protocol[field])) != protocol[field + "_sha256"]:
            raise ValueError("Source changed after preregistration")
    for row in protocol["schedule"]:
        if digest(root / f"{row['name']}-request.json") != row["request_sha256"]:
            raise ValueError("Prepared arm changed after preregistration")
    save_private_json(root / "started.json", {"time_unix": time.time(), "scheduled_calls": 8})
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
                        "tools_executed": 0,
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
        root / "completed.json", {"scheduled_calls_completed": 8, "tools_executed": 0}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    p = subparsers.add_parser("prepare")
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--document", type=Path, required=True)
    p.add_argument("--output-parent", type=Path, default=Path("artifacts"))
    p = subparsers.add_parser("run")
    p.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.capture, args.document, args.output_parent))
    else:
        asyncio.run(run(args.root.resolve()))


if __name__ == "__main__":
    main()
