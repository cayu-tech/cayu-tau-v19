"""Opt-in, synthetic policy-decision probe; never a tau score or runtime reviewer.

Only procedure availability changes within a pair. The deliberately shortened
context is NOT a reconstruction of a historical compaction summary. Expectations
stay outside model inputs. No proposed bank action is ever executed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import time
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path

from cayu import Message
from cayu.providers.base import ModelRequest, ModelStreamEventType

PROCEDURE = """Fictional bank procedure, revision 1:
After identity, closure eligibility, and retention history checks, establish why
the customer wants to leave. If a competing product motivated the request, compare
its relevant benefits with our documented alternatives before a generic retention
offer. If an alternative matches or exceeds those benefits, describe it and offer
application assistance; applying requires customer consent. If none does, explain
that and offer the permitted retention benefit. For general rewards dissatisfaction
without a competing product, explain how to maximize current rewards before offering
retention. Once the relevant comparison or rewards guidance is complete and the
customer declines alternatives and retention, proceed to closure disclosures and
obtain the required authorization. Never apply or close merely to finish a plan.
"""

PRODUCTS = """Fictional product facts:
Harbor Card: 1% flat rewards. Cedar Card: 4.5% on rail travel and professional
software, 1% otherwise. Neither has an annual fee. No other bank product exceeds
4.5% in these categories. Eligible retention benefit: 500 points.
"""


@dataclass(frozen=True)
class Case:
    id: str
    customer: str
    state: str
    expected_with_procedure: str


CASES = (
    Case(
        "competitor_match",
        "Another bank pays 3.7% for trains and design software. "
        "Those are my main expenses; Harbor's flat rewards aren't competitive.",
        "No comparison or retention offer has occurred.",
        "compare_alternatives",
    ),
    Case(
        "general_dissatisfaction",
        "The rewards disappoint me. I haven't looked at "
        "other banks; I don't understand how to make good use of my current card.",
        "No rewards guidance or retention offer has occurred.",
        "explain_current_rewards",
    ),
    Case(
        "no_matching_alternative",
        "The competitor pays 7% on trains and design "
        "software. I already heard your comparison and none of your cards matches it.",
        "The comparison is complete and confirmed no matching alternative. "
        "No retention offer has occurred.",
        "offer_retention",
    ),
    Case(
        "declined_alternatives",
        "You explained Cedar and the bonus points. "
        "I decline both. Please continue with closing Harbor.",
        "Comparison and retention offer completed and declined. Closure disclosures "
        "and final authorization remain outstanding.",
        "closure_disclosures",
    ),
)

STEPS = (
    "search_procedure",
    "compare_alternatives",
    "explain_current_rewards",
    "offer_retention",
    "closure_disclosures",
    "apply_for_card",
    "close_card",
)


def build_request(
    case: Case, *, procedure_present: bool, system_prompt: str, model: str = "gpt-5.6-sol"
) -> ModelRequest:
    context = (
        "Synthetic decision-only evaluation. Do not execute any bank actions. "
        "Report the next substantive step through record_next_step. "
        "The checklist already exists: resolve the Harbor closure request. "
        "Identity, closure eligibility and retention history checks succeeded.\n"
        f"Current state: {case.state}\n{PRODUCTS}\n"
        + (PROCEDURE if procedure_present else "Detailed closure procedure is not in this context.")
    )
    return ModelRequest(
        model=model,
        messages=[
            Message.text("system", system_prompt),
            Message.text("user", context),
            Message.text("user", case.customer),
        ],
        tools=[
            {
                "name": "record_next_step",
                "description": "Record a proposed step; executes nothing.",
                "input_schema": {
                    "type": "object",
                    "properties": {"step": {"type": "string", "enum": list(STEPS)}},
                    "required": ["step"],
                    "additionalProperties": False,
                },
            }
        ],
        options={
            "thinking": {"enabled": True, "effort": "xhigh"},
            "openai": {"parallel_tool_calls": False, "tool_choice": "required"},
        },
    )


def save_private_json(path: Path, value: object) -> None:
    """Exclusive creation; never overwrite or follow an existing output symlink."""
    data = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        output.write(data)


async def decide(provider, request: ModelRequest, *, timeout: float = 180) -> str:
    calls = []
    completed = 0
    async with asyncio.timeout(timeout):
        async with aclosing(provider.runtime_stream(request)) as stream:
            async for event in stream:
                if event.type is ModelStreamEventType.TOOL_CALL:
                    calls.append(event.payload)
                elif event.type is ModelStreamEventType.COMPLETED:
                    completed += 1
    if completed != 1 or len(calls) != 1:
        raise ValueError("Probe requires one completed response and exactly one decision.")
    call = calls[0]
    arguments = call.get("arguments")
    if (
        call.get("name") != "record_next_step"
        or not isinstance(arguments, dict)
        or set(arguments) != {"step"}
        or arguments["step"] not in STEPS
    ):
        raise ValueError("Probe returned an invalid decision.")
    return arguments["step"]


async def run(output: Path, repeats: int) -> None:
    # Lazy imports keep deterministic tests free of tau/provider initialization.
    from cayu_tau3_banking.agent import banking_system_prompt
    from cayu_tau3_banking.cli import build_provider

    prompt = banking_system_prompt(
        "Follow the supplied fictional bank procedure.", planning_mode="outcomes"
    )
    provider = build_provider(
        provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
    )
    try:
        for repeat in range(repeats):
            for case in CASES:
                # Counterbalance order across repetitions; no feedback to model.
                for present in (False, True) if repeat % 2 == 0 else (True, False):
                    stem = f"{repeat}-{case.id}-{'procedure' if present else 'missing'}"
                    request = build_request(case, procedure_present=present, system_prompt=prompt)
                    provider.preflight_portable_messages(
                        model=request.model, messages=request.messages, tools=request.tools
                    )
                    serialized = request.model_dump(mode="json")
                    save_private_json(output / f"{stem}-request.json", serialized)
                    started = time.monotonic()
                    try:
                        step = await decide(provider, request)
                    except Exception as error:
                        # Never persist provider exception text, credentials or raw payloads.
                        save_private_json(
                            output / f"{stem}-error.json", {"error_type": type(error).__name__}
                        )
                        raise RuntimeError(
                            f"Probe stopped: {type(error).__name__}; {stem}"
                        ) from None
                    expected = case.expected_with_procedure if present else "search_procedure"
                    result = {
                        "case": case.id,
                        "procedure_present": present,
                        "repeat": repeat,
                        "step": step,
                        "expected": expected,
                        "passed": step == expected,
                        "seconds": time.monotonic() - started,
                        "request_sha256": hashlib.sha256(
                            json.dumps(serialized, sort_keys=True).encode()
                        ).hexdigest(),
                        "scope": "synthetic decision probe, not tau or historical replay",
                    }
                    save_private_json(output / f"{stem}-result.json", result)
                    print(json.dumps(result), flush=True)
    finally:
        await provider.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts"))
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=1)
    args = parser.parse_args()
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="decision-probe-", dir=args.output_parent))
    print(f"Output: {output}", flush=True)
    asyncio.run(run(output, args.repeats))


if __name__ == "__main__":
    main()
