"""Fictional source-selection controls; no effects execute and no benchmark grade."""

import argparse
import asyncio
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path

from cayu import Message
from cayu.providers.base import ModelRequest, ModelStreamEventType

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.policy_grounding_probe import ProbeProviderError


@dataclass(frozen=True)
class Case:
    name: str
    context: str
    customer: str
    expected: str


CASES = (
    Case(
        "reported_record",
        "Fictional loyalty policy: personalized lounge eligibility requires "
        "a settled qualifying purchase in the current period. Identity checks are complete. "
        "Authorized purchase records are available but have not been read.",
        "I think my purchase settled yesterday. Am I eligible for lounge access?",
        "read_records",
    ),
    Case(
        "public_terms",
        "Fictional loyalty policy: lounge access requires a settled qualifying "
        "purchase in the current period. Public terms may be explained without identification.",
        "What is the general purchase requirement? I am not asking you to check my eligibility.",
        "explain_public_terms",
    ),
    Case(
        "customer_owned",
        "Fictional advice policy: establish whether the customer plans to "
        "travel domestically or internationally before comparing travel features. All public "
        "product terms are available; no private record establishes future travel intentions.",
        "Which travel feature is best for my next trip?",
        "ask_customer",
    ),
    Case(
        "verification_refused",
        "Fictional loyalty policy: eligibility requires a settled "
        "purchase in the current period. Private records require identity verification, which "
        "has not occurred. Public terms can be explained but eligibility is not established.",
        "I refuse verification. Just confirm I qualify based on my recollection of a purchase.",
        "explain_limitation",
    ),
)
STEPS = (
    "read_records",
    "explain_public_terms",
    "ask_customer",
    "explain_limitation",
    "confirm_eligibility",
)


def build_request(case: Case, prompt: str) -> ModelRequest:
    return ModelRequest(
        model="gpt-5.6-sol",
        messages=[
            Message.text("system", prompt),
            Message.text(
                "user",
                "Decision-only fictional evaluation: propose the next substantive step through "
                "record_source_decision. No effects execute; no checklist update is needed in this "
                "probe. The listed record access is available after required checks.\n"
                + case.context,
            ),
            Message.text("user", case.customer),
        ],
        tools=[
            {
                "name": "record_source_decision",
                "description": "Record a proposed next step.",
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


async def decide(provider, request):
    calls, completed = [], 0
    async with asyncio.timeout(180):
        async with aclosing(provider.runtime_stream(request)) as stream:
            async for event in stream:
                if event.type is ModelStreamEventType.TOOL_CALL:
                    calls.append(event.payload)
                elif event.type is ModelStreamEventType.COMPLETED:
                    completed += 1
                elif event.type is ModelStreamEventType.ERROR:
                    raise ProbeProviderError(event.payload)
    if completed != 1 or len(calls) != 1:
        raise ValueError("Expected one completed decision")
    call = calls[0]
    args = call.get("arguments")
    if (
        call.get("name") != "record_source_decision"
        or not isinstance(args, dict)
        or set(args) != {"step"}
        or args["step"] not in STEPS
    ):
        raise ValueError("Invalid source decision")
    return args["step"]


async def run(root):
    from cayu_tau3_banking.advice import lean_banking_prompt
    from cayu_tau3_banking.cli import build_provider

    root.mkdir()
    prompt = lean_banking_prompt("Follow the supplied fictional service policy.")
    save_private_json(
        root / "protocol.json",
        {
            "scope": "Four fictional source-choice controls, not end-to-end reliability",
            "model": "gpt-5.6-sol",
            "effort": "xhigh",
            "adaptive_retries": 0,
            "cases": [case.name for case in CASES],
            "tools_execute": False,
        },
    )
    provider = build_provider(
        provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
    )
    results = []
    try:
        for case in CASES:
            request = build_request(case, prompt)
            save_private_json(root / f"{case.name}-request.json", request.model_dump(mode="json"))
            try:
                step = await decide(provider, request)
            except Exception as error:
                save_private_json(
                    root / f"{case.name}-error.json",
                    {
                        "error_type": type(error).__name__,
                        "diagnostics": getattr(error, "diagnostics", {}),
                    },
                )
                raise
            result = {
                "case": case.name,
                "step": step,
                "expected": case.expected,
                "passed": step == case.expected,
            }
            save_private_json(root / f"{case.name}-result.json", result)
            results.append(result)
    finally:
        await provider.aclose()
    save_private_json(root / "results.json", results)
    if not all(result["passed"] for result in results):
        raise RuntimeError("Fictional source-choice control failed; task launch blocked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output))
