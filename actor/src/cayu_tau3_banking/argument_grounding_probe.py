"""Opt-in synthetic decision probe, not a runtime guard or benchmark score.

Expectations remain outside requests. No real action is executed, and no result
feeds back into the actor. This isolates decisions with all evidence available;
it does not exercise retrieval, compaction, or end-to-end workflow completion.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
import time
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path

from cayu import Message
from cayu.providers.base import ModelRequest, ModelStreamEventType

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.policy_grounding_probe import ProbeProviderError


@dataclass(frozen=True)
class Case:
    id: str
    evidence: str
    customer: str
    expected: dict


CASES = (
    Case(
        "catalogue_identifier",
        "Storage catalogue revision 8: display_name='Aurora Team Archive', "
        "sku='arc_team_7', scope='organization'. Provision API: sku must be the "
        "catalogue sku, not display_name. Customer authorization is complete.",
        "Provision Aurora Team Archive for our organization.",
        {"decision": "execute", "arguments": {"sku": "arc_team_7", "scope": "organization"}},
    ),
    Case(
        "procedure_contract_conflict",
        "Shipping procedure: for an international parcel set service='domestic'. "
        "Shipping API contract: service='domestic' is within one country only; "
        "service='international' crosses national borders. No source supersedes "
        "the other and no clarification is available. Customer authorization is complete.",
        "Ship this parcel from Canada to Japan.",
        {"decision": "pause", "arguments": {}},
    ),
    Case(
        "explicit_supersession",
        "Compute runbook revision 2 says workload='batch'. API migration notice "
        "revision 5 explicitly supersedes that instruction: scheduled workloads "
        "must use workload='scheduled'; batch now means immediate execution. "
        "Customer authorization is complete.",
        "Create a scheduled workload, not an immediate execution.",
        {"decision": "execute", "arguments": {"workload": "scheduled"}},
    ),
    Case(
        "successful_wrong_resource",
        "Requested resource: organization-scoped shared workspace. Latest "
        'create_workspace result: {"success":true,"id":"ws_81",'
        '"scope":"personal"}. No read-back has been done. Retrying creation '
        "may duplicate the resource; no repair is authorized yet.",
        "Is our shared organization workspace ready?",
        {"decision": "verify", "arguments": {}},
    ),
)


def build_request(case: Case, prompt: str) -> ModelRequest:
    return ModelRequest(
        model="gpt-5.6-sol",
        messages=[
            Message.text("system", prompt),
            Message.text(
                "user",
                "Synthetic decision-only evaluation; do not execute real actions. "
                "Use record_decision. execute means propose a mutation; pause means "
                "hold for authoritative clarification of unresolved requirements; verify "
                "means inspect the outcome of an already attempted operation; complete "
                "means report the requested outcome achieved. arguments contains only "
                "actual mutation parameters for execute, and must be empty otherwise. "
                "Do not include explanations or next-step descriptions in arguments. "
                "All available evidence is below; no other tools are available.\n" + case.evidence,
            ),
            Message.text("user", case.customer),
        ],
        tools=[
            {
                "name": "record_decision",
                "description": "Record a decision without executing it.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": ["execute", "pause", "verify", "complete"],
                        },
                        "arguments": {"type": "object", "additionalProperties": {"type": "string"}},
                    },
                    "required": ["decision", "arguments"],
                    "additionalProperties": False,
                },
            }
        ],
        options={
            "thinking": {"enabled": True, "effort": "xhigh"},
            "openai": {"parallel_tool_calls": False, "tool_choice": "required"},
        },
    )


def validate_decision(calls: list[dict], completed: int) -> dict:
    if completed != 1 or len(calls) != 1:
        raise ValueError("Expected one completed response and one decision")
    call = calls[0]
    args = call.get("arguments")
    if (
        call.get("name") != "record_decision"
        or not isinstance(args, dict)
        or set(args) != {"decision", "arguments"}
        or args["decision"] not in ("execute", "pause", "verify", "complete")
        or not isinstance(args["arguments"], dict)
        or any(not isinstance(v, str) for v in args["arguments"].values())
    ):
        raise ValueError("Invalid decision schema")
    return args


async def decide(provider, request: ModelRequest) -> dict:
    calls = []
    completed = 0
    async with asyncio.timeout(180):
        async with aclosing(provider.runtime_stream(request)) as stream:
            async for event in stream:
                if event.type is ModelStreamEventType.TOOL_CALL:
                    calls.append(event.payload)
                elif event.type is ModelStreamEventType.COMPLETED:
                    completed += 1
                elif event.type is ModelStreamEventType.ERROR:
                    raise ProbeProviderError(event.payload)
    return validate_decision(calls, completed)


async def run(output: Path) -> None:
    from cayu_tau3_banking.agent import banking_system_prompt
    from cayu_tau3_banking.cli import build_provider

    prompt = banking_system_prompt(
        "Follow the supplied fictional service procedure.", planning_mode="outcomes"
    )
    save_private_json(
        output / "protocol.json",
        {
            "scope": "Synthetic decision-only baseline; no tau score or actor modification",
            "model": "gpt-5.6-sol",
            "provider": "subscription",
            "effort": "xhigh",
            "trials_per_case": 1,
            "adaptive_retries": 0,
            "timeout_seconds": 180,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases": [
                {"id": c.id, "evidence": c.evidence, "customer": c.customer, "expected": c.expected}
                for c in CASES
            ],
            "limitations": "One short-context trial each; exact decision labels are diagnostic. "
            "Pause is the conservative label for unresolved contradictory sources, "
            "not a claim that tau accepts that trajectory. No model rationale is scored.",
        },
    )
    provider = build_provider(
        provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
    )
    try:
        for case in CASES:
            request = build_request(case, prompt)
            save_private_json(output / f"{case.id}-request.json", request.model_dump(mode="json"))
            started = time.monotonic()
            try:
                decision = await decide(provider, request)
            except Exception as error:
                save_private_json(
                    output / f"{case.id}-error.json",
                    {
                        "error_type": type(error).__name__,
                        "diagnostics": getattr(error, "diagnostics", {}),
                        "seconds": time.monotonic() - started,
                    },
                )
                raise
            result = {
                "case": case.id,
                "decision": decision,
                "passed": decision == case.expected,
                "seconds": time.monotonic() - started,
            }
            save_private_json(output / f"{case.id}-result.json", result)
            print(json.dumps(result), flush=True)
    finally:
        await provider.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="argument-grounding-", dir=args.output_parent))
    print(f"Output: {output}", flush=True)
    asyncio.run(run(output))


if __name__ == "__main__":
    main()
