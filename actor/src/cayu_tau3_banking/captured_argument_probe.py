"""Replay captured decision payloads without executing any proposed tools.

Only instructions change in treatment. Captured input and tool schemas remain
identical. This is counterfactual decision evidence, not a full task replay.
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

from cayu_tau3_banking.argument_contract import replace_legacy_naming
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.policy_grounding_probe import ProbeProviderError


class CapturedTransport:
    def __init__(self, base, payload):
        self.base = base
        self.payload = copy.deepcopy(payload)

    async def stream_response_events(self, **kwargs):
        kwargs["payload"] = copy.deepcopy(self.payload)
        async with aclosing(self.base.stream_response_events(**kwargs)) as stream:
            async for event in stream:
                yield event

    def __getattr__(self, name):
        return getattr(self.base, name)


def variant(payload, treatment, *, drop_reasoning=False, contract_precedence=False):
    result = copy.deepcopy(payload)
    if treatment or contract_precedence:
        result["instructions"] = replace_legacy_naming(
            result["instructions"], contract_precedence=contract_precedence and treatment
        )
    if drop_reasoning:
        result["input"] = [item for item in result["input"] if item.get("type") != "reasoning"]
    return result


async def run(root, source, repeats, *, drop_reasoning=False, contract_precedence=False):
    from cayu_tau3_banking.cli import build_provider

    inputs = {
        "checking": source / "dabc342439d542cb8b27ebd924dc25aa.request.json",
        "savings": source / "e1f8bab196bf4c5b84246c63376b0099.request.json",
    }
    save_private_json(
        root / "protocol.json",
        {
            "scope": "Historical prepared-request decision contrast; no tools execute and no tau score",
            "repeats": repeats,
            "drop_reasoning": drop_reasoning,
            "contract_precedence": contract_precedence,
            "baseline_definition": "current ARGUMENT_GROUNDING"
            if contract_precedence
            else "historical legacy prompt",
            "treatment_definition": "candidate CONTRACT_PRECEDENCE"
            if contract_precedence
            else "ARGUMENT_GROUNDING",
            "cases": list(inputs),
            "conditions": ["baseline", "grounded"],
            "timeout_seconds": 300,
            "adaptive_retries": 0,
            "source_hashes": {
                name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in inputs.items()
            },
            "changed_field": (
                "Remove prior reasoning items from input in both conditions; also replace "
                "one legacy product-naming paragraph in grounded condition only"
                if drop_reasoning
                else "instructions only; replace one legacy product-naming paragraph"
            ),
            "limitations": "Public development contexts; no private reasoning or seed replay guarantee. "
            "One step may propose more investigation instead of an effect. Raw outputs retained, "
            "no automatic semantic grading or reference answers sent to the actor.",
        },
    )
    for repeat in range(repeats):
        for name, path in inputs.items():
            original = json.loads(path.read_text())["payload"]
            for treatment in (False, True) if repeat % 2 == 0 else (True, False):
                label = f"{repeat}-{name}-{'grounded' if treatment else 'baseline'}"
                payload = variant(
                    original,
                    treatment,
                    drop_reasoning=drop_reasoning,
                    contract_precedence=contract_precedence,
                )
                save_private_json(root / f"{label}-request.json", payload)
                provider = build_provider(
                    provider="subscription",
                    codex_lb_base_url="",
                    semantic_progress_timeout_seconds=180,
                )
                provider.transport = CapturedTransport(provider.transport, payload)
                request = ModelRequest(
                    model=payload["model"],
                    messages=[Message.text("user", "Replay")],
                    tools=[
                        {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "input_schema": t["parameters"],
                        }
                        for t in payload["tools"]
                    ],
                )
                started = time.monotonic()
                calls, text = [], []
                completed = 0
                try:
                    async with asyncio.timeout(300):
                        async with aclosing(provider.runtime_stream(request)) as stream:
                            async for event in stream:
                                if event.type is ModelStreamEventType.TOOL_CALL:
                                    calls.append(event.payload)
                                elif event.type is ModelStreamEventType.TEXT_DELTA:
                                    text.append(event.delta)
                                elif event.type is ModelStreamEventType.COMPLETED:
                                    completed += 1
                                elif event.type is ModelStreamEventType.ERROR:
                                    raise ProbeProviderError(event.payload)
                    if completed != 1:
                        raise ValueError("No complete decision")
                    result = {
                        "case": label,
                        "seconds": time.monotonic() - started,
                        "calls": calls,
                        "text": "".join(text),
                        "completed": completed,
                    }
                    save_private_json(root / f"{label}-result.json", result)
                    print(json.dumps(result), flush=True)
                except Exception as error:
                    save_private_json(
                        root / f"{label}-error.json",
                        {
                            "error_type": type(error).__name__,
                            "diagnostics": getattr(error, "diagnostics", {}),
                            "seconds": time.monotonic() - started,
                        },
                    )
                    raise
                finally:
                    await provider.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=1)
    parser.add_argument("--drop-reasoning", action="store_true")
    parser.add_argument("--contract-precedence", action="store_true")
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="captured-arguments-", dir=args.output_parent))
    print(f"Output: {root}", flush=True)
    asyncio.run(
        run(
            root,
            args.source,
            args.repeats,
            drop_reasoning=args.drop_reasoning,
            contract_precedence=args.contract_precedence,
        )
    )


if __name__ == "__main__":
    main()
