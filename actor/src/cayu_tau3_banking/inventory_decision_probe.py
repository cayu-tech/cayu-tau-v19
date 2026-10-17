"""Counterfactual one-decision probe: frozen request, instructions-only difference."""

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
from cayu_tau3_banking.condition_preservation import replace_condition_preservation
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.evidence_scope import EVIDENCE_SCOPE_GUIDANCE


def variant(payload, treatment, *, document=None, condition_preservation=False):
    if document is not None and condition_preservation:
        raise ValueError("Document and instruction treatments are mutually exclusive")
    result = copy.deepcopy(payload)
    if treatment:
        if condition_preservation:
            result["instructions"] = replace_condition_preservation(result["instructions"])
        elif document is None:
            result["instructions"] += "\n\n" + EVIDENCE_SCOPE_GUIDANCE
        else:
            result["input"].append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Public tool documentation (exact existing corpus revision):\n"
                            + json.dumps(document, ensure_ascii=False),
                        }
                    ],
                }
            )
    return result


async def main():
    from cayu_tau3_banking.cli import build_provider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    treatments = parser.add_mutually_exclusive_group()
    treatments.add_argument("--document", type=Path)
    treatments.add_argument("--condition-preservation", action="store_true")
    args = parser.parse_args()
    document = json.loads(args.document.read_text()) if args.document else None
    capture = json.loads(args.request.read_text())
    if (
        capture.get("schema") != "cayu-tau3.provider-request.v1"
        or capture.get("omitted_field_count") != 0
    ):
        raise ValueError("A complete prepared-request capture is required")
    root = Path(tempfile.mkdtemp(prefix="inventory-decision-", dir="artifacts"))
    save_private_json(
        root / "protocol.json",
        {
            "source": str(args.request.resolve()),
            "source_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest(),
            "repeats": args.repeats,
            "treatment": "Replace the shared-condition planning paragraph; no added task facts"
            if args.condition_preservation
            else document
            if document is not None
            else EVIDENCE_SCOPE_GUIDANCE,
            "change": "Append one unchanged public document; instructions, prior input/state, tools/settings unchanged"
            if document is not None
            else "instructions only; input, prior provider state, tool schemas and settings unchanged",
            "tools_executed": 0,
            "retries": 0,
            "limitations": "Counterfactual development decision, not a full task or reliability estimate",
        },
    )
    print(json.dumps({"artifacts": str(root)}), flush=True)
    for repeat in range(args.repeats):
        for treatment in (False, True) if repeat % 2 == 0 else (True, False):
            label = f"{repeat}-{'candidate' if treatment else 'baseline'}"
            payload = variant(
                capture["payload"],
                treatment,
                document=document,
                condition_preservation=args.condition_preservation,
            )
            save_private_json(root / f"{label}-request.json", payload)
            provider = build_provider(
                provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=180
            )
            provider.transport = CapturedTransport(provider.transport, payload)
            request = ModelRequest(
                model=payload["model"],
                messages=[Message.text("user", "Decision replay")],
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
            calls, texts, completed = [], [], 0
            try:
                async with asyncio.timeout(300):
                    async with aclosing(provider.runtime_stream(request)) as stream:
                        async for event in stream:
                            if event.type is ModelStreamEventType.TOOL_CALL:
                                calls.append(event.payload)
                            elif event.type is ModelStreamEventType.TEXT_DELTA:
                                texts.append(event.delta)
                            elif event.type is ModelStreamEventType.COMPLETED:
                                completed += 1
                            elif event.type is ModelStreamEventType.ERROR:
                                raise RuntimeError("Provider stream reported an error")
                if completed != 1:
                    raise RuntimeError("No complete decision")
                result = {
                    "case": label,
                    "seconds": time.monotonic() - started,
                    "calls": calls,
                    "text": "".join(texts),
                    "tools_executed": 0,
                }
                save_private_json(root / f"{label}-result.json", result)
                print(json.dumps(result), flush=True)
            except Exception as error:
                save_private_json(
                    root / f"{label}-error.json", {"error_type": type(error).__name__}
                )
                raise RuntimeError(f"Probe stopped: {type(error).__name__}") from None
            finally:
                await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
