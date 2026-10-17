"""Local paired response-sensitivity test; no environment actions or tau scoring.

Preserves captured composed messages, including provider state. Only the chosen
summary passage differs within each pair. Tool schemas were not captured: this
requests a text continuation with no tools and is NOT an exact runtime replay.
"""

import argparse
import asyncio
import hashlib
import json
import tempfile
import time
from contextlib import aclosing
from copy import deepcopy
from pathlib import Path

from cayu import Message
from cayu.providers.base import ModelRequest, ModelStreamEventType
from cayu.providers.openai import build_openai_payload

from cayu_tau3_banking.decision_probe import save_private_json


def validate_native_request(request):
    # The portable-message preflight is for provider-independent messages and
    # rejects native reasoning/replay state. Preserve that state and validate
    # using the same native payload builder as the subscription stream instead.
    build_openai_payload(request, stream=True, reasoning_state="inline")


def paired_messages(snapshot, before: str, after: str):
    if not before or before == after or not after:
        raise ValueError("Intervention must replace one nonempty passage with different text.")
    summary = snapshot["compaction"]["summary"]
    if summary.count(before) != 1:
        raise ValueError("Expected exactly one intervention passage in the saved summary.")
    baseline = deepcopy(snapshot["messages"])
    changed = deepcopy(baseline)
    replacements = 0
    for message in changed:
        for part in message["content"]:
            if (
                part["type"] == "text"
                and part["text"] == "Previous session context summary:\n" + summary
            ):
                part["text"] = part["text"].replace(before, after, 1)
                replacements += 1
    if replacements != 1:
        raise ValueError(
            "Expected exactly one complete summary message; refusing approximate editing."
        )
    return baseline, changed


async def response(provider, request, *, timeout=180):
    chunks = []
    completed = 0
    async with asyncio.timeout(timeout):
        async with aclosing(provider.runtime_stream(request)) as stream:
            async for event in stream:
                if event.type is ModelStreamEventType.TEXT_DELTA:
                    chunks.append(event.delta or "")
                elif event.type is ModelStreamEventType.TOOL_CALL:
                    raise ValueError(
                        "Text-only sensitivity probe unexpectedly produced a tool call."
                    )
                elif event.type is ModelStreamEventType.COMPLETED:
                    completed += 1
    text = "".join(chunks)
    if completed != 1 or not text.strip():
        raise ValueError("Probe did not produce one complete nonempty response.")
    return text


async def run(snapshot, intervention, output, repeats):
    from cayu_tau3_banking.cli import build_provider

    baseline, changed = paired_messages(snapshot, intervention["before"], intervention["after"])
    provider = build_provider(
        provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
    )
    save_private_json(
        output / "manifest.json",
        {
            "source_model_step_id": snapshot["model_step_id"],
            "intervention": intervention,
            "repeats": repeats,
            "scope": "text-only continuation; original tool schemas unavailable",
            "model": snapshot["model"],
            "reasoning_effort": "xhigh",
            "automatic_retries": 0,
            "per_response_timeout_seconds": 180,
        },
    )
    try:
        for repeat in range(repeats):
            for edited in (False, True) if repeat % 2 == 0 else (True, False):
                stem = f"{repeat}-{'edited' if edited else 'baseline'}"
                request = ModelRequest(
                    model=snapshot["model"],
                    messages=[
                        Message.model_validate(message)
                        for message in (changed if edited else baseline)
                    ],
                    tools=[],
                    options={
                        "thinking": {"enabled": True, "effort": "xhigh"},
                        "openai": {"parallel_tool_calls": False},
                    },
                )
                started = time.monotonic()
                stage = "native_payload_validation"
                try:
                    validate_native_request(request)
                    stage = "request_snapshot"
                    serialized = request.model_dump(mode="json")
                    save_private_json(output / f"{stem}-request.json", serialized)
                    stage = "provider_stream"
                    text = await response(provider, request)
                except Exception as error:
                    save_private_json(
                        output / f"{stem}-error.json",
                        {"error_type": type(error).__name__, "stage": stage},
                    )
                    raise RuntimeError(f"Stopped at {stem}: {type(error).__name__}") from None
                save_private_json(
                    output / f"{stem}-result.json",
                    {
                        "response": text,
                        "seconds": time.monotonic() - started,
                        "request_sha256": hashlib.sha256(
                            json.dumps(serialized, sort_keys=True).encode()
                        ).hexdigest(),
                    },
                )
                print(
                    json.dumps(
                        {"condition": stem, "seconds": time.monotonic() - started, "response": text}
                    ),
                    flush=True,
                )
    finally:
        await provider.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("intervention", type=Path)
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts"))
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=3)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    intervention = json.loads(args.intervention.read_text())
    paired_messages(snapshot, intervention["before"], intervention["after"])
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="summary-probe-", dir=args.output_parent))
    print(f"Output: {output}", flush=True)
    asyncio.run(run(snapshot, intervention, output, args.repeats))


if __name__ == "__main__":
    main()
