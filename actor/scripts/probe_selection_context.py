"""Bounded instruction contrast at a saved decision; no tools execute or score changes.

The complete captured input, tools and reasoning are preserved. This tests one
decision under the recorded context, not retrieval or a resumed conversation.
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

GUIDANCE = """Before settling on a personalized product recommendation, inspect the
shortlisted product's material eligibility and cost conditions, not only its benefits
summary. Do this before spending the remaining research allowance on fine-grained
benefit comparisons. Apply customer facts already supplied; ask only for missing facts
that could change suitability. When terms or relevant customer facts remain unknown,
make the recommendation explicitly conditional and identify the unresolved condition.
Lack of research capacity is not evidence that a product is suitable. General public
comparisons may remain conditional without collecting unnecessary personal information.

Respect the scope of a selected combination. When the customer authorizes a bundle
conditionally, establish those conditions before making its component changes. If a
component becomes unavailable or a material term changes, explain the revised choice
and obtain the needed decision before partially executing or substituting products.
Do not treat later acceptance as prior authorization. Independently authorized work may
still proceed; do not invent coupling or reconfirm an unchanged, already-authorized choice.
""".strip()


def add_guidance(payload):
    candidate = copy.deepcopy(payload)
    instructions = candidate.get("instructions", "")
    if instructions.count("</instructions>") != 1 or GUIDANCE in instructions:
        raise ValueError("Expected one original instruction block without this treatment")
    candidate["instructions"] = instructions.replace(
        "</instructions>", f"\n{GUIDANCE}\n</instructions>"
    )
    return candidate


def add_document(payload, document):
    if not isinstance(document, dict) or any(
        not isinstance(document.get(key), str) or not document[key].strip()
        for key in ("id", "title", "content")
    ):
        raise ValueError("Expected one complete public corpus document")
    candidate = copy.deepcopy(payload)
    candidate["instructions"] += (
        "\n<additional_public_knowledge>\n"
        + json.dumps({k: document[k] for k in ("id", "title", "content")}, ensure_ascii=False)
        + "\n</additional_public_knowledge>"
    )
    return candidate


async def run(capture_path: Path, repeats: int, document_path: Path | None = None):
    from cayu_tau3_banking.cli import build_provider

    capture = json.loads(capture_path.read_text())
    if (
        capture.get("schema") != "cayu-tau3.provider-request.v1"
        or capture.get("omitted_field_count") != 0
    ):
        raise ValueError("Requires a complete saved provider request")
    original = capture["payload"]
    document = json.loads(document_path.read_text()) if document_path is not None else None
    treatment = "public_source" if document is not None else "guidance"
    candidate = add_document(original, document) if document is not None else add_guidance(original)
    root = Path(tempfile.mkdtemp(prefix="selection-context-", dir="artifacts"))
    save_private_json(
        root / "protocol.json",
        {
            "capture": str(capture_path.resolve()),
            "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "guidance": GUIDANCE if document is None else None,
            "additional_public_document": document,
            "additional_public_document_sha256": hashlib.sha256(
                document_path.read_bytes()
            ).hexdigest()
            if document_path is not None
            else None,
            "repeats_per_arm": repeats,
            "timeout_seconds": 300,
            "changes": "Append one complete public knowledge document; no guidance, customer facts or tools changed."
            if document is not None
            else "Instructions only. No source documents, customer facts or tool schema changed.",
            "limits": "One-step public-development contrast, not a full task or reliability estimate. "
            "No tools execute, no private reference outcomes are included, no adaptive retries.",
        },
    )
    print(f"Output: {root}", flush=True)
    for repeat in range(repeats):
        for arm in ("baseline", treatment) if repeat % 2 == 0 else (treatment, "baseline"):
            payload = original if arm == "baseline" else candidate
            label = f"{repeat}-{arm}"
            save_private_json(root / f"{label}-request.json", payload)
            provider = build_provider(
                provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
            )
            provider.transport = CapturedTransport(provider.transport, payload)
            request = ModelRequest(
                model=payload["model"],
                messages=[Message.text("user", "Diagnostic replay")],
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
                    raise ValueError("Expected exactly one completed decision")
                result = {
                    "arm": arm,
                    "repeat": repeat,
                    "seconds": time.monotonic() - started,
                    "calls": calls,
                    "text": "".join(text),
                }
                save_private_json(root / f"{label}-result.json", result)
                print(json.dumps(result), flush=True)
            except Exception as error:
                save_private_json(
                    root / f"{label}-error.json",
                    {
                        "error_type": type(error).__name__,
                        "seconds": time.monotonic() - started,
                        "diagnostics": getattr(error, "diagnostics", {}),
                    },
                )
                raise
            finally:
                await provider.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--treatment-document",
        type=Path,
        help="Contrast one extra public document instead of instruction guidance",
    )
    args = parser.parse_args()
    asyncio.run(run(args.capture, args.repeats, args.treatment_document))


if __name__ == "__main__":
    main()
