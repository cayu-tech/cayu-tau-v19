"""Matched model/effort decisions on one preserved banking failure context.

Diagnostic only: no tools execute, no prompts change, and no production model is
switched. A proposed research step is not a completed investigation or task pass.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import shutil
import tempfile
import time
from contextlib import aclosing
from pathlib import Path

from cayu.providers.base import ModelFinishReason, ModelRequest, ModelStreamEventType

from cayu_tau3_banking.captured_guidance_probe import checked_guidance_payload
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.provenance import CAYU_COMMIT, TAU2_COMMIT

CONFIGURATIONS = {
    "sol-xhigh": ("gpt-5.6-sol", "xhigh"),
    "astra-medium": ("gpt-6-astra", "medium"),
    "astra-high": ("gpt-6-astra", "high"),
    "astra-xhigh": ("gpt-6-astra", "xhigh"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variants(source):
    baseline = ModelRequest.model_validate(source)
    if baseline.model != "gpt-5.6-sol" or baseline.options.get("thinking") != {
        "enabled": True,
        "effort": "xhigh",
    }:
        raise ValueError("Expected the declared Sol xhigh baseline")
    if len([m for m in baseline.messages if m.role == "system"]) != 1:
        raise ValueError("Expected exactly one system message")
    if not baseline.messages or baseline.messages[0].role != "system":
        raise ValueError("Expected leading instructions")
    if len(baseline.messages[0].content) != 1 or baseline.messages[0].content[0].type != "text":
        raise ValueError("Expected a single instruction text block")
    if any(
        p.type not in {"text", "tool_call", "tool_result"}
        for m in baseline.messages
        for p in m.content
    ):
        raise ValueError("Expected portable context without opaque reasoning state")
    expected = baseline.messages[0].content[0].text
    base_payload = checked_guidance_payload(baseline, expected=expected)
    result = {}
    for name, (model, effort) in CONFIGURATIONS.items():
        raw = copy.deepcopy(source)
        raw["model"] = model
        raw["options"]["thinking"]["effort"] = effort
        request = ModelRequest.model_validate(raw)
        payload = checked_guidance_payload(request, expected=expected)
        normalized = copy.deepcopy(payload)
        normalized["model"] = base_payload["model"]
        normalized["reasoning"]["effort"] = base_payload["reasoning"]["effort"]
        if normalized != base_payload:
            raise ValueError("Prepared payload changed beyond model and reasoning effort")
        result[name] = raw
    return result


def prepare(parent, source):
    source = source.resolve()
    raw = json.loads(source.read_text())
    candidates = variants(raw)
    root = Path(tempfile.mkdtemp(prefix="model-efforts-", dir=parent)).resolve()
    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(
        repo / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(__file__, root / "run.py")
    save_private_json(root / "reference-request.json", raw)
    schedule = []
    order = tuple(CONFIGURATIONS)
    for repeat in range(2):
        for arm in order if repeat == 0 else reversed(order):
            name = f"{repeat}-{arm}"
            save_private_json(root / f"{name}-request.json", candidates[arm])
            schedule.append(
                {
                    "name": name,
                    "arm": arm,
                    "model": CONFIGURATIONS[arm][0],
                    "effort": CONFIGURATIONS[arm][1],
                }
            )
    save_private_json(
        root / "protocol.json",
        {
            "source_request": str(source),
            "source_request_sha256": digest(source),
            "framework_commit": CAYU_COMMIT,
            "tau_commit": TAU2_COMMIT,
            "schedule": schedule,
            "provider": "existing subscription",
            "timeout_seconds_per_request": 300,
            "semantic_progress_timeout_seconds": 180,
            "maximum_concurrency": 1,
            "whole_request_retries": 0,
            "max_requests": 8,
            "scope": "Eight fresh portable next decisions at the real task-066 recommendation boundary. Same retained conclusions, public sources, customer history, workflow, tools and newer generic guidance. Only requested model and reasoning effort differ.",
            "preflight": "Exact compiled-payload equality after normalizing only model and reasoning effort; single instruction block. Expected actual payload persisted for post-run audit.",
            "grade": "Manually review pursuit of the source-linked card benefit, whether cost/eligibility/consent stay unresolved when appropriate, interest-allocation disclosure, and preservation of mature-account sequencing. Do not grade a proposed research call as completed investigation or a final response.",
            "limits": "Two decisions per configuration on one known failure context are not a reliability estimate, exact historical wire replay, native-runtime continuation, benchmark pass or full-task score. Opaque reasoning/provider state was removed identically before this comparison. No reference action is supplied and no proposed tool executes.",
            "stopping": "Stop remaining calls on provider, timeout, capture or payload-integrity errors. No automatic whole-request retry or substitution of unavailable models.",
        },
    )
    for row in schedule:
        request = ModelRequest.model_validate(candidates[row["arm"]])
        save_private_json(
            root / f"{row['name']}-expected-payload.json",
            checked_guidance_payload(request, expected=request.messages[0].content[0].text),
        )
    save_private_json(
        root / "manifest.json",
        {"files": {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}},
    )
    return root


def verify(root):
    for path, expected in json.loads((root / "manifest.json").read_text())["files"].items():
        if digest(root / path) != expected:
            raise ValueError("Frozen comparison source changed")


async def decide(provider, request):
    calls, text, completion = [], [], None
    usage, returned_model = None, None
    started, first_text = time.monotonic(), None
    async with aclosing(provider.runtime_stream(request)) as stream:
        async for event in stream:
            if event.type is ModelStreamEventType.TOOL_CALL:
                calls.append(event.payload)
            elif event.type is ModelStreamEventType.TEXT_DELTA:
                text.append(event.delta)
                if event.delta and first_text is None:
                    first_text = time.monotonic() - started
            elif event.type is ModelStreamEventType.ERROR:
                raise RuntimeError("Provider error event; inspect captured diagnostics")
            elif event.type is ModelStreamEventType.COMPLETED:
                if completion is not None or event.completion is None:
                    raise ValueError("Expected one completion with finish metadata")
                completion = event.completion
                # Usage belongs to the terminal event payload, not ModelCompletion.
                usage = event.payload.get("usage")
                returned_model = event.payload.get("model")
    if (
        completion is None
        or completion.finish_reason not in {ModelFinishReason.STOP, ModelFinishReason.TOOL_CALLS}
        or not (calls or any(text))
    ):
        raise ValueError("Expected a nonempty, nontruncated completed decision")
    return {
        "text": "".join(text),
        "calls": calls,
        "usage": usage,
        "returned_model": returned_model,
        "completion": completion.model_dump(mode="json"),
        "seconds": time.monotonic() - started,
        "first_text_seconds": first_text,
        "tools_executed": 0,
    }


def audit_capture(root, name):
    paths = list((root / name).glob("provider-diagnostics/*/*.request.json"))
    statuses = list((root / name).glob("provider-diagnostics/*/*.capture.json"))
    if len(paths) != 1 or len(statuses) != 1:
        raise ValueError("Expected exactly one prepared-request capture and status")
    capture, status = json.loads(paths[0].read_text()), json.loads(statuses[0].read_text())
    expected = json.loads((root / f"{name}-expected-payload.json").read_text())
    if capture["payload_redacted"] or capture["payload"] != expected:
        raise ValueError("Actual request differs from qualified payload")
    if capture["capture_id"] != status["capture_id"] or any(
        status[k] != 0 for k in ("records_attempted", "sink_failures", "records_dropped")
    ):
        raise ValueError("Provider/capture errors require inspection")
    return {
        "actual_payload_matches": True,
        "capture_sha256": digest(paths[0]),
        "capture_status_sha256": digest(statuses[0]),
    }


async def run(root):
    import cayu_tau3_banking.cli as cli
    from cayu_tau3_banking.provider_diagnostics import DiagnosticOpenAITransport
    from cayu_tau3_banking.retrieval_eval import safe_error_diagnostic

    verify(root)
    if not Path(cli.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Live comparison must use frozen provider wrapper source")
    protocol = json.loads((root / "protocol.json").read_text())
    variants(json.loads((root / "reference-request.json").read_text()))
    save_private_json(root / "started.json", {"time_unix": time.time(), "max_requests": 8})
    for row in protocol["schedule"]:
        name = row["name"]
        request = ModelRequest.model_validate(
            json.loads((root / f"{name}-request.json").read_text())
        )
        provider = cli.build_provider(
            provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=180
        )
        provider.transport = DiagnosticOpenAITransport(
            provider.transport, root / name / "provider-diagnostics"
        )
        print(json.dumps({"starting": name}), flush=True)
        try:
            async with asyncio.timeout(300):
                result = await decide(provider, request)
            save_private_json(root / f"{name}-result.json", {**row, **result})
            audit = audit_capture(root, name)
            save_private_json(root / f"{name}-audit.json", audit)
            verify(root)
            print(
                json.dumps(
                    {
                        "completed": name,
                        "seconds": round(result["seconds"], 2),
                        "proposed_calls": len(result["calls"]),
                    }
                ),
                flush=True,
            )
        except Exception as error:
            save_private_json(root / f"{name}-error.json", safe_error_diagnostic(error))
            raise
        finally:
            await provider.aclose()
    save_private_json(root / "completed.json", {"requests_completed": 8, "tools_executed": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-parent", type=Path, default=Path("artifacts"))
    r = commands.add_parser("run")
    r.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.output_parent, args.source))
    else:
        asyncio.run(run(args.root.resolve()))


if __name__ == "__main__":
    main()
