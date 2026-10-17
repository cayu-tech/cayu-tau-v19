"""Paired Sol next decisions on preserved public-source binding context."""

import argparse
import asyncio
import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from cayu import Message
from cayu.providers.base import ModelRequest
from probe_model_efforts import decide

from cayu_tau3_banking.argument_contract import OPTIONAL_ARGUMENT_DEFAULTS, SOURCE_BINDING_RECOVERY
from cayu_tau3_banking.captured_argument_probe import CapturedTransport
from cayu_tau3_banking.decision_probe import save_private_json


def variants(source):
    if source["payload_redacted"]:
        raise ValueError("Source must be a complete captured payload")
    baseline = copy.deepcopy(source["payload"])
    if (
        baseline["instructions"].count(OPTIONAL_ARGUMENT_DEFAULTS) != 1
        or SOURCE_BINDING_RECOVERY in baseline["instructions"]
    ):
        raise ValueError("Unexpected captured guidance")
    baseline["model"] = "gpt-5.6-sol"
    baseline["reasoning"]["effort"] = "xhigh"
    baseline["input"] = [item for item in baseline["input"] if item.get("type") != "reasoning"]
    candidate = copy.deepcopy(baseline)
    candidate["instructions"] = baseline["instructions"].replace(
        OPTIONAL_ARGUMENT_DEFAULTS, OPTIONAL_ARGUMENT_DEFAULTS + "\n\n" + SOURCE_BINDING_RECOVERY
    )
    return {"baseline": baseline, "candidate": candidate}


def prepare(source):
    pair = variants(json.loads(source.read_text()))
    root = Path(tempfile.mkdtemp(prefix="binding-context-", dir="artifacts")).resolve()
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(
        project / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(__file__, root / "run.py")
    shutil.copy2(project / "scripts/probe_model_efforts.py", root / "probe_model_efforts.py")
    for name, payload in pair.items():
        save_private_json(root / f"{name}-payload.json", payload)
    save_private_json(
        root / "protocol.json",
        {
            "scope": "Two portable next decisions at the preserved final Astra binding failure context; no tools execute",
            "actor": "subscription/gpt-5.6-sol/xhigh",
            "retries": 0,
            "schedule": list(pair),
            "source": str(source.resolve()),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "changes": "Both arms use Sol xhigh and omit opaque reasoning items identically; only candidate adds generic source-binding recovery guidance",
            "limits": "Single decisions, not native continuation, original wire replay, full task, or reliability estimate. Existing erroneous plan conclusions are retained. No expected answer is supplied.",
        },
    )
    save_private_json(
        root / "manifest.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        },
    )
    return root


def verify(root):
    for name, digest in json.loads((root / "manifest.json").read_text()).items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen source changed")


async def run(root):
    import cayu_tau3_banking.cli as cli
    from cayu_tau3_banking.provider_diagnostics import DiagnosticOpenAITransport
    from cayu_tau3_banking.retrieval_eval import safe_error_diagnostic

    verify(root)
    if not Path(cli.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Use frozen source")
    for name in ("baseline", "candidate"):
        payload = json.loads((root / f"{name}-payload.json").read_text())
        provider = cli.build_provider(
            provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=180
        )
        provider.transport = CapturedTransport(
            DiagnosticOpenAITransport(provider.transport, root / name), payload
        )
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
        try:
            async with asyncio.timeout(300):
                result = await decide(provider, request)
            save_private_json(root / f"{name}-result.json", result)
            captures = list((root / name).glob("*/*.request.json"))
            statuses = list((root / name).glob("*/*.capture.json"))
            if len(captures) != 1 or len(statuses) != 1:
                raise ValueError("Missing unique capture")
            capture, status = (json.loads(p.read_text()) for p in (captures[0], statuses[0]))
            if capture["payload_redacted"] or capture["payload"] != payload:
                raise ValueError("Actual payload differs")
            if capture["capture_id"] != status["capture_id"] or any(
                status[k] for k in ("records_attempted", "sink_failures", "records_dropped")
            ):
                raise ValueError("Provider/capture errors")
            save_private_json(root / f"{name}-audit.json", {"actual_payload_matches": True})
            verify(root)
            print(
                json.dumps(
                    {
                        "arm": name,
                        "seconds": result["seconds"],
                        "calls": result["calls"],
                        "text": result["text"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            save_private_json(root / f"{name}-error.json", safe_error_diagnostic(error))
            raise
        finally:
            await provider.aclose()
    save_private_json(root / "completed.json", {"decisions": 2, "tools_executed": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.path))
    else:
        asyncio.run(run(args.path.resolve()))
