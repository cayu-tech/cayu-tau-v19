"""Diagnostic-only 2x2 saved decision: unrelated context and research availability.

No actor code changes, bank tools, answer feedback, or benchmark grading. The
interventions address one historical decision, not production selection rules.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

from cayu import Message
from cayu.providers.base import ModelRequest
from probe_model_efforts import decide

from cayu_tau3_banking.captured_argument_probe import CapturedTransport
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.provenance import CAYU_COMMIT, assert_source_provenance

READ = "cayu_read_product_documents"
UNRELATED = "doc_bank_accounts_bank_accounts_(general)_031"
ARMS = ("baseline", "without_dispute", "reads_restored", "both")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory_part(payload):
    parts = [
        p
        for m in payload["input"]
        if m.get("role") == "user"
        for p in m.get("content", [])
        if "<banking_context>" in p.get("text", "")
    ]
    if len(parts) != 1:
        raise ValueError("Expected one complete application context")
    return parts[0]


def packet(text, tag):
    matches = list(re.finditer(f"<{tag}>(.*?)</{tag}>", text, re.S))
    if len(matches) != 1:
        raise ValueError(f"Expected unique {tag} section")
    return matches[0]


def replace_span(text, match, value):
    return text[: match.start(1)] + value + text[match.end(1) :]


def variants(capture, tool_capture):
    for source in (capture, tool_capture):
        if source["payload_redacted"] or source["omitted_field_count"]:
            raise ValueError("Complete, unredacted captures required")
    baseline = copy.deepcopy(capture["payload"])
    if baseline["model"] != "gpt-5.6-sol" or baseline["reasoning"]["effort"] != "xhigh":
        raise ValueError("Expected original Sol xhigh decision")
    if any(t["name"] == READ for t in baseline["tools"]):
        raise ValueError("Baseline read tool must be unavailable")
    schemas = [t for t in tool_capture["payload"]["tools"] if t["name"] == READ]
    if len(schemas) != 1:
        raise ValueError("Expected exact historical read schema")
    original_memory = memory_part(baseline)["text"]
    doc_match = packet(original_memory, "active-tool-documentation")
    docs = json.loads(doc_match[1])
    removed = [d for d in docs["documents"] if d["entry_id"] == UNRELATED]
    if len(removed) != 1 or not removed[0]["text_complete"]:
        raise ValueError("Expected one complete unrelated dispute procedure")
    original_capacity = packet(original_memory, "research-capacity")
    cap_lines = original_capacity[1].splitlines(keepends=True)
    cap_index = next(i for i, line in enumerate(cap_lines) if line.startswith('{"remaining_calls"'))
    capacity = json.loads(cap_lines[cap_index])
    if capacity["remaining_calls"][READ] != {"customer_calls": 0, "conversation_calls": 24}:
        raise ValueError("Unexpected historical budget boundary")
    if "Silver +0.45%" not in json.dumps(baseline):
        raise ValueError("Expected existing source-linked lead; never insert it")
    results = {}
    for arm in ARMS:
        value = copy.deepcopy(baseline)
        part = memory_part(value)
        if arm in ("without_dispute", "both"):
            changed = copy.deepcopy(docs)
            changed["documents"] = [d for d in changed["documents"] if d["entry_id"] != UNRELATED]
            changed["omitted"].append(
                {
                    "entry_id": UNRELATED,
                    "entry_revision": removed[0]["entry_revision"],
                    "reason": "diagnostic_context_ablation",
                }
            )
            part["text"] = replace_span(
                part["text"],
                packet(part["text"], "active-tool-documentation"),
                "\n" + json.dumps(changed, ensure_ascii=False) + "\n",
            )
        if arm in ("reads_restored", "both"):
            changed_capacity = copy.deepcopy(capacity)
            changed_capacity["remaining_calls"][READ]["customer_calls"] = 2
            lines = cap_lines.copy()
            lines[cap_index] = json.dumps(changed_capacity) + "\n"
            part["text"] = replace_span(
                part["text"],
                packet(part["text"], "research-capacity"),
                "".join(lines),
            )
            # Preserve the original schema position, without enabling any other tool.
            value["tools"].insert(1, copy.deepcopy(schemas[0]))
        if value["instructions"] != baseline["instructions"]:
            raise ValueError("Instructions changed")
        if value["input"][:-1] != baseline["input"][:-1]:
            raise ValueError("Summary, prior results or opaque provider state changed")
        for tag in (
            "workflow-plan",
            "active-work-evidence",
            "observation-ledger",
            "cayu-work-context",
        ):
            if packet(part["text"], tag)[1] != packet(original_memory, tag)[1]:
                raise ValueError(f"Uncontrolled change to {tag}")
        results[arm] = value
    assert results["baseline"] == capture["payload"]
    return results, removed[0]


def prepare(source, tool_source, parent):
    values, removed = variants(json.loads(source.read_text()), json.loads(tool_source.read_text()))
    root = Path(tempfile.mkdtemp(prefix="research-exhaustion-", dir=parent)).resolve()
    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(
        repo / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(__file__, root / "run.py")
    shutil.copy2(repo / "scripts/probe_model_efforts.py", root / "probe_model_efforts.py")
    schedule = []
    for repeat in range(2):
        for arm in ARMS if repeat == 0 else reversed(ARMS):
            name = f"{repeat}-{arm}"
            save_private_json(root / f"{name}-payload.json", values[arm])
            schedule.append({"name": name, "arm": arm, "repeat": repeat})
    save_private_json(root / "removed-document.json", removed)
    save_private_json(
        root / "protocol.json",
        {
            "source": str(source.resolve()),
            "source_sha256": digest(source),
            "tool_source": str(tool_source.resolve()),
            "tool_source_sha256": digest(tool_source),
            "framework_commit": CAYU_COMMIT,
            "tau_repository": str((repo / ".vendor/tau2").resolve()),
            "actor": "subscription/gpt-5.6-sol/xhigh",
            "scope": "Next decision at event 1142, before the incomplete comparison was marked resolved; no tool executes.",
            "schedule": schedule,
            "max_requests": 8,
            "maximum_concurrency": 1,
            "whole_request_retries": 0,
            "timeout_seconds_per_request": 300,
            "semantic_progress_timeout_seconds": 180,
            "capacity_treatment": "Restore exactly two per-customer document reads and the unchanged historical read schema. Search remains unavailable. Prior output budgets remain historical, current capacity is authoritative.",
            "context_treatment": "Remove only the unrelated previously delivered debit-card dispute procedure from active-tool-documentation; append an explicit omission. Keep all other source text, summary, plan, prior outputs, opaque provider state, instructions, and settings identical.",
            "grading": "Manually inspect whether the proposed next step investigates any unresearched source-linked companion, merely repeats known evidence, declares the limited no-card comparison complete, or accurately reports an unresolved comparison. Proposed reads are not executed and are not task passes. No evaluator action or additional product fact supplied.",
            "limits": "Two samples per cell on one known decision cannot establish reliability or quantify causal shares. Removing one distractor tests that distractor, not all context load. Retained opaque state cannot be inspected. New inferences are counterfactual next decisions, not a historical continuation or a banking episode. Restoring reads includes the necessary tool-schema change, not just a number.",
            "stopping": "Stop on transport, timeout, source-integrity or request-capture failure; preserve diagnostics. No silent model substitution.",
        },
    )
    save_private_json(
        root / "manifest.json",
        {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()},
    )
    return root


def verify(root):
    for name, expected in json.loads((root / "manifest.json").read_text()).items():
        if digest(root / name) != expected:
            raise ValueError("Frozen diagnostic source changed")


async def run(root):
    import cayu_tau3_banking.cli as cli
    from cayu_tau3_banking.provider_diagnostics import DiagnosticOpenAITransport
    from cayu_tau3_banking.retrieval_eval import safe_error_diagnostic

    verify(root)
    if not Path(cli.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Use frozen provider wrapper source")
    protocol = json.loads((root / "protocol.json").read_text())
    assert_source_provenance(Path(protocol["tau_repository"]))
    for key in ("source", "tool_source"):
        if digest(Path(protocol[key])) != protocol[key + "_sha256"]:
            raise ValueError("Historical capture changed")
    save_private_json(root / "started.json", {"time_unix": time.time(), "max_requests": 8})
    for row in protocol["schedule"]:
        name = row["name"]
        payload = json.loads((root / f"{name}-payload.json").read_text())
        provider = cli.build_provider(
            provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=180
        )
        provider.transport = CapturedTransport(
            DiagnosticOpenAITransport(provider.transport, root / name),
            payload,
        )
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
        print(json.dumps({"starting": name}), flush=True)
        try:
            async with asyncio.timeout(300):
                result = await decide(provider, request)
            save_private_json(root / f"{name}-result.json", {**row, **result})
            captures = list((root / name).glob("*/*.request.json"))
            statuses = list((root / name).glob("*/*.capture.json"))
            if len(captures) != 1 or len(statuses) != 1:
                raise ValueError("Expected one prepared request and status")
            captured, status = (json.loads(p.read_text()) for p in (captures[0], statuses[0]))
            if captured["payload_redacted"] or captured["payload"] != payload:
                raise ValueError("Actual request differs from frozen treatment")
            if captured["capture_id"] != status["capture_id"] or any(
                status[k] for k in ("records_attempted", "sink_failures", "records_dropped")
            ):
                raise ValueError("Provider/capture errors")
            save_private_json(
                root / f"{name}-audit.json",
                {"actual_payload_matches": True, "capture_sha256": digest(captures[0])},
            )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("source", type=Path)
    p.add_argument("tool_source", type=Path)
    p.add_argument("--parent", type=Path, default=Path("artifacts"))
    r = commands.add_parser("run")
    r.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.source, args.tool_source, args.parent))
    else:
        asyncio.run(run(args.root.resolve()))
