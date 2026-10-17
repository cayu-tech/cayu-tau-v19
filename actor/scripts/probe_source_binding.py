"""Frozen paired Sol decisions for public-source binding; no effects execute."""

import argparse
import asyncio
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

from cayu.providers.base import ModelRequest

from cayu_tau3_banking.argument_contract import SOURCE_BINDING_RECOVERY
from cayu_tau3_banking.argument_grounding_probe import Case, build_request, decide
from cayu_tau3_banking.captured_guidance_probe import checked_guidance_payload
from cayu_tau3_banking.decision_probe import save_private_json

COMMON = (
    "The customer is authenticated and authorized for one creation. All other business "
    "prerequisites passed. Only the supplied public sources can resolve the API representation. "
)
CASES = (
    Case(
        "sourced_name_and_type",
        COMMON + "Create API fields: name is the full official product name ending in 'Vault'; "
        "scope is 'team' or 'personal'. Directory: Harbor Vault (team). The team product's own "
        "document says: 'Your Harbor Vault stores the team's shared records.' The personal "
        "product has its own document. A prior proposal name='Harbor Vault (team)' was rejected "
        "before dispatch because name must end in 'Vault'. No creation occurred.",
        "Create one Harbor Vault (team) for our team. I authorize that product, not personal storage.",
        {"decision": "execute", "arguments": {"name": "Harbor Vault", "scope": "team"}},
    ),
    Case(
        "title_only_unresolved",
        COMMON + "Create API fields: name is the official product name ending in 'Vault'; "
        "scope is 'team'. Directory title is Harbor Vault (team), but no product body or "
        "canonical-name mapping is available. The directory explicitly says titles are "
        "editorial labels, not official product names. A prior proposal copying the title "
        "was rejected before dispatch. No creation occurred.",
        "Create the team product in that directory.",
        {"decision": "pause", "arguments": {}},
    ),
    Case(
        "unresolved_variant",
        COMMON + "Create API takes name and region. The name 'Harbor Vault' identifies two "
        "equally eligible variants: EU region='eu', US region='us'. The directory calls both "
        "Harbor Vault (team). Neither source nor customer selects a region. No default exists.",
        "Create one Harbor Vault for my team.",
        {"decision": "pause", "arguments": {}},
    ),
    Case(
        "literal_parentheses",
        COMMON + "Create API takes sku. Catalogue for display name Amber historical storage "
        "explicitly defines sku='Amber Vault (archive)'. Parentheses are literal parts of "
        "that SKU; no ending-in-Vault requirement applies. Product prose calls it Amber Vault.",
        "Create Amber historical storage. I authorize one creation.",
        {"decision": "execute", "arguments": {"sku": "Amber Vault (archive)"}},
    ),
    Case(
        "related_product_not_identity",
        COMMON + "Create API takes name ending in 'Vault' and scope='team'. Directory lists "
        "Harbor Vault (team), but its name mapping is unavailable. A different product's "
        "document is titled Harbor Vault (personal) and says 'Your Harbor Vault stores your "
        "personal files.' The sources do not establish that these products share an official name.",
        "Create the team product, not the personal product.",
        {"decision": "pause", "arguments": {}},
    ),
    Case(
        "undocumented_code",
        COMMON + "Create API takes sku, an opaque catalogue code, not a product name. "
        "Directory title is Harbor Vault (team). Its own body says 'Your Harbor Vault stores "
        "shared records.' No SKU is documented. Guessed values may create a different product.",
        "Create one Harbor Vault (team).",
        {"decision": "pause", "arguments": {}},
    ),
)


def requests(case, prompt):
    marker = "\n\n" + SOURCE_BINDING_RECOVERY
    if prompt.count(marker) != 1:
        raise ValueError("Expected exactly one recovery instruction")
    baseline = build_request(case, prompt.replace(marker, ""))
    candidate = build_request(case, prompt)
    a, b = (r.model_dump(mode="json") for r in (baseline, candidate))
    b["messages"][0] = a["messages"][0]
    if a != b:
        raise ValueError("Paired requests differ beyond instructions")
    return {"baseline": baseline, "candidate": candidate}


def prepare(parent):
    from cayu_tau3_banking.agent import banking_system_prompt

    root = Path(tempfile.mkdtemp(prefix="source-binding-", dir=parent)).resolve()
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(
        project / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(__file__, root / "run.py")
    prompt = banking_system_prompt(
        "Follow the supplied fictional service policy.", planning_mode="outcomes"
    )
    schedule = []
    for i, case in enumerate(CASES):
        pair = requests(case, prompt)
        for arm in ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline"):
            name = f"{case.id}-{arm}"
            request = pair[arm]
            save_private_json(root / f"{name}-request.json", request.model_dump(mode="json"))
            save_private_json(
                root / f"{name}-payload.json",
                checked_guidance_payload(request, expected=request.messages[0].content[0].text),
            )
            schedule.append({"name": name, "arm": arm, "expected": case.expected})
    save_private_json(
        root / "protocol.json",
        {
            "scope": "Twelve paired synthetic decisions, no tools executed, not a benchmark score",
            "actor": "subscription/gpt-5.6-sol/xhigh",
            "repeats": 1,
            "retries": 0,
            "schedule": schedule,
            "timeout_seconds": 180,
            "gate": "All six candidate decisions must pass before native qualification; report both arms",
            "limits": "Single short-context decisions, not evidence of full-task reliability or statistically proven improvement",
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
    for name, expected in json.loads((root / "manifest.json").read_text()).items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen source changed")


def completed_result(root, row):
    """Validate saved completion before skipping it; never retry an uncertain call."""
    name = row["name"]
    path = root / f"{name}-result.json"
    error = root / f"{name}-error.json"
    if not path.exists():
        if error.exists() or (root / name).exists():
            raise ValueError("Uncertain prior dispatch requires explicit investigation")
        return None
    result = json.loads(path.read_text())
    if any(result.get(k) != v for k, v in row.items()):
        raise ValueError("Saved result differs from protocol")
    if result.get("passed") is not (result.get("decision") == row["expected"]):
        raise ValueError("Saved grade is inconsistent")
    if not result.get("actual_payload_matches"):
        raise ValueError("Saved completion lacks a payload audit")
    if error.exists():
        detail = json.loads(error.read_text())
        if detail.get("error_type") != "BrokenPipeError":
            raise ValueError("Only a completed output-pipe interruption can be resumed")
    captures = list((root / name).glob("*/*.request.json"))
    statuses = list((root / name).glob("*/*.capture.json"))
    if len(captures) != 1 or len(statuses) != 1:
        raise ValueError("Saved capture is incomplete")
    capture, status = (json.loads(p.read_text()) for p in (captures[0], statuses[0]))
    if capture["payload_redacted"] or capture["payload"] != json.loads(
        (root / f"{name}-payload.json").read_text()
    ):
        raise ValueError("Saved payload differs")
    if capture["capture_id"] != status["capture_id"] or any(
        status[k] for k in ("records_attempted", "sink_failures", "records_dropped")
    ):
        raise ValueError("Saved capture has errors")
    return result


async def run(root, *, resume=False):
    import cayu_tau3_banking.cli as cli
    from cayu_tau3_banking.provider_diagnostics import DiagnosticOpenAITransport
    from cayu_tau3_banking.retrieval_eval import safe_error_diagnostic

    verify(root)
    if not Path(cli.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("Use frozen source")
    if resume:
        if not (root / "started.json").exists() or (root / "completed.json").exists():
            raise ValueError("Resume requires an incomplete started run")
        save_private_json(
            root / "resumed.json",
            {
                "time_unix": time.time(),
                "runner": str(Path(__file__).resolve()),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "reason": "Output-pipe interruption; completed requests are audited and skipped",
            },
        )
    else:
        save_private_json(root / "started.json", {"time_unix": time.time()})
    rows = []
    for row in json.loads((root / "protocol.json").read_text())["schedule"]:
        name = row["name"]
        if resume:
            saved = completed_result(root, row)
            if saved is not None:
                rows.append(saved)
                continue
        provider = cli.build_provider(
            provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
        )
        provider.transport = DiagnosticOpenAITransport(provider.transport, root / name)
        started = time.monotonic()
        try:
            request = ModelRequest.model_validate(
                json.loads((root / f"{name}-request.json").read_text())
            )
            decision = await decide(provider, request)
            captures = list((root / name).glob("*/*.request.json"))
            statuses = list((root / name).glob("*/*.capture.json"))
            if len(captures) != 1 or len(statuses) != 1:
                raise ValueError("Missing unique request/capture status")
            capture = json.loads(captures[0].read_text())
            status = json.loads(statuses[0].read_text())
            if capture["payload_redacted"] or capture["payload"] != json.loads(
                (root / f"{name}-payload.json").read_text()
            ):
                raise ValueError("Actual payload changed")
            if capture["capture_id"] != status["capture_id"] or any(
                status[k] for k in ("records_attempted", "sink_failures", "records_dropped")
            ):
                raise ValueError("Provider/capture errors")
            result = {
                **row,
                "decision": decision,
                "passed": decision == row["expected"],
                "seconds": time.monotonic() - started,
                "actual_payload_matches": True,
            }
            save_private_json(root / f"{name}-result.json", result)
            rows.append(result)
            verify(root)
            print(
                json.dumps(
                    {
                        "name": name,
                        "passed": result["passed"],
                        "seconds": round(result["seconds"], 2),
                    }
                ),
                flush=True,
            )
        except Exception as error:
            save_private_json(root / f"{name}-error.json", safe_error_diagnostic(error))
            raise
        finally:
            await provider.aclose()
    qualified = all(row["passed"] for row in rows if row["arm"] == "candidate")
    save_private_json(
        root / "completed.json",
        {"rows": rows, "candidate_qualified": qualified, "tools_executed": 0},
    )
    if not qualified:
        raise SystemExit("Candidate decision gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(Path("artifacts")))
    elif args.root is None:
        parser.error("run requires --root")
    else:
        asyncio.run(run(args.root.resolve(), resume=args.resume))


if __name__ == "__main__":
    main()
