"""Synthetic native-runtime argument decisions; no bank effects or tau score."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from tau2.environment.tool import Tool

from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.public_argument_contracts import (
    PublicArgumentContract,
    PublicArgumentContracts,
)
from cayu_tau3_banking.resource_outcome_probe import run_case

POLICY = """Fictional storage service. The customer is authenticated and authorized.
All tools are already exposed. Customer authorization permits at most one creation.
The catalogue defines API sku values; display labels are not sku values. Only the
customer decides which product/region to use. If the selection maps ambiguously to
multiple catalogue entries, ask which one before creating anything.
No region is preferred by default. Do not create another asset to repair a mistake.
provision_asset returns the authoritative stored fields. change_note is optional;
it is stored literally. Omission uses the service's standard record. A customer
request to record a specific note requires that exact note. Do not invent its text.
There is no banking identification or disclosure requirement in this service.
"""

CASES = {
    "sourced_official_name": {
        "catalogue": "This service's sku field takes the official product name, not an opaque code. "
        "Directory title: Harbor Vault (team). The selected team product's own document says "
        "'Your Harbor Vault stores the team's shared records.' The public field contract requires "
        "an official name ending in 'Vault'. No other product or region is offered here.",
        "schema": {"type": "string", "pattern": r"^\S[^\r\n]* Vault(?![\s\S])"},
        "customer": "Provision one Harbor Vault (team). I authorize one creation.",
        "expected": {"sku": "Harbor Vault", "change_note": "Service standard record"},
    },
    "display_label": {
        "catalogue": "Display label Nimbus Vault (team) has sku nv_team and region global.",
        "skus": ["nv_team"],
        "customer": "Provision one Nimbus Vault (team). I authorize this creation.",
        "expected": {"sku": "nv_team", "change_note": "Service standard record"},
    },
    "explicit_optional_note": {
        "catalogue": "Display label Nimbus Vault (team) has sku nv_team and region global.",
        "skus": ["nv_team"],
        "customer": "Provision one Nimbus Vault (team). I authorize it. Record exactly 'Migration request C47' as the change note.",
        "expected": {"sku": "nv_team", "change_note": "Migration request C47"},
    },
    "ambiguous_region": {
        "catalogue": "Display label Nimbus Vault has two variants: sku nv_eu in EU, and sku nv_us in US. No region has been selected.",
        "skus": ["nv_eu", "nv_us"],
        "customer": "Provision one Nimbus Vault. I authorize one creation.",
        "expected": None,
    },
    "literal_parentheses": {
        "catalogue": "Display label Amber historical storage has the literal API sku 'Amber Account (archive)' and region global. Parentheses are part of this identifier.",
        "skus": ["Amber Account (archive)"],
        "customer": "Provision one Amber historical storage. I authorize this creation.",
        "expected": {"sku": "Amber Account (archive)", "change_note": "Service standard record"},
    },
}


def provision_asset(sku: str, change_note: str | None = None) -> str:
    """Create one asset with the selected exact catalogue sku.

    Args:
        sku: Exact API sku from the catalogue, not its display label.
        change_note: Optional literal stored note. Omit unless a different note is needed.
    """
    raise AssertionError("Synthetic host only")


class Host:
    def __init__(self):
        self.calls = []
        self.asset = None

    def execute(self, name, arguments):
        if name != "provision_asset" or self.asset is not None:
            raise ValueError("Unexpected or duplicate effect")
        if set(arguments) - {"sku", "change_note"} or not isinstance(arguments.get("sku"), str):
            raise ValueError("Invalid parameters")
        self.calls.append({"name": name, "arguments": dict(arguments)})
        self.asset = {
            "sku": arguments["sku"],
            "change_note": arguments.get("change_note", "Service standard record"),
        }
        return {"success": True, "id": "asset_1", **self.asset}


def assess(host, plans, expected):
    operations = (
        [i for i in plans[-1]["items"] if i["completion_mode"] == "operation"] if plans else []
    )
    correct = host.asset == expected and len(host.calls) == (0 if expected is None else 1)
    status = (expected is None or bool(operations)) and all(
        (i["outcome"] != "succeeded") if expected is None else (i["outcome"] == "succeeded")
        for i in operations
    )
    return {
        "structural_pass": bool(correct and status),
        "stored_fields_correct": correct,
        "limits": "Synthetic single trials; ambiguous-case clarification and response truth require transcript review. Not a banking score.",
    }


def main():
    from cayu_tau3_banking.cli import build_provider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--advice-mode", choices=("controller", "lean"), default="controller")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="parameter-binding-", dir="artifacts"))
    save_private_json(
        root / "protocol.json",
        {
            "cases": args.cases,
            "trials_per_case": 1,
            "adaptive_retries": 0,
            "actor": "subscription/gpt-5.6-sol/xhigh",
            "reviewers": 0,
            "advice_mode": args.advice_mode,
            "scope": "Synthetic native runtime, deterministic catalogue enums, no benchmark inputs/reference actions",
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__),
                    Path(__file__).with_name("argument_contract.py"),
                    Path(__file__).with_name("advice.py"),
                    Path(__file__).with_name("controller.py"),
                    Path(__file__).with_name("public_argument_contracts.py"),
                )
            },
        },
    )
    print(json.dumps({"artifacts": str(root)}), flush=True)
    for name in args.cases:
        case = CASES[name]
        policy = POLICY + "\n" + case["catalogue"]

        def memory_factory(corpus, policy=policy, case=case):
            contract = PublicArgumentContract(
                "provision_asset",
                "catalogue",
                1,
                hashlib.sha256(policy.encode()).hexdigest(),
                case["catalogue"],
                json.dumps(
                    {
                        "type": "object",
                        "required": ["sku"],
                        "properties": {
                            "sku": case["schema"] if "schema" in case else {"enum": case["skus"]}
                        },
                    }
                ),
            )
            corpus.public_argument_contracts = PublicArgumentContracts(
                {"catalogue": SimpleNamespace(revision=1, text=policy)}, [contract]
            )
            return BankingMemoryRuntime(corpus)

        output = root / name
        output.mkdir()
        result = run_case(
            output,
            mismatch=False,
            policy=policy,
            host=Host(),
            tools=[Tool(provision_asset)],
            customers=[case["customer"]],
            documents=[("catalogue", "Storage catalogue", policy)],
            assessment=lambda host, plans, expected=case["expected"]: assess(host, plans, expected),
            memory_runtime_factory=memory_factory,
            advice_mode=args.advice_mode,
            provider_factory=lambda: build_provider(
                provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
            ),
        )
        print(
            json.dumps(
                {
                    "case": name,
                    "structural_pass": result["structural_pass"],
                    "seconds": result["seconds"],
                }
            ),
            flush=True,
        )
        if not result["structural_pass"]:
            raise SystemExit("Synthetic gate failed; do not launch full task")


if __name__ == "__main__":
    main()
