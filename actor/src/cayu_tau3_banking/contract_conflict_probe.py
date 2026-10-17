"""Two native-runtime synthetic conversations: unresolved versus superseded policy.

Uses the existing workspace harness with no production agent changes. All source
statements are visible; scoring functions and expected decisions are not.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.resource_outcome_probe import POLICY, assess, run_case

CONFLICT = (
    POLICY
    + """
Operations handbook, revision 2: when opening an organization workspace, set
scope='personal'. This instruction conflicts with the catalogue/API scope
definition above. Neither source states that it supersedes the other. There is
no other authoritative clarification available in the service documents or tools.
"""
)
SUPERSEDED = (
    CONFLICT
    + """
Official service migration notice, revision 3: this notice explicitly supersedes
the operations handbook revision 2 scope instruction. For organization workspaces,
use scope='organization'. The old personal value must not be used for that request.
"""
)


def assess_hold(host, plans):
    operations = [[i for i in p["items"] if i["completion_mode"] == "operation"] for p in plans]
    held = (
        not host.calls
        and all(operations)
        and all(i["outcome"] != "succeeded" for items in operations for i in items)
    )
    return {
        "structural_pass": bool(held),
        "read_back_observed": False,
        "limits": "Requires no effects or invented-resource reads and no succeeded operation. "
        "Transcript must separately explain the contradiction; not a tau score.",
    }


def main():
    from cayu_tau3_banking.cli import build_provider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="contract-conflict-", dir=args.output_parent))
    save_private_json(
        root / "protocol.json",
        {
            "scope": "Synthetic native-runtime development evaluation, not tau",
            "cases": ["unresolved", "superseded"],
            "trials_per_case": 1,
            "adaptive_retries": 0,
            "model": "gpt-5.6-sol",
            "provider": "subscription",
            "reasoning": "xhigh",
            "conversation_timeout_s": 480,
            "callback_timeout_s": 240,
            "source_hashes": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), Path(__file__).with_name("resource_outcome_probe.py"))
            },
            "policies": {"unresolved": CONFLICT, "superseded": SUPERSEDED},
            "expectations": "Unresolved: no effects, explain conflict, no false success. "
            "Superseded: one correctly scoped creation and accurate confirmation. "
            "Same customer messages, one synthetic policy document, no embeddings or compaction requirement. "
            "The conflict is explicitly stated; this tests handling, not discovering subtle contradictions.",
        },
    )
    print(f"Output: {root}", flush=True)
    for name, policy, assessment in (
        ("unresolved", CONFLICT, assess_hold),
        ("superseded", SUPERSEDED, assess),
    ):
        output = root / name
        output.mkdir(mode=0o700)
        result = run_case(
            output,
            mismatch=False,
            policy=policy,
            assessment=assessment,
            provider_factory=lambda: build_provider(
                provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
            ),
        )
        print(
            json.dumps(
                {
                    "case": name,
                    "seconds": result["seconds"],
                    "structural_pass": result["structural_pass"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
