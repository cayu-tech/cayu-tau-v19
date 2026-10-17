"""LB-only summary controls; outputs require inspection, not a benchmark grade."""

import argparse
import asyncio
import json
from pathlib import Path

from cayu import Message
from cayu.providers.base import ModelRequest

from cayu_tau3_banking.cli import build_provider
from cayu_tau3_banking.compaction import SUMMARY_INSTRUCTIONS
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.summary_probe import response

CASES = {
    "attribution": (
        "Public procedure revision 7: waive an incorrectly charged service fee after identity "
        "verification and checking the account ledger. Verification succeeded (result v1). "
        "Ledger result l1: service fee 8; description branch service desk. Agent: the branch "
        "description establishes the service location; this fee should be waived. Customer: "
        "I authorize the waiver. No waiver call has yet occurred."
    ),
    "status_date": (
        "Environment date: 2030-04-08. Ledger result l2: status=posted, date=2030-04-09, "
        "amount=12. Agent: the dates appear inconsistent; I have not resolved the conflict. "
        "Public procedure revision 3 says review posted entries. No instruction says future "
        "dates change the status. The customer has not authorized an adjustment."
    ),
    "retain_requirement": (
        "Public procedure revision 9: independent merchant confirmation is required before "
        "this reversal. Ledger result l3 confirms the purchase but not merchant confirmation. "
        "Customer: reverse it now. Agent: merchant confirmation remains missing; the "
        "customer's permission does not establish that condition. No action was executed."
    ),
    "retain_uncertainty": (
        "Ledger result l4: account active, age 21 days. Public procedure revision 4: age "
        "must be at least 30 days. Customer reports 40 days. Agent: active status is not "
        "proof of age; these ages conflict and eligibility is unresolved. The customer "
        "withdrew permission to close the account. Proposed next step: explain the conflict."
    ),
}


async def run(output: Path, historical_input: Path | None = None):
    output.mkdir(parents=True, exist_ok=False)
    cases = dict(CASES)
    if historical_input is not None:
        # Only the original pre-decision transcript, never reference outcomes.
        cases["historical_first_compaction"] = historical_input.read_text()
    provider = build_provider(
        provider="codex-lb",
        codex_lb_base_url="https://codex-lb.cayu.ai",
        semantic_progress_timeout_seconds=120,
    )
    save_private_json(
        output / "protocol.json",
        {
            "provider": "codex-lb",
            "model": "gpt-5.6-sol",
            "effort": "low",
            "cases": list(cases),
            "tools_execute": False,
            "automatic_retries": 0,
            "assessment": "Manual source-to-summary fidelity inspection; no automatic semantic grade",
        },
    )
    try:
        for name, transcript in cases.items():
            request = ModelRequest(
                model="gpt-5.6-sol",
                messages=[
                    Message.text("system", SUMMARY_INSTRUCTIONS),
                    Message.text("user", transcript),
                ],
                tools=[],
                options={"thinking": {"enabled": True, "effort": "low"}},
            )
            save_private_json(output / f"{name}-request.json", request.model_dump(mode="json"))
            try:
                summary = await response(provider, request, timeout=180)
            except Exception as error:
                save_private_json(
                    output / f"{name}-error.json", {"error_type": type(error).__name__}
                )
                raise
            save_private_json(output / f"{name}-result.json", {"summary": summary})
            print(json.dumps({"case": name, "completed": True}), flush=True)
    finally:
        await provider.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-input", type=Path)
    args = parser.parse_args()
    asyncio.run(run(args.output, args.historical_input))


if __name__ == "__main__":
    main()
