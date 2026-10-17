"""Frozen synthetic discovery check; LB actor, no simulator or benchmark score."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

# Prevent the benchmark dependency from fetching a public price map on import.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from cayu_tau3_banking.cli import build_provider
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.navigation_probe import run_case

DOCUMENTS = (
    (
        "doc_savings_accounts_juniper_001",
        "Juniper account yield",
        "Juniper Savings pays 4.2% APY. Its annual fee is $0. Minimum balance is $100. "
        "The temporary opening reward is governed separately by the Juniper reward conditions.",
    ),
    (
        "doc_savings_accounts_juniper_002",
        "Juniper reward conditions",
        "# Opening reward\nThe Juniper opening reward is $90. It requires a monthly employer "
        "payroll deposit of at least $800 in each of the first three months. Personal transfers "
        "from another bank do not qualify. A customer without qualifying payroll earns no reward.",
    ),
    (
        "doc_savings_accounts_cobalt_001",
        "Cobalt savings terms",
        "Cobalt Savings pays 4.5% APY. Annual fee is $12. Minimum balance is $100. "
        "No opening reward or payroll requirement applies.",
    ),
    (
        "doc_savings_accounts_juniper_plus_001",
        "Juniper Plus account",
        "Juniper Plus is a distinct account requiring a $25000 balance. It pays 5.1% APY "
        "with no annual fee. Juniper Savings reward conditions do not apply to Juniper Plus.",
    ),
)
CASE = {
    "id": "payroll_changes_reward_eligibility",
    "customer": "Compare Juniper Savings and Cobalt for $6000 kept for one year. I will "
    "transfer $900 every month from my own other-bank savings, not from an employer. "
    "Treat the balance as fixed for this comparison. Include any opening reward I qualify "
    "for. Which has the higher first-year net? Advice only, no application.",
    "followups": [
        "Correction: those monthly $900 deposits will be employer payroll from "
        "the first month. Keep the other assumptions unchanged. Does your answer change?"
    ],
    "expected": "First: Juniper 252, no reward; Cobalt 270-12=258, Cobalt higher. "
    "After correction: Juniper 252+90=342, higher than Cobalt 258. Payroll must satisfy "
    "each of the first three months. No customer action or application.",
}
POLICY = """Fictional advice-only service. Public terms need no identity verification.
Use balance times APY without compounding, subtract annual fees and add only qualifying
opening rewards. Ignore taxes and changes in balance. All product terms are in the public
corpus. Follow linked conditions. Missing terms are unknown, not zero. Never execute an
application. Reconsider earlier conclusions when the customer changes their facts."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, required=True)
    args = parser.parse_args()
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="discovery-actor-", dir=args.output_parent))
    root = Path(__file__).resolve().parents[1]
    save_private_json(
        output / "protocol.json",
        {
            "case": CASE,
            "documents": DOCUMENTS,
            "policy": POLICY,
            "model": "gpt-5.6-sol",
            "provider": "codex-lb",
            "effort": "xhigh",
            "attempts": 1,
            "deadline_seconds": 600,
            "retrieval": "lexical synthetic corpus",
            "scope": "Fresh synthetic development check, not blind held-out evaluation or tau score",
            "source_hashes": {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((root / "src").rglob("*.py"))
            },
        },
    )
    print(f"Output: {output}", flush=True)
    result = run_case(
        output,
        CASE,
        documents=DOCUMENTS,
        policy=POLICY,
        advice_mode="lean",
        controller_mode="advisory",
        provider_factory=lambda: build_provider(
            provider="codex-lb",
            codex_lb_base_url="https://codex-lb.cayu.ai",
            semantic_progress_timeout_seconds=120,
        ),
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
