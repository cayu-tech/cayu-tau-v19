"""One frozen larger-corpus live discovery check through Codex LB."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from cayu_tau3_banking.cli import build_provider
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.navigation_probe import run_case

NAMES = (
    "Aster",
    "Birch",
    "Cedar",
    "Dahlia",
    "Elm",
    "Fern",
    "Grove",
    "Hazel",
    "Iris",
    "Jade",
    "Kelp",
    "Laurel",
)
DOCUMENTS = tuple(
    document
    for i, name in enumerate(NAMES)
    for document in (
        (
            f"doc_savings_accounts_{name.lower()}_001",
            f"{name} savings yield",
            f"{name} Savings pays {3 + i / 10:.1f}% APY. Annual fee and opening eligibility "
            f"are specified separately in {name} account conditions. No rewards or bonuses apply.",
        ),
        (
            f"doc_savings_accounts_{name.lower()}_002",
            f"{name} account conditions",
            f"{name} Savings annual account fee is ${i * 5}. Opening requires a balance of at "
            f"least ${20000 if i == 11 else 100}. No other fee or eligibility condition applies.",
        ),
    )
)
CASE = {
    "id": "catalogue_net_after_fee",
    "customer": "Compare every savings product in this catalogue for a fixed $10000 balance "
    "held for one year. List each product's rate, annual fee, eligibility and net interest "
    "after fees, then choose the highest-net eligible product. Advice only, no application.",
    "expected": "All 12 products covered. Aster through Kelp eligible, net 300+5*i dollars "
    "for indices 0..10. Laurel ineligible due to $20000 minimum. Kelp wins at $350 "
    "(4.0% APY less $50 fee). No invented terms or operations.",
}
POLICY = """Fictional advice-only catalogue. All 12 public savings products are in the
document directory. Public research requires no identity verification. Read yield and
conditions before choosing. Calculate annual interest as fixed balance times APY, with
no compounding, subtract annual fee. Ignore taxes. No applications or external actions.
Do not treat missing fees or eligibility terms as zero or satisfied."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, required=True)
    args = parser.parse_args()
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="discovery-inventory-", dir=args.output_parent))
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
            "retrieval": "lexical, 24 synthetic documents",
            "scope": "Fresh synthetic development qualification, not independent held-out or tau score",
            "source_hashes": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((root / "src").rglob("*.py"))
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
