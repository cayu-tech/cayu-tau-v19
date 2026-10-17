"""Source-complete product navigation on fictional, split-document catalogues."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.navigation_probe import run_case
from cayu_tau3_banking.provenance import CAYU_COMMIT
from cayu_tau3_banking.runtime_profile import profile_session

POLICY = """Fictional organization savings comparison. Advice only; no accounts may
be created. There are exactly two products: Nimbus and Nimbus Plus. They are
distinct products with independent qualifications. Public terms are in the supplied
corpus and need no banking identification. Accept the customer's stated organization
status without additional proof. Compare annual interest as balance times APY, with
no intra-year flows or extra costs. Missing qualifications must not be assumed met.
"""


def documents():
    result = []
    for name, slug, rate in (("Nimbus", "nimbus", "9"), ("Nimbus Plus", "nimbus_plus", "3")):
        for index, title, content in (
            (
                1,
                "benefits",
                f"{name} APY is {rate}%. Annual fee is $0. This is a benefits overview.",
            ),
            (
                2,
                "balance",
                f"{name} requires at least $100 to open and maintain. No balance-tier bonuses apply.",
            ),
            (
                3,
                "statements",
                f"{name} includes free electronic statements. Paper statements are not offered.",
            ),
            (4, "access", f"{name} includes a web dashboard. There is no mobile application."),
            (5, "withdrawals", f"{name} has no withdrawal fee and no withdrawal-count limit."),
            (6, "support", f"{name} provides online support during business hours."),
            (
                7,
                "qualification",
                f"{name} is available only to currently accredited nonprofit organizations."
                if name == "Nimbus"
                else f"{name} is available to all organizations, commercial or nonprofit.",
            ),
        ):
            result.append((f"doc_savings_accounts_{slug}_{index:03d}", f"{name} {title}", content))
    return tuple(result)


CASES = {
    "commercial": {
        "id": "commercial",
        "customer": "Recommend the eligible account with the highest annual interest on $2500, "
        "Nimbus or Nimbus Plus. We are a commercial studio with no nonprofit accreditation. "
        "This is advice only.",
    },
    "nonprofit": {
        "id": "nonprofit",
        "customer": "Recommend the eligible account with the highest annual interest on $2500, "
        "Nimbus or Nimbus Plus. We hold current nonprofit accreditation. This is advice only.",
    },
    "complete_comparison": {
        "id": "complete_comparison",
        "customer": "Compare Nimbus and Nimbus Plus for our commercial studio, which has no nonprofit "
        "accreditation. Before I decide, check both products' full published terms: interest on $2500 "
        "for a year, fees, minimum balance, qualifications, statements, withdrawals, dashboard access "
        "and support availability. Recommend only a product we qualify for. Advice only.",
    },
}
EXPECTED = {
    "commercial": "Nimbus Plus: 3%, $75; Nimbus ineligible.",
    "nonprofit": "Nimbus: 9%, $225; Nimbus Plus 3%, $75.",
    "complete_comparison": "Nimbus Plus eligible at 3%/$75; Nimbus ineligible at 9%/$225. Both $0 annual "
    "fee, $100 minimum, free electronic statements/no paper, no withdrawal fee/count limit, "
    "web dashboard/no mobile application, online support during business hours.",
}


def main():
    from cayu_tau3_banking.cli import build_provider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="product-packets-", dir="artifacts"))
    source = Path(__file__).resolve().parents[1] / "src/cayu_tau3_banking"
    save_private_json(
        root / "protocol.json",
        {
            "framework_commit": CAYU_COMMIT,
            "cases": [CASES[c] for c in args.cases],
            "expected_evaluator_only": EXPECTED,
            "documents": documents(),
            "policy": POLICY,
            "attempts_per_case": 1,
            "model": "subscription gpt-5.6-sol xhigh",
            "reviewers": False,
            "source_hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")
            },
            "limits": "Native runtime and real public navigation, fictional corpus. No reference answers "
            "in model inputs. Manual answer and tool-scope review required; not a tau score.",
        },
    )
    print(f"Output: {root}", flush=True)
    for case in args.cases:
        output = root / case
        output.mkdir(mode=0o700)
        result = run_case(
            output,
            CASES[case],
            documents=documents(),
            policy=POLICY,
            provider_factory=lambda: build_provider(
                provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
            ),
        )
        save_private_json(output / "profile.json", profile_session(output / "runtime.sqlite3"))
        print(
            json.dumps({"case": case, "reply": result["reply"], "seconds": result["seconds"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
