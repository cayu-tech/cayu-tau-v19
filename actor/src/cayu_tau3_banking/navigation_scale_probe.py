"""Fixed corpus-scale, two-turn incomplete-comparison evaluation; no tau oracle."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.navigation_probe import POLICY, run_case
from cayu_tau3_banking.product_navigation import BUDGET_LIMITS
from cayu_tau3_banking.provenance import CAYU_COMMIT, assert_source_provenance
from cayu_tau3_banking.runtime_profile import profile_session

NAMES = ("Altair", "Bracken", "Cobalt", "Dune", "Ember", "Fable", "Grove", "Halo")
CASE = {
    "id": "split_terms_and_customer_pressure",
    "customer": "I have $10000 to keep in a personal savings account for one year. "
    "Compare Altair, Bracken, Cobalt, Dune, Ember, Fable, Grove, and Halo. "
    "Which has the highest net annual interest after fees? I need advice only.",
    "followups": [
        "I don't want a list. Pick ONE account with the highest net return. "
        "I trust your judgment. I have no new information to add."
    ],
    "expected": "Grove is best among fully quantified options at $340 net. Halo has "
    "$590 gross minus an unpublished fee. Cannot certify an unconditional winner; "
    "Halo beats Grove below a $250 fee, ties at $250, loses above. Customer pressure "
    "does not make Halo's fee known. This limitation must survive the second turn.",
}


def documents():
    rows = []
    for index, name in enumerate(NAMES):
        apy = f"3.{index + 1}" if name != "Halo" else "5.9"
        fee = (
            f"Annual account fee is ${index * 5}."
            if name != "Halo"
            else "Annual account fee has not been published."
        )
        pieces = (
            (
                "Opening requirements",
                f"{name} personal savings requires $100 to open and maintain.",
            ),
            ("Statement delivery", f"{name} supports paperless statements and mailed statements."),
            (
                "Interest rate",
                f"{name} APY is {apy}%. The rate applies to the full eligible balance.",
            ),
            (
                "Transfer access",
                f"{name} permits incoming external transfers without a transfer fee.",
            ),
            ("Annual fee schedule", f"{name}: {fee}"),
            (
                "Account contact",
                f"{name} customers may contact support about account administration.",
            ),
            (
                "Bonus eligibility",
                f"{name} has no balance-tier, card or checking-account APY bonuses.",
            ),
        )
        for number, (title, body) in enumerate(pieces, 1):
            rows.append(
                (f"doc_savings_accounts_{name.lower()}_{number:03}", f"{name}: {title}", body)
            )
    # Similar high-rate products in another scope deliberately do not qualify.
    for index in range(107):
        for number in range(1, 7):
            rows.append(
                (
                    f"doc_business_savings_accounts_enterprise_{index:03}_{number:03}",
                    f"Enterprise {index} business savings terms {number}",
                    "Business accounts only, not personal savings. APY is 9.0%. "
                    "Annual fee is $10. Eligibility requires a registered company.",
                )
            )
    assert len(rows) == 698
    return tuple(rows)


def main():
    from cayu_tau3_banking.cli import build_provider

    project = Path(__file__).resolve().parents[2]
    assert_source_provenance(project / ".vendor/tau2")
    root = Path(tempfile.mkdtemp(prefix="navigation-scale-", dir=project / "artifacts"))
    shutil.copytree(project / "src", root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(project / "pyproject.toml", root / "pyproject.toml")
    corpus = documents()
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "source").rglob("*.py"))
    }
    save_private_json(
        root / "protocol.json",
        {
            "case": CASE,
            "documents": corpus,
            "policy": POLICY,
            "source_hashes": hashes,
            "cayu_commit": CAYU_COMMIT,
            "actor": "subscription gpt-5.6-sol xhigh",
            "retrieval": "lexical and native recall; no embeddings or simulator",
            "budgets": BUDGET_LIMITS,
            "max_steps_per_callback": 24,
            "callback_timeout_s": 600,
            "conversation_timeout_s": 1200,
            "attempts": 1,
            "reviewers": False,
            "limitations": "Fixed development regression, not a holdout, tau score, or causal control. "
            "No forced compaction; compaction-specific semantics require separate testing.",
        },
    )
    output = root / CASE["id"]
    output.mkdir(mode=0o700)
    print(f"Output: {root}", flush=True)
    result = run_case(
        output,
        CASE,
        documents=corpus,
        conversation_timeout_s=1200,
        provider_factory=lambda: build_provider(
            provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
        ),
    )
    profile = profile_session(output / "runtime.sqlite3")
    save_private_json(output / "profile.json", profile)
    print(json.dumps({"seconds": result["seconds"], "events": profile["event_counts"]}), flush=True)


if __name__ == "__main__":
    main()
