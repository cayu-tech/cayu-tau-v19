"""Paired synthetic test of incremental fact retention, not a benchmark run."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from cayu import ModelStreamEvent, ScriptedModelProvider
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor

from cayu_tau3_banking import agent
from cayu_tau3_banking.comparison_retention_probe import DOCUMENTS
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.navigation_probe import CASES, POLICY, run_case
from cayu_tau3_banking.navigation_probe import DOCUMENTS as NAVIGATION_DOCUMENTS
from cayu_tau3_banking.provenance import CAYU_COMMIT

RETENTION_GUIDANCE = """
For a comparison that spans several read batches or customer turns, retain a compact
source-backed comparison incrementally with cayu_check_comparison_facts while the
source passages are available. Keep just the decisive rates, costs, eligibility and
conditions, with exact quotes and revision references; partial worksheets may leave
gaps explicit. Do not wait for exhaustive research before retaining useful facts.
On a changed budget or preference, reuse still-applicable checked facts and recompute;
read again only for a specific missing condition, changed revision or source conflict.
Use a short candidate/decisive-gap plan to direct the next batch. Once the supported
comparison resolves the request, give the result instead of restarting discovery.
Unknown terms stay unknown; retained numeric facts do not prove their applicability
or catalogue-wide coverage. Simple one-step answers still need no worksheet.
""".strip()


def catalogue_case():
    documents = []
    for name, rate, fee in (
        ("Alder", "4", "0"),
        ("Birch", "5", "30"),
        ("Cedar", "3.2", "0"),
        ("Dune", "4.5", "14"),
    ):
        prefix = f"doc_savings_accounts_{name.lower()}"
        documents.extend(
            [
                (
                    prefix + "_001",
                    f"{name} savings rate",
                    f"{name} savings APY is {rate}%. No bonuses apply.",
                ),
                (
                    prefix + "_002",
                    f"{name} savings fees and requirements",
                    f"{name} savings annual fee is ${fee}. Minimum balance is $100. No fee waiver or other costs apply.",
                ),
            ]
        )
    for name, fee in (("Fern", "20"), ("Iris", "40"), ("Lark", "30"), ("Moss", "25")):
        prefix = f"doc_credit_cards_{name.lower()}"
        documents.extend(
            [
                (
                    prefix + "_001",
                    f"{name} card fee",
                    f"{name} card annual fee is ${fee}. No fee waiver applies.",
                ),
                (
                    prefix + "_002",
                    f"{name} card benefits",
                    f"{name} card rewards are zero. It offers no companion bonus. No other costs apply.",
                ),
            ]
        )
    return (
        tuple(documents),
        {
            "id": "catalogue_changing_balance",
            "customer": "Compare all four savings products Alder, Birch, Cedar and Dune, and all four "
            "cards Fern, Iris, Lark and Moss. I want one savings product and one card, advice only. "
            "I will keep $2000 in savings for one year and meet all requirements. Choose the pair "
            "with the greatest annual net interest after both fees. I will ask you to recalculate "
            "for another balance afterward using these same terms.",
            "followups": [
                "Now use $4000 instead with the same terms. Which pair wins and what is the annual net?"
            ],
        },
        ["Alder and Fern: $60", "Birch and Fern: $150"],
    )


def main():
    from cayu_tau3_banking.cli import build_provider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "retention"), required=True)
    parser.add_argument("--case", choices=("small", "catalogue", "unknown"), default="small")
    args = parser.parse_args()
    if args.arm == "retention":
        agent._MEMORY_INSTRUCTIONS += "\n\n" + RETENTION_GUIDANCE
    root = Path(tempfile.mkdtemp(prefix=f"comparison-workflow-{args.arm}-", dir="artifacts"))
    case = {
        "id": "changing_balance",
        "customer": "Compare Alder versus Birch savings and Fern versus Iris cards. "
        "I want one savings account and one card, advice only. I will leave $2000 in savings "
        "for a year, meet the stated minimums, and want the greatest annual net interest "
        "after both account and card fees. Which pair and net amount?",
        "followups": [
            "Recalculate with $3000 instead, using the same product terms. Which pair and net amount now?"
        ],
    }
    documents = DOCUMENTS
    expected = ["Alder and Fern: $60", "Alder and Fern: $100"]
    if args.case == "catalogue":
        documents, case, expected = catalogue_case()
    elif args.case == "unknown":
        selected = CASES[2]
        case = {key: value for key, value in selected.items() if key != "expected"}
        documents = NAVIGATION_DOCUMENTS
        expected = [selected["expected"]]
    source = Path(__file__).resolve().parent
    save_private_json(
        root / "protocol.json",
        {
            "arm": args.arm,
            "framework_commit": CAYU_COMMIT,
            "case": case,
            "documents": documents,
            "policy": POLICY,
            "memory_instructions": agent._MEMORY_INSTRUCTIONS,
            "retention_instructions": RETENTION_GUIDANCE if args.arm == "retention" else "",
            "system_prompt": agent.banking_system_prompt(POLICY, planning_mode="outcomes"),
            "source_hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")
            },
            "model": "subscription gpt-5.6-sol xhigh",
            "attempts": 1,
            "expected_evaluator_only": expected,
            "limits": "Single paired development case with artificial summary loss. Assess correctness, rereads, "
            "model calls and retained facts; not a banking score or reliable latency estimate.",
        },
    )
    summarizer = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta(
                    "Earlier comparison details omitted; use current structured state."
                ),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
            for _ in range(64)
        ]
    )
    output = root / "case"
    output.mkdir(mode=0o700)
    print(root, flush=True)
    result = run_case(
        output,
        case,
        documents=documents,
        policy=POLICY,
        conversation_timeout_s=600,
        compaction=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=summarizer, model="scripted-loss"),
            compact_after_messages=2,
            max_user_turns=1,
        ),
        provider_factory=lambda: build_provider(
            provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
        ),
    )
    print(
        json.dumps(
            {
                "arm": args.arm,
                "seconds": result["seconds"],
                "reply": result["reply"],
                "compactions": len(summarizer.requests),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
