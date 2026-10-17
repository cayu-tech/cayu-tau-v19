"""Frozen multi-category advice probe: forced summary loss versus no compaction.

No banking task inputs, reference actions, semantic reviewers or external effects.
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from cayu import ModelStreamEvent, ScriptedModelProvider
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.navigation_probe import POLICY, run_case

DOCUMENTS = (
    (
        "doc_savings_accounts_alder_001",
        "Alder savings",
        "Alder APY is 4%. Annual fee is $0. Minimum balance is $100. No other conditions or benefits apply.",
    ),
    (
        "doc_savings_accounts_birch_001",
        "Birch savings",
        "Birch APY is 3%. Annual fee is $0. Minimum balance is $100. No other conditions or benefits apply.",
    ),
    (
        "doc_credit_cards_fern_001",
        "Fern card",
        "Fern annual fee is $20. Card rewards are zero. No companion bonus or other costs apply.",
    ),
    (
        "doc_credit_cards_iris_001",
        "Iris card",
        "Iris annual fee is $40. Card rewards are zero. No companion bonus or other costs apply.",
    ),
)


def main():
    from cayu_tau3_banking.cli import build_provider

    repo = Path(__file__).resolve().parents[2]
    root = Path(tempfile.mkdtemp(prefix="comparison-retention-", dir=repo / "artifacts"))
    shutil.copytree(repo / "src", root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    case = {
        "id": "mixed_products",
        "customer": "Compare Alder versus Birch savings and Fern versus Iris cards. I want one savings account and one card, advice only. I will leave $2000 in savings for a year, meet the stated minimums, and want the greatest annual net interest after both account and card fees. Which pair and net amount?",
        "followups": [
            "Recalculate with $3000 instead, using the same product terms. Which pair and net amount now?"
        ],
    }
    save_private_json(
        root / "protocol.json",
        {
            "case": case,
            "documents": DOCUMENTS,
            "policy": POLICY,
            "arms": ["no_compaction", "forced_loss"],
            "attempts_per_arm": 1,
            "expected": "Alder plus Fern: $60 initially, $100 at $3000. No effects. Audit rereads, checked worksheet reuse, and exact contexts after forced summary loss.",
            "limits": "Both arms use treatment source. Tests compaction robustness, not causal baseline-versus-treatment speed improvement. One trial per arm; no tau score.",
            "model": "subscription gpt-5.6-sol xhigh",
            "conversation_timeout_s": 900,
            "source_hashes": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((root / "source").rglob("*.py"))
            },
        },
    )
    print(root, flush=True)
    for arm in ("no_compaction", "forced_loss"):
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta(
                        "Earlier conversation and comparison details omitted. Consult current structured state."
                    ),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
                for _ in range(64)
            ]
        )
        compaction = CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=provider, model="scripted-loss"),
            compact_after_messages=2 if arm == "forced_loss" else 10000,
            max_user_turns=1 if arm == "forced_loss" else 10000,
        )
        output = root / arm
        output.mkdir(mode=0o700)
        result = run_case(
            output,
            case,
            documents=DOCUMENTS,
            policy=POLICY,
            compaction=compaction,
            conversation_timeout_s=900,
            provider_factory=lambda: build_provider(
                provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
            ),
        )
        print(
            json.dumps(
                {
                    "arm": arm,
                    "seconds": result["seconds"],
                    "compactions": len(provider.requests),
                    "reply": result["reply"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
