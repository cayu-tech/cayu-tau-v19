"""Offline recorded-input probe; not a benchmark score or model evaluation."""

import argparse
import asyncio
import json
from pathlib import Path

from tau2.data_model.message import UserMessage

from cayu_tau3_banking.customer_sources import customer_source_context
from cayu_tau3_banking.incremental_comparison import check_incremental
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.ledger import ConversationLedger
from cayu_tau3_banking.product_navigation import COMPARISON_TOOL, ProductNavigation


async def main(args):
    results = {}
    for task in ["058", "059"]:
        nav = ProductNavigation(BankingKnowledgeCorpus.from_directory(args.corpus))
        seen = set()
        outcomes = []
        for event in json.loads((args.audit / f"current-{task}" / "events.json").read_text()):
            if (
                event["kind"] == "tool.call.completed"
                and event["tool"] == "cayu_read_product_documents"
            ):
                seen.update(d["entry_id"] for d in event["data"]["arguments"].get("documents", []))
            if (
                event["kind"] not in ["tool.call.completed", "tool.call.failed"]
                or event["tool"] != COMPARISON_TOOL
            ):
                continue
            inputs = event["data"]["arguments"]
            errors = list(nav.validators[COMPARISON_TOOL].iter_errors(inputs))
            if errors:
                outcomes.append({"sequence": event["sequence"], "malformed_request_rejected": True})
                continue
            result = await check_incremental(
                nav, inputs, lambda entry, revision, seen=seen: entry in seen and revision == "1"
            )
            outcomes.append(
                {
                    "sequence": event["sequence"],
                    "retained": result.get("retained_for_compaction"),
                    "source_checked_facts": len(result.get("facts", [])),
                    "cell_errors": len(result.get("cell_errors", [])),
                    "coverage_complete": result["declared_coverage_complete"],
                    "authorizes_completion": result["authorizes_completion"],
                }
            )
        results[task] = outcomes
    official = json.loads((args.audit / "current-041" / "official.json").read_text())
    ledger = ConversationLedger(interpret_customer_intent=False)
    for msg in official["simulations"][0]["messages"]:
        if msg["role"] == "user":
            ledger.observe_input(UserMessage.text(msg["content"]))
    projected = customer_source_context(ledger)
    results["041"] = {
        "partial_amount_retained": "69.99" in projected,
        "messages_retained": len(ledger.customer_message_ids),
        "projection_bytes": len(projected.encode()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
