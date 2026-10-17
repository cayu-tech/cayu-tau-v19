"""Offline feasibility audit, not a task rescore or a semantic correctness test."""

import argparse
import json
import os
import re
import sqlite3
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from cayu_tau3_banking.product_navigation import _unique_source_paragraph


def audit(documents, reports, archives=None):
    texts = {}
    for path in documents.glob("*.json"):
        document = json.loads(path.read_text())
        texts[document["id"]] = document["content"]
    counts = Counter()
    unique = set()
    for path in reports:
        report = json.loads(path.read_text())
        calls = report.get("research_calls", [])
        if archives is not None:
            calls = []
            with (
                zipfile.ZipFile(archives / path.stem / "diagnostics.zip") as archive,
                tempfile.TemporaryDirectory(prefix="tau-quote-audit-") as temporary,
            ):
                name = next(n for n in archive.namelist() if n.endswith(".sqlite3"))
                database = Path(archive.extract(name, temporary))
                with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
                    for event, raw in db.execute(
                        "SELECT event_type,payload_json FROM cayu_events WHERE "
                        "tool_name='cayu_check_comparison_facts' AND "
                        "event_type IN ('tool.call.completed','tool.call.failed')"
                    ):
                        payload = json.loads(raw)
                        counts[event] += 1
                        calls.append(
                            {
                                "tool": "cayu_check_comparison_facts",
                                "arguments": payload["arguments"],
                            }
                        )
        for call in calls:
            if call["tool"] != "cayu_check_comparison_facts":
                continue
            for fact in call["arguments"].get("facts", []):
                identity = (
                    fact.get("entry_id"),
                    fact.get("revision"),
                    fact.get("value"),
                    fact.get("quote"),
                )
                if identity in unique:
                    continue
                unique.add(identity)
                entry, revision, value, quote = identity
                counts["unique_fact_proposals"] += 1
                if (
                    entry not in texts
                    or revision != 1
                    or not isinstance(value, str)
                    or not isinstance(quote, str)
                ):
                    counts["invalid_or_missing_source_shape"] += 1
                    continue
                text = texts[entry]
                literal = re.search(
                    r"(?<![\w.])(?<![0-9],)" + re.escape(value) + r"(?!\w|[.,][0-9])", quote
                )
                quote_valid = quote in text and literal is not None
                counts["existing_quote_valid" if quote_valid else "existing_quote_invalid"] += 1
                try:
                    _unique_source_paragraph(text, value)
                except ValueError:
                    counts["explicit_quote_still_required"] += 1
                else:
                    counts["automatic_quote_possible"] += 1
                    if not quote_valid:
                        counts["invalid_quote_recoverable_with_unchanged_source_and_value"] += 1
    return {
        "purpose": "Mechanical source-selection feasibility on historical proposals, not predicted pass rate",
        "reports": [str(path) for path in reports],
        "counts": dict(counts),
        "includes_rejected_call_arguments": archives is not None,
        "limitations": [
            "No model rerun; does not establish fewer future errors or faster tasks",
            "Does not validate product family, delivery, scope, applicability, or semantic interpretation",
            "No expected evaluator actions or answers supplied to the actor",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archives", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    result = audit(args.documents, args.reports, args.archives)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result["counts"]))
