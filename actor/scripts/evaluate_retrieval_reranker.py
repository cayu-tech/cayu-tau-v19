"""One response per query over frozen candidates; default mode is offline preflight."""

import argparse
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

from cayu.providers.openai import build_openai_payload

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.embedding_diagnostics import embedding_failure_details
from cayu_tau3_banking.knowledge import _load_corpus_material
from cayu_tau3_banking.retrieval_reranker import (
    ID_PROMPT,
    RerankerFailure,
    build_id_request,
    select,
)

SOURCE = Path("artifacts/multiscale-selection-validation/run-aca4q3p4")
EMBEDDING_SOURCE = Path("artifacts/section-embedding-validation/run-eo81fta9")
DOCUMENTS = Path(".vendor/tau2/data/tau2/domains/banking_knowledge/documents")


async def run(root, live, resume_events=None):
    source = json.loads((SOURCE / "summary.json").read_text())
    original = json.loads((EMBEDDING_SOURCE / "manifest.json").read_text())
    material = _load_corpus_material(DOCUMENTS)
    if material.corpus_sha256 != original["corpus_sha256"]:
        raise ValueError("Public corpus no longer matches captured candidates.")
    entries = {e.id: e for e in material.entries}
    cases = {(c["suite"], c["id"]): c for c in original["cases"]}
    prepared = []
    for row in source["rows"]:
        case = cases[row["suite"], row["id"]]
        selected_ids = row["profiles"]["max_semantic"]["ids"][:16]
        # Stable query-conditioned permutation removes source-rank ordering from input.
        ordered = sorted(
            selected_ids,
            key=lambda key: hashlib.sha256((case["query"] + "\0" + key).encode()).hexdigest(),
        )
        mapping = {f"D{i:02d}": entries[key] for i, key in enumerate(ordered, 1)}
        documents = [
            {"candidate_id": key, "title": entry.title, "body": entry.text}
            for key, entry in mapping.items()
        ]
        request = build_id_request(case["query"], documents)
        build_openai_payload(request, stream=True, reasoning_state="inline")
        stem = f"{row['suite']}--{row['id']}"
        save_private_json(root / f"{stem}.request.json", request.model_dump(mode="json"))
        save_private_json(
            root / f"{stem}.mapping.json",
            {
                key: {
                    "entry_id": entry.id,
                    "revision": entry.revision,
                    "body_sha256": hashlib.sha256(entry.text.encode()).hexdigest(),
                }
                for key, entry in mapping.items()
            },
        )
        prepared.append((row, case, stem, documents, mapping, request))
    save_private_json(
        root / "manifest.json",
        {
            "model": "gpt-5.6-sol",
            "provider": "subscription",
            "reasoning_effort": "low",
            "max_calls": len(prepared),
            "max_candidates": 16,
            "max_selected": 4,
            "timeout_seconds_per_call": 120,
            "concurrency": 1,
            "repair_retries": 0,
            "prompt": ID_PROMPT,
            "output_contract": "candidate IDs only; authoritative host body hashes; no generated quotes",
            "corpus_sha256": material.corpus_sha256,
            "source": str(SOURCE),
            "candidate_order": "SHA256(query + NUL + entry ID)",
            "cases": [{"id": row["id"], "suite": row["suite"]} for row, *_ in prepared],
            "gate": "No per-query recall4 regression versus strongest-semantic selector or original passing retrieval, recover inherited APR miss. No automatic promotion without independent validation.",
            "limitations": "Previously inspected development queries. One response per query; no task pass or production latency claim. Exact quotes establish provenance, not semantic correctness.",
        },
    )
    print(json.dumps({"prepared": len(prepared), "live": live, "directory": str(root)}), flush=True)
    if not live:
        return
    from cayu_tau3_banking.cli import build_provider

    provider = build_provider(
        provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=60
    )
    rows = []
    active_case = None
    try:
        for row, case, stem, documents, mapping, request in prepared:
            active_case = row["id"]
            event_index = 0
            observed_completion = None

            def capture(event, case_stem=stem):
                nonlocal event_index, observed_completion
                save_private_json(root / f"{case_stem}.event-{event_index:03d}.json", event)
                event_index += 1
                if event["type"] == "completed":
                    observed_completion = event["payload"]

            cached_events = (
                [] if resume_events is None else sorted(resume_events.glob(f"{stem}.event-*.json"))
            )
            if cached_events:
                from cayu import ModelStreamEvent, ScriptedModelProvider

                prior_request = json.loads((resume_events / f"{stem}.request.json").read_text())
                if prior_request != request.model_dump(mode="json"):
                    raise ValueError("Cannot reuse a response for a changed request.")
                events = [json.loads(path.read_text()) for path in cached_events]
                replay = ScriptedModelProvider(
                    [[ModelStreamEvent(type=e["type"], payload=e["payload"]) for e in events]]
                )
                case_provider = replay
            else:
                case_provider = provider
            try:
                result = await select(case_provider, request, documents, capture=capture)
            except RerankerFailure as error:
                if error.diagnostic["stage"] not in {"response_shape", "evidence_validation"}:
                    raise
                result = {
                    "selections": [],
                    "elapsed_seconds": error.diagnostic["elapsed_seconds"],
                    "completion": observed_completion,
                    "validation_failure": error.diagnostic,
                }
            if cached_events:
                result["reused_response_from"] = str(resume_events)
                result["elapsed_seconds"] = None
            ids = [mapping[item["candidate_id"]].id for item in result["selections"]]
            required = set(case["required"])
            scored = {
                "id": row["id"],
                "suite": row["suite"],
                "selected_ids": ids,
                "recall4": len(required.intersection(ids)) / len(required),
                "selector_recall4": row["profiles"]["max_semantic"]["recall4"],
                "original_recall4": row["original_recall4"],
                **result,
            }
            save_private_json(root / f"{stem}.result.json", scored)
            rows.append(scored)
            print(
                json.dumps(
                    {
                        "case": row["id"],
                        "recall4": scored["recall4"],
                        "seconds": result["elapsed_seconds"],
                        "validation_failed": "validation_failure" in result,
                    }
                ),
                flush=True,
            )
        regressions = [
            row["id"]
            for row in rows
            if row["recall4"] < max(row["selector_recall4"], row["original_recall4"])
        ]
        apr = next(row for row in rows if row["id"] == "personal_apr_inventory")
        summary = {
            "rows": rows,
            "mean_recall4": sum(r["recall4"] for r in rows) / len(rows),
            "regressions": regressions,
            "gate_passed": not regressions
            and apr["recall4"] == 1.0
            and not any("validation_failure" in row for row in rows),
            "default_changed": False,
        }
        save_private_json(root / "summary.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    except Exception as error:
        diagnostic = {
            "case": active_case,
            "completed_cases": len(rows),
            "error": embedding_failure_details(error),
        }
        diagnostic["error"]["stage"] = "reranker"
        if hasattr(error, "diagnostic"):
            diagnostic["reranker_detail"] = error.diagnostic
        save_private_json(root / "failure.json", diagnostic)
        print(json.dumps({"stopped": diagnostic}), flush=True)
        raise SystemExit(1) from None
    finally:
        await provider.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--resume-events", type=Path)
    args = parser.parse_args()
    parent = Path("artifacts/retrieval-reranker-validation")
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    asyncio.run(run(root, args.live, args.resume_events))
