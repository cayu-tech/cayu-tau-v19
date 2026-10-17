"""Counterbalanced full-body versus best-section prompts on fixed candidates."""

import asyncio
import hashlib
import json
import statistics
import tempfile
from pathlib import Path

from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.retrieval_reranker import RerankerFailure, build_id_request, select

BASE = Path("artifacts/retrieval-reranker-validation/run-sq0gjqts")
CAPTURE = Path("artifacts/section-embedding-validation/run-eo81fta9")


def prepare():
    """Representations selected using saved semantic scores, never required IDs."""
    cases = json.loads((CAPTURE / "manifest.json").read_text())["cases"]
    prepared = []
    for case in cases:
        stem = case["suite"] + "--" + case["id"]
        request = json.loads((BASE / (stem + ".request.json")).read_text())
        original = json.loads(request["messages"][1]["content"][0]["text"])
        mapping = json.loads((BASE / (stem + ".mapping.json")).read_text())
        recorded = json.loads((CAPTURE / (stem + ".json")).read_text())
        sections = {r["entry_id"]: r["text"] for r in recorded["rankings"]["sections"]}
        views = {
            "full": original["documents"],
            "section": [
                {**d, "body": sections[mapping[d["candidate_id"]]["entry_id"]]}
                for d in original["documents"]
            ],
        }
        prepared.append((case, stem, mapping, views))
    return prepared


async def main():
    from cayu_tau3_banking.cli import build_provider

    parent = Path("artifacts/reranker-speed-validation")
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    cases = prepare()
    save_private_json(
        root / "manifest.json",
        {
            "model": "gpt-5.6-sol",
            "reasoning_effort": "low",
            "provider": "subscription",
            "candidate_count": 16,
            "max_selected": 4,
            "cases": len(cases),
            "max_calls": 46,
            "order": "alternate full/section and section/full by case index",
            "changed": "body representation only; same candidates/order/query/model/prompt",
            "scope": "Reranking latency only, not end-to-end search; existing development queries",
            "gate": "No recall regression on any case; target rerank p95 <= 1 second. No automatic promotion.",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    print(root, flush=True)
    provider = build_provider(
        provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=60
    )
    rows = []
    try:
        for index, (case, stem, mapping, views) in enumerate(cases):
            for arm in ("full", "section") if index % 2 == 0 else ("section", "full"):
                documents = views[arm]
                request = build_id_request(case["query"], documents)
                save_private_json(
                    root / f"{stem}--{arm}.request.json", request.model_dump(mode="json")
                )
                captured = []

                def capture(event, prefix=f"{stem}--{arm}", events=captured):
                    save_private_json(root / f"{prefix}.event-{len(events):03d}.json", event)
                    events.append(event)

                try:
                    result = await select(provider, request, documents, capture=capture)
                except RerankerFailure as error:
                    if error.diagnostic["stage"] not in {"response_shape", "evidence_validation"}:
                        save_private_json(root / "failure.json", error.diagnostic)
                        raise
                    result = {
                        "selections": [],
                        "elapsed_seconds": error.diagnostic["elapsed_seconds"],
                        "validation_failure": error.diagnostic,
                        "completion": next(
                            (e["payload"] for e in captured if e["type"] == "completed"), {}
                        ),
                    }
                ids = [mapping[s["candidate_id"]]["entry_id"] for s in result["selections"]]
                row = {
                    "case": case["id"],
                    "arm": arm,
                    "selected_ids": ids,
                    "recall4": len(set(ids) & set(case["required"])) / len(case["required"]),
                    "presented_body_bytes": sum(len(d["body"].encode()) for d in documents),
                    **result,
                }
                rows.append(row)
                save_private_json(root / f"{stem}--{arm}.result.json", row)
                print(
                    json.dumps(
                        {
                            "case": case["id"],
                            "arm": arm,
                            "recall4": row["recall4"],
                            "seconds": row["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )
        groups = {}
        for arm in ("full", "section"):
            chosen = [r for r in rows if r["arm"] == arm]
            times = [r["elapsed_seconds"] for r in chosen]
            groups[arm] = {
                "mean_recall4": statistics.mean(r["recall4"] for r in chosen),
                "median_seconds": statistics.median(times),
                "p95_seconds": statistics.quantiles(times, n=100, method="inclusive")[94],
                "input_tokens": sum(
                    r["completion"].get("usage", {}).get("input_tokens", 0) for r in chosen
                ),
            }
        save_private_json(
            root / "summary.json", {"rows": rows, "groups": groups, "default_changed": False}
        )
        print(json.dumps(groups), flush=True)
    finally:
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
