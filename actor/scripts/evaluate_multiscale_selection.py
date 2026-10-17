"""Offline, fixed candidate-union/selection experiment over recorded public hits.

No model calls, task execution, hidden evaluator inputs, or default changes.
Rankers accept only source ranks, never evaluation labels or required documents.
"""

import hashlib
import json
import tempfile
from pathlib import Path

ROOT = Path("artifacts/section-embedding-validation/run-eo81fta9")
BM25 = "cayu.tau3.banking.bm25"
KEYWORD = "knowledge.lexical"
WHOLE = "knowledge.semantic"
SECTION = "experiment.section.semantic"
OFFER_LEX = "cayu.tau3.banking.offer_facets.lexical"
OFFER_SEM = "cayu.tau3.banking.offer_facets.semantic"
WEIGHTS = {BM25: 1.0, KEYWORD: 0.5, WHOLE: 2.0, OFFER_LEX: 0.25, OFFER_SEM: 4.0}
ORDER = (BM25, WHOLE, SECTION, KEYWORD, OFFER_SEM, OFFER_LEX)


def source_lanes(data):
    """Project recorded chunk hits to canonical document revision identities."""
    lanes = {}
    for source in (*data["sources"], data["section_source"]):
        section = source is data["section_source"]
        records = {
            tuple(r["identity"][k] for k in ("record_type", "record_id", "revision")): r
            for r in source["records"]
        }
        for channel in source["channels"]:
            if section and channel["channel"] != WHOLE:
                continue
            name = SECTION if section else channel["channel"]
            if name in lanes:
                raise ValueError("Duplicate source channel.")
            ranks = {}
            for hit in sorted(channel["hits"], key=lambda h: h["rank"]):
                record = records[
                    tuple(hit["identity"][k] for k in ("record_type", "record_id", "revision"))
                ]
                identity = (record["locator"]["entry_id"], str(record["locator"]["entry_revision"]))
                ranks.setdefault(identity, hit["rank"])
            if len(ranks) > 16:
                raise ValueError("Captured source exceeds frozen candidate bound.")
            lanes[name] = ranks
    if set(lanes) != set(ORDER):
        raise ValueError("Unexpected source set.")
    return lanes


def weighted_selection(lanes, *, mode):
    """Reference RRF and two predeclared multi-representation selectors.

    max_semantic keeps lexical/facet weights and uses 2/(60+best semantic rank),
    never adding two votes for one document's two semantic representations.
    """
    if mode not in {"whole_control", "split_semantic", "max_semantic"}:
        raise ValueError("Unknown selector.")
    weights = dict(WEIGHTS)
    if mode == "split_semantic":
        weights[WHOLE] = weights[SECTION] = 1.0
    elif mode == "max_semantic":
        weights[SECTION] = 2.0
    candidates = set().union(*(set(lanes[name]) for name in weights))
    rows = []
    for identity in candidates:
        contributions = {
            name: weight / (60 + lanes[name][identity])
            for name, weight in weights.items()
            if identity in lanes[name]
        }
        ranks = [lanes[name][identity] for name in contributions]
        if mode == "max_semantic":
            semantic = [
                contributions.pop(name) for name in (WHOLE, SECTION) if name in contributions
            ]
            if semantic:
                contributions["semantic_family"] = max(semantic)
        rows.append((identity, sum(contributions.values()), min(ranks), len(contributions)))
    return [
        identity for identity, _, _, _ in sorted(rows, key=lambda r: (-r[1], r[2], -r[3], r[0]))
    ]


def round_robin(lanes):
    """Take the next unseen document from each source in a frozen source order."""
    remaining = {
        name: iter(sorted(lanes[name], key=lambda key: (lanes[name][key], key))) for name in ORDER
    }
    result, seen = [], set()
    while remaining:
        for name in tuple(remaining):
            for identity in remaining[name]:
                if identity not in seen:
                    seen.add(identity)
                    result.append(identity)
                    break
            else:
                del remaining[name]
    return result


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    output_parent = Path("artifacts/multiscale-selection-validation")
    output_parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=output_parent))
    plan = {
        "source": str(ROOT),
        "corpus_sha256": manifest["corpus_sha256"],
        "source_manifest_sha256": hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profiles": ["whole_control", "split_semantic", "max_semantic", "round_robin"],
        "pool": "Union of top16 canonical documents from six captured channels; at most 96. Same pool for the three candidates. No hidden labels in selectors.",
        "weights": WEIGHTS,
        "rrf_k": 60,
        "round_robin_order": ORDER,
        "gate": "No per-query recall4 regression versus either whole canonical control or original passing-profile retrieval; recover tier evidence. No automatic promotion. No tuning within this run.",
        "limitations": "Previously inspected development queries; no new blind validation, no model reranking, no runtime admission or full tau task evaluation.",
    }
    (output / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    for case in manifest["cases"]:
        data = json.loads((ROOT / f"{case['suite']}--{case['id']}.json").read_text())
        lanes = source_lanes(data)
        pool = set().union(*map(set, lanes.values()))
        selected = {
            name: weighted_selection(lanes, mode=name)
            for name in plan["profiles"]
            if name != "round_robin"
        }
        selected["round_robin"] = round_robin(lanes)
        for name in ("split_semantic", "max_semantic", "round_robin"):
            assert set(selected[name]) == pool and len(selected[name]) == len(pool)
        # Audit our reference formula/tie break against Cayu's recorded result.
        assert [key[0] for key in selected["whole_control"][:16]] == data["row"]["profiles"][
            "whole"
        ]["ids"]
        required = {(entry_id, "1") for entry_id in case["required"]}
        profiles = {
            name: {
                "ids": [key[0] for key in ids],
                "recall4": len(required.intersection(ids[:4])) / len(required),
                "recall16": len(required.intersection(ids[:16])) / len(required),
            }
            for name, ids in selected.items()
        }
        rows.append(
            {
                "id": case["id"],
                "suite": case["suite"],
                "pool_size": len(pool),
                "pool_recall": len(required & pool) / len(required),
                "profiles": profiles,
                "original_recall4": data["row"]["profiles"]["original"]["recall4"],
            }
        )
    candidates = {}
    for name in plan["profiles"][1:]:
        regressions = [
            r["id"]
            for r in rows
            if r["profiles"][name]["recall4"]
            < max(r["profiles"]["whole_control"]["recall4"], r["original_recall4"])
        ]
        tier = next(r for r in rows if r["suite"] == "observed-regression")
        candidates[name] = {
            "mean_recall4": sum(r["profiles"][name]["recall4"] for r in rows) / len(rows),
            "regressions": regressions,
            "gate_passed": not regressions and tier["profiles"][name]["recall4"] == 1.0,
        }
    summary = {
        "rows": rows,
        "candidates": candidates,
        "control_parity_queries": len(rows),
        "all_required_in_pool": all(r["pool_recall"] == 1.0 for r in rows),
        "default_changed": False,
        "model_calls": 0,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(output)
    print(
        json.dumps(
            {"candidates": candidates, "all_required_in_pool": summary["all_required_in_pool"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
