"""Frozen, offline lexical-first vs family-balanced retrieval experiment.

No model calls, hidden task answers, per-query rules or production ranking changes.
Reuses previously captured whole/section/source ranks, retaining canonical revisions.
"""

import hashlib
import json
import statistics
import tempfile
from pathlib import Path

from evaluate_multiscale_selection import (
    BM25,
    KEYWORD,
    OFFER_LEX,
    OFFER_SEM,
    ROOT,
    SECTION,
    WHOLE,
    source_lanes,
    weighted_selection,
)


def family_order(lanes, names):
    """Strongest rank per family; multiple representations do not multiply votes."""
    ids = set().union(*(set(lanes[name]) for name in names))
    return sorted(ids, key=lambda key: (min(lanes[n][key] for n in names if key in lanes[n]), key))


def merge_families(lexical, semantic, *, mode):
    if mode == "lexical_first":
        return list(dict.fromkeys([*lexical, *semantic]))
    if mode not in {"balanced_lexical_first", "balanced_semantic_first"}:
        raise ValueError("Unknown selection mode")
    lanes = [iter(lexical), iter(semantic)]
    if mode == "balanced_semantic_first":
        lanes.reverse()
    seen, result = set(), []
    while lanes:
        remaining = []
        for lane in lanes:
            for key in lane:
                if key not in seen:
                    seen.add(key)
                    result.append(key)
                    remaining.append(lane)
                    break
        lanes = remaining
    return result


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    parent = Path("artifacts/balanced-selection-validation")
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    profiles = ["lexical_first", "balanced_lexical_first", "balanced_semantic_first"]
    plan = {
        "source": str(ROOT),
        "source_sha256": hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profiles": profiles,
        "families": {
            "lexical": [BM25, KEYWORD, OFFER_LEX],
            "semantic": [WHOLE, SECTION, OFFER_SEM],
        },
        "family_selection": "Best source rank per document revision, tie by identity; no multiplying representation votes.",
        "gate": "No per-query recall4 regression against original passing profile or strongest-semantic reference; recover tier evidence. No tuning or promotion within run.",
        "limits": [4, 8, 12, 16],
        "limitations": "Previously inspected exact-ID labels, not exhaustive relevance judgments; bounded upstream sources; no task execution or blind validation.",
    }
    (output / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    for case in manifest["cases"]:
        data = json.loads((ROOT / f"{case['suite']}--{case['id']}.json").read_text())
        lanes = source_lanes(data)
        lexical = family_order(lanes, plan["families"]["lexical"])
        semantic = family_order(lanes, plan["families"]["semantic"])
        rankings = {mode: merge_families(lexical, semantic, mode=mode) for mode in profiles}
        rankings["max_semantic"] = weighted_selection(lanes, mode="max_semantic")
        required = {(key, "1") for key in case["required"]}
        metrics = {}
        for mode, ids in rankings.items():
            assert len(ids) == len(set(ids)) and set(ids) == set(lexical) | set(semantic)
            metrics[mode] = {
                "ids": [key[0] for key in ids],
                **{
                    f"recall{k}": len(required & set(ids[:k])) / len(required)
                    for k in plan["limits"]
                },
            }
        rows.append(
            {
                "id": case["id"],
                "profiles": metrics,
                "original_recall4": data["row"]["profiles"]["original"]["recall4"],
            }
        )
    summary = {}
    for mode in [*profiles, "max_semantic"]:
        regressions = [
            r["id"]
            for r in rows
            if r["profiles"][mode]["recall4"]
            < max(r["original_recall4"], r["profiles"]["max_semantic"]["recall4"])
        ]
        summary[mode] = {
            **{
                f"mean_recall{k}": statistics.mean(r["profiles"][mode][f"recall{k}"] for r in rows)
                for k in plan["limits"]
            },
            "regressions": regressions,
            "gate_passed": not regressions
            and next(r for r in rows if r["id"] == "observed-tier-query")["profiles"][mode][
                "recall4"
            ]
            == 1,
        }
    (output / "summary.json").write_text(
        json.dumps(
            {"profiles": summary, "rows": rows, "default_changed": False, "model_calls": 0},
            indent=2,
        )
        + "\n"
    )
    print(output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
