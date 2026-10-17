"""Retain independently checked evidence when a well-formed comparison fails.

Coverage and numeric conclusions are discarded on partial recovery. The ordinary
checker remains the only authority for literal source validation and storage bounds.
"""

import json
from collections import Counter
from copy import deepcopy

from cayu_tau3_banking.knowledge import BankingKnowledgeError
from cayu_tau3_banking.product_navigation import COMPARISON_TOOL


def brief_error(error):
    # Source excerpts already exist in the corpus; repeating one for every bad
    # cell can make the repair response larger than the rejected worksheet.
    return str(error).partition(" Already-delivered source for repair")[0][:400]


async def check_incremental(navigation, arguments, delivered):
    # Malformed requests retain the normal schema error and perform no mutation.
    if not navigation.validators[COMPARISON_TOOL].is_valid(arguments):
        raise ValueError("Invalid comparison worksheet arguments.")
    if "passage_handles" in arguments:
        arguments = navigation.normalize_comparison(arguments)
    # An unknown handle carries no trustworthy product identity. Do not guess
    # its row, but do not discard unrelated valid findings in the same capture.
    if arguments.get("capture") is True:
        unknown = [
            f
            for f in arguments["facts"]
            if "source_handle" in f
            and f["source_handle"] not in navigation.by_handle
            and "product_id" not in f
        ]
        remaining = [f for f in arguments["facts"] if f not in unknown]
        if unknown and remaining:
            result = await check_incremental(
                navigation, {**arguments, "facts": remaining}, delivered
            )
            result.setdefault("cell_errors", []).extend(
                {
                    "source_handle": f["source_handle"],
                    "field": f["field"],
                    "error": "Unknown source handle; no product inferred and this update was not applied. Copy a returned handle.",
                }
                for f in unknown
            )
            result.update(
                partial_recovery=True,
                original_request_complete=False,
                declared_cells_complete=False,
                declared_inputs_complete=False,
                declared_coverage_complete=False,
                authorizes_completion=False,
            )
            result["instruction"] = (
                "Known-source findings were checked independently. Unknown-handle updates were not applied; repair only those cells. No complete coverage is established."
            )
            navigation._comparison_readiness[arguments.get("worksheet_id", "default")] = False
            return result
    arguments = navigation.normalize_comparison(arguments)
    before = navigation.snapshot()
    readiness = dict(navigation.comparison_readiness())
    try:
        return await navigation.check(arguments, delivered)
    except (ValueError, BankingKnowledgeError) as error:
        original_error = brief_error(error)
    name = arguments.get("worksheet_id", "default")
    saved = navigation._worksheet(name)
    facts = deepcopy(saved["arguments"]["facts"]) if saved else []
    keyed = {(f["product_id"], f["field"]): f for f in facts}
    counts = Counter((f["product_id"], f["field"]) for f in arguments["facts"])
    rejected = []
    accepted = []
    repair_bytes_left = 12_000
    for fact in arguments["facts"]:
        key = (fact["product_id"], fact["field"])
        # An invalid update must not leave an older value looking current.
        keyed.pop(key, None)
        if counts[key] > 1:
            rejected.append(
                {
                    "product_id": key[0],
                    "field": key[1],
                    "error": "Duplicate cell; no update accepted.",
                }
            )
            continue
        single = {"product_ids": [key[0]], "required_fields": [key[1]], "facts": [fact]}
        try:
            checked = await navigation.check(single, delivered, remember=False)
        except (ValueError, BankingKnowledgeError) as error:
            detail = {"product_id": key[0], "field": key[1], "error": brief_error(error)}
            if repair := getattr(error, "repair", None):
                repair = deepcopy(repair)
                while (
                    repair["passages"]
                    and len(json.dumps(repair, ensure_ascii=False).encode()) > repair_bytes_left
                ):
                    repair["passages"].pop()
                    repair["omitted_matching_paragraphs"] += 1
                size = len(json.dumps(repair, ensure_ascii=False).encode())
                if size <= repair_bytes_left:
                    detail["source_repair"] = repair
                    repair_bytes_left -= size
                else:
                    detail["source_repair_omitted"] = True
            rejected.append(detail)
            continue
        checked_fact = deepcopy(checked["facts"][0])
        checked_fact.pop("evidence_ref", None)
        checked_fact.pop("quote", None)  # Validated exact span rehydrates the quote.
        keyed[key] = checked_fact
        accepted.append({"product_id": key[0], "field": key[1]})
    # Include missing cells in retained scope. Do not invent requirements for
    # an entirely undeclared product; report the original scope error instead.
    cells = {(f["product_id"], f["field"]) for f in keyed.values()}
    for source in ([saved["arguments"]] if saved else []) + [arguments]:
        products = source.get("product_ids", [])
        cells.update(
            (p, f)
            for p in products
            if p in navigation.products
            for f in source.get("required_fields", [])
        )
        cells.update(
            (c["product_id"], c["field"])
            for c in source.get("required_cells", [])
            if c["product_id"] in navigation.products
        )
    cells.update(
        (f["product_id"], f["field"]) for f in rejected if f["product_id"] in navigation.products
    )
    recovery = {
        "worksheet_id": name,
        "product_ids": sorted({p for p, _ in cells}),
        "required_fields": [],
        "required_cells": [{"product_id": p, "field": f} for p, f in sorted(cells)],
        "facts": list(keyed.values()),
    }
    navigation._comparison_readiness = readiness
    try:
        result = await navigation.check(recovery, delivered)
    except (ValueError, BankingKnowledgeError) as error:
        # No partial-capacity overwrite or corrupted checkpoint on failure.
        navigation.state = before
        navigation._comparison_readiness = readiness
        return {
            "partial_recovery": True,
            "original_request_complete": False,
            "retained_for_compaction": False,
            "worksheet_error": original_error,
            "recovery_error": brief_error(error),
            "cell_errors": rejected,
            "accepted_updates": [],
            "declared_coverage_complete": False,
            "authorizes_completion": False,
            "instruction": "Previous worksheet is unchanged. Reduce scope or repair the request; no new facts were saved.",
        }
    result.update(
        {
            "partial_recovery": True,
            "original_request_complete": False,
            "declared_cells_complete": False,
            "declared_inputs_complete": False,
            "worksheet_error": original_error,
            "cell_errors": rejected,
            "accepted_updates": accepted,
            "declared_coverage_complete": False,
            "authorizes_completion": False,
            "instruction": "Valid source cells are retained; coverage and calculations were not accepted. Patch only rejected/missing cells using this worksheet_id. This optional aid is not a requirement to answer or act on otherwise sufficient evidence.",
        }
    )
    navigation._comparison_readiness[name] = False
    return result
