"""Bounded passage retention over the existing revision-checked evidence store."""

import hashlib
import json
from copy import deepcopy

from cayu_tau3_banking.product_navigation import COMPARISON_TOOL

MAX_PASSAGES = 96
MAX_PRODUCTS = 16
READ_BYTES = 12_000


async def retain_passages(nav, args, delivered):
    if not nav.validators[COMPARISON_TOOL].is_valid(args):
        raise ValueError("Invalid passage request.")
    name = args.get("worksheet_id", "default")
    handles = list(dict.fromkeys(args.get("passage_handles", [])))
    entries = list(dict.fromkeys(args.get("entry_ids", [])))
    operation = args.get("operation", "append" if handles else "read")
    saved = nav._worksheet(name)
    existing = deepcopy(saved["arguments"]["facts"]) if saved else []
    errors = []
    accepted = []
    response = {
        "worksheet_id": name,
        "operation": operation,
        "accepted_handles": accepted,
        "errors": errors,
        "semantic_entailment_checked": False,
        "authorizes_completion": False,
    }
    if operation == "read":
        checked = (
            await nav.check(saved["arguments"], delivered, remember=False)
            if saved
            else {"facts": []}
        )
        by_handle = {f["field"]: f for f in checked["facts"]}
        if entries:
            requested = [h for h, f in by_handle.items() if f["entry_id"] in entries]
            present = {f["entry_id"] for f in checked["facts"]}
            errors.extend({"entry_id": entry, "error": "No retained passages for this source in this worksheet."}
                          for entry in entries if entry not in present)
        else:
            requested = handles or list(by_handle)
        identity = hashlib.sha256(
            json.dumps([name, nav.corpus.corpus_sha256, saved, handles, entries], sort_keys=True).encode()
        ).hexdigest()
        offset = 0
        if "cursor" in args:
            digest, separator, position = args["cursor"].partition(":")
            if digest != identity or not separator or not position.isdecimal():
                raise ValueError(
                    "Stale or invalid recovery cursor. Restart operation=read without cursor using the intended selection."
                )
            offset = int(position)
            if not 0 < offset < len(requested):
                raise ValueError("Invalid recovery cursor. Restart operation=read without cursor.")
        passages = []
        used = 0
        index = offset
        while index < len(requested):
            handle = requested[index]
            fact = by_handle.get(handle)
            if fact is None:
                errors.append(
                    {
                        "passage_handle": handle,
                        "error": "Passage is not retained in this worksheet.",
                    }
                )
                index += 1
                continue
            row = {
                "passage_handle": handle,
                "entry_id": fact["entry_id"],
                "revision": fact["revision"],
                "text": fact["quote"],
            }
            size = len(json.dumps(row, ensure_ascii=False).encode())
            if used + size > READ_BYTES:
                break
            passages.append(row)
            used += size
            index += 1
        response.update(
            pinned_passages=passages,
            next_cursor=f"{identity}:{index}" if index < len(requested) else None,
            retained_count=len(existing),
            retained_for_compaction=bool(saved),
        )
        return response
    if "cursor" in args:
        raise ValueError("cursor is only valid for operation=read.")
    valid = []
    for handle in handles:
        passage = nav.passages.get(handle)
        if passage is None or not delivered(passage["entry_id"], str(passage["revision"])):
            errors.append(
                {
                    "passage_handle": handle,
                    "error": "Unknown or unread passage; copy a handle from a complete document read.",
                }
            )
        else:
            valid.append(handle)
    if operation == "replace" and errors:
        return {
            **response,
            "retained_count": len(existing),
            "retained_for_compaction": bool(saved),
            "replacement_applied": False,
        }
    facts = {} if operation == "replace" else {(f["product_id"], f["field"]): f for f in existing}
    products = {key[0] for key in facts}
    for handle in valid:
        p = nav.passages[handle]
        key = (p["product_id"], handle)
        if p["product_id"] not in products and len(products) >= MAX_PRODUCTS:
            errors.append(
                {
                    "passage_handle": handle,
                    "error": "16-product capacity reached. Use operation=replace with a curated set of at most 16 products; other fitting append handles can still be saved.",
                }
            )
            continue
        if key not in facts and len(facts) >= MAX_PASSAGES:
            errors.append(
                {
                    "passage_handle": handle,
                    "error": "96-passage capacity reached. Use operation=replace with the curated set to retain; prior evidence is unchanged for this handle.",
                }
            )
            continue
        products.add(p["product_id"])
        facts[key] = {
            "product_id": p["product_id"],
            "entry_id": p["entry_id"],
            "revision": p["revision"],
            "field": handle,
            "value": p["text"].split()[0][:200],
            "source_span": p["source_span"],
        }
        accepted.append(handle)
    if operation == "replace" and errors:
        accepted.clear()
        return {
            **response,
            "retained_count": len(existing),
            "retained_for_compaction": bool(saved),
            "replacement_applied": False,
        }
    if not facts:
        if operation == "replace":
            nav._save_worksheet(name, None)
            nav._comparison_readiness.pop(name, None)
    elif accepted:
        proposal = {
            "worksheet_id": name,
            "product_ids": sorted({key[0] for key in facts}),
            "required_fields": [],
            "required_cells": [{"product_id": p, "field": f} for p, f in facts],
            "facts": list(facts.values()),
        }
        try:
            await nav.check(proposal, delivered)
        except ValueError as error:
            # Storage validation is atomic; preserve the previous collection.
            errors.extend({"passage_handle": h, "error": str(error)[:400]} for h in accepted)
            accepted.clear()
            return {
                **response,
                "retained_count": len(existing),
                "retained_for_compaction": bool(saved),
                "replacement_applied": False,
            }
    return {
        **response,
        "retained_count": len(facts),
        "retained_for_compaction": bool(facts),
        "replacement_applied": operation == "replace",
        "capacity": MAX_PASSAGES,
        "product_capacity": MAX_PRODUCTS,
        "retained_product_count": len(products),
    }
