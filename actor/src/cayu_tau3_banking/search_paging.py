"""Bounded public-document paging; cursors are consistency checks, not authority.

No source query or ranking changes here. A cursor binds an offset to the exact
query/configuration/candidate snapshot. Replaying after restart is safe only if
that snapshot can be reproduced; changed results require an explicit restart.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence, Set
from typing import Any

_CURSOR = re.compile(r"v1:([a-f0-9]{64}):([1-9][0-9]{0,5})\Z")


def parse_cursor(cursor: str | None) -> tuple[str, int] | None:
    if cursor is None:
        return None
    match = _CURSOR.fullmatch(cursor) if isinstance(cursor, str) else None
    if match is None:
        raise ValueError("Invalid search cursor; restart without cursor.")
    return match[1], int(match[2])


def document_page(
    base: Mapping[str, Any],
    documents: Sequence[dict[str, Any]],
    *,
    limit: int,
    exclude_entry_ids: Set[str],
    cursor: str | None,
) -> dict[str, Any]:
    """Page a fixed ordered candidate pool, skipping already delivered documents.

    Offset is into the original pool, never a filtered list. Exclusions may grow
    between calls without causing skips. Counts describe this pool only; a false
    has_more_in_pool says nothing about answer completeness or the entire corpus.
    """
    if type(limit) is not int or not 1 <= limit <= 4:
        raise ValueError("Search page limit must be an integer from 1 through 4.")
    parsed = parse_cursor(cursor)
    snapshot = hashlib.sha256(
        json.dumps(
            [base, documents], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    offset = 0
    if parsed is not None:
        if parsed[0] != snapshot or parsed[1] >= len(documents):
            raise ValueError(
                "Search cursor no longer matches these results; restart without cursor."
            )
        offset = parsed[1]
    selected = []
    scanned = offset
    skipped = 0
    while scanned < len(documents) and len(selected) < limit:
        document = documents[scanned]
        scanned += 1
        if document["entry_id"] in exclude_entry_ids:
            skipped += 1
        else:
            selected.append(document)
    remaining = sum(d["entry_id"] not in exclude_entry_ids for d in documents[scanned:])
    return {
        **base,
        "documents": selected,
        "returned": len(selected),
        "navigation": {
            "candidate_documents_in_pool": len(documents),
            "already_delivered_in_pool": sum(d["entry_id"] in exclude_entry_ids for d in documents),
            "already_delivered_skipped_on_page": skipped,
            "remaining_unseen_in_pool": remaining,
            "has_more_in_pool": remaining > 0,
            "next_cursor": f"v1:{snapshot}:{scanned}" if remaining else None,
            "pool_exhaustion_proves_answer_complete": False,
            "continuation": "Repeat the same query and mode with next_cursor; pages share the search budget. If the missing fact is a different subject, refine the query or switch mode instead.",
        },
    }
