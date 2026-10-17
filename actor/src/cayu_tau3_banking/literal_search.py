"""Exhaustive literal discovery over an already authorized immutable corpus.

Locators and excerpts are navigation, not proof that a complete source was read.
Matching is Unicode case-insensitive literal text, with non-overlapping matches.
No evaluator records, filesystem paths, embeddings or model calls are involved.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from cayu import KnowledgeEntry


def literal_candidates(entries: Iterable[KnowledgeEntry], query: str) -> list[dict[str, Any]]:
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError("Literal query must contain 1-500 characters and nonblank text.")
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    candidates = []
    for entry in sorted(entries, key=lambda entry: entry.id):
        fields = []
        occurrences = 0
        for field, text in (("title", entry.title or ""), ("text", entry.text)):
            first = None
            count = 0
            for match in pattern.finditer(text):
                count += 1
                if first is None:
                    first = match
            if first is None:
                continue
            occurrences += count
            start, end = first.span()
            excerpt_start = max(0, start - 120)
            excerpt_end = min(len(text), end + 120)
            fields.append(
                {
                    "field": field,
                    "occurrences": count,
                    "first_match": {"start": start, "end": end},
                    "excerpt": text[excerpt_start:excerpt_end],
                    "excerpt_start": excerpt_start,
                    "excerpt_end": excerpt_end,
                    "excerpt_complete": excerpt_start == 0 and excerpt_end == len(text),
                }
            )
        if fields:
            candidates.append(
                {
                    "entry_id": entry.id,
                    "entry_revision": entry.revision,
                    "title": entry.title,
                    "found_by": ["literal"],
                    "literal_occurrences": occurrences,
                    "matches": fields,
                }
            )
    return candidates
