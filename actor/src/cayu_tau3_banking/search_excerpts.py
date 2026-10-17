"""Source-faithful navigation excerpts; never claim a lexical hit is semantic evidence."""

from __future__ import annotations

import re


def matching_excerpt(text: str, query: str) -> dict[str, object] | None:
    terms = sorted(
        set(re.findall(r"\w[\w'-]*", query.casefold())),
        key=lambda value: (-len(value), value),
    )
    if not terms:
        return None
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(term) for term in terms) + r")(?!\w)",
        re.IGNORECASE,
    )
    # Favor a line matching multiple distinct query terms, rather than an
    # incidental occurrence near the beginning. No synthetic summary is produced.
    best = None
    offset = 0
    heading = None
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("#"):
            heading = line.strip()[:160]
        hits = list(pattern.finditer(line))
        if hits:
            score = len({hit.group().casefold() for hit in hits})
            if best is None or score > best[0]:
                best = (score, offset + hits[0].start(), heading)
        offset += len(line)
    if best is None:
        return None
    start = max(0, best[1] - 120)
    end = min(len(text), best[1] + 380)
    return {
        "kind": "lexical_navigation_excerpt",
        "field": "text",
        "start": start,
        "end": end,
        "offset_units": "unicode_code_points",
        "text": text[start:end],
        "heading": best[2],
        "complete_document": start == 0 and end == len(text),
        "note": "Query-word match, not a semantic relevance or policy-applicability judgment.",
    }
