"""Bounded public policy exposure, without classifying customer intent.

The corpus's numbered introductory articles are supplied whole when they fit.
All other entries retain exact locators and Markdown headings for model selection.
Only complete supplied documents count as delivered evidence.
"""

MAX_OVERVIEW_BYTES = 24_000
MAX_HEADINGS = 16
MAX_HEADING_CHARS = 240


def progressive_reward_policy_view(payload):
    documents = []
    delivered_ids = set()
    remaining = MAX_OVERVIEW_BYTES
    for document in payload["documents"]:
        item = {key: document[key] for key in ("entry_id", "entry_revision", "title")}
        headings = []
        heading_count = 0
        for line in document["text"].splitlines():
            stripped = line.lstrip()
            marker, separator, heading = stripped.partition(" ")
            if separator and 1 <= len(marker) <= 6 and set(marker) == {"#"}:
                heading_count += 1
                if len(headings) < MAX_HEADINGS:
                    headings.append(heading[:MAX_HEADING_CHARS])
        item["section_headings"] = headings
        item["headings_truncated"] = heading_count > len(headings)
        item["text_complete"] = False
        # Public document numbering, not a task ID or a semantic intent rule.
        overview = document["entry_id"].rsplit("_", 1)[-1] == "001"
        item["introductory_article"] = overview
        size = len(document["text"].encode("utf-8"))
        if overview and document.get("text_complete") is True and size <= remaining:
            item = {**document, **item, "text": document["text"], "text_complete": True}
            delivered_ids.add(document["entry_id"])
            remaining -= size
        documents.append(item)
    return {
        "policy_dossier_id": payload["policy_dossier_id"],
        "card_types": payload["card_types"],
        "documents": documents,
    }, delivered_ids
