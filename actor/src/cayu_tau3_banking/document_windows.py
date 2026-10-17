"""Exact-source navigation windows, without full-document evidence claims."""

from typing import Any


def document_window(document: dict[str, Any], span: dict[str, int]) -> dict[str, Any]:
    text = document["text"]
    start, end = span["start"], span["end"]
    context = span.get("context_chars", 400)
    if (
        type(start) is not int
        or type(end) is not int
        or type(context) is not int
        or not 0 <= start < end <= len(text)
        or end - start > 4000
        or not 0 <= context <= 2000
    ):
        raise ValueError(
            "Use valid source offsets, at most 4000 selected characters and 2000 context characters."
        )
    left, right = max(0, start - context), min(len(text), end + context)
    result = {
        key: value
        for key, value in document.items()
        if key not in {"evidence_ref", "content_sha256"}
    }
    if "content_sha256" in document:
        result["source_content_sha256"] = document["content_sha256"]
    result.update(
        text=text[left:right],
        text_complete=left == 0 and right == len(text),
        navigation_only=True,
        source_window={
            "field": "text",
            "offset_units": "unicode_code_points",
            "start": left,
            "end": right,
            "requested_start": start,
            "requested_end": end,
            "total_characters": len(text),
            "has_before": left > 0,
            "has_after": right < len(text),
        },
        reading_note="Exact source window, not full-document delivery or proof of policy applicability. Omit source_span to read the complete revision.",
    )
    return result
