"""Actor-selected public scope before retrieval; no intent or policy classifier."""

from __future__ import annotations

import copy
import json
import re

from rank_bm25 import BM25Plus

from cayu_tau3_banking.document_windows import document_window
from cayu_tau3_banking.search_paging import document_page

SCOPE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "minProperties": 1,
    "properties": {
        "category": {"type": "string", "minLength": 1, "maxLength": 200},
        "product_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 300},
        },
        "entry_ids": {
            "type": "array",
            "minItems": 1,
            "maxItems": 16,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 500},
        },
    },
    "description": "Exact public directory identifiers. Multiple fields intersect; values within a list form a union. Include general/procedure families when relevant.",
}
FIELDS = {
    "scope": SCOPE_SCHEMA,
    "terms": {
        "type": "array",
        "minItems": 1,
        "maxItems": 8,
        "items": {"type": "string", "minLength": 1, "maxLength": 100},
        "description": "Optional short literal substrings, case insensitive, in title/body. Omit for BM25 keyword ranking within scope. Not semantic conditions.",
    },
    "match": {"enum": ["any", "all"], "default": "any"},
}


def extend_search_schema(schema):
    schema["description"] = (
        "For a known product or source, add scope using exact public directory identifiers "
        "to search that subset BEFORE ranking. Omit mode (or use default/lexical/scoped): "
        "scope selects BM25 within that subset, not global semantic retrieval. "
        "With mode=literal, query is an exact case-insensitive substring within scope (up to 500 characters). "
        "Optional terms match any/all literal substrings; "
        "otherwise query supplies BM25 keywords. Returns complete short sources, bounded "
        "navigation windows for long sources. Empty matches never establish policy absence. "
        "Batch scope/terms/match belong inside each query, not at top level. "
        "Continuation must preserve query, scope, terms and match."
    ) + " " + schema["description"]
    props = schema["input_schema"]["properties"]
    for fields in (props, props["queries"]["items"]["properties"]):
        fields.update(copy.deepcopy(FIELDS))
        fields["mode"]["enum"].append("scoped")
    schema["input_schema"]["oneOf"][1]["not"]["anyOf"].extend(
        {"required": [name]} for name in FIELDS
    )
    return schema


async def scoped_search(
    corpus, query, *, scope, terms, match, limit, exclude_entry_ids, cursor, capture,
    literal=False,
):
    from jsonschema import Draft202012Validator

    from cayu_tau3_banking.knowledge import (
        _bounded_query,
        _knowledge_evidence_reference,
        _search_tokens,
    )

    if literal:
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("Literal query requires 1-500 characters.")
    else:
        query = _bounded_query(query)
    if not Draft202012Validator(SCOPE_SCHEMA).is_valid(scope):
        raise ValueError("Use exact directory scope identifiers.")
    if match not in ("any", "all"):
        raise ValueError("match must be any or all.")
    if terms is not None and (
        not Draft202012Validator(FIELDS["terms"]).is_valid(terms)
        or any(not term.strip() for term in terms)
    ):
        raise ValueError("Use 1-8 nonblank short literal terms.")
    if terms is None and match != "any":
        raise ValueError("match all requires terms.")
    if literal:
        if terms is not None:
            raise ValueError("Use query alone for literal mode, or terms with default/scoped mode.")
        terms = [query]
    directory = corpus.document_directory()
    for field, values in scope.items():
        key = {"product_ids": "product_id", "entry_ids": "entry_id"}.get(field, field)
        known = {row[key] for row in directory}
        if not set(values if isinstance(values, list) else [values]) <= known:
            raise ValueError("Unknown directory scope identifier.")
    rows = [
        row
        for row in directory
        if ("category" not in scope or row["category"] == scope["category"])
        and ("product_ids" not in scope or row["product_id"] in scope["product_ids"])
        and ("entry_ids" not in scope or row["entry_id"] in scope["entry_ids"])
    ]
    entries = [corpus._entries[row["entry_id"]] for row in rows]
    texts = [entry.title + "\n" + entry.text for entry in entries]
    if terms is not None:
        matched = [
            (row, 0.0)
            for row, text in zip(rows, texts, strict=True)
            if (all if match == "all" else any)(
                re.search(re.escape(term), text, re.IGNORECASE) is not None for term in terms
            )
        ]
    else:
        tokens = _search_tokens(query)
        if not tokens:
            raise ValueError("Supply query keywords or literal terms.")
        indexed = [_search_tokens(text) or [""] for text in texts]
        scores = BM25Plus(indexed).get_scores(tokens) if indexed else []
        matched = [
            (row, float(score))
            for row, words, score in zip(rows, indexed, scores, strict=True)
            if set(tokens).intersection(words)
        ]
        matched.sort(key=lambda item: (-item[1], item[0]["entry_id"]))
    candidates = [
        {
            "entry_id": row["entry_id"],
            "entry_revision": row["revision"],
            "product_id": row["product_id"],
            "category": row["category"],
            "score": score,
        }
        for row, score in matched
    ]
    payload = document_page(
        {
            "corpus_sha256": corpus.corpus_sha256,
            "query": query,
            "retrieval": {
                "mode": "scoped",
                "scope": scope,
                "terms": terms,
                "match": match,
                "ranking": "entry_id" if terms is not None else "bm25_within_scope",
            },
            "retrieval_bounds": {
                "documents_in_scope": len(rows),
                "matching_documents_in_scope": len(candidates),
                "scope_applied_before_ranking": True,
                "corpus_exhaustive": False,
            },
            "reading_instruction": "Counts describe lexical matches within this scope, not policy completeness or eligibility. No matches means unknown. Check relevant general/procedure families. Open incomplete sources with cayu_read_product_documents.",
        },
        candidates,
        limit=limit,
        exclude_entry_ids=exclude_entry_ids,
        cursor=cursor,
    )
    documents, references = [], []
    for candidate in payload["documents"]:
        document = {
            **candidate,
            **await corpus.read_document(candidate["entry_id"], candidate["entry_revision"]),
        }
        if len(document["text"].encode("utf-8")) > 8000:
            # Literal lexical navigation only; original Unicode offsets are preserved.
            hints = terms or _search_tokens(query)
            hits = [re.search(re.escape(term), document["text"], re.IGNORECASE) for term in hints]
            center = min((hit.start() for hit in hits if hit), default=0)
            start = max(0, center - 200)
            document = document_window(
                document,
                {
                    "start": start,
                    "end": min(len(document["text"]), start + 1800),
                    "context_chars": 0,
                },
            )
        else:
            references.append(_knowledge_evidence_reference(document))
        documents.append(document)
    payload["documents"] = documents
    payload["navigation_only"] = not references
    if capture is not None:
        payload["retrieval_trace_id"] = await capture(
            {
                "schema_version": 1,
                "mode": "scoped",
                "inventory": candidates,
                "page": copy.deepcopy(payload),
            }
        )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True), tuple(references)
