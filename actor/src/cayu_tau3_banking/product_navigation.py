"""Explicit public-product navigation and source-checked comparison worksheets.

No customer-language routing and no model calls. Worksheet completeness is
relative to the actor-declared rows/fields, not a claim of exhaustive coverage
or semantic entailment. Original excerpts remain available for model reasoning.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from decimal import Decimal, localcontext
from itertools import pairwise
from typing import Any

from jsonschema import Draft202012Validator

from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus

BROWSE_TOOL = "cayu_browse_product_documents"
READ_TOOL = "cayu_read_product_documents"
COMPARISON_TOOL = "cayu_check_comparison_facts"
NAVIGATION_TOOLS = frozenset({BROWSE_TOOL, READ_TOOL, COMPARISON_TOOL})
# Independent per-customer / per-conversation bounds. Metadata exploration must
# never consume the capacity reserved for source reads or arithmetic validation.
BUDGET_LIMITS = {BROWSE_TOOL: (72, 576), READ_TOOL: (96, 768), COMPARISON_TOOL: (48, 384)}
RECOVERY_LANE = "exact_source_recovery"
RECOVERY_LIMITS = (6, 24)
MAX_READ_DOCUMENTS = 16  # Same 72 KB result bound; complete sources or atomic rejection.
MAX_PRODUCT_DOCUMENTS = 16
MAX_READ_RESULT_BYTES = 72_000
MAX_RETAINED_LOCATORS = 64
_NUMBER = r"[+-]?[0-9]{1,16}(?:\.[0-9]{1,8})?"
_GROUPED_NUMBER = r"[+-]?[0-9]{1,3}(?:,[0-9]{3}){1,5}(?:\.[0-9]{1,8})?"


def _fact_number(value: str) -> Decimal | None:
    """Parse a literal US-style number without altering its stored source quote.

    Grouping is syntax, not a unit conversion. Reject malformed grouping,
    currency/percent suffixes, exponents and excessive precision as before.
    """
    if re.fullmatch(_NUMBER, value):
        return Decimal(value)
    if re.fullmatch(_GROUPED_NUMBER, value):
        ungrouped = value.replace(",", "")
        if re.fullmatch(_NUMBER, ungrouped):
            return Decimal(ungrouped)
    return None


def _schema(name, description, properties, required, **constraints):
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
            **constraints,
        },
    }


def tool_schemas():
    text = {"type": "string", "minLength": 1, "maxLength": 240}
    source = {
        "type": "object",
        "properties": {
            "entry_id": text,
            "source_handle": text,
            "revision": {"type": "integer", "minimum": 1},
        },
        "required": [],
        "oneOf": [
            {"required": ["entry_id"], "not": {"required": ["source_handle"]}},
            {
                "required": ["source_handle"],
                "not": {"anyOf": [{"required": ["entry_id"]}, {"required": ["revision"]}]},
            },
        ],
        "additionalProperties": False,
    }
    fact = {
        "type": "object",
        "properties": {
            "product_id": text,
            "field": text,
            "value": {"type": "string", "minLength": 1, "maxLength": 200},
            "entry_id": text,
            "source_handle": text,
            "revision": {"type": "integer", "minimum": 1},
            "quote": {"type": "string", "minLength": 8, "maxLength": 1600},
            "source_span": {
                "type": "object",
                "description": "Copy a returned exact source span instead of retyping its quote. Character offsets use an exclusive end; never estimate them.",
                "properties": {
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1},
                },
                "required": ["start", "end"],
                "additionalProperties": False,
            },
            "interpretation": {
                "type": "string",
                "minLength": 1,
                "maxLength": 800,
                "description": "Your explanation or inference, separate from the literal source value and quote. Retained but NOT verified; never used as a numeric calculation input.",
            },
            "source_ref": {
                "type": "object",
                "description": "Copy the source's delivered evidence_ref; no checksum reconstruction.",
                "properties": {
                    "kind": {"const": "knowledge_revision"},
                    "reference_id": text,
                    "version": {"type": "string", "pattern": "^[1-9][0-9]*$", "maxLength": 12},
                },
                "required": ["kind", "reference_id", "version"],
                "additionalProperties": False,
            },
        },
        "required": ["field", "value"],
        "additionalProperties": False,
        "not": {"required": ["quote", "source_span"]},
        "oneOf": [
            {
                "required": ["product_id", "entry_id", "revision"],
                "not": {"anyOf": [{"required": ["source_ref"]}, {"required": ["source_handle"]}]},
            },
            {
                "required": ["source_handle"],
                "not": {
                    "anyOf": [
                        {"required": ["entry_id"]},
                        {"required": ["revision"]},
                        {"required": ["source_ref"]},
                    ]
                },
            },
            {
                "required": ["product_id", "source_ref"],
                "not": {
                    "anyOf": [
                        {"required": ["entry_id"]},
                        {"required": ["revision"]},
                        {"required": ["source_handle"]},
                    ]
                },
            },
        ],
    }
    return (
        _schema(
            BROWSE_TOOL,
            "Browse the exact public document taxonomy without ranking. Omit filters for categories; "
            "supply category for products, then product_id for document titles and revision locators. "
            "Read source bodies with cayu_read_product_documents. Similar product names are distinct. "
            "General documents are also listed. Offset pages are bound to the immutable corpus. "
            "Use product_ids to list documents for up to 16 products in one bounded, paginated request; "
            "do not combine it with product_id. Directory entries are not factual evidence. "
            f"Discovery has {BUDGET_LIMITS[BROWSE_TOOL][0]} calls per customer / "
            f"{BUDGET_LIMITS[BROWSE_TOOL][1]} total, separate from reads, checking and search.",
            {
                "category": text,
                "product_id": text,
                "product_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 16,
                    "uniqueItems": True,
                    "items": text,
                },
                "offset": {"type": "integer", "minimum": 0},
            },
            [],
        ),
        _schema(
            READ_TOOL,
            f"Read up to {MAX_READ_DOCUMENTS} exact public document revisions from the directory, bypassing search ranking. "
            "In documents, copy source_handle or entry_id; omitted revision resolves to this immutable corpus revision. Never reconstruct identifiers. "
            "Alternatively, supply one exact product_id instead of documents to read ALL its directory documents "
            f"as one source-complete product packet (at most {MAX_PRODUCT_DOCUMENTS} documents). "
            "For a shortlisted product, a packet includes published qualifications as well as benefits, "
            "without guessing which FAQ title contains a decisive condition. General banking procedures "
            "remain separate; a complete packet does not establish customer eligibility or consent. "
            "Returns complete source text and evidence references. Requests containing any undelivered "
            f"revision use new-research capacity ({BUDGET_LIMITS[READ_TOOL][0]} calls per customer / "
            f"{BUDGET_LIMITS[READ_TOOL][1]} total). Requests containing "
            "ONLY previously delivered exact revisions automatically use separate source-recovery "
            "capacity (6 calls per customer / 24 total), not new-research slots. Recovery never supplies "
            "an unread revision and cannot borrow new-research slots when exhausted. "
            f"The complete result must fit {MAX_READ_RESULT_BYTES} UTF-8 bytes; split an oversized batch. "
            "Oversized product packets are rejected atomically; use the directory and exact-document batches instead. "
            "Use targeted exact revisions for a known question or recovery; use a full product packet "
            "when its whole scope is needed. Preserve decision facts in a comparison worksheet to "
            "avoid repeatedly recovering whole packets after compaction. "
            "For navigation, optionally supply source_span with ONE exact document: copy text-field "
            "Unicode offsets from a search excerpt and request surrounding context. This returns "
            "a source window without full-document evidence or tool unlocks; omit source_span for full reading.",
            {
                "documents": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_READ_DOCUMENTS,
                    "items": source,
                },
                "product_id": text,
                "source_span": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "integer", "minimum": 0},
                        "end": {"type": "integer", "minimum": 1},
                        "context_chars": {"type": "integer", "minimum": 0, "maximum": 2000},
                    },
                    "required": ["start", "end"],
                    "additionalProperties": False,
                },
            },
            [],
            allOf=[
                {
                    "if": {"required": ["source_span"]},
                    "then": {
                        "required": ["documents"],
                        "properties": {"documents": {"maxItems": 1}},
                    },
                }
            ],
            oneOf=[
                {"required": ["documents"], "not": {"required": ["product_id"]}},
                {"required": ["product_id"], "not": {"required": ["documents"]}},
            ],
        ),
        _schema(
            COMPARISON_TOOL,
            "Retain source passages directly: send passage_handles copied from document reads (and optional worksheet_id). "
            "This pins exact text without scalar extraction, quote reconstruction or coverage declarations. "
            "Use a small set of passages distinguishing plausible alternatives. Never submit passage_handles "
            "with facts, capture, calculations or coverage. Stored text is evidence, not a verified interpretation. "
            "Legacy detailed interface: send capture=true and facts containing source_handle, field and literal value; "
            "product and revision are resolved automatically. Optional interpretation is your unverified note. "
            "This merges into the named/default worksheet without declaring exhaustive scope or coverage. "
            "Save only decision-relevant changes, not every paragraph. Existing detailed mode: "
            "check a comparison worksheet against already-delivered exact sources. Declare product_ids and "
            "required_fields shared by all rows, plus required_cells for product-specific requirements "
            "(use required_fields=[] for mixed product types). Supply one fact per required cell, "
            "using the delivered source_ref or entry_id/revision. Quote is optional: if the literal value "
            "occurs in one complete source paragraph, the system copies that paragraph (up to 1600 "
            "characters). Ambiguous or oversized passages require an explicit exact quote. It never "
            "infers a value or repairs a supplied quote. Report missing cells and reject wrong product families, fabricated quotes or revisions. "
            "This checks source identity and literal support, NOT semantic entailment, units, applicability, "
            "or whether you declared every relevant product/field. Optional calculations sum coefficient * "
            "numeric fact value, using exact decimal arithmetic; you determine units, coefficients and "
            "applicable policies. Copy numeric values literally without currency/percent suffixes; "
            "valid thousands grouping such as '12,500.00' is accepted in calculations. "
            "Values must be literal substrings: quote 'no rewards' as text, not inferred 0. "
            "Put paraphrases, applicability judgments and inferred meaning in optional interpretation, "
            "not value or quote. Interpretations are retained model notes, not source-verified facts. "
            "Declare numeric_cells for required quantitative inputs; an unknown text value does not satisfy them. "
            f"Checks have {BUDGET_LIMITS[COMPARISON_TOOL][0]} calls per customer / "
            f"{BUDGET_LIMITS[COMPARISON_TOOL][1]} total, independent of discovery and reads. "
            "Keep conflicting facts unresolved using distinct stable fields for each source assertion, "
            "including conflicting passages within one document. A product/field key holds one "
            "assertion: updating it replaces the old value. Link competing assertions to their "
            "common disputed measure in interpretation; never submit duplicate cells. "
            "Successful checks retain one bounded worksheet across compaction, including partial results. "
            "Use stable worksheet_id values for independent comparison scopes; up to four named/default "
            "worksheets share 256000 retained bytes, at most 64000 per worksheet. Automatic context is separately bounded; a compact locator replaces worksheets too large for that context. Quotes are stored as exact source spans; oversized updates fail atomically and preserve the previous worksheet. Split unrelated categories into separate worksheets. "
            "For an existing worksheet send worksheet_id and changed facts only; omitted scope "
            "automatically reuses its saved facts. Send facts=[] to recheck it. "
            "If supplying both product_ids and required_fields, set reuse_checked_facts=true "
            "to merge, or omit it for a complete replacement. Reuse avoids resending scope, "
            "coverage or calculations; omitted scope fields are retained. Explicit scope lists may "
            "add products or required fields/cells but cannot remove existing requirements during reuse. "
            "All reused facts are revalidated. Supply a complete replacement to intentionally narrow scope. "
            "Changed facts or calculations without an explicit updated coverage assessment reopen "
            "the saved alternatives for review; previous conclusions are not silently reused. "
            "Does not complete the workflow or authorize actions.",
            {
                "coverage": {
                    "type": "object",
                    "description": "Omitted open_questions means coverage is unresolved, never complete. Keep the customer's objective, plausible alternatives and material unknowns across compaction. Include source-linked companions that could change the result, even if not currently held. Declaring an exclusion is the actor's interpretation, not automatically verified by this checker.",
                    "properties": {
                        "objective": {"type": "string", "minLength": 1, "maxLength": 600},
                        "alternatives": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 16,
                            "uniqueItems": True,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "product_ids": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 8,
                                        "uniqueItems": True,
                                        "items": text,
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": ["unresolved", "compared", "excluded"],
                                    },
                                    "reason": {"type": "string", "minLength": 1, "maxLength": 600},
                                },
                                "required": ["product_ids", "status", "reason"],
                                "additionalProperties": False,
                            },
                        },
                        "open_questions": {
                            "type": "array",
                            "maxItems": 16,
                            "uniqueItems": True,
                            "items": {"type": "string", "minLength": 1, "maxLength": 400},
                        },
                    },
                    "required": ["objective", "alternatives"],
                    "additionalProperties": False,
                },
                "passage_handles": {
                    "type": "array", "minItems": 0, "maxItems": 24,
                    "items": text,
                },
                "entry_ids": {"type": "array", "minItems": 1, "maxItems": 16, "items": text},
                "operation": {"enum": ["append", "replace", "read"]},
                "cursor": {"type": "string", "minLength": 1, "maxLength": 100},
                "capture": {
                    "type": "boolean",
                    "description": "Merge small source-backed findings now; scope derives from cells, not a completeness claim.",
                },
                "reuse_checked_facts": {"type": "boolean"},
                "worksheet_id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]{1,64}$"},
                "product_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 16,
                    "uniqueItems": True,
                    "items": text,
                },
                "required_fields": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 12,
                    "uniqueItems": True,
                    "items": text,
                },
                "facts": {"type": "array", "maxItems": 96, "items": fact},
                "required_cells": {
                    "type": "array",
                    "maxItems": 96,
                    "uniqueItems": True,
                    "items": {
                        "type": "object",
                        "properties": {"product_id": text, "field": text},
                        "required": ["product_id", "field"],
                        "additionalProperties": False,
                    },
                },
                "numeric_cells": {
                    "type": "array",
                    "maxItems": 96,
                    "uniqueItems": True,
                    "items": {
                        "type": "object",
                        "properties": {"product_id": text, "field": text},
                        "required": ["product_id", "field"],
                        "additionalProperties": False,
                    },
                },
                "calculations": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": text,
                            "terms": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 24,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "product_id": text,
                                        "field": text,
                                        "coefficient": {
                                            "type": "string",
                                            "pattern": f"^{_NUMBER}$",
                                        },
                                    },
                                    "required": ["product_id", "field", "coefficient"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["label", "terms"],
                        "additionalProperties": False,
                    },
                },
            },
            [],
            allOf=[{
                "if": {"anyOf": [{"required": ["passage_handles"]}, {"required": ["entry_ids"]}]},
                "then": {"not": {"anyOf": [{"required": [key]} for key in
                    ["facts", "capture", "product_ids", "required_fields", "required_cells",
                     "numeric_cells", "reuse_checked_facts", "calculations", "coverage"]]}},
                "else": {"required": ["facts"], "not": {"anyOf": [{"required": ["operation"]}, {"required": ["cursor"]}]}},
            }, {
                "if": {"required": ["entry_ids"]},
                "then": {"required": ["operation"], "properties": {"operation": {"const": "read"}},
                         "not": {"required": ["passage_handles"]}},
            }],
            anyOf=[
                {"required": ["passage_handles"]},
                {"required": ["entry_ids"]},
                {"required": ["capture"], "properties": {"capture": {"const": True}}},
                {"required": ["product_ids", "required_fields"]},
                {
                    "required": ["worksheet_id"],
                    "properties": {"reuse_checked_facts": {"const": True}},
                },
            ],
        ),
    )


class ProductNavigation:
    # Durable evidence capacity is separate from automatic prompt exposure.
    MAX_WORKSHEET_BYTES = 64_000
    MAX_ALL_WORKSHEETS_BYTES = 256_000
    MAX_COMPARISON_CONTEXT_BYTES = 36_000

    def __init__(self, corpus: BankingKnowledgeCorpus, state=None):
        # Runtime-owned mode, never restored from actor-controlled arguments.
        # Already delivered immutable sources are local evidence reuse, not search.
        self.allow_delivered_reuse = False
        self.corpus = corpus
        self.records = corpus.document_directory()
        self.by_id = {record["entry_id"]: record for record in self.records}
        self.by_handle = {}
        for record in self.records:
            identity = json.dumps([corpus.corpus_sha256, record["entry_id"], record["revision"]])
            record["source_handle"] = "src_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
            if record["source_handle"] in self.by_handle:
                raise ValueError("Duplicate source handle.")
            self.by_handle[record["source_handle"]] = record
        self.passages = {}
        self.document_passages = {}
        for record in self.records:
            text = corpus._entries[record["entry_id"]].text
            rows = []
            # Literal text boundaries only. No relevance or intent classification.
            boundaries = [0] + [m.end() for m in re.finditer(r"\n[ \t\r]*\n", text)] + [len(text)]
            for start, end in pairwise(boundaries):
                while start < end:
                    stop = min(start + 1600, end)
                    if stop < end:
                        cut = text.rfind(" ", start + 800, stop)
                        if cut > start:
                            stop = cut
                    if text[start:stop].strip():
                        handle = "pass_" + hashlib.sha256(
                            f"{record['source_handle']}:{start}:{stop}".encode()
                        ).hexdigest()[:20]
                        row = {"passage_handle": handle, "source_span": {"start": start, "end": stop}}
                        rows.append(row)
                        self.passages[handle] = {**record, **row, "text": text[start:stop]}
                    start = stop
            self.document_passages[record["entry_id"]] = rows
        self.products = {r["product_id"] for r in self.records}
        self.product_records: dict[str, list[dict[str, Any]]] = {}
        for record in self.records:
            self.product_records.setdefault(record["product_id"], []).append(record)
        self.categories = sorted({r["category"] for r in self.records})
        self.validators = {
            s["name"]: Draft202012Validator(s["input_schema"]) for s in tool_schemas()
        }
        self.state = self._restore(state)
        self._comparison_readiness: dict[str, bool] = {}

    def _restore(self, state):
        if state is None:
            return {
                "version": 6,
                "customer": None,
                "lanes": {name: {"turn": 0, "total": 0} for name in BUDGET_LIMITS},
                "products": [],
                "documents": [],
                "worksheet": None,
                "named_worksheets": {},
                "recovery": {"turn": 0, "total": 0},
            }
        # Historical snapshots cannot identify which tool consumed the shared
        # budget. Charge every lane conservatively, never replenish on migration.
        if type(state) is dict and set(state) == {"total", "turn", "customer"}:
            if (
                type(state["total"]) is not int
                or not 0 <= state["total"] <= 48
                or type(state["turn"]) is not int
                or not 0 <= state["turn"] <= min(12, state["total"])
            ):
                raise ValueError("Invalid legacy navigation budget state.")
            state = {
                "version": 2,
                "customer": state["customer"],
                "lanes": {
                    name: {
                        "turn": per_turn if state["turn"] == 12 else min(state["turn"], per_turn),
                        "total": total if state["total"] == 48 else min(state["total"], total),
                    }
                    for name, (per_turn, total) in BUDGET_LIMITS.items()
                },
            }
        if type(state) is dict and type(state.get("version")) is int and state["version"] == 2:
            if set(state) != {"version", "lanes", "customer"}:
                raise ValueError("Invalid version-2 navigation state.")
            state = {**state, "version": 3, "products": [], "documents": []}
        if type(state) is dict and type(state.get("version")) is int and state["version"] == 3:
            if set(state) != {"version", "lanes", "customer", "products", "documents"}:
                raise ValueError("Invalid version-3 navigation state.")
            state = {**state, "version": 4, "worksheet": None}
        if type(state) is dict and type(state.get("version")) is int and state["version"] == 4:
            if set(state) != {"version", "lanes", "customer", "products", "documents", "worksheet"}:
                raise ValueError("Invalid version-4 navigation state.")
            state = {**state, "version": 5, "named_worksheets": {}}
        if type(state) is dict and type(state.get("version")) is int and state["version"] == 5:
            if set(state) != {
                "version",
                "lanes",
                "customer",
                "products",
                "documents",
                "worksheet",
                "named_worksheets",
            }:
                raise ValueError("Invalid version-5 navigation state.")
            # Old snapshots have no recovery accounting. Never grant fresh
            # capacity by restoring an old-format session.
            state = {
                **state,
                "version": 6,
                "recovery": {"turn": RECOVERY_LIMITS[0], "total": RECOVERY_LIMITS[1]},
            }
        if (
            type(state) is not dict
            or set(state)
            != {
                "version",
                "lanes",
                "customer",
                "products",
                "documents",
                "worksheet",
                "named_worksheets",
                "recovery",
            }
            or type(state["version"]) is not int
            or state["version"] != 6
            or (state["customer"] is not None and not isinstance(state["customer"], str))
            or type(state["lanes"]) is not dict
            or set(state["lanes"]) != set(BUDGET_LIMITS)
        ):
            raise ValueError("Invalid navigation budget state.")
        for name, (per_turn, total) in {**BUDGET_LIMITS, RECOVERY_LANE: RECOVERY_LIMITS}.items():
            lane = state["recovery"] if name == RECOVERY_LANE else state["lanes"][name]
            if (
                type(lane) is not dict
                or set(lane) != {"turn", "total"}
                or type(lane["total"]) is not int
                or not 0 <= lane["total"] <= total
                or type(lane["turn"]) is not int
                or not 0 <= lane["turn"] <= min(per_turn, lane["total"])
            ):
                raise ValueError("Invalid navigation lane budget state.")
        for key, known in (("products", self.products), ("documents", self.by_id)):
            values = state[key]
            if (
                type(values) is not list
                or len(values) > MAX_RETAINED_LOCATORS
                or any(type(v) is not str or v not in known for v in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("Invalid retained navigation locators.")
        named = state["named_worksheets"]
        if (
            type(named) is not dict
            or len(named) > 4
            or len(named) + int(state["worksheet"] is not None) > 4
            or any(
                type(k) is not str
                or k == "default"
                or re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", k) is None
                for k in named
            )
        ):
            raise ValueError("Invalid named comparison worksheets.")
        state = deepcopy(state)
        worksheets = {**named, "default": state["worksheet"]}
        if len(json.dumps(worksheets, ensure_ascii=False).encode()) > self.MAX_ALL_WORKSHEETS_BYTES:
            raise ValueError("Retained comparison worksheets exceed total byte limit.")
        for name, worksheet in worksheets.items():
            if worksheet is None:
                if name != "default":
                    raise ValueError("Named comparison worksheet cannot be null.")
                continue
            if (
                type(worksheet) is not dict
                or set(worksheet) != {"corpus_sha256", "arguments"}
                or not isinstance(worksheet["corpus_sha256"], str)
                or len(json.dumps(worksheet, ensure_ascii=False).encode())
                > self.MAX_WORKSHEET_BYTES
                or not self.validators[COMPARISON_TOOL].is_valid(worksheet["arguments"])
                or not {"product_ids", "required_fields"} <= worksheet["arguments"].keys()
                or worksheet["arguments"].get("worksheet_id", "default") != name
            ):
                raise ValueError("Invalid retained comparison worksheet.")
            if worksheet["corpus_sha256"] != self.corpus.corpus_sha256:
                if name == "default":
                    state["worksheet"] = None
                else:
                    state["named_worksheets"].pop(name)
        return deepcopy(state)

    def _worksheet(self, name):
        return (
            self.state["worksheet"]
            if name == "default"
            else self.state["named_worksheets"].get(name)
        )

    def _save_worksheet(self, name, value):
        if name == "default":
            self.state["worksheet"] = value
        elif value is None:
            self.state["named_worksheets"].pop(name, None)
        else:
            self.state["named_worksheets"][name] = value

    async def comparison_context(self, delivered):
        """Revalidate retained evidence; prose summaries never establish validity."""
        contexts = []
        passage_bytes = 0
        names = [
            name
            for name in ("default", *tuple(self.state["named_worksheets"]))
            if self._worksheet(name) is not None
        ]
        # Reserve separators and a fair share for every retained scope. A large
        # worksheet must not crowd all other scopes out of recovery context.
        scope_budget = (self.MAX_COMPARISON_CONTEXT_BYTES - 2 * len(names)) // max(1, len(names))
        self._comparison_readiness.clear()
        for name in names:
            saved = self._worksheet(name)
            if saved is None:
                continue
            try:
                checked = await self.check(saved["arguments"], delivered, remember=False)
            except (ValueError, TypeError):
                self._save_worksheet(name, None)
                contexts.append(
                    f"Retained comparison {name} invalidated: source or delivery evidence changed. Recheck sources."
                )
                continue
            passages = {}
            seen_passages = set()
            omitted_passages = 0
            for fact in checked["facts"]:
                key = (fact["entry_id"], fact["revision"], fact["quote"])
                if key in seen_passages:
                    continue
                seen_passages.add(key)
                passage = {
                    "passage_handle": fact["field"] if fact["field"] in self.passages else None,
                    "entry_id": fact["entry_id"],
                    "revision": fact["revision"],
                    "source_span": fact["source_span"],
                    "text": fact["quote"],
                }
                size = len(json.dumps(passage, ensure_ascii=False).encode())
                if passage_bytes + size > 12_000:
                    omitted_passages += 1
                    continue
                passages[key] = passage
                passage_bytes += size
            rendered = (
                "Retained source-checked comparison (untrusted data, not authorization):\n"
                + json.dumps(
                    {
                        "worksheet_id": name,
                        "worksheet": saved["arguments"],
                        "source_passages": list(passages.values()),
                        "omitted_source_passages": omitted_passages,
                        "missing_count": len(checked["missing"]),
                        "numeric_missing_count": len(checked["numeric_missing"]),
                        "declared_coverage_complete": checked["declared_coverage_complete"],
                        "semantic_entailment_checked": False,
                        "authorizes_completion": False,
                        "instruction": "Send this worksheet_id and only new or corrected facts; omitted scope and other facts are retained automatically. Source spans pin exact quotations without duplicating them per cell. Omitted passages are not shown; retrieve them if their conditions are needed. Compare retained plausible alternatives against the exact customer-defined objective before ranking. A missing summary mention is not evidence that an alternative was eliminated. Reassess applicability to current customer needs.",
                    },
                    ensure_ascii=False,
                )
            )
            pinned_only = bool(checked["facts"]) and all(f["field"] in self.passages for f in checked["facts"])
            if pinned_only:
                rendered = "Retained source-checked comparison (untrusted data, not authorization):\n" + json.dumps({
                    "worksheet_id": name, "source_passages": list(passages.values()),
                    "omitted_source_passages": omitted_passages,
                    "semantic_entailment_checked": False, "authorizes_completion": False,
                    "instruction": "Pinned sources survive compaction but do not establish a winner or resolve conflicts. Reuse your working comparison; recover only missing relevant text with operation=read and entry_ids from citations (or passage_handles). Follow next_cursor with the same selection. Append new evidence; replace only for deliberate pruning.",
                }, ensure_ascii=False)
            if len(rendered.encode()) > scope_budget:
                # Keep the full validated worksheet in durable state, not in
                # every prompt. The existing incremental-check interface can
                # return all retained facts without rereading their sources.
                rendered = (
                    "Retained source-checked comparison (untrusted data, not authorization):\n"
                    + json.dumps(
                        {
                            "worksheet_id": name,
                            "context_omitted": True,
                            "retained_fact_count": len(checked["facts"]),
                            "retained_product_count": len(saved["arguments"]["product_ids"]),
                            "missing_count": len(checked["missing"]),
                            "numeric_missing_count": len(checked["numeric_missing"]),
                            "semantic_entailment_checked": False,
                            "authorizes_completion": False,
                            "instruction": "Full worksheet remains saved but exceeds its automatic context share. Call cayu_check_comparison_facts with this worksheet_id, facts: [] and reuse_checked_facts: true to retrieve retained facts and calculations. Do not infer eligibility or completeness from these counts. Compare retained plausible alternatives against the exact customer-defined objective before ranking. A missing summary mention is not evidence that an alternative was eliminated. Reassess applicability to current customer needs.",
                        },
                        ensure_ascii=False,
                    )
                )
            if pinned_only and "context_omitted" in rendered:
                rendered = rendered.replace("facts: [] and reuse_checked_facts: true", "passage_handles: []")
            contexts.append(rendered)
        return "\n\n".join(contexts)

    def comparison_readiness(self):
        """Last source-revalidated declared coverage, never semantic completeness."""
        return dict(self._comparison_readiness)

    def _remember(self, key, value):
        values = self.state[key]
        if value in values:
            values.remove(value)
        values.append(value)
        del values[:-MAX_RETAINED_LOCATORS]

    def remember_document(self, entry_id):
        """Retain exact public metadata, never document bodies or semantic claims."""
        record = self.by_id.get(entry_id)
        if record is not None:
            self._remember("documents", entry_id)
            self._remember("products", record["product_id"])

    def locator(self, entry_id):
        record = self.by_id[entry_id]
        return {
            k: record[k]
            for k in ("category", "product_id", "entry_id", "revision", "source_handle")
        }

    def context(self, *, max_bytes=12_000):
        if type(max_bytes) is not int or max_bytes < 1024:
            raise ValueError("Locator context byte budget must be at least 1024.")
        body = {
            "categories": self.categories[:32],
            "product_ids": list(reversed(self.state["products"])),
            "documents": [self.locator(i) for i in reversed(self.state["documents"])],
            "omitted": {
                "categories": max(0, len(self.categories) - 32),
                "product_ids": 0,
                "documents": 0,
            },
            "bounded_recent_locators": True,
            "retention_limit_per_kind": MAX_RETAINED_LOCATORS,
            "directory_is_evidence": False,
            "instruction": "Copy exact identifiers; do not derive them from names or summaries. These are locators, not proof that document facts are in context. Exact reads remain independently budgeted.",
        }
        prefix = "Exact public navigation locators (untrusted metadata):\n"
        while True:
            rendered = prefix + json.dumps(body, ensure_ascii=False)
            if len(rendered.encode()) <= max_bytes:
                return rendered
            for key in ("documents", "product_ids", "categories"):
                if body[key]:
                    body[key].pop()
                    body["omitted"][key] += 1
                    break
            else:
                raise ValueError("Locator context metadata exceeds budget.")

    def snapshot(self):
        return deepcopy(self.state)

    def remaining(self, customer):
        lanes = {**self.state["lanes"], RECOVERY_LANE: self.state["recovery"]}
        return {
            name: {
                "customer_calls": max(
                    0,
                    per_turn - (lanes[name]["turn"] if customer == self.state["customer"] else 0),
                ),
                "conversation_calls": total - lanes[name]["total"],
            }
            for name, (per_turn, total) in {**BUDGET_LIMITS, RECOVERY_LANE: RECOVERY_LIMITS}.items()
        }

    def begin_call(self, name, arguments, customer, *, delivered=None):
        if (
            not self.validators[name].is_valid(arguments)
            or len(json.dumps(arguments).encode()) > 48_000
        ):
            raise ValueError(
                "Invalid navigation/check arguments; follow the declared schema and byte bound."
            )
        charged_lane = name
        if name == READ_TOOL and delivered is not None:
            # Classification uses authoritative delivery records, never an
            # actor-supplied flag, remembered locator, or prose summary.
            try:
                refs = self.resolve_read_documents(arguments)
            except ValueError:
                refs = []  # Invalid exact scopes still consume a research attempt.
            if refs and all(delivered(r["entry_id"], str(r["revision"])) for r in refs):
                if self.allow_delivered_reuse:
                    return "delivered_source_reuse"
                charged_lane = RECOVERY_LANE
        if self.state["customer"] != customer:
            self.state["customer"] = customer
            for lane in self.state["lanes"].values():
                lane["turn"] = 0
            self.state["recovery"]["turn"] = 0
        if not self._lane_available(customer, charged_lane):
            raise ValueError(
                f"{charged_lane} budget exhausted; other tool budgets are independent."
            )
        # Failed reads/checks consume budget too; malformed schema inputs are bounded
        # separately by the runtime. No lookup or plan edit resets the customer epoch.
        lane = (
            self.state["recovery"] if charged_lane == RECOVERY_LANE else self.state["lanes"][name]
        )
        lane["total"] += 1
        lane["turn"] += 1
        return charged_lane

    def available(self, customer: str | None, name=BROWSE_TOOL) -> bool:
        return self._lane_available(customer, name) or (
            name == READ_TOOL
            and (self.allow_delivered_reuse or self._lane_available(customer, RECOVERY_LANE))
        )

    def _lane_available(self, customer, name):
        lane = self.state["recovery"] if name == RECOVERY_LANE else self.state["lanes"][name]
        turn = lane["turn"] if self.state["customer"] == customer else 0
        per_turn, total = RECOVERY_LIMITS if name == RECOVERY_LANE else BUDGET_LIMITS[name]
        return lane["total"] < total and turn < per_turn

    def browse(self, arguments: dict[str, Any]) -> dict[str, Any]:
        category, product = arguments.get("category"), arguments.get("product_id")
        products = arguments.get("product_ids")
        if product is not None and products is not None:
            raise ValueError("Use product_id or product_ids, not both.")
        offset = arguments.get("offset", 0)
        records = self.records
        if category is not None:
            records = tuple(r for r in records if r["category"] == category)
            if not records:
                raise ValueError(
                    "Unknown category: "
                    + json.dumps(category)
                    + "; exact categories: "
                    + json.dumps(self.categories[:32])
                )
        if product is not None or products is not None:
            selected = set(products) if products is not None else {product}
            rows = [dict(r) for r in records if r["product_id"] in selected]
            if {r["product_id"] for r in rows} != selected:
                missing = sorted(selected - {r["product_id"] for r in rows})
                raise ValueError(
                    "Unknown product in the selected category: "
                    + json.dumps(missing)
                    + "; copy exact product_ids from retained locator context or the directory, not display names."
                )
        else:
            key = "category" if category is None else "product_id"
            values = sorted({r[key] for r in records})
            rows = [
                {key: value, "document_count": sum(r[key] == value for r in records)}
                for value in values
            ]
        if offset and offset >= len(rows):
            raise ValueError("Offset is outside this directory listing.")
        for row in rows[offset : offset + 32]:
            if "entry_id" in row:
                self.remember_document(row["entry_id"])
            elif "product_id" in row:
                self._remember("products", row["product_id"])
        return {
            "corpus_sha256": self.corpus.corpus_sha256,
            "entries": rows[offset : offset + 32],
            "total": len(rows),
            "next_offset": offset + 32 if offset + 32 < len(rows) else None,
            "directory_is_evidence": False,
        }

    def resolve_read_documents(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Resolve a bounded exact scope without inferring relevance from customer text.

        The caller still reads every revision through the access-controlled store
        and checks the complete serialized byte size before publishing any evidence.
        """
        if not self.validators[READ_TOOL].is_valid(arguments):
            raise ValueError("Supply documents or one exact product_id, not both.")
        if "documents" in arguments:
            resolved = []
            for item in arguments["documents"]:
                record = (
                    self.by_handle.get(item["source_handle"])
                    if "source_handle" in item
                    else self.by_id.get(item["entry_id"])
                )
                if record is None:
                    raise ValueError(
                        "Unknown source; copy source_handle or entry_id from the directory."
                    )
                resolved.append(
                    {
                        "entry_id": record["entry_id"],
                        "revision": item.get("revision", record["revision"]),
                    }
                )
            return resolved
        records = self.product_records.get(arguments["product_id"])
        if records is None:
            raise ValueError("Unknown product_id; copy its exact identifier from the directory.")
        if len(records) > MAX_PRODUCT_DOCUMENTS:
            raise ValueError(
                f"Product has {len(records)} documents, exceeding the {MAX_PRODUCT_DOCUMENTS}-document "
                "packet bound. Browse its directory and use exact-document batches. "
                "No document from this call was delivered."
            )
        return [{"entry_id": r["entry_id"], "revision": r["revision"]} for r in records]

    def resolve_knowledge_reference(self, reference):
        """Canonicalize a returned handle without granting source delivery."""
        reference = deepcopy(reference)
        if reference.get("kind") != "knowledge_revision":
            return reference
        record = self.by_handle.get(reference["reference_id"])
        if record is None:
            return reference  # Ordinary IDs still undergo the caller's delivery check.
        version = str(record["revision"])
        if reference.get("version", version) != version:
            raise ValueError("Source handle and explicit knowledge revision disagree.")
        return {
            "kind": "knowledge_revision",
            "reference_id": record["entry_id"],
            "version": version,
        }

    def normalize_comparison(self, arguments):
        """Resolve exact handles and expand an optional small evidence patch, without inference."""
        if not self.validators[COMPARISON_TOOL].is_valid(arguments):
            raise ValueError("Invalid comparison worksheet arguments.")
        arguments = deepcopy(arguments)
        if "passage_handles" in arguments:
            facts = []
            for handle in arguments.pop("passage_handles"):
                passage = self.passages.get(handle)
                if passage is None:
                    raise ValueError("Unknown passage handle; copy a locator returned by a document read.")
                facts.append({
                    "source_handle": passage["source_handle"], "field": handle,
                    "value": passage["text"].split()[0][:200],
                    "source_span": passage["source_span"],
                    "interpretation": "Pinned source passage; no scalar value, relevance or conclusion inferred.",
                })
            arguments.update(capture=True, facts=facts)
        for fact in arguments["facts"]:
            if "source_handle" not in fact:
                continue
            record = self.by_handle.get(fact.pop("source_handle"))
            if record is None:
                raise ValueError("Unknown source handle; copy a returned handle.")
            if fact.get("product_id", record["product_id"]) != record["product_id"]:
                raise ValueError("Source handle and product_id disagree.")
            fact.update({k: record[k] for k in ("product_id", "entry_id", "revision")})
        if arguments.pop("capture", False):
            if any(
                k in arguments
                for k in (
                    "product_ids",
                    "required_fields",
                    "required_cells",
                    "numeric_cells",
                    "reuse_checked_facts",
                    "calculations",
                    "coverage",
                )
            ):
                raise ValueError(
                    "Capture accepts worksheet_id and facts only; use detailed mode to assess coverage."
                )
            saved = self._worksheet(arguments.get("worksheet_id", "default"))
            previous = saved["arguments"] if saved else {}
            products = set(previous.get("product_ids", []))
            products.update(f["product_id"] for f in arguments["facts"])
            cells = {(c["product_id"], c["field"]) for c in previous.get("required_cells", [])}
            cells.update((f["product_id"], f["field"]) for f in arguments["facts"])
            arguments.update(
                product_ids=sorted(products),
                required_fields=previous.get("required_fields", []),
                required_cells=[{"product_id": p, "field": f} for p, f in sorted(cells)],
                reuse_checked_facts=bool(saved),
            )
        return arguments

    async def check(self, arguments: dict[str, Any], delivered, *, remember=True) -> dict[str, Any]:
        arguments = self.normalize_comparison(arguments)
        if not self.validators[COMPARISON_TOOL].is_valid(arguments):
            raise ValueError("Invalid comparison worksheet arguments.")
        name = arguments.get("worksheet_id", "default")
        coverage_supplied = "coverage" in arguments
        previous = None
        reuse = arguments.pop(
            "reuse_checked_facts",
            "worksheet_id" in arguments
            and not {"product_ids", "required_fields"} <= arguments.keys(),
        )
        if reuse:
            saved = self._worksheet(name)
            if saved is None or saved["corpus_sha256"] != self.corpus.corpus_sha256:
                raise ValueError("No retained worksheet is available; supply source-backed facts.")
            previous = saved["arguments"]
            for key in ("coverage", "calculations"):
                if key not in arguments and key in previous:
                    arguments[key] = deepcopy(previous[key])
            for key in ("product_ids", "required_fields", "required_cells", "numeric_cells"):
                if key not in arguments and key in previous:
                    arguments[key] = deepcopy(previous[key])
                old_scope = {json.dumps(value, sort_keys=True) for value in previous.get(key, [])}
                new_scope = {json.dumps(value, sort_keys=True) for value in arguments.get(key, [])}
                if not old_scope <= new_scope:
                    raise ValueError(
                        "Reuse cannot remove declared comparison requirements; supply a complete replacement for narrowed scope."
                    )
            updates = {(f["product_id"], f["field"]) for f in arguments["facts"]}
            arguments["facts"] = [
                deepcopy(f)
                for f in previous["facts"]
                if (f["product_id"], f["field"]) not in updates
            ] + arguments["facts"]
        if not self.validators[COMPARISON_TOOL].is_valid(arguments):
            raise ValueError("Invalid comparison worksheet arguments.")
        for fact in arguments["facts"]:
            if ref := fact.pop("source_ref", None):
                fact["entry_id"] = ref["reference_id"]
                fact["revision"] = int(ref["version"])
        products, fields = arguments["product_ids"], arguments["required_fields"]
        coverage = arguments.get("coverage")
        if coverage is not None:
            # Omitted metadata must not discard valid source cells or mean that
            # no uncertainty exists. Preserve an explicit unresolved assessment.
            if "open_questions" not in coverage:
                coverage["open_questions"] = [
                    "Coverage was not assessed: explicitly state remaining open questions "
                    "or confirm that none remain."
                ]
            alternatives = coverage["alternatives"]
            # A product set can have different configurations, quantities or
            # eligibility assumptions. Only identical declarations are duplicate;
            # interpretation of the stated distinction remains the actor's job.
            if any(not set(a["product_ids"]) <= self.products for a in alternatives):
                raise ValueError(
                    "Unknown product in comparison coverage; use exact directory identifiers."
                )
            if any(
                a["status"] == "compared" and not set(a["product_ids"]) <= set(products)
                for a in alternatives
            ):
                raise ValueError(
                    "Compared alternatives require declared source-fact rows for every component."
                )
            if (
                not coverage["objective"].strip()
                or any(not a["reason"].strip() for a in alternatives)
                or any(not q.strip() for q in coverage["open_questions"])
            ):
                raise ValueError("Comparison objective, reasons and questions must be nonblank.")
            if (
                any(a["status"] == "unresolved" for a in alternatives)
                and not coverage["open_questions"]
            ):
                raise ValueError("Unresolved alternatives require an explicit open question.")
        if not set(products) <= self.products:
            # Recover from exact source identities, never fuzzy-match product names
            # or silently correct the proposed comparison. The actor must resubmit.
            source_ids = dict.fromkeys(f["entry_id"] for f in arguments["facts"])
            recovery = [self.locator(i) for i in source_ids if i in self.by_id][:16]
            raise ValueError(
                "Unknown product identifiers: "
                + json.dumps(sorted(set(products) - self.products))
                + "; exact source locators (metadata only, not validated facts): "
                + json.dumps(recovery)
                + "; resubmit with exact product_id values. No facts were validated."
            )
        required = dict.fromkeys((p, f) for p in products for f in fields)
        for cell in arguments.get("required_cells", []):
            if cell["product_id"] not in products:
                raise ValueError("Required cell belongs to an undeclared product.")
            required[(cell["product_id"], cell["field"])] = None
        if not required or any(not any(key[0] == p for key in required) for p in products):
            raise ValueError("Declare required cells for every selected product.")
        numeric = {(c["product_id"], c["field"]) for c in arguments.get("numeric_cells", [])}
        if not numeric <= required.keys():
            raise ValueError("Numeric cells must also be declared required cells.")
        facts = {}
        for fact in arguments["facts"]:
            key = (fact["product_id"], fact["field"])
            if key in facts:
                raise ValueError(f"Duplicate comparison cell: {key}; supply one update per cell.")
            if key not in required:
                raise ValueError(
                    "Undeclared comparison cell: "
                    + json.dumps(key)
                    + "; declared fields for this product: "
                    + json.dumps([f for p, f in required if p == key[0]])
                    + ". Copy the existing field ID or add the new field to required_fields/required_cells in this patch."
                )
            record = self.by_id.get(fact["entry_id"])
            if record is None or record["product_id"] != key[0]:
                raise ValueError("Fact source belongs to a different product.")
            if not delivered(fact["entry_id"], str(fact["revision"])):
                raise ValueError("Read the exact source revision before citing it.")
            source = await self.corpus.read_document(fact["entry_id"], fact["revision"])
            if "interpretation" in fact and not fact["interpretation"].strip():
                raise ValueError("Interpretation must be nonblank if supplied.")
            pinned = self.passages.get(fact["field"])
            exact_pin = pinned is not None and all((
                fact["entry_id"] == pinned["entry_id"],
                str(fact["revision"]) == str(pinned["revision"]),
                fact.get("source_span") == pinned["source_span"],
                fact["value"] == pinned["text"].split()[0][:200],
            ))
            if "source_span" in fact:
                start, end = fact["source_span"]["start"], fact["source_span"]["end"]
                if not (0 <= start < end <= len(source["text"]) and (1 if exact_pin else 8) <= end - start <= 1600):
                    raise ValueError("Invalid exact source span; copy a returned span or quote.")
                fact["quote"] = source["text"][start:end]
            elif "quote" not in fact:
                try:
                    fact["quote"] = _unique_source_paragraph(source["text"], fact["value"])
                except ValueError as error:
                    raise _source_repair_error(fact, source, str(error)) from None
            literal = re.search(
                r"(?<![\w.])(?<![0-9],)" + re.escape(fact["value"]) + r"(?!\w|[.,][0-9])",
                fact["quote"],
            )
            if fact["quote"] not in source["text"] or (literal is None and not exact_pin):
                raise _source_repair_error(
                    fact,
                    source,
                    "Value and quote must occur literally in the pinned source; "
                    "copy the source wording, not a paraphrase or inferred numeric value.",
                )
            start = fact["source_span"]["start"] if "source_span" in fact else source["text"].find(fact["quote"])
            fact["source_span"] = {"start": start, "end": start + len(fact["quote"])}
            facts[key] = {**fact, "evidence_ref": source["evidence_ref"]}
            # Storage uses the immutable revision and exact span. Response facts
            # still contain the full quote; shared context deduplicates passages.
            fact.pop("quote")
        if previous is not None and coverage is not None and not coverage_supplied:
            old_facts = {(f["product_id"], f["field"]): f for f in previous["facts"]}
            new_facts = {(f["product_id"], f["field"]): f for f in arguments["facts"]}
            if old_facts != new_facts or previous.get("calculations", []) != arguments.get(
                "calculations", []
            ):
                # Invalidate only the conclusion, not the source facts. Semantic
                # reassessment belongs to the actor; no keyword classification.
                for alternative in coverage["alternatives"]:
                    alternative["status"] = "unresolved"
                question = (
                    "Reassess alternatives because source facts or calculation inputs changed."
                )
                if (
                    question not in coverage["open_questions"]
                    and len(coverage["open_questions"]) < 16
                ):
                    coverage["open_questions"].append(question)
        missing = [{"product_id": p, "field": f} for p, f in required if (p, f) not in facts]
        numeric_missing = [
            {"product_id": p, "field": f}
            for p, f in sorted(numeric)
            if (p, f) not in facts or _fact_number(facts[(p, f)]["value"]) is None
        ]
        calculations = []
        for calculation in arguments.get("calculations", []):
            with localcontext() as context:
                context.prec = 64
                total = Decimal(0)
                for term in calculation["terms"]:
                    fact = facts.get((term["product_id"], term["field"]))
                    number = None if fact is None else _fact_number(fact["value"])
                    if number is None:
                        raise ValueError(
                            f"Calculation {calculation['label']!r} requires a checked numeric fact "
                            f"for {term['product_id']}/{term['field']}; supply that fact in this "
                            "worksheet or reuse its retained facts with identical scope."
                        )
                    total += Decimal(term["coefficient"]) * number
                calculations.append({"label": calculation["label"], "value": format(total, "f")})
        body = {
            "facts": list(facts.values()),
            "missing": missing,
            "declared_cells_complete": not missing,
            "numeric_missing": numeric_missing,
            "declared_inputs_complete": not missing and not numeric_missing,
            "semantic_entailment_checked": False,
            "exhaustive_product_coverage_checked": False,
            "calculation_results": calculations,
            "arithmetic_checked": bool(calculations),
            "authorizes_completion": False,
            "coverage": coverage,
            "declared_coverage_complete": coverage is not None
            and not coverage["open_questions"]
            and all(a["status"] != "unresolved" for a in coverage["alternatives"]),
        }
        body["worksheet_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()
        ).hexdigest()
        if remember:
            saved = {"corpus_sha256": self.corpus.corpus_sha256, "arguments": arguments}
            retained = (
                len(json.dumps(saved, ensure_ascii=False).encode()) <= self.MAX_WORKSHEET_BYTES
            )
            proposed = {
                **self.state["named_worksheets"],
                "default": self.state["worksheet"],
                name: saved,
            }
            retained = (
                retained
                and sum(value is not None for value in proposed.values()) <= 4
                and len(json.dumps(proposed, ensure_ascii=False).encode())
                <= self.MAX_ALL_WORKSHEETS_BYTES
            )
            if not retained:
                raise ValueError(
                    "Comparison update exceeds retained evidence capacity; previous worksheet is unchanged. "
                    "Keep only decision-relevant facts/notes or split independent scopes. "
                    f"Limit per worksheet: {self.MAX_WORKSHEET_BYTES} bytes; total: {self.MAX_ALL_WORKSHEETS_BYTES} bytes."
                )
            self._save_worksheet(name, deepcopy(saved))
            body["retained_for_compaction"] = retained
            body["worksheet_id"] = name
            body["total_retention_byte_limit"] = self.MAX_ALL_WORKSHEETS_BYTES
            body["retention_byte_limit"] = self.MAX_WORKSHEET_BYTES
        self._comparison_readiness[name] = bool(
            body["declared_inputs_complete"] and body["declared_coverage_complete"]
        )
        return body


class SourceRepairError(ValueError):
    """Carry exact repair evidence without parsing prose or inferring meaning."""

    def __init__(self, message, repair):
        super().__init__(message)
        self.repair = repair


def _source_repair_error(fact, source, reason):
    """Return bounded, already-delivered evidence without granting validation."""
    position = source["text"].find(fact["value"])
    start = max(0, position - 400)
    excerpt = source["text"][start : start + 1600]
    recovery = {
        "entry_id": fact["entry_id"],
        "revision": fact["revision"],
        "source_excerpt": excerpt,
        "excerpt_complete": start == 0 and len(excerpt) == len(source["text"]),
        "excerpt_start": start,
        "omitted_characters": len(source["text"]) - len(excerpt),
        "evidence_ref": source["evidence_ref"],
        "facts_accepted": False,
    }
    spans = _matching_paragraph_spans(source["text"], fact["value"])
    candidates = [
        {"source_span": {"start": start, "end": end}, "text": source["text"][start:end]}
        for start, end in spans
        if 8 <= end - start <= 1600
    ]
    repair = {
        "entry_id": fact["entry_id"],
        "revision": fact["revision"],
        "passages": candidates[:3],
        "omitted_matching_paragraphs": len(spans) - min(3, len(candidates)),
        "facts_accepted": False,
        "instruction": "Select the passage supporting this field, including its conditions, and copy its source_span in a patch. No passage is selected automatically. If none supports the claim, correct the literal value or leave it unresolved.",
    }
    return SourceRepairError(
        f"Cell {fact['product_id']}/{fact['field']}: {reason} "
        "Keep inferred meaning in interpretation, not in a literal value or quote. "
        "Already-delivered source for repair (not a validated cell): "
        + json.dumps(recovery, ensure_ascii=False),
        repair,
    )


def _unique_source_paragraph(text: str, value: str) -> str:
    """Copy source text, never infer semantics or fabricate an exact quotation.

    Paragraph boundaries retain adjacent conditions/table rows. We deliberately
    refuse long or ambiguous passages instead of choosing the first occurrence
    or truncating away qualifications. Explicit quotations remain supported.
    """
    spans = _matching_paragraph_spans(text, value)
    if not spans:
        raise ValueError(
            "Literal value must occur in one complete paragraph for automatic quotation; "
            "supply an exact quote from the intended source passage. No facts were accepted."
        )
    if len(spans) != 1:
        raise ValueError(
            "Literal value occurs in different source paragraphs; select an exact quote."
        )
    start, end = next(iter(spans))
    quote = text[start:end]
    if not 8 <= len(quote) <= 1600:
        raise ValueError(
            "The complete source paragraph does not fit the quote bound; "
            "supply an exact bounded quote, keeping applicable conditions visible."
        )
    return quote


def _matching_paragraph_spans(text: str, value: str):
    """Literal boundary matching only; never classify intent or relevance."""
    pattern = r"(?<![\w.])(?<![0-9],)" + re.escape(value) + r"(?!\w|[.,][0-9])"
    matches = list(re.finditer(pattern, text))
    boundaries = list(re.finditer(r"\n[ \t\r]*\n", text))
    spans = {
        (
            max((b.end() for b in boundaries if b.end() <= match.start()), default=0),
            min((b.start() for b in boundaries if b.start() >= match.end()), default=len(text)),
        )
        for match in matches
    }
    return sorted(spans)
