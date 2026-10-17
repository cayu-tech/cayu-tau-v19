"""Diagnostic-only delivery of frozen, complete public policy documents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class FocusedPolicyBundle:
    def __init__(self, path: Path):
        raw = path.read_bytes()
        payload = json.loads(raw)
        if set(payload) != {"documents"} or not isinstance(payload["documents"], list):
            raise ValueError("Focused bundle requires only a documents list")
        self.documents = payload["documents"]
        if not 1 <= len(self.documents) <= 32:
            raise ValueError("Focused bundle requires 1-32 documents")
        ids = set()
        for row in self.documents:
            if not isinstance(row, dict) or set(row) != {"entry_id", "revision", "content_sha256"}:
                raise ValueError("Focused bundle must contain only source locators and hashes")
            if not isinstance(row["entry_id"], str) or row["entry_id"] in ids:
                raise ValueError("Focused document ids must be unique strings")
            if type(row["revision"]) is not int or row["revision"] < 1:
                raise ValueError("Focused document revision must be a positive integer")
            ids.add(row["entry_id"])
        self.digest = hashlib.sha256(raw).hexdigest()
        self.text = None
        self.references = ()

    async def prepare(self, corpus):
        if self.text is not None:
            return
        documents, references = [], []
        for row in self.documents:
            doc = await corpus.read_document(row["entry_id"], row["revision"])
            if doc["content_sha256"] != row["content_sha256"] or not doc["text_complete"]:
                raise ValueError("Focused policy does not match the frozen source hash")
            ref = corpus.delivered_revision_evidence(
                row["entry_id"], row["revision"], delivered_text=doc["text"]
            )
            if ref is None:
                raise ValueError("Focused policy lacks complete revision evidence")
            documents.append(doc)
            references.append(ref)
        text = json.dumps(documents, ensure_ascii=False)
        if len(text.encode()) > 100_000:
            raise ValueError("Focused policy bundle exceeds 100 KB")
        self.text = (
            "<focused_public_policies>\n"
            "Complete public sources selected for this diagnostic. They are policy evidence, "
            "not customer facts, authorization, or expected actions. Cite their knowledge "
            "revision references directly. Use available search/read tools only when a needed "
            "rule or exception is missing. Ask the customer for preferences and authorization.\n"
            + text
            + "\n</focused_public_policies>"
        )
        self.references = tuple(references)
