"""Bounded exact-revision evidence for unfinished, model-selected obligations.

This is application context retention, not another search or semantic reviewer.
Only previously delivered references accepted by the controller can enter the pack.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from cayu import WorkEvidenceReference

from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.ledger import ToolActionStatus

if TYPE_CHECKING:
    from cayu_tau3_banking.controller import BankingWorkflowController
    from cayu_tau3_banking.ledger import ConversationLedger


def recent_tool_documentation_references(
    corpus: BankingKnowledgeCorpus,
    ledger: ConversationLedger,
    controller: BankingWorkflowController,
    *,
    max_tools: int = 4,
    documents_per_tool: int = 2,
) -> tuple[WorkEvidenceReference, ...]:
    """Round-robin previously delivered docs for bounded recent successful tools.

    Selection uses exact public capability identities, not language classification.
    No lookup execution, new retrieval, evidence registration or plan mutation occurs.
    """
    if (
        type(max_tools) is not int
        or type(documents_per_tool) is not int
        or not (1 <= max_tools <= 4 and 1 <= documents_per_tool <= 2)
    ):
        raise ValueError("Tool-document retention bounds exceed the supported maximum")
    names = set()
    groups = []
    for action in reversed(ledger.actions):
        if action.status is not ToolActionStatus.SUCCEEDED:
            continue
        name = action.name
        if name == "call_discoverable_agent_tool":
            name = action.arguments.get("agent_tool_name")
        if not isinstance(name, str) or name in names:
            continue
        names.add(name)
        candidates = []
        for entry_id, revision in corpus.capability_document_locators(name):
            reference = controller.delivered_knowledge_reference(entry_id, str(revision))
            if reference is not None:
                candidates.append(reference)
        focused = corpus.focused_capability_document_ids(name)
        focused_candidates = [ref for ref in candidates if ref.reference_id in focused]
        # Prefer already-delivered focused definitions to unrelated workflows
        # that merely use this tool. If none were delivered, retain the bounded
        # general fallback. This does not claim exhaustive procedural coverage.
        candidates = (focused_candidates or candidates)[:documents_per_tool]
        if candidates:
            groups.append(candidates)
            if len(groups) == max_tools:
                break
    references = {}
    for index in range(documents_per_tool):
        for group in groups:
            if index < len(group):
                ref = group[index]
                references.setdefault((ref.reference_id, ref.version), ref)
    return tuple(references.values())


async def active_evidence_context(
    corpus: BankingKnowledgeCorpus,
    references: tuple[WorkEvidenceReference, ...],
    *,
    max_documents: int = 4,
    max_bytes: int = 24_000,
    purpose: str = "work",
) -> str | None:
    """Reload immutable public revisions under the corpus access scope.

    Whole records only: never silently turn a truncated procedure into full
    evidence. Omissions are visible and do not change checklist completion.
    """
    if max_documents < 1 or max_bytes < 1:
        raise ValueError("Active evidence bounds must be positive.")
    if purpose not in {"work", "tool_contracts"}:
        raise ValueError("Unknown evidence retention purpose")
    documents = []
    omitted = []
    seen = set()
    used = 0
    for reference in references:
        if reference.kind != "knowledge_revision":
            raise ValueError("Active context accepts only knowledge revisions.")
        identity = (reference.reference_id, reference.version)
        if identity in seen:
            continue
        seen.add(identity)
        locator = {"entry_id": reference.reference_id, "entry_revision": reference.version}
        if len(documents) >= max_documents:
            omitted.append({**locator, "reason": "document_limit"})
            continue
        # The corpus is pinned, but reading still uses Cayu's access boundary.
        entry = await corpus.store.get_entry(
            reference.reference_id,
            revision=int(reference.version),
            access_scope=corpus.scope,
        )
        if entry is None:
            omitted.append({**locator, "reason": "revision_unavailable"})
            continue
        canonical = corpus.delivered_revision_evidence(
            entry.id, entry.revision, delivered_text=entry.text
        )
        if canonical is None or (
            reference.digest is not None and reference.digest != canonical.digest
        ):
            raise ValueError("Active evidence does not match its accepted revision.")
        document = {
            "entry_id": entry.id,
            "entry_revision": entry.revision,
            "title": entry.title,
            "text": entry.text,
            "text_complete": True,
            "evidence_ref": canonical.model_dump(
                mode="json", include={"kind", "reference_id", "version"}
            ),
        }
        size = len(json.dumps(document, ensure_ascii=False).encode("utf-8"))
        if used + size > max_bytes:
            omitted.append({**locator, "reason": "byte_limit"})
            continue
        documents.append(document)
        used += size
    if not documents and not omitted:
        return None
    if purpose == "tool_contracts":
        return (
            "Previously delivered public documentation for recently successful tools. "
            "Retained with their results so lookup scope and policy conditions survive compaction. "
            "Check what each result covers before inferring presence or absence; a tool name or "
            "earlier summary is not its contract. This is a bounded selection, not all tool "
            "documentation. Omitted documents are not supplied. This does not establish eligibility, "
            "authorize new effects, or complete work.\n<active-tool-documentation>\n"
            + json.dumps({"documents": documents, "omitted": omitted}, ensure_ascii=False)
            + "\n</active-tool-documentation>"
        )
    return (
        "Previously delivered public evidence cited by unfinished work. These exact "
        "revisions remain available here even when automatic recall changes focus. "
        "A checklist citation alone is not policy text or proof of eligibility. "
        "Use the actual terms and conditional sequence; do not invent amounts, "
        "conditions or permission from a plan or earlier assistant claim. "
        "Omitted references are NOT supplied here; retrieve missing terms before "
        "relying on them. This context does not complete any obligation.\n"
        "<active-work-evidence>\n"
        + json.dumps({"documents": documents, "omitted": omitted}, ensure_ascii=False)
        + "\n</active-work-evidence>"
    )
