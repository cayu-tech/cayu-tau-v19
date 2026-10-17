"""Experimental heading-based embedding projections; not enabled by the agent.

Split only at Markdown ATX headings outside fenced code. Preserve each section
verbatim, including tables, and carry the ancestor headings into its projection.
This is deliberately not a general Markdown parser or a token-budget splitter.
"""

import hashlib
import re

from cayu import KnowledgeChunk, KnowledgeEntry

VERSION = "cayu.tau3.section-embedding-experiment.v1"
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+\S")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def section_embedding_chunks(entry: KnowledgeEntry) -> list[KnowledgeChunk]:
    """Return deterministic projections with exact body offsets and provenance."""
    sections = []
    ancestors: list[tuple[int, str]] = []
    prefix: tuple[str, ...] = ()
    start = 0
    offset = 0
    fence = None
    for line in entry.text.splitlines(keepends=True):
        marker = _FENCE.match(line)
        if fence is not None:
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not line[marker.end() :].strip()
            ):
                fence = None
        elif marker:
            fence = marker[1]
        else:
            heading = _HEADING.match(line)
            if heading:
                if offset > start:
                    sections.append((start, offset, prefix))
                level = len(heading[1])
                while ancestors and ancestors[-1][0] >= level:
                    ancestors.pop()
                prefix = tuple(text for _, text in ancestors)
                ancestors.append((level, line.rstrip("\r\n")))
                start = offset
        offset += len(line)
    if offset > start:
        sections.append((start, offset, prefix))
    chunks = []
    for start, end, prefix in sections:
        body = entry.text[start:end]
        if not body.strip():
            continue
        text = "\n\n".join(part for part in (entry.title, *prefix, body) if part)
        index = len(chunks)
        chunks.append(
            KnowledgeChunk(
                id=f"{entry.id}:r{entry.revision}:section:{index}",
                entry_id=entry.id,
                entry_revision=entry.revision,
                text=text,
                chunk_index=index,
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
                source_uri=entry.source_uri,
                metadata={
                    "embedding_representation": VERSION,
                    "body_start_char": start,
                    "body_end_char": end,
                    "ancestor_headings": list(prefix),
                },
            )
        )
    if not chunks:
        raise ValueError("Cannot embed a document with an empty body.")
    return chunks
