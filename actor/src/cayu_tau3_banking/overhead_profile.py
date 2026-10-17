"""Read-only phase accounting for archived runs, without exporting customer text.

These are elapsed event intervals, not CPU attribution. In particular, uncovered
time is not automatically checkpoint time. Explicit post-effect timestamps give
a narrower delay up to publication start, not its completion or CPU cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
import zipfile
from collections import Counter
from datetime import datetime
from itertools import pairwise
from pathlib import Path

PLAN_TOOLS = {"controller_sync_work_plan", "controller_update_work_plan"}
READ = "cayu_read_product_documents"
WORKSHEET = "cayu_check_comparison_facts"
RESEARCH_TOOLS = {READ, "cayu_browse_product_documents", "cayu_search_banking_knowledge"}


def returned_document_revisions(payload: dict) -> list[tuple[str, int]]:
    """Count delivered revisions, including product-packet reads with no ID list."""
    result = payload.get("result")
    if not isinstance(result, dict) or result.get("is_error"):
        raise ValueError("Expected a successful read result")
    content = result.get("structured")
    if not isinstance(content, dict) or "documents" not in content:
        try:
            content = json.loads(result["content"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Read result has no structured document inventory") from None
    if not isinstance(content, dict) or not isinstance(content.get("documents"), list):
        raise ValueError("Read result has no document inventory")
    revisions = []
    for doc in content["documents"]:
        if (
            not isinstance(doc, dict)
            or not isinstance(doc.get("entry_id"), str)
            or type(doc.get("entry_revision")) is not int
        ):
            raise ValueError("Read result has an invalid document identity")
        if doc.get("navigation_only") or doc.get("text_complete") is False:
            continue
        revisions.append((doc["entry_id"], doc["entry_revision"]))
    return revisions


def seconds(value: str) -> float:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Phase timestamps require an explicit timezone")
    return result.timestamp()


def union_seconds(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = float("-inf")
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("An event interval ends before it starts")
        total += max(0.0, stop - max(start, end))
        end = max(stop, end)
    return total


def summarize(events: list[dict]) -> dict:
    if not events or len({e["session_id"] for e in events}) != 1:
        raise ValueError("Expected one nonempty session")
    stamps = [seconds(e["timestamp"]) for e in events]
    timestamp_reversals = sum(a > b for a, b in pairwise(stamps))
    starts: dict[tuple, tuple[float, str | None]] = {}
    intervals: dict[str, list[tuple[float, float]]] = {}
    counts, failed_tools, tool_counts = Counter(), Counter(), Counter()
    unmatched_ends, replaced_starts = Counter(), Counter()
    model_intervals = {}
    plan_attempts, worksheet_attempts, research_attempts = set(), set(), set()
    reads, post_effect = [], []
    recall_reported = []
    compaction_unstarted = []

    def begin(key, at, name=None):
        if key in starts:
            replaced_starts[key[0]] += 1
        starts[key] = (at, name)

    def finish(key, at):
        item = starts.pop(key, None)
        if item is None:
            unmatched_ends[key[0]] += 1
            return None
        start, name = item
        if at < start:
            raise ValueError("Negative elapsed interval")
        intervals.setdefault(key[0], []).append((start, at))
        return start, name

    for event, at in zip(events, stamps, strict=True):
        kind, payload, name = event["event_type"], event["payload"], event.get("tool_name")
        counts[kind] += 1
        attempt = payload.get("model_attempt_id")
        step = payload.get("model_step_id")
        call = payload.get("tool_call_id")
        if kind == "model.started" and attempt:
            begin(("model", attempt), at)
        elif kind in {"model.completed", "model.failed"} and attempt:
            matched = finish(("model", attempt), at)
            if matched:
                model_intervals[attempt] = (matched[0], at)
        elif kind == "tool.call.started" and call:
            begin(("tool", call), at, name)
            tool_counts[name] += 1
            if name in PLAN_TOOLS:
                plan_attempts.add(attempt)
            if name == WORKSHEET:
                worksheet_attempts.add(attempt)
            if name in RESEARCH_TOOLS:
                research_attempts.add(attempt)
        elif kind in {"tool.call.completed", "tool.call.failed", "tool.call.blocked"} and call:
            matched = finish(("tool", call), at)
            if matched:
                intervals.setdefault("tool:" + str(name), []).append((matched[0], at))
            if kind != "tool.call.completed":
                failed_tools[name] += 1
            effect_at = payload.get("tool_effect_completed_at")
            publication_at = payload.get("tool_terminal_publication_started_at")
            if effect_at and publication_at:
                effect = seconds(effect_at)
                publication = seconds(publication_at)
                if effect > publication or (matched and effect < matched[0]):
                    raise ValueError("Tool effect timestamp is outside its recorded interval")
                delay = publication - effect
                post_effect.append({"tool": name, "seconds": delay})
                intervals.setdefault("post_effect_to_publication_start", []).append(
                    (effect, publication)
                )
            if kind == "tool.call.completed" and name == READ:
                reads.extend(returned_document_revisions(payload))
        elif kind == "memory.recall.started":
            begin(("recall", step, payload.get("policy_sha256")), at)
        elif kind in {"memory.recall.completed", "memory.recall.failed"}:
            finish(("recall", step, payload.get("policy_sha256")), at)
            duration = payload.get("duration_seconds")
            if (
                isinstance(duration, (float, int))
                and not isinstance(duration, bool)
                and duration >= 0
            ):
                recall_reported.append(duration)
        elif kind == "context.compaction.started":
            begin(("compaction", step), at)
        elif kind in {"context.compaction.completed", "context.compaction.failed"}:
            # A single started event can be followed by multiple completed chunks.
            # A partial-prefix event is not the end of the complete compaction.
            if payload.get("coverage_mode") == "partial_prefix":
                continue
            key = ("compaction", step)
            if key in starts:
                finish(key, at)
            else:
                compaction_unstarted.append(step)

    measured = [
        i for phase in ("model", "tool", "recall", "compaction") for i in intervals.get(phase, [])
    ]
    total_span = max(stamps) - min(stamps)
    model_seconds = {
        category: sum(end - start for a, (start, end) in model_intervals.items() if a in attempts)
        for category, attempts in (
            ("plan_proposals", plan_attempts),
            ("worksheet_proposals", worksheet_attempts),
            ("research_proposals", research_attempts),
        )
    }
    return {
        "event_span_seconds": round(total_span, 3),
        "event_counts": dict(sorted(counts.items())),
        "timestamp_reversals_in_publication_order": timestamp_reversals,
        "phase_union_seconds": {
            k: round(union_seconds(v), 3) for k, v in sorted(intervals.items())
        },
        "covered_interval_union_seconds": round(union_seconds(measured), 3),
        "uncovered_event_span_seconds": round(total_span - union_seconds(measured), 3),
        "model_seconds_by_proposed_work_nonexclusive": {
            k: round(v, 3) for k, v in model_seconds.items()
        },
        "tool_starts": dict(sorted(tool_counts.items())),
        "tool_failures_or_blocks": dict(sorted(failed_tools.items())),
        "successful_document_revision_reads": len(reads),
        "unique_document_revisions_read": len(set(reads)),
        "repeat_revision_reads": len(reads) - len(set(reads)),
        "reported_recall_engine_seconds": round(sum(recall_reported), 3),
        "recall_engine_timed_calls": len(recall_reported),
        "post_effect_to_publication_start": {
            "timed_calls": len(post_effect),
            "sum_seconds_nonexclusive": round(sum(r["seconds"] for r in post_effect), 3),
            "maximum_seconds": round(max((r["seconds"] for r in post_effect), default=0), 3),
        },
        "unmatched_starts": dict(Counter(k[0] for k in starts)),
        "unmatched_ends": dict(unmatched_ends),
        "replaced_starts": dict(replaced_starts),
        "unstarted_terminal_compactions": len(compaction_unstarted),
        "limits": [
            "Phase unions can overlap and must not be added as exclusive wall time.",
            "Model proposal time includes reasoning, not just emitting plan or worksheet JSON.",
            "Uncovered time includes uninstrumented work, waits, callback/simulator time, and persistence; it is not measured checkpoint time.",
            "Post-effect delays end at publication START. They do not measure publication completion, CPU ownership or a framework root cause.",
            "Event timestamps may precede publication of later-sequenced hooks. Pairing uses explicit operation IDs, not assumed monotonic publication timestamps.",
            "Repeated revision reads may be justified; this count does not establish whether the source remained visible.",
            "An incomplete interval is counted as unmatched, never filled with an assumed duration.",
        ],
    }


def audit_archive(task_directory: Path) -> dict:
    if not (task_directory / "assessment.json").is_file():
        raise ValueError("Only completed tasks may be profiled")
    archive_path = task_directory / "diagnostics.zip"
    with (
        zipfile.ZipFile(archive_path) as archive,
        tempfile.TemporaryDirectory(prefix="tau-phases-") as td,
    ):
        names = [
            i
            for i in archive.infolist()
            if i.filename.startswith("sessions/") and i.filename.endswith(".sqlite3")
        ]
        if len(names) != 1 or names[0].file_size > 256 * 1024 * 1024:
            raise ValueError("Expected one bounded SQLite session archive")
        target = Path(td) / "session.sqlite3"
        with archive.open(names[0]) as source, target.open("xb") as dest:
            shutil.copyfileobj(source, dest)
        with sqlite3.connect(target.as_uri() + "?mode=ro", uri=True) as db:
            events = [
                {
                    "session_id": session,
                    "event_type": kind,
                    "timestamp": timestamp,
                    "tool_name": tool,
                    "payload": json.loads(payload),
                }
                for session, kind, timestamp, tool, payload in db.execute(
                    "SELECT session_id,event_type,timestamp,tool_name,payload_json FROM cayu_events ORDER BY sequence"
                )
            ]
            checkpoints = db.execute("SELECT state_json FROM cayu_checkpoints").fetchall()
        if len(checkpoints) != 1:
            raise ValueError("Expected one retained checkpoint")
        checkpoint = json.loads(checkpoints[0][0])
        result = summarize(events)
        with archive_path.open("rb") as source:
            archive_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        result.update(
            task=task_directory.name,
            archive_sha256=archive_sha256,
            checkpoint_bytes=len(checkpoints[0][0].encode()),
            checkpoint_root_field_bytes={
                k: len(json.dumps(v, separators=(",", ":")).encode()) for k, v in checkpoint.items()
            },
            invocation_receipts=len(
                checkpoint.get("invocation_lifecycle_receipt", {}).get("receipts", [])
            ),
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [audit_archive(p) for p in args.tasks]
    with args.output.open("x") as output:
        json.dump({"schema_version": 1, "tasks": results}, output, indent=2)
    for result in results:
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "task",
                        "event_span_seconds",
                        "phase_union_seconds",
                        "uncovered_event_span_seconds",
                        "model_seconds_by_proposed_work_nonexclusive",
                        "post_effect_to_publication_start",
                        "repeat_revision_reads",
                        "invocation_receipts",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
