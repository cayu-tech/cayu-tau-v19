"""Bound private bookkeeping without treating narrative rewrites as execution."""

from __future__ import annotations

import hashlib
import json


class PlanningStallError(RuntimeError):
    """The actor exhausted its bounded opportunity to repair a planning-only loop."""


def plan_fingerprint(items: list[dict]) -> str:
    """Compare structured decisions, not revision numbers, goals or evidence prose."""
    projection = []
    for item in items:
        value = {k: v for k, v in item.items() if k not in {"goal", "evidence"}}
        if "depends_on" in value:
            value["depends_on"] = sorted(value["depends_on"])
        value["obligations"] = [
            {k: v for k, v in obligation.items() if k not in {"goal", "evidence"}}
            for obligation in item.get("obligations", [])
        ]
        value["obligations"].sort(key=lambda o: o["id"])
        if "next_operation" in value:
            value["next_operation"] = {
                k: v for k, v in value["next_operation"].items() if k != "summary"
            }
        # Reference order carries no new authority or information.
        for record in [value, *value["obligations"]]:
            record["evidence_refs"] = sorted(
                record.get("evidence_refs", []), key=lambda ref: json.dumps(ref, sort_keys=True)
            )
        projection.append(value)
    projection.sort(key=lambda item: item["id"])
    return hashlib.sha256(json.dumps(projection, sort_keys=True).encode()).hexdigest()


class PlanningProgress:
    """Conversation-local bound; authoritative observations reset the window.

    A new structured plan is useful bookkeeping, but never proof of task completion.
    A hard per-observation ceiling also bounds oscillating or endlessly expanded plans.
    These counters intentionally survive external tool callbacks within one conversation.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.updates = 0
        self.stalls = 0
        self.seen: set[str] = set()
        self.private_results: set[str] = set()

    def observe_private_result(self, name: str, content: str, *, failed: bool) -> None:
        """Only a new private result clears the planning-only window.

        This detects exact repeated observations, not semantic progress. Ignore
        the adapter's top-level capacity counters: spending another slot is not
        new evidence. External observations reset this conversation-local cache.
        """
        if failed:
            return
        try:
            value = json.loads(content)
        except (ValueError, TypeError):
            value = content
        if isinstance(value, dict):
            value = {
                key: item
                for key, item in value.items()
                if key not in {"navigation_budget", "remaining_calls", "search_budget"}
            }
        fingerprint = hashlib.sha256(
            json.dumps([name, value], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        if fingerprint in self.private_results:
            return
        self.private_results.add(fingerprint)
        self.updates = 0
        self.stalls = 0
        self.seen.clear()

    def observe_plan(self, before: str, after: str, *, failed: bool) -> None:
        self.updates += 1
        self.seen.add(before)
        if failed or after in self.seen:
            self.stalls += 1
        else:
            self.stalls = 0
        self.seen.add(after)

    @property
    def warning(self) -> str | None:
        if self.stalls < 2 and self.updates < 8:
            return None
        return (
            "Planning is not advancing execution; repeated private lookups are not new evidence. "
            "Accepted plan updates are already saved in the current workflow-plan context. "
            "An unchanged-update reply means the requested state is already current, not that "
            "persistence failed. Do not rewrite explanations or toggle states. "
            "Use an available tool for genuinely executable work, or reconcile a real blocker "
            "and deliver an honest limited response/customer choice. Document lookups use kind "
            "knowledge_search; a depleted search budget is not customer authorization and does "
            "not establish completion. Keep unverified side effects unresolved. Do not ask an "
            "unnecessary customer question just to reset a search budget."
        )

    def check(self) -> None:
        if self.stalls >= 5 or self.updates >= 12:
            raise PlanningStallError("planning_no_progress")
