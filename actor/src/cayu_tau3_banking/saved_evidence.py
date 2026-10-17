"""Read-only recovery of already observed sources; no task/reference access."""

import copy
import hashlib
import json
from dataclasses import asdict

from cayu_tau3_banking.decision_packets import normalized_operation
from cayu_tau3_banking.ledger import ToolActionStatus, _parse_bank_account_transactions

SAVED_EVIDENCE_TOOL = "cayu_read_saved_evidence"
SAVED_EVIDENCE_SCHEMA = {
    "name": SAVED_EVIDENCE_TOOL,
    "description": "Recover exact previously observed documents, transaction rows or saved notes. No new research or bank operation. Use index for available sources; documents for a batch of known exact text windows (up to 8, 16000 requested characters total); document for a single window; transactions for saved rows with original result references and freshness; notes for retained interpretations. Unknown fields stay unknown. offset/next_offset page the same source; document offsets are Unicode characters, others are row offsets.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind"],
        "properties": {
            "kind": {"enum": ["index", "document", "documents", "transactions", "notes"]},
            "documents": {
                "type": "array", "minItems": 1, "maxItems": 8,
                "description": "For kind=documents, recover up to 8 exact saved windows in one call; total requested characters at most 16000. Each result independently reports success or error.",
                "items": {
                    "type": "object", "additionalProperties": False, "required": ["entry_id"],
                    "properties": {
                        "entry_id": {"type": "string", "minLength": 1, "maxLength": 256},
                        "revision": {"type": "integer", "minimum": 1},
                        "offset": {"type": "integer", "minimum": 0},
                        "characters": {"type": "integer", "minimum": 1, "maximum": 6000},
                    },
                },
            },
            "entry_id": {"type": "string", "maxLength": 256},
            "revision": {"type": "integer", "minimum": 1},
            "account_id": {"type": "string", "maxLength": 128},
            "scope": {"type": "string", "maxLength": 64, "description": "For notes, recover the saved reconciliation or analysis for this exact account/topic scope."},
            "ids": {"type": "array", "maxItems": 32, "items": {"type": "string", "maxLength": 256}},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 30},
            "characters": {"type": "integer", "minimum": 1, "maximum": 6000},
        },
    },
}
# Exact public read/administrative names, not a classifier of customer intent.
_READ_ONLY = {
    "get_bank_account_transactions_9173",
    "get_all_user_accounts_by_user_id_3847",
    "get_credit_card_accounts_by_user",
    "get_credit_card_transactions_by_user",
    "get_current_time",
    "get_user_information_by_email",
    "get_user_information_by_id",
    "get_user_information_by_name",
    "list_discoverable_agent_tools",
    "unlock_discoverable_agent_tool",
    "give_discoverable_user_tool",
    "log_verification",
    "get_debit_cards_by_account_id_7823",
    "get_user_dispute_history_7291",
    "get_pending_replacement_orders_5765",
    "get_credit_limit_increase_history_4829",
    "get_closure_reason_history_8293",
    "get_payment_history_6183",
    "get_referrals_by_user",
    "get_atm_deposit_images_8473",
    "get_debit_dispute_status_7483",
}
_ACCOUNT_FIELDS = {
    "account_id",
    "bank_account_id",
    "source_account_id",
    "destination_account_id",
    "credit_card_account_id",
    "checking_account_id",
    "savings_account_id",
}


class SavedEvidence:
    def __init__(self, ledger, facts):
        self.ledger = ledger
        self.facts = facts
        self.documents = {}
        self.transactions = {}

    def observe_result(self, call_id, content):
        if self.ledger.tool_action_status(call_id) is not ToolActionStatus.SUCCEEDED:
            return
        name, raw = normalized_operation(
            self.ledger.tool_action_name(call_id), self.ledger.tool_action_arguments(call_id)
        )
        args = json.loads(raw)
        if name == "get_bank_account_transactions_9173":
            account = args.get("account_id")
            rows = _parse_bank_account_transactions(content)
            if rows is not None and any(row.account_id != account for row in rows):
                rows = None
            if rows is not None:
                observed_rows = [{**asdict(row), "amount": str(row.amount)} for row in rows]
                previous = self.transactions.get(account)
                repeated = (
                    previous.get("unchanged_refreshes", 0) + 1
                    if previous and previous["rows"] == observed_rows
                    else 0
                )
                self.transactions[account] = {
                    "account_id": account,
                    "source": {"kind": "tool_result", "reference_id": call_id},
                    "freshness": "latest_observed_snapshot",
                    "invalidated_by": [],
                    "rows": observed_rows,
                    "unchanged_refreshes": repeated,
                }
            elif account in self.transactions:
                self.transactions[account]["freshness"] = "refresh_required"
                self.transactions[account]["invalidated_by"].append(call_id)
            return
        if name in _READ_ONLY:
            return
        accounts = {v for k, v in args.items() if k in _ACCOUNT_FIELDS and isinstance(v, str)}
        # Unknown effects with no explicit account binding conservatively stale all
        # snapshots. This never prevents a fresh bank lookup or independent action.
        for account, saved in self.transactions.items():
            if not accounts or account in accounts:
                saved["freshness"] = "refresh_required"
                saved["invalidated_by"] = (saved["invalidated_by"] + [call_id])[-8:]

    def observe_search_documents(self, content):
        """Retain only complete documents actually delivered by successful searches."""
        try:
            result = json.loads(content)
        except (ValueError, TypeError):
            return
        if not isinstance(result, dict):
            return
        if isinstance(result.get("batch"), list):
            for item in result["batch"]:
                if isinstance(item, dict) and item.get("is_error") is False:
                    self.observe_search_documents(json.dumps(item.get("result")))
        documents = result.get("documents")
        if isinstance(documents, list):
            complete = [d for d in documents if isinstance(d, dict) and d.get("text_complete") is True]
            self.project_documents(json.dumps({"documents": complete}), retain_only=True)

    def project_documents(self, content, *, targeted=False, retain_only=False):
        """First delivery stays complete. Repeat bodies become recovery locators."""
        try:
            result = json.loads(content)
        except (ValueError, TypeError):
            return content
        if not isinstance(result, dict) or not isinstance(result.get("documents"), list):
            return content
        if targeted:
            return content
        for doc in result["documents"]:
            if not isinstance(doc, dict) or not doc.get("text_complete", True) or targeted:
                continue
            if not isinstance(doc.get("text"), str) or not doc.get("evidence_ref"):
                continue
            ref = doc["evidence_ref"]
            revision = int(ref.get("version", doc.get("entry_revision", 1)))
            key = (doc["entry_id"], revision)
            previous = self.documents.get(key)
            text = doc["text"]
            digest = hashlib.sha256(text.encode()).hexdigest()
            self.documents[key] = {
                "entry_id": key[0],
                "revision": revision,
                "title": doc.get("title", ""),
                "source": copy.deepcopy(ref),
                "sha256": digest,
                "text": text,
            }
            if retain_only or previous is None or previous["sha256"] != digest:
                continue
            # Preserve a short exact opening plus heading offsets, never pretend
            # the shortened result contains the complete document again.
            headings = []
            offset = 0
            for line in text.splitlines(keepends=True):
                if line.startswith("#"):
                    headings.append({"offset": offset, "heading": line.strip()[:160]})
                offset += len(line)
            identity = {
                k: doc[k]
                for k in ("entry_id", "entry_revision", "title", "evidence_ref", "source_handle")
                if k in doc
            }
            doc.clear()
            doc.update(identity)
            doc.update(
                text="",
                text_complete=False,
                saved_source={"characters": len(text), "sections": headings[:12]},
            )
            result["recovery_note"] = (
                "Sources marked saved_source are unchanged revisions already delivered and retained, "
                "not missing evidence. Recover exact needed text with cayu_read_saved_evidence "
                "kind=documents with a batch of entry_id/revision/offset/characters windows, kind=document for one window, or a targeted source_span. "
                "Identical rereads cannot establish facts absent from these sources."
            )
        return json.dumps(result)

    def read(self, args):
        kind = args["kind"]
        if kind == "documents":
            windows = args.get("documents", [])
            if not 1 <= len(windows) <= 8:
                raise ValueError("Supply 1-8 saved document windows.")
            if sum(w.get("characters", 4000) for w in windows) > 16000:
                raise ValueError("Batch recovery exceeds 16000 requested characters; narrow the windows or split the batch.")
            results = []
            for index, window in enumerate(windows):
                try:
                    document = self.read({**window, "kind": "document"})
                except ValueError as error:
                    results.append({"index": index, "entry_id": window.get("entry_id"), "is_error": True, "error": str(error)})
                else:
                    results.append({"index": index, "is_error": False, "document": document})
            failures = sum(item["is_error"] for item in results)
            return {
                "status": "rejected" if failures == len(results) else "partial" if failures else "accepted",
                "results": results,
                "new_research": False,
            }
        offset = args.get("offset", 0)
        limit = args.get("limit", 20)
        if kind == "document":
            matches = [
                d
                for (entry, rev), d in self.documents.items()
                if entry == args.get("entry_id")
                and (args.get("revision") is None or rev == args["revision"])
            ]
            if not matches:
                raise ValueError("No delivered saved revision; retrieve the public source first.")
            doc = max(matches, key=lambda d: d["revision"])
            if offset > len(doc["text"]):
                raise ValueError("Offset exceeds the saved source length.")
            end = min(len(doc["text"]), offset + args.get("characters", 4000))
            return {
                **{k: v for k, v in doc.items() if k != "text"},
                "text": doc["text"][offset:end],
                "text_complete": offset == 0 and end == len(doc["text"]),
                "offset": offset,
                "next_offset": end if end < len(doc["text"]) else None,
            }
        if kind == "transactions":
            saved = self.transactions.get(args.get("account_id"))
            if saved is None:
                raise ValueError(
                    "No observed transaction snapshot for that account; use the bank lookup first."
                )
            rows = saved["rows"]
            if args.get("ids"):
                rows = [r for r in rows if r["transaction_id"] in args["ids"]]
            metadata = {k: copy.deepcopy(v) for k, v in saved.items() if k != "rows"}
            metadata["freshness_note"] = (
                "Observed history, not live bank state. Refresh after changes; no fee ownership or matching inferred."
            )
        elif kind == "notes":
            rows = list(self.facts.facts.values())
            if args.get("scope") is not None:
                rows = [r for r in rows if r.get("scope") == args["scope"]]
            if args.get("ids"):
                rows = [r for r in rows if r["id"] in args["ids"]]
            metadata = {"revision": self.facts.revision, "meaning_verified": False}
        else:
            rows = [
                {
                    "kind": "document",
                    **{k: v for k, v in d.items() if k != "text"},
                    "characters": len(d["text"]),
                }
                for d in self.documents.values()
            ]
            rows += [
                {
                    "kind": "transactions",
                    **{k: v for k, v in s.items() if k != "rows"},
                    "row_count": len(s["rows"]),
                }
                for s in self.transactions.values()
            ]
            rows += [
                {"kind": "note", "id": f["id"], "state": f["state"], "claim": f["claim"]}
                for f in self.facts.facts.values()
            ]
            metadata = {"available_sources": len(rows)}
        page = []
        for row in rows[offset : offset + limit]:
            if page and len(json.dumps([*page, row])) > 14000:
                break
            page.append(copy.deepcopy(row))
        end = offset + len(page)
        return {
            **metadata,
            "rows": page,
            "offset": offset,
            "next_offset": end if end < len(rows) else None,
        }

    def context(self):
        return json.dumps(
            {
                "saved_documents": len(self.documents),
                "transaction_snapshots": [
                    {
                        "account_id": k,
                        "rows": len(v["rows"]),
                        "source": v["source"],
                        "freshness": v["freshness"],
                        "unchanged_refreshes": v.get("unchanged_refreshes", 0),
                    }
                    for k, v in self.transactions.items()
                ],
                "recovery_tool": SAVED_EVIDENCE_TOOL,
                "note": "Recover exact rows/passages here instead of rereading whole histories. Snapshot is not live state; no inference about fee ownership.",
            }
        )
