"""Small actor-authored factual notes, separate from outcomes and action packets.

Source checks prove availability, not entailment. No prose/intent classifier lives
here, and notes never authorize an action or mark an outcome complete.
"""

import copy
import json

from jsonschema import Draft202012Validator

from cayu_tau3_banking.decision_packets import normalized_operation
from cayu_tau3_banking.ledger import ToolActionStatus

DECISION_NOTE_TOOL = "cayu_update_decision_facts"
_REF = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "reference_id"],
    "properties": {
        "kind": {"enum": ["customer_message", "tool_result", "knowledge_revision"]},
        "reference_id": {"type": "string", "minLength": 1, "maxLength": 256},
        "version": {"type": "string", "maxLength": 64},
    },
}
_CONFLICT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["affected_outcome", "operation", "arguments", "prerequisite"],
    "properties": {
        "affected_outcome": {"type": "string", "minLength": 1, "maxLength": 64},
        "operation": {"type": "string", "minLength": 1, "maxLength": 128},
        "arguments": {
            "type": "object",
            "minProperties": 1,
            "maxProperties": 6,
            "additionalProperties": {"type": ["string", "number", "boolean"]},
            "description": "Exact identifying arguments of the operation creating the blocker; no aliases or intent matching.",
        },
        "prerequisite": {"type": "string", "minLength": 1, "maxLength": 360},
        "customer_choice": {
            "type": "object",
            "additionalProperties": False,
            "required": ["disclosure_sequence", "response_sequence"],
            "properties": {
                "disclosure_sequence": {"type": "integer", "minimum": 1},
                "response_sequence": {"type": "integer", "minimum": 1},
            },
            "description": "Delivered dialogue sequence numbers: explain the lost/delayed outcome, then customer's informed choice. Actor interprets meaning; chronological references are checked.",
        },
    },
}
DECISION_NOTE_SCHEMA = {
    "name": DECISION_NOTE_TOOL,
    "description": "Optional compact working memory, not an authorization or completion worksheet. For comparisons retain the objective, calculations with conditions, exclusions and material unknowns; cite source inputs and distinguish your conclusions from source assertions. Patch only changed facts. Preserve unknown/contradicted states, including conflicts inside one source. Reuse IDs. Actor owns meaning. Independent facts are accepted separately; the response explicitly lists accepted and rejected IDs. Existing facts survive rejected patches. Remove obsolete facts by id. Saved notes beyond the compact projection remain available through cayu_read_saved_evidence kind=notes.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "facts": {
                "type": "array",
                "maxItems": 32,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "claim", "state", "sources"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1, "maxLength": 64},
                        "claim": {"type": "string", "minLength": 1, "maxLength": 360},
                        "scope": {
                            "type": "string",
                            "maxLength": 64,
                            "description": "Optional outcome or topic ID for recovering related notes; no new plan required.",
                        },
                        "conflict": _CONFLICT,
                        "comparison": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "alternative",
                                "calculation",
                                "conditions",
                                "next_missing_fact",
                            ],
                            "properties": {
                                "alternative": {"type": "string", "maxLength": 128},
                                "calculation": {"type": "string", "maxLength": 360},
                                "conditions": {"type": "string", "maxLength": 360},
                                "next_missing_fact": {"type": "string", "maxLength": 240},
                            },
                            "description": "Optional compact comparison row. Empty next_missing_fact means none known, not proven exhaustive coverage. Retained verbatim across compaction; no separate worksheet required.",
                        },
                        "state": {"enum": ["true", "false", "unknown", "contradicted"]},
                        "sources": {"type": "array", "maxItems": 32, "items": _REF},
                        "invalidated_by": {
                            "type": "array",
                            "maxItems": 6,
                            "items": {"type": "string", "maxLength": 100},
                        },
                        "resource": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["field", "value"],
                            "properties": {
                                "field": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 100,
                                    "description": "Exact argument name used by the invalidating tool; no inferred aliases.",
                                },
                                "value": {"type": "string", "minLength": 1, "maxLength": 128},
                            },
                        },
                    },
                },
            },
            "remove": {
                "type": "array",
                "maxItems": 32,
                "items": {"type": "string", "maxLength": 64},
            },
        },
        "anyOf": [{"required": ["facts"]}, {"required": ["remove"]}],
    },
}


class DecisionFacts:
    def __init__(self, ledger, controller):
        self.ledger = ledger
        self.controller = controller
        self.facts = {}
        self.revision = 0

    def update(self, patch):
        candidate = copy.deepcopy(self.facts)
        seen = set()
        for fact in patch.get("facts", []):
            if fact["id"] in seen or fact["id"] in patch.get("remove", []):
                raise ValueError("Each fact ID can be changed only once per patch")
            seen.add(fact["id"])
            refs = fact["sources"]
            # A single delivered document can contain two conflicting assertions.
            # References establish source availability, never semantic entailment.
            minimum = 0 if fact["state"] == "unknown" else 1
            if len({json.dumps(ref, sort_keys=True) for ref in refs}) < minimum:
                raise ValueError("Known or contradicted facts need an available source")
            for ref in refs:
                kind, rid = ref["kind"], ref["reference_id"]
                valid = (
                    self.ledger.has_customer_message(rid)
                    if kind == "customer_message"
                    else self.ledger.tool_action_status(rid) is ToolActionStatus.SUCCEEDED
                    if kind == "tool_result"
                    else self.controller.has_delivered_knowledge_revision(rid, ref.get("version"))
                )
                if not valid:
                    raise ValueError("Fact source is unavailable or not successful: " + rid)
            conflict = fact.get("conflict")
            if conflict:
                items = {item["id"]: item for item in self.controller.export_state()["items"]}
                if conflict["affected_outcome"] not in items:
                    raise ValueError("Conflict must identify an existing customer outcome")
                if not any(ref["kind"] == "knowledge_revision" for ref in refs):
                    raise ValueError("A sequencing conflict needs a delivered policy source")
                choice = conflict.get("customer_choice")
                if choice:
                    dialogue = {
                        m["sequence"]: m
                        for m in json.loads(self.ledger.render_dialogue_context())["messages"]
                    }
                    disclosure = dialogue.get(choice["disclosure_sequence"], {})
                    response = dialogue.get(choice["response_sequence"], {})
                    if (
                        disclosure.get("role") != "assistant"
                        or response.get("role") != "customer"
                        or choice["response_sequence"] <= choice["disclosure_sequence"]
                    ):
                        raise ValueError(
                            "Informed choice requires a delivered explanation followed by a customer response"
                        )
            candidate[fact["id"]] = copy.deepcopy(fact)
        for key in patch.get("remove", []):
            candidate.pop(key, None)
        if len(candidate) > 64 or len(json.dumps(candidate)) > 96_000:
            raise ValueError(
                "Saved-note capacity is 64 facts / 96000 characters. Remove obsolete IDs; existing notes are retained."
            )
        self.facts = candidate
        self.revision += 1

    def apply_patch(self, patch):
        """Best-effort independent entries; never silently accept invalid evidence.

        The legacy update method stays atomic for internal callers. Tool callers
        receive explicit per-entry receipts and repair only rejected entries.
        """
        accepted, rejected = [], []
        entries = patch.get("facts", [])
        ids = [f.get("id") for f in entries if isinstance(f, dict)]
        removals = patch.get("remove", [])
        validator = Draft202012Validator(
            DECISION_NOTE_SCHEMA["input_schema"]["properties"]["facts"]["items"]
        )
        for index, fact in enumerate(entries):
            key = fact.get("id") if isinstance(fact, dict) else None
            errors = list(validator.iter_errors(fact))
            if errors or not key or ids.count(key) > 1 or key in removals:
                details = [
                    {
                        "path": ["facts", index, *error.absolute_path],
                        "rule": error.validator,
                        "constraint": error.validator_value,
                        "message": (
                            "Missing required fields: " + ", ".join(
                                field for field in error.validator_value if field not in fact
                            )
                            if error.validator == "required" and isinstance(fact, dict)
                            else f"Field violates {error.validator}; repair this field."
                        ),
                    }
                    for error in errors[:3]
                ]
                rejected.append({
                    "index": index,
                    "id": key,
                    "reason": "Invalid fact fields" if errors else "Each ID must be changed once",
                    "errors": details,
                })
                continue
            try:
                self.update({"facts": [fact]})
            except ValueError as error:
                rejected.append({"index": index, "id": key, "reason": str(error)})
            else:
                accepted.append(key)
        for key in removals:
            if key not in ids:
                self.update({"remove": [key]})
        removed = [k for k in removals if k not in ids]
        return {
            "status": "partial" if rejected and (accepted or removed) else "rejected" if rejected else "accepted",
            "accepted_ids": accepted,
            "rejected": rejected,
            "removed_ids": removed,
            "revision": self.revision,
            "stored_fact_count": len(self.facts),
            "note": "Only accepted IDs changed. Recover saved notes with cayu_read_saved_evidence kind=notes; repair only rejected entries.",
        }

    def context_snapshot(self):
        selected = []
        # Recent active interpretations, with conflict declarations prioritized.
        ordered = sorted(
            reversed(list(self.facts.values())), key=lambda f: (not bool(f.get("conflict")), not bool(f.get("comparison")))
        )
        for fact in ordered:
            if len(selected) >= 16 or len(json.dumps([*selected, fact])) > 14000:
                continue
            selected.append(copy.deepcopy(fact))
        return {
            "revision": self.revision,
            "owner": "acting_model",
            "facts": selected,
            "stored_fact_count": len(self.facts),
            "omitted_from_projection": len(self.facts) - len(selected),
            "omitted_comparisons": [
                {"id": f["id"], "alternative": f["comparison"]["alternative"], "state": f["state"]}
                for f in ordered if f.get("comparison") and f["id"] not in {x["id"] for x in selected}
            ],
            "recovery": "cayu_read_saved_evidence kind=notes; omitted notes are retained, not deleted",
        }

    def observe_result(self, call_id):
        if self.ledger.tool_action_status(call_id) is not ToolActionStatus.SUCCEEDED:
            return
        name, raw = normalized_operation(
            self.ledger.tool_action_name(call_id), self.ledger.tool_action_arguments(call_id)
        )
        arguments = json.loads(raw)
        changed = False
        for fact in self.facts.values():
            scope = fact.get("resource")
            if name not in fact.get("invalidated_by", []):
                continue
            if scope and arguments.get(scope["field"]) != scope["value"]:
                continue
            fact["state"] = "unknown"
            fact["invalidated_by_result"] = call_id
            changed = True
        if changed:
            self.revision += 1

    def conflict_error(self, name, arguments):
        """Only model-declared, source-backed, exactly scoped conflicts affect admission.

        Meaning is not proven by citation presence. No text/intent classifier and
        no requirement to manufacture a conflict for every operation.
        """
        items = {item["id"]: item for item in self.controller.export_state()["items"]}
        for fact in self.facts.values():
            conflict = fact.get("conflict")
            if not conflict or fact["state"] == "false" or conflict.get("customer_choice"):
                continue
            if name != conflict["operation"] or not all(
                key in arguments and type(arguments[key]) is type(value) and arguments[key] == value
                for key, value in conflict["arguments"].items()
            ):
                continue
            item = items.get(conflict["affected_outcome"], {})
            if item.get("status") == "completed" and item.get("outcome") in {
                "succeeded",
                "answered",
                "declined",
            }:
                continue
            return (
                f"Recorded conflict {fact['id']} affects outcome {conflict['affected_outcome']}: "
                f"{conflict['prerequisite']}. Complete that outcome first, or explain the tradeoff "
                "and cite the customer's subsequent informed choice. If the recorded rule is wrong, "
                "correct it from policy evidence. Independent operations remain available."
            )
        return None

    def snapshot(self):
        return {
            "revision": self.revision,
            "owner": "acting_model",
            "facts": copy.deepcopy(list(self.facts.values())),
        }
