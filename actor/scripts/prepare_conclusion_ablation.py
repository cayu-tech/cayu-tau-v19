"""Freeze a diagnostic ablation of prior model judgments, never edit the acting agent.

The exact passage selectors below describe one recorded development context.
They are evaluator-side interventions, not runtime rules or suggested answers.
All source documents, account results, workflow states and consent stay unchanged.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

from cayu.providers.base import ModelRequest

from cayu_tau3_banking.captured_guidance_probe import checked_guidance_payload
from cayu_tau3_banking.decision_probe import save_private_json

PLAN_TOOL = "controller_update_work_plan"
SUMMARY_PREFIX = "Previous session context summary:\n"
OPEN_TAG, CLOSE_TAG = "<workflow-plan>\n", "\n</workflow-plan>"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_replace(text, before, after):
    if not before or not after or before == after or text.count(before) != 1:
        raise ValueError("Expected exactly one nonempty intervention passage.")
    return text.replace(before, after, 1)


def edit_plan(plan, edits):
    result = deepcopy(plan)
    for edit in edits:
        items = [i for i in result["items"] if i["id"] == edit["item_id"]]
        if len(items) != 1:
            raise ValueError("Expected exactly one selected work item.")
        target = items[0]
        if edit.get("obligation_id"):
            obligations = [o for o in target["obligations"] if o["id"] == edit["obligation_id"]]
            if len(obligations) != 1:
                raise ValueError("Expected exactly one selected obligation.")
            target = obligations[0]
        if target.get("evidence") != edit["before"]:
            raise ValueError("Selected planning evidence differs from the frozen passage.")
        target["evidence"] = edit["after"]
    return result


def intervene(request, spec):
    """Change only selected summary passages and private planning evidence fields."""
    changed = deepcopy(request)
    summaries = plans = contexts = 0
    for message in changed["messages"]:
        for part in message["content"]:
            if message["role"] == "user" and part["type"] == "text":
                text = part["text"]
                if text.startswith(SUMMARY_PREFIX):
                    summaries += 1
                    for edit in spec["summary"]:
                        text = exact_replace(text, edit["before"], edit["after"])
                    part["text"] = text
                if OPEN_TAG in text or CLOSE_TAG in text:
                    contexts += 1
                    if text.count(OPEN_TAG) != 1 or text.count(CLOSE_TAG) != 1:
                        raise ValueError("Expected one complete workflow section.")
                    prefix, rest = text.split(OPEN_TAG)
                    body, suffix = rest.split(CLOSE_TAG)
                    expected = edit_plan(json.loads(body), spec["plan"])
                    for edit in spec["plan"]:
                        body = exact_replace(
                            body, json.dumps(edit["before"]), json.dumps(edit["after"])
                        )
                    if json.loads(body) != expected:
                        raise ValueError("Workflow serialization changed unintended fields.")
                    part["text"] = prefix + OPEN_TAG + body + CLOSE_TAG + suffix
            if (
                message["role"] == "assistant"
                and part["type"] == "tool_call"
                and part["tool_name"] == PLAN_TOOL
                and part["tool_call_id"] == spec["plan_call_id"]
            ):
                plans += 1
                part["arguments"] = edit_plan(part["arguments"], spec["plan"])
    if (summaries, plans, contexts) != (1, 1, 1):
        raise ValueError("Expected one summary, one selected private plan call and one projection.")
    return changed


def intervention_spec():
    return {
        "plan_call_id": "call_EkMzeXGVB8MWD9PydxvQFOTj",
        "summary": [
            {
                "before": "- Current provisional comparison: Green has the best confirmed fee-adjusted illustration without relationship assumptions; Gold advertises a higher rate but has unresolved below-$10,000 eligibility and fee treatment. No definitive highest attainable product for Riley until linked products are inspected.",
                "after": "- Current provisional comparison: Gold advertises a higher rate but has unresolved below-$10,000 eligibility and fee treatment. No definitive highest attainable product for Riley until linked products are inspected.",
            },
            {
                "before": "- Based on stated preferences and ability to maintain about $500, **Evergreen + Green Savings is a tentative leading setup**, not a final recommendation or required sequence. Existing relationships could produce a better attainable rate.",
                "after": "- No final recommendation or required sequence has been established. Existing relationships could produce a better attainable rate.",
            },
        ],
        "plan": [
            {
                "item_id": "compare-checking",
                "before": "Evergreen best matches Riley's ability to maintain about $500 and priority of maximizing linked savings APY while avoiding fees: its $6 monthly fee is waived at a $500 minimum daily balance and it provides a +0.55% APY boost to linked Green Savings.",
                "after": "Riley can maintain about $500 and prioritizes maximizing linked savings APY while avoiding fees. Evergreen's $6 monthly fee is waived at a $500 minimum daily balance and it provides a +0.55% APY boost to linked Green Savings.",
            },
            {
                "item_id": "open-checking",
                "before": "Evergreen is the recommended product, but Riley must explicitly select and authorize opening it before fresh application eligibility evidence is retrieved and the opening tool is used.",
                "after": "For Evergreen, Riley must explicitly select and authorize opening it before fresh application eligibility evidence is retrieved and the opening tool is used.",
            },
            {
                "item_id": "open-checking",
                "obligation_id": "checking-selection-consent",
                "before": "The account and relationship comparison now supports recommending Evergreen, so Riley can be asked to select it and authorize opening with the disclosed $6 fee and $500 daily-balance waiver.",
                "after": "For Evergreen, Riley can be asked to select it and authorize opening with the disclosed $6 fee and $500 daily-balance waiver.",
            },
            {
                "item_id": "compare-savings",
                "before": "Riley has no credit cards and only a Light Blue checking account, so no current relationship bonus makes Gold's below-$10,000 conditions more attainable. Green Savings linked to Evergreen gives a documented 4.0% base plus 0.55%, with no monthly maintenance fee; at a constant $6,000 that is about $273 annually before rate or balance changes.",
                "after": "Riley has no credit cards and only a Light Blue checking account. Green Savings linked to Evergreen gives a documented 4.0% base plus 0.55%, with no monthly maintenance fee; at a constant $6,000 that is about $273 annually before rate or balance changes.",
            },
            {
                "item_id": "open-savings",
                "before": "Green Savings is the recommended product, but Riley must select and authorize it, accept paperless statements, and identify the funding source for the $6,000 before opening/funding actions proceed.",
                "after": "For Green Savings, Riley must select and authorize it, accept paperless statements, and identify the funding source for the $6,000 before opening/funding actions proceed.",
            },
            {
                "item_id": "open-savings",
                "obligation_id": "savings-selection-consent",
                "before": "The account and relationship review now supports recommending Green Savings linked to Evergreen, so Riley can be asked to select it, authorize opening, and accept its paperless requirement.",
                "after": "For Green Savings linked to Evergreen, Riley can be asked to select it, authorize opening, and accept its paperless requirement.",
            },
        ],
    }


def prepare(project, reference):
    source = reference / "1-candidate-request.json"
    old_protocol = json.loads((reference / "protocol.json").read_text())
    entry = next(r for r in old_protocol["schedule"] if r["name"] == "1-candidate")
    if digest(source) != entry["request_sha256"]:
        raise ValueError("Reference request changed.")
    retained = json.loads(source.read_text())
    spec = intervention_spec()
    neutral = intervene(retained, spec)
    reverse = deepcopy(spec)
    for edit in [*reverse["summary"], *reverse["plan"]]:
        edit["before"], edit["after"] = edit["after"], edit["before"]
    if intervene(neutral, reverse) != retained:
        raise ValueError("Intervention is not exactly reversible.")
    for arm in (retained, neutral):
        request = ModelRequest.model_validate(arm)
        checked_guidance_payload(request, expected=retained["messages"][0]["content"][0]["text"])
    root = Path(tempfile.mkdtemp(prefix="conclusion-ablation-", dir=project / "artifacts"))
    shutil.copy2(reference / "contrast.py", root / "contrast.py")
    shutil.copy2(Path(__file__), root / "prepare.py")
    save_private_json(root / "intervention.json", spec)
    schedule = []
    for i, arm in enumerate(("retained", "neutral", "neutral", "retained")):
        name = f"{i}-{arm}"
        save_private_json(root / f"{name}-request.json", retained if arm == "retained" else neutral)
        schedule.append(
            {"name": name, "arm": arm, "request_sha256": digest(root / f"{name}-request.json")}
        )
    protocol = {
        "schedule": schedule,
        "script_sha256": digest(root / "contrast.py"),
        "source_context_sha256": old_protocol["source_context_sha256"],
        "source_capture_sha256": old_protocol["source_capture_sha256"],
        "public_schema_reconstruction": old_protocol["public_schema_reconstruction"],
        "reference_request": str(source),
        "reference_request_sha256": digest(source),
        "prepare_script_sha256": digest(root / "prepare.py"),
        "intervention_sha256": digest(root / "intervention.json"),
        "max_requests": 4,
        "whole_request_retries": 0,
        "timeout_seconds_per_request": 300,
        "tools_executed": 0,
        "max_concurrency": 1,
        "model": retained["model"],
        "effort": "xhigh",
        "provider": "existing subscription",
        "hypothesis": "Do the selected earlier recommendation judgments steer the next decision away from an unresolved source-linked benefit?",
        "treatment": "Fourteen evaluator-side text edits: two summary passages and six private planning evidence fields in both the historical planner call and current workflow projection. No acting-agent implementation change.",
        "preserved": "Same new guidance in both arms, every tool/schema/option, all public-document and environment tool results, customer requirements and consent, all workflow IDs/states/obligations/references/dependencies, response contract, research limits, ledger, active evidence and capability directory. No additional answer facts or reference actions.",
        "workflow_fact": "Comparison items are in_progress, not completed. Research obligations are resolved, communication obligations ready, and response_contract.mode is customer_input_only. These controls remain unchanged and their causal effect is not tested here.",
        "limits": "Selected-conclusion wording ablation, not elimination of every possible anchor or a token-length-matched intervention. Shorter text may alter salience. Four one-step decisions on one known development context cannot prove a general causal mechanism or a task pass. No tool execution or continuation.",
        "grading": "After all four complete, manually inspect proposed search/research targets, handling of potential linked-card benefits, gross/net disclosure, and consent/sequence preservation. An unresolved tool-call step has no final-response disclosure score. No grading feedback to actor.",
    }
    save_private_json(root / "protocol.json", protocol)
    print(json.dumps({"frozen": str(root), "requests": 4, "model_dispatches": 0}), flush=True)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[1], args.reference.resolve())
