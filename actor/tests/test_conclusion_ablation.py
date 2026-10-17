import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prepare_conclusion_ablation",
    Path(__file__).parents[1] / "scripts/prepare_conclusion_ablation.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def fixture():
    plan = {
        "items": [
            {
                "id": "compare",
                "status": "in_progress",
                "evidence": "Best option.",
                "obligations": [
                    {"id": "consent", "status": "ready", "evidence": "Recommend it; need consent."}
                ],
            }
        ],
        "response_contract": {"mode": "customer_input_only"},
    }
    request = {
        "model": "same",
        "tools": [{"name": "same"}],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "same prompt"}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": probe.SUMMARY_PREFIX + "Customer wants capacity. Best option.",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_name": probe.PLAN_TOOL,
                        "tool_call_id": "private",
                        "arguments": deepcopy(plan),
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "Public source says Best option.; authoritative result unchanged.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Ledger unchanged\n"
                        + probe.OPEN_TAG
                        + json.dumps(plan)
                        + probe.CLOSE_TAG
                        + "\nSources unchanged",
                    }
                ],
            },
        ],
    }
    edits = {
        "plan_call_id": "private",
        "summary": [{"before": "Best option.", "after": "Not yet ranked."}],
        "plan": [
            {"item_id": "compare", "before": "Best option.", "after": "Option available."},
            {
                "item_id": "compare",
                "obligation_id": "consent",
                "before": "Recommend it; need consent.",
                "after": "Need consent.",
            },
        ],
    }
    return request, edits


def test_only_selected_derived_text_changes_and_all_state_and_results_survive():
    original, edits = fixture()
    before = deepcopy(original)
    changed = probe.intervene(original, edits)
    assert original == before
    assert changed["messages"][0] == original["messages"][0]
    assert changed["messages"][3] == original["messages"][3]
    assert changed["tools"] == original["tools"]
    plan = changed["messages"][2]["content"][0]["arguments"]
    assert plan["items"][0]["status"] == "in_progress"
    assert plan["items"][0]["obligations"][0]["status"] == "ready"
    assert plan["response_contract"] == {"mode": "customer_input_only"}
    reverse = deepcopy(edits)
    for row in [*reverse["summary"], *reverse["plan"]]:
        row["before"], row["after"] = row["after"], row["before"]
    assert probe.intervene(changed, reverse) == original


@pytest.mark.parametrize(
    "kind",
    [
        "wrong_call",
        "bank_tool",
        "duplicate_summary",
        "missing_summary",
        "missing_context",
        "duplicate_tag",
        "wrong_evidence",
        "duplicate_item",
        "wrong_obligation",
    ],
)
def test_fails_closed_on_ambiguous_or_unexpected_structure(kind):
    request, edits = fixture()
    if kind == "wrong_call":
        edits["plan_call_id"] = "other"
    elif kind == "bank_tool":
        request["messages"][2]["content"][0]["tool_name"] = "bank_mutation"
    elif kind == "duplicate_summary":
        request["messages"].append(deepcopy(request["messages"][1]))
    elif kind == "missing_summary":
        request["messages"].pop(1)
    elif kind == "missing_context":
        request["messages"].pop()
    elif kind == "duplicate_tag":
        request["messages"][-1]["content"][0]["text"] += probe.OPEN_TAG
    elif kind == "wrong_evidence":
        edits["plan"][0]["before"] = "Other"
    elif kind == "duplicate_item":
        plan = request["messages"][2]["content"][0]["arguments"]
        plan["items"].append(deepcopy(plan["items"][0]))
    elif kind == "wrong_obligation":
        edits["plan"][1]["obligation_id"] = "other"
    with pytest.raises(ValueError):
        probe.intervene(request, edits)


def test_development_spec_does_not_add_missing_answer_facts_or_change_workflow_fields():
    edits = probe.intervention_spec()
    assert len(edits["summary"]) == 2 and len(edits["plan"]) == 6
    for edit in [*edits["summary"], *edits["plan"]]:
        assert all(term not in edit["after"] for term in ("Silver", "carbon", "10%", "0.45"))
        assert "status" not in edit and "goal" not in edit
