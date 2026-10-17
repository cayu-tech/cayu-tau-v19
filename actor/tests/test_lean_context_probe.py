import asyncio
import copy
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with patch.dict(sys.modules, {"probe_fee_plan_context": load_script("probe_fee_plan_context")}):
    probe = load_script("probe_lean_context")


def payload():
    return {
        "model": "gpt-5.6-sol",
        "reasoning": {"effort": "xhigh"},
        "instructions": "<instructions>Generic original instructions"
        + probe.INSTRUCTION_BOUNDARY
        + " extra controller guidance</instructions>\n<policy>Original bank policy</policy>",
        "tools": [{"name": "mutate", "parameters": {"type": "object"}}],
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Previous session context summary:\nCustomer constraints.\n## Observed facts\nAll observed product facts."
                        + probe.SUMMARY_BOUNDARY
                        + "Old plan status.\n## Pending work\nOld next steps.\n## Tentative proposals—not required procedures\nOld preferred answer.",
                    }
                ],
            },
            {"type": "function_call", "call_id": "source", "name": "read", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "source",
                "output": "Unchanged document body",
            },
            {
                "type": "function_call",
                "call_id": "plan",
                "name": "controller_update_work_plan",
                "arguments": "{}",
            },
            {"type": "function_call_output", "call_id": "plan", "output": "accepted"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "<banking_context><research-capacity>0</research-capacity><workflow-plan>Old plan</workflow-plan><observation-ledger>Exact customer messages, effects and pending calls</observation-ledger><active-work-evidence>Complete source text</active-work-evidence><restored-public-source>Complete restored fee text</restored-public-source></banking_context>",
                    }
                ],
            },
        ],
    }


def test_only_registered_sections_change():
    original = payload()
    before = copy.deepcopy(original)
    arms = probe.variants(original)
    assert original == before == arms["full"]
    for arm in (arms["plan_reduced"], arms["lean"]):
        assert arm["tools"] == before["tools"]
        assert arm["reasoning"] == before["reasoning"]
        assert arm["input"][1:3] == before["input"][1:3]
        assert (
            arm["input"][0]["content"][0]["text"]
            == before["input"][0]["content"][0]["text"].split(probe.SUMMARY_BOUNDARY)[0]
        )
        assert probe.memory_block(arm)["text"] == probe.memory_block(before)["text"].replace(
            "<workflow-plan>Old plan</workflow-plan>", ""
        )
        assert len(arm["input"]) == len(before["input"]) - 2
    assert arms["plan_reduced"]["instructions"] == before["instructions"]
    assert arms["lean"]["input"] == arms["plan_reduced"]["input"]
    assert (
        arms["lean"]["instructions"]
        == "<instructions>Generic original instructions\n</instructions>\n<policy>Original bank policy</policy>"
    )


@pytest.mark.parametrize(
    "change",
    [
        "reasoning",
        "missing_summary",
        "missing_boundary",
        "missing_observations",
        "duplicate_call",
        "missing_output",
        "duplicate_output",
        "duplicate_plan",
    ],
)
def test_malformed_or_ambiguous_input_rejected(change):
    p = payload()
    if change == "reasoning":
        p["input"].append({"type": "reasoning", "encrypted_content": "fixture"})
    elif change == "missing_summary":
        p["input"].pop(0)
    elif change == "missing_boundary":
        p["input"][0]["content"][0]["text"] = "Previous session context summary:\nIncomplete"
    elif change == "missing_observations":
        p["input"][0]["content"][0]["text"] = p["input"][0]["content"][0]["text"].replace(
            "## Observed facts", "different"
        )
    elif change == "duplicate_call":
        p["input"].append(copy.deepcopy(p["input"][3]))
    elif change == "missing_output":
        p["input"].pop(4)
    elif change == "duplicate_output":
        p["input"].append(copy.deepcopy(p["input"][4]))
    else:
        probe.memory_block(p)["text"] += "<workflow-plan>Another plan</workflow-plan>"
    with pytest.raises(ValueError):
        probe.reduce_plan(p)


@pytest.mark.parametrize(
    "instruction",
    [
        "<instructions>Unknown structure</instructions><policy>Same</policy>",
        "<instructions>Same",
        "<instructions>x</instructions><instructions>y</instructions>",
    ],
)
def test_unknown_instruction_structure_rejected(instruction):
    p = payload()
    p["instructions"] = instruction
    with pytest.raises(ValueError):
        probe.reduce_instructions(p)


def prepare_fixture(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "completed.json").write_text(json.dumps({"scheduled_calls_completed": 4}))
    (parent / "0-fresh_reasoning-request.json").write_text(json.dumps(payload()))
    return probe.prepare(parent, tmp_path)


def test_six_requests_frozen_before_any_dispatch(tmp_path):
    root = prepare_fixture(tmp_path)
    protocol = json.loads((root / "protocol.json").read_text())
    assert [r["arm"] for r in protocol["schedule"]] == list(probe.ARMS) + list(reversed(probe.ARMS))
    assert all(
        probe.digest(root / (r["name"] + "-request.json")) == r["request_sha256"]
        for r in protocol["schedule"]
    )
    assert json.loads((root / "0-full-request.json").read_text()) == payload()
    assert not (root / "started.json").exists()


def test_modified_request_rejected_before_provider_creation(tmp_path, monkeypatch):
    root = prepare_fixture(tmp_path)
    (root / "0-full-request.json").write_text("{}")
    import cayu_tau3_banking.cli as cli

    def forbidden(**kwargs):
        pytest.fail("Provider must not be created for changed input")

    monkeypatch.setattr(cli, "build_provider", forbidden)
    with pytest.raises(ValueError, match="Prepared arm changed"):
        asyncio.run(probe.run(root))
    assert not (root / "started.json").exists()
