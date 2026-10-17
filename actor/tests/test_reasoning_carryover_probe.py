import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "probe_reasoning_carryover", Path(__file__).parents[1] / "scripts/probe_reasoning_carryover.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_only_whole_opaque_items_removed_not_visible_text_or_new_effort():
    payload = {
        "instructions": "same",
        "model": "same",
        "reasoning": {"effort": "xhigh"},
        "input": [
            {"type": "reasoning", "encrypted_content": "opaque fixture"},
            {"role": "user", "content": "Visible evidence"},
            {"type": "function_call", "arguments": "same"},
            {"type": "function_call_output", "output": "same"},
        ],
    }
    before = copy.deepcopy(payload)
    result = probe.fresh_state(payload)
    assert payload == before
    assert result == {**before, "input": before["input"][1:]}
    with pytest.raises(ValueError):
        probe.fresh_state(result)


def test_requires_completed_parent_and_registers_four_calls(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "completed.json").write_text(json.dumps({"scheduled_calls_completed": 7}))
    with pytest.raises(ValueError):
        probe.prepare(parent)
    (parent / "completed.json").write_text(json.dumps({"scheduled_calls_completed": 8}))
    payload = {
        "model": "fixture",
        "reasoning": {"effort": "xhigh"},
        "input": [
            {"type": "reasoning", "encrypted_content": "opaque"},
            {"role": "user", "content": "Same"},
        ],
    }
    (parent / "0-both-request.json").write_text(json.dumps(payload))
    root = probe.prepare(parent)
    protocol = json.loads((root / "protocol.json").read_text())
    assert [x["arm"] for x in protocol["schedule"]] == [
        "carryover",
        "fresh_reasoning",
        "fresh_reasoning",
        "carryover",
    ]
    assert protocol["opaque_items_removed_in_fresh_arm"] == 1
    assert all(
        probe.digest(root / f"{x['name']}-request.json") == x["request_sha256"]
        for x in protocol["schedule"]
    )
