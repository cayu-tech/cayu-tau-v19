import copy
import json
import sys
from unittest.mock import patch

import pytest
from test_lean_context_probe import load_script

with patch.dict(sys.modules, {"probe_fee_plan_context": load_script("probe_fee_plan_context")}):
    probe = load_script("probe_native_tools")


def payload():
    return {
        "model": "fixture",
        "reasoning": {"effort": "xhigh"},
        "input": [{"role": "user", "content": "All evidence unchanged"}],
        "instructions": "Same",
        "tools": [
            {"name": name, "parameters": {"type": "object"}} for name in sorted(probe.HELPERS)
        ]
        + [{"name": "banking_tool", "parameters": {"type": "object"}}],
    }


def test_only_helper_schemas_removed():
    original = payload()
    before = copy.deepcopy(original)
    result = probe.native_tools(original)
    assert original == before
    assert result == {**before, "tools": [before["tools"][-1]]}


def test_missing_duplicate_or_only_helpers_rejected():
    p = payload()
    p["tools"].pop(0)
    with pytest.raises(ValueError):
        probe.native_tools(p)
    p = payload()
    p["tools"].append(copy.deepcopy(p["tools"][0]))
    with pytest.raises(ValueError):
        probe.native_tools(p)
    p = payload()
    p["tools"].pop()
    with pytest.raises(ValueError):
        probe.native_tools(p)


def test_parent_must_complete_and_exactly_two_requests_registered(tmp_path):
    p = tmp_path / "parent"
    p.mkdir()
    (p / "completed.json").write_text(json.dumps({"scheduled_calls_completed": 5}))
    with pytest.raises(ValueError):
        probe.prepare(p)
    (p / "completed.json").write_text(json.dumps({"scheduled_calls_completed": 6}))
    (p / "0-lean-request.json").write_text(json.dumps(payload()))
    root = probe.prepare(p)
    protocol = json.loads((root / "protocol.json").read_text())
    assert len(protocol["schedule"]) == 2
    for row in protocol["schedule"]:
        assert probe.digest(root / (row["name"] + "-request.json")) == row["request_sha256"]
    assert not (root / "started.json").exists()
