import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "audit_frozen_task", Path(__file__).parents[1] / "scripts/audit_frozen_task.py"
)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fixture(root):
    write(root / "protocol.json", {"task_ids": ["task_fixture"]})
    write(root / "source-manifest.json", {"files": {}})
    task = root / "task_fixture"
    write(task / "assessment.json", {"rows": []})
    messages = [
        {"role": "user", "content": "Advice only"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call-1", "name": "read", "arguments": {"id": "x"}}],
        },
        {"role": "tool", "id": "call-1", "content": '{"value": 7}', "error": False},
        {"role": "assistant", "content": "The value is 7."},
    ]
    write(
        task / "official/results.json",
        {
            "simulations": [
                {
                    "task_id": "task_fixture",
                    "duration": 10,
                    "termination_reason": "user_stop",
                    "reward_info": {"reward": 1},
                    "messages": messages,
                }
            ]
        },
    )
    call = {
        "type": "tool_call",
        "tool_name": "cayu_read_product_documents",
        "tool_call_id": "packet",
        "arguments": {"product_id": "fixture/product"},
    }
    for name in ("first", "later"):
        write(task / f"contexts/context/{name}.json", {"messages": [{"content": [call]}]})
    return task


def test_extracts_actual_results_and_deduplicates_captured_calls(tmp_path):
    task = fixture(tmp_path)
    result = audit_module.audit(tmp_path, "task_fixture")
    assert result["source_unchanged"]
    assert result["external_calls"][0]["result"] == {"value": 7}
    assert [m["text"] for m in result["dialogue"]] == ["Advice only", "The value is 7."]
    assert list(result["product_packet_calls"]) == ["packet"]
    assert (task / "post-run-evidence.json").is_file()
    with pytest.raises(FileExistsError):
        audit_module.audit(tmp_path, "task_fixture")


def test_refuses_running_or_unregistered_task(tmp_path):
    write(tmp_path / "protocol.json", {"task_ids": ["task_fixture"]})
    with pytest.raises(ValueError, match="completed assessment"):
        audit_module.audit(tmp_path, "task_fixture")
    with pytest.raises(ValueError, match="registration"):
        audit_module.audit(tmp_path, "unselected")


def test_rejects_changed_source_before_producing_audit(tmp_path):
    task = fixture(tmp_path)
    write(tmp_path / "source-manifest.json", {"files": {"protocol.json": "wrong"}})
    with pytest.raises(ValueError, match="Frozen file changed"):
        audit_module.audit(tmp_path, "task_fixture")
    assert not (task / "post-run-evidence.json").exists()


def test_rejects_conflicting_call_identity_in_contexts(tmp_path):
    task = fixture(tmp_path)
    write(
        task / "contexts/context/conflict.json",
        {
            "messages": [
                {
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_name": "cayu_read_product_documents",
                            "tool_call_id": "packet",
                            "arguments": {"product_id": "different/product"},
                        }
                    ]
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="Inconsistent call identity"):
        audit_module.audit(tmp_path, "task_fixture")
    assert not (task / "post-run-evidence.json").exists()
