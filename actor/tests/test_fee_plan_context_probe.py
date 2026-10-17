import asyncio
import copy
import importlib.util
import json
from pathlib import Path

import pytest
from cayu import ScriptedModelProvider
from cayu.providers.base import ModelStreamEvent

spec = importlib.util.spec_from_file_location(
    "probe_fee_plan_context", Path(__file__).parents[1] / "scripts/probe_fee_plan_context.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def payload():
    plan = {
        "items": [
            {
                "id": parent,
                "status": "in_progress",
                "obligations": [
                    {
                        "id": child,
                        "status": "ready",
                        "evidence": "Old claim for " + parent,
                        "evidence_refs": [
                            {"kind": "customer_message", "reference_id": "customer:1"}
                        ],
                    }
                ],
            }
            for parent, child in sorted(probe.TARGETS)
        ]
    }
    return {
        "model": "gpt-5.6-sol",
        "reasoning": {"effort": "xhigh"},
        "instructions": "Original rules",
        "tools": [{"name": "mutate", "parameters": {"type": "object", "properties": {}}}],
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Customer's unchanged request"}],
            },
            {"type": "reasoning", "encrypted_content": "opaque fixture"},
            {
                "type": "function_call",
                "name": "controller_update_work_plan",
                "arguments": json.dumps(plan),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "<banking_context><research-capacity>0</research-capacity><workflow-plan>"
                        + json.dumps(plan)
                        + "</workflow-plan></banking_context>",
                    }
                ],
            },
        ],
    }


DOC = {
    "id": "public-fixture",
    "title": "Card conditions",
    "content": "Annual fee: 0. Approval required.",
}


def test_factorial_changes_only_intended_fields_and_preserves_source():
    original = payload()
    before = copy.deepcopy(original)
    arms = probe.variants(original, DOC)
    assert original == before and arms["baseline"] == before
    for arm in arms.values():
        assert {k: v for k, v in arm.items() if k != "input"} == {
            k: v for k, v in before.items() if k != "input"
        }
        assert arm["input"][:2] == before["input"][:2]
    assert arms["fee_only"]["input"][2] == before["input"][2]
    assert DOC["content"] in probe.memory_block(arms["fee_only"])["text"]
    assert probe.NEUTRAL not in json.dumps(arms["fee_only"])
    plan = json.loads(arms["plan_only"]["input"][2]["arguments"])
    original_plan = json.loads(before["input"][2]["arguments"])
    for item, old in zip(plan["items"], original_plan["items"], strict=True):
        item["obligations"][0]["evidence"] = old["obligations"][0]["evidence"]
    assert plan == original_plan
    assert arms["both"] == probe.neutralize_plan(arms["fee_only"])


def test_plan_change_is_fail_closed_for_missing_inconsistent_or_repeated_target():
    with pytest.raises(ValueError):
        probe.neutralize_plan(probe.neutralize_plan(payload()))
    p = payload()
    p["input"][2]["arguments"] = '{"items": []}'
    with pytest.raises(ValueError):
        probe.neutralize_plan(p)
    p = payload()
    probe.memory_block(p)["text"] = probe.memory_block(p)["text"].replace(
        "Old claim", "Different claim", 1
    )
    with pytest.raises(ValueError):
        probe.neutralize_plan(p)


@pytest.mark.parametrize("document", [None, {}, {**DOC, "content": ""}])
def test_incomplete_document_rejected(document):
    with pytest.raises(ValueError):
        probe.restore_fee(payload(), document)


def test_ambiguous_or_repeated_source_injection_rejected():
    with pytest.raises(ValueError):
        probe.restore_fee(probe.restore_fee(payload(), DOC), DOC)
    p = payload()
    p["input"].append(copy.deepcopy(p["input"][-1]))
    with pytest.raises(ValueError):
        probe.restore_fee(p, DOC)


def test_eight_preregistered_requests_and_unchanged_baseline(tmp_path):
    capture = tmp_path / "capture.json"
    document = tmp_path / "document.json"
    capture.write_text(
        json.dumps(
            {
                "schema": "cayu-tau3.provider-request.v1",
                "omitted_field_count": 0,
                "payload": payload(),
            }
        )
    )
    document.write_text(json.dumps(DOC))
    root = probe.prepare(capture, document, tmp_path)
    protocol = json.loads((root / "protocol.json").read_text())
    assert [r["arm"] for r in protocol["schedule"]] == list(probe.ARMS) + list(reversed(probe.ARMS))
    assert json.loads((root / "0-baseline-request.json").read_text()) == payload()
    assert all(
        probe.digest(root / f"{r['name']}-request.json") == r["request_sha256"]
        for r in protocol["schedule"]
    )
    assert not (root / "started.json").exists()


def test_proposed_tools_are_captured_not_executed():
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(name="mutate", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    result = asyncio.run(probe.decide(provider, payload()))
    assert len(result["calls"]) == 1 and result["tools_executed"] == 0


def test_incomplete_stream_is_not_a_decision():
    class IncompleteProvider:
        async def runtime_stream(self, request):
            if False:
                yield

    provider = IncompleteProvider()
    with pytest.raises(ValueError):
        asyncio.run(probe.decide(provider, payload()))


def test_error_event_is_not_graded_as_a_model_decision():
    class ErrorProvider:
        async def runtime_stream(self, request):
            yield ModelStreamEvent(type="error", payload={"error_type": "fixture_error"})

    with pytest.raises(probe.ProbeProviderError):
        asyncio.run(probe.decide(ErrorProvider(), payload()))


def test_request_has_a_hard_time_bound():
    class SlowProvider:
        async def runtime_stream(self, request):
            await asyncio.sleep(1)
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    with pytest.raises(TimeoutError):
        asyncio.run(probe.decide(SlowProvider(), payload(), timeout=0.01))
