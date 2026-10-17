import asyncio
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
from cayu import Message
from cayu.providers.base import (
    ModelCompletion,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
)

spec = importlib.util.spec_from_file_location(
    "probe_model_efforts", Path(__file__).parents[1] / "scripts/probe_model_efforts.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def source():
    return ModelRequest(
        model="gpt-5.6-sol",
        messages=[
            Message.text("system", "One prompt"),
            Message.text("user", "Customer and public facts"),
        ],
        tools=[],
        options={
            "thinking": {"enabled": True, "effort": "xhigh"},
            "openai": {"parallel_tool_calls": False},
        },
    ).model_dump(mode="json")


def test_only_model_and_effort_change_and_original_is_preserved():
    original = source()
    before = deepcopy(original)
    for name, value in probe.variants(original).items():
        model, effort = probe.CONFIGURATIONS[name]
        assert value["model"] == model and value["options"]["thinking"]["effort"] == effort
        value["model"] = original["model"]
        value["options"]["thinking"]["effort"] = "xhigh"
        assert value == original
    assert original == before


@pytest.mark.parametrize("change", ["model", "effort", "duplicate", "late"])
def test_unexpected_baseline_rejected(change):
    raw = source()
    if change == "model":
        raw["model"] = "other"
    elif change == "effort":
        raw["options"]["thinking"]["effort"] = "medium"
    elif change == "duplicate":
        raw["messages"].append(raw["messages"][0])
    else:
        raw["messages"].reverse()
    with pytest.raises(ValueError):
        probe.variants(raw)


class Provider:
    def __init__(self, events):
        self.events = events

    async def runtime_stream(self, request):
        for e in self.events:
            yield e


def test_terminal_usage_is_read_from_payload_not_finish_metadata():
    usage = {"input_tokens": 200, "output_tokens": 50}
    provider = Provider(
        [
            ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, delta="Answer"),
            ModelStreamEvent(
                type=ModelStreamEventType.COMPLETED,
                payload={"usage": usage, "model": "gpt-6-astra"},
                completion=ModelCompletion(finish_reason="stop"),
            ),
        ]
    )
    result = asyncio.run(probe.decide(provider, ModelRequest.model_validate(source())))
    assert result["usage"] == usage and result["returned_model"] == "gpt-6-astra"
    assert result["text"] == "Answer" and result["tools_executed"] == 0


@pytest.mark.parametrize("finish", ["length", "error", "unknown"])
def test_truncated_or_failed_output_is_not_a_completed_decision(finish):
    provider = Provider(
        [
            ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, delta="Partial"),
            ModelStreamEvent(
                type=ModelStreamEventType.COMPLETED,
                completion=ModelCompletion(finish_reason=finish),
            ),
        ]
    )
    with pytest.raises(ValueError, match="nontruncated"):
        asyncio.run(probe.decide(provider, ModelRequest.model_validate(source())))
