import asyncio
from copy import deepcopy

import pytest
from cayu import ModelStreamEvent, ScriptedModelProvider
from cayu.providers.base import ModelRequest
from cayu.providers.openai import build_openai_payload
from cayu.providers.openai_subscription import OpenAISubscriptionProvider

from cayu_tau3_banking.summary_probe import paired_messages, response, validate_native_request


def test_native_reasoning_is_preserved_instead_of_using_portable_preflight():
    request = ModelRequest(
        model="gpt-5.6-sol",
        messages=[
            {
                "role": "assistant",
                "content": [{"type": "thinking", "text": "Native prior reasoning."}],
            },
            {"role": "user", "content": [{"type": "text", "text": "Continue."}]},
        ],
    )
    original = request.model_dump(mode="json")
    with pytest.raises(ValueError, match="provider state or thinking"):
        OpenAISubscriptionProvider.preflight_portable_messages(
            None, model=request.model, messages=request.messages, tools=request.tools
        )
    validate_native_request(request)
    assert request.model_dump(mode="json") == original


def fixture():
    return {
        "compaction": {"summary": "Keep facts. Incorrect ordering."},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Previous session context summary:\nKeep facts. Incorrect ordering.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Customer request."}]},
        ],
    }


def test_only_target_summary_passage_changes_and_input_is_untouched():
    source = fixture()
    original = deepcopy(source)
    baseline, edited = paired_messages(source, "Incorrect ordering.", "Correct ordering.")
    assert source == original
    assert baseline == original["messages"]
    expected = deepcopy(baseline)
    expected[0]["content"][0]["text"] = expected[0]["content"][0]["text"].replace(
        "Incorrect", "Correct"
    )
    assert edited == expected
    for messages in (baseline, edited):
        request = ModelRequest(model="gpt-5.6-sol", messages=messages, tools=[])
        build_openai_payload(request, stream=True, reasoning_state="inline")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_prefix"])
def test_fail_closed_on_ambiguous_or_missing_summary(mutation):
    source = fixture()
    if mutation == "missing":
        source["messages"] = source["messages"][1:]
    elif mutation == "duplicate":
        source["messages"].append(deepcopy(source["messages"][0]))
    else:
        source["messages"][0]["content"][0]["text"] = "Keep facts. Incorrect ordering."
    with pytest.raises(ValueError, match="complete summary"):
        paired_messages(source, "Incorrect ordering.", "Correct ordering.")


def test_stream_is_text_only_and_requires_completion():
    request = ModelRequest(model="test", messages=[])
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("Hello"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    assert asyncio.run(response(provider, request)) == "Hello"
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(name="mutate", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    with pytest.raises(ValueError, match="tool call"):
        asyncio.run(response(provider, request))
