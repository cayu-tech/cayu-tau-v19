import asyncio
import dataclasses

import pytest
from cayu import ModelStreamEvent, ScriptedModelProvider
from cayu.providers.base import ModelStreamEventType
from cayu.providers.openai import build_openai_payload
from jsonschema import Draft202012Validator

from cayu_tau3_banking.argument_grounding_probe import (
    CASES,
    build_request,
    decide,
    validate_decision,
)
from cayu_tau3_banking.policy_grounding_probe import ProbeProviderError


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_expectations_do_not_enter_requests(case):
    changed = dataclasses.replace(case, expected={"hidden": "DO_NOT_EXPOSE"})
    assert build_request(case, "prompt") == build_request(changed, "prompt")
    request = build_request(case, "prompt")
    schema = request.tools[0]["input_schema"]
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(case.expected)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_valid_decisions(case):
    assert (
        validate_decision([{"name": "record_decision", "arguments": case.expected}], 1)
        == case.expected
    )


@pytest.mark.parametrize(
    "calls,completed",
    [
        ([], 1),
        ([{"name": "record_decision", "arguments": CASES[0].expected}], 0),
        ([{"name": "other", "arguments": CASES[0].expected}], 1),
        (
            [
                {
                    "name": "record_decision",
                    "arguments": {"decision": "execute", "arguments": {"x": 2}},
                }
            ],
            1,
        ),
        (
            [
                {
                    "name": "record_decision",
                    "arguments": {"decision": "execute", "arguments": {}, "extra": True},
                }
            ],
            1,
        ),
    ],
)
def test_invalid_response_rejected(calls, completed):
    with pytest.raises(ValueError):
        validate_decision(calls, completed)


def test_real_payload_and_stream_contract():
    request = build_request(CASES[0], "prompt")
    payload = build_openai_payload(request, stream=True, reasoning_state="inline")
    assert payload["tools"][0]["name"] == "record_decision"
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(name="record_decision", arguments=CASES[0].expected),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    assert asyncio.run(decide(provider, request)) == CASES[0].expected


def test_provider_error_is_not_misclassified_as_decision_failure():
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent(
                    type=ModelStreamEventType.ERROR, payload={"error_type": "APIError"}
                ),
                ModelStreamEvent.completed({"finish_reason": "error"}),
            ]
        ]
    )
    with pytest.raises(ProbeProviderError) as raised:
        asyncio.run(decide(provider, build_request(CASES[0], "prompt")))
    assert raised.value.diagnostics == {"error_type": "APIError"}
