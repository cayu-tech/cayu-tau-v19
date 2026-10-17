import asyncio
import json
import stat

import pytest
from cayu import ModelStreamEvent, ScriptedModelProvider
from cayu.providers.openai import build_openai_payload

from cayu_tau3_banking.decision_probe import (
    CASES,
    PROCEDURE,
    build_request,
    decide,
    save_private_json,
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_pair_changes_only_procedure_availability(case):
    absent = build_request(case, procedure_present=False, system_prompt="unchanged")
    present = build_request(case, procedure_present=True, system_prompt="unchanged")
    left, right = absent.model_dump(mode="json"), present.model_dump(mode="json")
    assert left.pop("messages") != right.pop("messages")
    assert left == right
    assert absent.messages[0] == present.messages[0]
    assert absent.messages[2] == present.messages[2]
    serialized = present.model_dump_json()
    assert "expected_with_procedure" not in serialized
    assert PROCEDURE in present.messages[1].content[0].text


def test_output_is_private_and_cannot_overwrite(tmp_path):
    path = tmp_path / "request.json"
    save_private_json(path, {"value": 1})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        save_private_json(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


@pytest.mark.parametrize("present", [False, True])
def test_real_provider_payload_accepts_cayu_tool_schema_without_network(present):
    request = build_request(CASES[0], procedure_present=present, system_prompt="test")
    payload = build_openai_payload(request, stream=True, reasoning_state="inline")
    assert payload["tools"][0]["name"] == "record_next_step"
    assert payload["tools"][0]["parameters"]["required"] == ["step"]


def test_decision_is_scored_without_executing_tools():
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    name="record_next_step", arguments={"step": "search_procedure"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    request = build_request(CASES[0], procedure_present=False, system_prompt="test")
    assert asyncio.run(decide(provider, request)) == "search_procedure"


@pytest.mark.parametrize("step", ["invented", None, []])
def test_invalid_decision_is_an_error_not_a_scored_failure(step):
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(name="record_next_step", arguments={"step": step}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    request = build_request(CASES[0], procedure_present=False, system_prompt="test")
    with pytest.raises(ValueError, match="invalid decision"):
        asyncio.run(decide(provider, request))
