import asyncio
import json

import pytest
from cayu import ModelStreamEvent, ScriptedModelProvider
from cayu.providers.openai import build_openai_payload

from cayu_tau3_banking.policy_grounding_probe import (
    CASES,
    Decision,
    ProbeProviderError,
    assess,
    build_request,
    decide,
    invalid_decision_result,
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_pair_changes_only_evidence(case):
    absent = build_request(case, system_prompt="fixed", active_context=None)
    present = build_request(case, system_prompt="fixed", active_context="Public exact revision")
    left, right = absent.model_dump(), present.model_dump()
    assert left.pop("messages") != right.pop("messages")
    assert left == right
    assert absent.messages[0] == present.messages[0]
    assert absent.messages[2] == present.messages[2]
    assert "expected_step" not in present.model_dump_json()
    payload = build_openai_payload(present, stream=True, reasoning_state="inline")
    assert payload["tools"][0]["name"] == "record_decision"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_correct_decisions_and_missing_evidence_scored_separately(case):
    expected = Decision(
        step=case.expected_step,
        credit_dollars=case.expected_credit,
        conditions=[],
        evidence_ids=["procedure"],
    )
    assert assess(case, expected, retained=True)
    assert not assess(case, expected, retained=False)
    search = Decision(step=case.missing_step, credit_dollars=None, conditions=[], evidence_ids=[])
    assert assess(case, search, retained=False)
    assert not assess(case, search, retained=True)
    assert not assess(
        case, expected.model_copy(update={"conditions": ["invented condition"]}), retained=True
    )


def test_live_event_contract_without_dispatch():
    result = dict(
        step="offer_retention", credit_dollars=17, conditions=[], evidence_ids=["procedure"]
    )
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(name="record_decision", arguments=result),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    request = build_request(CASES[1], system_prompt="fixed", active_context="Public exact revision")
    assert asyncio.run(decide(provider, request)).model_dump() == result


def test_provider_error_is_not_mislabeled_as_an_invalid_decision():
    class ErrorProvider:
        async def runtime_stream(self, request):
            yield ModelStreamEvent.error("private error message")

    provider = ErrorProvider()
    request = build_request(CASES[0], system_prompt="fixed", active_context=None)
    with pytest.raises(ProbeProviderError, match="Provider stream failed") as error:
        asyncio.run(decide(provider, request))
    assert "private error" not in str(error.value)


def test_invalid_model_proposal_is_captured_before_validation(tmp_path):
    from pydantic import ValidationError

    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(name="record_decision", arguments={"step": "invented"}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    request = build_request(CASES[0], system_prompt="fixed", active_context=None)
    path = tmp_path / "response.json"
    with pytest.raises(ValidationError):
        asyncio.run(decide(provider, request, capture_path=path))
    assert json.loads(path.read_text())["calls"][0]["arguments"] == {"step": "invented"}


def test_wrong_case_field_is_a_failed_result_not_silently_repaired():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as caught:
        Decision.model_validate(
            {
                "Step": "offer_retention",
                "credit_dollars": 17,
                "conditions": [],
                "evidence_ids": ["procedure"],
            }
        )
    row = invalid_decision_result("fixture", True, caught.value, 1.0)
    assert row["passed"] is False
    assert row["decision"] is None
    assert row["failure_kind"] == "invalid_decision_schema"
    assert {e["type"] for e in row["validation_errors"]} == {"missing", "extra_forbidden"}


def test_run_preserves_schema_failure_and_continues_without_retry(tmp_path, monkeypatch):
    from cayu_tau3_banking import cli
    from cayu_tau3_banking import policy_grounding_probe as probe

    class Provider:
        closed = False

        def preflight_portable_messages(self, **kwargs):
            pass

        async def aclose(self):
            self.closed = True

    provider = Provider()
    monkeypatch.setattr(cli, "build_provider", lambda **kwargs: provider)
    calls = []

    async def fake_decide(provider, request, *, capture_path):
        calls.append(capture_path.name)
        if len(calls) == 1:
            return Decision.model_validate({"Step": "search_procedure"})
        return Decision(
            step="search_procedure", credit_dollars=None, conditions=[], evidence_ids=[]
        )

    monkeypatch.setattr(probe, "decide", fake_decide)
    asyncio.run(probe.run(tmp_path))
    result = json.loads((tmp_path / "results.json").read_text())
    assert len(calls) == len(set(calls)) == result["total"] == 10
    assert result["rows"][0]["failure_kind"] == "invalid_decision_schema"
    assert result["rows"][0]["passed"] is False
    assert provider.closed
