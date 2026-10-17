from __future__ import annotations

import json

import pytest
from cayu import Event, EventType
from tau2.data_model.message import AssistantMessage

from cayu_tau3_banking.runtime_usage import RuntimeUsageAccounting


def completion(*, input_tokens=10, output_tokens=3, missing=False, auxiliary=False):
    payload = (
        {}
        if missing
        else {
            "usage_metrics": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cache": {"cached_input_tokens": 2},
                "reasoning_output_tokens": 1,
            },
        }
    )
    if not auxiliary:
        payload["transcript_cursor"] = 2
    return Event(type=EventType.MODEL_COMPLETED, session_id="test", payload=payload)


def test_interval_usage_is_not_repeated_on_multiple_tool_callbacks():
    accounting = RuntimeUsageAccounting()
    accounting.observe(completion())
    first = accounting.attach(AssistantMessage.text("first"))
    second = accounting.attach(AssistantMessage.text("second"))
    accounting.observe(completion(input_tokens=20, output_tokens=5))
    third = accounting.attach(AssistantMessage.text("third"))
    assert first.usage == {"prompt_tokens": 10, "completion_tokens": 3}
    assert second.usage == {"prompt_tokens": 0, "completion_tokens": 0}
    assert third.usage == {"prompt_tokens": 20, "completion_tokens": 5}
    assert accounting.snapshot()["input_tokens"] == 30
    assert accounting.snapshot()["cached_input_tokens"] == 4
    assert accounting.snapshot()["reasoning_output_tokens"] == 2
    assert first.cost is None


def test_missing_usage_remains_unknown_without_poisoning_later_complete_intervals():
    accounting = RuntimeUsageAccounting()
    accounting.observe(completion())
    accounting.observe(completion(missing=True))
    first = accounting.attach(AssistantMessage.text("first"))
    assert first.usage is None
    assert first.raw_data["cayu_runtime_usage"]["input_tokens"] == 10
    accounting.observe(completion(input_tokens=20))
    second = accounting.attach(AssistantMessage.text("second"))
    assert second.usage == {"prompt_tokens": 20, "completion_tokens": 3}
    assert not accounting.snapshot()["complete"]
    assert accounting.snapshot()["missing_usage_completions"] == 1


@pytest.mark.parametrize("kind", [EventType.MODEL_ERROR, EventType.MODEL_ATTEMPT_DISCARDED])
def test_failed_or_discarded_attempt_is_not_assumed_free(kind):
    accounting = RuntimeUsageAccounting()
    accounting.observe(Event(type=kind, session_id="test", payload={"secret": "do not export"}))
    accounting.observe(completion())
    reply = accounting.attach(AssistantMessage.text("reply"))
    assert reply.usage is None
    assert reply.raw_data["cayu_runtime_usage"]["uncertain_usage_events"] == 1
    assert "do not export" not in json.dumps(reply.raw_data)


def test_compactor_completion_is_included_and_does_not_finish_acting_call():
    accounting = RuntimeUsageAccounting()
    accounting.observe(Event(type=EventType.MODEL_STARTED, session_id="test"))
    accounting.observe(completion(auxiliary=True))
    assert accounting.snapshot()["model_in_flight"]
    assert not accounting.snapshot()["complete"]
    accounting.observe(completion(input_tokens=20))
    reply = accounting.attach(AssistantMessage.text("reply"))
    assert reply.usage == {"prompt_tokens": 30, "completion_tokens": 6}
    assert accounting.snapshot()["complete"]


def test_normalization_failure_does_not_fall_back_to_raw_provider_usage():
    event = completion()
    event.payload.update(usage_normalization_failed=True, usage={"input_tokens": 900})
    accounting = RuntimeUsageAccounting()
    accounting.observe(event)
    assert accounting.attach(AssistantMessage.text("reply")).usage is None
    assert accounting.snapshot()["input_tokens"] == 0


def test_invalid_normalized_usage_is_unknown_and_content_free():
    event = completion()
    event.payload["usage_metrics"]["input_tokens"] = "secret-invalid-counter"
    accounting = RuntimeUsageAccounting()
    accounting.observe(event)
    reply = accounting.attach(AssistantMessage.text("reply"))
    assert reply.usage is None
    assert "secret-invalid-counter" not in json.dumps(reply.raw_data)


def test_untracked_application_helpers_prevent_incomplete_tau_totals():
    accounting = RuntimeUsageAccounting(untracked_helpers=True)
    accounting.observe(completion())
    reply = accounting.attach(AssistantMessage.text("reply"))
    assert reply.usage is None
    assert accounting.snapshot()["input_tokens"] == 10
    assert accounting.snapshot()["untracked_helpers"]
