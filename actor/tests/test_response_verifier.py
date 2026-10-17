from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from cayu import MessageRole
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent

from cayu_tau3_banking.response_verifier import (
    BankingResponseVerifier,
    ResponseReviewCorrection,
    ResponseVerifierError,
)


class ScriptedProvider(ModelProvider):
    name = "scripted"

    def __init__(self, scripts: list[list[ModelStreamEvent]]) -> None:
        self.scripts = list(scripts)
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self.scripts.pop(0):
            yield event


def completed() -> ModelStreamEvent:
    return ModelStreamEvent.completed(
        {
            "model": "gpt-test",
            "status": "completed",
            "usage": {"input_tokens": 31, "output_tokens": 5},
            "provider_state": [],
            "finish_reason": "stop",
            "end_turn": True,
        }
    )


def run(coroutine):
    return asyncio.run(coroutine)


def review(verifier: BankingResponseVerifier):
    return run(
        verifier.review(
            customer_messages=("Please handle both requests.",),
            observation_context="<observation-ledger>{}</observation-ledger>",
            response_state={
                "plan_revision": 2,
                "items": [
                    {"id": "first", "response_kind": "ask_customer"},
                    {"id": "second", "response_kind": "ask_customer"},
                ],
            },
            proposed_response="Please provide the first and second missing values.",
        )
    )


def test_response_verifier_returns_closed_decision_and_usage() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="review-1",
                    name="submit_banking_response_review",
                    arguments={
                        "accepted": False,
                        "correction": "missing_required_item",
                        "affected_item_ids": ["second"],
                    },
                ),
                completed(),
            ]
        ]
    )
    verifier = BankingResponseVerifier(
        provider=provider,
        model="gpt-test",
        options={"openai": {"tool_choice": "required"}},
    )

    decision = review(verifier)

    assert decision.accepted is False
    assert decision.correction is ResponseReviewCorrection.MISSING_REQUIRED_ITEM
    assert decision.affected_item_ids == ("second",)
    assert decision.prompt_tokens == 31
    assert decision.completion_tokens == 5
    request = provider.requests[0]
    assert [message.role for message in request.messages] == [MessageRole.SYSTEM, MessageRole.USER]
    assert "human_transfer_request_policy is authoritative" in request.messages[0].content[0].text
    assert [tool["name"] for tool in request.tools] == ["submit_banking_response_review"]
    assert request.tools[0]["input_schema"]["properties"]["affected_item_ids"] == {
        "type": "array",
        "items": {"type": "string", "enum": ["first", "second"]},
        "maxItems": 2,
        "uniqueItems": True,
    }
    assert request.options["openai"]["tool_choice"] == "required"


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "accepted": True,
            "correction": "missing_required_item",
            "affected_item_ids": [],
        },
        {"accepted": False, "correction": "none", "affected_item_ids": ["first"]},
        {"accepted": True, "correction": "unknown", "affected_item_ids": []},
        {"accepted": True, "correction": "none", "affected_item_ids": ["first"]},
        {"accepted": False, "correction": "missing_required_item", "affected_item_ids": []},
        {
            "accepted": False,
            "correction": "missing_required_item",
            "affected_item_ids": ["unknown"],
        },
        {"accepted": True, "correction": "none"},
    ],
)
def test_response_verifier_rejects_inconsistent_or_invalid_decisions(arguments) -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="review-1",
                    name="submit_banking_response_review",
                    arguments=arguments,
                ),
                completed(),
            ]
        ]
    )

    with pytest.raises(ResponseVerifierError):
        review(BankingResponseVerifier(provider=provider, model="gpt-test"))


def test_response_verifier_rejects_multiple_review_tool_calls() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="review-1",
                    name="submit_banking_response_review",
                    arguments={"accepted": True, "correction": "none", "affected_item_ids": []},
                ),
                ModelStreamEvent.tool_call(
                    id="review-2",
                    name="submit_banking_response_review",
                    arguments={"accepted": True, "correction": "none", "affected_item_ids": []},
                ),
                completed(),
            ]
        ]
    )

    with pytest.raises(ResponseVerifierError, match="exactly one"):
        review(BankingResponseVerifier(provider=provider, model="gpt-test"))
