from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from cayu import MessageRole
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent

from cayu_tau3_banking.plan_verifier import (
    BankingPlanVerifier,
    PlanReviewCorrection,
    PlanVerifierError,
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
            "usage": {"input_tokens": 23, "output_tokens": 5},
            "provider_state": [],
            "finish_reason": "stop",
            "end_turn": True,
        }
    )


def run(coroutine):
    return asyncio.run(coroutine)


def review(
    verifier: BankingPlanVerifier,
    *,
    reclassifiable_item_ids: tuple[str, ...] = ("compare",),
    dialogue_context: str = "",
):
    return run(
        verifier.review(
            customer_messages=("Please compare the options and apply for my selection.",),
            plan_state={
                "revision": 1,
                "items": [
                    {
                        "id": "compare",
                        "goal": "Compare the available options",
                        "completion_mode": "information",
                        "status": "completed",
                    }
                ],
            },
            reclassifiable_item_ids=reclassifiable_item_ids,
            dialogue_context=dialogue_context,
        )
    )


@pytest.mark.parametrize("context", [None, 17, "x" * 24_001])
def test_plan_verifier_rejects_invalid_dialogue_before_provider_call(context):
    provider = ScriptedProvider([])
    verifier = BankingPlanVerifier(provider=provider, model="gpt-test")
    with pytest.raises(ValueError, match="dialogue_context"):
        review(verifier, dialogue_context=context)
    assert not provider.requests


def test_plan_verifier_delivers_context_without_interpreting_customer_language():
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="review-1",
                    name="submit_banking_plan_review",
                    arguments={
                        "accepted": True,
                        "correction": "none",
                        "affected_item_ids": [],
                        "required_new_item_count": 0,
                    },
                ),
                completed(),
            ]
        ]
    )
    verifier = BankingPlanVerifier(provider=provider, model="gpt-test")
    context = json.dumps(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Are you asking about the requirements, not applying?",
                },
                {"role": "customer", "content": "Yes, information only."},
            ]
        }
    )
    decision = review(verifier, dialogue_context=context)
    assert decision.accepted
    payload = json.loads(provider.requests[0].messages[1].content[0].text)
    assert payload["dialogue_context"] == context
    assert "apply" in payload["customer_messages"][0]


def test_actor_and_reviewer_instructions_do_not_impose_unrequested_acquisition():
    # This guards the instruction contract, not live model semantic accuracy.
    from cayu_tau3_banking.agent import _AGENT_INSTRUCTIONS
    from cayu_tau3_banking.plan_verifier import _SYSTEM_PROMPT

    assert "do not by themselves request execution" in _AGENT_INSTRUCTIONS
    assert "do not by themselves request execution" in _SYSTEM_PROMPT
    assert "including a conditional request" in _AGENT_INSTRUCTIONS
    assert "including a conditional request" in _SYSTEM_PROMPT
    assert "conservative classification" not in _SYSTEM_PROMPT
    assert "Treat a request to find or acquire" not in _AGENT_INSTRUCTIONS


def test_plan_verifier_returns_one_normalized_structured_decision() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="review-1",
                    name="submit_banking_plan_review",
                    arguments={
                        "accepted": False,
                        "correction": "missing_operation_outcome",
                        "affected_item_ids": [],
                        "required_new_item_count": 2,
                    },
                ),
                completed(),
            ]
        ]
    )
    verifier = BankingPlanVerifier(
        provider=provider,
        model="gpt-test",
        options={
            "thinking": {"enabled": True, "effort": "low"},
            "openai": {"parallel_tool_calls": False, "tool_choice": "required"},
        },
    )

    decision = review(verifier)

    assert decision.accepted is False
    assert decision.correction is PlanReviewCorrection.MISSING_OPERATION_OUTCOME
    assert decision.affected_item_ids == ()
    assert decision.required_new_item_count == 2
    assert decision.prompt_tokens == 23
    assert decision.completion_tokens == 5
    request = provider.requests[0]
    assert [message.role for message in request.messages] == [MessageRole.SYSTEM, MessageRole.USER]
    assert [tool["name"] for tool in request.tools] == ["submit_banking_plan_review"]
    assert request.tools[0]["input_schema"]["properties"]["affected_item_ids"] == {
        "type": "array",
        "items": {"type": "string", "enum": ["compare"]},
        "maxItems": 1,
        "uniqueItems": True,
    }
    payload = request.messages[1].content[0].text
    assert '"reclassifiable_item_ids":["compare"]' in payload
    assert request.options == {
        "thinking": {"enabled": True, "effort": "low"},
        "openai": {"parallel_tool_calls": False, "tool_choice": "required"},
    }


def test_plan_verifier_uses_a_valid_closed_schema_without_reclassification_authority() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="review-1",
                    name="submit_banking_plan_review",
                    arguments={
                        "accepted": True,
                        "correction": "none",
                        "affected_item_ids": [],
                        "required_new_item_count": 0,
                    },
                ),
                completed(),
            ]
        ]
    )

    decision = review(
        BankingPlanVerifier(provider=provider, model="gpt-test"),
        reclassifiable_item_ids=(),
    )

    assert decision.accepted is True
    affected_schema = provider.requests[0].tools[0]["input_schema"]["properties"][
        "affected_item_ids"
    ]
    assert affected_schema == {
        "type": "array",
        "items": {"type": "string"},
        "maxItems": 0,
        "uniqueItems": True,
    }
    payload = provider.requests[0].messages[1].content[0].text
    assert '"reclassifiable_item_ids":[]' in payload


def test_plan_verifier_rejects_wrong_mode_for_an_item_outside_revision_authority() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="review-1",
                    name="submit_banking_plan_review",
                    arguments={
                        "accepted": False,
                        "correction": "wrong_completion_mode_to_operation",
                        "affected_item_ids": ["compare"],
                        "required_new_item_count": 0,
                    },
                ),
                completed(),
            ]
        ]
    )

    with pytest.raises(PlanVerifierError, match="invalid decision fields"):
        review(
            BankingPlanVerifier(provider=provider, model="gpt-test"),
            reclassifiable_item_ids=(),
        )


def test_plan_verifier_rejects_unstructured_or_ambiguous_provider_output() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="review-1",
                    name="submit_banking_plan_review",
                    arguments={
                        "accepted": True,
                        "correction": "none",
                        "affected_item_ids": [],
                        "required_new_item_count": 0,
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="review-2",
                    name="submit_banking_plan_review",
                    arguments={
                        "accepted": False,
                        "correction": "wrong_completion_mode_to_operation",
                        "affected_item_ids": ["compare"],
                        "required_new_item_count": 0,
                    },
                ),
                completed(),
            ]
        ]
    )

    with pytest.raises(PlanVerifierError, match="exactly one"):
        review(BankingPlanVerifier(provider=provider, model="gpt-test"))


def test_plan_verifier_accepts_only_with_no_correction() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="review-1",
                    name="submit_banking_plan_review",
                    arguments={
                        "accepted": True,
                        "correction": "none",
                        "affected_item_ids": [],
                        "required_new_item_count": 0,
                    },
                ),
                completed(),
            ]
        ]
    )

    decision = review(BankingPlanVerifier(provider=provider, model="gpt-test"))

    assert decision.accepted is True
    assert decision.correction is PlanReviewCorrection.NONE


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "accepted": True,
            "correction": "missing_operation_outcome",
            "affected_item_ids": [],
            "required_new_item_count": 1,
        },
        {
            "accepted": False,
            "correction": "missing_operation_outcome",
            "affected_item_ids": [],
            "required_new_item_count": 0,
        },
        {
            "accepted": False,
            "correction": "missing_operation_outcome",
            "affected_item_ids": ["compare"],
            "required_new_item_count": 1,
        },
        {
            "accepted": False,
            "correction": "wrong_completion_mode_to_operation",
            "affected_item_ids": [],
            "required_new_item_count": 0,
        },
        {
            "accepted": False,
            "correction": "wrong_completion_mode_to_operation",
            "affected_item_ids": ["unknown"],
            "required_new_item_count": 0,
        },
    ],
)
def test_plan_verifier_rejects_inconsistent_acceptance_and_correction(arguments) -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="review-1",
                    name="submit_banking_plan_review",
                    arguments=arguments,
                ),
                completed(),
            ]
        ]
    )

    with pytest.raises(PlanVerifierError, match="decision"):
        review(BankingPlanVerifier(provider=provider, model="gpt-test"))
