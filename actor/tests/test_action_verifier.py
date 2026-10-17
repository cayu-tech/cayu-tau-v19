from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from cayu import MessageRole, ToolCallPart
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent

from cayu_tau3_banking.action_verifier import (
    ActionReviewCorrection,
    ActionVerifierError,
    BankingActionVerifier,
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
            "usage": {"input_tokens": 29, "output_tokens": 7},
            "provider_state": [],
            "finish_reason": "stop",
            "end_turn": True,
        }
    )


def run(coroutine):
    return asyncio.run(coroutine)


def calls() -> tuple[ToolCallPart, ...]:
    return (
        ToolCallPart(
            tool_call_id="call-1",
            tool_name="call_discoverable_agent_tool",
            arguments={
                "agent_tool_name": "open_bank_account_4821",
                "arguments": '{"user_id":"user-1","account_type":"savings"}',
            },
        ),
        ToolCallPart(
            tool_call_id="call-2",
            tool_name="call_discoverable_agent_tool",
            arguments={
                "agent_tool_name": "close_bank_account_7392",
                "arguments": '{"account_id":"account-1"}',
            },
        ),
    )


def review(verifier: BankingActionVerifier):
    return run(
        verifier.review(
            customer_messages=("Open savings, but do not close my existing account.",),
            observation_context="<observation-ledger>{}</observation-ledger>",
            workflow_context="<workflow-plan>{}</workflow-plan>",
            calls=calls(),
        )
    )


def test_action_verifier_returns_ordered_closed_decisions_and_usage() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="review-1",
                    name="submit_banking_action_review",
                    arguments={
                        "decisions": [
                            {"call_index": 0, "accepted": True, "correction": "none"},
                            {
                                "call_index": 1,
                                "accepted": False,
                                "correction": "customer_intent_mismatch",
                            },
                        ]
                    },
                ),
                completed(),
            ]
        ]
    )
    verifier = BankingActionVerifier(
        provider=provider,
        model="gpt-test",
        options={
            "thinking": {"enabled": True, "effort": "low"},
            "openai": {"parallel_tool_calls": False, "tool_choice": "required"},
        },
    )

    batch = review(verifier)

    assert [decision.accepted for decision in batch.decisions] == [True, False]
    assert batch.decisions[1].correction is ActionReviewCorrection.CUSTOMER_INTENT_MISMATCH
    assert batch.prompt_tokens == 29
    assert batch.completion_tokens == 7
    request = provider.requests[0]
    assert [message.role for message in request.messages] == [MessageRole.SYSTEM, MessageRole.USER]
    assert [tool["name"] for tool in request.tools] == ["submit_banking_action_review"]
    assert request.options["openai"]["tool_choice"] == "required"
    assert "deterministic Cayu audit results" in request.messages[0].content[0].text


def test_action_verifier_accepts_closed_premature_handoff_correction() -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="review-1",
                    name="submit_banking_action_review",
                    arguments={
                        "decisions": [
                            {
                                "call_index": 0,
                                "accepted": False,
                                "correction": "premature_handoff",
                            }
                        ]
                    },
                ),
                completed(),
            ]
        ]
    )
    verifier = BankingActionVerifier(provider=provider, model="gpt-test")
    transfer = ToolCallPart(
        tool_call_id="transfer-1",
        tool_name="transfer_to_human_agents",
        arguments={"summary": "Customer requested a human.", "reason": "other"},
    )

    batch = run(
        verifier.review(
            customer_messages=("Please perform the requested operation, then transfer me.",),
            observation_context="<observation-ledger>{}</observation-ledger>",
            workflow_context="<workflow-plan>{}</workflow-plan>",
            calls=(transfer,),
        )
    )

    assert batch.decisions[0].accepted is False
    assert batch.decisions[0].correction is ActionReviewCorrection.PREMATURE_HANDOFF


@pytest.mark.parametrize(
    "decisions",
    [
        [{"call_index": 0, "accepted": True, "correction": "none"}],
        [
            {"call_index": 1, "accepted": True, "correction": "none"},
            {"call_index": 0, "accepted": True, "correction": "none"},
        ],
        [
            {
                "call_index": 0,
                "accepted": True,
                "correction": "argument_semantic_mismatch",
            },
            {"call_index": 1, "accepted": True, "correction": "none"},
        ],
    ],
)
def test_action_verifier_rejects_wrong_count_order_or_inconsistent_decisions(decisions) -> None:
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="review-1",
                    name="submit_banking_action_review",
                    arguments={"decisions": decisions},
                ),
                completed(),
            ]
        ]
    )

    with pytest.raises(ActionVerifierError):
        review(BankingActionVerifier(provider=provider, model="gpt-test"))


def test_action_verifier_rejects_multiple_review_tool_calls() -> None:
    decision = {
        "decisions": [
            {"call_index": 0, "accepted": True, "correction": "none"},
            {"call_index": 1, "accepted": True, "correction": "none"},
        ]
    }
    provider = ScriptedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="review-1",
                    name="submit_banking_action_review",
                    arguments=decision,
                ),
                ModelStreamEvent.tool_call(
                    id="review-2",
                    name="submit_banking_action_review",
                    arguments=decision,
                ),
                completed(),
            ]
        ]
    )

    with pytest.raises(ActionVerifierError, match="exactly one"):
        review(BankingActionVerifier(provider=provider, model="gpt-test"))
