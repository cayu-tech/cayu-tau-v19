"""Offline reproduction and verification of bounded partial compaction.

Uses fictional tool results and a scripted summarizer only. Re-entering the policy
below is a diagnostic, not a production retry implementation: runtime ownership,
checkpoint publication, spend accounting and cancellation belong to the runtime.
Use --expect-first-build success to verify merged continuation without re-entry.
"""

import argparse
import asyncio
import json

from cayu import AgentSpec, Message
from cayu.context.base import ContextBuildError, _estimate_model_facing_context_pressure
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runtime import (
    CheckpointCompactionContextPolicy,
    ContextRequest,
    InMemorySessionStore,
    ModelCompactor,
    RunRequest,
    SessionIdentity,
)


class ScriptedSummarizer(ModelProvider):
    name = "partial-compaction-reproducer"

    def __init__(self):
        self.calls = 0
        self.prompt_chars = []

    async def stream(self, request):
        self.calls += 1
        self.prompt_chars.append(len(request.messages[-1].content[0].text))
        yield ModelStreamEvent.text_delta(
            "Fictional read results remain available in the transcript."
        )
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


async def reproduce(*, expected_first_build="failure"):
    if expected_first_build not in {"failure", "success"}:
        raise ValueError("Expected first-build outcome must be failure or success")
    provider = ScriptedSummarizer()
    store = InMemorySessionStore()
    session = await store.create(
        RunRequest(agent_name="assistant", messages=[]),
        identity=SessionIdentity(provider_name=provider.name, model="synthetic"),
    )
    messages = [
        Message.text("system", "Preserve the original request."),
        Message.text("user", "ORIGINAL_FICTIONAL_REQUEST"),
    ]
    for index in range(6):
        messages.extend(
            [
                Message.tool_call(tool_call_id=f"read-{index}", tool_name="read", arguments={}),
                Message.tool_result(
                    tool_call_id=f"read-{index}", tool_name="read", content="x" * 7_000
                ),
            ]
        )
    request = ContextRequest(
        session=session,
        agent=AgentSpec(name="assistant", model="synthetic"),
        messages=messages,
        step=7,
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(
            provider=provider, model="synthetic", max_input_chars=10_000, max_hierarchy_calls=8
        ),
        compact_after_estimated_context_tokens=1_000,
        max_recent_context_tokens=500,
        reserved_output_tokens=100,
        reserved_summary_tokens=50,
    )
    checkpoint = None
    previous_cursor = 2
    attempts = []
    for attempt in range(1, 9):
        try:
            result = await policy.build_with_checkpoint(request, checkpoint=checkpoint)
        except ContextBuildError as failure:
            if expected_first_build == "success":
                raise AssertionError(
                    "Expected the first context build to fit without re-entry"
                ) from failure
            assert str(failure).startswith(
                "Checkpoint compaction did not produce a model-facing context "
                "within the configured size bounds."
            )
            assert failure.checkpoint is not None
            checkpoint = failure.checkpoint
            cursor = checkpoint["context_compaction"]["compacted_transcript_cursor"]
            assert cursor > previous_cursor, "No progress: a retry would be unjustified"
            previous_cursor = cursor
            attempts.append({"pass": attempt, "status": "does_not_fit", "cursor": cursor})
        else:
            pressure = _estimate_model_facing_context_pressure(
                request=request, messages=result.messages, reserved_output_tokens=100
            )
            assert pressure.estimated_context_input_tokens <= 500
            assert pressure.estimated_context_window_tokens < 1_000
            assert any(
                "ORIGINAL_FICTIONAL_REQUEST" in str(message.content) for message in result.messages
            )
            if expected_first_build == "failure":
                assert attempts, "Expected to reproduce at least one failed partial projection"
            else:
                assert attempt == 1 and not attempts, "Qualification must not use external re-entry"
            attempts.append(
                {
                    "pass": attempt,
                    "status": "fits",
                    "cursor": result.checkpoint["context_compaction"][
                        "compacted_transcript_cursor"
                    ],
                    "estimated_input_tokens": pressure.estimated_context_input_tokens,
                }
            )
            assert max(provider.prompt_chars) <= 10_000
            return {
                "synthetic_only": True,
                "network_calls": 0,
                "automatic_first_build_fails": len(attempts) > 1,
                "automatic_first_build_fits": len(attempts) == 1,
                "bounded_reentry_with_same_limits_fits": len(attempts) > 1,
                "scripted_summary_calls": provider.calls,
                "max_prompt_chars": max(provider.prompt_chars),
                "attempts": attempts,
            }
    raise AssertionError("Diagnostic did not converge within its explicit bound")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-first-build", choices=("failure", "success"), default="failure")
    args = parser.parse_args()
    print(
        json.dumps(asyncio.run(reproduce(expected_first_build=args.expect_first_build)), indent=2)
    )
