from types import SimpleNamespace

import pytest
from cayu import Message, ScriptedModelProvider
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    ContextBuildResult,
    ContextPressureOverhead,
    ContextUsageState,
    ModelCompactor,
)
from tau2.data_model.message import ToolMessage, UserMessage
from test_runtime_banking import assembly, call, text

from cayu_tau3_banking.context_pressure import COMPACTION_TRIGGER, warn_before_compaction


def warning(result, checkpoint=None, reserve=8_256):
    request = SimpleNamespace(
        messages=result.messages,
        context_usage=ContextUsageState(),
        pressure_overhead=ContextPressureOverhead(),
    )
    return warn_before_compaction(
        result,
        request,
        previous_checkpoint=checkpoint,
        threshold=COMPACTION_TRIGGER,
        reserved_output=reserve,
    )


def test_warning_counts_reserved_projection_and_is_once_per_persisted_cursor():
    result = ContextBuildResult(messages=[Message.text("user", "x" * 325_000)])
    assert warning(result) is result
    warned = warning(result, reserve=30_000)
    assert "<context_pressure" in warned.messages[-1].content[0].text
    assert "tau_context_pressure_warning" in warned.checkpoint
    assert warning(result, checkpoint=warned.checkpoint, reserve=30_000) is result
    # A new compaction cursor starts a new warning cycle; unrelated state survives.
    after = result.model_copy(
        update={
            "checkpoint": {
                **warned.checkpoint,
                "context_compaction": {"compacted_transcript_cursor": 42},
                "other_state": {"preserved": True},
            }
        }
    )
    next_warning = warning(after, reserve=30_000)
    assert next_warning.checkpoint["tau_context_pressure_warning"]["cursor"] == 42
    assert next_warning.checkpoint["other_state"] == {"preserved": True}
    assert result.checkpoint is None


@pytest.mark.parametrize("legacy", [True, False])
def test_default_larger_context_preserves_source_that_old_threshold_compacts(
    tmp_path, monkeypatch, legacy
):
    import cayu_tau3_banking.runtime_banking as runtime

    summaries = ScriptedModelProvider([text("Old result summarized; exact detail omitted.")] * 10)
    compactor = ModelCompactor(provider=summaries, model="summary")
    monkeypatch.setattr(runtime, "banking_compactor", lambda *_: compactor)
    policy = (
        CheckpointCompactionContextPolicy(
            compactor=compactor,
            compact_after_estimated_context_tokens=60_000,
            max_recent_context_tokens=20_000,
            reserved_output_tokens=8_000,
            reserved_summary_tokens=4_000,
        )
        if legacy
        else None
    )
    agent, actor, _ = assembly(
        tmp_path,
        [call("host_action", {"value": 7}, "read"), text("Done.")],
        compaction=policy,
        controller_mode="advisory",
        planning_mode="outcomes",
        advice_mode="lean",
        callback_timeout_s=180,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Perform the requested action."), state
        )
        assert reply.tool_calls
        body = "Exact qualification: marker-Q791.\n" + "ordinary source text " * 14_000
        agent.generate_next_message(
            ToolMessage(role="tool", id=reply.tool_calls[0].id, content=body), state
        )
        assert bool(summaries.requests) is legacy
        if not legacy:
            assert "marker-Q791" in repr(actor.requests[-1].messages)
    finally:
        state.stop()


def test_warning_reaches_actor_once_across_runtime_turns(tmp_path, monkeypatch):
    import cayu_tau3_banking.runtime_banking as runtime

    summaries = ScriptedModelProvider([text("Summary.")] * 10)
    monkeypatch.setattr(
        runtime, "banking_compactor", lambda *_: ModelCompactor(provider=summaries, model="summary")
    )
    agent, actor, _ = assembly(
        tmp_path,
        [
            call("host_action", {"value": 7}, "read"),
            text("Ready."),
            text("Done."),
        ],
        persistent=True,
        controller_mode="advisory",
        planning_mode="outcomes",
        advice_mode="lean",
        callback_timeout_s=180,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Perform the action."), state)
        agent.generate_next_message(
            ToolMessage(role="tool", id=reply.tool_calls[0].id, content="x" * 375_000), state
        )
        agent.generate_next_message(UserMessage.text("Continue."), state)
        assert not summaries.requests
        assert "<context_pressure" in repr(actor.requests[1].messages)
        assert "<context_pressure" not in repr(actor.requests[2].messages)
    finally:
        state.stop()
