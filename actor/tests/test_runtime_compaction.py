from __future__ import annotations

import asyncio

import pytest
from cayu import ModelStreamEvent, ScriptedModelProvider, ToolResultPart
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    ModelCompactor,
    TranscriptDigestCompactor,
)
from cayu.providers.base import ModelProviderError
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from test_runtime_banking import assembly, call, text, work

from cayu_tau3_banking.compaction import SUMMARY_INSTRUCTIONS, banking_compactor
from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL, UPDATE_WORK_PLAN_TOOL
from cayu_tau3_banking.ledger import ToolActionStatus
from cayu_tau3_banking.runtime_agent import RuntimeAgentError


def test_lossy_compactor_cannot_remove_retained_navigation_locators(tmp_path):
    from test_product_navigation import ENTRY, PRODUCT, document, worksheet

    from cayu_tau3_banking.product_navigation import BROWSE_TOOL, COMPARISON_TOOL, READ_TOOL

    document(tmp_path)
    summary = "Product research happened. Exact identifiers deliberately omitted."
    summary_provider = ScriptedModelProvider([text(summary) for _ in range(20)])
    policy = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(provider=summary_provider, model="lossy-script"),
        compact_after_messages=2,
        max_user_turns=1,
    )
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [work(reference=ENTRY)]}),
            call(BROWSE_TOOL, {"category": "personal_savings"}, "browse"),
            call(READ_TOOL, {"documents": [{"entry_id": ENTRY, "revision": 1}]}, "read"),
            call(COMPARISON_TOOL, worksheet(), "check"),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [work(done=True, reference=ENTRY)]}, "done"),
            text("The comparison is complete."),
            text("The same comparison remains valid."),
        ],
        persistent=True,
        compaction=policy,
        callback_timeout_s=60,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Compare the terms."), state)
        assert reply.content == "The comparison is complete."
        reply, _ = agent.generate_next_message(UserMessage.text("Summarize the same terms."), state)
        assert reply.content == "The same comparison remains valid."
        assert summary_provider.requests
        for request in provider.requests[2:]:
            context = next(t for t in message_texts(request) if "<banking_context>" in t)
            assert PRODUCT in context
        assert PRODUCT not in summary
        final_context = next(
            t for t in message_texts(provider.requests[-1]) if "<banking_context>" in t
        )
        assert "Retained source-checked comparison" in final_context
        assert worksheet()["facts"][0]["quote"] in final_context
        assert any(summary in t for r in provider.requests[2:] for t in message_texts(r))
        assert captured["state"].memory.navigation.state["documents"] == [ENTRY]
        assert not any(
            part.is_error
            for request in provider.requests
            for message in request.messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        )
    finally:
        state.stop()


def message_texts(request):
    return [
        part.text for message in request.messages for part in message.content if part.type == "text"
    ]


@pytest.mark.parametrize("failures", [1, 2])
def test_summary_unknown_retry_is_bounded_and_never_replays_banking_action(tmp_path, failures):
    error = [
        ModelStreamEvent.error(
            "Unknown provider error",
            cause=ModelProviderError("Unknown provider error", provider="scripted"),
        ),
        ModelStreamEvent.completed({"finish_reason": "error"}),
    ]
    summaries = ScriptedModelProvider(
        [error for _ in range(failures)]
        + [text("The recorded action succeeded.") for _ in range(8)]
    )
    compactor = banking_compactor(summaries, "scripted")
    policy = CheckpointCompactionContextPolicy(
        compactor=compactor,
        compact_after_estimated_context_tokens=20_000,
        max_recent_context_tokens=5_000,
        reserved_output_tokens=2_000,
        reserved_summary_tokens=1_000,
    )
    agent, actor, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [work(operation=True)]}),
            call("host_action", {"value": 7}, "bank-write"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {"items": [work(done=True, operation=True, reference="bank-write")]},
            ),
            text("Done."),
        ],
        persistent=True,
        compaction=policy,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Execute the action."), state)
        assert len(reply.tool_calls) == 1
        result = ToolMessage(
            role="tool",
            id=reply.tool_calls[0].id,
            content="completed " + "public reference data " * 4_000,
        )
        if failures == 2:
            with pytest.raises(RuntimeAgentError):
                agent.generate_next_message(result, state)
            assert len(summaries.requests) == 2
            assert len(actor.requests) == 2
        else:
            reply, _ = agent.generate_next_message(result, state)
            assert reply.content == "Done."
            assert len(summaries.requests) >= 2
        assert summaries.requests[0].messages == summaries.requests[1].messages
        assert all(not request.tools for request in summaries.requests)
        assert all(SUMMARY_INSTRUCTIONS in message_texts(request) for request in summaries.requests)
        assert [a.call_id for a in captured["state"].ledger.actions] == ["bank-write"]
        assert (
            captured["state"].ledger.tool_action_status("bank-write") is ToolActionStatus.SUCCEEDED
        )
    finally:
        state.stop()


def test_lossy_native_compaction_keeps_open_comparison_and_separate_recovery(tmp_path):
    from test_outcome_controller import obligation, outcome
    from test_product_navigation import ENTRY, document
    from test_research_workspace import comparison_args, resolve_args

    from cayu_tau3_banking.product_navigation import COMPARISON_TOOL, READ_TOOL

    document(tmp_path)
    summary_provider = ScriptedModelProvider(
        [
            text("Prior research occurred; exact facts and open questions omitted.")
            for _ in range(30)
        ]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(provider=summary_provider, model="lossy-script"),
        compact_after_messages=2,
        max_user_turns=1,
    )
    plan = {
        **outcome(completion_mode="information"),
        "obligations": [obligation(kind="knowledge_search", comparison_worksheet_id="value")],
    }
    read = {"documents": [{"entry_id": ENTRY, "revision": 1}]}
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(READ_TOOL, read, "new-source"),
            call(COMPARISON_TOOL, comparison_args(), "partial"),
            text("The companion's cost remains an open question."),
            call(SYNC_WORK_PLAN_TOOL, {"items": [plan]}),
            call(UPDATE_WORK_PLAN_TOOL, resolve_args(), "premature"),
            call(READ_TOOL, read, "recovery"),
            call(COMPARISON_TOOL, comparison_args(unresolved=False), "complete"),
            call(UPDATE_WORK_PLAN_TOOL, resolve_args(), "resolve"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {"items": [{"id": "request", "status": "completed", "outcome": "answered"}]},
                "done",
            ),
            text("The source-backed comparison is ready."),
        ],
        planning_mode="outcomes",
        advice_mode="lean",
        persistent=True,
        compaction=policy,
        callback_timeout_s=90,
        host_tools=[],
        max_steps=16,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Compare the terms."), state)
        assert reply.content == "The companion's cost remains an open question."
        reply, _ = agent.generate_next_message(UserMessage.text("Continue the comparison."), state)
        assert reply.content == "The source-backed comparison is ready."
        assert summary_provider.requests
        context = next(t for t in message_texts(provider.requests[5]) if "<banking_context>" in t)
        assert "Does the companion change the result after its own costs?" in context
        assert '"declared_coverage_complete": false' in context
        # Automatic recall delivered this exact revision before either explicit
        # read. Both are recovery, including the read after real compaction.
        assert captured["state"].memory.navigation.state["lanes"][READ_TOOL]["total"] == 0
        assert captured["state"].memory.navigation.state["recovery"]["total"] == 2
        rejected = {
            part.tool_call_id
            for request in provider.requests
            for message in request.messages
            for part in message.content
            if isinstance(part, ToolResultPart) and part.is_error
        }
        assert rejected == {"premature"}
        assert captured["state"].controller.export_state()["items"][0]["status"] == "completed"
    finally:
        state.stop()


@pytest.mark.parametrize("persistent", [False, True], ids=["memory", "sqlite"])
def test_compaction_keeps_current_evidence_and_original_durable_history(tmp_path, persistent):
    policy = CheckpointCompactionContextPolicy(
        compactor=TranscriptDigestCompactor(max_summary_chars=8_000),
        max_user_turns=1,
        compact_after_messages=2,
    )
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [work(done=True)]}),
            text("The purchase limit is $4,500."),
        ],
        persistent=persistent,
        compaction=policy,
    )
    history = [
        message
        for i in range(6)
        for message in (
            UserMessage.text(f"Earlier customer question {i}."),
            AssistantMessage.text(f"Earlier accepted answer {i}."),
        )
    ]
    state = agent.get_init_state(history)
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Explain the purchase limit."), state
        )
        assert reply.content == "The purchase limit is $4,500."
        texts = message_texts(provider.requests[0])
        assert "Earlier accepted answer 5." not in texts
        assert any("Previous session context summary" in value for value in texts)
        assert any("$4,500" in value and "<banking_context>" in value for value in texts)
        bank = captured["state"]
        assert len(bank.ledger.customer_message_ids) == 7
        assert bank.controller.export_state()["items"][0]["outcome"] == "answered"

        async def inspect():
            store = captured["app"].session_store
            return (
                await store.load_checkpoint(state._conversation.session_id),
                await store.load_transcript(state._conversation.session_id),
            )

        checkpoint, transcript = asyncio.run_coroutine_threadsafe(inspect(), state._loop).result(10)
        assert checkpoint["context_compaction"]["compacted_transcript_cursor"] > 0
        assert "summary" in checkpoint["context_compaction"]
        original = [
            part.text for message in transcript for part in message.content if part.type == "text"
        ]
        assert "Earlier accepted answer 5." in original
        assert not any("<banking_context>" in value for value in original)
    finally:
        state.stop()


@pytest.mark.parametrize(
    ("oversized", "hierarchy_calls"),
    [(False, 8), (True, 8), (True, 2)],
    ids=["bounded", "hierarchical", "hierarchy_budget_exhausted"],
)
def test_model_compaction_preserves_evidence_or_fails_closed_at_input_bound(
    tmp_path, oversized, hierarchy_calls, monkeypatch
):
    observed_headroom = []
    observed_compactions = []
    original_build = CheckpointCompactionContextPolicy.build_with_checkpoint

    async def observe_build(self, request, *, checkpoint):
        observed_headroom.append(self.reserved_summary_tokens)
        built = await original_build(self, request, checkpoint=checkpoint)
        observed_compactions.extend(
            item.payload
            for item in built.compaction_telemetry
            if item.event_type.value == "context.compaction.completed"
        )
        return built

    monkeypatch.setattr(CheckpointCompactionContextPolicy, "build_with_checkpoint", observe_build)
    summary_provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta(
                    "The environment action returned successfully; see authoritative work evidence."
                ),
                ModelStreamEvent.completed(
                    {"usage": {"input_tokens": 100, "output_tokens": 20}, "finish_reason": "stop"}
                ),
            ]
            for _ in range(8)
        ]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(
            provider=summary_provider,
            model="scripted-summary",
            max_hierarchy_calls=hierarchy_calls,
        ),
        compact_after_estimated_context_tokens=20_000,
        max_recent_context_tokens=5_000,
        reserved_output_tokens=2_000,
        reserved_summary_tokens=1_000,
    )
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [work(operation=True)]}),
            call("host_action", {"value": 7}, "model-action"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {"items": [work(done=True, operation=True, reference="model-action")]},
                "update",
            ),
            text("Done."),
        ],
        compaction=policy,
    )
    state = agent.get_init_state()
    # A large atomic round may be covered by bounded hierarchy work. The runtime
    # must continue past an earlier ordinary prefix, but reject work that cannot
    # fit the hierarchy budget without dropping or repeating the external effect.
    large_result = "completed " + "public reference data " * (20_000 if oversized else 4_000)
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Execute the requested action."), state
        )
        result = ToolMessage(
            role="tool",
            id=reply.tool_calls[0].id,
            content=large_result,
        )
        if hierarchy_calls == 2:
            with pytest.raises(RuntimeAgentError):
                agent.generate_next_message(result, state)
            # A context failure cannot undo an external effect or authorize a retry.
            assert len(provider.requests) == 2
            assert (
                captured["state"].ledger.tool_action_status("model-action")
                is ToolActionStatus.SUCCEEDED
            )
            assert not state._thread.is_alive()
            # The ordinary prefix succeeds; the oversized unit's hierarchy is
            # rejected during budget preflight before any hierarchy dispatch.
            assert len(summary_provider.requests) == 1
            assert observed_headroom and set(observed_headroom) == {1_000}
            return
        reply, _ = agent.generate_next_message(result, state)
        assert reply.content == "Done."
        assert summary_provider.requests
        assert all(not request.tools for request in summary_provider.requests)
        assert all(
            len(request.messages[-1].content[0].text) <= policy.compactor.max_input_chars
            for request in summary_provider.requests
        )
        if oversized:
            assert len(observed_compactions) == 2
            assert [item["represented_source_end"] for item in observed_compactions] == [4, 6]
            assert observed_compactions[-1]["chunk_mode"] == "hierarchical_atomic_unit"
        usage = reply.raw_data["cayu_runtime_usage"]
        assert usage["input_tokens"] == 100 * len(summary_provider.requests)
        assert usage["output_tokens"] == 20 * len(summary_provider.requests)
        assert reply.usage is None  # Acting script omitted usage; never imply a complete total.
        after = provider.requests[2]
        assert not any(
            isinstance(part, ToolResultPart) and part.content == large_result
            for message in after.messages
            for part in message.content
        )
        assert any(
            "model-action" in value and "<banking_context>" in value
            for value in message_texts(after)
        )
        bank = captured["state"]
        assert [action.call_id for action in bank.ledger.actions] == ["model-action"]
        assert bank.ledger.tool_action_status("model-action") is ToolActionStatus.SUCCEEDED
        assert bank.controller.export_state()["items"][0]["outcome"] == "succeeded"
        transcript = asyncio.run_coroutine_threadsafe(
            captured["app"].session_store.load_transcript(state._conversation.session_id),
            state._loop,
        ).result(10)
        assert any(
            isinstance(part, ToolResultPart) and part.content == large_result
            for message in transcript
            for part in message.content
        )
    finally:
        state.stop()

    assert observed_headroom and set(observed_headroom) == {1_000}


def test_transient_evidence_reserves_context_capacity_without_silent_truncation(tmp_path):
    policy = CheckpointCompactionContextPolicy(
        compact_after_estimated_context_tokens=10_000,
        max_recent_context_tokens=5_000,
        reserved_output_tokens=1_000,
    )
    agent, provider, captured = assembly(tmp_path, [text("must not run")], compaction=policy)
    state = agent.get_init_state()

    async def oversized_context(**kwargs):
        return "Evidence that must not be silently dropped. " * 10_000

    captured["state"].memory.runtime_context = oversized_context
    try:
        with pytest.raises(RuntimeAgentError):
            agent.generate_next_message(UserMessage.text("Explain the purchase limit."), state)
        assert not provider.requests
    finally:
        state.stop()
