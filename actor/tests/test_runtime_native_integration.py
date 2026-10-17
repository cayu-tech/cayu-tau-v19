import asyncio

import pytest
from cayu import RecallEvidenceQuery
from tau2.data_model.message import UserMessage
from test_runtime_banking import assembly, call, text, work

from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL


@pytest.mark.parametrize("persistent", [False, True])
def test_native_recall_delivers_once_per_step_and_binds_controller_evidence(tmp_path, persistent):
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [work(done=True)]}),
            text("The purchase limit is $4,500."),
        ],
        native_recall=True,
        persistent=persistent,
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Explain the purchase limit."), state
        )
        assert reply.content == "The purchase limit is $4,500."
        assert captured["state"].controller.export_state()["items"][0]["outcome"] == "answered"
        for request in provider.requests:
            content = "\n".join(
                p.text for m in request.messages for p in m.content if p.type == "text"
            )
            assert content.count('<cayu_automatic_memory version="2">') == 1
            assert "<banking-knowledge-recall>" not in content

        async def inspect():
            store = captured["app"].session_store
            query = RecallEvidenceQuery(session_id=state._conversation.session_id)
            return await store.list_recall_receipts(query), await store.list_context_exposures(
                query
            )

        receipts, exposures = asyncio.run_coroutine_threadsafe(inspect(), state._loop).result(10)
        assert len(receipts.items) == 1
        assert len(exposures.items) == 2
        assert len(captured["state"].observed_recall_exposures) == 2
    finally:
        state.stop()
