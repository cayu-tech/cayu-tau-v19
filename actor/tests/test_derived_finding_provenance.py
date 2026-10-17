"""Internal calculations must not advertise unresolvable plan evidence IDs."""

import json

from tau2.data_model.message import UserMessage
from test_runtime_banking import assembly, call, text, work

from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL
from cayu_tau3_banking.runtime_banking import BankingRuntimeSession


def test_derived_finding_context_allows_source_backed_answer_without_fake_tool_evidence(
    tmp_path, monkeypatch
):
    original = BankingRuntimeSession.history

    async def history(self, messages):
        await original(self, messages)
        self.record_structured_findings(
            tool_name="synthetic_calculator",
            tool_call_id="private-calculation-123",
            structured={"finding_kind": "calculation", "findings": [{"value": 4500}]},
        )

    monkeypatch.setattr(BankingRuntimeSession, "history", history)
    answered = work(done=True)
    answered.pop("next_operation")
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [answered]}),
            text("The purchase limit is $4,500."),
        ],
        planning_mode="outcomes",
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Explain the limit."), state)
        assert reply.content == "The purchase limit is $4,500."
        contexts = [
            part.text
            for request in provider.requests
            for message in request.messages
            for part in message.content
            if part.type == "text" and "<structured_findings>" in part.text
        ]
        assert contexts
        for context in contexts:
            content = context.split("<structured_findings>\n", 1)[1].split(
                "\n</structured_findings>", 1
            )[0]
            findings = json.loads(content)
            assert findings["items"][0]["calculation_id"] == "private-calculation-123"
            assert "evidence_ref" not in findings["items"][0]
            assert "not a plan evidence_ref" in findings["notice"]
        bank = captured["state"]
        assert not bank.ledger.actions  # No private calculation became a banking effect.
        assert bank.controller.export_state()["items"][0]["evidence_refs"] == [
            {"kind": "knowledge_revision", "reference_id": "limits", "version": "1", "available": True}
        ]
    finally:
        state.stop()
