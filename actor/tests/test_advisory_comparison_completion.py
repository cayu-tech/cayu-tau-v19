"""Research completion must agree with the production advisory tool contract."""
import asyncio
import json

import pytest
from cayu import ScriptedModelProvider
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from tau2.data_model.message import UserMessage
from test_product_navigation import document
from test_read_batch_capacity import make_session
from test_runtime_banking import assembly, call, text
from test_runtime_compaction import message_texts

from cayu_tau3_banking.product_navigation import COMPARISON_TOOL, READ_TOOL, ProductNavigation


def plan():
    return {"items": [{"id": "comparison", "goal": "Compare the accounts.",
        "completion_mode": "information", "status": "in_progress", "outcome": "unresolved",
        "evidence_refs": [{"kind": "customer_message", "reference_id": "customer-message:1"}],
        "obligations": [{"id": "research", "goal": "Read applicable terms.",
            "kind": "knowledge_search", "status": "ready", "comparison_worksheet_id": "comparison"}]}]}


def resolution(entry):
    return {"items": [{"id": "comparison", "obligations": [{"id": "research",
        "status": "resolved", "evidence": "Terms conflict; give a conditional comparison, not an unconditional winner.",
        "evidence_refs": [{"kind": "knowledge_revision", "reference_id": entry, "version": "1"}]}]}]}


@pytest.mark.parametrize("enforced", [False, True])
def test_existing_binding_resolves_only_in_advisory_mode(tmp_path, enforced):
    async def run():
        session = make_session(tmp_path, count=1)
        session.workflow_controller_enabled = True
        session.advisory_workflow = True
        session.controller.planning_mode = "outcomes"
        # Create the legacy binding before changing the host's enforcement mode.
        assert not (await session.invoke("controller_sync_work_plan", plan())).is_error
        session.controller.enforce_comparison_coverage = enforced
        entry = "doc_savings_accounts_nimbus_000"
        body = json.loads((await session.invoke(READ_TOOL, {"documents": [{"entry_id": entry}]})).content)
        saved = await session.invoke(COMPARISON_TOOL, {"worksheet_id": "comparison", "passage_handles": list(body["documents"][0]["passages"])})
        assert not saved.is_error
        result = await session.invoke("controller_update_work_plan", resolution(entry))
        assert result.is_error is enforced, result.content
        assert session.navigation.comparison_readiness() == {"comparison": False}
        if enforced:
            assert "no complete revalidated declared coverage" in result.content
        else:
            assert not session.controller.invalid_comparison_bindings()
    asyncio.run(run())


def test_compact_outcomes_and_uncertainty_survive_compaction_in_production_assembly(tmp_path):
    entry = "doc_savings_accounts_nimbus_001"
    terms = "Stated annual yield is 4.2%. The example says 4.8%. The applicable rate is unresolved."
    document(tmp_path, entry, terms)
    steps = []
    def configure(corpus):
        nav = ProductNavigation(corpus)
        handles = [p["passage_handle"] for p in nav.document_passages[entry]]
        initial = plan()
        initial["items"][0].pop("obligations")
        explained = {"items": [{"id": "comparison", "status": "completed", "outcome": "answered",
            "evidence": "Explained the conditional comparison and unresolved rate; neither rate is confirmed.",
            "evidence_refs": [{"kind": "knowledge_revision", "reference_id": entry, "version": "1"}]}]}
        steps.extend([
            call("controller_sync_work_plan", initial, "plan"),
            call(READ_TOOL, {"documents": [{"entry_id": entry}]}, "read"),
            call(COMPARISON_TOOL, {"worksheet_id": "comparison", "passage_handles": handles}, "pin"),
            call("cayu_update_decision_facts", {"facts": [
                {"id": "rate", "claim": "The stated 4.2% rate conflicts with the 4.8% example in the same source.", "state": "contradicted",
                 "sources": [{"kind": "knowledge_revision", "reference_id": entry, "version": "1"}]},
                {"id": "eligibility", "claim": "Customer qualification is unknown; compare conditionally.", "state": "unknown", "sources": []},
            ]}, "notes"),
            text("The rate is contradictory: I can compare both scenarios but cannot promise either rate."),
            call("controller_update_work_plan", explained, "explained"),
            text("The rate remains unresolved; the conditional comparison still applies."),
        ])
    summarizer = ScriptedModelProvider([text("The rate is confirmed at 4.8%; all qualifications are satisfied.") for _ in range(24)])
    policy = CheckpointCompactionContextPolicy(compactor=ModelCompactor(provider=summarizer, model="summary"), compact_after_messages=2, max_user_turns=1)
    agent, provider, _ = assembly(tmp_path, steps, configure_corpus=configure, compaction=policy,
        planning_mode="outcomes", advice_mode="lean", controller_mode="advisory",
        task_memory_mode="constraints", task_memory_directory=tmp_path / "notes")
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(UserMessage.text("Compare the accounts."), state)
        assert "cannot promise" in reply.content
        agent.generate_next_message(UserMessage.text("Is the rate confirmed?"), state)
        rendered = "\n".join(t for req in provider.requests for t in message_texts(req))
        assert "Plan rejected" not in rendered
        assert '"status": "completed"' in rendered
        assert "Private tool arguments do not satisfy" not in rendered
        assert terms in "\n".join(message_texts(provider.requests[-1]))
        projection = next(t for t in reversed(message_texts(provider.requests[-1])) if t.startswith("<banking_context>"))
        assert '"state": "contradicted"' in projection
        assert '"state": "unknown"' in projection
        assert "Customer qualification is unknown" in projection
        assert summarizer.requests
    finally:
        state.stop()
