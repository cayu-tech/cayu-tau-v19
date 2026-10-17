import asyncio
import json

import pytest
from cayu import WorkEvidenceReference
from test_outcome_controller import obligation, outcome, setup_plan, update

from cayu_tau3_banking.active_evidence import active_evidence_context
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus


@pytest.fixture
def corpus(tmp_path):
    for name in ("procedure", "other"):
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"id": name, "title": name, "content": f"{name}: credit is 17 dollars."})
        )
    return BankingKnowledgeCorpus.from_directory(tmp_path)


def ref(corpus, name="procedure"):
    return corpus.delivered_revision_evidence(
        name, 1, delivered_text=f"{name}: credit is 17 dollars."
    )


def payload(context):
    return json.loads(context.split("<active-work-evidence>\n")[1].split("\n</")[0])


def citation(reference):
    return reference.model_dump(include={"kind", "reference_id", "version"})


def test_exact_revision_retained_without_search_or_state_change(corpus):
    controller, _ = setup_plan(outcome())
    reference = ref(corpus)
    controller.observe_knowledge_evidence((reference,))
    assert not update(
        controller,
        obligations=[
            obligation(
                kind="communication",
                evidence="Use the procedure",
                evidence_refs=[citation(reference)],
            )
        ],
    ).is_error
    before = controller.export_state()
    result = payload(
        asyncio.run(active_evidence_context(corpus, controller.active_knowledge_references()))
    )
    assert result["documents"][0]["text"] == "procedure: credit is 17 dollars."
    assert result["documents"][0]["text_complete"] is True
    assert result["omitted"] == []
    assert before == controller.export_state()


def test_resolved_obligations_and_completed_work_release_references(corpus):
    controller, _ = setup_plan(outcome(completion_mode="information"))
    reference = ref(corpus)
    controller.observe_knowledge_evidence((reference,))
    assert not update(
        controller, obligations=[obligation("resolved", evidence_refs=[citation(reference)])]
    ).is_error
    assert controller.active_knowledge_references() == ()
    assert not update(
        controller,
        status="completed",
        outcome="answered",
        evidence="Answered",
        evidence_refs=[citation(reference)],
    ).is_error
    assert controller.active_knowledge_references() == ()


def test_condition_on_unfinished_parent_survives_completed_research(corpus):
    controller, _ = setup_plan(outcome())
    reference = ref(corpus)
    controller.observe_knowledge_evidence((reference,))
    assert not update(
        controller,
        evidence="The procedure's conditions must remain satisfied until execution.",
        evidence_refs=[citation(reference)],
        obligations=[
            obligation("resolved", kind="knowledge_search", evidence_refs=[citation(reference)])
        ],
    ).is_error
    assert controller.active_knowledge_references() == (reference,)
    pack = payload(
        asyncio.run(active_evidence_context(corpus, controller.active_knowledge_references()))
    )
    assert pack["documents"][0]["text_complete"] is True
    assert pack["documents"][0]["text"] == "procedure: credit is 17 dollars."


def test_completed_research_stays_with_unfinished_outcome_without_parent_copy(corpus):
    controller, _ = setup_plan(outcome(completion_mode="information"))
    reference = ref(corpus)
    controller.observe_knowledge_evidence((reference,))
    assert not update(
        controller,
        obligations=[
            obligation("resolved", kind="knowledge_search", evidence_refs=[citation(reference)])
        ],
    ).is_error
    assert controller.active_knowledge_references() == (reference,)
    before = controller.export_state()
    pack = payload(
        asyncio.run(active_evidence_context(corpus, controller.active_knowledge_references()))
    )
    assert pack["documents"][0]["text"] == "procedure: credit is 17 dollars."
    assert controller.export_state() == before
    assert not update(
        controller,
        status="completed",
        outcome="answered",
        evidence="Answered",
        evidence_refs=[citation(reference)],
    ).is_error
    assert controller.active_knowledge_references() == ()


def test_deduplicates_and_reports_bound_omissions(corpus):
    first, second = ref(corpus), ref(corpus, "other")
    result = payload(
        asyncio.run(active_evidence_context(corpus, (first, first, second), max_documents=1))
    )
    assert len(result["documents"]) == 1
    assert result["omitted"] == [
        {"entry_id": "other", "entry_revision": "1", "reason": "document_limit"}
    ]
    result = payload(asyncio.run(active_evidence_context(corpus, (first,), max_bytes=1)))
    assert not result["documents"]
    assert result["omitted"][0]["reason"] == "byte_limit"


def test_wrong_digest_fails_closed(corpus):
    reference = ref(corpus).model_copy(update={"digest": "0" * 64})
    with pytest.raises(ValueError, match="accepted revision"):
        asyncio.run(active_evidence_context(corpus, (reference,)))


def test_no_references_adds_no_context(corpus):
    assert asyncio.run(active_evidence_context(corpus, ())) is None


def test_missing_revision_is_not_substituted(corpus):
    reference = WorkEvidenceReference(
        kind="knowledge_revision", reference_id="missing", version="1"
    )
    result = payload(asyncio.run(active_evidence_context(corpus, (reference,))))
    assert result["documents"] == []
    assert result["omitted"][0]["reason"] == "revision_unavailable"


def test_native_context_keeps_cited_policy_across_customer_turns(tmp_path):
    from tau2.data_model.message import UserMessage
    from test_runtime_banking import assembly, call, text

    from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL

    reference = {"kind": "knowledge_revision", "reference_id": "limits", "version": "1"}
    plan = {
        **outcome(completion_mode="information"),
        "obligations": [
            obligation(
                kind="communication", evidence="Explain the known policy", evidence_refs=[reference]
            )
        ],
    }
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [plan]}),
            text("Which use case?"),
            text("The purchase limit is $4,500. Which use case?"),
        ],
        native_recall=True,
        planning_mode="outcomes",
    )
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Explain the purchase limit."), state)
        agent.generate_next_message(UserMessage.text("A different use case."), state)
        serialized = provider.requests[-1].model_dump_json()
        assert "active-work-evidence" in serialized
        assert "The purchase limit is $4,500." in serialized
        assert not captured["state"].ledger.actions
        assert len(provider.requests) == 3
    finally:
        state.stop()
