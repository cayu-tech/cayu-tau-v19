"""Generic recovery and declared-comparison contracts; no benchmark answers."""

import asyncio
import json
from copy import deepcopy

import pytest
from cayu import ToolCallPart
from tau2.data_model.message import AssistantMessage, UserMessage
from test_outcome_controller import obligation, outcome, setup_plan, update
from test_product_navigation import ENTRY, PRODUCT, document, worksheet
from test_read_batch_capacity import make_session, request

from cayu_tau3_banking.admission import BankingToolAdmissionGate
from cayu_tau3_banking.agent_memory import _MAX_KNOWLEDGE_SEARCHES, BankingMemoryRuntime
from cayu_tau3_banking.controller import UPDATE_WORK_PLAN_TOOL, BankingWorkflowController
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.product_navigation import (
    BUDGET_LIMITS,
    COMPARISON_TOOL,
    READ_TOOL,
    RECOVERY_LANE,
    ProductNavigation,
)


def invoke(session, name, args):
    return asyncio.run(session.invoke(name, args))


def test_known_source_recovery_does_not_consume_new_research(tmp_path):
    session = make_session(tmp_path, count=2)
    first = invoke(session, READ_TOOL, request(1))
    assert not first.is_error
    assert json.loads(first.content)["navigation_budget"]["charged_lane"] == READ_TOOL
    for _ in range(6):
        restored = invoke(session, READ_TOOL, request(1))
        assert not restored.is_error
        assert json.loads(restored.content)["navigation_budget"]["charged_lane"] == RECOVERY_LANE
    saved = session.navigation.snapshot()
    assert saved["lanes"][READ_TOOL] == {"turn": 1, "total": 1}
    assert saved["recovery"] == {"turn": 6, "total": 6}
    assert invoke(session, READ_TOOL, request(1)).is_error
    assert session.navigation.snapshot() == saved  # No borrowing research capacity.
    assert not invoke(session, READ_TOOL, {"documents": [request(2)["documents"][1]]}).is_error
    assert session.navigation.state["lanes"][READ_TOOL]["turn"] == 2


def test_recovery_remains_available_after_new_read_exhaustion_but_cannot_read_unknown(tmp_path):
    limit = BUDGET_LIMITS[READ_TOOL][0]
    session = make_session(tmp_path, count=limit + 1)
    for index in range(limit):
        assert not invoke(
            session, READ_TOOL, {"documents": [request(limit + 1)["documents"][index]]}
        ).is_error
    assert READ_TOOL in {s["name"] for s in session.tool_schemas()}
    assert not invoke(session, READ_TOOL, request(1)).is_error
    saved = session.navigation.snapshot()
    mixed = {"documents": [request(1)["documents"][0], request(limit + 1)["documents"][limit]]}
    rejected = invoke(session, READ_TOOL, mixed)
    assert rejected.is_error and READ_TOOL in rejected.content
    assert session.navigation.snapshot() == saved
    assert not session.controller.has_delivered_knowledge_revision(
        mixed["documents"][1]["entry_id"], "1"
    )


def test_locator_or_wrong_revision_never_grants_recovery(tmp_path):
    session = make_session(tmp_path, count=1)
    session.navigation.remember_document(request(1)["documents"][0]["entry_id"])
    assert not invoke(session, READ_TOOL, request(1)).is_error
    wrong = request(1)
    wrong["documents"][0]["revision"] = 2
    assert invoke(session, READ_TOOL, wrong).is_error
    assert session.navigation.state["lanes"][READ_TOOL]["total"] == 2
    assert session.navigation.state["recovery"]["total"] == 0


def test_recovery_counter_is_durable_and_only_customer_turn_resets(tmp_path):
    session = make_session(tmp_path, count=1)
    assert not invoke(session, READ_TOOL, request(1)).is_error
    assert not invoke(session, READ_TOOL, request(1)).is_error
    saved = session.navigation.snapshot()
    session.navigation = ProductNavigation(session.corpus, saved)
    assert not invoke(session, READ_TOOL, request(1)).is_error
    assert session.navigation.state["recovery"] == {"turn": 2, "total": 2}
    assert saved["recovery"] == {"turn": 1, "total": 1}
    session.ledger.observe_input(UserMessage.text("A genuinely new request."))
    assert not invoke(session, READ_TOOL, request(1)).is_error
    assert session.navigation.state["recovery"] == {"turn": 1, "total": 3}


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {"turn": True, "total": 1},
        {"turn": 7, "total": 7},
        {"turn": 1, "total": 0},
        {"turn": 0, "total": 25},
    ],
)
def test_invalid_recovery_snapshot_is_rejected(tmp_path, bad):
    session = make_session(tmp_path, count=1)
    saved = session.navigation.snapshot()
    saved["recovery"] = bad
    with pytest.raises(ValueError):
        ProductNavigation(session.corpus, saved)


def test_old_snapshots_do_not_receive_unaccounted_recovery_capacity(tmp_path):
    session = make_session(tmp_path, count=1)
    saved = session.navigation.snapshot()
    saved["version"] = 5
    saved.pop("recovery")
    restored = ProductNavigation(session.corpus, saved)
    assert restored.state["recovery"] == {"turn": 6, "total": 24}
    assert restored.state["lanes"] == saved["lanes"]


def coverage(*, unresolved=True):
    return {
        "objective": "Compare retained annual value after all fees.",
        "alternatives": [
            {
                "product_ids": [PRODUCT],
                "status": "unresolved" if unresolved else "compared",
                "reason": "Verify which advertised benefit applies to this request.",
            }
        ],
        "open_questions": ["Does the companion change the result after its own costs?"]
        if unresolved
        else [],
    }


@pytest.fixture
def comparison_session(tmp_path):
    document(tmp_path)
    controller, ledger = setup_plan(
        {
            **outcome(completion_mode="information"),
            "obligations": [obligation(kind="knowledge_search", comparison_worksheet_id="value")],
        }
    )
    session = BankingMemoryRuntime(BankingKnowledgeCorpus.from_directory(tmp_path)).new_session(
        ledger=ledger,
        controller=controller,
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=True,
    )
    assert not invoke(
        session, READ_TOOL, {"documents": [{"entry_id": ENTRY, "revision": 1}]}
    ).is_error
    return session


def comparison_args(*, unresolved=True):
    return {**worksheet(), "worksheet_id": "value", "coverage": coverage(unresolved=unresolved)}


def resolve_args():
    return {
        "items": [
            {
                "id": "request",
                "obligations": [
                    {
                        "id": "validate-address",
                        "status": "resolved",
                        "evidence": "The declared comparison is complete.",
                        "evidence_refs": [
                            {"kind": "knowledge_revision", "reference_id": ENTRY, "version": "1"}
                        ],
                    }
                ],
            }
        ]
    }


def test_unknowns_survive_worksheet_reuse_and_prevent_plan_closure(comparison_session):
    session = comparison_session
    checked = invoke(session, COMPARISON_TOOL, comparison_args())
    assert not checked.is_error
    assert json.loads(checked.content)["declared_cells_complete"]
    assert not json.loads(checked.content)["declared_coverage_complete"]
    before = session.controller.export_state()
    rejected = invoke(session, UPDATE_WORK_PLAN_TOOL, resolve_args())
    assert rejected.is_error and "coverage" in rejected.content
    assert session.controller.export_state() == before
    restored = ProductNavigation(session.corpus, session.navigation.snapshot())
    context = asyncio.run(
        restored.comparison_context(session.controller.has_delivered_knowledge_revision)
    )
    assert "Does the companion change" in context
    reuse = comparison_args()
    reuse.pop("coverage")
    reuse["facts"] = []
    reuse["reuse_checked_facts"] = True
    result = asyncio.run(restored.check(reuse, session.controller.has_delivered_knowledge_revision))
    assert not result["declared_coverage_complete"]


def test_complete_declared_coverage_allows_resolution_and_binding_survives_restore(
    comparison_session,
):
    session = comparison_session
    assert not invoke(session, COMPARISON_TOOL, comparison_args(unresolved=False)).is_error
    accepted = invoke(session, UPDATE_WORK_PLAN_TOOL, resolve_args())
    assert not accepted.is_error, accepted.content
    saved = session.controller.export_state()
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Checkpoint", raw_data={"cayu_tau3_workflow": saved})],
        ledger=session.ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == saved
    assert restored.invalid_comparison_bindings()  # No readiness inferred from persisted prose.
    restored.set_comparison_readiness({"value": True})
    assert not restored.invalid_comparison_bindings()


def test_comparison_binding_cannot_be_removed_or_reassigned(comparison_session):
    session = comparison_session
    before = session.controller.export_state()
    for replacement in (None, "other"):
        result = update(
            session.controller,
            obligations=[{"id": "validate-address", "comparison_worksheet_id": replacement}],
        )
        assert result.is_error
        assert session.controller.export_state() == before


def test_invalidated_worksheet_blocks_terminal_until_comparison_is_reopened(comparison_session):
    session = comparison_session
    assert not invoke(session, COMPARISON_TOOL, comparison_args(unresolved=False)).is_error
    assert not invoke(session, UPDATE_WORK_PLAN_TOOL, resolve_args()).is_error
    assert not invoke(session, COMPARISON_TOOL, comparison_args(unresolved=True)).is_error
    call = ToolCallPart(tool_call_id="effect", tool_name="host_action", arguments={})
    for gate in (session, session.controller):
        (decision,) = gate.review((call,))
        assert not decision.allowed
        assert decision.diagnostic_code == "comparison_coverage_invalidated"
    result = asyncio.run(session.review_terminal_output("That settles the comparison."))
    assert not result.allowed and result.correction_code == "comparison_coverage_invalidated"
    assert not invoke(
        session,
        UPDATE_WORK_PLAN_TOOL,
        {
            "items": [
                {"id": "request", "obligations": [{"id": "validate-address", "status": "ready"}]}
            ]
        },
    ).is_error
    assert session.controller.review((call,))[0].allowed


@pytest.mark.parametrize(
    "change", ["unknown", "duplicate", "missing_question", "blank", "new_compared_component"]
)
def test_coverage_declaration_rejects_invalid_scope_without_overwriting(comparison_session, change):
    session = comparison_session
    assert not invoke(session, COMPARISON_TOOL, comparison_args()).is_error
    before = deepcopy(session.navigation.state["named_worksheets"])
    args = comparison_args()
    if change == "unknown":
        args["coverage"]["alternatives"][0]["product_ids"] = ["not-a-product"]
    elif change == "duplicate":
        args["coverage"]["alternatives"] *= 2
    elif change == "missing_question":
        args["coverage"]["open_questions"] = []
    elif change == "blank":
        args["coverage"]["objective"] = " "
    else:
        args["coverage"]["alternatives"][0].update(
            status="compared", product_ids=["unlisted-component"]
        )
    result = invoke(session, COMPARISON_TOOL, args)
    assert result.is_error
    assert session.navigation.state["named_worksheets"] == before


def test_directory_alone_is_not_usable_document_research(tmp_path):
    session = make_session(tmp_path, count=1)
    session.navigation.state["lanes"][READ_TOOL] = {
        "turn": BUDGET_LIMITS[READ_TOOL][0],
        "total": BUDGET_LIMITS[READ_TOOL][1],
    }
    session.navigation.state["recovery"] = {"turn": 6, "total": 24}
    session._knowledge_searches = _MAX_KNOWLEDGE_SEARCHES
    payload = json.loads(session.research_capacity_context().split("\n")[1])
    assert payload["directory_browse_available"]
    assert payload["research_exhausted"]
    assert not payload["new_source_reads_available"]
    assert not payload["source_recovery_available"]


def test_same_products_can_have_different_unresolved_configurations(comparison_session):
    session = comparison_session
    args = comparison_args(unresolved=False)
    args["coverage"]["alternatives"].append(
        {
            "product_ids": [PRODUCT],
            "status": "unresolved",
            "reason": "A different term length may change the value; its terms are not yet verified.",
        }
    )
    args["coverage"]["open_questions"] = ["Does the longer term have the same costs?"]
    result = invoke(session, COMPARISON_TOOL, args)
    assert not result.is_error, result.content
    assert not json.loads(result.content)["declared_coverage_complete"]
    assert invoke(session, UPDATE_WORK_PLAN_TOOL, resolve_args()).is_error
