"""Model-facing bookkeeping stays small without weakening execution evidence."""

import json
from copy import deepcopy

import pytest
from cayu import WorkEvidenceReference
from test_case_conditions import condition, execute, read, setup
from test_debit_resource_conditions import card_read
from test_outcome_controller import outcome, setup_plan

from cayu_tau3_banking.case_conditions import condition_schema, parse_condition
from cayu_tau3_banking.controller import (
    SYNC_WORK_PLAN_TOOL,
    UPDATE_WORK_PLAN_TOOL,
    BankingWorkflowController,
)
from cayu_tau3_banking.ledger import ConversationLedger


def identity_observation(kind):
    ledger = ConversationLedger()
    if kind == "bank_account":
        read(ledger)
        return ledger, "account-a", "account_id", "read-1"
    if kind == "debit_card":
        card_read(ledger)
        return ledger, "debit-a", "card_id", "card-read"
    execute(
        ledger,
        "credit-read",
        "call_discoverable_agent_tool",
        {
            "agent_tool_name": "get_all_user_accounts_by_user_id_3847",
            "arguments": json.dumps({"user_id": "person-a"}),
        },
        "Bank Accounts:\nNo bank accounts found.\n\n"
        "Credit Card Accounts:\nFound 1 record(s) in 'credit_card_accounts':\n\n"
        "1. Record ID: credit-a\n   account_id: credit-a\n   user_id: person-a\n"
        "   card_type: Fictional Card\n   account_status: ACTIVE\n",
    )
    return ledger, "credit-a", "account_id", "credit-read"


@pytest.mark.parametrize("kind", ["bank_account", "credit_card", "debit_card"])
@pytest.mark.parametrize("operator", ["equal", "not_equal"])
@pytest.mark.parametrize("same_identity", [False, True])
def test_identity_predicates_compare_authoritative_resource_identity(kind, operator, same_identity):
    ledger, resource, field, call_id = identity_observation(kind)
    assert ledger.resource_evidence(call_id, kind, resource) is not None
    parsed = parse_condition(
        condition(
            resource_kind=kind,
            resource_id=resource,
            observed_in=call_id,
            checks=[
                {
                    "field": field,
                    "operator": operator,
                    "value": resource if same_identity else "unrelated-resource",
                }
            ],
        )
    )
    satisfied = same_identity if operator == "equal" else not same_identity
    assert parsed.gap(ledger) == (None if satisfied else "recorded_condition_not_satisfied")
    execute(
        ledger,
        "mutation",
        "call_discoverable_agent_tool",
        {
            "agent_tool_name": "apply_adjustment",
            "arguments": json.dumps({field: resource}),
        },
    )
    assert parsed.gap(ledger) == "resource_observation_stale_or_unavailable"


@pytest.mark.parametrize("kind", ["bank_account", "credit_card", "debit_card"])
def test_identity_fields_do_not_accept_ordering_operators(kind):
    _, resource, field, call_id = identity_observation(kind)
    with pytest.raises(ValueError, match="numeric comparison requires current_holdings"):
        parse_condition(
            condition(
                resource_kind=kind,
                resource_id=resource,
                observed_in=call_id,
                checks=[{"field": field, "operator": "at_least", "value": resource}],
            )
        )


def test_condition_schema_advertises_both_account_and_card_identity():
    fields = condition_schema({})["properties"]["checks"]["items"]["properties"]["field"]
    assert {"account_id", "card_id"} <= set(fields["enum"])


def question(**changes):
    return {
        "id": "missing_address",
        "goal": "Ask for the destination required to proceed",
        "kind": "customer_input",
        "status": "ready",
        "evidence": "The request does not provide the required destination.",
        "evidence_refs": [{"kind": "customer_message", "reference_id": "customer-message:1"}],
        **changes,
    }


@pytest.mark.parametrize("status", ["pending", "in_progress"])
@pytest.mark.parametrize("updating", [False, True])
def test_explicit_question_derives_unresolved_parent_state(status, updating):
    controller, ledger = setup_plan()
    item = {"id": "request", "status": status, "obligations": [question()]}
    if not updating:
        controller = BankingWorkflowController(ledger, planning_mode="outcomes")
        item = {**outcome(status=status), "obligations": [question()]}
    original = deepcopy(item)
    result = controller.invoke(
        UPDATE_WORK_PLAN_TOOL if updating else SYNC_WORK_PLAN_TOOL, {"items": [item]}
    )
    assert not result.is_error, result.content
    assert item == original
    saved = controller.export_state()["items"][0]
    assert saved["status"] == "awaiting_customer"
    assert saved["outcome"] == "unresolved"
    assert saved["obligations"][0]["status"] == "ready"
    rendered = json.loads(
        controller.render_context().split("<workflow-plan>\n")[1].split("\n</workflow-plan>")[0]
    )
    assert rendered["response_contract"]["mode"] == "customer_input_only"
    assert not rendered["response_contract"]["task_complete"]


def test_question_derivation_still_requires_real_evidence_atomically():
    controller, _ = setup_plan()
    before = controller.export_state()
    for change in (
        {"evidence_refs": []},
        {"evidence": ""},
        {"evidence_refs": [{"kind": "customer_message", "reference_id": "invented"}]},
    ):
        result = controller.invoke(
            UPDATE_WORK_PLAN_TOOL,
            {"items": [{"id": "request", "obligations": [question(**change)]}]},
        )
        assert result.is_error
        assert controller.export_state() == before


@pytest.mark.parametrize(
    "status,outcome_value", [("completed", "succeeded"), ("blocked", "blocked")]
)
def test_question_cannot_silently_replace_terminal_or_blocked_intent(status, outcome_value):
    controller, _ = setup_plan()
    before = controller.export_state()
    result = controller.invoke(
        UPDATE_WORK_PLAN_TOOL,
        {
            "items": [
                {
                    "id": "request",
                    "status": status,
                    "outcome": outcome_value,
                    "evidence": "Need customer input",
                    "obligations": [question()],
                    "evidence_refs": question()["evidence_refs"],
                }
            ]
        },
    )
    assert result.is_error
    assert controller.export_state() == before


def test_derivation_does_not_replace_an_explicitly_invalid_parent_reference():
    controller, _ = setup_plan()
    before = controller.export_state()
    result = controller.invoke(
        UPDATE_WORK_PLAN_TOOL,
        {
            "items": [
                {
                    "id": "request",
                    "status": "in_progress",
                    "obligations": [question()],
                    "evidence_refs": [{"kind": "customer_message", "reference_id": "invented"}],
                }
            ]
        },
    )
    assert result.is_error
    assert controller.export_state() == before


@pytest.mark.parametrize(
    "field,value",
    [("evidence", value) for value in (None, 0, False, {}, [])]
    + [("evidence_refs", value) for value in (None, 0, False, {}, "")],
)
def test_phase_derivation_does_not_normalize_invalid_parent_field_types(field, value):
    controller, _ = setup_plan()
    before = controller.export_state()
    result = controller.invoke(
        UPDATE_WORK_PLAN_TOOL,
        {
            "items": [
                {
                    "id": "request",
                    "status": "in_progress",
                    "obligations": [question()],
                    field: value,
                }
            ]
        },
    )
    assert result.is_error
    assert controller.export_state() == before


def test_no_question_or_waiting_question_does_not_derive_customer_handoff():
    for obligations in ([], [question(status="waiting")]):
        controller, _ = setup_plan()
        result = controller.invoke(
            UPDATE_WORK_PLAN_TOOL,
            {
                "items": [
                    {
                        "id": "request",
                        "status": "in_progress",
                        "obligations": obligations,
                    }
                ]
            },
        )
        assert not result.is_error
        assert controller.export_state()["items"][0]["status"] == "in_progress"


def test_derived_question_does_not_skip_its_own_preparation():
    controller, _ = setup_plan()
    result = controller.invoke(
        UPDATE_WORK_PLAN_TOOL,
        {
            "items": [
                {
                    "id": "request",
                    "obligations": [
                        question(),
                        {
                            "id": "verify_destination_options",
                            "goal": "Inspect permitted destinations",
                            "kind": "action",
                            "status": "ready",
                        },
                    ],
                }
            ]
        },
    )
    assert not result.is_error, result.content
    assert not controller.preview_terminal_output("Which destination?").allowed


def test_outcome_schema_does_not_ask_model_to_copy_integrity_metadata():
    controller, _ = setup_plan()
    for tool in controller.registration_tool_schemas():
        item = tool["input_schema"]["properties"]["items"]["items"]
        obligation = item["properties"]["obligations"]["items"]["properties"]
        for ref in (
            item["properties"]["evidence_refs"]["items"],
            obligation["evidence_refs"]["items"],
            obligation["condition"]["properties"]["procedure"],
        ):
            assert set(ref["properties"]) == {"kind", "reference_id", "version"}


def test_context_projects_copyable_refs_but_persistence_keeps_integrity():
    controller, _, result = setup()
    assert not result.is_error
    ref = {
        "kind": "knowledge_revision",
        "reference_id": "additional-policy",
        "version": "1",
        "digest": "a" * 64,
    }
    controller.observe_knowledge_evidence([WorkEvidenceReference(**ref)])
    result = controller.invoke(
        UPDATE_WORK_PLAN_TOOL,
        {
            "items": [
                {
                    "id": "request",
                    "evidence": "Additional policy",
                    "evidence_refs": [ref],
                    "obligations": [{"id": "eligibility", "evidence_refs": [ref]}],
                }
            ]
        },
    )
    assert not result.is_error, result.content
    before = controller.export_state()
    context = controller.render_context()
    assert '"digest"' not in context and '"available"' not in context
    assert controller.export_state() == before
    assert before["items"][0]["evidence_refs"][0]["digest"] == "a" * 64
    result = controller.invoke(
        UPDATE_WORK_PLAN_TOOL,
        {
            "items": [
                {
                    "id": "request",
                    "evidence_refs": [{**ref, "digest": "b" * 64}],
                }
            ]
        },
    )
    assert result.is_error and "digest does not match" in result.content
    assert controller.export_state() == before


@pytest.mark.parametrize("value", ["04/05/2026", "2026-02-30", "20260405"])
def test_invalid_date_has_actionable_error_and_is_never_guessed(value):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_condition(
            condition(checks=[{"field": "opened_on", "operator": "equal", "value": value}])
        )


def test_date_schema_explains_exact_format_before_first_attempt():
    schema = condition_schema({})
    value = schema["properties"]["checks"]["items"]["properties"]["value"]
    assert "YYYY-MM-DD" in value["description"]
    assert "current_holdings" in value["description"]
