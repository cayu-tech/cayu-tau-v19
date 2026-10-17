"""Fictional account transitions: no task catalog, reference answers or live APIs."""

import json
from copy import deepcopy
from decimal import Decimal

import pytest
from cayu import ToolCallPart, WorkEvidenceReference
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from test_outcome_controller import outcome

from cayu_tau3_banking.case_conditions import ConditionEvaluation, parse_condition
from cayu_tau3_banking.controller import (
    SYNC_WORK_PLAN_TOOL,
    UPDATE_WORK_PLAN_TOOL,
    BankingWorkflowController,
)
from cayu_tau3_banking.ledger import ConversationLedger

POLICY = {"kind": "knowledge_revision", "reference_id": "fictional-eligibility", "version": "1"}


def execute(ledger, call_id, name, arguments, content="Done", error=False):
    ledger.observe_assistant(
        AssistantMessage.text("", tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])
    )
    ledger.observe_input(ToolMessage(role="tool", id=call_id, content=content, error=error))


def read(
    ledger,
    call_id="read-1",
    account_id="account-a",
    opened="01/01/2020",
    status="ACTIVE",
    user="person-a",
):
    content = (
        "Bank Accounts:\nFound 1 record(s) in 'accounts':\n\n"
        f"1. Record ID: {account_id}\n   account_id: {account_id}\n   user_id: {user}\n"
        f"   class: Checking\n   level: Standard\n   status: {status}\n   current_holdings: $100\n"
        + (f"   date_opened: {opened}\n" if opened is not None else "")
        + "\nCredit Card Accounts:\nNo credit card accounts found."
    )
    execute(
        ledger,
        call_id,
        "call_discoverable_agent_tool",
        {
            "agent_tool_name": "get_all_user_accounts_by_user_id_3847",
            "arguments": json.dumps({"user_id": user}),
        },
        content,
    )
    return content


def condition(**changes):
    return {
        "resource_kind": "bank_account",
        "resource_id": "account-a",
        "observed_in": "read-1",
        "procedure": dict(POLICY),
        "for_tools": ["apply_adjustment"],
        "checks": [
            {"field": "status", "operator": "equal", "value": "ACTIVE"},
            {"field": "opened_on", "operator": "on_or_before", "value": "2025-01-01"},
        ],
        **changes,
    }


def obligation(**changes):
    return {
        "id": "eligibility",
        "goal": "Establish all adjustment prerequisites",
        "kind": "action",
        "status": "resolved",
        "condition": condition(),
        "evidence": "The declared conditions hold for this account.",
        "evidence_refs": [dict(POLICY), {"kind": "tool_result", "reference_id": "read-1"}],
        **changes,
    }


def setup(*, opened="01/01/2020", obligations=None):
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Please perform the adjustment if eligible."))
    read(ledger, opened=opened)
    controller = BankingWorkflowController(ledger, planning_mode="outcomes")
    controller.observe_knowledge_evidence([WorkEvidenceReference(**POLICY)])
    planned = outcome()
    planned["obligations"] = [obligation()] if obligations is None else obligations
    result = controller.invoke(SYNC_WORK_PLAN_TOOL, {"items": [planned]})
    return controller, ledger, result


def patch(controller, **changes):
    return controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [{"id": "request", **changes}]})


def admitted(controller, name="apply_adjustment", account_id="account-a"):
    return controller.review(
        [
            ToolCallPart(
                tool_call_id="proposed",
                tool_name="call_discoverable_agent_tool",
                arguments={
                    "agent_tool_name": name,
                    "arguments": json.dumps({"account_id": account_id}),
                },
            )
        ]
    )[0].allowed


def test_active_and_old_enough_are_independent_conditions():
    controller, ledger, result = setup()
    assert not result.is_error
    assert admitted(controller)
    assert setup(opened="09/01/2026")[2].is_error
    assert setup(opened=None)[2].is_error
    read(ledger, "new-read", "account-new", "09/01/2026")
    assert not admitted(controller, account_id="account-new")
    update = obligation(
        condition=condition(resource_id="account-new", observed_in="new-read"),
        evidence_refs=[POLICY, {"kind": "tool_result", "reference_id": "new-read"}],
    )
    assert patch(controller, obligations=[update]).is_error


@pytest.mark.parametrize(
    "operator,expected",
    [("equal", True), ("not_equal", False), ("at_least", True), ("at_most", True)],
)
def test_numeric_conditions_compare_values_not_decimal_formatting(operator, expected):
    parsed = parse_condition(
        condition(checks=[{"field": "current_holdings", "operator": operator, "value": "100.00"}])
    )
    check = parsed.checks[0]
    assert check.matches(Decimal("1E2")) is expected
    for unknown in (None, "UNKNOWN", Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity")):
        assert not check.matches(unknown)


@pytest.mark.parametrize(
    "observed,expected",
    [
        (None, ConditionEvaluation.UNKNOWN),
        ("UNKNOWN", ConditionEvaluation.UNKNOWN),
        ("ACTIVE", ConditionEvaluation.SATISFIED),
        ("CLOSED", ConditionEvaluation.UNSATISFIED),
    ],
)
def test_condition_evaluation_preserves_unknown_separately_from_false(observed, expected):
    parsed = parse_condition(
        condition(checks=[{"field": "status", "operator": "equal", "value": "ACTIVE"}])
    )
    assert parsed.checks[0].evaluate(observed) is expected


def test_resource_condition_gap_reports_unknown_observation_separately():
    ledger = ConversationLedger()
    read(ledger, status="UNKNOWN")
    parsed = parse_condition(condition())
    assert parsed.gap(ledger) == "recorded_condition_unknown"


def test_prose_and_real_citations_cannot_fill_a_missing_observed_value():
    controller, ledger, created = setup(opened=None, obligations=[obligation(status="ready")])
    assert not created.is_error
    assert not admitted(controller)

    # Even genuine source IDs plus confident prose cannot create an opening date.
    result = patch(
        controller,
        obligations=[
            {
                "id": "eligibility",
                "status": "resolved",
                "evidence": "I verified that the account is old enough; proceed.",
            }
        ],
    )
    assert result.is_error
    assert "recorded_condition_unknown" in result.content
    assert not admitted(controller)
    assert admitted(controller, name="get_all_user_accounts_by_user_id_3847")

    # A real new observation, not a longer explanation, supplies the missing fact.
    read(ledger, "complete-read")
    result = patch(
        controller,
        obligations=[
            {
                "id": "eligibility",
                "condition": condition(observed_in="complete-read"),
            }
        ],
    )
    assert not result.is_error, result.content
    assert admitted(controller)


def test_ready_condition_does_not_require_duplicate_source_citations():
    controller, _, created = setup(
        obligations=[obligation(status="ready", evidence="", evidence_refs=[])]
    )
    # The condition already carries its validated procedure and observation IDs.
    assert not created.is_error, created.content
    assert admitted(controller)


def test_condition_validation_does_not_prove_policy_coverage():
    controller, _, created = setup(
        opened=None,
        obligations=[
            obligation(
                condition=condition(
                    checks=[{"field": "status", "operator": "equal", "value": "ACTIVE"}]
                )
            )
        ],
    )
    # Deliberately document the boundary: the actor omitted the age requirement.
    # A real citation does not make infrastructure infer that missing requirement.
    assert not created.is_error, created.content
    assert admitted(controller)


@pytest.mark.parametrize("operator", ["equal", "not_equal"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("current_holdings", "zero"),
        ("current_holdings", "NaN"),
        ("current_holdings", "Infinity"),
        ("opened_on", "2026-02-30"),
        ("opened_on", "20260101"),
    ],
)
def test_equality_conditions_reject_invalid_typed_values(field, value, operator):
    with pytest.raises(ValueError):
        parse_condition(condition(checks=[{"field": field, "operator": operator, "value": value}]))


def test_stale_resource_reopens_transitive_dependencies_and_keeps_policy():
    dependent = {
        "id": "decision",
        "goal": "Assess next action",
        "status": "resolved",
        "depends_on": ["eligibility"],
        "evidence": "Prerequisite satisfied",
        "evidence_refs": [POLICY],
    }
    controller, ledger, result = setup(obligations=[obligation(), dependent])
    assert not result.is_error
    before = controller.export_state()
    read(ledger, "read-2", status="CLOSED")
    after = controller.export_state()
    assert after["revision"] == before["revision"] + 1
    assert [o["status"] for o in after["items"][0]["obligations"]] == ["ready", "ready"]
    assert not admitted(controller)
    assert admitted(controller, name="get_all_user_accounts_by_user_id_3847")
    assert not controller.preview_terminal_output("Finished.").allowed
    assert controller.export_state() == after  # Reconciliation is idempotent.
    assert controller.active_knowledge_references()[0].reference_id == POLICY["reference_id"]


def test_customer_change_reopens_only_customer_sensitive_conditions():
    controller, ledger, result = setup(
        obligations=[obligation(condition=condition(customer_message_id="customer-message:1"))]
    )
    assert not result.is_error
    ledger.observe_input(UserMessage.text("Change the intended usage."))
    assert controller.export_state()["items"][0]["obligations"][0]["status"] == "ready"
    changed = obligation(condition=condition(customer_message_id="customer-message:2"))
    assert not patch(controller, obligations=[changed]).is_error
    other, other_ledger, _ = setup()
    other_ledger.observe_input(UserMessage.text("Thanks for checking."))
    assert other.export_state()["items"][0]["obligations"][0]["status"] == "resolved"


@pytest.mark.parametrize("error", [False, True])
def test_successful_effect_invalidates_reads_but_failed_effect_does_not(error):
    controller, ledger, _ = setup()
    execute(
        ledger,
        "mutation",
        "call_discoverable_agent_tool",
        {"agent_tool_name": "change_account", "arguments": json.dumps({"account_id": "account-a"})},
        error=error,
    )
    assert admitted(controller) is error
    if not error:
        assert patch(controller, obligations=[obligation()]).is_error
        read(ledger, "fresh")
        assert not patch(
            controller,
            obligations=[
                obligation(
                    condition=condition(observed_in="fresh"),
                    evidence_refs=[POLICY, {"kind": "tool_result", "reference_id": "fresh"}],
                )
            ],
        ).is_error
        assert admitted(controller)


def test_unrelated_resource_write_does_not_invalidate_same_read():
    controller, ledger, _ = setup()
    execute(
        ledger,
        "mutation",
        "call_discoverable_agent_tool",
        {
            "agent_tool_name": "change_account",
            "arguments": json.dumps({"account_id": "different-account"}),
        },
    )
    assert admitted(controller)
    read(ledger, "same-again")
    assert controller.export_state()["items"][0]["obligations"][0]["status"] == "resolved"


@pytest.mark.parametrize(
    "change",
    [
        {"resource_id": "unknown"},
        {"observed_in": "missing"},
        {"resource_kind": "credit_card"},
        {"procedure": {**POLICY, "version": "2"}},
        {"customer_message_id": "missing"},
        {"checks": []},
        {"checks": [{"field": "opened_on", "operator": "at_most", "value": "1"}]},
        {"checks": [{"field": "current_holdings", "operator": "at_least", "value": "NaN"}]},
        {"checks": [{"field": "opened_on", "operator": "on_or_before", "value": "not-a-date"}]},
    ],
)
def test_invalid_or_unavailable_condition_rejects_atomically(change):
    controller, _, _ = setup()
    before = controller.export_state()
    assert patch(controller, obligations=[obligation(condition=condition(**change))]).is_error
    assert controller.export_state() == before


def test_cycles_unknown_dependencies_and_removing_condition_reject():
    controller, _, _ = setup()
    for deps in (["eligibility"], ["unknown"]):
        assert patch(controller, obligations=[{"id": "eligibility", "depends_on": deps}]).is_error
    assert patch(controller, obligations=[{"id": "eligibility", "condition": None}]).is_error


def test_dependency_source_change_requires_reassessment_but_prose_does_not():
    dependent = {
        "id": "decision",
        "goal": "Assess next action",
        "status": "resolved",
        "depends_on": ["eligibility"],
        "evidence": "Prerequisite satisfied",
        "evidence_refs": [POLICY],
    }
    controller, _, _ = setup(obligations=[obligation(), dependent])
    assert not patch(
        controller,
        obligations=[
            {"id": "eligibility", "evidence": "Additional interpretation of unchanged facts"}
        ],
    ).is_error
    assert controller.export_state()["items"][0]["obligations"][1]["status"] == "resolved"
    assert not patch(
        controller, obligations=[{"id": "eligibility", "evidence_refs": [POLICY]}]
    ).is_error
    assert controller.export_state()["items"][0]["obligations"][1]["status"] == "ready"


def test_snapshot_roundtrip_and_legacy_schema_cannot_smuggle_conditions():
    controller, ledger, _ = setup()
    saved = controller.export_state()
    assert saved["schema_version"] == 18
    history = [AssistantMessage.text("Continuing", raw_data={"cayu_tau3_workflow": saved})]
    restored = BankingWorkflowController.from_history(
        history, ledger=ledger, planning_mode="outcomes"
    )
    assert restored.export_state() == saved
    assert restored.active_knowledge_references() == controller.active_knowledge_references()
    legacy = deepcopy(saved)
    legacy["schema_version"] = 16
    assert not BankingWorkflowController.from_history(
        [AssistantMessage.text("Old", raw_data={"cayu_tau3_workflow": legacy})],
        ledger=ledger,
        planning_mode="outcomes",
    ).has_work_items()


def test_literals_and_missing_values_are_not_inferred():
    assert parse_condition(condition()) is not None
    assert parse_condition(condition()).checks[0].matches("ACTIVE")
    assert not parse_condition(condition()).checks[0].matches(None)


def test_customer_tool_handoff_obeys_same_resource_prerequisite():
    controller, ledger, _ = setup()
    read(ledger, "changed", opened="09/01/2026")
    decision = controller.review(
        [
            ToolCallPart(
                tool_call_id="handoff",
                tool_name="give_discoverable_user_tool",
                arguments={
                    "discoverable_tool_name": "apply_adjustment",
                    "arguments": json.dumps({"account_id": "account-a"}),
                },
            )
        ]
    )[0]
    assert not decision.allowed


def test_resource_reads_are_bounded_and_replayable():
    ledger = ConversationLedger()
    for number in range(70):
        read(ledger, f"read-{number}")
    assert len(ledger._resource_reads) == 64
    assert ledger.resource_evidence("read-0", "bank_account", "account-a") is None
    assert ledger.resource_evidence_refs() == [
        {"resource_kind": "bank_account", "resource_id": "account-a", "observed_in": "read-69"}
    ]


def test_completed_effect_is_not_reopened_by_later_resource_changes():
    controller, ledger, _ = setup()
    execute(ledger, "done", "apply_adjustment", {"account_id": "account-a"})
    assert not patch(
        controller,
        status="completed",
        outcome="succeeded",
        evidence="Successful authoritative operation",
        evidence_refs=[{"kind": "tool_result", "reference_id": "done"}],
    ).is_error
    saved = controller.export_state()["items"][0]
    read(ledger, "later", status="CLOSED")
    ledger.observe_input(UserMessage.text("Please handle another request."))
    assert not controller.invoke(UPDATE_WORK_PLAN_TOOL, {"items": [outcome("other")]}).is_error
    assert controller.export_state()["items"][0] == saved
    assert patch(
        controller, obligations=[{"id": "eligibility", "condition": condition(resource_id="other")}]
    ).is_error


def test_pre_execution_receipt_survives_own_effect_but_cannot_authorize_a_repeat():
    controller, ledger, _ = setup()
    call = ToolCallPart(
        tool_call_id="effect",
        tool_name="call_discoverable_agent_tool",
        arguments={
            "agent_tool_name": "apply_adjustment",
            "arguments": json.dumps({"account_id": "account-a"}),
        },
    )
    controller.before_condition_execution(call)
    execute(ledger, call.tool_call_id, call.tool_name, call.arguments)
    controller.after_condition_execution(call.tool_call_id)
    saved = controller.export_state()
    checked = saved["items"][0]["obligations"][0]
    assert checked["status"] == "resolved"
    assert checked["applied_to"] == "effect"
    assert not admitted(controller)  # A second effect needs fresh evidence.
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Effect observed", raw_data={"cayu_tau3_workflow": saved})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert restored.export_state() == saved
    assert not patch(
        restored,
        status="completed",
        outcome="succeeded",
        evidence="Authoritative effect succeeded",
        evidence_refs=[{"kind": "tool_result", "reference_id": "effect"}],
    ).is_error
    assert patch(restored, obligations=[{"id": "eligibility", "applied_to": "effect"}]).is_error


def test_failed_execution_never_mints_a_condition_receipt():
    controller, ledger, _ = setup()
    call = ToolCallPart(
        tool_call_id="effect",
        tool_name="call_discoverable_agent_tool",
        arguments={
            "agent_tool_name": "apply_adjustment",
            "arguments": json.dumps({"account_id": "account-a"}),
        },
    )
    controller.before_condition_execution(call)
    execute(ledger, call.tool_call_id, call.tool_name, call.arguments, error=True)
    controller.after_condition_execution(call.tool_call_id)
    assert "applied_to" not in controller.export_state()["items"][0]["obligations"][0]


@pytest.mark.parametrize("wrapped", [False, True])
def test_customer_scoped_effect_retains_the_existing_resource_prerequisite(wrapped):
    controller, ledger, _ = setup()
    arguments = {"user_id": "person-a"}
    call = ToolCallPart(
        tool_call_id="new-resource-effect",
        tool_name="call_discoverable_agent_tool" if wrapped else "apply_adjustment",
        arguments={"agent_tool_name": "apply_adjustment", "arguments": json.dumps(arguments)}
        if wrapped
        else arguments,
    )
    assert controller.review([call])[0].allowed
    controller.before_condition_execution(call)
    execute(ledger, call.tool_call_id, call.tool_name, call.arguments)
    controller.after_condition_execution(call.tool_call_id)
    saved = controller.export_state()
    assert saved["items"][0]["obligations"][0]["applied_to"] == call.tool_call_id
    read(ledger, "changed-after-effect", status="CLOSED")
    assert not controller.review([call])[0].allowed
    restored = BankingWorkflowController.from_history(
        [AssistantMessage.text("Observed", raw_data={"cayu_tau3_workflow": saved})],
        ledger=ledger,
        planning_mode="outcomes",
    )
    assert not patch(
        restored,
        status="completed",
        outcome="succeeded",
        evidence="The effect succeeded",
        evidence_refs=[{"kind": "tool_result", "reference_id": call.tool_call_id}],
        obligations=[{"id": "eligibility", "status": "resolved"}],
    ).is_error


@pytest.mark.parametrize(
    "arguments",
    [
        {"user_id": "another-person"},
        {"user_id": "another-person", "account_id": "account-a"},
        {"user_id": "person-a", "account_id": "another-account"},
        {"user_id": "person-a", "source_account_id": "another-account"},
    ],
)
def test_customer_scope_never_overrides_a_different_explicit_resource(arguments):
    controller, ledger, _ = setup()
    call = ToolCallPart(
        tool_call_id="wrong-scope", tool_name="apply_adjustment", arguments=arguments
    )
    assert not controller.review([call])[0].allowed
    controller.before_condition_execution(call)
    execute(ledger, call.tool_call_id, call.tool_name, call.arguments)
    controller.after_condition_execution(call.tool_call_id)
    assert "applied_to" not in controller.export_state()["items"][0]["obligations"][0]


def test_account_inventory_accepts_created_record_shape_and_rejects_conflicting_aliases():
    ledger = ConversationLedger()
    original = read(ledger)
    created = original.replace("class: Checking", "account_type: Checking").replace(
        "level: Standard", "account_class: Standard"
    )
    arguments = {
        "agent_tool_name": "get_all_user_accounts_by_user_id_3847",
        "arguments": json.dumps({"user_id": "person-a"}),
    }
    execute(ledger, "created-shape", "call_discoverable_agent_tool", arguments, created)
    assert ledger.resource_evidence("created-shape", "bank_account", "account-a") is not None
    mixed = original.replace("Found 1 record(s)", "Found 2 record(s)").replace(
        "\nCredit Card Accounts:",
        "\n"
        + created.split("\n\n", 1)[1]
        .split("\nCredit Card Accounts:")[0]
        .replace("1. Record ID", "2. Record ID")
        .replace("account-a", "account-b")
        + "\nCredit Card Accounts:",
    )
    execute(ledger, "mixed", "call_discoverable_agent_tool", arguments, mixed)
    assert {a.account_id for a in ledger.bank_accounts("person-a")} == {"account-a", "account-b"}
    for index, conflict in enumerate(("class: Other", "level: Other")):
        execute(
            ledger,
            f"conflict-{index}",
            "call_discoverable_agent_tool",
            arguments,
            created.replace("account_type: Checking", "account_type: Checking\n   " + conflict),
        )
        assert ledger.resource_evidence(f"conflict-{index}", "bank_account", "account-a") is None


@pytest.mark.parametrize("dependency", [None, "eligibility", [[]], [1], {}, ["eligibility"] * 8])
def test_malformed_dependency_patches_never_escape_as_runtime_exceptions(dependency):
    controller, _, _ = setup()
    before = controller.export_state()
    assert patch(controller, obligations=[{"id": "eligibility", "depends_on": dependency}]).is_error
    assert controller.export_state() == before


def test_native_resource_replacement_after_compaction_blocks_effect_not_repair(tmp_path):
    import asyncio

    from cayu import EventType, ScriptedModelProvider, SQLiteSessionStore, ToolResultPart
    from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
    from tau2.environment.tool import Tool
    from test_runtime_banking import assembly, call, text

    from cayu_tau3_banking.product_navigation import READ_TOOL

    def get_all_user_accounts_by_user_id_3847(user_id: str) -> str:
        """Read public account records.

        Args:
            user_id: Customer whose accounts should be read.
        """
        raise AssertionError("Only the external host executes bank tools")

    def apply_adjustment(account_id: str) -> str:
        """Apply the requested adjustment.

        Args:
            account_id: Target account.
        """
        raise AssertionError("Ineligible effect must never be dispatched")

    (tmp_path / f"{POLICY['reference_id']}.json").write_text(
        json.dumps(
            {
                "id": POLICY["reference_id"],
                "title": "Fictional adjustment procedure",
                "content": "Adjustment requires ACTIVE status and opening on or before January 1, 2025.",
            }
        )
    )
    summary = ScriptedModelProvider([text("Prior detail omitted.") for _ in range(30)])
    compaction = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(provider=summary, model="lossy"),
        compact_after_messages=2,
        max_user_turns=1,
    )
    initial_check = obligation(
        condition=condition(observed_in="initial-read", customer_message_id="customer-message:1"),
        evidence_refs=[POLICY, {"kind": "tool_result", "reference_id": "initial-read"}],
    )
    blocked_check = obligation(
        status="waiting",
        condition=condition(
            resource_id="replacement",
            observed_in="replacement-read",
            customer_message_id="customer-message:2",
        ),
        evidence="Replacement is active but too new under the procedure.",
        evidence_refs=[POLICY, {"kind": "tool_result", "reference_id": "replacement-read"}],
    )
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [outcome()]}, "create-plan"),
            call(
                READ_TOOL,
                {"documents": [{"entry_id": POLICY["reference_id"], "revision": 1}]},
                "policy",
            ),
            call("get_all_user_accounts_by_user_id_3847", {"user_id": "person-a"}, "initial-read"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "request",
                            "status": "awaiting_customer",
                            "evidence": "Awaiting customer confirmation of target",
                            "obligations": [initial_check],
                        }
                    ]
                },
                "initial-check",
            ),
            text("Which account should I use?"),
            call(
                "get_all_user_accounts_by_user_id_3847", {"user_id": "person-a"}, "replacement-read"
            ),
            call("apply_adjustment", {"account_id": "replacement"}, "must-not-execute"),
            call(
                UPDATE_WORK_PLAN_TOOL,
                {
                    "items": [
                        {
                            "id": "request",
                            "status": "blocked",
                            "outcome": "blocked",
                            "evidence": "The requested replacement account does not meet the age condition.",
                            "evidence_refs": [
                                POLICY,
                                {"kind": "tool_result", "reference_id": "replacement-read"},
                            ],
                            "obligations": [blocked_check],
                        }
                    ]
                },
                "repair",
            ),
            text(
                "The replacement is active but does not meet the required age. No adjustment was made."
            ),
        ],
        planning_mode="outcomes",
        advice_mode="lean",
        persistent=True,
        compaction=compaction,
        callback_timeout_s=90,
        host_tools=[Tool(get_all_user_accounts_by_user_id_3847), Tool(apply_adjustment)],
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Please adjust my account if eligible."), state
        )
        assert reply.tool_calls[0].name == "get_all_user_accounts_by_user_id_3847"
        reply, _ = agent.generate_next_message(
            ToolMessage(role="tool", id=reply.tool_calls[0].id, content=read(ConversationLedger())),
            state,
        )
        assert reply.content == "Which account should I use?"
        reply, _ = agent.generate_next_message(
            UserMessage.text("Use the replacement instead."), state
        )
        assert reply.tool_calls[0].name == "get_all_user_accounts_by_user_id_3847"
        reply, _ = agent.generate_next_message(
            ToolMessage(
                role="tool",
                id=reply.tool_calls[0].id,
                content=read(ConversationLedger(), account_id="replacement", opened="09/01/2026"),
            ),
            state,
        )
        assert "No adjustment was made" in reply.content
        bank = captured["state"]
        assert bank.ledger.tool_action_status("must-not-execute") is None
        saved = bank.controller.export_state()
        assert saved["items"][0]["status"] == "blocked"
        assert saved["items"][0]["obligations"][0]["condition"]["resource_id"] == "replacement"
        assert summary.requests
        results = [
            part
            for request in provider.requests
            for message in request.messages
            for part in message.content
            if isinstance(part, ToolResultPart)
        ]
        assert any(part.is_error for part in results)
        session_id = bank.session_id
    finally:
        agent.stop(state=state)

    async def persisted():
        store = SQLiteSessionStore(tmp_path / "runtime.sqlite3")
        try:
            events = await store.load_events(session_id)
            starts = [e for e in events if e.type == EventType.TOOL_CALL_STARTED]
            assert (
                len([e for e in starts if e.tool_name == "get_all_user_accounts_by_user_id_3847"])
                == 2
            )
            # Cayu records a started attempt even for a pre-execution policy denial.
            denied = [e for e in events if e.payload.get("tool_call_id") == "must-not-execute"]
            blocked = [e for e in denied if e.type == EventType.TOOL_CALL_BLOCKED]
            assert len(blocked) == 1
            assert blocked[0].payload["denied_by"] == "tool_policy"
            assert "different resource" in blocked[0].payload["reason"]
            assert not any(e.type == EventType.TOOL_CALL_COMPLETED for e in denied)
        finally:
            await store.close()

    asyncio.run(persisted())
