from tau2.data_model.message import AssistantMessage, UserMessage
from test_outcome_controller import obligation, outcome, setup_plan

from cayu_tau3_banking.contract_conflict_probe import CONFLICT, SUPERSEDED, assess_hold
from cayu_tau3_banking.resource_outcome_probe import SyntheticWorkspace


def test_explicit_supersession_is_the_only_policy_difference():
    assert SUPERSEDED.startswith(CONFLICT)
    assert "explicitly supersedes" in SUPERSEDED[len(CONFLICT) :]


def test_conflict_assessment_requires_plan_and_no_effects():
    host = SyntheticWorkspace(mismatch=False)
    pending = {"items": [outcome()]}
    assert assess_hold(host, [pending, pending])["structural_pass"]
    assert not assess_hold(host, [{"items": []}])["structural_pass"]
    assert not assess_hold(host, [{"items": [outcome(status="completed")]}])["structural_pass"]
    host.execute("create_workspace", {"sku": "arc_team_7", "scope": "organization"})
    assert not assess_hold(host, [pending, pending])["structural_pass"]


def test_delivered_information_does_not_automatically_reconcile_on_next_customer_turn():
    controller, ledger = setup_plan(
        {
            **outcome(completion_mode="information"),
            "obligations": [obligation(kind="communication", goal="Explain the result")],
        }
    )
    before = controller.export_state()["items"]
    assert controller.review_terminal_output("Here is the result.").allowed
    ledger.observe_assistant(AssistantMessage.text("Here is the result."))
    assert controller.export_state()["items"] == before
    ledger.observe_input(UserMessage.text("Thanks. What can we do next?"))
    assert controller.export_state()["items"] == before
    # The next customer epoch permits another response, but does not declare
    # the earlier communication semantically complete without a plan update.
    assert controller.preview_terminal_output("Let us discuss next steps.").allowed
