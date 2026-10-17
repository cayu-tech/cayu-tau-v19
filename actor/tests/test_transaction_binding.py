import json
from types import SimpleNamespace

import pytest

from cayu_tau3_banking.transaction_binding import bind_transaction_fields


def ledger(date="02/17/2025"):
    return SimpleNamespace(
        credit_card_transactions=lambda user: [
            SimpleNamespace(user_id="u1", transaction_id="t1", transaction_date=date)
        ]
    )


def call(date="@transaction_date", **kwargs):
    return {
        "agent_tool_name": "file_credit_card_transaction_dispute_4829",
        "arguments": json.dumps(
            {
                "user_id": "u1",
                "transaction_id": "t1",
                "purchase_date": date,
                "resolution_requested": "partial_refund",
                "amount": 12.34,
                "issue_noticed_date": "02/20/2025",
                **kwargs,
            }
        ),
    }


def test_explicit_binding_copies_record_and_preserves_all_other_fields():
    args = call()
    bound = bind_transaction_fields(ledger(), "call_discoverable_agent_tool", args)
    old = json.loads(args["arguments"])
    new = json.loads(bound["arguments"])
    assert new == {**old, "purchase_date": "02/17/2025"}
    assert old["purchase_date"] == "@transaction_date"
    assert bind_transaction_fields(ledger(), "call_discoverable_agent_tool", bound) == bound


def test_wrong_literal_is_rejected_not_silently_corrected():
    with pytest.raises(ValueError, match="disagrees"):
        bind_transaction_fields(ledger(), "call_discoverable_agent_tool", call("02/16/2025"))


@pytest.mark.parametrize("kwargs", [{"user_id": "other"}, {"transaction_id": "other"}])
def test_cross_user_and_unknown_transaction_cannot_bind(kwargs):
    with pytest.raises(ValueError, match="Read"):
        bind_transaction_fields(ledger(), "call_discoverable_agent_tool", call(**kwargs))


def test_unrelated_tool_and_literal_without_observation_are_unchanged():
    args = call()
    assert bind_transaction_fields(ledger(), "another_tool", args) == args
    args = call("02/17/2025")
    assert (
        bind_transaction_fields(
            SimpleNamespace(credit_card_transactions=lambda _: None),
            "call_discoverable_agent_tool",
            args,
        )
        == args
    )


def test_runtime_dispatch_and_receipt_use_bound_date(tmp_path):
    from tau2.data_model.message import ToolMessage, UserMessage
    from tau2.environment.tool import Tool
    from test_ledger import assistant_call
    from test_runtime_banking import assembly
    from test_runtime_banking import call as model_call

    def call_discoverable_agent_tool(agent_tool_name: str, arguments: str) -> str:
        """Dispatch a discovered bank operation.

        Args:
            agent_tool_name: Operation name.
            arguments: Operation arguments encoded as JSON.
        """
        raise AssertionError("Only the external bank executes this operation")

    class AllowGate:
        async def review(self, calls):
            return tuple(
                SimpleNamespace(
                    tool_call_id=c.tool_call_id, allowed=True, reason=None, diagnostic_code=None
                )
                for c in calls
            )

    args = call()
    agent, _, captured = assembly(
        tmp_path,
        [model_call("call_discoverable_agent_tool", args, "dispute")],
        controller_mode="advisory",
        advice_mode="lean",
        planning_mode="outcomes",
        host_tools=[Tool(call_discoverable_agent_tool)],
    )
    state = agent.get_init_state()
    try:
        bank = captured["state"]
        # Isolate binding/dispatch from independently tested banking eligibility gates.
        bank.batch_gate = bank.memory_gate = bank.gate = AllowGate()
        bank.ledger.observe_assistant(
            assistant_call("lookup", "get_credit_card_transactions_by_user", user_id="u1")
        )
        bank.ledger.observe_input(
            ToolMessage(
                id="lookup",
                role="tool",
                requestor="assistant",
                content=(
                    "Found 1 record(s) in 'credit_card_transaction_history':\n\n"
                    "1. Record ID: t1\n"
                    "   transaction_id: t1\n"
                    "   user_id: u1\n"
                    "   credit_card_type: Silver Rewards Card\n"
                    "   merchant_name: Merchant\n"
                    "   transaction_amount: $12.34\n"
                    "   transaction_date: 02/17/2025\n"
                    "   category: Shopping\n"
                    "   status: COMPLETED\n"
                    "   rewards_earned: 0 points"
                ),
            )
        )
        reply, _ = agent.generate_next_message(UserMessage.text("File the agreed dispute."), state)
        assert len(reply.tool_calls) == 1
        dispatched = reply.tool_calls[0].arguments
        assert json.loads(dispatched["arguments"]) == {
            **json.loads(args["arguments"]),
            "purchase_date": "02/17/2025",
        }
        action = next(a for a in bank.ledger.actions if a.call_id == "dispute")
        assert action.arguments == dispatched
    finally:
        state.stop()
