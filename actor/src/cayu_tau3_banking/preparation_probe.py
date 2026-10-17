"""Opt-in live, tool-executing synthetic probe; never a tau score.

Uses the production outcome agent and native Cayu runtime unchanged. Expectations
are evaluated outside the actor context. Every case gets fresh state and tools.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from cayu import SQLiteSessionStore
from tau2.data_model.message import MultiToolMessage, ToolMessage, UserMessage
from tau2.environment.tool import Tool

from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.runtime_agent import CayuRuntimeAgent
from cayu_tau3_banking.runtime_banking import build_banking_runtime_resources

POLICY = """Fictional document-delivery service policy, revision 1.
The customer is authenticated and authorized for delivery record D17. Identity and
access checks are complete; do not request bank account or card details. There is
no fee waiver or alternate fulfillment method. The customer's request authorizes
dispatch once prerequisites are satisfied.
Use read_delivery_record to obtain the current saved address and payment status.
An address is eligible for validate_delivery_address only when the record contains
a complete address. The validation tool records the outcome of address validation.
dispatch_documents requires both successful address validation and paid postage.
Only the customer can supply a missing address or pay postage. Ask for what is
missing; after the customer reports a change, read the record to verify it. Never
claim dispatch without a successful dispatch_documents result.
All executable service tools are already exposed. They do not require discovery.
"""


def read_delivery_record(record_id: str) -> str:
    """Read the saved delivery address, postage status and validation status.

    Args:
        record_id: Delivery record identifier.
    """
    raise AssertionError("Only the synthetic host executes service tools")


def validate_delivery_address(record_id: str) -> str:
    """Validate the record's complete saved address. Does not dispatch documents.

    Args:
        record_id: Delivery record identifier.
    """
    raise AssertionError("Only the synthetic host executes service tools")


def dispatch_documents(record_id: str) -> str:
    """Dispatch documents, requiring a validated address and paid postage.

    Args:
        record_id: Delivery record identifier.
    """
    raise AssertionError("Only the synthetic host executes service tools")


def verify_delivery_access(access_code: str) -> str:
    """Verify the customer's supplied delivery access code before record access.

    Args:
        access_code: Access code supplied by the customer.
    """
    raise AssertionError("Only the synthetic host executes service tools")


class SyntheticDelivery:
    def __init__(self, *, address_present: bool, require_access: bool = False):
        self.address = "12 Example Lane, Example City, EX 12345" if address_present else None
        self.paid = False
        self.validated = False
        self.dispatched = False
        self.calls: list[dict] = []
        self.access_granted = not require_access

    def execute(self, name: str, arguments: dict) -> dict:
        self.calls.append({"name": name, "arguments": arguments})
        if name == "verify_delivery_access":
            if arguments != {"access_code": "D17-ACCESS"}:
                raise ValueError("Invalid delivery access code")
            self.access_granted = True
            return {"record_id": "D17", "access_granted": True}
        if not self.access_granted:
            raise ValueError("Record access requires successful code verification")
        if arguments != {"record_id": "D17"}:
            raise ValueError("Synthetic tool received invalid record arguments")
        if name == "read_delivery_record":
            return {
                "record_id": "D17",
                "address": self.address,
                "postage_paid": self.paid,
                "address_validated": self.validated,
                "dispatched": self.dispatched,
            }
        if name == "validate_delivery_address":
            if not self.address:
                raise ValueError("Actor tried to validate a missing address")
            self.validated = True
            return {"record_id": "D17", "address_validated": True}
        if name == "dispatch_documents":
            if not self.paid or not self.validated or self.dispatched:
                raise ValueError("Actor tried unauthorized or duplicate dispatch")
            self.dispatched = True
            return {"record_id": "D17", "dispatched": True}
        raise ValueError("Unknown synthetic tool")

    def customer_update(self):
        self.address = "12 Example Lane, Example City, EX 12345"
        self.paid = True


def run_case(
    output: Path,
    *,
    address_present: bool,
    provider_factory,
    model="gpt-5.6-sol",
    require_access: bool = False,
    parallel_tools: bool = False,
):
    policy = POLICY
    if require_access:
        policy = policy.replace(
            "The customer is authenticated and authorized for delivery record D17. Identity and\n"
            "access checks are complete; do not request bank account or card details.",
            "Access to delivery record D17 is not authorized yet. Ask the customer for the delivery\n"
            "access code and verify it using verify_delivery_access before calling any record tools.\n"
            "Do not request bank account or card details.",
        )
    corpus_dir = output / "corpus"
    corpus_dir.mkdir()
    save_private_json(
        corpus_dir / "delivery-policy.json",
        {
            "id": "delivery-policy",
            "title": "Document delivery procedure",
            "content": policy,
        },
    )
    corpus = BankingKnowledgeCorpus.from_directory(corpus_dir)
    context_dir = output / "contexts"
    context_dir.mkdir()
    host = SyntheticDelivery(address_present=address_present, require_access=require_access)
    captured = {}

    def factory(broker, specs):
        provider = provider_factory()
        store = SQLiteSessionStore(output / "runtime.sqlite3")

        async def close():
            try:
                close_provider = getattr(provider, "aclose", None)
                if close_provider is not None:
                    await close_provider()
            finally:
                await store.close()

        resources = build_banking_runtime_resources(
            broker,
            specs,
            provider=provider,
            model=model,
            domain_policy=policy,
            memory_runtime=BankingMemoryRuntime(corpus),
            session_store=store,
            aclose=close,
            native_recall=True,
            planning_mode="outcomes",
            context_capture_directory=context_dir,
            provider_options={
                "thinking": {"enabled": True, "effort": "xhigh"},
                "openai": {"parallel_tool_calls": parallel_tools},
            },
        )
        captured["bank"] = resources.observer
        return resources

    agent = CayuRuntimeAgent(
        tools=[
            Tool(read_delivery_record),
            Tool(validate_delivery_address),
            Tool(dispatch_documents),
        ]
        + ([Tool(verify_delivery_access)] if require_access else []),
        domain_policy=policy,
        resource_factory=factory,
        max_steps=24,
        max_tool_calls=64,
        callback_timeout_s=240,
        conversation_timeout_s=480,
        diagnostic_stop_on_tool_error=True,
    )
    state = None
    transcript: list[dict] = []
    started = time.monotonic()

    def turn(customer: str):
        message = UserMessage.text(customer)
        transcript.append(message.model_dump(mode="json"))
        while True:
            reply, _ = agent.generate_next_message(message, state)
            transcript.append(reply.model_dump(mode="json"))
            if not reply.tool_calls:
                return reply.content
            results = []
            for call in reply.tool_calls:
                result = ToolMessage(
                    role="tool",
                    id=call.id,
                    content=json.dumps(host.execute(call.name, call.arguments)),
                )
                transcript.append(result.model_dump(mode="json"))
                results.append(result)
            message = (
                results[0]
                if len(results) == 1
                else MultiToolMessage(role="tool", tool_messages=results)
            )

    try:
        state = agent.get_init_state()
        first = turn(
            "Please send the documents for delivery record D17. I have not paid postage yet."
        )
        first_calls = list(host.calls)
        first_plan = captured["bank"].controller.export_state()
        first_ok = (
            host.validated == address_present
            and not host.dispatched
            and any(c["name"] == "read_delivery_record" for c in first_calls)
        )
        host.customer_update()
        if require_access:
            first_ok = not first_calls and not host.validated and not host.dispatched
        final = turn(
            "I have now paid the postage and saved my complete delivery address. Please proceed."
            + (" My delivery access code is D17-ACCESS." if require_access else "")
        )
        result = {
            "scope": "synthetic native-runtime tool execution, not tau",
            "address_present_initially": address_present,
            "access_required_initially": require_access,
            "first_phase_passed": first_ok,
            "final_dispatch_passed": host.dispatched,
            "passed": first_ok and host.dispatched,
            "first_reply": first,
            "final_reply": final,
            "first_phase_calls": first_calls,
            "first_plan": first_plan,
            "final_plan": captured["bank"].controller.export_state(),
            "calls": host.calls,
            "seconds": time.monotonic() - started,
            "context_snapshots": len(list(context_dir.rglob("*.json"))),
        }
        save_private_json(output / "result.json", result)
        return result
    except Exception as error:
        save_private_json(
            output / "error.json",
            {
                "error_type": type(error).__name__,
                "seconds": time.monotonic() - started,
                "calls": host.calls,
            },
        )
        raise
    finally:
        try:
            if state is not None:
                state.stop()
        finally:
            save_private_json(output / "transcript.json", transcript)


def main():
    from cayu_tau3_banking.cli import build_provider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, default=Path("artifacts"))
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    parser.add_argument("--access-only", action="store_true")
    parser.add_argument("--parallel-tools", action="store_true")
    args = parser.parse_args()
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="preparation-probe-", dir=args.output_parent))
    save_private_json(
        output / "manifest.json",
        {
            "model": "gpt-5.6-sol",
            "provider": "subscription",
            "reasoning": "xhigh",
            "repeats": args.repeats,
            "access_only": args.access_only,
            "parallel_tools": args.parallel_tools,
            "case_timeout_seconds": 480,
            "max_steps_per_turn": 24,
            "policy": POLICY,
            "retrieval": "lexical single synthetic document",
            "reviewers": False,
            "expectations": "Validate before waiting iff address exists; dispatch only after both prerequisites.",
        },
    )
    print(f"Output: {output}", flush=True)
    for repeat in range(args.repeats):
        conditions = (
            (True,) if args.access_only else ((True, False) if repeat % 2 == 0 else (False, True))
        )
        for present in conditions:
            label = (
                "access-required" if args.access_only else ("independent" if present else "blocked")
            )
            case_dir = output / f"{repeat}-{label}"
            case_dir.mkdir(mode=0o700)
            try:
                result = run_case(
                    case_dir,
                    address_present=present,
                    require_access=args.access_only,
                    parallel_tools=args.parallel_tools,
                    provider_factory=lambda: build_provider(
                        provider="subscription",
                        codex_lb_base_url="",
                        semantic_progress_timeout_seconds=120,
                    ),
                )
            except Exception as error:
                print(
                    json.dumps(
                        {"stopped": True, "case": case_dir.name, "error_type": type(error).__name__}
                    ),
                    flush=True,
                )
                raise SystemExit(1) from None
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "address_present_initially",
                            "passed",
                            "seconds",
                            "context_snapshots",
                        )
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
