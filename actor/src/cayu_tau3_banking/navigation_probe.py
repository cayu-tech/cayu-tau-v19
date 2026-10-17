"""Fixed synthetic navigation evaluation; never a tau score or actor-side oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

from cayu import SQLiteSessionStore
from tau2.data_model.message import UserMessage

from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.provenance import CAYU_COMMIT, assert_source_provenance
from cayu_tau3_banking.runtime_agent import CayuRuntimeAgent
from cayu_tau3_banking.runtime_banking import build_banking_runtime_resources
from cayu_tau3_banking.runtime_profile import profile_session

POLICY = """Fictional public product comparison service. Public product information
requires no identity verification. This is advice only: no application, account
creation or other customer operation is available. All current public terms are
in the knowledge corpus. Missing terms are unknown, not zero. Product names with
suffixes are different products. Apply eligibility and bonus conditions to the
customer's stated situation. Do not request sensitive personal information.
For this comparison use balance times stated APY, without intra-year compounding,
minus annual fees; assume rates remain fixed for one year. There are no taxes or
other costs besides documented fees. Card rewards are zero. Do not invent rates
or fees. Explain any uncertainty that prevents a complete comparison.
"""

DOCUMENTS = (
    (
        "doc_savings_accounts_meadow_001",
        "Meadow savings base terms",
        "Meadow Savings has base APY 3.1%. Annual account fee is $0. Minimum balance is $100.",
    ),
    (
        "doc_savings_accounts_meadow_002",
        "Meadow savings companion bonus",
        "Meadow Savings adds 1.7 percentage points of APY when the customer holds a "
        "Lark Card and spends at least $500 on that card every month. Below that monthly "
        "spend there is no companion bonus. This bonus applies only to Meadow Savings.",
    ),
    (
        "doc_savings_accounts_meadow_plus_001",
        "Meadow Plus savings",
        "Meadow Plus is a separate product. APY is 5.6%. Annual fee is $0. "
        "Opening and maintaining this account requires a balance of at least $50000. "
        "No companion bonus applies.",
    ),
    (
        "doc_savings_accounts_quartz_001",
        "Quartz savings",
        "Quartz Savings APY is 3.9%. Annual account fee is $0. Minimum balance is $100. "
        "No card is needed and no companion bonus applies.",
    ),
    (
        "doc_credit_cards_lark_001",
        "Lark card terms",
        "Lark Card annual fee is $70. It offers no cashback or other rewards. "
        "It can qualify Meadow Savings for a companion bonus under the savings terms.",
    ),
    (
        "doc_savings_accounts_tide_001",
        "Tide savings incomplete publication",
        "Tide Savings APY is 4.3%. Minimum balance is $100. "
        "Its annual account fee has not been published in this catalogue. "
        "No companion bonus applies.",
    ),
)

# Expectations never enter domain policy, documents, or customer messages.
CASES = (
    {
        "id": "eligible_bundle",
        "customer": "Compare Meadow Savings with a new Lark Card, Meadow Plus, and Quartz Savings. "
        "I have $10000 to keep deposited for a year and can spend $600 every month on Lark. "
        "Which eligible option gives the highest annual net? Include rates, fees and calculations. "
        "I only want advice, not an application.",
        "expected": "Meadow+Lark: 4.8%, $480 interest minus $70 = $410. "
        "Quartz $390. Meadow Plus ineligible ($50000 minimum). Choose bundle.",
    },
    {
        "id": "bonus_ineligible",
        "customer": "Compare Meadow Savings with a new Lark Card, Meadow Plus, and Quartz Savings. "
        "I have $10000 to keep deposited for a year and can spend only $450 every month on Lark. "
        "Which eligible option gives the highest annual net? Include rates, fees and calculations. "
        "I only want advice, not an application.",
        "expected": "Bonus unavailable. Meadow+Lark: 3.1%, $310 minus $70 = $240. "
        "Quartz $390. Meadow Plus ineligible. Choose Quartz.",
    },
    {
        "id": "unknown_fee",
        "customer": "Compare Tide Savings and Quartz Savings for $10000 held for a year. "
        "What is each annual net and which is better? I need you to flag any missing "
        "terms instead of assuming them. I only want advice.",
        "expected": "Tide $430 gross, fee unknown, net $430 minus unknown fee. "
        "Quartz $390 net. Cannot determine unconditional winner. "
        "Tide wins if fee below $40, ties at $40, loses above $40.",
    },
)


def run_case(
    output: Path,
    case: dict,
    *,
    provider_factory,
    model="gpt-5.6-sol",
    documents=DOCUMENTS,
    conversation_timeout_s=600,
    policy=POLICY,
    compaction=None,
    memory_runtime_factory=BankingMemoryRuntime,
    advice_mode="controller",
    controller_mode="enforced",
) -> dict:
    corpus_dir = output / "corpus"
    corpus_dir.mkdir()
    for identifier, title, content in documents:
        save_private_json(
            corpus_dir / f"{identifier}.json",
            {"id": identifier, "title": title, "content": content},
        )
    corpus = BankingKnowledgeCorpus.from_directory(corpus_dir)
    captured = {}

    def factory(broker, specs):
        provider = provider_factory()
        store = SQLiteSessionStore(output / "runtime.sqlite3")

        async def close():
            try:
                if getattr(provider, "aclose", None):
                    await provider.aclose()
            finally:
                await store.close()

        resources = build_banking_runtime_resources(
            broker,
            specs,
            provider=provider,
            model=model,
            domain_policy=policy,
            memory_runtime=memory_runtime_factory(corpus),
            session_store=store,
            aclose=close,
            native_recall=True,
            planning_mode="outcomes",
            advice_mode=advice_mode,
            controller_mode=controller_mode,
            compaction=compaction,
            context_capture_directory=output / "contexts",
            provider_options={
                "thinking": {"enabled": True, "effort": "xhigh"},
                "openai": {"parallel_tool_calls": False},
            },
        )
        captured["bank"] = resources.observer
        return resources

    agent = CayuRuntimeAgent(
        tools=[],
        domain_policy=policy,
        resource_factory=factory,
        max_steps=24,
        max_tool_calls=64,
        callback_timeout_s=600,
        conversation_timeout_s=conversation_timeout_s,
        diagnostic_stop_on_tool_error=False,
    )
    state = None
    started = time.monotonic()
    transcript = []
    turns = []
    try:
        state = agent.get_init_state()
        for customer in (case["customer"], *case.get("followups", ())):
            transcript.append(UserMessage.text(customer).model_dump(mode="json"))
            reply, _ = agent.generate_next_message(UserMessage.text(customer), state)
            transcript.append(reply.model_dump(mode="json"))
            if reply.tool_calls:
                raise RuntimeError(
                    "Advice-only evaluation unexpectedly requested an external action"
                )
            turns.append(
                {"reply": reply.content, "plan": captured["bank"].controller.export_state()}
            )
        result = {
            "case": case["id"],
            "seconds": time.monotonic() - started,
            "reply": reply.content,
            "turns": turns,
            "plan": captured["bank"].controller.export_state(),
            "assessment": "Unscored: inspect answer and persisted tool evidence against frozen expectations.",
        }
        save_private_json(output / "result.json", result)
        return result
    except Exception as error:
        save_private_json(
            output / "error.json",
            {"error_type": type(error).__name__, "seconds": time.monotonic() - started},
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
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    assert_source_provenance(repo / ".vendor" / "tau2")
    args.output_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="navigation-fixed-", dir=args.output_parent))
    # Freeze all actor sources before the first dispatch, not just this probe.
    shutil.copytree(repo / "src", root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(repo / "pyproject.toml", root / "pyproject.toml")
    manifest = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "source").rglob("*.py"))
    }
    save_private_json(
        root / "protocol.json",
        {
            "cases": CASES,
            "documents": DOCUMENTS,
            "policy": POLICY,
            "cayu_commit": CAYU_COMMIT,
            "model": "gpt-5.6-sol",
            "provider": "subscription",
            "effort": "xhigh",
            "attempts_per_case": 1,
            "callback_timeout_s": 600,
            "conversation_timeout_s": 600,
            "max_steps": 24,
            "max_tool_calls": 64,
            "native_recall": True,
            "retrieval": "lexical, six-document synthetic corpus",
            "reviewers": False,
            "simulator": False,
            "source_hashes": manifest,
            "limits": "No control arm or tau score. Manual answer review required. "
            "Stop subsequent launches on infrastructure errors; no adaptive retries.",
        },
    )
    print(f"Output: {root}", flush=True)
    for case in CASES:
        output = root / case["id"]
        output.mkdir(mode=0o700)
        result = run_case(
            output,
            case,
            provider_factory=lambda: build_provider(
                provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=120
            ),
        )
        profile = profile_session(output / "runtime.sqlite3")
        save_private_json(output / "profile.json", profile)
        print(
            json.dumps(
                {
                    "case": case["id"],
                    "seconds": result["seconds"],
                    "events": profile["event_counts"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
