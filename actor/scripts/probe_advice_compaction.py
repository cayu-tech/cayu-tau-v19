"""Frozen multi-turn advice comparison with genuine provider-backed compaction.

Local diagnostic only. Reuses both advice assemblies without changing their prompts,
tool policies or production defaults. Expected answers never enter actor state.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from cayu import RetryPolicy, SQLiteSessionStore
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from probe_advice_paths import ARMS, MODEL, build_minimal_resources, digest, source_hashes, verify
from tau2.data_model.message import UserMessage

from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.navigation_probe import POLICY
from cayu_tau3_banking.provenance import CAYU_COMMIT, assert_source_provenance
from cayu_tau3_banking.provider_diagnostics import DiagnosticOpenAITransport
from cayu_tau3_banking.runtime_agent import CayuRuntimeAgent
from cayu_tau3_banking.runtime_banking import build_banking_runtime_resources
from cayu_tau3_banking.runtime_profile import profile_session

TIMEOUT = 600
MAX_STEPS = 32
MAX_CALLS = 48
# Same generic preservation instructions as the production assembly, not answer hints.
SUMMARY_INSTRUCTIONS = (
    "Summarize prior banking dialogue compactly. Preserve customer constraints, "
    "consent and refusals, pending handoffs, exact tool result IDs and errors. "
    "Separate observed facts from proposals. Do not invent successful actions. "
    "Preserve known prerequisites shared across pending outcomes, including "
    "capabilities or eligibility an irreversible action would remove. Label "
    "unverified proposed ordering as tentative, not a required procedure. "
    "The fresh application evidence projection remains authoritative; this "
    "summary cannot grant authorization or prove completion. Return only the "
    "summary, aiming for at most 8000 characters. Do not call tools."
)


def catalogue():
    documents = []
    for name, rate, fee, minimum in (
        ("Aspen", "3.6", "0", "100"),
        ("Beacon", "4.4", "60", "100"),
        ("Clover", "3.0", "0", "100"),
        ("Delta", "5.0", "0", "50000"),
        ("Elm", "5.8", "0", "100"),
        ("Fjord", "4.0", "0", "100"),
        ("Grove", "4.45", "80", "100"),
        ("Heath", "4.1", "40", "100"),
        ("Indigo", "3.3", "0", "100"),
        ("Juniper", "4.2", "65", "100"),
        ("Kestrel", "3.8", "0", "100"),
        ("Linden", "4.25", "50", "100"),
    ):
        bonus = (
            "Clover Savings adds 1.8 percentage points to its APY only while an Aurora Card "
            "is held and at least $500 is spent on that card every month. Below that spend "
            "there is no bonus. Only Clover receives this bonus."
            if name == "Clover"
            else "No companion bonus applies."
        )
        donation = (
            "Exactly 20% of all earned interest is donated automatically, with no opt-out."
            if name == "Elm"
            else "All earned interest is retained; there is no donation."
        )
        prefix = f"doc_savings_accounts_{name.lower()}"
        documents.extend(
            [
                (
                    prefix + "_001",
                    f"{name} Savings rate and benefits",
                    f"{name} Savings base APY is {rate}%. {bonus} {donation}",
                ),
                (
                    prefix + "_002",
                    f"{name} Savings fees and eligibility",
                    f"{name} Savings annual fee is ${fee}. Opening and maintaining the account "
                    f"requires a balance of at least ${minimum}. No fee waiver or other costs apply.",
                ),
            ]
        )
    for name, fee in (("Aurora", "40"), ("Cirrus", "0"), ("Orbit", "60"), ("Vela", "20")):
        prefix = f"doc_credit_cards_{name.lower()}"
        benefit = (
            "Holding Aurora can qualify Clover Savings for its conditional companion bonus; "
            "see Clover's rate and benefits document for the spending condition."
            if name == "Aurora"
            else "No companion savings bonus applies."
        )
        documents.extend(
            [
                (
                    prefix + "_001",
                    f"{name} Card fees and eligibility",
                    f"{name} Card annual fee is ${fee}. No fee waiver applies. "
                    "For this fictional comparison the customer meets all card eligibility requirements.",
                ),
                (
                    prefix + "_002",
                    f"{name} Card benefits",
                    f"{name} Card offers no cashback or other card rewards. {benefit} No other costs apply.",
                ),
            ]
        )
    return documents


def cases():
    scope = (
        "Compare the entire published catalogue of savings accounts, with or without a card. "
        "I hold none of these products. Choose the eligible option with the highest annual "
        "interest I retain after all account/card fees and any mandatory donations. "
        "Show the decisive comparisons and conditions. Assume a constant balance for one year. "
        "Advice only; do not open anything. "
    )
    return [
        {
            "id": "changing_eligibility",
            "customer": scope
            + "My balance is $10000. I can spend $600 every month on a new card if useful. "
            "I will not accept any product that donates my earned interest.",
            "followups": [
                "Actually my monthly card spending will be only $300. Everything else stays the same. "
                "Recalculate the best option and its annual net.",
                "Now increase my savings balance to $50000; keep my other requirements from the "
                "previous turns unchanged. Recalculate the best option and annual net.",
            ],
        },
        {
            "id": "changing_preferences",
            "customer": scope
            + "My balance is $20000. I will not get any card. Mandatory donations are acceptable "
            "as long as you subtract them before comparing what I keep.",
            "followups": [
                "I have changed my mind about donations: I will not accept any product that "
                "donates my interest. Everything else stays the same. Recalculate the best option.",
                "I am now willing to get a card and can spend $600 on it every month. "
                "Keep all my other requirements unchanged. Recalculate the best option and annual net.",
            ],
        },
    ]


def expectations():
    return {
        "changing_eligibility": [
            {
                "winner": "Clover + Aurora",
                "net": 440,
                "checks": "4.8% with qualifying spend, subtract $40 card fee; Delta ineligible; Elm prohibited.",
            },
            {
                "winner": "Fjord (no paid card)",
                "net": 400,
                "checks": "$300 spend cannot earn Clover bonus; preserve donation restriction and $10000 balance.",
            },
            {
                "winner": "Delta (no paid card)",
                "net": 2500,
                "checks": "Exactly $50000 meets Delta minimum; preserve $300 spend and donation restriction.",
            },
        ],
        "changing_preferences": [
            {
                "winner": "Elm (no card)",
                "net": 928,
                "checks": "$1160 gross minus $232 donation; no cards; Delta ineligible.",
            },
            {
                "winner": "Beacon (no card)",
                "net": 820,
                "checks": "$880 minus $60; Elm now prohibited; card prohibition and $20000 balance persist.",
            },
            {
                "winner": "Clover + Aurora",
                "net": 920,
                "checks": "$960 minus $40; spend qualifies; donation prohibition and $20000 balance persist.",
            },
        ],
    }


def compaction_metrics(path):
    """Count native compactor completions separately from acting-model completions."""
    calls = {"actor": [], "compactor": []}
    compactions = []
    started = None
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT event_type,timestamp,payload_json FROM cayu_events ORDER BY sequence"
        )
        for kind, timestamp, raw in rows:
            payload = json.loads(raw)
            if kind == "model.completed":
                purpose = "compactor" if payload.get("purpose") == "context_compaction" else "actor"
                calls[purpose].append(
                    {
                        "usage": payload.get("usage"),
                        "model": payload.get("model"),
                        "outcome": payload.get("compaction_outcome"),
                    }
                )
            elif kind == "context.compaction.started":
                started = datetime.fromisoformat(timestamp)
            elif kind in {"context.compaction.completed", "context.compaction.failed"}:
                compactions.append(
                    {
                        "event": kind,
                        "payload": payload,
                        "seconds": (datetime.fromisoformat(timestamp) - started).total_seconds()
                        if started
                        else None,
                    }
                )
                started = None
    return {"calls": calls, "compactions": compactions}


def run_case(
    output,
    case,
    *,
    arm,
    corpus_directory,
    policy,
    provider_factory,
    model=MODEL,
    timeout=TIMEOUT,
    max_steps=MAX_STEPS,
    max_calls=MAX_CALLS,
    compactor_provider_factory=None,
    capture_transport=False,
):
    if arm not in ARMS or set(case) != {"id", "customer", "followups"}:
        raise ValueError("Only frozen actor-visible case fields are accepted")
    if not isinstance(case["followups"], list) or not all(
        isinstance(t, str) and t.strip() for t in [case["customer"], *case["followups"]]
    ):
        raise ValueError("Customer turns must be nonempty text")
    captured = {}

    async def factory(broker, specs):
        store = SQLiteSessionStore(output / "runtime.sqlite3")
        providers = []

        async def close():
            try:
                for provider in reversed(providers):
                    if getattr(provider, "aclose", None):
                        await provider.aclose()
            finally:
                await store.close()

        try:
            provider = provider_factory()
            providers.append(provider)
            if capture_transport:
                provider.transport = DiagnosticOpenAITransport(
                    provider.transport, output / "transport"
                )
            summary_provider = provider
            if compactor_provider_factory is not None:
                summary_provider = compactor_provider_factory()
                if summary_provider is not provider:
                    providers.append(summary_provider)
            compaction = CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(
                    provider=summary_provider,
                    model=model,
                    system_prompt=SUMMARY_INSTRUCTIONS,
                    options={"thinking": {"enabled": True, "effort": "low"}},
                    max_input_chars=120_000,
                    max_hierarchy_calls=8,
                    retry_policy=RetryPolicy(max_attempts=2, max_unknown_attempts=1),
                    retry_empty_summaries=True,
                ),
                compact_after_messages=2,
                max_user_turns=1,
            )
            common = dict(
                provider=provider,
                model=model,
                domain_policy=policy,
                memory_runtime=BankingMemoryRuntime(
                    BankingKnowledgeCorpus.from_directory(corpus_directory)
                ),
                session_store=store,
                aclose=close,
                compaction=compaction,
                context_capture_directory=output / "contexts",
                provider_options={
                    "thinking": {"enabled": True, "effort": "xhigh"},
                    "openai": {"parallel_tool_calls": False},
                },
            )
            resources = (
                build_banking_runtime_resources(
                    broker, specs, **common, native_recall=True, planning_mode="outcomes"
                )
                if arm == "controller"
                else build_minimal_resources(broker, specs, **common)
            )
            captured["state"] = resources.observer
            return resources
        except BaseException:
            await close()
            raise

    agent = CayuRuntimeAgent(
        tools=[],
        domain_policy=policy,
        resource_factory=factory,
        max_steps=max_steps,
        max_tool_calls=max_calls,
        callback_timeout_s=timeout + 10,
        conversation_timeout_s=timeout,
        diagnostic_stop_on_tool_error=False,
    )
    state = None
    started = time.monotonic()
    transcript, turns = [], []
    result = {"case": case["id"], "arm": arm, "turns": turns, "bank_tools_executed": 0}
    try:
        state = agent.get_init_state()
        for index, content in enumerate([case["customer"], *case["followups"]]):
            turn_started = time.monotonic()
            customer = UserMessage.text(content)
            transcript.append(customer.model_dump(mode="json"))
            reply, _ = agent.generate_next_message(customer, state)
            transcript.append(reply.model_dump(mode="json"))
            if reply.tool_calls or not isinstance(reply.content, str) or not reply.content.strip():
                raise RuntimeError("Expected nonempty final advice, not an external action")
            turn = {
                "index": index,
                "reply": reply.content,
                "seconds": time.monotonic() - turn_started,
                "plan": captured["state"].controller.export_state(),
            }
            turns.append(turn)
            save_private_json(output / f"turn-{index}.json", turn)
            print(
                json.dumps(
                    {
                        "case": case["id"],
                        "arm": arm,
                        "turn": index,
                        "status": "answered",
                        "seconds": turn["seconds"],
                    }
                ),
                flush=True,
            )
        result["status"] = "answered"
    except Exception as error:
        result.update(
            status="error",
            error_type=type(error).__name__,
            failure_type=getattr(error, "failure_type", None),
        )
    finally:
        try:
            if state is not None:
                state.stop()
        except Exception as error:
            result.update(status="error", cleanup_error_type=type(error).__name__)
        finally:
            result["seconds"] = time.monotonic() - started
            save_private_json(output / "transcript.json", transcript)
            save_private_json(output / "result.json", result)
    try:
        save_private_json(output / "profile.json", profile_session(output / "runtime.sqlite3"))
        save_private_json(
            output / "compaction-metrics.json", compaction_metrics(output / "runtime.sqlite3")
        )
    except Exception as error:
        save_private_json(output / "diagnostic-error.json", {"error_type": type(error).__name__})
        result["status"] = "error"
        result["diagnostic_error_type"] = type(error).__name__
        save_private_json(output / "diagnostic-result.json", result)
    return result


def prepare(parent):
    root = Path(tempfile.mkdtemp(prefix="advice-compaction-", dir=parent)).resolve()
    corpus = root / "corpus"
    corpus.mkdir(mode=0o700)
    for identifier, title, content in catalogue():
        save_private_json(
            corpus / f"{identifier}.json", dict(id=identifier, title=title, content=content)
        )
    save_private_json(root / "cases.json", cases())
    save_private_json(root / "expectations-evaluator-only.json", expectations())
    sources = source_hashes()
    sources[str(Path(__file__).resolve())] = digest(Path(__file__))
    save_private_json(
        root / "protocol.json",
        {
            "framework_commit": CAYU_COMMIT,
            "model": MODEL,
            "actor_effort": "xhigh",
            "compactor_effort": "low",
            "summary_instructions": SUMMARY_INSTRUCTIONS,
            "policy": POLICY,
            "arms": list(ARMS),
            "attempts_per_arm_per_case": 1,
            "conversation_timeout_seconds": TIMEOUT,
            "max_model_steps_per_turn": MAX_STEPS,
            "max_tool_calls_per_session": MAX_CALLS,
            "max_concurrent_runs": 2,
            "compaction": {"compact_after_messages": 2, "max_user_turns": 1},
            "source_hashes": sources,
            "input_hashes": {
                str(p.relative_to(root)): digest(p)
                for p in [
                    root / "cases.json",
                    root / "expectations-evaluator-only.json",
                    *sorted(corpus.glob("*.json")),
                ]
            },
            "scope": "32 fictional documents: 12 savings accounts and four cards, each with two documents. "
            "Two three-turn conversations, both paths. Same persistent runtime state throughout each conversation. "
            "No external bank actions, user simulator or benchmark submission.",
            "assessment": "Review every final reply against separate expectations; inspect actual composed contexts, "
            "prepared summary inputs, summaries, tool results and event telemetry. Do not count an answer as correct "
            "just because execution finished. Require real completed compaction before interpreting retention results.",
            "limits": "Deliberately frequent real compaction at customer-turn boundaries, not production thresholds. "
            "One sample per path/case, no no-compaction control, no held-out or tau score. Same lexical retrieval as "
            "previous advice panel, no embeddings. Shared model, corpus and public policy, but paths differ in prompts, "
            "planning, tools and context. A difference cannot identify one component's causal effect. "
            "Later case pairs stop on runtime failures; no adaptive retries or answer-specific tuning.",
        },
    )
    return root


def run(root):
    from cayu_tau3_banking.cli import build_provider

    assert_source_provenance(Path(__file__).resolve().parents[1] / ".vendor/tau2")
    protocol = verify(root)
    frozen = json.loads((root / "cases.json").read_text())
    save_private_json(root / "started.json", {"time_unix": time.time(), "scheduled_runs": 4})
    completed = []
    for index, case in enumerate(frozen):
        verify(root)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for arm in ARMS if index % 2 == 0 else reversed(ARMS):
                output = root / f"{case['id']}-{arm}"
                output.mkdir(mode=0o700)
                futures.append(
                    executor.submit(
                        run_case,
                        output,
                        case,
                        arm=arm,
                        corpus_directory=root / "corpus",
                        policy=protocol["policy"],
                        capture_transport=True,
                        provider_factory=lambda: build_provider(
                            provider="subscription",
                            codex_lb_base_url="",
                            semantic_progress_timeout_seconds=180,
                        ),
                    )
                )
            results = [future.result() for future in futures]
        completed.extend(results)
        if any(r["status"] == "error" for r in results):
            save_private_json(
                root / "stopped.json",
                {
                    "runs": completed,
                    "reason": "Runtime or diagnostic failure; later pairs not launched",
                },
            )
            return
    verify(root)
    save_private_json(
        root / "completed.json", {"runs_finished": len(completed), "bank_tools_executed": 0}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("root", type=Path, nargs="?", default=Path("artifacts"))
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.root))
    else:
        run(args.root.resolve())


if __name__ == "__main__":
    main()
