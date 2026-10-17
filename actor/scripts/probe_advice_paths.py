"""Paired advice-only evaluations through Cayu's real runtime to final replies.

Diagnostic assembly only: no production defaults change. No provider replay,
fake tool acknowledgements, customer simulator or external bank tools are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cayu import (
    AgentSpec,
    BeforeStopDecision,
    CayuApp,
    Environment,
    EnvironmentSpec,
    LoopPolicy,
    RequestFootprintConfig,
    SQLiteSessionStore,
)
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from cayu.tools.base import ToolSpec
from cayu.tools.exposure import ToolExposureDecision, ToolExposurePolicy
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyResult
from tau2.data_model.message import UserMessage

from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.context_capture import CapturingContextPolicy
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.knowledge import KNOWLEDGE_SEARCH_TOOL, BankingKnowledgeCorpus
from cayu_tau3_banking.navigation_probe import CASES, DOCUMENTS, POLICY
from cayu_tau3_banking.product_navigation import BROWSE_TOOL, READ_TOOL
from cayu_tau3_banking.provenance import CAYU_COMMIT, assert_source_provenance
from cayu_tau3_banking.runtime_agent import CayuRuntimeAgent, RuntimeResources
from cayu_tau3_banking.runtime_banking import (
    BankingNativeTool,
    BankingRuntimeContext,
    BankingRuntimeSession,
    build_banking_runtime_resources,
)
from cayu_tau3_banking.runtime_profile import profile_session
from cayu_tau3_banking.runtime_recall import banking_recall_policy, observe_delivered_recall

ARMS = ("controller", "minimal")
READ_ONLY_TOOLS = frozenset({BROWSE_TOOL, READ_TOOL, KNOWLEDGE_SEARCH_TOOL})
MODEL = "gpt-5.6-sol"
TIMEOUT = 420
MAX_STEPS = 10
MAX_CALLS = 24
MINIMAL_INSTRUCTIONS = """You are a banking customer-service agent. Follow the supplied
public policy and use only the supplied tools. Ground the requested comparison in
source evidence and the customer's stated constraints. Distinguish known terms from
unknown conditions. Do not invent facts, consent or completed actions.
Return either a customer-facing answer or tool calls, not both in the same step.
Tools run before your next step. Once the evidence supports an answer, give it;
otherwise explain the material uncertainty or ask for the necessary missing input.
"""


def frozen_cases():
    cases = [dict(case) for case in CASES]
    cases.append(
        {
            "id": "interest_diversion_restricted",
            "customer": "Compare Cedar Savings and Pine Savings for $10000 held for a year. "
            "I want the highest net interest, but I will not accept any account that sends "
            "part of my earned interest to donations. Which should I choose, and what is "
            "the annual net? Explain the tradeoff. Advice only; do not open anything.",
            "expected": "Choose Pine: $420 net, no donation. Cedar grosses $480 and retains "
            "$432 after mandatory $48 diversion; despite greater retained interest it "
            "violates the explicit customer constraint and is not a suitable recommendation.",
        }
    )
    return cases


def frozen_documents():
    return [
        *DOCUMENTS,
        (
            "doc_savings_accounts_cedar_001",
            "Cedar Savings terms",
            "Cedar Savings APY is 4.8%. "
            "Annual account fee is $0. Minimum balance is $100. Exactly 10% of interest earned "
            "is automatically donated. There is no opt-out. No other bonuses apply.",
        ),
        (
            "doc_savings_accounts_pine_001",
            "Pine Savings terms",
            "Pine Savings APY is 4.2%. "
            "Annual account fee is $0. Minimum balance is $100. All interest is retained "
            "by the customer. There are no donations or companion bonuses.",
        ),
    ]


class AdviceExposure(ToolExposurePolicy):
    def __init__(self, state):
        self.state = state

    def select(self, request):
        return ToolExposureDecision(
            profile_id="diagnostic:read-only-advice:v1",
            tool_names=tuple(
                tool.name
                for tool in request.eligible_tools
                if tool.name in READ_ONLY_TOOLS & self.state.available_private_names
            ),
        )


class AdviceToolPolicy(ToolPolicy):
    def __init__(self, state):
        self.state = state

    async def authorize(self, request):
        await observe_delivered_recall(self.state)
        self.state.refresh_exposure()
        allowed = request.tool_name in READ_ONLY_TOOLS & self.state.available_private_names
        return ToolPolicyResult(
            decision=ToolPolicyDecision.ALLOW if allowed else ToolPolicyDecision.DENY,
            reason=None if allowed else "Only currently available public-source reads are allowed.",
        )


class AdviceReplyPolicy(LoopPolicy):
    def __init__(self, state):
        self.state = state

    async def before_stop(self, context):
        await observe_delivered_recall(self.state)
        return BeforeStopDecision.complete()


def build_minimal_resources(
    broker,
    specs,
    *,
    provider,
    model,
    domain_policy,
    memory_runtime,
    session_store,
    aclose,
    compaction,
    context_capture_directory,
    provider_options,
):
    if specs:
        raise ValueError("Diagnostic minimal advice path cannot register bank actions")
    state = BankingRuntimeSession(memory_runtime, (), planning_mode="outcomes")
    # Reuse knowledge/exposure bookkeeping, but no model-maintained work plan.
    state.memory = memory_runtime.new_session(
        ledger=state.ledger,
        controller=state.controller,
        admission_gate=state.admission,
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=False,
    )
    state.native_recall = True
    state.refresh_exposure()
    app = CayuApp(
        session_store=session_store,
        enable_logging=False,
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="advice-diagnostic", fingerprint_key=secrets.token_hex(32)
        ),
    )
    state.session_store = app.session_store
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="public-corpus"), knowledge_store=memory_runtime.corpus.store
        ),
        default=True,
    )
    context = CapturingContextPolicy(
        banking_recall_policy(
            memory_runtime.corpus, BankingRuntimeContext(state, compaction=compaction)
        ),
        context_capture_directory,
    )
    tools = [
        BankingNativeTool(
            ToolSpec(
                name=s["name"],
                description=s["description"],
                input_schema=s["input_schema"],
                parallel_safe=False,
            ),
            state,
        )
        for s in state.memory.tool_schemas()
        if s["name"] in READ_ONLY_TOOLS
    ]
    app.register_agent(
        AgentSpec(
            name="banking",
            model=model,
            system_prompt=MINIMAL_INSTRUCTIONS + "\n<policy>\n" + domain_policy + "\n</policy>",
            provider_options=provider_options,
        ),
        tools=tools,
        context_policy=context,
        tool_exposure_policy=AdviceExposure(state),
        tool_policy=AdviceToolPolicy(state),
        loop_policies=[AdviceReplyPolicy(state)],
    )
    return RuntimeResources(app=app, aclose=aclose, observer=state)


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
):
    if arm not in ARMS or set(case) != {"id", "customer"}:
        raise ValueError("Only frozen actor-visible case fields are accepted")
    captured = {}

    async def factory(broker, specs):
        provider = provider_factory()
        store = SQLiteSessionStore(output / "runtime.sqlite3")

        async def close():
            try:
                if getattr(provider, "aclose", None):
                    await provider.aclose()
            finally:
                await store.close()

        try:
            corpus = BankingKnowledgeCorpus.from_directory(corpus_directory)
            compaction = CheckpointCompactionContextPolicy(
                compactor=ModelCompactor(provider=provider, model=model),
                compact_after_messages=1000,
                max_user_turns=100,
            )
            common = dict(
                provider=provider,
                model=model,
                domain_policy=policy,
                memory_runtime=BankingMemoryRuntime(corpus),
                session_store=store,
                aclose=close,
                compaction=compaction,
                context_capture_directory=output / "contexts",
                provider_options={
                    "thinking": {"enabled": True, "effort": "xhigh"},
                    "openai": {"parallel_tool_calls": False},
                },
            )
            if arm == "controller":
                resources = build_banking_runtime_resources(
                    broker, specs, **common, native_recall=True, planning_mode="outcomes"
                )
            else:
                resources = build_minimal_resources(broker, specs, **common)
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
    transcript = []
    result = None
    try:
        state = agent.get_init_state()
        customer = UserMessage.text(case["customer"])
        transcript.append(customer.model_dump(mode="json"))
        reply, _ = agent.generate_next_message(customer, state)
        transcript.append(reply.model_dump(mode="json"))
        if reply.tool_calls or not isinstance(reply.content, str) or not reply.content.strip():
            raise RuntimeError("Expected a nonempty final advice reply, not an external action")
        result = {
            "case": case["id"],
            "arm": arm,
            "status": "answered",
            "reply": reply.content,
            "seconds": time.monotonic() - started,
            "plan": captured["state"].controller.export_state(),
            "bank_tools_executed": 0,
        }
    except Exception as error:
        result = {
            "case": case["id"],
            "arm": arm,
            "status": "error",
            "error_type": type(error).__name__,
            "failure_type": getattr(error, "failure_type", None),
            "seconds": time.monotonic() - started,
            "bank_tools_executed": 0,
        }
    finally:
        try:
            if state is not None:
                state.stop()
        finally:
            save_private_json(output / "transcript.json", transcript)
            if result is not None:
                save_private_json(output / "result.json", result)
    save_private_json(output / "profile.json", profile_session(output / "runtime.sqlite3"))
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    repo = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), *sorted((repo / "src/cayu_tau3_banking").glob("*.py"))]
    return {str(p.resolve()): digest(p) for p in paths}


def prepare(parent):
    root = Path(tempfile.mkdtemp(prefix="advice-paths-", dir=parent)).resolve()
    corpus = root / "corpus"
    corpus.mkdir(mode=0o700)
    for identifier, title, content in frozen_documents():
        save_private_json(
            corpus / f"{identifier}.json", {"id": identifier, "title": title, "content": content}
        )
    cases = frozen_cases()
    save_private_json(
        root / "cases.json", [{"id": c["id"], "customer": c["customer"]} for c in cases]
    )
    save_private_json(
        root / "expectations-evaluator-only.json", {c["id"]: c["expected"] for c in cases}
    )
    save_private_json(
        root / "protocol.json",
        {
            "framework_commit": CAYU_COMMIT,
            "model": MODEL,
            "effort": "xhigh",
            "policy": POLICY,
            "minimal_instructions": MINIMAL_INSTRUCTIONS,
            "case_ids": [c["id"] for c in cases],
            "arms": list(ARMS),
            "attempts_per_arm_per_case": 1,
            "max_concurrent_runs": 2,
            "conversation_timeout_seconds": TIMEOUT,
            "max_model_steps": MAX_STEPS,
            "max_tool_calls": MAX_CALLS,
            "source_hashes": source_hashes(),
            "input_hashes": {
                str(p.relative_to(root)): digest(p)
                for p in [
                    root / "cases.json",
                    root / "expectations-evaluator-only.json",
                    *sorted(corpus.glob("*.json")),
                ]
            },
            "scope": "Four frozen fictional advice decisions, two paths, real CayuApp scheduling and tool execution until a final reply or hard limit. No tau simulator, bank accounts or external actions.",
            "minimal_path": "Native Cayu runtime, SQLite transcript/events, immutable knowledge, native automatic recall/exposure, existing bounded public search/browse/read tools and source-evidence bookkeeping. No plan tools, worksheets, specialized audit helpers or semantic reply reviewer.",
            "controller_path": "Existing build_banking_runtime_resources with outcome planning, native recall, mandatory plan, all application helpers and existing terminal-readiness policy. Internal plan, read and worksheet results execute genuinely and the runtime continues.",
            "shared": "Same provider/model/effort, customer input, eight-document corpus, bank policy, retrieval implementation, research budgets and model/tool/wall limits. Separate runtime/knowledge state per run. Compaction threshold set above the bounded test length in both arms to avoid a summarization confound.",
            "assessment": "Manual final-answer review against separately stored expectations; retain all tool results, errors, final contexts and runtime timings. Missing final answers are limit/infrastructure outcomes, not successful recommendations. No expected answers are given to the actor.",
            "limits": "One sample per arm/case; familiar synthetic development cases, not unseen holdouts or tau scores. Does not isolate individual prompt/helper/context changes and does not reproduce task 066's long history. Concurrent pair timings include provider contention; not a statistically reliable speed estimate. No configuration promoted on this small panel alone.",
        },
    )
    return root


def verify(root):
    protocol = json.loads((root / "protocol.json").read_text())
    for path, value in protocol["source_hashes"].items():
        if digest(Path(path)) != value:
            raise ValueError("Actor/diagnostic source changed after freezing")
    for path, value in protocol["input_hashes"].items():
        if digest(root / path) != value:
            raise ValueError("Frozen input or evaluator expectations changed")
    return protocol


def run(root):
    from cayu_tau3_banking.cli import build_provider

    assert_source_provenance(Path(__file__).resolve().parents[1] / ".vendor/tau2")
    protocol = verify(root)
    cases = json.loads((root / "cases.json").read_text())
    save_private_json(root / "started.json", {"time_unix": time.time(), "scheduled_runs": 8})
    completed = []
    for index, case in enumerate(cases):
        verify(root)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for arm in ARMS if index % 2 == 0 else reversed(ARMS):
                output = root / f"{case['id']}-{arm}"
                output.mkdir(mode=0o700)
                print(json.dumps({"starting": output.name}), flush=True)
                futures.append(
                    executor.submit(
                        run_case,
                        output,
                        case,
                        arm=arm,
                        corpus_directory=root / "corpus",
                        policy=protocol["policy"],
                        provider_factory=lambda: build_provider(
                            provider="subscription",
                            codex_lb_base_url="",
                            semantic_progress_timeout_seconds=180,
                        ),
                    )
                )
            results = [f.result() for f in futures]
        for result in results:
            completed.append(result)
            print(
                json.dumps({k: result[k] for k in ("case", "arm", "status", "seconds")}), flush=True
            )
        if any(r["status"] == "error" for r in results):
            save_private_json(
                root / "stopped.json",
                {
                    "reason": "A run failed; no subsequent pairs launched",
                    "runs_finished": len(completed),
                },
            )
            return
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
