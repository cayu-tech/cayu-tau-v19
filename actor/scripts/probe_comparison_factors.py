"""Native-runtime diagnostic of research status and irrelevant-history burden.

The bootstrap is an explicitly synthetic planning snapshot, not model work or a
historical replay. Only diagnostic files are created; no actor defaults change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from cayu import SQLiteSessionStore
from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
from tau2.data_model.message import UserMessage

from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL
from cayu_tau3_banking.decision_probe import save_private_json
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.product_navigation import READ_TOOL
from cayu_tau3_banking.provenance import CAYU_COMMIT, assert_source_provenance
from cayu_tau3_banking.runtime_agent import CayuRuntimeAgent
from cayu_tau3_banking.runtime_banking import build_banking_runtime_resources
from cayu_tau3_banking.runtime_profile import profile_session

MODEL = "gpt-5.6-sol"
SOURCE_ID = "service_catalogue"
POLICY = """Fictional storage service. Use the supplied current catalogue and customer
constraints. Catalogue terms are complete; unlisted charges and benefits are zero.
Usable capacity excludes mandatory service reservations. All prices are monthly.
Advice does not authorize enrollment or changes. No banking identification or
tool discovery is needed. No provisioning or other external tool is available.
"""
OBJECTIVE = "Recommend the setup with the most usable storage within the customer's budget"
REFERENCE = {"kind": "knowledge_revision", "reference_id": SOURCE_ID, "version": "1"}
SINGLE_BASE = (
    " Choose exactly one base subscription, optionally with one compatible companion. "
    "Do not pool capacity across base subscriptions or buy repeated subscriptions."
)


def catalogue(case_id):
    if case_id not in {"case_a", "case_b"}:
        raise ValueError("Unknown case")
    companion_fee = 4 if case_id == "case_a" else 16
    rows = [
        "Sable: 120 GB usable storage for $18. No minimum seats or reservations. "
        "Moss is an optional companion subscription; see its separate terms.",
        "Birch: 150 GB usable storage for $23. No minimum seats, companions or reservations.",
        "Dune: 200 GB advertised storage for $29. Exactly 35% is reserved by the service "
        "and cannot hold customer data. No companions or other conditions.",
        f"Moss: ${companion_fee} in addition to the base subscription. Requires at least "
        "four seats. Adds 55 GB usable storage to Sable, and does not benefit other products. "
        "No reservation or other benefit applies. Advice is not enrollment consent.",
    ]
    rows.extend(
        f"Archive {i:02}: {80 + i} GB usable storage for ${12 + i % 8}. "
        "No companions, reservations or minimum-seat conditions."
        for i in range(1, 31)
    )
    return "\n".join(rows)


def customer(case_id, history):
    if history not in {"compact", "dense"}:
        raise ValueError("Unknown history condition")
    records = (
        "\n".join(
            f"Closed support ticket {i:03}: export of {i % 7 + 1} files verified; temporary "
            "download expired. No product change or current product terms are established by this record."
            for i in range(240)
        )
        if history == "dense"
        else ""
    )
    return (
        "Background: I have Sable, five seats and no companion subscriptions. "
        "An earlier standalone comparison suggested Birch, but I have not selected it.\n"
        + ("Archived completed support history:\n" + records + "\n" if records else "")
        + "Current complete catalogue:\n"
        + catalogue(case_id)
        + "\nCurrent request: Which setup gives me the most usable storage for no more than "
        "$30 total per month? Keep the seat count fixed at five. Explain the decisive "
        "costs, usable capacities and conditions. Advice only; do not change any subscription."
    )


def seed_plan(research_status):
    if research_status not in {"ready", "resolved"}:
        raise ValueError("Unknown research status")
    return {
        "items": [
            {
                "id": "compare",
                "goal": OBJECTIVE,
                "completion_mode": "information",
                "status": "in_progress",
                "outcome": "unresolved",
                "depends_on": [],
                "evidence": "An earlier standalone comparison suggested Birch; the catalogue is available.",
                "evidence_refs": [REFERENCE],
                "obligations": [
                    {
                        "id": "research",
                        "goal": "Compare eligible setups, costs and usable capacity",
                        "kind": "knowledge_search",
                        "status": research_status,
                        "evidence": "The complete current catalogue has been supplied.",
                        "evidence_refs": [REFERENCE],
                    },
                    {
                        "id": "explain",
                        "goal": "Explain the recommendation and material conditions",
                        "kind": "communication",
                        "status": "ready",
                    },
                ],
            }
        ]
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(parent, *, confirmation=False):
    root = Path(tempfile.mkdtemp(prefix="comparison-factors-", dir=parent)).resolve()
    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(
        repo / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(__file__, root / "run.py")
    schedule = []
    order = (
        ("compact", "ready"),
        ("dense", "resolved"),
        ("compact", "resolved"),
        ("dense", "ready"),
    )
    for index, case_id in enumerate(("case_a", "case_b")):
        for history, research in order if index == 0 else reversed(order):
            if confirmation and (history, research) != ("dense", "resolved"):
                continue
            name = f"{case_id}-{history}-{research}"
            schedule.append(
                {"name": name, "case": case_id, "history": history, "research": research}
            )
            save_private_json(
                root / f"{name}-input.json",
                {
                    "customer": customer(case_id, history) + (SINGLE_BASE if confirmation else ""),
                    "catalogue": catalogue(case_id),
                    "plan": seed_plan(research),
                    "policy": POLICY,
                },
            )
    save_private_json(
        root / "expectations-evaluator-only.json",
        {
            "case_a": {
                "winner": "Sable + Moss",
                "usable_gb": 175,
                "monthly_cost": 22,
                "condition": "Five seats meet the four-seat condition; separate consideration from consent.",
            },
            "case_b": {
                "winner": "Birch",
                "usable_gb": 150,
                "monthly_cost": 23,
                "condition": "Sable+Moss costs $34 and exceeds the $30 cap. Dune retains 130 GB.",
            },
        },
    )
    save_private_json(
        root / "protocol.json",
        {
            "framework_commit": CAYU_COMMIT,
            "single_base_confirmation": confirmation,
            "model": MODEL,
            "effort": "xhigh",
            "provider": "existing subscription",
            "schedule": schedule,
            "max_steps_per_episode": 10,
            "max_tool_calls_per_episode": 24,
            "timeout_seconds_per_episode": 360,
            "whole_episode_retries": 0,
            "maximum_concurrent_episodes": 1,
            "hypotheses": [
                "Does a resolved research marker steer an incomplete comparison?",
                "Does unrelated history reduce use of identical source facts?",
            ],
            "bootstrap": "Before native inference, a diagnostic observer reads the real local source through the memory API and creates a validated synthetic checklist. Bootstrap is separately recorded, not claimed model work or native tool events.",
            "differences": "Within each case, only irrelevant completed-history text and one seeded research status differ. Same model, prompt, tools, objectives, source facts and checklist evidence. No extra reviewer or changed runtime gate.",
            "limits": (
                "Two clarified single-base controls, not a corrected eight-cell factorial panel. "
                if confirmation
                else "Eight legacy factorial episodes with an ambiguous number of base subscriptions; overall optimality is unscored. "
            )
            + "Not a tau replay, historical checkpoint, compaction test, vector-search comparison, model comparison or reliability estimate. The sources explicitly document the optional subscription. These simplified controls cannot explain away the harder banking failure.",
            "grading": "Inspect complete final answers, required exclusions, calculations, plan transitions and actual captures. No answer feedback to actor; expectations loaded only after runs. Stop further launches on infrastructure errors.",
        },
    )
    save_private_json(
        root / "manifest.json",
        {"files": {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}},
    )
    return root


def verify(root):
    for name, expected in json.loads((root / "manifest.json").read_text())["files"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen input or implementation changed")


class BootstrapObserver:
    def __init__(self, state, plan, output):
        self.state, self.plan, self.output = state, plan, output
        self.used = False

    async def history(self, messages):
        await self.state.history(messages)

    async def customer(self, text):
        await self.state.customer(text)
        if self.used:
            raise ValueError("Diagnostic bootstrap accepts exactly one customer turn")
        self.used = True
        read = await self.state.memory.invoke(
            READ_TOOL,
            {"documents": [{"entry_id": SOURCE_ID, "revision": 1}]},
            allow_unplanned_public_research=True,
        )
        if read.is_error:
            raise ValueError("Diagnostic source bootstrap rejected")
        plan = await self.state.memory.invoke(SYNC_WORK_PLAN_TOOL, self.plan)
        if plan.is_error:
            raise ValueError("Diagnostic checklist bootstrap rejected: " + plan.content)
        self.state.refresh_exposure()
        save_private_json(
            self.output / "bootstrap.json",
            {
                "source_result": read.content,
                "plan_result": plan.content,
                "plan": self.state.controller.export_state(),
                "native_model_calls": 0,
                "synthetic_initialization": True,
            },
        )

    async def reply(self, text):
        await self.state.reply(text)


def run(root, name, *, provider_factory=None, model=MODEL, timeout=360):
    verify(root)
    protocol = json.loads((root / "protocol.json").read_text())
    if not any(r["name"] == name for r in protocol["schedule"]):
        raise ValueError("Unknown scheduled episode")
    case = json.loads((root / f"{name}-input.json").read_text())
    output = root / "results" / name
    output.mkdir(parents=True, mode=0o700)
    save_private_json(output / "started.json", {"name": name})
    corpus_path = output / "corpus"
    corpus_path.mkdir()
    save_private_json(
        corpus_path / f"{SOURCE_ID}.json",
        {
            "id": SOURCE_ID,
            "title": "Current complete storage catalogue",
            "content": case["catalogue"],
        },
    )
    captured = {}
    if provider_factory is None:
        import cayu_tau3_banking.advice as advice
        from cayu_tau3_banking.cli import build_provider
        from cayu_tau3_banking.provider_diagnostics import DiagnosticOpenAITransport

        if not Path(advice.__file__).resolve().is_relative_to(root / "src"):
            raise ValueError("Live actor must use frozen source")
        assert_source_provenance(Path(__file__).resolve().parents[2] / ".vendor/tau2")

        def provider_factory():
            provider = build_provider(
                provider="subscription", codex_lb_base_url="", semantic_progress_timeout_seconds=180
            )
            provider.transport = DiagnosticOpenAITransport(
                provider.transport, output / "provider-diagnostics"
            )
            return provider

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
            resources = build_banking_runtime_resources(
                broker,
                specs,
                provider=provider,
                model=model,
                domain_policy=case["policy"],
                memory_runtime=BankingMemoryRuntime(
                    BankingKnowledgeCorpus.from_directory(corpus_path)
                ),
                session_store=store,
                aclose=close,
                native_recall=True,
                planning_mode="outcomes",
                advice_mode="lean",
                context_capture_directory=output / "contexts",
                compaction=CheckpointCompactionContextPolicy(
                    compactor=ModelCompactor(provider=provider, model=model),
                    compact_after_messages=1000,
                    max_user_turns=100,
                ),
                provider_options={
                    "thinking": {"enabled": True, "effort": "xhigh"},
                    "openai": {"parallel_tool_calls": False},
                },
            )
        except BaseException:
            await close()
            raise
        captured["state"] = resources.observer
        return replace(
            resources, observer=BootstrapObserver(resources.observer, case["plan"], output)
        )

    agent = CayuRuntimeAgent(
        tools=[],
        domain_policy=case["policy"],
        resource_factory=factory,
        max_steps=10,
        max_tool_calls=24,
        callback_timeout_s=timeout + 10,
        conversation_timeout_s=timeout,
        diagnostic_stop_on_tool_error=False,
    )
    state = None
    transcript = []
    try:
        state = agent.get_init_state()
        message = UserMessage.text(case["customer"])
        transcript.append(message.model_dump(mode="json"))
        reply, _ = agent.generate_next_message(message, state)
        transcript.append(reply.model_dump(mode="json"))
        if reply.tool_calls or not isinstance(reply.content, str) or not reply.content.strip():
            raise ValueError("Expected complete advice, not an external action")
        result = {
            "name": name,
            "status": "answered",
            "reply": reply.content,
            "plan": captured["state"].controller.export_state(),
            "external_tools_executed": 0,
        }
    except Exception as error:
        result = {
            "name": name,
            "status": "error",
            "error_type": type(error).__name__,
            "failure_type": getattr(error, "failure_type", None),
        }
    finally:
        try:
            if state is not None:
                state.stop()
        finally:
            save_private_json(output / "transcript.json", transcript)
    save_private_json(output / "result.json", result)
    save_private_json(output / "profile.json", profile_session(output / "runtime.sqlite3"))
    verify(root)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--output-parent", type=Path, default=Path("artifacts"))
    p.add_argument("--confirmation", action="store_true")
    r = commands.add_parser("run")
    r.add_argument("root", type=Path)
    r.add_argument("name")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.output_parent, confirmation=args.confirmation))
    else:
        result = run(args.root.resolve(), args.name)
        print(json.dumps({k: result[k] for k in ("name", "status")}))
        if result["status"] != "answered":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
