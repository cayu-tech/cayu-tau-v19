"""Loop-owned production resources and explicit native τ agent registration."""

from dataclasses import replace
from uuid import uuid4

from cayu import RetryPolicy, SQLiteSessionStore
from tau2.registry import registry

from cayu_tau3_banking.action_verifier import BankingActionVerifier
from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.plan_verifier import BankingPlanVerifier
from cayu_tau3_banking.response_verifier import BankingResponseVerifier
from cayu_tau3_banking.runtime_agent import CayuRuntimeAgent
from cayu_tau3_banking.runtime_banking import build_banking_runtime_resources
from cayu_tau3_banking.runtime_helpers import TrackedHelperProvider
from cayu_tau3_banking.runtime_usage import RuntimeUsageAccounting


def register_native_agent(args, corpus):
    from cayu_tau3_banking.cli import build_provider

    name = "cayu_native_banking"
    args.runtime_store_directory.mkdir(parents=True, exist_ok=True)

    def factory(tools, domain_policy, **kwargs):
        # Never inspect τ task/oracle arguments.
        del kwargs

        async def resources(broker, specs):
            provider = build_provider(
                provider=args.provider,
                codex_lb_base_url=args.codex_lb_base_url,
                semantic_progress_timeout_seconds=args.provider_semantic_progress_timeout_seconds,
            )
            store = None

            async def cleanup():
                try:
                    await provider.aclose()
                finally:
                    if store is not None:
                        await store.close()

            try:
                diagnostic_directory = getattr(args, "provider_diagnostics_directory", None)
                if diagnostic_directory is not None:
                    from cayu_tau3_banking.provider_diagnostics import DiagnosticOpenAITransport

                    provider.transport = DiagnosticOpenAITransport(
                        provider.transport, diagnostic_directory
                    )
                store = SQLiteSessionStore(args.runtime_store_directory / f"{uuid4().hex}.sqlite3")
                return assemble(broker, specs, provider, store, cleanup)
            except BaseException:
                await cleanup()
                raise

        def assemble(broker, specs, provider, store, cleanup):
            usage = RuntimeUsageAccounting()
            helper = TrackedHelperProvider(provider, usage)
            options = {
                "thinking": {"enabled": True, "effort": "low"},
                "openai": {"parallel_tool_calls": False, "tool_choice": "required"},
            }
            memory = BankingMemoryRuntime(
                corpus,
                plan_verifier=BankingPlanVerifier(
                    provider=helper, model=args.model, options=options
                )
                if args.semantic_plan_verifier
                else None,
                action_verifier=BankingActionVerifier(
                    provider=helper, model=args.model, options=options
                )
                if args.semantic_action_verifier
                else None,
                response_verifier=BankingResponseVerifier(
                    provider=helper, model=args.model, options=options
                )
                if args.semantic_response_verifier
                else None,
            )

            result = build_banking_runtime_resources(
                broker,
                specs,
                provider=provider,
                model=args.model,
                domain_policy=domain_policy,
                memory_runtime=memory,
                session_store=store,
                aclose=cleanup,
                native_recall=True,
                lexical_navigation=getattr(args, "lexical_navigation", False),
                controller_mode=getattr(args, "controller_mode", "enforced"),
                decision_packets=getattr(args, "decision_packets", False),
                task_memory_mode={"constraints-v4": "constraints", "notebook-v4": "markdown"}.get(getattr(args, "candidate_profile", None)),
                task_memory_directory=args.runtime_store_directory / "task-memory",
                planning_mode=args.planning_mode,
                advice_mode=args.advice_mode,
                focused_policy_file=getattr(args, "focused_policy_file", None),
                curated_context=(
                    args.curated_context_file.read_text()
                    if getattr(args, "curated_context_file", None) is not None
                    else None
                ),
                context_capture_directory=args.context_capture_directory,
                provider_options={
                    "thinking": {"enabled": True, "effort": args.reasoning_effort},
                    "openai": {
                        "parallel_tool_calls": getattr(args, "runtime_parallel_tools", False)
                    },
                },
            )
            return replace(result, usage_accounting=usage, untracked_model_helpers=False)

        return CayuRuntimeAgent(
            tools=tools,
            domain_policy=domain_policy,
            resource_factory=resources,
            callback_timeout_s=args.completion_timeout_seconds,
            conversation_timeout_s=getattr(args, "runtime_conversation_timeout_seconds", None) or args.simulation_timeout_seconds,
            grade_budget_stops=getattr(args, "grade_agent_budget_stops", False),
            max_steps=args.runtime_max_steps,
            retry_policy=RetryPolicy(
                max_attempts=getattr(args, "runtime_model_max_attempts", 3),
                max_unknown_attempts=1,
                initial_delay_s=2.0,
                max_delay_s=15.0,
            ),
        )

    registry.register_agent_factory(factory, name)
    return name
