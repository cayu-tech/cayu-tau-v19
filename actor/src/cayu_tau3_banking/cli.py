"""Command-line entry point for reproducible tau3 banking runs."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from cayu import TextEmbeddingProvider
from cayu.providers import OpenAIProvider
from cayu.providers.deadlines import ProviderStreamDeadlines
from tau2.config import DEFAULT_RETRY_ATTEMPTS
from tau2.data_model.simulation import TextRunConfig
from tau2.runner import run_domain

from cayu_tau3_banking.action_verifier import BankingActionVerifier
from cayu_tau3_banking.agent import register_cayu_bridge_agent
from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.attempt_diagnostics import AttemptDiagnosticsRecorder
from cayu_tau3_banking.bridge import (
    DEFAULT_COMPLETION_TIMEOUT_SECONDS,
    CayuModelGateway,
    CayuTauBridge,
)
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus, BankingKnowledgeError
from cayu_tau3_banking.plan_verifier import BankingPlanVerifier
from cayu_tau3_banking.provenance import assert_source_provenance
from cayu_tau3_banking.response_verifier import BankingResponseVerifier
from cayu_tau3_banking.subscription_recovery import RecoveringSubscriptionProvider

_DEFAULT_TAU2_REPOSITORY = Path(__file__).resolve().parents[2] / ".vendor" / "tau2"
_DEFAULT_CODEX_LB_BASE_URL = "https://codex-lb.cayu.ai"
_CODEX_LB_API_KEY_ENV = "CODEX_LB_API_KEY"
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
_DEFAULT_USER_COMPLETION_TIMEOUT_SECONDS = 300.0
_DEFAULT_SIMULATION_TIMEOUT_SECONDS = 3_600.0
_DEFAULT_PROVIDER_SEMANTIC_PROGRESS_TIMEOUT_SECONDS = 300.0
_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3_072,
    "text-embedding-3-small": 1_536,
}


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-profile", choices=("evidence-v2", "evidence-v2-no-packets", "simplified-v3", "constraints-v4", "notebook-v4"), default=None)
    parser.add_argument("--runtime", choices=("native", "bridge"), default="native")
    parser.add_argument(
        "--lexical-navigation",
        action="store_true",
        help="Add bounded lexical document locators alongside native recall; not read evidence.",
    )
    parser.add_argument(
        "--controller-mode",
        choices=("enforced", "advisory"),
        default="enforced",
        help="Advisory checklists do not veto execution or replies; authoritative tool safeguards remain.",
    )
    parser.add_argument(
        "--planning-mode",
        choices=("outcomes", "exact"),
        default="outcomes",
        help="Native outcome checklist, or legacy exact-operation baseline.",
    )
    parser.add_argument(
        "--decision-packets",
        action="store_true",
        help="Opt-in concrete mutation and cross-outcome checks in advisory mode.",
    )
    parser.add_argument("--runtime-max-steps", type=_positive_int, default=64)
    parser.add_argument(
        "--advice-mode",
        choices=("controller", "lean"),
        default="controller",
        help="Opt-in concise public advice without a mandatory plan; banking-tool guards remain enabled.",
    )
    parser.add_argument(
        "--runtime-parallel-tools",
        action="store_true",
        help="Allow multiple native actor tool calls per model round; existing authorization still applies.",
    )
    parser.add_argument(
        "--context-capture-directory",
        type=Path,
        help="Opt-in native composed-context snapshots. Contains sensitive conversation text; keep local.",
    )
    parser.add_argument(
        "--provider-diagnostics-directory",
        type=Path,
        help="Opt-in native prepared requests and safe error categories; contains private conversation data.",
    )
    parser.add_argument(
        "--runtime-store-directory", type=Path, default=Path("data/native-sessions")
    )
    parser.add_argument(
        "--provider",
        choices=("subscription", "api", "codex-lb"),
        default="codex-lb",
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="xhigh",
    )
    parser.add_argument("--user-model", default="openai/gpt-5.2")
    parser.add_argument(
        "--user-provider",
        choices=("platform", "codex-lb"),
        default="platform",
        help=(
            "Route the tau user simulator through the OpenAI Platform or Codex LB. "
            "Codex LB is useful for local smoke tests but is not leaderboard-comparable "
            "unless it exposes the required simulator model."
        ),
    )
    parser.add_argument(
        "--codex-lb-base-url",
        default=_DEFAULT_CODEX_LB_BASE_URL,
        help="Codex LB origin without the /v1 suffix.",
    )
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--grade-agent-budget-stops", action="store_true")
    parser.add_argument("--runtime-conversation-timeout-seconds", type=_positive_float, default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--runtime-model-max-attempts",
        type=int,
        choices=range(1, 11),
        default=3,
        help="Native Cayu attempts per model step for classified transient failures only; unknown errors and whole-task replay remain separate.",
    )
    parser.add_argument(
        "--completion-timeout-seconds",
        type=_positive_float,
        default=DEFAULT_COMPLETION_TIMEOUT_SECONDS,
        help="Hard wall-clock deadline for one generated assistant turn.",
    )
    parser.add_argument(
        "--provider-semantic-progress-timeout-seconds",
        type=_positive_float,
        default=_DEFAULT_PROVIDER_SEMANTIC_PROGRESS_TIMEOUT_SECONDS,
        help=(
            "Maximum time one Cayu model stream may produce no normalized semantic "
            "progress before failing closed."
        ),
    )
    parser.add_argument(
        "--simulation-timeout-seconds",
        type=_positive_float,
        default=_DEFAULT_SIMULATION_TIMEOUT_SECONDS,
        help="Maximum wall-clock time for one complete tau simulation.",
    )
    parser.add_argument(
        "--user-completion-timeout-seconds",
        type=_positive_float,
        default=_DEFAULT_USER_COMPLETION_TIMEOUT_SECONDS,
        help="Network deadline for one generated user-simulator turn.",
    )
    parser.add_argument(
        "--max-retries",
        type=_nonnegative_int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help=(
            "Maximum full-task retries after a simulation exception. Use 0 for a "
            "single-attempt diagnostic gate."
        ),
    )
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--save-to")
    parser.add_argument(
        "--attempt-diagnostics",
        type=Path,
        help=(
            "Append oracle-free discarded-attempt metadata to this private JSONL file. "
            "Records contain opaque equality tags, tool names, and bounded counters."
        ),
    )
    parser.add_argument(
        "--cayu-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Cayu knowledge recall and work context instead of tau retrieval tools.",
    )
    parser.add_argument(
        "--memory-retrieval",
        choices=("lexical", "hybrid"),
        default="lexical",
        help=(
            "Select the Cayu knowledge-store treatment. Hybrid uses the OpenAI Platform "
            "embedding endpoint but keeps tau retrieval disabled."
        ),
    )
    parser.add_argument("--embedding-model", default=_DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dimensions", type=int)
    parser.add_argument(
        "--focused-policy-file",
        type=Path,
        help="Diagnostic frozen source manifest; suppresses automatic retrieval, retains fallback tools.",
    )
    parser.add_argument(
        "--curated-context-file",
        type=Path,
        help="Diagnostic-only file of source-backed context; disables public research tools.",
    )
    parser.add_argument(
        "--retrieval-config",
        help=(
            "tau retrieval configuration. Defaults to no_knowledge with Cayu memory and "
            "alltools without it. Cayu memory requires no_knowledge."
        ),
    )
    parser.add_argument(
        "--ledger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inject the bounded public-observation ledger (use --no-ledger for ablation).",
    )
    parser.add_argument(
        "--workflow-controller",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable private structured work planning for compound requests.",
    )
    parser.add_argument(
        "--semantic-plan-verifier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use an isolated read-only model review to validate plan coverage and completion "
            "modes before customer-facing text."
        ),
    )
    parser.add_argument(
        "--semantic-action-verifier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use an isolated read-only model review to validate consequential environment "
            "calls against public intent, observations, and workflow state before execution."
        ),
    )
    parser.add_argument(
        "--semantic-response-verifier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use an isolated read-only model review to require each customer-facing draft "
            "to cover the controller's complete response disposition."
        ),
    )
    parser.add_argument(
        "--tau2-repository",
        type=Path,
        default=_DEFAULT_TAU2_REPOSITORY,
        help="Pinned tau2 source checkout to verify before model or benchmark execution.",
    )
    return parser


def build_provider(
    *,
    provider: str,
    codex_lb_base_url: str,
    semantic_progress_timeout_seconds: float,
):
    """Build the selected Cayu provider without persisting credentials in run metadata."""

    deadlines = ProviderStreamDeadlines(
        semantic_progress_timeout_s=semantic_progress_timeout_seconds
    )
    if provider == "subscription":
        return RecoveringSubscriptionProvider(stream_deadlines=deadlines)
    if provider == "api":
        return OpenAIProvider(stream_deadlines=deadlines)
    if provider != "codex-lb":
        raise ValueError(f"Unsupported provider: {provider!r}.")
    return OpenAIProvider(
        api_key=_required_codex_lb_api_key(),
        base_url=_validated_codex_lb_base_url(codex_lb_base_url),
        name="codex-lb",
        stream_deadlines=deadlines,
    )


def resolve_user_runtime(
    *,
    user_provider: str,
    user_model: str,
    codex_lb_base_url: str,
    completion_timeout_seconds: float,
) -> tuple[dict[str, object], str | None]:
    """Return public LiteLLM arguments plus an optional process-only API key override."""

    if (
        type(completion_timeout_seconds) not in {int, float}
        or isinstance(completion_timeout_seconds, bool)
        or not math.isfinite(completion_timeout_seconds)
        or completion_timeout_seconds <= 0
    ):
        raise ValueError("completion_timeout_seconds must be a positive number.")
    llm_args: dict[str, object] = {
        "reasoning_effort": "low",
        "timeout": float(completion_timeout_seconds),
    }
    if user_provider == "platform":
        return llm_args, None
    if user_provider != "codex-lb":
        raise ValueError(f"Unsupported user provider: {user_provider!r}.")
    if not user_model.startswith("openai/"):
        raise ValueError("The Codex LB user model must use LiteLLM's openai/ prefix.")
    base_url = _validated_codex_lb_base_url(codex_lb_base_url)
    llm_args["api_base"] = f"{base_url}/v1"
    return llm_args, _required_codex_lb_api_key()


def _required_codex_lb_api_key() -> str:
    api_key = os.environ.get(_CODEX_LB_API_KEY_ENV, "").strip()
    if not api_key:
        raise ValueError(f"Codex LB requires {_CODEX_LB_API_KEY_ENV} in the trusted host process.")
    return api_key


def _validated_codex_lb_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("codex_lb_base_url must be nonblank text.")
    normalized = value.strip().rstrip("/")
    if normalized.endswith("/v1"):
        raise ValueError("codex_lb_base_url must not include the /v1 suffix.")
    return normalized


@contextmanager
def _temporary_openai_api_key(api_key: str | None) -> Iterator[None]:
    """Expose a proxy key to LiteLLM for one run, then restore the host environment."""

    if api_key is None:
        yield
        return
    previous = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous


def resolve_retrieval_config(*, cayu_memory: bool, requested: str | None) -> str:
    """Keep Cayu retrieval and tau retrieval mutually exclusive by construction."""

    if type(cayu_memory) is not bool:
        raise TypeError("cayu_memory must be a boolean.")
    if requested is not None and (not isinstance(requested, str) or not requested.strip()):
        raise ValueError("requested retrieval configuration must be nonblank text or null.")
    resolved = requested.strip() if requested is not None else None
    resolved = resolved or ("no_knowledge" if cayu_memory else "alltools")
    if cayu_memory and resolved != "no_knowledge":
        raise ValueError("Cayu memory requires tau retrieval_config=no_knowledge.")
    return resolved


def resolve_embedding_dimensions(*, model: str, requested: int | None) -> int:
    """Resolve a reproducible embedding dimension for one supported model."""

    if not isinstance(model, str) or not model.strip():
        raise ValueError("embedding model must be nonblank text.")
    if requested is not None:
        if type(requested) is not int or requested <= 0:
            raise ValueError("embedding dimensions must be a positive integer.")
        return requested
    dimensions = _EMBEDDING_DIMENSIONS.get(model.strip())
    if dimensions is None:
        raise ValueError(
            "Unknown embedding model dimensions; pass --embedding-dimensions explicitly."
        )
    return dimensions


async def build_memory_corpus(
    documents: Path,
    *,
    retrieval_mode: str,
    embedding_model: str,
    embedding_dimensions: int | None,
    embedding_provider: TextEmbeddingProvider | None = None,
) -> BankingKnowledgeCorpus:
    """Build the selected Cayu corpus while keeping lexical and hybrid runs explicit."""

    if retrieval_mode == "lexical":
        if embedding_dimensions is not None:
            raise ValueError("Lexical retrieval does not use embedding dimensions.")
        return BankingKnowledgeCorpus.from_directory(documents)
    if retrieval_mode != "hybrid":
        raise ValueError("memory retrieval must be lexical or hybrid.")
    dimensions = resolve_embedding_dimensions(
        model=embedding_model,
        requested=embedding_dimensions,
    )
    provider = embedding_provider
    if provider is None:
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise ValueError("Hybrid Cayu memory requires OPENAI_API_KEY for embeddings.")
        provider = OpenAIProvider()
    return await BankingKnowledgeCorpus.from_directory_with_embeddings(
        documents,
        embedding_provider=provider,
        embedding_model=embedding_model,
        embedding_dimensions=dimensions,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.memory_retrieval == "hybrid":
        from cayu_tau3_banking.runtime_embeddings import OwnedEmbeddingProvider

        if not os.environ.get("OPENAI_API_KEY", "").strip():
            raise SystemExit("Hybrid Cayu memory requires OPENAI_API_KEY for embeddings.")
        # Capture the Platform credential before a simulator proxy override.
        api_key = os.environ["OPENAI_API_KEY"]
        with OwnedEmbeddingProvider(lambda: OpenAIProvider(api_key=api_key)) as embeddings:
            _run(args, embedding_provider=embeddings)
    else:
        _run(args)


def _run(args, *, embedding_provider=None) -> None:
    if getattr(args, "candidate_profile", None) in {"evidence-v2", "evidence-v2-no-packets", "simplified-v3", "constraints-v4", "notebook-v4"}:
        expected = {
            "runtime": "native", "planning_mode": "outcomes",
            "controller_mode": "advisory", "advice_mode": "lean",
            "decision_packets": args.candidate_profile == "evidence-v2", "grade_agent_budget_stops": True,
            "semantic_plan_verifier": False, "semantic_action_verifier": False,
            "semantic_response_verifier": False,
        }
        wrong = [key for key, value in expected.items() if getattr(args, key, None) != value]
        if wrong:
            raise ValueError("evidence-v2 configuration mismatch: " + ", ".join(wrong))
        if getattr(args, "focused_policy_file", None) or getattr(args, "curated_context_file", None):
            raise ValueError("evidence-v2 cannot use task-selected policy bundles")
    if getattr(args, "lexical_navigation", False) and args.runtime != "native":
        raise SystemExit("Lexical navigation requires native runtime.")
    if getattr(args, "decision_packets", False) and (
        args.runtime != "native" or args.controller_mode != "advisory"
    ):
        raise ValueError("Decision packets require native advisory mode")
    if getattr(args, "controller_mode", "enforced") == "advisory" and (
        args.runtime != "native" or args.planning_mode != "outcomes" or args.advice_mode != "lean"
    ):
        raise SystemExit(
            "Advisory control requires native runtime, outcome planning and lean advice."
        )
    if args.advice_mode == "lean" and (
        args.runtime != "native" or args.planning_mode != "outcomes"
    ):
        raise SystemExit("Lean advice requires native runtime and outcome planning.")
    if args.planning_mode == "outcomes":
        if args.runtime != "native":
            raise SystemExit(
                "Outcome planning requires --runtime native; use --planning-mode exact for bridge."
            )
        if any(
            (
                args.semantic_plan_verifier,
                args.semantic_action_verifier,
                args.semantic_response_verifier,
            )
        ):
            raise SystemExit(
                "Outcome planning disables legacy veto reviewers; use --planning-mode exact for that baseline."
            )
    assert_source_provenance(args.tau2_repository)
    if args.runtime == "native" and not (
        args.cayu_memory and args.ledger and args.workflow_controller
    ):
        raise SystemExit("Native runtime requires Cayu memory, ledger and workflow controller.")
    if args.runtime == "native" and args.attempt_diagnostics is not None:
        raise SystemExit(
            "Native runtime records SQLite events; --attempt-diagnostics is bridge-only."
        )
    try:
        retrieval_config = resolve_retrieval_config(
            cayu_memory=args.cayu_memory,
            requested=args.retrieval_config,
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if not args.cayu_memory and args.memory_retrieval != "lexical":
        raise SystemExit("--memory-retrieval requires Cayu memory.")
    corpus = None
    if args.cayu_memory:
        documents = (
            args.tau2_repository / "data" / "tau2" / "domains" / "banking_knowledge" / "documents"
        )
        try:
            corpus = asyncio.run(
                build_memory_corpus(
                    documents,
                    retrieval_mode=args.memory_retrieval,
                    embedding_model=args.embedding_model,
                    embedding_dimensions=args.embedding_dimensions,
                    embedding_provider=embedding_provider,
                )
            )
        except (BankingKnowledgeError, TypeError, ValueError) as exc:
            from cayu_tau3_banking.embedding_diagnostics import EmbeddingInitializationError
            from cayu_tau3_banking.runtime_embeddings import OwnedEmbeddingProvider

            if isinstance(embedding_provider, OwnedEmbeddingProvider):
                diagnostic = embedding_provider.failure_diagnostic
                if diagnostic is not None:
                    raise EmbeddingInitializationError(diagnostic) from exc
            raise SystemExit(str(exc)) from exc
    if args.runtime == "native":
        from cayu_tau3_banking.provenance import CAYU_COMMIT, TAU2_COMMIT
        from cayu_tau3_banking.runtime_cli import register_native_agent

        llm_args_user, user_api_key = resolve_user_runtime(
            user_provider=args.user_provider,
            user_model=args.user_model,
            codex_lb_base_url=args.codex_lb_base_url,
            completion_timeout_seconds=args.user_completion_timeout_seconds,
        )
        agent_name = register_native_agent(args, corpus)
        with _temporary_openai_api_key(user_api_key):
            run_domain(
                TextRunConfig(
                    domain="banking_knowledge",
                    agent=agent_name,
                    llm_agent=args.model,
                    llm_args_agent={
                        "runtime": "native",
                        "provider": args.provider,
                        "user_provider": args.user_provider,
                        "runtime_max_steps": args.runtime_max_steps,
                        "planning_mode": args.planning_mode,
                        "controller_mode": args.controller_mode,
                        "decision_packets": getattr(args, "decision_packets", False),
                        "candidate_profile": getattr(args, "candidate_profile", None),
                        "intent_interpretation": "acting_model" if args.controller_mode == "advisory" else "legacy_guards",
                        "focused_policy_file": str(args.focused_policy_file) if args.focused_policy_file else None,
                        "lexical_navigation": getattr(args, "lexical_navigation", False),
                        "cayu_commit": CAYU_COMMIT,
                        "tau2_commit": TAU2_COMMIT,
                        "reasoning_effort": args.reasoning_effort,
                        "memory_retrieval": args.memory_retrieval,
                        "corpus_sha256": corpus.corpus_sha256,
                        "semantic_plan_verifier": args.semantic_plan_verifier,
                        "semantic_action_verifier": args.semantic_action_verifier,
                        "semantic_response_verifier": args.semantic_response_verifier,
                    },
                    llm_user=args.user_model,
                    llm_args_user=llm_args_user,
                    retrieval_config=retrieval_config,
                    task_ids=args.task_ids,
                    num_tasks=args.num_tasks,
                    num_trials=args.num_trials,
                    max_concurrency=args.max_concurrency,
                    max_steps=args.max_steps,
                    timeout=args.simulation_timeout_seconds,
                    max_retries=args.max_retries,
                    seed=args.seed,
                    save_to=args.save_to,
                    enforce_communication_protocol=True,
                )
            )
        return
    try:
        provider = build_provider(
            provider=args.provider,
            codex_lb_base_url=args.codex_lb_base_url,
            semantic_progress_timeout_seconds=(args.provider_semantic_progress_timeout_seconds),
        )
        llm_args_user, user_api_key = resolve_user_runtime(
            user_provider=args.user_provider,
            user_model=args.user_model,
            codex_lb_base_url=args.codex_lb_base_url,
            completion_timeout_seconds=args.user_completion_timeout_seconds,
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    memory_runtime = (
        None
        if corpus is None
        else BankingMemoryRuntime(
            corpus,
            plan_verifier=(
                BankingPlanVerifier(
                    provider=provider,
                    model=args.model,
                    options={
                        "thinking": {"enabled": True, "effort": "low"},
                        "openai": {
                            "parallel_tool_calls": False,
                            "tool_choice": "required",
                        },
                    },
                )
                if args.semantic_plan_verifier and args.workflow_controller
                else None
            ),
            action_verifier=(
                BankingActionVerifier(
                    provider=provider,
                    model=args.model,
                    options={
                        "thinking": {"enabled": True, "effort": "low"},
                        "openai": {
                            "parallel_tool_calls": False,
                            "tool_choice": "required",
                        },
                    },
                )
                if args.semantic_action_verifier and args.workflow_controller
                else None
            ),
            response_verifier=(
                BankingResponseVerifier(
                    provider=provider,
                    model=args.model,
                    options={
                        "thinking": {"enabled": True, "effort": "low"},
                        "openai": {
                            "parallel_tool_calls": False,
                            "tool_choice": "required",
                        },
                    },
                )
                if args.semantic_response_verifier and args.workflow_controller
                else None
            ),
        )
    )
    bridge = CayuTauBridge(
        provider=provider,
        model=args.model,
        options={
            "thinking": {"enabled": True, "effort": args.reasoning_effort},
        },
    )
    attempt_diagnostics_recorder = (
        None
        if args.attempt_diagnostics is None
        else AttemptDiagnosticsRecorder(args.attempt_diagnostics)
    )
    with CayuModelGateway(
        bridge,
        completion_timeout_s=args.completion_timeout_seconds,
        attempt_diagnostics_recorder=attempt_diagnostics_recorder,
    ) as gateway:
        agent_name = register_cayu_bridge_agent(
            gateway,
            memory_runtime=memory_runtime,
            ledger_enabled=args.ledger,
            workflow_controller_enabled=args.workflow_controller,
        )
        with _temporary_openai_api_key(user_api_key):
            run_domain(
                TextRunConfig(
                    domain="banking_knowledge",
                    agent=agent_name,
                    llm_agent=args.model,
                    llm_args_agent={
                        "reasoning_effort": args.reasoning_effort,
                        "provider_semantic_progress_timeout_seconds": (
                            args.provider_semantic_progress_timeout_seconds
                        ),
                    },
                    llm_user=args.user_model,
                    llm_args_user=llm_args_user,
                    retrieval_config=retrieval_config,
                    task_ids=args.task_ids,
                    num_tasks=args.num_tasks,
                    num_trials=args.num_trials,
                    max_concurrency=args.max_concurrency,
                    max_steps=args.max_steps,
                    timeout=args.simulation_timeout_seconds,
                    max_retries=args.max_retries,
                    seed=args.seed,
                    save_to=args.save_to,
                    enforce_communication_protocol=True,
                )
            )


if __name__ == "__main__":
    main()
