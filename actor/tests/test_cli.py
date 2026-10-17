from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from cayu.providers import OpenAIProvider, OpenAISubscriptionProvider

from cayu_tau3_banking.cli import (
    _parser,
    _temporary_openai_api_key,
    build_memory_corpus,
    build_provider,
    resolve_embedding_dimensions,
    resolve_retrieval_config,
    resolve_user_runtime,
)


def test_retry_control_preserves_upstream_default_and_allows_zero() -> None:
    assert _parser().parse_args([]).max_retries == 3
    assert _parser().parse_args(["--max-retries", "0"]).max_retries == 0


def test_default_actor_uses_lb_without_changing_the_platform_simulator(monkeypatch) -> None:
    args = _parser().parse_args([])
    assert args.provider == "codex-lb"
    assert args.model == "gpt-5.6-sol"
    assert args.user_provider == "platform"
    assert args.user_model == "openai/gpt-5.2"
    monkeypatch.delenv("CODEX_LB_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CODEX_LB_API_KEY"):
        build_provider(
            provider=args.provider,
            codex_lb_base_url=args.codex_lb_base_url,
            semantic_progress_timeout_seconds=45,
        )


def test_outcome_planning_defaults_to_no_semantic_veto_reviewers() -> None:
    defaults = _parser().parse_args([])
    disabled = _parser().parse_args(
        [
            "--no-semantic-plan-verifier",
            "--no-semantic-action-verifier",
            "--no-semantic-response-verifier",
        ]
    )

    assert defaults.planning_mode == "outcomes"
    assert defaults.semantic_plan_verifier is False
    assert defaults.semantic_action_verifier is False
    assert defaults.semantic_response_verifier is False
    assert disabled.semantic_plan_verifier is False
    assert disabled.semantic_action_verifier is False
    assert disabled.semantic_response_verifier is False


def test_retry_control_rejects_negative_values() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["--max-retries", "-1"])


def test_completion_timeout_is_bounded_and_configurable() -> None:
    assert _parser().parse_args([]).completion_timeout_seconds == 600.0
    assert (
        _parser().parse_args(["--completion-timeout-seconds", "12.5"]).completion_timeout_seconds
        == 12.5
    )
    assert _parser().parse_args([]).provider_semantic_progress_timeout_seconds == 300.0
    assert (
        _parser()
        .parse_args(["--provider-semantic-progress-timeout-seconds", "45"])
        .provider_semantic_progress_timeout_seconds
        == 45.0
    )
    assert _parser().parse_args([]).simulation_timeout_seconds == 3_600.0
    assert (
        _parser().parse_args(["--simulation-timeout-seconds", "45"]).simulation_timeout_seconds
        == 45.0
    )
    assert _parser().parse_args([]).user_completion_timeout_seconds == 300.0
    assert (
        _parser()
        .parse_args(["--user-completion-timeout-seconds", "17.5"])
        .user_completion_timeout_seconds
        == 17.5
    )


@pytest.mark.parametrize(
    "option",
    [
        "--completion-timeout-seconds",
        "--provider-semantic-progress-timeout-seconds",
        "--simulation-timeout-seconds",
        "--user-completion-timeout-seconds",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_timeouts_reject_nonpositive_or_nonfinite_values(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args([option, value])


def test_cayu_memory_disables_tau_retrieval_tools_by_default() -> None:
    assert resolve_retrieval_config(cayu_memory=True, requested=None) == "no_knowledge"


def test_control_agent_retains_tau_alltools_default() -> None:
    assert resolve_retrieval_config(cayu_memory=False, requested=None) == "alltools"


def test_cayu_memory_rejects_duplicate_tau_retrieval() -> None:
    with pytest.raises(ValueError, match="requires tau retrieval_config=no_knowledge"):
        resolve_retrieval_config(cayu_memory=True, requested="alltools")


def test_control_agent_accepts_explicit_tau_retrieval() -> None:
    assert resolve_retrieval_config(cayu_memory=False, requested="bm25") == "bm25"


@pytest.mark.parametrize(
    ("model", "dimensions"),
    (("text-embedding-3-large", 3_072), ("text-embedding-3-small", 1_536)),
)
def test_embedding_dimensions_are_derived_from_supported_models(
    model: str,
    dimensions: int,
) -> None:
    assert resolve_embedding_dimensions(model=model, requested=None) == dimensions


def test_unknown_embedding_model_requires_explicit_dimensions() -> None:
    with pytest.raises(ValueError, match="pass --embedding-dimensions"):
        resolve_embedding_dimensions(model="custom-embedding", requested=None)


@pytest.mark.parametrize("dimensions", (True, 0, -1))
def test_embedding_dimensions_must_be_positive_integers(dimensions: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        resolve_embedding_dimensions(
            model="text-embedding-3-large",
            requested=dimensions,  # type: ignore[arg-type]
        )


def test_hybrid_memory_requires_a_platform_embedding_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        asyncio.run(
            build_memory_corpus(
                tmp_path,
                retrieval_mode="hybrid",
                embedding_model="text-embedding-3-large",
                embedding_dimensions=None,
            )
        )


def test_lexical_memory_rejects_embedding_dimensions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not use embedding dimensions"):
        asyncio.run(
            build_memory_corpus(
                tmp_path,
                retrieval_mode="lexical",
                embedding_model="text-embedding-3-large",
                embedding_dimensions=3_072,
            )
        )


def test_build_provider_retains_subscription_path() -> None:
    provider = build_provider(
        provider="subscription",
        codex_lb_base_url="https://unused.example",
        semantic_progress_timeout_seconds=45,
    )

    assert isinstance(provider, OpenAISubscriptionProvider)
    assert provider.stream_deadlines.semantic_progress_timeout_s == 45


def test_build_provider_uses_process_only_codex_lb_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_LB_API_KEY", "test-proxy-key")

    provider = build_provider(
        provider="codex-lb",
        codex_lb_base_url="https://proxy.example/",
        semantic_progress_timeout_seconds=45,
    )

    assert isinstance(provider, OpenAIProvider)
    assert provider.api_key == "test-proxy-key"
    assert provider.base_url == "https://proxy.example"
    assert provider.name == "codex-lb"
    assert provider.stream_deadlines.semantic_progress_timeout_s == 45


def test_build_provider_requires_codex_lb_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEX_LB_API_KEY", raising=False)

    with pytest.raises(ValueError, match="CODEX_LB_API_KEY"):
        build_provider(
            provider="codex-lb",
            codex_lb_base_url="https://proxy.example",
            semantic_progress_timeout_seconds=45,
        )


def test_codex_lb_user_runtime_keeps_key_out_of_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_LB_API_KEY", "test-proxy-key")

    llm_args, api_key = resolve_user_runtime(
        user_provider="codex-lb",
        user_model="openai/gpt-test",
        codex_lb_base_url="https://proxy.example/",
        completion_timeout_seconds=45,
    )

    assert llm_args == {
        "reasoning_effort": "low",
        "api_base": "https://proxy.example/v1",
        "timeout": 45.0,
    }
    assert "test-proxy-key" not in repr(llm_args)
    assert api_key == "test-proxy-key"


def test_platform_user_runtime_records_request_timeout_without_key_override() -> None:
    llm_args, api_key = resolve_user_runtime(
        user_provider="platform",
        user_model="openai/gpt-5.2",
        codex_lb_base_url="https://unused.example",
        completion_timeout_seconds=120,
    )

    assert llm_args == {"reasoning_effort": "low", "timeout": 120.0}
    assert api_key is None


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf")])
def test_user_runtime_rejects_invalid_completion_timeout(timeout: object) -> None:
    with pytest.raises(ValueError, match="positive number"):
        resolve_user_runtime(
            user_provider="platform",
            user_model="openai/gpt-5.2",
            codex_lb_base_url="https://unused.example",
            completion_timeout_seconds=timeout,  # type: ignore[arg-type]
        )


def test_codex_lb_user_runtime_requires_openai_litellm_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEX_LB_API_KEY", "test-proxy-key")

    with pytest.raises(ValueError, match="openai/ prefix"):
        resolve_user_runtime(
            user_provider="codex-lb",
            user_model="gpt-test",
            codex_lb_base_url="https://proxy.example",
            completion_timeout_seconds=45,
        )


def test_codex_lb_base_url_rejects_v1_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_LB_API_KEY", "test-proxy-key")

    with pytest.raises(ValueError, match="must not include the /v1 suffix"):
        resolve_user_runtime(
            user_provider="codex-lb",
            user_model="openai/gpt-test",
            codex_lb_base_url="https://proxy.example/v1",
            completion_timeout_seconds=45,
        )


def test_temporary_openai_api_key_restores_existing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "platform-key")

    with _temporary_openai_api_key("proxy-key"):
        assert os.environ["OPENAI_API_KEY"] == "proxy-key"

    assert os.environ["OPENAI_API_KEY"] == "platform-key"


def test_temporary_openai_api_key_removes_new_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with _temporary_openai_api_key("proxy-key"):
        assert os.environ["OPENAI_API_KEY"] == "proxy-key"

    assert "OPENAI_API_KEY" not in os.environ
