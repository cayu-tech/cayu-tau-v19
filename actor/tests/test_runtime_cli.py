import json

import pytest
from cayu import ScriptedModelProvider
from tau2.data_model.message import UserMessage
from tau2.environment.tool import Tool
from tau2.registry import Registry
from test_runtime_banking import call, host_action, text, work

from cayu_tau3_banking import cli, runtime_cli
from cayu_tau3_banking.controller import SYNC_WORK_PLAN_TOOL
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.runtime_agent import RuntimeAgentError


@pytest.mark.parametrize("value", ["0", "-1"])
def test_native_runtime_rejects_invalid_step_limit_before_launch(value):
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["--runtime-max-steps", value])


@pytest.mark.parametrize("value", ["0", "11", "-1", "1.5"])
def test_native_retry_attempts_are_bounded(value):
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["--runtime-model-max-attempts", value])


def test_cli_defaults_enable_only_bounded_typed_provider_retries(tmp_path, monkeypatch):
    factory, _ = setup_factory(tmp_path, monkeypatch)
    agent = factory(tools=[Tool(host_action)], domain_policy="Follow public policy.")
    assert agent.retry_policy.max_attempts == 3
    assert agent.retry_policy.max_unknown_attempts == 1


class ClosingProvider(ScriptedModelProvider):
    closed = False

    async def aclose(self):
        self.closed = True


def setup_factory(tmp_path, monkeypatch, *, capture=False, parallel=False, lexical=False, extra_args=()):
    (tmp_path / "limits.json").write_text(
        json.dumps(
            {"id": "limits", "title": "Purchase limit", "content": "The purchase limit is $4,500."}
        )
    )
    corpus = BankingKnowledgeCorpus.from_directory(tmp_path)
    outcome = work(done=True)
    outcome.pop("next_operation")
    provider = ClosingProvider(
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [outcome]}),
            text("The purchase limit is $4,500."),
        ]
    )
    monkeypatch.setattr(cli, "build_provider", lambda **kwargs: provider)
    registry = Registry()
    monkeypatch.setattr(runtime_cli, "registry", registry)
    args = cli._parser().parse_args(
        [
            "--runtime-store-directory",
            str(tmp_path / "sessions"),
            "--reasoning-effort",
            "high",
            "--no-semantic-plan-verifier",
            "--no-semantic-action-verifier",
            "--no-semantic-response-verifier",
            *extra_args,
        ]
    )
    if capture:
        args.context_capture_directory = tmp_path / "contexts"
    args.runtime_parallel_tools = parallel
    args.lexical_navigation = lexical
    name = runtime_cli.register_native_agent(args, corpus)
    return registry.get_agent_factory(name), provider


def test_release_retry_and_time_budgets_reach_actor(tmp_path, monkeypatch):
    factory, _ = setup_factory(tmp_path, monkeypatch, extra_args=[
        "--runtime-model-max-attempts", "10",
        "--runtime-conversation-timeout-seconds", "9000",
        "--completion-timeout-seconds", "9150",
        "--simulation-timeout-seconds", "9300",
    ])
    agent = factory(tools=[Tool(host_action)], domain_policy="Follow public policy.")
    assert agent.retry_policy.max_attempts == 10
    assert agent.retry_policy.max_unknown_attempts == 1
    assert agent.conversation_timeout_s == 9000
    assert agent.callback_timeout_s == 9150


@pytest.mark.parametrize("enabled", [False, True])
def test_lexical_navigation_flag_reaches_native_resources(tmp_path, monkeypatch, enabled):
    original = runtime_cli.build_banking_runtime_resources
    observed = []

    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(
            (
                kwargs["lexical_navigation"],
                result.diagnostic_snapshot()["cayu_native_runtime"]["lexical_navigation"],
            )
        )
        return result

    monkeypatch.setattr(runtime_cli, "build_banking_runtime_resources", build)
    factory, _ = setup_factory(tmp_path, monkeypatch, lexical=enabled)
    agent = factory(tools=[Tool(host_action)], domain_policy="Follow public policy.")
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Explain the purchase limit."), state)
        assert observed == [(enabled, enabled)]
    finally:
        state.stop()


def test_context_capture_contains_actual_native_recall_messages(tmp_path, monkeypatch):
    factory, provider = setup_factory(tmp_path, monkeypatch, capture=True)
    agent = factory(tools=[Tool(host_action)], domain_policy="Follow public policy.")
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Explain the purchase limit."), state)
    finally:
        state.stop()
    snapshots = [json.loads(path.read_text()) for path in (tmp_path / "contexts").glob("*/*.json")]
    assert len(snapshots) == len(provider.requests) == 2
    for snapshot in snapshots:
        # The runtime may prepend the system message after context composition.
        assert any(
            all(
                message in request.model_dump(mode="json")["messages"]
                for message in snapshot["messages"]
            )
            for request in provider.requests
        )
        assert "4,500" in json.dumps(snapshot["messages"])
        assert snapshot["model_step_id"]


def test_parallel_calls_option_reaches_actor_provider(tmp_path, monkeypatch):
    assert cli._parser().parse_args(["--runtime-parallel-tools"]).runtime_parallel_tools
    factory, provider = setup_factory(tmp_path, monkeypatch, parallel=True)
    agent = factory(tools=[Tool(host_action)], domain_policy="Follow public policy.")
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Explain the purchase limit."), state)
        assert provider.requests[0].options["openai"]["parallel_tool_calls"] is True
    finally:
        state.stop()


def test_cli_factory_runs_native_recall_with_requested_thinking_and_owned_cleanup(
    tmp_path, monkeypatch
):
    factory, provider = setup_factory(tmp_path, monkeypatch)

    class OracleTrap:
        def __str__(self):
            raise AssertionError("Oracle must not be inspected")

    agent = factory(
        tools=[Tool(host_action)], domain_policy="Follow public policy.", task=OracleTrap()
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Explain the purchase limit."), state
        )
        assert reply.content == "The purchase limit is $4,500."
        assert reply.raw_data["cayu_tau3_workflow"]["revision"] > 0
        assert reply.raw_data["cayu_tau3_memory"]["knowledge_retrieval"]["mode"] == "lexical"
        assert reply.raw_data["cayu_native_runtime"]["observed_recall_exposures"] == 2
        assert provider.requests[0].options["thinking"] == {"enabled": True, "effort": "high"}
        assert provider.requests[0].options["openai"]["parallel_tool_calls"] is False
        assert state._conversation.resources.observer.native_recall
        assert not state._conversation.usage.untracked_helpers
        assert len(list((tmp_path / "sessions").glob("*.sqlite3"))) == 1
    finally:
        state.stop()
    assert provider.closed


def test_failed_cli_assembly_closes_provider_before_loop_shutdown(tmp_path, monkeypatch):
    factory, provider = setup_factory(tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        raise ValueError("Assembly failure")

    monkeypatch.setattr(runtime_cli, "build_banking_runtime_resources", fail)
    agent = factory(tools=[Tool(host_action)], domain_policy="Follow public policy.")
    with pytest.raises(RuntimeAgentError, match="initialization failed"):
        agent.get_init_state()
    assert provider.closed
